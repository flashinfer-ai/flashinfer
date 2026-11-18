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
#define SMEM_RED_SUM_STAGE_BYTES 24576
#define SMEM_RED_SUM_STRIDE 24576
#define SMEM_RED_Q_OFF 24576
#define SMEM_RED_Q_STAGE_BYTES 24576
#define SMEM_RED_Q_STRIDE 24576
#define SMEM_FLAG_OFF 49152
#define SMEM_FLAG_STAGE_BYTES 48
#define SMEM_FLAG_STRIDE 48
#define SMEM_ITEM_CELL_OFF 0
#define SMEM_ITEM_CELL_STAGE_BYTES 48
#define SMEM_ITEM_CELL_STRIDE 48
#define SMEM_TOTAL 49536
#define THREADS 768
#define LAUNCH_MIN_BLOCKS 1

#include <math_constants.h>

extern "C" {

__global__ __launch_bounds__(768, LAUNCH_MIN_BLOCKS) void
kernel_cake_minimax_h3_sm120_nvfp4_varlen_attention_b7747937e84437b2ac99(__nv_bfloat16* __restrict__ Q, __nv_bfloat16* __restrict__ K, __nv_bfloat16* __restrict__ V, int* __restrict__ tile_table, int* __restrict__ seg_tile_begin, int* __restrict__ tok_seg, int* __restrict__ cu_seqlens, float* __restrict__ partials, unsigned int* __restrict__ counters, float* __restrict__ mean_k, float* __restrict__ q_mean, unsigned int* __restrict__ q4, unsigned int* __restrict__ k4, uint8_t* __restrict__ q_sf, uint8_t* __restrict__ k_sf, unsigned int* __restrict__ qm4, uint8_t* __restrict__ qm_sf, unsigned int* __restrict__ vt4, uint8_t* __restrict__ v_sf, unsigned int* __restrict__ hctr, int num_tiles, int total_tokens, int total_padded, int num_heads, int num_kblocks)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    __shared__ __align__(16) unsigned char smem_static_raw[48];
    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    float* red_sum = reinterpret_cast<float*>(smem_raw + 0);
    const int red_sum_addr = smem + 0;
    float* red_q = reinterpret_cast<float*>(smem_raw + 24576);
    const int red_q_addr = smem + 24576;
    int* flag = reinterpret_cast<int*>(smem_raw + 49152);
    const int flag_addr = smem + 49152;
    int* item_cell = reinterpret_cast<int*>(smem_static_raw + 0);
    const int item_cell_addr = (int)(unsigned long long)__cvta_generic_to_shared(item_cell);

    // ---- Role: wk0 ----
    if (warp <= 7) {
        { // wk0_main
            int wtid = tid;
            int col = (wtid & 15) * 8;
            int rgrp = wtid >> 4;
            int sub = wtid & 7;
            int trow = wtid >> 3;
            int col16 = sub * 16;
            int num_tblk = total_padded / 32;
            int vwarp = wtid >> 5;
            int vlane = wtid & 31;
            int vblk = vwarp & 1;
            int vdp = (vwarp >> 1) * 16 + (vlane & 15);
            int vhalf = vlane >> 4;
            int n_abc = 2 * num_tiles + num_tblk;
            int per = n_abc + num_tblk;
            asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
            unsigned int nxt_r[2];
            if (wtid == 0) {
                unsigned int _atomic_fetch_add_0 = atomicAdd(&hctr[5 * num_heads + 4], 1);
                item_cell[0] = (int)_atomic_fetch_add_0;
            }
            asm volatile("bar.sync 1, 256;" ::: "memory");
            #pragma unroll 1
            for (int it = 0; it < per * num_heads; it++) {
                int w = item_cell[0 + (it & 1)];
                if (w >= per * num_heads) {
                    break;
                }
                if (wtid == 0) {
                    unsigned int _atomic_fetch_add_1 = atomicAdd(&hctr[5 * num_heads + 4], 1);
                    nxt_r[0] = _atomic_fetch_add_1;
                }
                int w2 = ((w >= n_abc) ? w - n_abc : 0);
                int s = w2 / per;
                int r = w2 - s * per;
                int last_s = ((s + 1 >= num_heads) ? 1 : 0);
                int is_d = ((w >= n_abc && (last_s == 1 || r >= n_abc)) ? 1 : 0);
                int head = ((w < n_abc) ? 0 : ((is_d == 1) ? s : s + 1));
                int ri = ((w < n_abc) ? w : ((last_s == 1) ? r : ((is_d == 1) ? r - n_abc : r)));
                if (is_d == 0) {
                    if (ri < num_tiles) {
                        int tile = (ri * num_heads + head) / num_heads;
                        int head_0 = (ri * num_heads + head) % num_heads;
                        int _vec_load_0[4];
                        {
                            const int4* _ivptr_2 = reinterpret_cast<const int4*>(tile_table + (8 * tile) + 0);
                            int4 _ivld_2;
                            _ivld_2 = *_ivptr_2;
                            _vec_load_0[0 + 0] = _ivld_2.x;
                            _vec_load_0[0 + 1] = _ivld_2.y;
                            _vec_load_0[0 + 2] = _ivld_2.z;
                            _vec_load_0[0 + 3] = _ivld_2.w;
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
                                const uint4* _vptr_3 = reinterpret_cast<const uint4*>(Q + ((((row_end > brow + rgrp) ? brow + rgrp : row_end - 1) * num_heads + head_0) * 128 + col) + 0);
                                uint4 _vld_3[1];
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
                                    _vld_3[_blk] = _vptr_3[_blk];
                                    uint32_t* _vpairs_3 = reinterpret_cast<uint32_t*>(&_vld_3[_blk]);
                                    #pragma unroll
                                    for (int _pair = 0; _pair < 4; _pair++) {
                                        (&vec_load[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_3[_pair]) << 16);
                                        (&vec_load[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_3[_pair]) & 0xffff0000u);
                                    }
                                }
                            }
                            float vec_load_1[8];
                            {
                                const uint4* _vptr_4 = reinterpret_cast<const uint4*>(Q + ((((row_end > brow + 16 + rgrp) ? brow + 16 + rgrp : row_end - 1) * num_heads + head_0) * 128 + col) + 0);
                                uint4 _vld_4[1];
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
                                    _vld_4[_blk] = _vptr_4[_blk];
                                    uint32_t* _vpairs_4 = reinterpret_cast<uint32_t*>(&_vld_4[_blk]);
                                    #pragma unroll
                                    for (int _pair = 0; _pair < 4; _pair++) {
                                        (&vec_load_1[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_4[_pair]) << 16);
                                        (&vec_load_1[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_4[_pair]) & 0xffff0000u);
                                    }
                                }
                            }
                            float vec_load_2[8];
                            {
                                const uint4* _vptr_5 = reinterpret_cast<const uint4*>(Q + ((((row_end > brow + 32 + rgrp) ? brow + 32 + rgrp : row_end - 1) * num_heads + head_0) * 128 + col) + 0);
                                uint4 _vld_5[1];
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
                                    _vld_5[_blk] = _vptr_5[_blk];
                                    uint32_t* _vpairs_5 = reinterpret_cast<uint32_t*>(&_vld_5[_blk]);
                                    #pragma unroll
                                    for (int _pair = 0; _pair < 4; _pair++) {
                                        (&vec_load_2[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_5[_pair]) << 16);
                                        (&vec_load_2[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_5[_pair]) & 0xffff0000u);
                                    }
                                }
                            }
                            float vec_load_3[8];
                            {
                                const uint4* _vptr_6 = reinterpret_cast<const uint4*>(Q + ((((row_end > brow + 48 + rgrp) ? brow + 48 + rgrp : row_end - 1) * num_heads + head_0) * 128 + col) + 0);
                                uint4 _vld_6[1];
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
                                    _vld_6[_blk] = _vptr_6[_blk];
                                    uint32_t* _vpairs_6 = reinterpret_cast<uint32_t*>(&_vld_6[_blk]);
                                    #pragma unroll
                                    for (int _pair = 0; _pair < 4; _pair++) {
                                        (&vec_load_3[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_6[_pair]) << 16);
                                        (&vec_load_3[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_6[_pair]) & 0xffff0000u);
                                    }
                                }
                            }
                            float vec_load_4[8];
                            {
                                const uint4* _vptr_7 = reinterpret_cast<const uint4*>(Q + ((((row_end > brow + 64 + rgrp) ? brow + 64 + rgrp : row_end - 1) * num_heads + head_0) * 128 + col) + 0);
                                uint4 _vld_7[1];
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
                                    _vld_7[_blk] = _vptr_7[_blk];
                                    uint32_t* _vpairs_7 = reinterpret_cast<uint32_t*>(&_vld_7[_blk]);
                                    #pragma unroll
                                    for (int _pair = 0; _pair < 4; _pair++) {
                                        (&vec_load_4[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_7[_pair]) << 16);
                                        (&vec_load_4[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_7[_pair]) & 0xffff0000u);
                                    }
                                }
                            }
                            float vec_load_5[8];
                            {
                                const uint4* _vptr_8 = reinterpret_cast<const uint4*>(Q + ((((row_end > brow + 80 + rgrp) ? brow + 80 + rgrp : row_end - 1) * num_heads + head_0) * 128 + col) + 0);
                                uint4 _vld_8[1];
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
                                    _vld_8[_blk] = _vptr_8[_blk];
                                    uint32_t* _vpairs_8 = reinterpret_cast<uint32_t*>(&_vld_8[_blk]);
                                    #pragma unroll
                                    for (int _pair = 0; _pair < 4; _pair++) {
                                        (&vec_load_5[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_8[_pair]) << 16);
                                        (&vec_load_5[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_8[_pair]) & 0xffff0000u);
                                    }
                                }
                            }
                            float vec_load_6[8];
                            {
                                const uint4* _vptr_9 = reinterpret_cast<const uint4*>(Q + ((((row_end > brow + 96 + rgrp) ? brow + 96 + rgrp : row_end - 1) * num_heads + head_0) * 128 + col) + 0);
                                uint4 _vld_9[1];
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
                                    _vld_9[_blk] = _vptr_9[_blk];
                                    uint32_t* _vpairs_9 = reinterpret_cast<uint32_t*>(&_vld_9[_blk]);
                                    #pragma unroll
                                    for (int _pair = 0; _pair < 4; _pair++) {
                                        (&vec_load_6[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_9[_pair]) << 16);
                                        (&vec_load_6[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_9[_pair]) & 0xffff0000u);
                                    }
                                }
                            }
                            float vec_load_7[8];
                            {
                                const uint4* _vptr_10 = reinterpret_cast<const uint4*>(Q + ((((row_end > brow + 112 + rgrp) ? brow + 112 + rgrp : row_end - 1) * num_heads + head_0) * 128 + col) + 0);
                                uint4 _vld_10[1];
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
                                    _vld_10[_blk] = _vptr_10[_blk];
                                    uint32_t* _vpairs_10 = reinterpret_cast<uint32_t*>(&_vld_10[_blk]);
                                    #pragma unroll
                                    for (int _pair = 0; _pair < 4; _pair++) {
                                        (&vec_load_7[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_10[_pair]) << 16);
                                        (&vec_load_7[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_10[_pair]) & 0xffff0000u);
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
                                red_q[0 + rgrp * 128 + col + e_1] = qsum[e_1];
                            }
                            asm volatile("bar.sync 1, 256;" ::: "memory");
                            int qt = qt_base + batch;
                            if (wtid < 128) {
                                float qtot = 0.0f;
                                for (int g = 0; g < 16; g++) {
                                    qtot = qtot + red_q[0 + g * 128 + wtid];
                                }
                                int brows = ((row_end - brow < 128) ? row_end - brow : 128);
                                float _fdiv_11 = __fdiv_rn(qtot, (float)brows);
                                *(reinterpret_cast<float*>(q_mean + ((qt * num_heads + head_0) * 128 + wtid)) + (0)) = _fdiv_11;
                            }
                            asm volatile("bar.sync 1, 256;" ::: "memory");
                            #pragma unroll 1
                            for (int blk = 0; blk < 4; blk++) {
                                int tok = brow + blk * 32 + trow;
                                int valid = ((tok < row_end) ? 1 : 0);
                                int tok_c = ((tok < row_end) ? tok : row_end - 1);
                                int is_first = ((valid == 1 && tok == brow) ? 1 : 0);
                                float vec_load_0[16];
                                {
                                    const uint4* _vptr_12 = reinterpret_cast<const uint4*>(Q + ((tok_c * num_heads + head_0) * 128 + col16) + 0);
                                    uint4 _vld_12[2];
                                    #pragma unroll
                                    for (int _blk = 0; _blk < 2; _blk++) {
                                        _vld_12[_blk] = _vptr_12[_blk];
                                        uint32_t* _vpairs_12 = reinterpret_cast<uint32_t*>(&_vld_12[_blk]);
                                        #pragma unroll
                                        for (int _pair = 0; _pair < 4; _pair++) {
                                            (&vec_load_0[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_12[_pair]) << 16);
                                            (&vec_load_0[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_12[_pair]) & 0xffff0000u);
                                        }
                                    }
                                }
                                int qmbase = (qt * num_heads + head_0) * 128 + col16;
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
                                uint16_t _cvt_satfinite_e4m3x2_f32_14;
                                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_14) : "f"(0.0f), "f"(max_1 * 0.16666667f));
                                uint16_t _e4m3x2_decode_15 = (uint16_t)((unsigned int)_cvt_satfinite_e4m3x2_f32_14 & 0xFFu);
                                uint32_t _f16x2_decode_15;
                                float fp8_decode;
                                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_decode_15) : "h"(_e4m3x2_decode_15));
                                uint16_t _f16_decode_15 = (uint16_t)_f16x2_decode_15;
                                asm("cvt.f32.f16 %0, %1;" : "=f"(fp8_decode) : "h"(_f16_decode_15));
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
                                unsigned int _shfl_xor_sync_16 = __shfl_xor_sync(0xFFFFFFFF, qw[0], 1);
                                qstore[2] = _shfl_xor_sync_16;
                                unsigned int _shfl_xor_sync_17 = __shfl_xor_sync(0xFFFFFFFF, qw[1], 1);
                                qstore[3] = _shfl_xor_sync_17;
                                if (valid == 1) {
                                    uint16_t _cvt_satfinite_e4m3x2_f32_18;
                                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_18) : "f"(0.0f), "f"(qsf_a[0]));
                                    *reinterpret_cast<unsigned char*>(q_sf + ((tok * num_heads + head_0) * 8 + sub)) = (unsigned char)((unsigned int)(unsigned char)(_cvt_satfinite_e4m3x2_f32_18 & 127));
                                    if ((sub & 1) == 0) {
                                        reinterpret_cast<int4*>(q4 + (((tok * num_heads + head_0) * 64 + sub * 8) / 4))[0] = reinterpret_cast<int4*>(qstore)[0];
                                    }
                                }
                                if (is_first == 1) {
                                    unsigned int mw[2];
                                    float msf_a[1];
                                    float max_1_0 = fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fabsf(qmv[0]), fabsf(qmv[1])), fabsf(qmv[2])), fabsf(qmv[3])), fabsf(qmv[4])), fabsf(qmv[5])), fabsf(qmv[6])), fabsf(qmv[7])), fabsf(qmv[8])), fabsf(qmv[9])), fabsf(qmv[10])), fabsf(qmv[11])), fabsf(qmv[12])), fabsf(qmv[13])), fabsf(qmv[14])), fabsf(qmv[15]));
                                    uint16_t _cvt_satfinite_e4m3x2_f32_19;
                                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_19) : "f"(0.0f), "f"(max_1_0 * 0.16666667f));
                                    uint16_t _e4m3x2_decode_20 = (uint16_t)((unsigned int)_cvt_satfinite_e4m3x2_f32_19 & 0xFFu);
                                    uint32_t _f16x2_decode_20;
                                    float fp8_decode_0;
                                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_decode_20) : "h"(_e4m3x2_decode_20));
                                    uint16_t _f16_decode_20 = (uint16_t)_f16x2_decode_20;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(fp8_decode_0) : "h"(_f16_decode_20));
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
                                    int qmrow = qt * num_heads + head_0;
                                    *(reinterpret_cast<unsigned int*>(qm4 + (qmrow * 16 + sub * 2)) + (0)) = mw[0];
                                    *(reinterpret_cast<unsigned int*>(qm4 + (qmrow * 16 + sub * 2 + 1)) + (0)) = mw[1];
                                    uint16_t _cvt_satfinite_e4m3x2_f32_21;
                                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_21) : "f"(0.0f), "f"(msf_a[0]));
                                    *reinterpret_cast<unsigned char*>(qm_sf + (qmrow * 8 + sub)) = (unsigned char)((unsigned int)(unsigned char)(_cvt_satfinite_e4m3x2_f32_21 & 127));
                                }
                            }
                        }
                    } else if (ri < num_tiles + num_tblk) {
                        int vlive[8];
                        vlive[0] = (((ri - num_tiles) * 32 + (vblk * 4 + vhalf * 2) < total_tokens) ? 1 : 0);
                        float _vec_load_11[2];
                        {
                            uint32_t _bf16x2_bits_22;
                            _bf16x2_bits_22 = *reinterpret_cast<const uint32_t*>(V + (((((ri - num_tiles) * 32 + (vblk * 4 + vhalf * 2) < total_tokens) ? (ri - num_tiles) * 32 + (vblk * 4 + vhalf * 2) : total_tokens - 1) * num_heads + head) * 128 + vdp * 2) + 0);
                            (&_vec_load_11[0])[0] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_22) << 16);
                            (&_vec_load_11[0])[1] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_22) & 0xffff0000u);
                        }
                        vlive[1] = (((ri - num_tiles) * 32 + (vblk * 4 + vhalf * 2 + 1) < total_tokens) ? 1 : 0);
                        float _vec_load_12[2];
                        {
                            uint32_t _bf16x2_bits_23;
                            _bf16x2_bits_23 = *reinterpret_cast<const uint32_t*>(V + (((((ri - num_tiles) * 32 + (vblk * 4 + vhalf * 2 + 1) < total_tokens) ? (ri - num_tiles) * 32 + (vblk * 4 + vhalf * 2 + 1) : total_tokens - 1) * num_heads + head) * 128 + vdp * 2) + 0);
                            (&_vec_load_12[0])[0] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_23) << 16);
                            (&_vec_load_12[0])[1] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_23) & 0xffff0000u);
                        }
                        vlive[2] = (((ri - num_tiles) * 32 + (8 + vblk * 4 + vhalf * 2) < total_tokens) ? 1 : 0);
                        float _vec_load_13[2];
                        {
                            uint32_t _bf16x2_bits_24;
                            _bf16x2_bits_24 = *reinterpret_cast<const uint32_t*>(V + (((((ri - num_tiles) * 32 + (8 + vblk * 4 + vhalf * 2) < total_tokens) ? (ri - num_tiles) * 32 + (8 + vblk * 4 + vhalf * 2) : total_tokens - 1) * num_heads + head) * 128 + vdp * 2) + 0);
                            (&_vec_load_13[0])[0] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_24) << 16);
                            (&_vec_load_13[0])[1] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_24) & 0xffff0000u);
                        }
                        vlive[3] = (((ri - num_tiles) * 32 + (8 + vblk * 4 + vhalf * 2 + 1) < total_tokens) ? 1 : 0);
                        float _vec_load_14[2];
                        {
                            uint32_t _bf16x2_bits_25;
                            _bf16x2_bits_25 = *reinterpret_cast<const uint32_t*>(V + (((((ri - num_tiles) * 32 + (8 + vblk * 4 + vhalf * 2 + 1) < total_tokens) ? (ri - num_tiles) * 32 + (8 + vblk * 4 + vhalf * 2 + 1) : total_tokens - 1) * num_heads + head) * 128 + vdp * 2) + 0);
                            (&_vec_load_14[0])[0] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_25) << 16);
                            (&_vec_load_14[0])[1] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_25) & 0xffff0000u);
                        }
                        vlive[4] = (((ri - num_tiles) * 32 + (16 + vblk * 4 + vhalf * 2) < total_tokens) ? 1 : 0);
                        float _vec_load_15[2];
                        {
                            uint32_t _bf16x2_bits_26;
                            _bf16x2_bits_26 = *reinterpret_cast<const uint32_t*>(V + (((((ri - num_tiles) * 32 + (16 + vblk * 4 + vhalf * 2) < total_tokens) ? (ri - num_tiles) * 32 + (16 + vblk * 4 + vhalf * 2) : total_tokens - 1) * num_heads + head) * 128 + vdp * 2) + 0);
                            (&_vec_load_15[0])[0] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_26) << 16);
                            (&_vec_load_15[0])[1] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_26) & 0xffff0000u);
                        }
                        vlive[5] = (((ri - num_tiles) * 32 + (16 + vblk * 4 + vhalf * 2 + 1) < total_tokens) ? 1 : 0);
                        float _vec_load_16[2];
                        {
                            uint32_t _bf16x2_bits_27;
                            _bf16x2_bits_27 = *reinterpret_cast<const uint32_t*>(V + (((((ri - num_tiles) * 32 + (16 + vblk * 4 + vhalf * 2 + 1) < total_tokens) ? (ri - num_tiles) * 32 + (16 + vblk * 4 + vhalf * 2 + 1) : total_tokens - 1) * num_heads + head) * 128 + vdp * 2) + 0);
                            (&_vec_load_16[0])[0] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_27) << 16);
                            (&_vec_load_16[0])[1] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_27) & 0xffff0000u);
                        }
                        vlive[6] = (((ri - num_tiles) * 32 + (24 + vblk * 4 + vhalf * 2) < total_tokens) ? 1 : 0);
                        float _vec_load_17[2];
                        {
                            uint32_t _bf16x2_bits_28;
                            _bf16x2_bits_28 = *reinterpret_cast<const uint32_t*>(V + (((((ri - num_tiles) * 32 + (24 + vblk * 4 + vhalf * 2) < total_tokens) ? (ri - num_tiles) * 32 + (24 + vblk * 4 + vhalf * 2) : total_tokens - 1) * num_heads + head) * 128 + vdp * 2) + 0);
                            (&_vec_load_17[0])[0] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_28) << 16);
                            (&_vec_load_17[0])[1] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_28) & 0xffff0000u);
                        }
                        vlive[7] = (((ri - num_tiles) * 32 + (24 + vblk * 4 + vhalf * 2 + 1) < total_tokens) ? 1 : 0);
                        float _vec_load_18[2];
                        {
                            uint32_t _bf16x2_bits_29;
                            _bf16x2_bits_29 = *reinterpret_cast<const uint32_t*>(V + (((((ri - num_tiles) * 32 + (24 + vblk * 4 + vhalf * 2 + 1) < total_tokens) ? (ri - num_tiles) * 32 + (24 + vblk * 4 + vhalf * 2 + 1) : total_tokens - 1) * num_heads + head) * 128 + vdp * 2) + 0);
                            (&_vec_load_18[0])[0] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_29) << 16);
                            (&_vec_load_18[0])[1] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_29) & 0xffff0000u);
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
                        float _shfl_xor_sync_30 = __shfl_xor_sync(0xFFFFFFFF, amax_a, 16);
                        amax_a = fmaxf(amax_a, _shfl_xor_sync_30);
                        float _shfl_xor_sync_31 = __shfl_xor_sync(0xFFFFFFFF, amax_b, 16);
                        amax_b = fmaxf(amax_b, _shfl_xor_sync_31);
                        float vsf_a = amax_a * 0.16666667f;
                        float vsf_b = amax_b * 0.16666667f;
                        uint16_t _e4m3x2_f32_0;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_0) : "f"(vsf_b), "f"(vsf_a));
                        uint16_t _e4m3x2_decode_32 = (uint16_t)((unsigned int)_e4m3x2_f32_0 & 0xFFu);
                        uint32_t _f16x2_decode_32;
                        float vdec_a;
                        asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_decode_32) : "h"(_e4m3x2_decode_32));
                        uint16_t _f16_decode_32 = (uint16_t)_f16x2_decode_32;
                        asm("cvt.f32.f16 %0, %1;" : "=f"(vdec_a) : "h"(_f16_decode_32));
                        uint16_t _e4m3x2_decode_33 = (uint16_t)((unsigned int)_e4m3x2_f32_0 >> 8 & 0xFFu);
                        uint32_t _f16x2_decode_33;
                        float vdec_b;
                        asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_decode_33) : "h"(_e4m3x2_decode_33));
                        uint16_t _f16_decode_33 = (uint16_t)_f16x2_decode_33;
                        asm("cvt.f32.f16 %0, %1;" : "=f"(vdec_b) : "h"(_f16_decode_33));
                        float _fdiv_34 = __fdiv_rn(1.0f, vdec_a);
                        float vinv_a = ((vdec_a != 0.0f) ? _fdiv_34 : 0.0f);
                        float _fdiv_35 = __fdiv_rn(1.0f, vdec_b);
                        float vinv_b = ((vdec_b != 0.0f) ? _fdiv_35 : 0.0f);
                        for (int ii_1 = 0; ii_1 < 8; ii_1++) {
                            va[ii_1] = va[ii_1] * vinv_a;
                            vb[ii_1] = vb[ii_1] * vinv_b;
                        }
                        unsigned int vwa[1];
                        unsigned int vwb[1];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(vwa[0]) : "f"(va[0]), "f"(va[1]), "f"(va[2]), "f"(va[3]), "f"(va[4]), "f"(va[5]), "f"(va[6]), "f"(va[7]));
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(vwb[0]) : "f"(vb[0]), "f"(vb[1]), "f"(vb[2]), "f"(vb[3]), "f"(vb[4]), "f"(vb[5]), "f"(vb[6]), "f"(vb[7]));
                        int row_words = total_padded >> 3;
                        int vt_word = (head * 128 + vdp * 2) * row_words + ((ri - num_tiles) * 32 >> 3) + vblk * 2 + vhalf;
                        *(reinterpret_cast<unsigned int*>(vt4 + vt_word) + (0)) = vwa[0];
                        *(reinterpret_cast<unsigned int*>(vt4 + (vt_word + row_words)) + (0)) = vwb[0];
                        if (vhalf == 0) {
                            int vsf_base = ((head * num_kblocks + ((ri - num_tiles) * 32 >> 7)) * 128 + vdp * 2) * 8 + ((((ri - num_tiles) * 32 & 127) >> 4) + vblk);
                            uint16_t _cvt_satfinite_e4m3x2_f32_36;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_36) : "f"(0.0f), "f"(vsf_a));
                            *reinterpret_cast<unsigned char*>(v_sf + vsf_base) = (unsigned char)((unsigned int)(unsigned char)(_cvt_satfinite_e4m3x2_f32_36 & 127));
                            uint16_t _cvt_satfinite_e4m3x2_f32_37;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_37) : "f"(0.0f), "f"(vsf_b));
                            *reinterpret_cast<unsigned char*>(v_sf + (vsf_base + 8)) = (unsigned char)((unsigned int)(unsigned char)(_cvt_satfinite_e4m3x2_f32_37 & 127));
                        }
                    } else {
                        int tile_1 = ((ri - num_tiles - num_tblk) * num_heads + head) / num_heads;
                        int head_0_1 = ((ri - num_tiles - num_tblk) * num_heads + head) % num_heads;
                        int _vec_load_19[4];
                        {
                            const int4* _ivptr_38 = reinterpret_cast<const int4*>(tile_table + (8 * tile_1) + 0);
                            int4 _ivld_38;
                            _ivld_38 = *_ivptr_38;
                            _vec_load_19[0 + 0] = _ivld_38.x;
                            _vec_load_19[0 + 1] = _ivld_38.y;
                            _vec_load_19[0 + 2] = _ivld_38.z;
                            _vec_load_19[0 + 3] = _ivld_38.w;
                        }
                        int _vec_load_20[4];
                        {
                            const int4* _ivptr_39 = reinterpret_cast<const int4*>(tile_table + (8 * tile_1 + 4) + 0);
                            int4 _ivld_39;
                            _ivld_39 = *_ivptr_39;
                            _vec_load_20[0 + 0] = _ivld_39.x;
                            _vec_load_20[0 + 1] = _ivld_39.y;
                            _vec_load_20[0 + 2] = _ivld_39.z;
                            _vec_load_20[0 + 3] = _ivld_39.w;
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
                                const uint4* _vptr_40 = reinterpret_cast<const uint4*>(K + ((((row_end_1 > brow_1 + rgrp) ? brow_1 + rgrp : row_end_1 - 1) * num_heads + head_0_1) * 128 + col) + 0);
                                uint4 _vld_40[1];
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
                                    _vld_40[_blk] = _vptr_40[_blk];
                                    uint32_t* _vpairs_40 = reinterpret_cast<uint32_t*>(&_vld_40[_blk]);
                                    #pragma unroll
                                    for (int _pair = 0; _pair < 4; _pair++) {
                                        (&vec_load_8[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_40[_pair]) << 16);
                                        (&vec_load_8[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_40[_pair]) & 0xffff0000u);
                                    }
                                }
                            }
                            float vec_load_1_1[8];
                            {
                                const uint4* _vptr_41 = reinterpret_cast<const uint4*>(K + ((((row_end_1 > brow_1 + 16 + rgrp) ? brow_1 + 16 + rgrp : row_end_1 - 1) * num_heads + head_0_1) * 128 + col) + 0);
                                uint4 _vld_41[1];
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
                                    _vld_41[_blk] = _vptr_41[_blk];
                                    uint32_t* _vpairs_41 = reinterpret_cast<uint32_t*>(&_vld_41[_blk]);
                                    #pragma unroll
                                    for (int _pair = 0; _pair < 4; _pair++) {
                                        (&vec_load_1_1[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_41[_pair]) << 16);
                                        (&vec_load_1_1[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_41[_pair]) & 0xffff0000u);
                                    }
                                }
                            }
                            float vec_load_2_1[8];
                            {
                                const uint4* _vptr_42 = reinterpret_cast<const uint4*>(K + ((((row_end_1 > brow_1 + 32 + rgrp) ? brow_1 + 32 + rgrp : row_end_1 - 1) * num_heads + head_0_1) * 128 + col) + 0);
                                uint4 _vld_42[1];
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
                                    _vld_42[_blk] = _vptr_42[_blk];
                                    uint32_t* _vpairs_42 = reinterpret_cast<uint32_t*>(&_vld_42[_blk]);
                                    #pragma unroll
                                    for (int _pair = 0; _pair < 4; _pair++) {
                                        (&vec_load_2_1[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_42[_pair]) << 16);
                                        (&vec_load_2_1[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_42[_pair]) & 0xffff0000u);
                                    }
                                }
                            }
                            float vec_load_3_1[8];
                            {
                                const uint4* _vptr_43 = reinterpret_cast<const uint4*>(K + ((((row_end_1 > brow_1 + 48 + rgrp) ? brow_1 + 48 + rgrp : row_end_1 - 1) * num_heads + head_0_1) * 128 + col) + 0);
                                uint4 _vld_43[1];
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
                                    _vld_43[_blk] = _vptr_43[_blk];
                                    uint32_t* _vpairs_43 = reinterpret_cast<uint32_t*>(&_vld_43[_blk]);
                                    #pragma unroll
                                    for (int _pair = 0; _pair < 4; _pair++) {
                                        (&vec_load_3_1[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_43[_pair]) << 16);
                                        (&vec_load_3_1[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_43[_pair]) & 0xffff0000u);
                                    }
                                }
                            }
                            float vec_load_4_1[8];
                            {
                                const uint4* _vptr_44 = reinterpret_cast<const uint4*>(K + ((((row_end_1 > brow_1 + 64 + rgrp) ? brow_1 + 64 + rgrp : row_end_1 - 1) * num_heads + head_0_1) * 128 + col) + 0);
                                uint4 _vld_44[1];
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
                                    _vld_44[_blk] = _vptr_44[_blk];
                                    uint32_t* _vpairs_44 = reinterpret_cast<uint32_t*>(&_vld_44[_blk]);
                                    #pragma unroll
                                    for (int _pair = 0; _pair < 4; _pair++) {
                                        (&vec_load_4_1[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_44[_pair]) << 16);
                                        (&vec_load_4_1[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_44[_pair]) & 0xffff0000u);
                                    }
                                }
                            }
                            float vec_load_5_1[8];
                            {
                                const uint4* _vptr_45 = reinterpret_cast<const uint4*>(K + ((((row_end_1 > brow_1 + 80 + rgrp) ? brow_1 + 80 + rgrp : row_end_1 - 1) * num_heads + head_0_1) * 128 + col) + 0);
                                uint4 _vld_45[1];
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
                                    _vld_45[_blk] = _vptr_45[_blk];
                                    uint32_t* _vpairs_45 = reinterpret_cast<uint32_t*>(&_vld_45[_blk]);
                                    #pragma unroll
                                    for (int _pair = 0; _pair < 4; _pair++) {
                                        (&vec_load_5_1[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_45[_pair]) << 16);
                                        (&vec_load_5_1[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_45[_pair]) & 0xffff0000u);
                                    }
                                }
                            }
                            float vec_load_6_1[8];
                            {
                                const uint4* _vptr_46 = reinterpret_cast<const uint4*>(K + ((((row_end_1 > brow_1 + 96 + rgrp) ? brow_1 + 96 + rgrp : row_end_1 - 1) * num_heads + head_0_1) * 128 + col) + 0);
                                uint4 _vld_46[1];
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
                                    _vld_46[_blk] = _vptr_46[_blk];
                                    uint32_t* _vpairs_46 = reinterpret_cast<uint32_t*>(&_vld_46[_blk]);
                                    #pragma unroll
                                    for (int _pair = 0; _pair < 4; _pair++) {
                                        (&vec_load_6_1[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_46[_pair]) << 16);
                                        (&vec_load_6_1[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_46[_pair]) & 0xffff0000u);
                                    }
                                }
                            }
                            float vec_load_7_1[8];
                            {
                                const uint4* _vptr_47 = reinterpret_cast<const uint4*>(K + ((((row_end_1 > brow_1 + 112 + rgrp) ? brow_1 + 112 + rgrp : row_end_1 - 1) * num_heads + head_0_1) * 128 + col) + 0);
                                uint4 _vld_47[1];
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
                                    _vld_47[_blk] = _vptr_47[_blk];
                                    uint32_t* _vpairs_47 = reinterpret_cast<uint32_t*>(&_vld_47[_blk]);
                                    #pragma unroll
                                    for (int _pair = 0; _pair < 4; _pair++) {
                                        (&vec_load_7_1[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_47[_pair]) << 16);
                                        (&vec_load_7_1[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_47[_pair]) & 0xffff0000u);
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
                            red_sum[0 + rgrp * 128 + col + e_4] = ksum[e_4];
                        }
                        asm volatile("bar.sync 1, 256;" ::: "memory");
                        int direct = ((t1 - t0_1 == 1) ? 1 : 0);
                        if (direct == 0) {
                            if (wtid < 128) {
                                float total = 0.0f;
                                for (int g_1 = 0; g_1 < 16; g_1++) {
                                    total = total + red_sum[0 + g_1 * 128 + wtid];
                                }
                                *(reinterpret_cast<float*>(partials + (((ri - num_tiles - num_tblk) * num_heads + head) * 128 + wtid)) + (0)) = total;
                            }
                            asm volatile("bar.sync 1, 256;" ::: "memory");
                            if (wtid == 0) {
                                unsigned int old;
                                asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], %2;"
                                    : "=r"(old) : "l"(&counters[seg_1 * num_heads + head_0_1]), "r"(static_cast<uint32_t>(1)) : "memory");
                                int last = (((int)old == t1 - t0_1 - 1) ? 1 : 0);
                                flag[0] = last;
                                if (last == 1) {
                                    *(reinterpret_cast<unsigned int*>(counters + (seg_1 * num_heads + head_0_1)) + (0)) = (unsigned int)0;
                                }
                            }
                            asm volatile("bar.sync 1, 256;" ::: "memory");
                        }
                        int is_last = ((direct == 0) ? flag[0] : 1);
                        if (is_last == 1) {
                            if (wtid < 128) {
                                float acc_a[1];
                                float fold_v[4];
                                acc_a[0] = 0.0f;
                                if (direct == 1) {
                                    for (int g_2 = 0; g_2 < 16; g_2++) {
                                        acc_a[0] = acc_a[0] + red_sum[0 + g_2 * 128 + wtid];
                                    }
                                } else {
                                    #pragma unroll 1
                                    for (int tt = t0_1; tt < t1; tt += 4) {
                                        for (int j = 0; j < 4; j++) {
                                            int ttj = ((t1 > tt + j) ? tt + j : t1 - 1);
                                            fold_v[j] = partials[(ttj * num_heads + head_0_1) * 128 + wtid];
                                        }
                                        for (int j_1 = 0; j_1 < 4; j_1++) {
                                            float fv = ((t1 > tt + j_1) ? fold_v[j_1] : 0.0f);
                                            acc_a[0] = acc_a[0] + fv;
                                        }
                                    }
                                }
                                if (length > 0) {
                                    float lenf = (float)length;
                                    float _fdiv_48 = __fdiv_rn(acc_a[0], lenf);
                                    *(reinterpret_cast<float*>(mean_k + ((seg_1 * num_heads + head_0_1) * 128 + wtid)) + (0)) = _fdiv_48;
                                }
                            }
                        }
                        asm volatile("bar.sync 1, 256;" ::: "memory");
                        if (wtid == 0) {
                            asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(hctr) + (num_heads + head))), "r"(static_cast<unsigned int>(1)) : "memory");
                        }
                    }
                } else {
                    if (wtid == 0) {
                        {
                        unsigned int _acquire_observed;
                        do {
                        asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(_acquire_observed) : "l"((reinterpret_cast<unsigned int*>(hctr) + (num_heads + head))) : "memory");
                        } while (static_cast<unsigned int>(_acquire_observed - static_cast<unsigned int>(num_tiles)) >= static_cast<unsigned int>(1));
                        }
                    }
                    asm volatile("bar.sync 1, 256;" ::: "memory");
                    int tok_1 = ri * 32 + trow;
                    int valid_1 = ((tok_1 < total_tokens) ? 1 : 0);
                    int tok_c_1 = ((tok_1 < total_tokens) ? tok_1 : total_tokens - 1);
                    float vec_load_9[16];
                    {
                        const uint4* _vptr_49 = reinterpret_cast<const uint4*>(K + ((tok_c_1 * num_heads + head) * 128 + col16) + 0);
                        uint4 _vld_49[2];
                        #pragma unroll
                        for (int _blk = 0; _blk < 2; _blk++) {
                            _vld_49[_blk] = _vptr_49[_blk];
                            uint32_t* _vpairs_49 = reinterpret_cast<uint32_t*>(&_vld_49[_blk]);
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
                                (&vec_load_9[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_49[_pair]) << 16);
                                (&vec_load_9[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_49[_pair]) & 0xffff0000u);
                            }
                        }
                    }
                    int seg_2 = tok_seg[tok_c_1];
                    int mbase = (seg_2 * num_heads + head) * 128 + col16;
                    float kc[16];
                    for (int quad_1 = 0; quad_1 < 4; quad_1++) {
                        float _vec_load_30[4];
                        {
                            float4 _v4 = *reinterpret_cast<const float4*>(mean_k + (mbase + quad_1 * 4) + 0);
                            _vec_load_30[0 + 0] = _v4.x;
                            _vec_load_30[0 + 1] = _v4.y;
                            _vec_load_30[0 + 2] = _v4.z;
                            _vec_load_30[0 + 3] = _v4.w;
                        }
                        for (int e_5 = 0; e_5 < 4; e_5++) {
                            float kf = vec_load_9[quad_1 * 4 + e_5];
                            kc[quad_1 * 4 + e_5] = ((valid_1 == 1) ? kf - _vec_load_30[e_5] : 0.0f);
                        }
                    }
                    unsigned int kw[2];
                    float ksf_a[1];
                    float max_1_1 = fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fabsf(kc[0]), fabsf(kc[1])), fabsf(kc[2])), fabsf(kc[3])), fabsf(kc[4])), fabsf(kc[5])), fabsf(kc[6])), fabsf(kc[7])), fabsf(kc[8])), fabsf(kc[9])), fabsf(kc[10])), fabsf(kc[11])), fabsf(kc[12])), fabsf(kc[13])), fabsf(kc[14])), fabsf(kc[15]));
                    uint16_t _cvt_satfinite_e4m3x2_f32_51;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_51) : "f"(0.0f), "f"(max_1_1 * 0.16666667f));
                    uint16_t _e4m3x2_decode_52 = (uint16_t)((unsigned int)_cvt_satfinite_e4m3x2_f32_51 & 0xFFu);
                    uint32_t _f16x2_decode_52;
                    float fp8_decode_1;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_decode_52) : "h"(_e4m3x2_decode_52));
                    uint16_t _f16_decode_52 = (uint16_t)_f16x2_decode_52;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(fp8_decode_1) : "h"(_f16_decode_52));
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
                    unsigned int _shfl_xor_sync_53 = __shfl_xor_sync(0xFFFFFFFF, kw[0], 1);
                    kstore[0] = _shfl_xor_sync_53;
                    unsigned int _shfl_xor_sync_54 = __shfl_xor_sync(0xFFFFFFFF, kw[1], 1);
                    kstore[1] = _shfl_xor_sync_54;
                    kstore[2] = kw[0];
                    kstore[3] = kw[1];
                    uint16_t _cvt_satfinite_e4m3x2_f32_55;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_55) : "f"(0.0f), "f"(ksf_a[0]));
                    *reinterpret_cast<unsigned char*>(k_sf + (((head * num_kblocks + (tok_1 >> 7)) * 128 + (tok_1 & 127)) * 8 + sub)) = (unsigned char)((unsigned int)(unsigned char)(_cvt_satfinite_e4m3x2_f32_55 & 127));
                    if (valid_1 == 1) {
                        if ((sub & 1) == 1) {
                            reinterpret_cast<int4*>(k4 + (((tok_1 * num_heads + head) * 64 + (sub - 1) * 8) / 4))[0] = reinterpret_cast<int4*>(kstore)[0];
                        }
                    }
                }
                if (wtid == 0) {
                    item_cell[0 + (it + 1 & 1)] = (int)nxt_r[0];
                }
                asm volatile("bar.sync 1, 256;" ::: "memory");
                if (wtid == 0) {
                    if (is_d == 1) {
                        unsigned int dold;
                        asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], %2;"
                            : "=r"(dold) : "l"(&hctr[2 * num_heads + head]), "r"(static_cast<uint32_t>(1)) : "memory");
                        if ((int)dold == num_tblk - 1) {
                            *(reinterpret_cast<unsigned int*>(hctr + (num_heads + head)) + (0)) = (unsigned int)0;
                            *(reinterpret_cast<unsigned int*>(hctr + (2 * num_heads + head)) + (0)) = (unsigned int)0;
                        }
                    }
                    unsigned int old_1;
                    asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], %2;"
                        : "=r"(old_1) : "l"(&hctr[head]), "r"(static_cast<uint32_t>(1)) : "memory");
                    if ((int)old_1 == per - 1) {
                        *(reinterpret_cast<unsigned int*>(hctr + head) + (0)) = (unsigned int)0;
                        asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(hctr) + (3 * num_heads + head))), "r"(static_cast<unsigned int>(1)) : "memory");
                    }
                }
            }
            asm volatile("barrier.sync 4, 768;" ::: "memory");
            if (tid == 0) {
                unsigned int eold;
                asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], %2;"
                    : "=r"(eold) : "l"(&hctr[5 * num_heads + 5]), "r"(static_cast<uint32_t>(1)) : "memory");
                if ((int)eold == num_bids - 1) {
                    *(reinterpret_cast<unsigned int*>(hctr + (5 * num_heads + 4)) + (0)) = (unsigned int)0;
                    *(reinterpret_cast<unsigned int*>(hctr + (5 * num_heads + 5)) + (0)) = (unsigned int)0;
                    asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(hctr) + (5 * num_heads + 2))), "r"(static_cast<unsigned int>(1)) : "memory");
                }
            }
        }
    }
    // ---- Role: wk1 ----
    if (warp >= 8 && warp <= 15) {
        { // wk1_main
            int wtid_1 = tid - 256;
            int col_1 = (wtid_1 & 15) * 8;
            int rgrp_1 = wtid_1 >> 4;
            int sub_1 = wtid_1 & 7;
            int trow_1 = wtid_1 >> 3;
            int col16_1 = sub_1 * 16;
            int num_tblk_1 = total_padded / 32;
            int vwarp_1 = wtid_1 >> 5;
            int vlane_1 = wtid_1 & 31;
            int vblk_1 = vwarp_1 & 1;
            int vdp_1 = (vwarp_1 >> 1) * 16 + (vlane_1 & 15);
            int vhalf_1 = vlane_1 >> 4;
            int n_abc_1 = 2 * num_tiles + num_tblk_1;
            int per_1 = n_abc_1 + num_tblk_1;
            asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
            unsigned int nxt_r_1[2];
            if (wtid_1 == 0) {
                unsigned int _atomic_fetch_add_0_1 = atomicAdd(&hctr[5 * num_heads + 4], 1);
                item_cell[4] = (int)_atomic_fetch_add_0_1;
            }
            asm volatile("bar.sync 2, 256;" ::: "memory");
            #pragma unroll 1
            for (int it_1 = 0; it_1 < per_1 * num_heads; it_1++) {
                int w_1 = item_cell[4 + (it_1 & 1)];
                if (w_1 >= per_1 * num_heads) {
                    break;
                }
                if (wtid_1 == 0) {
                    unsigned int _atomic_fetch_add_1_1 = atomicAdd(&hctr[5 * num_heads + 4], 1);
                    nxt_r_1[0] = _atomic_fetch_add_1_1;
                }
                int w2_1 = ((w_1 >= n_abc_1) ? w_1 - n_abc_1 : 0);
                int s_1 = w2_1 / per_1;
                int r_1 = w2_1 - s_1 * per_1;
                int last_s_1 = ((s_1 + 1 >= num_heads) ? 1 : 0);
                int is_d_1 = ((w_1 >= n_abc_1 && (last_s_1 == 1 || r_1 >= n_abc_1)) ? 1 : 0);
                int head_1 = ((w_1 < n_abc_1) ? 0 : ((is_d_1 == 1) ? s_1 : s_1 + 1));
                int ri_1 = ((w_1 < n_abc_1) ? w_1 : ((last_s_1 == 1) ? r_1 : ((is_d_1 == 1) ? r_1 - n_abc_1 : r_1)));
                if (is_d_1 == 0) {
                    if (ri_1 < num_tiles) {
                        int tile_2 = (ri_1 * num_heads + head_1) / num_heads;
                        int head_0_2 = (ri_1 * num_heads + head_1) % num_heads;
                        int _vec_load_0_1[4];
                        {
                            const int4* _ivptr_2 = reinterpret_cast<const int4*>(tile_table + (8 * tile_2) + 0);
                            int4 _ivld_2;
                            _ivld_2 = *_ivptr_2;
                            _vec_load_0_1[0 + 0] = _ivld_2.x;
                            _vec_load_0_1[0 + 1] = _ivld_2.y;
                            _vec_load_0_1[0 + 2] = _ivld_2.z;
                            _vec_load_0_1[0 + 3] = _ivld_2.w;
                        }
                        int seg_3 = _vec_load_0_1[0];
                        int row_begin_2 = _vec_load_0_1[1];
                        int row_end_2 = _vec_load_0_1[2];
                        int t0_2 = _vec_load_0_1[3];
                        int qt_base_1 = seg_tile_begin[seg_3] + (tile_2 - t0_2);
                        float qsum_1[8];
                        float qf8_1[8];
                        #pragma unroll 1
                        for (int batch_2 = 0; batch_2 < (row_end_2 - row_begin_2 + 128 - 1) / 128; batch_2++) {
                            for (int e_6 = 0; e_6 < 8; e_6++) {
                                qsum_1[e_6] = 0.0f;
                            }
                            int brow_2 = row_begin_2 + batch_2 * 128;
                            float vec_load_10[8];
                            {
                                const uint4* _vptr_3 = reinterpret_cast<const uint4*>(Q + ((((row_end_2 > brow_2 + rgrp_1) ? brow_2 + rgrp_1 : row_end_2 - 1) * num_heads + head_0_2) * 128 + col_1) + 0);
                                uint4 _vld_3[1];
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
                                    _vld_3[_blk] = _vptr_3[_blk];
                                    uint32_t* _vpairs_3 = reinterpret_cast<uint32_t*>(&_vld_3[_blk]);
                                    #pragma unroll
                                    for (int _pair = 0; _pair < 4; _pair++) {
                                        (&vec_load_10[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_3[_pair]) << 16);
                                        (&vec_load_10[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_3[_pair]) & 0xffff0000u);
                                    }
                                }
                            }
                            float vec_load_1_2[8];
                            {
                                const uint4* _vptr_4 = reinterpret_cast<const uint4*>(Q + ((((row_end_2 > brow_2 + 16 + rgrp_1) ? brow_2 + 16 + rgrp_1 : row_end_2 - 1) * num_heads + head_0_2) * 128 + col_1) + 0);
                                uint4 _vld_4[1];
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
                                    _vld_4[_blk] = _vptr_4[_blk];
                                    uint32_t* _vpairs_4 = reinterpret_cast<uint32_t*>(&_vld_4[_blk]);
                                    #pragma unroll
                                    for (int _pair = 0; _pair < 4; _pair++) {
                                        (&vec_load_1_2[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_4[_pair]) << 16);
                                        (&vec_load_1_2[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_4[_pair]) & 0xffff0000u);
                                    }
                                }
                            }
                            float vec_load_2_2[8];
                            {
                                const uint4* _vptr_5 = reinterpret_cast<const uint4*>(Q + ((((row_end_2 > brow_2 + 32 + rgrp_1) ? brow_2 + 32 + rgrp_1 : row_end_2 - 1) * num_heads + head_0_2) * 128 + col_1) + 0);
                                uint4 _vld_5[1];
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
                                    _vld_5[_blk] = _vptr_5[_blk];
                                    uint32_t* _vpairs_5 = reinterpret_cast<uint32_t*>(&_vld_5[_blk]);
                                    #pragma unroll
                                    for (int _pair = 0; _pair < 4; _pair++) {
                                        (&vec_load_2_2[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_5[_pair]) << 16);
                                        (&vec_load_2_2[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_5[_pair]) & 0xffff0000u);
                                    }
                                }
                            }
                            float vec_load_3_2[8];
                            {
                                const uint4* _vptr_6 = reinterpret_cast<const uint4*>(Q + ((((row_end_2 > brow_2 + 48 + rgrp_1) ? brow_2 + 48 + rgrp_1 : row_end_2 - 1) * num_heads + head_0_2) * 128 + col_1) + 0);
                                uint4 _vld_6[1];
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
                                    _vld_6[_blk] = _vptr_6[_blk];
                                    uint32_t* _vpairs_6 = reinterpret_cast<uint32_t*>(&_vld_6[_blk]);
                                    #pragma unroll
                                    for (int _pair = 0; _pair < 4; _pair++) {
                                        (&vec_load_3_2[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_6[_pair]) << 16);
                                        (&vec_load_3_2[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_6[_pair]) & 0xffff0000u);
                                    }
                                }
                            }
                            float vec_load_4_2[8];
                            {
                                const uint4* _vptr_7 = reinterpret_cast<const uint4*>(Q + ((((row_end_2 > brow_2 + 64 + rgrp_1) ? brow_2 + 64 + rgrp_1 : row_end_2 - 1) * num_heads + head_0_2) * 128 + col_1) + 0);
                                uint4 _vld_7[1];
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
                                    _vld_7[_blk] = _vptr_7[_blk];
                                    uint32_t* _vpairs_7 = reinterpret_cast<uint32_t*>(&_vld_7[_blk]);
                                    #pragma unroll
                                    for (int _pair = 0; _pair < 4; _pair++) {
                                        (&vec_load_4_2[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_7[_pair]) << 16);
                                        (&vec_load_4_2[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_7[_pair]) & 0xffff0000u);
                                    }
                                }
                            }
                            float vec_load_5_2[8];
                            {
                                const uint4* _vptr_8 = reinterpret_cast<const uint4*>(Q + ((((row_end_2 > brow_2 + 80 + rgrp_1) ? brow_2 + 80 + rgrp_1 : row_end_2 - 1) * num_heads + head_0_2) * 128 + col_1) + 0);
                                uint4 _vld_8[1];
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
                                    _vld_8[_blk] = _vptr_8[_blk];
                                    uint32_t* _vpairs_8 = reinterpret_cast<uint32_t*>(&_vld_8[_blk]);
                                    #pragma unroll
                                    for (int _pair = 0; _pair < 4; _pair++) {
                                        (&vec_load_5_2[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_8[_pair]) << 16);
                                        (&vec_load_5_2[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_8[_pair]) & 0xffff0000u);
                                    }
                                }
                            }
                            float vec_load_6_2[8];
                            {
                                const uint4* _vptr_9 = reinterpret_cast<const uint4*>(Q + ((((row_end_2 > brow_2 + 96 + rgrp_1) ? brow_2 + 96 + rgrp_1 : row_end_2 - 1) * num_heads + head_0_2) * 128 + col_1) + 0);
                                uint4 _vld_9[1];
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
                                    _vld_9[_blk] = _vptr_9[_blk];
                                    uint32_t* _vpairs_9 = reinterpret_cast<uint32_t*>(&_vld_9[_blk]);
                                    #pragma unroll
                                    for (int _pair = 0; _pair < 4; _pair++) {
                                        (&vec_load_6_2[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_9[_pair]) << 16);
                                        (&vec_load_6_2[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_9[_pair]) & 0xffff0000u);
                                    }
                                }
                            }
                            float vec_load_7_2[8];
                            {
                                const uint4* _vptr_10 = reinterpret_cast<const uint4*>(Q + ((((row_end_2 > brow_2 + 112 + rgrp_1) ? brow_2 + 112 + rgrp_1 : row_end_2 - 1) * num_heads + head_0_2) * 128 + col_1) + 0);
                                uint4 _vld_10[1];
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
                                    _vld_10[_blk] = _vptr_10[_blk];
                                    uint32_t* _vpairs_10 = reinterpret_cast<uint32_t*>(&_vld_10[_blk]);
                                    #pragma unroll
                                    for (int _pair = 0; _pair < 4; _pair++) {
                                        (&vec_load_7_2[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_10[_pair]) << 16);
                                        (&vec_load_7_2[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_10[_pair]) & 0xffff0000u);
                                    }
                                }
                            }
                            qf8_1[0] = vec_load_10[0];
                            qf8_1[1] = vec_load_10[1];
                            qf8_1[2] = vec_load_10[2];
                            qf8_1[3] = vec_load_10[3];
                            qf8_1[4] = vec_load_10[4];
                            qf8_1[5] = vec_load_10[5];
                            qf8_1[6] = vec_load_10[6];
                            qf8_1[7] = vec_load_10[7];
                            qsum_1[0] = qsum_1[0] + ((row_end_2 > brow_2 + rgrp_1) ? qf8_1[0] : 0.0f);
                            qsum_1[1] = qsum_1[1] + ((row_end_2 > brow_2 + rgrp_1) ? qf8_1[1] : 0.0f);
                            qsum_1[2] = qsum_1[2] + ((row_end_2 > brow_2 + rgrp_1) ? qf8_1[2] : 0.0f);
                            qsum_1[3] = qsum_1[3] + ((row_end_2 > brow_2 + rgrp_1) ? qf8_1[3] : 0.0f);
                            qsum_1[4] = qsum_1[4] + ((row_end_2 > brow_2 + rgrp_1) ? qf8_1[4] : 0.0f);
                            qsum_1[5] = qsum_1[5] + ((row_end_2 > brow_2 + rgrp_1) ? qf8_1[5] : 0.0f);
                            qsum_1[6] = qsum_1[6] + ((row_end_2 > brow_2 + rgrp_1) ? qf8_1[6] : 0.0f);
                            qsum_1[7] = qsum_1[7] + ((row_end_2 > brow_2 + rgrp_1) ? qf8_1[7] : 0.0f);
                            qf8_1[0] = vec_load_1_2[0];
                            qf8_1[1] = vec_load_1_2[1];
                            qf8_1[2] = vec_load_1_2[2];
                            qf8_1[3] = vec_load_1_2[3];
                            qf8_1[4] = vec_load_1_2[4];
                            qf8_1[5] = vec_load_1_2[5];
                            qf8_1[6] = vec_load_1_2[6];
                            qf8_1[7] = vec_load_1_2[7];
                            qsum_1[0] = qsum_1[0] + ((row_end_2 > brow_2 + 16 + rgrp_1) ? qf8_1[0] : 0.0f);
                            qsum_1[1] = qsum_1[1] + ((row_end_2 > brow_2 + 16 + rgrp_1) ? qf8_1[1] : 0.0f);
                            qsum_1[2] = qsum_1[2] + ((row_end_2 > brow_2 + 16 + rgrp_1) ? qf8_1[2] : 0.0f);
                            qsum_1[3] = qsum_1[3] + ((row_end_2 > brow_2 + 16 + rgrp_1) ? qf8_1[3] : 0.0f);
                            qsum_1[4] = qsum_1[4] + ((row_end_2 > brow_2 + 16 + rgrp_1) ? qf8_1[4] : 0.0f);
                            qsum_1[5] = qsum_1[5] + ((row_end_2 > brow_2 + 16 + rgrp_1) ? qf8_1[5] : 0.0f);
                            qsum_1[6] = qsum_1[6] + ((row_end_2 > brow_2 + 16 + rgrp_1) ? qf8_1[6] : 0.0f);
                            qsum_1[7] = qsum_1[7] + ((row_end_2 > brow_2 + 16 + rgrp_1) ? qf8_1[7] : 0.0f);
                            qf8_1[0] = vec_load_2_2[0];
                            qf8_1[1] = vec_load_2_2[1];
                            qf8_1[2] = vec_load_2_2[2];
                            qf8_1[3] = vec_load_2_2[3];
                            qf8_1[4] = vec_load_2_2[4];
                            qf8_1[5] = vec_load_2_2[5];
                            qf8_1[6] = vec_load_2_2[6];
                            qf8_1[7] = vec_load_2_2[7];
                            qsum_1[0] = qsum_1[0] + ((row_end_2 > brow_2 + 32 + rgrp_1) ? qf8_1[0] : 0.0f);
                            qsum_1[1] = qsum_1[1] + ((row_end_2 > brow_2 + 32 + rgrp_1) ? qf8_1[1] : 0.0f);
                            qsum_1[2] = qsum_1[2] + ((row_end_2 > brow_2 + 32 + rgrp_1) ? qf8_1[2] : 0.0f);
                            qsum_1[3] = qsum_1[3] + ((row_end_2 > brow_2 + 32 + rgrp_1) ? qf8_1[3] : 0.0f);
                            qsum_1[4] = qsum_1[4] + ((row_end_2 > brow_2 + 32 + rgrp_1) ? qf8_1[4] : 0.0f);
                            qsum_1[5] = qsum_1[5] + ((row_end_2 > brow_2 + 32 + rgrp_1) ? qf8_1[5] : 0.0f);
                            qsum_1[6] = qsum_1[6] + ((row_end_2 > brow_2 + 32 + rgrp_1) ? qf8_1[6] : 0.0f);
                            qsum_1[7] = qsum_1[7] + ((row_end_2 > brow_2 + 32 + rgrp_1) ? qf8_1[7] : 0.0f);
                            qf8_1[0] = vec_load_3_2[0];
                            qf8_1[1] = vec_load_3_2[1];
                            qf8_1[2] = vec_load_3_2[2];
                            qf8_1[3] = vec_load_3_2[3];
                            qf8_1[4] = vec_load_3_2[4];
                            qf8_1[5] = vec_load_3_2[5];
                            qf8_1[6] = vec_load_3_2[6];
                            qf8_1[7] = vec_load_3_2[7];
                            qsum_1[0] = qsum_1[0] + ((row_end_2 > brow_2 + 48 + rgrp_1) ? qf8_1[0] : 0.0f);
                            qsum_1[1] = qsum_1[1] + ((row_end_2 > brow_2 + 48 + rgrp_1) ? qf8_1[1] : 0.0f);
                            qsum_1[2] = qsum_1[2] + ((row_end_2 > brow_2 + 48 + rgrp_1) ? qf8_1[2] : 0.0f);
                            qsum_1[3] = qsum_1[3] + ((row_end_2 > brow_2 + 48 + rgrp_1) ? qf8_1[3] : 0.0f);
                            qsum_1[4] = qsum_1[4] + ((row_end_2 > brow_2 + 48 + rgrp_1) ? qf8_1[4] : 0.0f);
                            qsum_1[5] = qsum_1[5] + ((row_end_2 > brow_2 + 48 + rgrp_1) ? qf8_1[5] : 0.0f);
                            qsum_1[6] = qsum_1[6] + ((row_end_2 > brow_2 + 48 + rgrp_1) ? qf8_1[6] : 0.0f);
                            qsum_1[7] = qsum_1[7] + ((row_end_2 > brow_2 + 48 + rgrp_1) ? qf8_1[7] : 0.0f);
                            qf8_1[0] = vec_load_4_2[0];
                            qf8_1[1] = vec_load_4_2[1];
                            qf8_1[2] = vec_load_4_2[2];
                            qf8_1[3] = vec_load_4_2[3];
                            qf8_1[4] = vec_load_4_2[4];
                            qf8_1[5] = vec_load_4_2[5];
                            qf8_1[6] = vec_load_4_2[6];
                            qf8_1[7] = vec_load_4_2[7];
                            qsum_1[0] = qsum_1[0] + ((row_end_2 > brow_2 + 64 + rgrp_1) ? qf8_1[0] : 0.0f);
                            qsum_1[1] = qsum_1[1] + ((row_end_2 > brow_2 + 64 + rgrp_1) ? qf8_1[1] : 0.0f);
                            qsum_1[2] = qsum_1[2] + ((row_end_2 > brow_2 + 64 + rgrp_1) ? qf8_1[2] : 0.0f);
                            qsum_1[3] = qsum_1[3] + ((row_end_2 > brow_2 + 64 + rgrp_1) ? qf8_1[3] : 0.0f);
                            qsum_1[4] = qsum_1[4] + ((row_end_2 > brow_2 + 64 + rgrp_1) ? qf8_1[4] : 0.0f);
                            qsum_1[5] = qsum_1[5] + ((row_end_2 > brow_2 + 64 + rgrp_1) ? qf8_1[5] : 0.0f);
                            qsum_1[6] = qsum_1[6] + ((row_end_2 > brow_2 + 64 + rgrp_1) ? qf8_1[6] : 0.0f);
                            qsum_1[7] = qsum_1[7] + ((row_end_2 > brow_2 + 64 + rgrp_1) ? qf8_1[7] : 0.0f);
                            qf8_1[0] = vec_load_5_2[0];
                            qf8_1[1] = vec_load_5_2[1];
                            qf8_1[2] = vec_load_5_2[2];
                            qf8_1[3] = vec_load_5_2[3];
                            qf8_1[4] = vec_load_5_2[4];
                            qf8_1[5] = vec_load_5_2[5];
                            qf8_1[6] = vec_load_5_2[6];
                            qf8_1[7] = vec_load_5_2[7];
                            qsum_1[0] = qsum_1[0] + ((row_end_2 > brow_2 + 80 + rgrp_1) ? qf8_1[0] : 0.0f);
                            qsum_1[1] = qsum_1[1] + ((row_end_2 > brow_2 + 80 + rgrp_1) ? qf8_1[1] : 0.0f);
                            qsum_1[2] = qsum_1[2] + ((row_end_2 > brow_2 + 80 + rgrp_1) ? qf8_1[2] : 0.0f);
                            qsum_1[3] = qsum_1[3] + ((row_end_2 > brow_2 + 80 + rgrp_1) ? qf8_1[3] : 0.0f);
                            qsum_1[4] = qsum_1[4] + ((row_end_2 > brow_2 + 80 + rgrp_1) ? qf8_1[4] : 0.0f);
                            qsum_1[5] = qsum_1[5] + ((row_end_2 > brow_2 + 80 + rgrp_1) ? qf8_1[5] : 0.0f);
                            qsum_1[6] = qsum_1[6] + ((row_end_2 > brow_2 + 80 + rgrp_1) ? qf8_1[6] : 0.0f);
                            qsum_1[7] = qsum_1[7] + ((row_end_2 > brow_2 + 80 + rgrp_1) ? qf8_1[7] : 0.0f);
                            qf8_1[0] = vec_load_6_2[0];
                            qf8_1[1] = vec_load_6_2[1];
                            qf8_1[2] = vec_load_6_2[2];
                            qf8_1[3] = vec_load_6_2[3];
                            qf8_1[4] = vec_load_6_2[4];
                            qf8_1[5] = vec_load_6_2[5];
                            qf8_1[6] = vec_load_6_2[6];
                            qf8_1[7] = vec_load_6_2[7];
                            qsum_1[0] = qsum_1[0] + ((row_end_2 > brow_2 + 96 + rgrp_1) ? qf8_1[0] : 0.0f);
                            qsum_1[1] = qsum_1[1] + ((row_end_2 > brow_2 + 96 + rgrp_1) ? qf8_1[1] : 0.0f);
                            qsum_1[2] = qsum_1[2] + ((row_end_2 > brow_2 + 96 + rgrp_1) ? qf8_1[2] : 0.0f);
                            qsum_1[3] = qsum_1[3] + ((row_end_2 > brow_2 + 96 + rgrp_1) ? qf8_1[3] : 0.0f);
                            qsum_1[4] = qsum_1[4] + ((row_end_2 > brow_2 + 96 + rgrp_1) ? qf8_1[4] : 0.0f);
                            qsum_1[5] = qsum_1[5] + ((row_end_2 > brow_2 + 96 + rgrp_1) ? qf8_1[5] : 0.0f);
                            qsum_1[6] = qsum_1[6] + ((row_end_2 > brow_2 + 96 + rgrp_1) ? qf8_1[6] : 0.0f);
                            qsum_1[7] = qsum_1[7] + ((row_end_2 > brow_2 + 96 + rgrp_1) ? qf8_1[7] : 0.0f);
                            qf8_1[0] = vec_load_7_2[0];
                            qf8_1[1] = vec_load_7_2[1];
                            qf8_1[2] = vec_load_7_2[2];
                            qf8_1[3] = vec_load_7_2[3];
                            qf8_1[4] = vec_load_7_2[4];
                            qf8_1[5] = vec_load_7_2[5];
                            qf8_1[6] = vec_load_7_2[6];
                            qf8_1[7] = vec_load_7_2[7];
                            qsum_1[0] = qsum_1[0] + ((row_end_2 > brow_2 + 112 + rgrp_1) ? qf8_1[0] : 0.0f);
                            qsum_1[1] = qsum_1[1] + ((row_end_2 > brow_2 + 112 + rgrp_1) ? qf8_1[1] : 0.0f);
                            qsum_1[2] = qsum_1[2] + ((row_end_2 > brow_2 + 112 + rgrp_1) ? qf8_1[2] : 0.0f);
                            qsum_1[3] = qsum_1[3] + ((row_end_2 > brow_2 + 112 + rgrp_1) ? qf8_1[3] : 0.0f);
                            qsum_1[4] = qsum_1[4] + ((row_end_2 > brow_2 + 112 + rgrp_1) ? qf8_1[4] : 0.0f);
                            qsum_1[5] = qsum_1[5] + ((row_end_2 > brow_2 + 112 + rgrp_1) ? qf8_1[5] : 0.0f);
                            qsum_1[6] = qsum_1[6] + ((row_end_2 > brow_2 + 112 + rgrp_1) ? qf8_1[6] : 0.0f);
                            qsum_1[7] = qsum_1[7] + ((row_end_2 > brow_2 + 112 + rgrp_1) ? qf8_1[7] : 0.0f);
                            for (int e_7 = 0; e_7 < 8; e_7++) {
                                red_q[2048 + rgrp_1 * 128 + col_1 + e_7] = qsum_1[e_7];
                            }
                            asm volatile("bar.sync 2, 256;" ::: "memory");
                            int qt_1 = qt_base_1 + batch_2;
                            if (wtid_1 < 128) {
                                float qtot_1 = 0.0f;
                                for (int g_3 = 0; g_3 < 16; g_3++) {
                                    qtot_1 = qtot_1 + red_q[2048 + g_3 * 128 + wtid_1];
                                }
                                int brows_1 = ((row_end_2 - brow_2 < 128) ? row_end_2 - brow_2 : 128);
                                float _fdiv_11_1 = __fdiv_rn(qtot_1, (float)brows_1);
                                *(reinterpret_cast<float*>(q_mean + ((qt_1 * num_heads + head_0_2) * 128 + wtid_1)) + (0)) = _fdiv_11_1;
                            }
                            asm volatile("bar.sync 2, 256;" ::: "memory");
                            #pragma unroll 1
                            for (int blk_1 = 0; blk_1 < 4; blk_1++) {
                                int tok_2 = brow_2 + blk_1 * 32 + trow_1;
                                int valid_2 = ((tok_2 < row_end_2) ? 1 : 0);
                                int tok_c_2 = ((tok_2 < row_end_2) ? tok_2 : row_end_2 - 1);
                                int is_first_1 = ((valid_2 == 1 && tok_2 == brow_2) ? 1 : 0);
                                float vec_load_8_1[16];
                                {
                                    const uint4* _vptr_12 = reinterpret_cast<const uint4*>(Q + ((tok_c_2 * num_heads + head_0_2) * 128 + col16_1) + 0);
                                    uint4 _vld_12[2];
                                    #pragma unroll
                                    for (int _blk = 0; _blk < 2; _blk++) {
                                        _vld_12[_blk] = _vptr_12[_blk];
                                        uint32_t* _vpairs_12 = reinterpret_cast<uint32_t*>(&_vld_12[_blk]);
                                        #pragma unroll
                                        for (int _pair = 0; _pair < 4; _pair++) {
                                            (&vec_load_8_1[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_12[_pair]) << 16);
                                            (&vec_load_8_1[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_12[_pair]) & 0xffff0000u);
                                        }
                                    }
                                }
                                int qmbase_1 = (qt_1 * num_heads + head_0_2) * 128 + col16_1;
                                float qmv_1[16];
                                float qc_1[16];
                                for (int quad_2 = 0; quad_2 < 4; quad_2++) {
                                    float _vec_load_10_1[4];
                                    {
                                        float4 _v4 = *reinterpret_cast<const float4*>(q_mean + (qmbase_1 + quad_2 * 4) + 0);
                                        _vec_load_10_1[0 + 0] = _v4.x;
                                        _vec_load_10_1[0 + 1] = _v4.y;
                                        _vec_load_10_1[0 + 2] = _v4.z;
                                        _vec_load_10_1[0 + 3] = _v4.w;
                                    }
                                    for (int e_8 = 0; e_8 < 4; e_8++) {
                                        qmv_1[quad_2 * 4 + e_8] = _vec_load_10_1[e_8];
                                        float qf_1 = vec_load_8_1[quad_2 * 4 + e_8];
                                        qc_1[quad_2 * 4 + e_8] = ((valid_2 == 1) ? qf_1 - _vec_load_10_1[e_8] : 0.0f);
                                    }
                                }
                                unsigned int qw_1[2];
                                float qsf_a_1[1];
                                float max_1_2 = fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fabsf(qc_1[0]), fabsf(qc_1[1])), fabsf(qc_1[2])), fabsf(qc_1[3])), fabsf(qc_1[4])), fabsf(qc_1[5])), fabsf(qc_1[6])), fabsf(qc_1[7])), fabsf(qc_1[8])), fabsf(qc_1[9])), fabsf(qc_1[10])), fabsf(qc_1[11])), fabsf(qc_1[12])), fabsf(qc_1[13])), fabsf(qc_1[14])), fabsf(qc_1[15]));
                                uint16_t _cvt_satfinite_e4m3x2_f32_14_1;
                                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_14_1) : "f"(0.0f), "f"(max_1_2 * 0.16666667f));
                                uint16_t _e4m3x2_decode_15 = (uint16_t)((unsigned int)_cvt_satfinite_e4m3x2_f32_14_1 & 0xFFu);
                                uint32_t _f16x2_decode_15;
                                float fp8_decode_2;
                                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_decode_15) : "h"(_e4m3x2_decode_15));
                                uint16_t _f16_decode_15 = (uint16_t)_f16x2_decode_15;
                                asm("cvt.f32.f16 %0, %1;" : "=f"(fp8_decode_2) : "h"(_f16_decode_15));
                                float fdiv_rn_2 = __fdiv_rn(1.0f, fp8_decode_2);
                                qc_1[0] = qc_1[0] * ((fp8_decode_2 != 0.0f) ? fdiv_rn_2 : 0.0f);
                                qc_1[1] = qc_1[1] * ((fp8_decode_2 != 0.0f) ? fdiv_rn_2 : 0.0f);
                                qc_1[2] = qc_1[2] * ((fp8_decode_2 != 0.0f) ? fdiv_rn_2 : 0.0f);
                                qc_1[3] = qc_1[3] * ((fp8_decode_2 != 0.0f) ? fdiv_rn_2 : 0.0f);
                                qc_1[4] = qc_1[4] * ((fp8_decode_2 != 0.0f) ? fdiv_rn_2 : 0.0f);
                                qc_1[5] = qc_1[5] * ((fp8_decode_2 != 0.0f) ? fdiv_rn_2 : 0.0f);
                                qc_1[6] = qc_1[6] * ((fp8_decode_2 != 0.0f) ? fdiv_rn_2 : 0.0f);
                                qc_1[7] = qc_1[7] * ((fp8_decode_2 != 0.0f) ? fdiv_rn_2 : 0.0f);
                                qc_1[8] = qc_1[8] * ((fp8_decode_2 != 0.0f) ? fdiv_rn_2 : 0.0f);
                                qc_1[9] = qc_1[9] * ((fp8_decode_2 != 0.0f) ? fdiv_rn_2 : 0.0f);
                                qc_1[10] = qc_1[10] * ((fp8_decode_2 != 0.0f) ? fdiv_rn_2 : 0.0f);
                                qc_1[11] = qc_1[11] * ((fp8_decode_2 != 0.0f) ? fdiv_rn_2 : 0.0f);
                                qc_1[12] = qc_1[12] * ((fp8_decode_2 != 0.0f) ? fdiv_rn_2 : 0.0f);
                                qc_1[13] = qc_1[13] * ((fp8_decode_2 != 0.0f) ? fdiv_rn_2 : 0.0f);
                                qc_1[14] = qc_1[14] * ((fp8_decode_2 != 0.0f) ? fdiv_rn_2 : 0.0f);
                                qc_1[15] = qc_1[15] * ((fp8_decode_2 != 0.0f) ? fdiv_rn_2 : 0.0f);
                                asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(qw_1[0]) : "f"(qc_1[0]), "f"(qc_1[1]), "f"(qc_1[2]), "f"(qc_1[3]), "f"(qc_1[4]), "f"(qc_1[5]), "f"(qc_1[6]), "f"(qc_1[7]));
                                asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(qw_1[1]) : "f"(qc_1[8]), "f"(qc_1[9]), "f"(qc_1[10]), "f"(qc_1[11]), "f"(qc_1[12]), "f"(qc_1[13]), "f"(qc_1[14]), "f"(qc_1[15]));
                                qsf_a_1[0] = max_1_2 * 0.16666667f;
                                unsigned int qstore_1[4];
                                qstore_1[0] = qw_1[0];
                                qstore_1[1] = qw_1[1];
                                unsigned int _shfl_xor_sync_16_1 = __shfl_xor_sync(0xFFFFFFFF, qw_1[0], 1);
                                qstore_1[2] = _shfl_xor_sync_16_1;
                                unsigned int _shfl_xor_sync_17_1 = __shfl_xor_sync(0xFFFFFFFF, qw_1[1], 1);
                                qstore_1[3] = _shfl_xor_sync_17_1;
                                if (valid_2 == 1) {
                                    uint16_t _cvt_satfinite_e4m3x2_f32_18_1;
                                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_18_1) : "f"(0.0f), "f"(qsf_a_1[0]));
                                    *reinterpret_cast<unsigned char*>(q_sf + ((tok_2 * num_heads + head_0_2) * 8 + sub_1)) = (unsigned char)((unsigned int)(unsigned char)(_cvt_satfinite_e4m3x2_f32_18_1 & 127));
                                    if ((sub_1 & 1) == 0) {
                                        reinterpret_cast<int4*>(q4 + (((tok_2 * num_heads + head_0_2) * 64 + sub_1 * 8) / 4))[0] = reinterpret_cast<int4*>(qstore_1)[0];
                                    }
                                }
                                if (is_first_1 == 1) {
                                    unsigned int mw_1[2];
                                    float msf_a_1[1];
                                    float max_1_1_1 = fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fabsf(qmv_1[0]), fabsf(qmv_1[1])), fabsf(qmv_1[2])), fabsf(qmv_1[3])), fabsf(qmv_1[4])), fabsf(qmv_1[5])), fabsf(qmv_1[6])), fabsf(qmv_1[7])), fabsf(qmv_1[8])), fabsf(qmv_1[9])), fabsf(qmv_1[10])), fabsf(qmv_1[11])), fabsf(qmv_1[12])), fabsf(qmv_1[13])), fabsf(qmv_1[14])), fabsf(qmv_1[15]));
                                    uint16_t _cvt_satfinite_e4m3x2_f32_19_1;
                                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_19_1) : "f"(0.0f), "f"(max_1_1_1 * 0.16666667f));
                                    uint16_t _e4m3x2_decode_20 = (uint16_t)((unsigned int)_cvt_satfinite_e4m3x2_f32_19_1 & 0xFFu);
                                    uint32_t _f16x2_decode_20;
                                    float fp8_decode_1_1;
                                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_decode_20) : "h"(_e4m3x2_decode_20));
                                    uint16_t _f16_decode_20 = (uint16_t)_f16x2_decode_20;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(fp8_decode_1_1) : "h"(_f16_decode_20));
                                    float fdiv_rn_1_1 = __fdiv_rn(1.0f, fp8_decode_1_1);
                                    qmv_1[0] = qmv_1[0] * ((fp8_decode_1_1 != 0.0f) ? fdiv_rn_1_1 : 0.0f);
                                    qmv_1[1] = qmv_1[1] * ((fp8_decode_1_1 != 0.0f) ? fdiv_rn_1_1 : 0.0f);
                                    qmv_1[2] = qmv_1[2] * ((fp8_decode_1_1 != 0.0f) ? fdiv_rn_1_1 : 0.0f);
                                    qmv_1[3] = qmv_1[3] * ((fp8_decode_1_1 != 0.0f) ? fdiv_rn_1_1 : 0.0f);
                                    qmv_1[4] = qmv_1[4] * ((fp8_decode_1_1 != 0.0f) ? fdiv_rn_1_1 : 0.0f);
                                    qmv_1[5] = qmv_1[5] * ((fp8_decode_1_1 != 0.0f) ? fdiv_rn_1_1 : 0.0f);
                                    qmv_1[6] = qmv_1[6] * ((fp8_decode_1_1 != 0.0f) ? fdiv_rn_1_1 : 0.0f);
                                    qmv_1[7] = qmv_1[7] * ((fp8_decode_1_1 != 0.0f) ? fdiv_rn_1_1 : 0.0f);
                                    qmv_1[8] = qmv_1[8] * ((fp8_decode_1_1 != 0.0f) ? fdiv_rn_1_1 : 0.0f);
                                    qmv_1[9] = qmv_1[9] * ((fp8_decode_1_1 != 0.0f) ? fdiv_rn_1_1 : 0.0f);
                                    qmv_1[10] = qmv_1[10] * ((fp8_decode_1_1 != 0.0f) ? fdiv_rn_1_1 : 0.0f);
                                    qmv_1[11] = qmv_1[11] * ((fp8_decode_1_1 != 0.0f) ? fdiv_rn_1_1 : 0.0f);
                                    qmv_1[12] = qmv_1[12] * ((fp8_decode_1_1 != 0.0f) ? fdiv_rn_1_1 : 0.0f);
                                    qmv_1[13] = qmv_1[13] * ((fp8_decode_1_1 != 0.0f) ? fdiv_rn_1_1 : 0.0f);
                                    qmv_1[14] = qmv_1[14] * ((fp8_decode_1_1 != 0.0f) ? fdiv_rn_1_1 : 0.0f);
                                    qmv_1[15] = qmv_1[15] * ((fp8_decode_1_1 != 0.0f) ? fdiv_rn_1_1 : 0.0f);
                                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(mw_1[0]) : "f"(qmv_1[0]), "f"(qmv_1[1]), "f"(qmv_1[2]), "f"(qmv_1[3]), "f"(qmv_1[4]), "f"(qmv_1[5]), "f"(qmv_1[6]), "f"(qmv_1[7]));
                                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(mw_1[1]) : "f"(qmv_1[8]), "f"(qmv_1[9]), "f"(qmv_1[10]), "f"(qmv_1[11]), "f"(qmv_1[12]), "f"(qmv_1[13]), "f"(qmv_1[14]), "f"(qmv_1[15]));
                                    msf_a_1[0] = max_1_1_1 * 0.16666667f;
                                    int qmrow_1 = qt_1 * num_heads + head_0_2;
                                    *(reinterpret_cast<unsigned int*>(qm4 + (qmrow_1 * 16 + sub_1 * 2)) + (0)) = mw_1[0];
                                    *(reinterpret_cast<unsigned int*>(qm4 + (qmrow_1 * 16 + sub_1 * 2 + 1)) + (0)) = mw_1[1];
                                    uint16_t _cvt_satfinite_e4m3x2_f32_21_1;
                                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_21_1) : "f"(0.0f), "f"(msf_a_1[0]));
                                    *reinterpret_cast<unsigned char*>(qm_sf + (qmrow_1 * 8 + sub_1)) = (unsigned char)((unsigned int)(unsigned char)(_cvt_satfinite_e4m3x2_f32_21_1 & 127));
                                }
                            }
                        }
                    } else if (ri_1 < num_tiles + num_tblk_1) {
                        int vlive_1[8];
                        vlive_1[0] = (((ri_1 - num_tiles) * 32 + (vblk_1 * 4 + vhalf_1 * 2) < total_tokens) ? 1 : 0);
                        float _vec_load_11_1[2];
                        {
                            uint32_t _bf16x2_bits_22;
                            _bf16x2_bits_22 = *reinterpret_cast<const uint32_t*>(V + (((((ri_1 - num_tiles) * 32 + (vblk_1 * 4 + vhalf_1 * 2) < total_tokens) ? (ri_1 - num_tiles) * 32 + (vblk_1 * 4 + vhalf_1 * 2) : total_tokens - 1) * num_heads + head_1) * 128 + vdp_1 * 2) + 0);
                            (&_vec_load_11_1[0])[0] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_22) << 16);
                            (&_vec_load_11_1[0])[1] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_22) & 0xffff0000u);
                        }
                        vlive_1[1] = (((ri_1 - num_tiles) * 32 + (vblk_1 * 4 + vhalf_1 * 2 + 1) < total_tokens) ? 1 : 0);
                        float _vec_load_12_1[2];
                        {
                            uint32_t _bf16x2_bits_23;
                            _bf16x2_bits_23 = *reinterpret_cast<const uint32_t*>(V + (((((ri_1 - num_tiles) * 32 + (vblk_1 * 4 + vhalf_1 * 2 + 1) < total_tokens) ? (ri_1 - num_tiles) * 32 + (vblk_1 * 4 + vhalf_1 * 2 + 1) : total_tokens - 1) * num_heads + head_1) * 128 + vdp_1 * 2) + 0);
                            (&_vec_load_12_1[0])[0] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_23) << 16);
                            (&_vec_load_12_1[0])[1] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_23) & 0xffff0000u);
                        }
                        vlive_1[2] = (((ri_1 - num_tiles) * 32 + (8 + vblk_1 * 4 + vhalf_1 * 2) < total_tokens) ? 1 : 0);
                        float _vec_load_13_1[2];
                        {
                            uint32_t _bf16x2_bits_24;
                            _bf16x2_bits_24 = *reinterpret_cast<const uint32_t*>(V + (((((ri_1 - num_tiles) * 32 + (8 + vblk_1 * 4 + vhalf_1 * 2) < total_tokens) ? (ri_1 - num_tiles) * 32 + (8 + vblk_1 * 4 + vhalf_1 * 2) : total_tokens - 1) * num_heads + head_1) * 128 + vdp_1 * 2) + 0);
                            (&_vec_load_13_1[0])[0] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_24) << 16);
                            (&_vec_load_13_1[0])[1] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_24) & 0xffff0000u);
                        }
                        vlive_1[3] = (((ri_1 - num_tiles) * 32 + (8 + vblk_1 * 4 + vhalf_1 * 2 + 1) < total_tokens) ? 1 : 0);
                        float _vec_load_14_1[2];
                        {
                            uint32_t _bf16x2_bits_25;
                            _bf16x2_bits_25 = *reinterpret_cast<const uint32_t*>(V + (((((ri_1 - num_tiles) * 32 + (8 + vblk_1 * 4 + vhalf_1 * 2 + 1) < total_tokens) ? (ri_1 - num_tiles) * 32 + (8 + vblk_1 * 4 + vhalf_1 * 2 + 1) : total_tokens - 1) * num_heads + head_1) * 128 + vdp_1 * 2) + 0);
                            (&_vec_load_14_1[0])[0] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_25) << 16);
                            (&_vec_load_14_1[0])[1] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_25) & 0xffff0000u);
                        }
                        vlive_1[4] = (((ri_1 - num_tiles) * 32 + (16 + vblk_1 * 4 + vhalf_1 * 2) < total_tokens) ? 1 : 0);
                        float _vec_load_15_1[2];
                        {
                            uint32_t _bf16x2_bits_26;
                            _bf16x2_bits_26 = *reinterpret_cast<const uint32_t*>(V + (((((ri_1 - num_tiles) * 32 + (16 + vblk_1 * 4 + vhalf_1 * 2) < total_tokens) ? (ri_1 - num_tiles) * 32 + (16 + vblk_1 * 4 + vhalf_1 * 2) : total_tokens - 1) * num_heads + head_1) * 128 + vdp_1 * 2) + 0);
                            (&_vec_load_15_1[0])[0] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_26) << 16);
                            (&_vec_load_15_1[0])[1] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_26) & 0xffff0000u);
                        }
                        vlive_1[5] = (((ri_1 - num_tiles) * 32 + (16 + vblk_1 * 4 + vhalf_1 * 2 + 1) < total_tokens) ? 1 : 0);
                        float _vec_load_16_1[2];
                        {
                            uint32_t _bf16x2_bits_27;
                            _bf16x2_bits_27 = *reinterpret_cast<const uint32_t*>(V + (((((ri_1 - num_tiles) * 32 + (16 + vblk_1 * 4 + vhalf_1 * 2 + 1) < total_tokens) ? (ri_1 - num_tiles) * 32 + (16 + vblk_1 * 4 + vhalf_1 * 2 + 1) : total_tokens - 1) * num_heads + head_1) * 128 + vdp_1 * 2) + 0);
                            (&_vec_load_16_1[0])[0] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_27) << 16);
                            (&_vec_load_16_1[0])[1] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_27) & 0xffff0000u);
                        }
                        vlive_1[6] = (((ri_1 - num_tiles) * 32 + (24 + vblk_1 * 4 + vhalf_1 * 2) < total_tokens) ? 1 : 0);
                        float _vec_load_17_1[2];
                        {
                            uint32_t _bf16x2_bits_28;
                            _bf16x2_bits_28 = *reinterpret_cast<const uint32_t*>(V + (((((ri_1 - num_tiles) * 32 + (24 + vblk_1 * 4 + vhalf_1 * 2) < total_tokens) ? (ri_1 - num_tiles) * 32 + (24 + vblk_1 * 4 + vhalf_1 * 2) : total_tokens - 1) * num_heads + head_1) * 128 + vdp_1 * 2) + 0);
                            (&_vec_load_17_1[0])[0] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_28) << 16);
                            (&_vec_load_17_1[0])[1] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_28) & 0xffff0000u);
                        }
                        vlive_1[7] = (((ri_1 - num_tiles) * 32 + (24 + vblk_1 * 4 + vhalf_1 * 2 + 1) < total_tokens) ? 1 : 0);
                        float _vec_load_18_1[2];
                        {
                            uint32_t _bf16x2_bits_29;
                            _bf16x2_bits_29 = *reinterpret_cast<const uint32_t*>(V + (((((ri_1 - num_tiles) * 32 + (24 + vblk_1 * 4 + vhalf_1 * 2 + 1) < total_tokens) ? (ri_1 - num_tiles) * 32 + (24 + vblk_1 * 4 + vhalf_1 * 2 + 1) : total_tokens - 1) * num_heads + head_1) * 128 + vdp_1 * 2) + 0);
                            (&_vec_load_18_1[0])[0] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_29) << 16);
                            (&_vec_load_18_1[0])[1] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_29) & 0xffff0000u);
                        }
                        float va_1[8];
                        float vb_1[8];
                        va_1[0] = _vec_load_11_1[0];
                        vb_1[0] = _vec_load_11_1[1];
                        va_1[1] = _vec_load_12_1[0];
                        vb_1[1] = _vec_load_12_1[1];
                        va_1[2] = _vec_load_13_1[0];
                        vb_1[2] = _vec_load_13_1[1];
                        va_1[3] = _vec_load_14_1[0];
                        vb_1[3] = _vec_load_14_1[1];
                        va_1[4] = _vec_load_15_1[0];
                        vb_1[4] = _vec_load_15_1[1];
                        va_1[5] = _vec_load_16_1[0];
                        vb_1[5] = _vec_load_16_1[1];
                        va_1[6] = _vec_load_17_1[0];
                        vb_1[6] = _vec_load_17_1[1];
                        va_1[7] = _vec_load_18_1[0];
                        vb_1[7] = _vec_load_18_1[1];
                        va_1[0] = ((vlive_1[0] == 1) ? va_1[0] : 0.0f);
                        vb_1[0] = ((vlive_1[0] == 1) ? vb_1[0] : 0.0f);
                        va_1[1] = ((vlive_1[1] == 1) ? va_1[1] : 0.0f);
                        vb_1[1] = ((vlive_1[1] == 1) ? vb_1[1] : 0.0f);
                        va_1[2] = ((vlive_1[2] == 1) ? va_1[2] : 0.0f);
                        vb_1[2] = ((vlive_1[2] == 1) ? vb_1[2] : 0.0f);
                        va_1[3] = ((vlive_1[3] == 1) ? va_1[3] : 0.0f);
                        vb_1[3] = ((vlive_1[3] == 1) ? vb_1[3] : 0.0f);
                        va_1[4] = ((vlive_1[4] == 1) ? va_1[4] : 0.0f);
                        vb_1[4] = ((vlive_1[4] == 1) ? vb_1[4] : 0.0f);
                        va_1[5] = ((vlive_1[5] == 1) ? va_1[5] : 0.0f);
                        vb_1[5] = ((vlive_1[5] == 1) ? vb_1[5] : 0.0f);
                        va_1[6] = ((vlive_1[6] == 1) ? va_1[6] : 0.0f);
                        vb_1[6] = ((vlive_1[6] == 1) ? vb_1[6] : 0.0f);
                        va_1[7] = ((vlive_1[7] == 1) ? va_1[7] : 0.0f);
                        vb_1[7] = ((vlive_1[7] == 1) ? vb_1[7] : 0.0f);
                        float amax_a_1 = 0.0f;
                        float amax_b_1 = 0.0f;
                        for (int ii_2 = 0; ii_2 < 8; ii_2++) {
                            amax_a_1 = fmaxf(amax_a_1, fabsf(va_1[ii_2]));
                            amax_b_1 = fmaxf(amax_b_1, fabsf(vb_1[ii_2]));
                        }
                        float _shfl_xor_sync_30_1 = __shfl_xor_sync(0xFFFFFFFF, amax_a_1, 16);
                        amax_a_1 = fmaxf(amax_a_1, _shfl_xor_sync_30_1);
                        float _shfl_xor_sync_31_1 = __shfl_xor_sync(0xFFFFFFFF, amax_b_1, 16);
                        amax_b_1 = fmaxf(amax_b_1, _shfl_xor_sync_31_1);
                        float vsf_a_1 = amax_a_1 * 0.16666667f;
                        float vsf_b_1 = amax_b_1 * 0.16666667f;
                        uint16_t _e4m3x2_f32_0_1;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_0_1) : "f"(vsf_b_1), "f"(vsf_a_1));
                        uint16_t _e4m3x2_decode_32 = (uint16_t)((unsigned int)_e4m3x2_f32_0_1 & 0xFFu);
                        uint32_t _f16x2_decode_32;
                        float vdec_a_1;
                        asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_decode_32) : "h"(_e4m3x2_decode_32));
                        uint16_t _f16_decode_32 = (uint16_t)_f16x2_decode_32;
                        asm("cvt.f32.f16 %0, %1;" : "=f"(vdec_a_1) : "h"(_f16_decode_32));
                        uint16_t _e4m3x2_decode_33 = (uint16_t)((unsigned int)_e4m3x2_f32_0_1 >> 8 & 0xFFu);
                        uint32_t _f16x2_decode_33;
                        float vdec_b_1;
                        asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_decode_33) : "h"(_e4m3x2_decode_33));
                        uint16_t _f16_decode_33 = (uint16_t)_f16x2_decode_33;
                        asm("cvt.f32.f16 %0, %1;" : "=f"(vdec_b_1) : "h"(_f16_decode_33));
                        float _fdiv_34_1 = __fdiv_rn(1.0f, vdec_a_1);
                        float vinv_a_1 = ((vdec_a_1 != 0.0f) ? _fdiv_34_1 : 0.0f);
                        float _fdiv_35_1 = __fdiv_rn(1.0f, vdec_b_1);
                        float vinv_b_1 = ((vdec_b_1 != 0.0f) ? _fdiv_35_1 : 0.0f);
                        for (int ii_3 = 0; ii_3 < 8; ii_3++) {
                            va_1[ii_3] = va_1[ii_3] * vinv_a_1;
                            vb_1[ii_3] = vb_1[ii_3] * vinv_b_1;
                        }
                        unsigned int vwa_1[1];
                        unsigned int vwb_1[1];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(vwa_1[0]) : "f"(va_1[0]), "f"(va_1[1]), "f"(va_1[2]), "f"(va_1[3]), "f"(va_1[4]), "f"(va_1[5]), "f"(va_1[6]), "f"(va_1[7]));
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(vwb_1[0]) : "f"(vb_1[0]), "f"(vb_1[1]), "f"(vb_1[2]), "f"(vb_1[3]), "f"(vb_1[4]), "f"(vb_1[5]), "f"(vb_1[6]), "f"(vb_1[7]));
                        int row_words_1 = total_padded >> 3;
                        int vt_word_1 = (head_1 * 128 + vdp_1 * 2) * row_words_1 + ((ri_1 - num_tiles) * 32 >> 3) + vblk_1 * 2 + vhalf_1;
                        *(reinterpret_cast<unsigned int*>(vt4 + vt_word_1) + (0)) = vwa_1[0];
                        *(reinterpret_cast<unsigned int*>(vt4 + (vt_word_1 + row_words_1)) + (0)) = vwb_1[0];
                        if (vhalf_1 == 0) {
                            int vsf_base_1 = ((head_1 * num_kblocks + ((ri_1 - num_tiles) * 32 >> 7)) * 128 + vdp_1 * 2) * 8 + ((((ri_1 - num_tiles) * 32 & 127) >> 4) + vblk_1);
                            uint16_t _cvt_satfinite_e4m3x2_f32_36_1;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_36_1) : "f"(0.0f), "f"(vsf_a_1));
                            *reinterpret_cast<unsigned char*>(v_sf + vsf_base_1) = (unsigned char)((unsigned int)(unsigned char)(_cvt_satfinite_e4m3x2_f32_36_1 & 127));
                            uint16_t _cvt_satfinite_e4m3x2_f32_37_1;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_37_1) : "f"(0.0f), "f"(vsf_b_1));
                            *reinterpret_cast<unsigned char*>(v_sf + (vsf_base_1 + 8)) = (unsigned char)((unsigned int)(unsigned char)(_cvt_satfinite_e4m3x2_f32_37_1 & 127));
                        }
                    } else {
                        int tile_3 = ((ri_1 - num_tiles - num_tblk_1) * num_heads + head_1) / num_heads;
                        int head_0_3 = ((ri_1 - num_tiles - num_tblk_1) * num_heads + head_1) % num_heads;
                        int _vec_load_19_1[4];
                        {
                            const int4* _ivptr_38 = reinterpret_cast<const int4*>(tile_table + (8 * tile_3) + 0);
                            int4 _ivld_38;
                            _ivld_38 = *_ivptr_38;
                            _vec_load_19_1[0 + 0] = _ivld_38.x;
                            _vec_load_19_1[0 + 1] = _ivld_38.y;
                            _vec_load_19_1[0 + 2] = _ivld_38.z;
                            _vec_load_19_1[0 + 3] = _ivld_38.w;
                        }
                        int _vec_load_20_1[4];
                        {
                            const int4* _ivptr_39 = reinterpret_cast<const int4*>(tile_table + (8 * tile_3 + 4) + 0);
                            int4 _ivld_39;
                            _ivld_39 = *_ivptr_39;
                            _vec_load_20_1[0 + 0] = _ivld_39.x;
                            _vec_load_20_1[0 + 1] = _ivld_39.y;
                            _vec_load_20_1[0 + 2] = _ivld_39.z;
                            _vec_load_20_1[0 + 3] = _ivld_39.w;
                        }
                        int seg_4 = _vec_load_19_1[0];
                        int row_begin_3 = _vec_load_19_1[1];
                        int row_end_3 = _vec_load_19_1[2];
                        int t0_3 = _vec_load_19_1[3];
                        int t1_1 = _vec_load_20_1[0];
                        int length_1 = _vec_load_20_1[1];
                        float ksum_1[8];
                        for (int e_9 = 0; e_9 < 8; e_9++) {
                            ksum_1[e_9] = 0.0f;
                        }
                        float kf8_1[8];
                        #pragma unroll 1
                        for (int batch_3 = 0; batch_3 < (row_end_3 - row_begin_3 + 128 - 1) / 128; batch_3++) {
                            int brow_3 = row_begin_3 + batch_3 * 128;
                            float vec_load_11[8];
                            {
                                const uint4* _vptr_40 = reinterpret_cast<const uint4*>(K + ((((row_end_3 > brow_3 + rgrp_1) ? brow_3 + rgrp_1 : row_end_3 - 1) * num_heads + head_0_3) * 128 + col_1) + 0);
                                uint4 _vld_40[1];
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
                                    _vld_40[_blk] = _vptr_40[_blk];
                                    uint32_t* _vpairs_40 = reinterpret_cast<uint32_t*>(&_vld_40[_blk]);
                                    #pragma unroll
                                    for (int _pair = 0; _pair < 4; _pair++) {
                                        (&vec_load_11[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_40[_pair]) << 16);
                                        (&vec_load_11[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_40[_pair]) & 0xffff0000u);
                                    }
                                }
                            }
                            float vec_load_1_3[8];
                            {
                                const uint4* _vptr_41 = reinterpret_cast<const uint4*>(K + ((((row_end_3 > brow_3 + 16 + rgrp_1) ? brow_3 + 16 + rgrp_1 : row_end_3 - 1) * num_heads + head_0_3) * 128 + col_1) + 0);
                                uint4 _vld_41[1];
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
                                    _vld_41[_blk] = _vptr_41[_blk];
                                    uint32_t* _vpairs_41 = reinterpret_cast<uint32_t*>(&_vld_41[_blk]);
                                    #pragma unroll
                                    for (int _pair = 0; _pair < 4; _pair++) {
                                        (&vec_load_1_3[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_41[_pair]) << 16);
                                        (&vec_load_1_3[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_41[_pair]) & 0xffff0000u);
                                    }
                                }
                            }
                            float vec_load_2_3[8];
                            {
                                const uint4* _vptr_42 = reinterpret_cast<const uint4*>(K + ((((row_end_3 > brow_3 + 32 + rgrp_1) ? brow_3 + 32 + rgrp_1 : row_end_3 - 1) * num_heads + head_0_3) * 128 + col_1) + 0);
                                uint4 _vld_42[1];
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
                                    _vld_42[_blk] = _vptr_42[_blk];
                                    uint32_t* _vpairs_42 = reinterpret_cast<uint32_t*>(&_vld_42[_blk]);
                                    #pragma unroll
                                    for (int _pair = 0; _pair < 4; _pair++) {
                                        (&vec_load_2_3[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_42[_pair]) << 16);
                                        (&vec_load_2_3[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_42[_pair]) & 0xffff0000u);
                                    }
                                }
                            }
                            float vec_load_3_3[8];
                            {
                                const uint4* _vptr_43 = reinterpret_cast<const uint4*>(K + ((((row_end_3 > brow_3 + 48 + rgrp_1) ? brow_3 + 48 + rgrp_1 : row_end_3 - 1) * num_heads + head_0_3) * 128 + col_1) + 0);
                                uint4 _vld_43[1];
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
                                    _vld_43[_blk] = _vptr_43[_blk];
                                    uint32_t* _vpairs_43 = reinterpret_cast<uint32_t*>(&_vld_43[_blk]);
                                    #pragma unroll
                                    for (int _pair = 0; _pair < 4; _pair++) {
                                        (&vec_load_3_3[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_43[_pair]) << 16);
                                        (&vec_load_3_3[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_43[_pair]) & 0xffff0000u);
                                    }
                                }
                            }
                            float vec_load_4_3[8];
                            {
                                const uint4* _vptr_44 = reinterpret_cast<const uint4*>(K + ((((row_end_3 > brow_3 + 64 + rgrp_1) ? brow_3 + 64 + rgrp_1 : row_end_3 - 1) * num_heads + head_0_3) * 128 + col_1) + 0);
                                uint4 _vld_44[1];
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
                                    _vld_44[_blk] = _vptr_44[_blk];
                                    uint32_t* _vpairs_44 = reinterpret_cast<uint32_t*>(&_vld_44[_blk]);
                                    #pragma unroll
                                    for (int _pair = 0; _pair < 4; _pair++) {
                                        (&vec_load_4_3[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_44[_pair]) << 16);
                                        (&vec_load_4_3[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_44[_pair]) & 0xffff0000u);
                                    }
                                }
                            }
                            float vec_load_5_3[8];
                            {
                                const uint4* _vptr_45 = reinterpret_cast<const uint4*>(K + ((((row_end_3 > brow_3 + 80 + rgrp_1) ? brow_3 + 80 + rgrp_1 : row_end_3 - 1) * num_heads + head_0_3) * 128 + col_1) + 0);
                                uint4 _vld_45[1];
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
                                    _vld_45[_blk] = _vptr_45[_blk];
                                    uint32_t* _vpairs_45 = reinterpret_cast<uint32_t*>(&_vld_45[_blk]);
                                    #pragma unroll
                                    for (int _pair = 0; _pair < 4; _pair++) {
                                        (&vec_load_5_3[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_45[_pair]) << 16);
                                        (&vec_load_5_3[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_45[_pair]) & 0xffff0000u);
                                    }
                                }
                            }
                            float vec_load_6_3[8];
                            {
                                const uint4* _vptr_46 = reinterpret_cast<const uint4*>(K + ((((row_end_3 > brow_3 + 96 + rgrp_1) ? brow_3 + 96 + rgrp_1 : row_end_3 - 1) * num_heads + head_0_3) * 128 + col_1) + 0);
                                uint4 _vld_46[1];
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
                                    _vld_46[_blk] = _vptr_46[_blk];
                                    uint32_t* _vpairs_46 = reinterpret_cast<uint32_t*>(&_vld_46[_blk]);
                                    #pragma unroll
                                    for (int _pair = 0; _pair < 4; _pair++) {
                                        (&vec_load_6_3[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_46[_pair]) << 16);
                                        (&vec_load_6_3[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_46[_pair]) & 0xffff0000u);
                                    }
                                }
                            }
                            float vec_load_7_3[8];
                            {
                                const uint4* _vptr_47 = reinterpret_cast<const uint4*>(K + ((((row_end_3 > brow_3 + 112 + rgrp_1) ? brow_3 + 112 + rgrp_1 : row_end_3 - 1) * num_heads + head_0_3) * 128 + col_1) + 0);
                                uint4 _vld_47[1];
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
                                    _vld_47[_blk] = _vptr_47[_blk];
                                    uint32_t* _vpairs_47 = reinterpret_cast<uint32_t*>(&_vld_47[_blk]);
                                    #pragma unroll
                                    for (int _pair = 0; _pair < 4; _pair++) {
                                        (&vec_load_7_3[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_47[_pair]) << 16);
                                        (&vec_load_7_3[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_47[_pair]) & 0xffff0000u);
                                    }
                                }
                            }
                            kf8_1[0] = vec_load_11[0];
                            kf8_1[1] = vec_load_11[1];
                            kf8_1[2] = vec_load_11[2];
                            kf8_1[3] = vec_load_11[3];
                            kf8_1[4] = vec_load_11[4];
                            kf8_1[5] = vec_load_11[5];
                            kf8_1[6] = vec_load_11[6];
                            kf8_1[7] = vec_load_11[7];
                            ksum_1[0] = ksum_1[0] + ((row_end_3 > brow_3 + rgrp_1) ? kf8_1[0] : 0.0f);
                            ksum_1[1] = ksum_1[1] + ((row_end_3 > brow_3 + rgrp_1) ? kf8_1[1] : 0.0f);
                            ksum_1[2] = ksum_1[2] + ((row_end_3 > brow_3 + rgrp_1) ? kf8_1[2] : 0.0f);
                            ksum_1[3] = ksum_1[3] + ((row_end_3 > brow_3 + rgrp_1) ? kf8_1[3] : 0.0f);
                            ksum_1[4] = ksum_1[4] + ((row_end_3 > brow_3 + rgrp_1) ? kf8_1[4] : 0.0f);
                            ksum_1[5] = ksum_1[5] + ((row_end_3 > brow_3 + rgrp_1) ? kf8_1[5] : 0.0f);
                            ksum_1[6] = ksum_1[6] + ((row_end_3 > brow_3 + rgrp_1) ? kf8_1[6] : 0.0f);
                            ksum_1[7] = ksum_1[7] + ((row_end_3 > brow_3 + rgrp_1) ? kf8_1[7] : 0.0f);
                            kf8_1[0] = vec_load_1_3[0];
                            kf8_1[1] = vec_load_1_3[1];
                            kf8_1[2] = vec_load_1_3[2];
                            kf8_1[3] = vec_load_1_3[3];
                            kf8_1[4] = vec_load_1_3[4];
                            kf8_1[5] = vec_load_1_3[5];
                            kf8_1[6] = vec_load_1_3[6];
                            kf8_1[7] = vec_load_1_3[7];
                            ksum_1[0] = ksum_1[0] + ((row_end_3 > brow_3 + 16 + rgrp_1) ? kf8_1[0] : 0.0f);
                            ksum_1[1] = ksum_1[1] + ((row_end_3 > brow_3 + 16 + rgrp_1) ? kf8_1[1] : 0.0f);
                            ksum_1[2] = ksum_1[2] + ((row_end_3 > brow_3 + 16 + rgrp_1) ? kf8_1[2] : 0.0f);
                            ksum_1[3] = ksum_1[3] + ((row_end_3 > brow_3 + 16 + rgrp_1) ? kf8_1[3] : 0.0f);
                            ksum_1[4] = ksum_1[4] + ((row_end_3 > brow_3 + 16 + rgrp_1) ? kf8_1[4] : 0.0f);
                            ksum_1[5] = ksum_1[5] + ((row_end_3 > brow_3 + 16 + rgrp_1) ? kf8_1[5] : 0.0f);
                            ksum_1[6] = ksum_1[6] + ((row_end_3 > brow_3 + 16 + rgrp_1) ? kf8_1[6] : 0.0f);
                            ksum_1[7] = ksum_1[7] + ((row_end_3 > brow_3 + 16 + rgrp_1) ? kf8_1[7] : 0.0f);
                            kf8_1[0] = vec_load_2_3[0];
                            kf8_1[1] = vec_load_2_3[1];
                            kf8_1[2] = vec_load_2_3[2];
                            kf8_1[3] = vec_load_2_3[3];
                            kf8_1[4] = vec_load_2_3[4];
                            kf8_1[5] = vec_load_2_3[5];
                            kf8_1[6] = vec_load_2_3[6];
                            kf8_1[7] = vec_load_2_3[7];
                            ksum_1[0] = ksum_1[0] + ((row_end_3 > brow_3 + 32 + rgrp_1) ? kf8_1[0] : 0.0f);
                            ksum_1[1] = ksum_1[1] + ((row_end_3 > brow_3 + 32 + rgrp_1) ? kf8_1[1] : 0.0f);
                            ksum_1[2] = ksum_1[2] + ((row_end_3 > brow_3 + 32 + rgrp_1) ? kf8_1[2] : 0.0f);
                            ksum_1[3] = ksum_1[3] + ((row_end_3 > brow_3 + 32 + rgrp_1) ? kf8_1[3] : 0.0f);
                            ksum_1[4] = ksum_1[4] + ((row_end_3 > brow_3 + 32 + rgrp_1) ? kf8_1[4] : 0.0f);
                            ksum_1[5] = ksum_1[5] + ((row_end_3 > brow_3 + 32 + rgrp_1) ? kf8_1[5] : 0.0f);
                            ksum_1[6] = ksum_1[6] + ((row_end_3 > brow_3 + 32 + rgrp_1) ? kf8_1[6] : 0.0f);
                            ksum_1[7] = ksum_1[7] + ((row_end_3 > brow_3 + 32 + rgrp_1) ? kf8_1[7] : 0.0f);
                            kf8_1[0] = vec_load_3_3[0];
                            kf8_1[1] = vec_load_3_3[1];
                            kf8_1[2] = vec_load_3_3[2];
                            kf8_1[3] = vec_load_3_3[3];
                            kf8_1[4] = vec_load_3_3[4];
                            kf8_1[5] = vec_load_3_3[5];
                            kf8_1[6] = vec_load_3_3[6];
                            kf8_1[7] = vec_load_3_3[7];
                            ksum_1[0] = ksum_1[0] + ((row_end_3 > brow_3 + 48 + rgrp_1) ? kf8_1[0] : 0.0f);
                            ksum_1[1] = ksum_1[1] + ((row_end_3 > brow_3 + 48 + rgrp_1) ? kf8_1[1] : 0.0f);
                            ksum_1[2] = ksum_1[2] + ((row_end_3 > brow_3 + 48 + rgrp_1) ? kf8_1[2] : 0.0f);
                            ksum_1[3] = ksum_1[3] + ((row_end_3 > brow_3 + 48 + rgrp_1) ? kf8_1[3] : 0.0f);
                            ksum_1[4] = ksum_1[4] + ((row_end_3 > brow_3 + 48 + rgrp_1) ? kf8_1[4] : 0.0f);
                            ksum_1[5] = ksum_1[5] + ((row_end_3 > brow_3 + 48 + rgrp_1) ? kf8_1[5] : 0.0f);
                            ksum_1[6] = ksum_1[6] + ((row_end_3 > brow_3 + 48 + rgrp_1) ? kf8_1[6] : 0.0f);
                            ksum_1[7] = ksum_1[7] + ((row_end_3 > brow_3 + 48 + rgrp_1) ? kf8_1[7] : 0.0f);
                            kf8_1[0] = vec_load_4_3[0];
                            kf8_1[1] = vec_load_4_3[1];
                            kf8_1[2] = vec_load_4_3[2];
                            kf8_1[3] = vec_load_4_3[3];
                            kf8_1[4] = vec_load_4_3[4];
                            kf8_1[5] = vec_load_4_3[5];
                            kf8_1[6] = vec_load_4_3[6];
                            kf8_1[7] = vec_load_4_3[7];
                            ksum_1[0] = ksum_1[0] + ((row_end_3 > brow_3 + 64 + rgrp_1) ? kf8_1[0] : 0.0f);
                            ksum_1[1] = ksum_1[1] + ((row_end_3 > brow_3 + 64 + rgrp_1) ? kf8_1[1] : 0.0f);
                            ksum_1[2] = ksum_1[2] + ((row_end_3 > brow_3 + 64 + rgrp_1) ? kf8_1[2] : 0.0f);
                            ksum_1[3] = ksum_1[3] + ((row_end_3 > brow_3 + 64 + rgrp_1) ? kf8_1[3] : 0.0f);
                            ksum_1[4] = ksum_1[4] + ((row_end_3 > brow_3 + 64 + rgrp_1) ? kf8_1[4] : 0.0f);
                            ksum_1[5] = ksum_1[5] + ((row_end_3 > brow_3 + 64 + rgrp_1) ? kf8_1[5] : 0.0f);
                            ksum_1[6] = ksum_1[6] + ((row_end_3 > brow_3 + 64 + rgrp_1) ? kf8_1[6] : 0.0f);
                            ksum_1[7] = ksum_1[7] + ((row_end_3 > brow_3 + 64 + rgrp_1) ? kf8_1[7] : 0.0f);
                            kf8_1[0] = vec_load_5_3[0];
                            kf8_1[1] = vec_load_5_3[1];
                            kf8_1[2] = vec_load_5_3[2];
                            kf8_1[3] = vec_load_5_3[3];
                            kf8_1[4] = vec_load_5_3[4];
                            kf8_1[5] = vec_load_5_3[5];
                            kf8_1[6] = vec_load_5_3[6];
                            kf8_1[7] = vec_load_5_3[7];
                            ksum_1[0] = ksum_1[0] + ((row_end_3 > brow_3 + 80 + rgrp_1) ? kf8_1[0] : 0.0f);
                            ksum_1[1] = ksum_1[1] + ((row_end_3 > brow_3 + 80 + rgrp_1) ? kf8_1[1] : 0.0f);
                            ksum_1[2] = ksum_1[2] + ((row_end_3 > brow_3 + 80 + rgrp_1) ? kf8_1[2] : 0.0f);
                            ksum_1[3] = ksum_1[3] + ((row_end_3 > brow_3 + 80 + rgrp_1) ? kf8_1[3] : 0.0f);
                            ksum_1[4] = ksum_1[4] + ((row_end_3 > brow_3 + 80 + rgrp_1) ? kf8_1[4] : 0.0f);
                            ksum_1[5] = ksum_1[5] + ((row_end_3 > brow_3 + 80 + rgrp_1) ? kf8_1[5] : 0.0f);
                            ksum_1[6] = ksum_1[6] + ((row_end_3 > brow_3 + 80 + rgrp_1) ? kf8_1[6] : 0.0f);
                            ksum_1[7] = ksum_1[7] + ((row_end_3 > brow_3 + 80 + rgrp_1) ? kf8_1[7] : 0.0f);
                            kf8_1[0] = vec_load_6_3[0];
                            kf8_1[1] = vec_load_6_3[1];
                            kf8_1[2] = vec_load_6_3[2];
                            kf8_1[3] = vec_load_6_3[3];
                            kf8_1[4] = vec_load_6_3[4];
                            kf8_1[5] = vec_load_6_3[5];
                            kf8_1[6] = vec_load_6_3[6];
                            kf8_1[7] = vec_load_6_3[7];
                            ksum_1[0] = ksum_1[0] + ((row_end_3 > brow_3 + 96 + rgrp_1) ? kf8_1[0] : 0.0f);
                            ksum_1[1] = ksum_1[1] + ((row_end_3 > brow_3 + 96 + rgrp_1) ? kf8_1[1] : 0.0f);
                            ksum_1[2] = ksum_1[2] + ((row_end_3 > brow_3 + 96 + rgrp_1) ? kf8_1[2] : 0.0f);
                            ksum_1[3] = ksum_1[3] + ((row_end_3 > brow_3 + 96 + rgrp_1) ? kf8_1[3] : 0.0f);
                            ksum_1[4] = ksum_1[4] + ((row_end_3 > brow_3 + 96 + rgrp_1) ? kf8_1[4] : 0.0f);
                            ksum_1[5] = ksum_1[5] + ((row_end_3 > brow_3 + 96 + rgrp_1) ? kf8_1[5] : 0.0f);
                            ksum_1[6] = ksum_1[6] + ((row_end_3 > brow_3 + 96 + rgrp_1) ? kf8_1[6] : 0.0f);
                            ksum_1[7] = ksum_1[7] + ((row_end_3 > brow_3 + 96 + rgrp_1) ? kf8_1[7] : 0.0f);
                            kf8_1[0] = vec_load_7_3[0];
                            kf8_1[1] = vec_load_7_3[1];
                            kf8_1[2] = vec_load_7_3[2];
                            kf8_1[3] = vec_load_7_3[3];
                            kf8_1[4] = vec_load_7_3[4];
                            kf8_1[5] = vec_load_7_3[5];
                            kf8_1[6] = vec_load_7_3[6];
                            kf8_1[7] = vec_load_7_3[7];
                            ksum_1[0] = ksum_1[0] + ((row_end_3 > brow_3 + 112 + rgrp_1) ? kf8_1[0] : 0.0f);
                            ksum_1[1] = ksum_1[1] + ((row_end_3 > brow_3 + 112 + rgrp_1) ? kf8_1[1] : 0.0f);
                            ksum_1[2] = ksum_1[2] + ((row_end_3 > brow_3 + 112 + rgrp_1) ? kf8_1[2] : 0.0f);
                            ksum_1[3] = ksum_1[3] + ((row_end_3 > brow_3 + 112 + rgrp_1) ? kf8_1[3] : 0.0f);
                            ksum_1[4] = ksum_1[4] + ((row_end_3 > brow_3 + 112 + rgrp_1) ? kf8_1[4] : 0.0f);
                            ksum_1[5] = ksum_1[5] + ((row_end_3 > brow_3 + 112 + rgrp_1) ? kf8_1[5] : 0.0f);
                            ksum_1[6] = ksum_1[6] + ((row_end_3 > brow_3 + 112 + rgrp_1) ? kf8_1[6] : 0.0f);
                            ksum_1[7] = ksum_1[7] + ((row_end_3 > brow_3 + 112 + rgrp_1) ? kf8_1[7] : 0.0f);
                        }
                        for (int e_10 = 0; e_10 < 8; e_10++) {
                            red_sum[2048 + rgrp_1 * 128 + col_1 + e_10] = ksum_1[e_10];
                        }
                        asm volatile("bar.sync 2, 256;" ::: "memory");
                        int direct_1 = ((t1_1 - t0_3 == 1) ? 1 : 0);
                        if (direct_1 == 0) {
                            if (wtid_1 < 128) {
                                float total_1 = 0.0f;
                                for (int g_4 = 0; g_4 < 16; g_4++) {
                                    total_1 = total_1 + red_sum[2048 + g_4 * 128 + wtid_1];
                                }
                                *(reinterpret_cast<float*>(partials + (((ri_1 - num_tiles - num_tblk_1) * num_heads + head_1) * 128 + wtid_1)) + (0)) = total_1;
                            }
                            asm volatile("bar.sync 2, 256;" ::: "memory");
                            if (wtid_1 == 0) {
                                unsigned int old_2;
                                asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], %2;"
                                    : "=r"(old_2) : "l"(&counters[seg_4 * num_heads + head_0_3]), "r"(static_cast<uint32_t>(1)) : "memory");
                                int last_1 = (((int)old_2 == t1_1 - t0_3 - 1) ? 1 : 0);
                                flag[4] = last_1;
                                if (last_1 == 1) {
                                    *(reinterpret_cast<unsigned int*>(counters + (seg_4 * num_heads + head_0_3)) + (0)) = (unsigned int)0;
                                }
                            }
                            asm volatile("bar.sync 2, 256;" ::: "memory");
                        }
                        int is_last_1 = ((direct_1 == 0) ? flag[4] : 1);
                        if (is_last_1 == 1) {
                            if (wtid_1 < 128) {
                                float acc_a_1[1];
                                float fold_v_1[4];
                                acc_a_1[0] = 0.0f;
                                if (direct_1 == 1) {
                                    for (int g_5 = 0; g_5 < 16; g_5++) {
                                        acc_a_1[0] = acc_a_1[0] + red_sum[2048 + g_5 * 128 + wtid_1];
                                    }
                                } else {
                                    #pragma unroll 1
                                    for (int tt_1 = t0_3; tt_1 < t1_1; tt_1 += 4) {
                                        for (int j_2 = 0; j_2 < 4; j_2++) {
                                            int ttj_1 = ((t1_1 > tt_1 + j_2) ? tt_1 + j_2 : t1_1 - 1);
                                            fold_v_1[j_2] = partials[(ttj_1 * num_heads + head_0_3) * 128 + wtid_1];
                                        }
                                        for (int j_3 = 0; j_3 < 4; j_3++) {
                                            float fv_1 = ((t1_1 > tt_1 + j_3) ? fold_v_1[j_3] : 0.0f);
                                            acc_a_1[0] = acc_a_1[0] + fv_1;
                                        }
                                    }
                                }
                                if (length_1 > 0) {
                                    float lenf_1 = (float)length_1;
                                    float _fdiv_48_1 = __fdiv_rn(acc_a_1[0], lenf_1);
                                    *(reinterpret_cast<float*>(mean_k + ((seg_4 * num_heads + head_0_3) * 128 + wtid_1)) + (0)) = _fdiv_48_1;
                                }
                            }
                        }
                        asm volatile("bar.sync 2, 256;" ::: "memory");
                        if (wtid_1 == 0) {
                            asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(hctr) + (num_heads + head_1))), "r"(static_cast<unsigned int>(1)) : "memory");
                        }
                    }
                } else {
                    if (wtid_1 == 0) {
                        {
                        unsigned int _acquire_observed;
                        do {
                        asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(_acquire_observed) : "l"((reinterpret_cast<unsigned int*>(hctr) + (num_heads + head_1))) : "memory");
                        } while (static_cast<unsigned int>(_acquire_observed - static_cast<unsigned int>(num_tiles)) >= static_cast<unsigned int>(1));
                        }
                    }
                    asm volatile("bar.sync 2, 256;" ::: "memory");
                    int tok_3 = ri_1 * 32 + trow_1;
                    int valid_3 = ((tok_3 < total_tokens) ? 1 : 0);
                    int tok_c_3 = ((tok_3 < total_tokens) ? tok_3 : total_tokens - 1);
                    float vec_load_12[16];
                    {
                        const uint4* _vptr_49 = reinterpret_cast<const uint4*>(K + ((tok_c_3 * num_heads + head_1) * 128 + col16_1) + 0);
                        uint4 _vld_49[2];
                        #pragma unroll
                        for (int _blk = 0; _blk < 2; _blk++) {
                            _vld_49[_blk] = _vptr_49[_blk];
                            uint32_t* _vpairs_49 = reinterpret_cast<uint32_t*>(&_vld_49[_blk]);
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
                                (&vec_load_12[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_49[_pair]) << 16);
                                (&vec_load_12[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_49[_pair]) & 0xffff0000u);
                            }
                        }
                    }
                    int seg_5 = tok_seg[tok_c_3];
                    int mbase_1 = (seg_5 * num_heads + head_1) * 128 + col16_1;
                    float kc_1[16];
                    for (int quad_3 = 0; quad_3 < 4; quad_3++) {
                        float _vec_load_30_1[4];
                        {
                            float4 _v4 = *reinterpret_cast<const float4*>(mean_k + (mbase_1 + quad_3 * 4) + 0);
                            _vec_load_30_1[0 + 0] = _v4.x;
                            _vec_load_30_1[0 + 1] = _v4.y;
                            _vec_load_30_1[0 + 2] = _v4.z;
                            _vec_load_30_1[0 + 3] = _v4.w;
                        }
                        for (int e_11 = 0; e_11 < 4; e_11++) {
                            float kf_1 = vec_load_12[quad_3 * 4 + e_11];
                            kc_1[quad_3 * 4 + e_11] = ((valid_3 == 1) ? kf_1 - _vec_load_30_1[e_11] : 0.0f);
                        }
                    }
                    unsigned int kw_1[2];
                    float ksf_a_1[1];
                    float max_1_3 = fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fabsf(kc_1[0]), fabsf(kc_1[1])), fabsf(kc_1[2])), fabsf(kc_1[3])), fabsf(kc_1[4])), fabsf(kc_1[5])), fabsf(kc_1[6])), fabsf(kc_1[7])), fabsf(kc_1[8])), fabsf(kc_1[9])), fabsf(kc_1[10])), fabsf(kc_1[11])), fabsf(kc_1[12])), fabsf(kc_1[13])), fabsf(kc_1[14])), fabsf(kc_1[15]));
                    uint16_t _cvt_satfinite_e4m3x2_f32_51_1;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_51_1) : "f"(0.0f), "f"(max_1_3 * 0.16666667f));
                    uint16_t _e4m3x2_decode_52 = (uint16_t)((unsigned int)_cvt_satfinite_e4m3x2_f32_51_1 & 0xFFu);
                    uint32_t _f16x2_decode_52;
                    float fp8_decode_3;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_decode_52) : "h"(_e4m3x2_decode_52));
                    uint16_t _f16_decode_52 = (uint16_t)_f16x2_decode_52;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(fp8_decode_3) : "h"(_f16_decode_52));
                    float fdiv_rn_3 = __fdiv_rn(1.0f, fp8_decode_3);
                    kc_1[0] = kc_1[0] * ((fp8_decode_3 != 0.0f) ? fdiv_rn_3 : 0.0f);
                    kc_1[1] = kc_1[1] * ((fp8_decode_3 != 0.0f) ? fdiv_rn_3 : 0.0f);
                    kc_1[2] = kc_1[2] * ((fp8_decode_3 != 0.0f) ? fdiv_rn_3 : 0.0f);
                    kc_1[3] = kc_1[3] * ((fp8_decode_3 != 0.0f) ? fdiv_rn_3 : 0.0f);
                    kc_1[4] = kc_1[4] * ((fp8_decode_3 != 0.0f) ? fdiv_rn_3 : 0.0f);
                    kc_1[5] = kc_1[5] * ((fp8_decode_3 != 0.0f) ? fdiv_rn_3 : 0.0f);
                    kc_1[6] = kc_1[6] * ((fp8_decode_3 != 0.0f) ? fdiv_rn_3 : 0.0f);
                    kc_1[7] = kc_1[7] * ((fp8_decode_3 != 0.0f) ? fdiv_rn_3 : 0.0f);
                    kc_1[8] = kc_1[8] * ((fp8_decode_3 != 0.0f) ? fdiv_rn_3 : 0.0f);
                    kc_1[9] = kc_1[9] * ((fp8_decode_3 != 0.0f) ? fdiv_rn_3 : 0.0f);
                    kc_1[10] = kc_1[10] * ((fp8_decode_3 != 0.0f) ? fdiv_rn_3 : 0.0f);
                    kc_1[11] = kc_1[11] * ((fp8_decode_3 != 0.0f) ? fdiv_rn_3 : 0.0f);
                    kc_1[12] = kc_1[12] * ((fp8_decode_3 != 0.0f) ? fdiv_rn_3 : 0.0f);
                    kc_1[13] = kc_1[13] * ((fp8_decode_3 != 0.0f) ? fdiv_rn_3 : 0.0f);
                    kc_1[14] = kc_1[14] * ((fp8_decode_3 != 0.0f) ? fdiv_rn_3 : 0.0f);
                    kc_1[15] = kc_1[15] * ((fp8_decode_3 != 0.0f) ? fdiv_rn_3 : 0.0f);
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(kw_1[0]) : "f"(kc_1[0]), "f"(kc_1[1]), "f"(kc_1[2]), "f"(kc_1[3]), "f"(kc_1[4]), "f"(kc_1[5]), "f"(kc_1[6]), "f"(kc_1[7]));
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(kw_1[1]) : "f"(kc_1[8]), "f"(kc_1[9]), "f"(kc_1[10]), "f"(kc_1[11]), "f"(kc_1[12]), "f"(kc_1[13]), "f"(kc_1[14]), "f"(kc_1[15]));
                    ksf_a_1[0] = max_1_3 * 0.16666667f;
                    unsigned int kstore_1[4];
                    unsigned int _shfl_xor_sync_53_1 = __shfl_xor_sync(0xFFFFFFFF, kw_1[0], 1);
                    kstore_1[0] = _shfl_xor_sync_53_1;
                    unsigned int _shfl_xor_sync_54_1 = __shfl_xor_sync(0xFFFFFFFF, kw_1[1], 1);
                    kstore_1[1] = _shfl_xor_sync_54_1;
                    kstore_1[2] = kw_1[0];
                    kstore_1[3] = kw_1[1];
                    uint16_t _cvt_satfinite_e4m3x2_f32_55_1;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_55_1) : "f"(0.0f), "f"(ksf_a_1[0]));
                    *reinterpret_cast<unsigned char*>(k_sf + (((head_1 * num_kblocks + (tok_3 >> 7)) * 128 + (tok_3 & 127)) * 8 + sub_1)) = (unsigned char)((unsigned int)(unsigned char)(_cvt_satfinite_e4m3x2_f32_55_1 & 127));
                    if (valid_3 == 1) {
                        if ((sub_1 & 1) == 1) {
                            reinterpret_cast<int4*>(k4 + (((tok_3 * num_heads + head_1) * 64 + (sub_1 - 1) * 8) / 4))[0] = reinterpret_cast<int4*>(kstore_1)[0];
                        }
                    }
                }
                if (wtid_1 == 0) {
                    item_cell[4 + (it_1 + 1 & 1)] = (int)nxt_r_1[0];
                }
                asm volatile("bar.sync 2, 256;" ::: "memory");
                if (wtid_1 == 0) {
                    if (is_d_1 == 1) {
                        unsigned int dold_1;
                        asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], %2;"
                            : "=r"(dold_1) : "l"(&hctr[2 * num_heads + head_1]), "r"(static_cast<uint32_t>(1)) : "memory");
                        if ((int)dold_1 == num_tblk_1 - 1) {
                            *(reinterpret_cast<unsigned int*>(hctr + (num_heads + head_1)) + (0)) = (unsigned int)0;
                            *(reinterpret_cast<unsigned int*>(hctr + (2 * num_heads + head_1)) + (0)) = (unsigned int)0;
                        }
                    }
                    unsigned int old_3;
                    asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], %2;"
                        : "=r"(old_3) : "l"(&hctr[head_1]), "r"(static_cast<uint32_t>(1)) : "memory");
                    if ((int)old_3 == per_1 - 1) {
                        *(reinterpret_cast<unsigned int*>(hctr + head_1) + (0)) = (unsigned int)0;
                        asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(hctr) + (3 * num_heads + head_1))), "r"(static_cast<unsigned int>(1)) : "memory");
                    }
                }
            }
            asm volatile("barrier.sync 4, 768;" ::: "memory");
        }
    }
    // ---- Role: wk2 ----
    if (warp >= 16 && warp <= 23) {
        { // wk2_main
            int wtid_2 = tid - 512;
            int col_2 = (wtid_2 & 15) * 8;
            int rgrp_2 = wtid_2 >> 4;
            int sub_2 = wtid_2 & 7;
            int trow_2 = wtid_2 >> 3;
            int col16_2 = sub_2 * 16;
            int num_tblk_2 = total_padded / 32;
            int vwarp_2 = wtid_2 >> 5;
            int vlane_2 = wtid_2 & 31;
            int vblk_2 = vwarp_2 & 1;
            int vdp_2 = (vwarp_2 >> 1) * 16 + (vlane_2 & 15);
            int vhalf_2 = vlane_2 >> 4;
            int n_abc_2 = 2 * num_tiles + num_tblk_2;
            int per_2 = n_abc_2 + num_tblk_2;
            asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
            unsigned int nxt_r_2[2];
            if (wtid_2 == 0) {
                unsigned int _atomic_fetch_add_0_2 = atomicAdd(&hctr[5 * num_heads + 4], 1);
                item_cell[8] = (int)_atomic_fetch_add_0_2;
            }
            asm volatile("bar.sync 3, 256;" ::: "memory");
            #pragma unroll 1
            for (int it_2 = 0; it_2 < per_2 * num_heads; it_2++) {
                int w_2 = item_cell[8 + (it_2 & 1)];
                if (w_2 >= per_2 * num_heads) {
                    break;
                }
                if (wtid_2 == 0) {
                    unsigned int _atomic_fetch_add_1_2 = atomicAdd(&hctr[5 * num_heads + 4], 1);
                    nxt_r_2[0] = _atomic_fetch_add_1_2;
                }
                int w2_2 = ((w_2 >= n_abc_2) ? w_2 - n_abc_2 : 0);
                int s_2 = w2_2 / per_2;
                int r_2 = w2_2 - s_2 * per_2;
                int last_s_2 = ((s_2 + 1 >= num_heads) ? 1 : 0);
                int is_d_2 = ((w_2 >= n_abc_2 && (last_s_2 == 1 || r_2 >= n_abc_2)) ? 1 : 0);
                int head_2 = ((w_2 < n_abc_2) ? 0 : ((is_d_2 == 1) ? s_2 : s_2 + 1));
                int ri_2 = ((w_2 < n_abc_2) ? w_2 : ((last_s_2 == 1) ? r_2 : ((is_d_2 == 1) ? r_2 - n_abc_2 : r_2)));
                if (is_d_2 == 0) {
                    if (ri_2 < num_tiles) {
                        int tile_4 = (ri_2 * num_heads + head_2) / num_heads;
                        int head_0_4 = (ri_2 * num_heads + head_2) % num_heads;
                        int _vec_load_0_2[4];
                        {
                            const int4* _ivptr_2 = reinterpret_cast<const int4*>(tile_table + (8 * tile_4) + 0);
                            int4 _ivld_2;
                            _ivld_2 = *_ivptr_2;
                            _vec_load_0_2[0 + 0] = _ivld_2.x;
                            _vec_load_0_2[0 + 1] = _ivld_2.y;
                            _vec_load_0_2[0 + 2] = _ivld_2.z;
                            _vec_load_0_2[0 + 3] = _ivld_2.w;
                        }
                        int seg_6 = _vec_load_0_2[0];
                        int row_begin_4 = _vec_load_0_2[1];
                        int row_end_4 = _vec_load_0_2[2];
                        int t0_4 = _vec_load_0_2[3];
                        int qt_base_2 = seg_tile_begin[seg_6] + (tile_4 - t0_4);
                        float qsum_2[8];
                        float qf8_2[8];
                        #pragma unroll 1
                        for (int batch_4 = 0; batch_4 < (row_end_4 - row_begin_4 + 128 - 1) / 128; batch_4++) {
                            for (int e_12 = 0; e_12 < 8; e_12++) {
                                qsum_2[e_12] = 0.0f;
                            }
                            int brow_4 = row_begin_4 + batch_4 * 128;
                            float vec_load_13[8];
                            {
                                const uint4* _vptr_3 = reinterpret_cast<const uint4*>(Q + ((((row_end_4 > brow_4 + rgrp_2) ? brow_4 + rgrp_2 : row_end_4 - 1) * num_heads + head_0_4) * 128 + col_2) + 0);
                                uint4 _vld_3[1];
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
                                    _vld_3[_blk] = _vptr_3[_blk];
                                    uint32_t* _vpairs_3 = reinterpret_cast<uint32_t*>(&_vld_3[_blk]);
                                    #pragma unroll
                                    for (int _pair = 0; _pair < 4; _pair++) {
                                        (&vec_load_13[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_3[_pair]) << 16);
                                        (&vec_load_13[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_3[_pair]) & 0xffff0000u);
                                    }
                                }
                            }
                            float vec_load_1_4[8];
                            {
                                const uint4* _vptr_4 = reinterpret_cast<const uint4*>(Q + ((((row_end_4 > brow_4 + 16 + rgrp_2) ? brow_4 + 16 + rgrp_2 : row_end_4 - 1) * num_heads + head_0_4) * 128 + col_2) + 0);
                                uint4 _vld_4[1];
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
                                    _vld_4[_blk] = _vptr_4[_blk];
                                    uint32_t* _vpairs_4 = reinterpret_cast<uint32_t*>(&_vld_4[_blk]);
                                    #pragma unroll
                                    for (int _pair = 0; _pair < 4; _pair++) {
                                        (&vec_load_1_4[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_4[_pair]) << 16);
                                        (&vec_load_1_4[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_4[_pair]) & 0xffff0000u);
                                    }
                                }
                            }
                            float vec_load_2_4[8];
                            {
                                const uint4* _vptr_5 = reinterpret_cast<const uint4*>(Q + ((((row_end_4 > brow_4 + 32 + rgrp_2) ? brow_4 + 32 + rgrp_2 : row_end_4 - 1) * num_heads + head_0_4) * 128 + col_2) + 0);
                                uint4 _vld_5[1];
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
                                    _vld_5[_blk] = _vptr_5[_blk];
                                    uint32_t* _vpairs_5 = reinterpret_cast<uint32_t*>(&_vld_5[_blk]);
                                    #pragma unroll
                                    for (int _pair = 0; _pair < 4; _pair++) {
                                        (&vec_load_2_4[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_5[_pair]) << 16);
                                        (&vec_load_2_4[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_5[_pair]) & 0xffff0000u);
                                    }
                                }
                            }
                            float vec_load_3_4[8];
                            {
                                const uint4* _vptr_6 = reinterpret_cast<const uint4*>(Q + ((((row_end_4 > brow_4 + 48 + rgrp_2) ? brow_4 + 48 + rgrp_2 : row_end_4 - 1) * num_heads + head_0_4) * 128 + col_2) + 0);
                                uint4 _vld_6[1];
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
                                    _vld_6[_blk] = _vptr_6[_blk];
                                    uint32_t* _vpairs_6 = reinterpret_cast<uint32_t*>(&_vld_6[_blk]);
                                    #pragma unroll
                                    for (int _pair = 0; _pair < 4; _pair++) {
                                        (&vec_load_3_4[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_6[_pair]) << 16);
                                        (&vec_load_3_4[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_6[_pair]) & 0xffff0000u);
                                    }
                                }
                            }
                            float vec_load_4_4[8];
                            {
                                const uint4* _vptr_7 = reinterpret_cast<const uint4*>(Q + ((((row_end_4 > brow_4 + 64 + rgrp_2) ? brow_4 + 64 + rgrp_2 : row_end_4 - 1) * num_heads + head_0_4) * 128 + col_2) + 0);
                                uint4 _vld_7[1];
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
                                    _vld_7[_blk] = _vptr_7[_blk];
                                    uint32_t* _vpairs_7 = reinterpret_cast<uint32_t*>(&_vld_7[_blk]);
                                    #pragma unroll
                                    for (int _pair = 0; _pair < 4; _pair++) {
                                        (&vec_load_4_4[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_7[_pair]) << 16);
                                        (&vec_load_4_4[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_7[_pair]) & 0xffff0000u);
                                    }
                                }
                            }
                            float vec_load_5_4[8];
                            {
                                const uint4* _vptr_8 = reinterpret_cast<const uint4*>(Q + ((((row_end_4 > brow_4 + 80 + rgrp_2) ? brow_4 + 80 + rgrp_2 : row_end_4 - 1) * num_heads + head_0_4) * 128 + col_2) + 0);
                                uint4 _vld_8[1];
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
                                    _vld_8[_blk] = _vptr_8[_blk];
                                    uint32_t* _vpairs_8 = reinterpret_cast<uint32_t*>(&_vld_8[_blk]);
                                    #pragma unroll
                                    for (int _pair = 0; _pair < 4; _pair++) {
                                        (&vec_load_5_4[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_8[_pair]) << 16);
                                        (&vec_load_5_4[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_8[_pair]) & 0xffff0000u);
                                    }
                                }
                            }
                            float vec_load_6_4[8];
                            {
                                const uint4* _vptr_9 = reinterpret_cast<const uint4*>(Q + ((((row_end_4 > brow_4 + 96 + rgrp_2) ? brow_4 + 96 + rgrp_2 : row_end_4 - 1) * num_heads + head_0_4) * 128 + col_2) + 0);
                                uint4 _vld_9[1];
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
                                    _vld_9[_blk] = _vptr_9[_blk];
                                    uint32_t* _vpairs_9 = reinterpret_cast<uint32_t*>(&_vld_9[_blk]);
                                    #pragma unroll
                                    for (int _pair = 0; _pair < 4; _pair++) {
                                        (&vec_load_6_4[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_9[_pair]) << 16);
                                        (&vec_load_6_4[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_9[_pair]) & 0xffff0000u);
                                    }
                                }
                            }
                            float vec_load_7_4[8];
                            {
                                const uint4* _vptr_10 = reinterpret_cast<const uint4*>(Q + ((((row_end_4 > brow_4 + 112 + rgrp_2) ? brow_4 + 112 + rgrp_2 : row_end_4 - 1) * num_heads + head_0_4) * 128 + col_2) + 0);
                                uint4 _vld_10[1];
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
                                    _vld_10[_blk] = _vptr_10[_blk];
                                    uint32_t* _vpairs_10 = reinterpret_cast<uint32_t*>(&_vld_10[_blk]);
                                    #pragma unroll
                                    for (int _pair = 0; _pair < 4; _pair++) {
                                        (&vec_load_7_4[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_10[_pair]) << 16);
                                        (&vec_load_7_4[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_10[_pair]) & 0xffff0000u);
                                    }
                                }
                            }
                            qf8_2[0] = vec_load_13[0];
                            qf8_2[1] = vec_load_13[1];
                            qf8_2[2] = vec_load_13[2];
                            qf8_2[3] = vec_load_13[3];
                            qf8_2[4] = vec_load_13[4];
                            qf8_2[5] = vec_load_13[5];
                            qf8_2[6] = vec_load_13[6];
                            qf8_2[7] = vec_load_13[7];
                            qsum_2[0] = qsum_2[0] + ((row_end_4 > brow_4 + rgrp_2) ? qf8_2[0] : 0.0f);
                            qsum_2[1] = qsum_2[1] + ((row_end_4 > brow_4 + rgrp_2) ? qf8_2[1] : 0.0f);
                            qsum_2[2] = qsum_2[2] + ((row_end_4 > brow_4 + rgrp_2) ? qf8_2[2] : 0.0f);
                            qsum_2[3] = qsum_2[3] + ((row_end_4 > brow_4 + rgrp_2) ? qf8_2[3] : 0.0f);
                            qsum_2[4] = qsum_2[4] + ((row_end_4 > brow_4 + rgrp_2) ? qf8_2[4] : 0.0f);
                            qsum_2[5] = qsum_2[5] + ((row_end_4 > brow_4 + rgrp_2) ? qf8_2[5] : 0.0f);
                            qsum_2[6] = qsum_2[6] + ((row_end_4 > brow_4 + rgrp_2) ? qf8_2[6] : 0.0f);
                            qsum_2[7] = qsum_2[7] + ((row_end_4 > brow_4 + rgrp_2) ? qf8_2[7] : 0.0f);
                            qf8_2[0] = vec_load_1_4[0];
                            qf8_2[1] = vec_load_1_4[1];
                            qf8_2[2] = vec_load_1_4[2];
                            qf8_2[3] = vec_load_1_4[3];
                            qf8_2[4] = vec_load_1_4[4];
                            qf8_2[5] = vec_load_1_4[5];
                            qf8_2[6] = vec_load_1_4[6];
                            qf8_2[7] = vec_load_1_4[7];
                            qsum_2[0] = qsum_2[0] + ((row_end_4 > brow_4 + 16 + rgrp_2) ? qf8_2[0] : 0.0f);
                            qsum_2[1] = qsum_2[1] + ((row_end_4 > brow_4 + 16 + rgrp_2) ? qf8_2[1] : 0.0f);
                            qsum_2[2] = qsum_2[2] + ((row_end_4 > brow_4 + 16 + rgrp_2) ? qf8_2[2] : 0.0f);
                            qsum_2[3] = qsum_2[3] + ((row_end_4 > brow_4 + 16 + rgrp_2) ? qf8_2[3] : 0.0f);
                            qsum_2[4] = qsum_2[4] + ((row_end_4 > brow_4 + 16 + rgrp_2) ? qf8_2[4] : 0.0f);
                            qsum_2[5] = qsum_2[5] + ((row_end_4 > brow_4 + 16 + rgrp_2) ? qf8_2[5] : 0.0f);
                            qsum_2[6] = qsum_2[6] + ((row_end_4 > brow_4 + 16 + rgrp_2) ? qf8_2[6] : 0.0f);
                            qsum_2[7] = qsum_2[7] + ((row_end_4 > brow_4 + 16 + rgrp_2) ? qf8_2[7] : 0.0f);
                            qf8_2[0] = vec_load_2_4[0];
                            qf8_2[1] = vec_load_2_4[1];
                            qf8_2[2] = vec_load_2_4[2];
                            qf8_2[3] = vec_load_2_4[3];
                            qf8_2[4] = vec_load_2_4[4];
                            qf8_2[5] = vec_load_2_4[5];
                            qf8_2[6] = vec_load_2_4[6];
                            qf8_2[7] = vec_load_2_4[7];
                            qsum_2[0] = qsum_2[0] + ((row_end_4 > brow_4 + 32 + rgrp_2) ? qf8_2[0] : 0.0f);
                            qsum_2[1] = qsum_2[1] + ((row_end_4 > brow_4 + 32 + rgrp_2) ? qf8_2[1] : 0.0f);
                            qsum_2[2] = qsum_2[2] + ((row_end_4 > brow_4 + 32 + rgrp_2) ? qf8_2[2] : 0.0f);
                            qsum_2[3] = qsum_2[3] + ((row_end_4 > brow_4 + 32 + rgrp_2) ? qf8_2[3] : 0.0f);
                            qsum_2[4] = qsum_2[4] + ((row_end_4 > brow_4 + 32 + rgrp_2) ? qf8_2[4] : 0.0f);
                            qsum_2[5] = qsum_2[5] + ((row_end_4 > brow_4 + 32 + rgrp_2) ? qf8_2[5] : 0.0f);
                            qsum_2[6] = qsum_2[6] + ((row_end_4 > brow_4 + 32 + rgrp_2) ? qf8_2[6] : 0.0f);
                            qsum_2[7] = qsum_2[7] + ((row_end_4 > brow_4 + 32 + rgrp_2) ? qf8_2[7] : 0.0f);
                            qf8_2[0] = vec_load_3_4[0];
                            qf8_2[1] = vec_load_3_4[1];
                            qf8_2[2] = vec_load_3_4[2];
                            qf8_2[3] = vec_load_3_4[3];
                            qf8_2[4] = vec_load_3_4[4];
                            qf8_2[5] = vec_load_3_4[5];
                            qf8_2[6] = vec_load_3_4[6];
                            qf8_2[7] = vec_load_3_4[7];
                            qsum_2[0] = qsum_2[0] + ((row_end_4 > brow_4 + 48 + rgrp_2) ? qf8_2[0] : 0.0f);
                            qsum_2[1] = qsum_2[1] + ((row_end_4 > brow_4 + 48 + rgrp_2) ? qf8_2[1] : 0.0f);
                            qsum_2[2] = qsum_2[2] + ((row_end_4 > brow_4 + 48 + rgrp_2) ? qf8_2[2] : 0.0f);
                            qsum_2[3] = qsum_2[3] + ((row_end_4 > brow_4 + 48 + rgrp_2) ? qf8_2[3] : 0.0f);
                            qsum_2[4] = qsum_2[4] + ((row_end_4 > brow_4 + 48 + rgrp_2) ? qf8_2[4] : 0.0f);
                            qsum_2[5] = qsum_2[5] + ((row_end_4 > brow_4 + 48 + rgrp_2) ? qf8_2[5] : 0.0f);
                            qsum_2[6] = qsum_2[6] + ((row_end_4 > brow_4 + 48 + rgrp_2) ? qf8_2[6] : 0.0f);
                            qsum_2[7] = qsum_2[7] + ((row_end_4 > brow_4 + 48 + rgrp_2) ? qf8_2[7] : 0.0f);
                            qf8_2[0] = vec_load_4_4[0];
                            qf8_2[1] = vec_load_4_4[1];
                            qf8_2[2] = vec_load_4_4[2];
                            qf8_2[3] = vec_load_4_4[3];
                            qf8_2[4] = vec_load_4_4[4];
                            qf8_2[5] = vec_load_4_4[5];
                            qf8_2[6] = vec_load_4_4[6];
                            qf8_2[7] = vec_load_4_4[7];
                            qsum_2[0] = qsum_2[0] + ((row_end_4 > brow_4 + 64 + rgrp_2) ? qf8_2[0] : 0.0f);
                            qsum_2[1] = qsum_2[1] + ((row_end_4 > brow_4 + 64 + rgrp_2) ? qf8_2[1] : 0.0f);
                            qsum_2[2] = qsum_2[2] + ((row_end_4 > brow_4 + 64 + rgrp_2) ? qf8_2[2] : 0.0f);
                            qsum_2[3] = qsum_2[3] + ((row_end_4 > brow_4 + 64 + rgrp_2) ? qf8_2[3] : 0.0f);
                            qsum_2[4] = qsum_2[4] + ((row_end_4 > brow_4 + 64 + rgrp_2) ? qf8_2[4] : 0.0f);
                            qsum_2[5] = qsum_2[5] + ((row_end_4 > brow_4 + 64 + rgrp_2) ? qf8_2[5] : 0.0f);
                            qsum_2[6] = qsum_2[6] + ((row_end_4 > brow_4 + 64 + rgrp_2) ? qf8_2[6] : 0.0f);
                            qsum_2[7] = qsum_2[7] + ((row_end_4 > brow_4 + 64 + rgrp_2) ? qf8_2[7] : 0.0f);
                            qf8_2[0] = vec_load_5_4[0];
                            qf8_2[1] = vec_load_5_4[1];
                            qf8_2[2] = vec_load_5_4[2];
                            qf8_2[3] = vec_load_5_4[3];
                            qf8_2[4] = vec_load_5_4[4];
                            qf8_2[5] = vec_load_5_4[5];
                            qf8_2[6] = vec_load_5_4[6];
                            qf8_2[7] = vec_load_5_4[7];
                            qsum_2[0] = qsum_2[0] + ((row_end_4 > brow_4 + 80 + rgrp_2) ? qf8_2[0] : 0.0f);
                            qsum_2[1] = qsum_2[1] + ((row_end_4 > brow_4 + 80 + rgrp_2) ? qf8_2[1] : 0.0f);
                            qsum_2[2] = qsum_2[2] + ((row_end_4 > brow_4 + 80 + rgrp_2) ? qf8_2[2] : 0.0f);
                            qsum_2[3] = qsum_2[3] + ((row_end_4 > brow_4 + 80 + rgrp_2) ? qf8_2[3] : 0.0f);
                            qsum_2[4] = qsum_2[4] + ((row_end_4 > brow_4 + 80 + rgrp_2) ? qf8_2[4] : 0.0f);
                            qsum_2[5] = qsum_2[5] + ((row_end_4 > brow_4 + 80 + rgrp_2) ? qf8_2[5] : 0.0f);
                            qsum_2[6] = qsum_2[6] + ((row_end_4 > brow_4 + 80 + rgrp_2) ? qf8_2[6] : 0.0f);
                            qsum_2[7] = qsum_2[7] + ((row_end_4 > brow_4 + 80 + rgrp_2) ? qf8_2[7] : 0.0f);
                            qf8_2[0] = vec_load_6_4[0];
                            qf8_2[1] = vec_load_6_4[1];
                            qf8_2[2] = vec_load_6_4[2];
                            qf8_2[3] = vec_load_6_4[3];
                            qf8_2[4] = vec_load_6_4[4];
                            qf8_2[5] = vec_load_6_4[5];
                            qf8_2[6] = vec_load_6_4[6];
                            qf8_2[7] = vec_load_6_4[7];
                            qsum_2[0] = qsum_2[0] + ((row_end_4 > brow_4 + 96 + rgrp_2) ? qf8_2[0] : 0.0f);
                            qsum_2[1] = qsum_2[1] + ((row_end_4 > brow_4 + 96 + rgrp_2) ? qf8_2[1] : 0.0f);
                            qsum_2[2] = qsum_2[2] + ((row_end_4 > brow_4 + 96 + rgrp_2) ? qf8_2[2] : 0.0f);
                            qsum_2[3] = qsum_2[3] + ((row_end_4 > brow_4 + 96 + rgrp_2) ? qf8_2[3] : 0.0f);
                            qsum_2[4] = qsum_2[4] + ((row_end_4 > brow_4 + 96 + rgrp_2) ? qf8_2[4] : 0.0f);
                            qsum_2[5] = qsum_2[5] + ((row_end_4 > brow_4 + 96 + rgrp_2) ? qf8_2[5] : 0.0f);
                            qsum_2[6] = qsum_2[6] + ((row_end_4 > brow_4 + 96 + rgrp_2) ? qf8_2[6] : 0.0f);
                            qsum_2[7] = qsum_2[7] + ((row_end_4 > brow_4 + 96 + rgrp_2) ? qf8_2[7] : 0.0f);
                            qf8_2[0] = vec_load_7_4[0];
                            qf8_2[1] = vec_load_7_4[1];
                            qf8_2[2] = vec_load_7_4[2];
                            qf8_2[3] = vec_load_7_4[3];
                            qf8_2[4] = vec_load_7_4[4];
                            qf8_2[5] = vec_load_7_4[5];
                            qf8_2[6] = vec_load_7_4[6];
                            qf8_2[7] = vec_load_7_4[7];
                            qsum_2[0] = qsum_2[0] + ((row_end_4 > brow_4 + 112 + rgrp_2) ? qf8_2[0] : 0.0f);
                            qsum_2[1] = qsum_2[1] + ((row_end_4 > brow_4 + 112 + rgrp_2) ? qf8_2[1] : 0.0f);
                            qsum_2[2] = qsum_2[2] + ((row_end_4 > brow_4 + 112 + rgrp_2) ? qf8_2[2] : 0.0f);
                            qsum_2[3] = qsum_2[3] + ((row_end_4 > brow_4 + 112 + rgrp_2) ? qf8_2[3] : 0.0f);
                            qsum_2[4] = qsum_2[4] + ((row_end_4 > brow_4 + 112 + rgrp_2) ? qf8_2[4] : 0.0f);
                            qsum_2[5] = qsum_2[5] + ((row_end_4 > brow_4 + 112 + rgrp_2) ? qf8_2[5] : 0.0f);
                            qsum_2[6] = qsum_2[6] + ((row_end_4 > brow_4 + 112 + rgrp_2) ? qf8_2[6] : 0.0f);
                            qsum_2[7] = qsum_2[7] + ((row_end_4 > brow_4 + 112 + rgrp_2) ? qf8_2[7] : 0.0f);
                            for (int e_13 = 0; e_13 < 8; e_13++) {
                                red_q[4096 + rgrp_2 * 128 + col_2 + e_13] = qsum_2[e_13];
                            }
                            asm volatile("bar.sync 3, 256;" ::: "memory");
                            int qt_2 = qt_base_2 + batch_4;
                            if (wtid_2 < 128) {
                                float qtot_2 = 0.0f;
                                for (int g_6 = 0; g_6 < 16; g_6++) {
                                    qtot_2 = qtot_2 + red_q[4096 + g_6 * 128 + wtid_2];
                                }
                                int brows_2 = ((row_end_4 - brow_4 < 128) ? row_end_4 - brow_4 : 128);
                                float _fdiv_11_2 = __fdiv_rn(qtot_2, (float)brows_2);
                                *(reinterpret_cast<float*>(q_mean + ((qt_2 * num_heads + head_0_4) * 128 + wtid_2)) + (0)) = _fdiv_11_2;
                            }
                            asm volatile("bar.sync 3, 256;" ::: "memory");
                            #pragma unroll 1
                            for (int blk_2 = 0; blk_2 < 4; blk_2++) {
                                int tok_4 = brow_4 + blk_2 * 32 + trow_2;
                                int valid_4 = ((tok_4 < row_end_4) ? 1 : 0);
                                int tok_c_4 = ((tok_4 < row_end_4) ? tok_4 : row_end_4 - 1);
                                int is_first_2 = ((valid_4 == 1 && tok_4 == brow_4) ? 1 : 0);
                                float vec_load_9_1[16];
                                {
                                    const uint4* _vptr_12 = reinterpret_cast<const uint4*>(Q + ((tok_c_4 * num_heads + head_0_4) * 128 + col16_2) + 0);
                                    uint4 _vld_12[2];
                                    #pragma unroll
                                    for (int _blk = 0; _blk < 2; _blk++) {
                                        _vld_12[_blk] = _vptr_12[_blk];
                                        uint32_t* _vpairs_12 = reinterpret_cast<uint32_t*>(&_vld_12[_blk]);
                                        #pragma unroll
                                        for (int _pair = 0; _pair < 4; _pair++) {
                                            (&vec_load_9_1[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_12[_pair]) << 16);
                                            (&vec_load_9_1[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_12[_pair]) & 0xffff0000u);
                                        }
                                    }
                                }
                                int qmbase_2 = (qt_2 * num_heads + head_0_4) * 128 + col16_2;
                                float qmv_2[16];
                                float qc_2[16];
                                for (int quad_4 = 0; quad_4 < 4; quad_4++) {
                                    float _vec_load_10_2[4];
                                    {
                                        float4 _v4 = *reinterpret_cast<const float4*>(q_mean + (qmbase_2 + quad_4 * 4) + 0);
                                        _vec_load_10_2[0 + 0] = _v4.x;
                                        _vec_load_10_2[0 + 1] = _v4.y;
                                        _vec_load_10_2[0 + 2] = _v4.z;
                                        _vec_load_10_2[0 + 3] = _v4.w;
                                    }
                                    for (int e_14 = 0; e_14 < 4; e_14++) {
                                        qmv_2[quad_4 * 4 + e_14] = _vec_load_10_2[e_14];
                                        float qf_2 = vec_load_9_1[quad_4 * 4 + e_14];
                                        qc_2[quad_4 * 4 + e_14] = ((valid_4 == 1) ? qf_2 - _vec_load_10_2[e_14] : 0.0f);
                                    }
                                }
                                unsigned int qw_2[2];
                                float qsf_a_2[1];
                                float max_1_4 = fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fabsf(qc_2[0]), fabsf(qc_2[1])), fabsf(qc_2[2])), fabsf(qc_2[3])), fabsf(qc_2[4])), fabsf(qc_2[5])), fabsf(qc_2[6])), fabsf(qc_2[7])), fabsf(qc_2[8])), fabsf(qc_2[9])), fabsf(qc_2[10])), fabsf(qc_2[11])), fabsf(qc_2[12])), fabsf(qc_2[13])), fabsf(qc_2[14])), fabsf(qc_2[15]));
                                uint16_t _cvt_satfinite_e4m3x2_f32_14_2;
                                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_14_2) : "f"(0.0f), "f"(max_1_4 * 0.16666667f));
                                uint16_t _e4m3x2_decode_15 = (uint16_t)((unsigned int)_cvt_satfinite_e4m3x2_f32_14_2 & 0xFFu);
                                uint32_t _f16x2_decode_15;
                                float fp8_decode_4;
                                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_decode_15) : "h"(_e4m3x2_decode_15));
                                uint16_t _f16_decode_15 = (uint16_t)_f16x2_decode_15;
                                asm("cvt.f32.f16 %0, %1;" : "=f"(fp8_decode_4) : "h"(_f16_decode_15));
                                float fdiv_rn_4 = __fdiv_rn(1.0f, fp8_decode_4);
                                qc_2[0] = qc_2[0] * ((fp8_decode_4 != 0.0f) ? fdiv_rn_4 : 0.0f);
                                qc_2[1] = qc_2[1] * ((fp8_decode_4 != 0.0f) ? fdiv_rn_4 : 0.0f);
                                qc_2[2] = qc_2[2] * ((fp8_decode_4 != 0.0f) ? fdiv_rn_4 : 0.0f);
                                qc_2[3] = qc_2[3] * ((fp8_decode_4 != 0.0f) ? fdiv_rn_4 : 0.0f);
                                qc_2[4] = qc_2[4] * ((fp8_decode_4 != 0.0f) ? fdiv_rn_4 : 0.0f);
                                qc_2[5] = qc_2[5] * ((fp8_decode_4 != 0.0f) ? fdiv_rn_4 : 0.0f);
                                qc_2[6] = qc_2[6] * ((fp8_decode_4 != 0.0f) ? fdiv_rn_4 : 0.0f);
                                qc_2[7] = qc_2[7] * ((fp8_decode_4 != 0.0f) ? fdiv_rn_4 : 0.0f);
                                qc_2[8] = qc_2[8] * ((fp8_decode_4 != 0.0f) ? fdiv_rn_4 : 0.0f);
                                qc_2[9] = qc_2[9] * ((fp8_decode_4 != 0.0f) ? fdiv_rn_4 : 0.0f);
                                qc_2[10] = qc_2[10] * ((fp8_decode_4 != 0.0f) ? fdiv_rn_4 : 0.0f);
                                qc_2[11] = qc_2[11] * ((fp8_decode_4 != 0.0f) ? fdiv_rn_4 : 0.0f);
                                qc_2[12] = qc_2[12] * ((fp8_decode_4 != 0.0f) ? fdiv_rn_4 : 0.0f);
                                qc_2[13] = qc_2[13] * ((fp8_decode_4 != 0.0f) ? fdiv_rn_4 : 0.0f);
                                qc_2[14] = qc_2[14] * ((fp8_decode_4 != 0.0f) ? fdiv_rn_4 : 0.0f);
                                qc_2[15] = qc_2[15] * ((fp8_decode_4 != 0.0f) ? fdiv_rn_4 : 0.0f);
                                asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(qw_2[0]) : "f"(qc_2[0]), "f"(qc_2[1]), "f"(qc_2[2]), "f"(qc_2[3]), "f"(qc_2[4]), "f"(qc_2[5]), "f"(qc_2[6]), "f"(qc_2[7]));
                                asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(qw_2[1]) : "f"(qc_2[8]), "f"(qc_2[9]), "f"(qc_2[10]), "f"(qc_2[11]), "f"(qc_2[12]), "f"(qc_2[13]), "f"(qc_2[14]), "f"(qc_2[15]));
                                qsf_a_2[0] = max_1_4 * 0.16666667f;
                                unsigned int qstore_2[4];
                                qstore_2[0] = qw_2[0];
                                qstore_2[1] = qw_2[1];
                                unsigned int _shfl_xor_sync_16_2 = __shfl_xor_sync(0xFFFFFFFF, qw_2[0], 1);
                                qstore_2[2] = _shfl_xor_sync_16_2;
                                unsigned int _shfl_xor_sync_17_2 = __shfl_xor_sync(0xFFFFFFFF, qw_2[1], 1);
                                qstore_2[3] = _shfl_xor_sync_17_2;
                                if (valid_4 == 1) {
                                    uint16_t _cvt_satfinite_e4m3x2_f32_18_2;
                                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_18_2) : "f"(0.0f), "f"(qsf_a_2[0]));
                                    *reinterpret_cast<unsigned char*>(q_sf + ((tok_4 * num_heads + head_0_4) * 8 + sub_2)) = (unsigned char)((unsigned int)(unsigned char)(_cvt_satfinite_e4m3x2_f32_18_2 & 127));
                                    if ((sub_2 & 1) == 0) {
                                        reinterpret_cast<int4*>(q4 + (((tok_4 * num_heads + head_0_4) * 64 + sub_2 * 8) / 4))[0] = reinterpret_cast<int4*>(qstore_2)[0];
                                    }
                                }
                                if (is_first_2 == 1) {
                                    unsigned int mw_2[2];
                                    float msf_a_2[1];
                                    float max_1_2_1 = fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fabsf(qmv_2[0]), fabsf(qmv_2[1])), fabsf(qmv_2[2])), fabsf(qmv_2[3])), fabsf(qmv_2[4])), fabsf(qmv_2[5])), fabsf(qmv_2[6])), fabsf(qmv_2[7])), fabsf(qmv_2[8])), fabsf(qmv_2[9])), fabsf(qmv_2[10])), fabsf(qmv_2[11])), fabsf(qmv_2[12])), fabsf(qmv_2[13])), fabsf(qmv_2[14])), fabsf(qmv_2[15]));
                                    uint16_t _cvt_satfinite_e4m3x2_f32_19_2;
                                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_19_2) : "f"(0.0f), "f"(max_1_2_1 * 0.16666667f));
                                    uint16_t _e4m3x2_decode_20 = (uint16_t)((unsigned int)_cvt_satfinite_e4m3x2_f32_19_2 & 0xFFu);
                                    uint32_t _f16x2_decode_20;
                                    float fp8_decode_2_1;
                                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_decode_20) : "h"(_e4m3x2_decode_20));
                                    uint16_t _f16_decode_20 = (uint16_t)_f16x2_decode_20;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(fp8_decode_2_1) : "h"(_f16_decode_20));
                                    float fdiv_rn_2_1 = __fdiv_rn(1.0f, fp8_decode_2_1);
                                    qmv_2[0] = qmv_2[0] * ((fp8_decode_2_1 != 0.0f) ? fdiv_rn_2_1 : 0.0f);
                                    qmv_2[1] = qmv_2[1] * ((fp8_decode_2_1 != 0.0f) ? fdiv_rn_2_1 : 0.0f);
                                    qmv_2[2] = qmv_2[2] * ((fp8_decode_2_1 != 0.0f) ? fdiv_rn_2_1 : 0.0f);
                                    qmv_2[3] = qmv_2[3] * ((fp8_decode_2_1 != 0.0f) ? fdiv_rn_2_1 : 0.0f);
                                    qmv_2[4] = qmv_2[4] * ((fp8_decode_2_1 != 0.0f) ? fdiv_rn_2_1 : 0.0f);
                                    qmv_2[5] = qmv_2[5] * ((fp8_decode_2_1 != 0.0f) ? fdiv_rn_2_1 : 0.0f);
                                    qmv_2[6] = qmv_2[6] * ((fp8_decode_2_1 != 0.0f) ? fdiv_rn_2_1 : 0.0f);
                                    qmv_2[7] = qmv_2[7] * ((fp8_decode_2_1 != 0.0f) ? fdiv_rn_2_1 : 0.0f);
                                    qmv_2[8] = qmv_2[8] * ((fp8_decode_2_1 != 0.0f) ? fdiv_rn_2_1 : 0.0f);
                                    qmv_2[9] = qmv_2[9] * ((fp8_decode_2_1 != 0.0f) ? fdiv_rn_2_1 : 0.0f);
                                    qmv_2[10] = qmv_2[10] * ((fp8_decode_2_1 != 0.0f) ? fdiv_rn_2_1 : 0.0f);
                                    qmv_2[11] = qmv_2[11] * ((fp8_decode_2_1 != 0.0f) ? fdiv_rn_2_1 : 0.0f);
                                    qmv_2[12] = qmv_2[12] * ((fp8_decode_2_1 != 0.0f) ? fdiv_rn_2_1 : 0.0f);
                                    qmv_2[13] = qmv_2[13] * ((fp8_decode_2_1 != 0.0f) ? fdiv_rn_2_1 : 0.0f);
                                    qmv_2[14] = qmv_2[14] * ((fp8_decode_2_1 != 0.0f) ? fdiv_rn_2_1 : 0.0f);
                                    qmv_2[15] = qmv_2[15] * ((fp8_decode_2_1 != 0.0f) ? fdiv_rn_2_1 : 0.0f);
                                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(mw_2[0]) : "f"(qmv_2[0]), "f"(qmv_2[1]), "f"(qmv_2[2]), "f"(qmv_2[3]), "f"(qmv_2[4]), "f"(qmv_2[5]), "f"(qmv_2[6]), "f"(qmv_2[7]));
                                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(mw_2[1]) : "f"(qmv_2[8]), "f"(qmv_2[9]), "f"(qmv_2[10]), "f"(qmv_2[11]), "f"(qmv_2[12]), "f"(qmv_2[13]), "f"(qmv_2[14]), "f"(qmv_2[15]));
                                    msf_a_2[0] = max_1_2_1 * 0.16666667f;
                                    int qmrow_2 = qt_2 * num_heads + head_0_4;
                                    *(reinterpret_cast<unsigned int*>(qm4 + (qmrow_2 * 16 + sub_2 * 2)) + (0)) = mw_2[0];
                                    *(reinterpret_cast<unsigned int*>(qm4 + (qmrow_2 * 16 + sub_2 * 2 + 1)) + (0)) = mw_2[1];
                                    uint16_t _cvt_satfinite_e4m3x2_f32_21_2;
                                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_21_2) : "f"(0.0f), "f"(msf_a_2[0]));
                                    *reinterpret_cast<unsigned char*>(qm_sf + (qmrow_2 * 8 + sub_2)) = (unsigned char)((unsigned int)(unsigned char)(_cvt_satfinite_e4m3x2_f32_21_2 & 127));
                                }
                            }
                        }
                    } else if (ri_2 < num_tiles + num_tblk_2) {
                        int vlive_2[8];
                        vlive_2[0] = (((ri_2 - num_tiles) * 32 + (vblk_2 * 4 + vhalf_2 * 2) < total_tokens) ? 1 : 0);
                        float _vec_load_11_2[2];
                        {
                            uint32_t _bf16x2_bits_22;
                            _bf16x2_bits_22 = *reinterpret_cast<const uint32_t*>(V + (((((ri_2 - num_tiles) * 32 + (vblk_2 * 4 + vhalf_2 * 2) < total_tokens) ? (ri_2 - num_tiles) * 32 + (vblk_2 * 4 + vhalf_2 * 2) : total_tokens - 1) * num_heads + head_2) * 128 + vdp_2 * 2) + 0);
                            (&_vec_load_11_2[0])[0] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_22) << 16);
                            (&_vec_load_11_2[0])[1] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_22) & 0xffff0000u);
                        }
                        vlive_2[1] = (((ri_2 - num_tiles) * 32 + (vblk_2 * 4 + vhalf_2 * 2 + 1) < total_tokens) ? 1 : 0);
                        float _vec_load_12_2[2];
                        {
                            uint32_t _bf16x2_bits_23;
                            _bf16x2_bits_23 = *reinterpret_cast<const uint32_t*>(V + (((((ri_2 - num_tiles) * 32 + (vblk_2 * 4 + vhalf_2 * 2 + 1) < total_tokens) ? (ri_2 - num_tiles) * 32 + (vblk_2 * 4 + vhalf_2 * 2 + 1) : total_tokens - 1) * num_heads + head_2) * 128 + vdp_2 * 2) + 0);
                            (&_vec_load_12_2[0])[0] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_23) << 16);
                            (&_vec_load_12_2[0])[1] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_23) & 0xffff0000u);
                        }
                        vlive_2[2] = (((ri_2 - num_tiles) * 32 + (8 + vblk_2 * 4 + vhalf_2 * 2) < total_tokens) ? 1 : 0);
                        float _vec_load_13_2[2];
                        {
                            uint32_t _bf16x2_bits_24;
                            _bf16x2_bits_24 = *reinterpret_cast<const uint32_t*>(V + (((((ri_2 - num_tiles) * 32 + (8 + vblk_2 * 4 + vhalf_2 * 2) < total_tokens) ? (ri_2 - num_tiles) * 32 + (8 + vblk_2 * 4 + vhalf_2 * 2) : total_tokens - 1) * num_heads + head_2) * 128 + vdp_2 * 2) + 0);
                            (&_vec_load_13_2[0])[0] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_24) << 16);
                            (&_vec_load_13_2[0])[1] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_24) & 0xffff0000u);
                        }
                        vlive_2[3] = (((ri_2 - num_tiles) * 32 + (8 + vblk_2 * 4 + vhalf_2 * 2 + 1) < total_tokens) ? 1 : 0);
                        float _vec_load_14_2[2];
                        {
                            uint32_t _bf16x2_bits_25;
                            _bf16x2_bits_25 = *reinterpret_cast<const uint32_t*>(V + (((((ri_2 - num_tiles) * 32 + (8 + vblk_2 * 4 + vhalf_2 * 2 + 1) < total_tokens) ? (ri_2 - num_tiles) * 32 + (8 + vblk_2 * 4 + vhalf_2 * 2 + 1) : total_tokens - 1) * num_heads + head_2) * 128 + vdp_2 * 2) + 0);
                            (&_vec_load_14_2[0])[0] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_25) << 16);
                            (&_vec_load_14_2[0])[1] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_25) & 0xffff0000u);
                        }
                        vlive_2[4] = (((ri_2 - num_tiles) * 32 + (16 + vblk_2 * 4 + vhalf_2 * 2) < total_tokens) ? 1 : 0);
                        float _vec_load_15_2[2];
                        {
                            uint32_t _bf16x2_bits_26;
                            _bf16x2_bits_26 = *reinterpret_cast<const uint32_t*>(V + (((((ri_2 - num_tiles) * 32 + (16 + vblk_2 * 4 + vhalf_2 * 2) < total_tokens) ? (ri_2 - num_tiles) * 32 + (16 + vblk_2 * 4 + vhalf_2 * 2) : total_tokens - 1) * num_heads + head_2) * 128 + vdp_2 * 2) + 0);
                            (&_vec_load_15_2[0])[0] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_26) << 16);
                            (&_vec_load_15_2[0])[1] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_26) & 0xffff0000u);
                        }
                        vlive_2[5] = (((ri_2 - num_tiles) * 32 + (16 + vblk_2 * 4 + vhalf_2 * 2 + 1) < total_tokens) ? 1 : 0);
                        float _vec_load_16_2[2];
                        {
                            uint32_t _bf16x2_bits_27;
                            _bf16x2_bits_27 = *reinterpret_cast<const uint32_t*>(V + (((((ri_2 - num_tiles) * 32 + (16 + vblk_2 * 4 + vhalf_2 * 2 + 1) < total_tokens) ? (ri_2 - num_tiles) * 32 + (16 + vblk_2 * 4 + vhalf_2 * 2 + 1) : total_tokens - 1) * num_heads + head_2) * 128 + vdp_2 * 2) + 0);
                            (&_vec_load_16_2[0])[0] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_27) << 16);
                            (&_vec_load_16_2[0])[1] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_27) & 0xffff0000u);
                        }
                        vlive_2[6] = (((ri_2 - num_tiles) * 32 + (24 + vblk_2 * 4 + vhalf_2 * 2) < total_tokens) ? 1 : 0);
                        float _vec_load_17_2[2];
                        {
                            uint32_t _bf16x2_bits_28;
                            _bf16x2_bits_28 = *reinterpret_cast<const uint32_t*>(V + (((((ri_2 - num_tiles) * 32 + (24 + vblk_2 * 4 + vhalf_2 * 2) < total_tokens) ? (ri_2 - num_tiles) * 32 + (24 + vblk_2 * 4 + vhalf_2 * 2) : total_tokens - 1) * num_heads + head_2) * 128 + vdp_2 * 2) + 0);
                            (&_vec_load_17_2[0])[0] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_28) << 16);
                            (&_vec_load_17_2[0])[1] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_28) & 0xffff0000u);
                        }
                        vlive_2[7] = (((ri_2 - num_tiles) * 32 + (24 + vblk_2 * 4 + vhalf_2 * 2 + 1) < total_tokens) ? 1 : 0);
                        float _vec_load_18_2[2];
                        {
                            uint32_t _bf16x2_bits_29;
                            _bf16x2_bits_29 = *reinterpret_cast<const uint32_t*>(V + (((((ri_2 - num_tiles) * 32 + (24 + vblk_2 * 4 + vhalf_2 * 2 + 1) < total_tokens) ? (ri_2 - num_tiles) * 32 + (24 + vblk_2 * 4 + vhalf_2 * 2 + 1) : total_tokens - 1) * num_heads + head_2) * 128 + vdp_2 * 2) + 0);
                            (&_vec_load_18_2[0])[0] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_29) << 16);
                            (&_vec_load_18_2[0])[1] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_29) & 0xffff0000u);
                        }
                        float va_2[8];
                        float vb_2[8];
                        va_2[0] = _vec_load_11_2[0];
                        vb_2[0] = _vec_load_11_2[1];
                        va_2[1] = _vec_load_12_2[0];
                        vb_2[1] = _vec_load_12_2[1];
                        va_2[2] = _vec_load_13_2[0];
                        vb_2[2] = _vec_load_13_2[1];
                        va_2[3] = _vec_load_14_2[0];
                        vb_2[3] = _vec_load_14_2[1];
                        va_2[4] = _vec_load_15_2[0];
                        vb_2[4] = _vec_load_15_2[1];
                        va_2[5] = _vec_load_16_2[0];
                        vb_2[5] = _vec_load_16_2[1];
                        va_2[6] = _vec_load_17_2[0];
                        vb_2[6] = _vec_load_17_2[1];
                        va_2[7] = _vec_load_18_2[0];
                        vb_2[7] = _vec_load_18_2[1];
                        va_2[0] = ((vlive_2[0] == 1) ? va_2[0] : 0.0f);
                        vb_2[0] = ((vlive_2[0] == 1) ? vb_2[0] : 0.0f);
                        va_2[1] = ((vlive_2[1] == 1) ? va_2[1] : 0.0f);
                        vb_2[1] = ((vlive_2[1] == 1) ? vb_2[1] : 0.0f);
                        va_2[2] = ((vlive_2[2] == 1) ? va_2[2] : 0.0f);
                        vb_2[2] = ((vlive_2[2] == 1) ? vb_2[2] : 0.0f);
                        va_2[3] = ((vlive_2[3] == 1) ? va_2[3] : 0.0f);
                        vb_2[3] = ((vlive_2[3] == 1) ? vb_2[3] : 0.0f);
                        va_2[4] = ((vlive_2[4] == 1) ? va_2[4] : 0.0f);
                        vb_2[4] = ((vlive_2[4] == 1) ? vb_2[4] : 0.0f);
                        va_2[5] = ((vlive_2[5] == 1) ? va_2[5] : 0.0f);
                        vb_2[5] = ((vlive_2[5] == 1) ? vb_2[5] : 0.0f);
                        va_2[6] = ((vlive_2[6] == 1) ? va_2[6] : 0.0f);
                        vb_2[6] = ((vlive_2[6] == 1) ? vb_2[6] : 0.0f);
                        va_2[7] = ((vlive_2[7] == 1) ? va_2[7] : 0.0f);
                        vb_2[7] = ((vlive_2[7] == 1) ? vb_2[7] : 0.0f);
                        float amax_a_2 = 0.0f;
                        float amax_b_2 = 0.0f;
                        for (int ii_4 = 0; ii_4 < 8; ii_4++) {
                            amax_a_2 = fmaxf(amax_a_2, fabsf(va_2[ii_4]));
                            amax_b_2 = fmaxf(amax_b_2, fabsf(vb_2[ii_4]));
                        }
                        float _shfl_xor_sync_30_2 = __shfl_xor_sync(0xFFFFFFFF, amax_a_2, 16);
                        amax_a_2 = fmaxf(amax_a_2, _shfl_xor_sync_30_2);
                        float _shfl_xor_sync_31_2 = __shfl_xor_sync(0xFFFFFFFF, amax_b_2, 16);
                        amax_b_2 = fmaxf(amax_b_2, _shfl_xor_sync_31_2);
                        float vsf_a_2 = amax_a_2 * 0.16666667f;
                        float vsf_b_2 = amax_b_2 * 0.16666667f;
                        uint16_t _e4m3x2_f32_0_2;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_0_2) : "f"(vsf_b_2), "f"(vsf_a_2));
                        uint16_t _e4m3x2_decode_32 = (uint16_t)((unsigned int)_e4m3x2_f32_0_2 & 0xFFu);
                        uint32_t _f16x2_decode_32;
                        float vdec_a_2;
                        asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_decode_32) : "h"(_e4m3x2_decode_32));
                        uint16_t _f16_decode_32 = (uint16_t)_f16x2_decode_32;
                        asm("cvt.f32.f16 %0, %1;" : "=f"(vdec_a_2) : "h"(_f16_decode_32));
                        uint16_t _e4m3x2_decode_33 = (uint16_t)((unsigned int)_e4m3x2_f32_0_2 >> 8 & 0xFFu);
                        uint32_t _f16x2_decode_33;
                        float vdec_b_2;
                        asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_decode_33) : "h"(_e4m3x2_decode_33));
                        uint16_t _f16_decode_33 = (uint16_t)_f16x2_decode_33;
                        asm("cvt.f32.f16 %0, %1;" : "=f"(vdec_b_2) : "h"(_f16_decode_33));
                        float _fdiv_34_2 = __fdiv_rn(1.0f, vdec_a_2);
                        float vinv_a_2 = ((vdec_a_2 != 0.0f) ? _fdiv_34_2 : 0.0f);
                        float _fdiv_35_2 = __fdiv_rn(1.0f, vdec_b_2);
                        float vinv_b_2 = ((vdec_b_2 != 0.0f) ? _fdiv_35_2 : 0.0f);
                        for (int ii_5 = 0; ii_5 < 8; ii_5++) {
                            va_2[ii_5] = va_2[ii_5] * vinv_a_2;
                            vb_2[ii_5] = vb_2[ii_5] * vinv_b_2;
                        }
                        unsigned int vwa_2[1];
                        unsigned int vwb_2[1];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(vwa_2[0]) : "f"(va_2[0]), "f"(va_2[1]), "f"(va_2[2]), "f"(va_2[3]), "f"(va_2[4]), "f"(va_2[5]), "f"(va_2[6]), "f"(va_2[7]));
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(vwb_2[0]) : "f"(vb_2[0]), "f"(vb_2[1]), "f"(vb_2[2]), "f"(vb_2[3]), "f"(vb_2[4]), "f"(vb_2[5]), "f"(vb_2[6]), "f"(vb_2[7]));
                        int row_words_2 = total_padded >> 3;
                        int vt_word_2 = (head_2 * 128 + vdp_2 * 2) * row_words_2 + ((ri_2 - num_tiles) * 32 >> 3) + vblk_2 * 2 + vhalf_2;
                        *(reinterpret_cast<unsigned int*>(vt4 + vt_word_2) + (0)) = vwa_2[0];
                        *(reinterpret_cast<unsigned int*>(vt4 + (vt_word_2 + row_words_2)) + (0)) = vwb_2[0];
                        if (vhalf_2 == 0) {
                            int vsf_base_2 = ((head_2 * num_kblocks + ((ri_2 - num_tiles) * 32 >> 7)) * 128 + vdp_2 * 2) * 8 + ((((ri_2 - num_tiles) * 32 & 127) >> 4) + vblk_2);
                            uint16_t _cvt_satfinite_e4m3x2_f32_36_2;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_36_2) : "f"(0.0f), "f"(vsf_a_2));
                            *reinterpret_cast<unsigned char*>(v_sf + vsf_base_2) = (unsigned char)((unsigned int)(unsigned char)(_cvt_satfinite_e4m3x2_f32_36_2 & 127));
                            uint16_t _cvt_satfinite_e4m3x2_f32_37_2;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_37_2) : "f"(0.0f), "f"(vsf_b_2));
                            *reinterpret_cast<unsigned char*>(v_sf + (vsf_base_2 + 8)) = (unsigned char)((unsigned int)(unsigned char)(_cvt_satfinite_e4m3x2_f32_37_2 & 127));
                        }
                    } else {
                        int tile_5 = ((ri_2 - num_tiles - num_tblk_2) * num_heads + head_2) / num_heads;
                        int head_0_5 = ((ri_2 - num_tiles - num_tblk_2) * num_heads + head_2) % num_heads;
                        int _vec_load_19_2[4];
                        {
                            const int4* _ivptr_38 = reinterpret_cast<const int4*>(tile_table + (8 * tile_5) + 0);
                            int4 _ivld_38;
                            _ivld_38 = *_ivptr_38;
                            _vec_load_19_2[0 + 0] = _ivld_38.x;
                            _vec_load_19_2[0 + 1] = _ivld_38.y;
                            _vec_load_19_2[0 + 2] = _ivld_38.z;
                            _vec_load_19_2[0 + 3] = _ivld_38.w;
                        }
                        int _vec_load_20_2[4];
                        {
                            const int4* _ivptr_39 = reinterpret_cast<const int4*>(tile_table + (8 * tile_5 + 4) + 0);
                            int4 _ivld_39;
                            _ivld_39 = *_ivptr_39;
                            _vec_load_20_2[0 + 0] = _ivld_39.x;
                            _vec_load_20_2[0 + 1] = _ivld_39.y;
                            _vec_load_20_2[0 + 2] = _ivld_39.z;
                            _vec_load_20_2[0 + 3] = _ivld_39.w;
                        }
                        int seg_7 = _vec_load_19_2[0];
                        int row_begin_5 = _vec_load_19_2[1];
                        int row_end_5 = _vec_load_19_2[2];
                        int t0_5 = _vec_load_19_2[3];
                        int t1_2 = _vec_load_20_2[0];
                        int length_2 = _vec_load_20_2[1];
                        float ksum_2[8];
                        for (int e_15 = 0; e_15 < 8; e_15++) {
                            ksum_2[e_15] = 0.0f;
                        }
                        float kf8_2[8];
                        #pragma unroll 1
                        for (int batch_5 = 0; batch_5 < (row_end_5 - row_begin_5 + 128 - 1) / 128; batch_5++) {
                            int brow_5 = row_begin_5 + batch_5 * 128;
                            float vec_load_14[8];
                            {
                                const uint4* _vptr_40 = reinterpret_cast<const uint4*>(K + ((((row_end_5 > brow_5 + rgrp_2) ? brow_5 + rgrp_2 : row_end_5 - 1) * num_heads + head_0_5) * 128 + col_2) + 0);
                                uint4 _vld_40[1];
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
                                    _vld_40[_blk] = _vptr_40[_blk];
                                    uint32_t* _vpairs_40 = reinterpret_cast<uint32_t*>(&_vld_40[_blk]);
                                    #pragma unroll
                                    for (int _pair = 0; _pair < 4; _pair++) {
                                        (&vec_load_14[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_40[_pair]) << 16);
                                        (&vec_load_14[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_40[_pair]) & 0xffff0000u);
                                    }
                                }
                            }
                            float vec_load_1_5[8];
                            {
                                const uint4* _vptr_41 = reinterpret_cast<const uint4*>(K + ((((row_end_5 > brow_5 + 16 + rgrp_2) ? brow_5 + 16 + rgrp_2 : row_end_5 - 1) * num_heads + head_0_5) * 128 + col_2) + 0);
                                uint4 _vld_41[1];
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
                                    _vld_41[_blk] = _vptr_41[_blk];
                                    uint32_t* _vpairs_41 = reinterpret_cast<uint32_t*>(&_vld_41[_blk]);
                                    #pragma unroll
                                    for (int _pair = 0; _pair < 4; _pair++) {
                                        (&vec_load_1_5[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_41[_pair]) << 16);
                                        (&vec_load_1_5[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_41[_pair]) & 0xffff0000u);
                                    }
                                }
                            }
                            float vec_load_2_5[8];
                            {
                                const uint4* _vptr_42 = reinterpret_cast<const uint4*>(K + ((((row_end_5 > brow_5 + 32 + rgrp_2) ? brow_5 + 32 + rgrp_2 : row_end_5 - 1) * num_heads + head_0_5) * 128 + col_2) + 0);
                                uint4 _vld_42[1];
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
                                    _vld_42[_blk] = _vptr_42[_blk];
                                    uint32_t* _vpairs_42 = reinterpret_cast<uint32_t*>(&_vld_42[_blk]);
                                    #pragma unroll
                                    for (int _pair = 0; _pair < 4; _pair++) {
                                        (&vec_load_2_5[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_42[_pair]) << 16);
                                        (&vec_load_2_5[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_42[_pair]) & 0xffff0000u);
                                    }
                                }
                            }
                            float vec_load_3_5[8];
                            {
                                const uint4* _vptr_43 = reinterpret_cast<const uint4*>(K + ((((row_end_5 > brow_5 + 48 + rgrp_2) ? brow_5 + 48 + rgrp_2 : row_end_5 - 1) * num_heads + head_0_5) * 128 + col_2) + 0);
                                uint4 _vld_43[1];
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
                                    _vld_43[_blk] = _vptr_43[_blk];
                                    uint32_t* _vpairs_43 = reinterpret_cast<uint32_t*>(&_vld_43[_blk]);
                                    #pragma unroll
                                    for (int _pair = 0; _pair < 4; _pair++) {
                                        (&vec_load_3_5[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_43[_pair]) << 16);
                                        (&vec_load_3_5[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_43[_pair]) & 0xffff0000u);
                                    }
                                }
                            }
                            float vec_load_4_5[8];
                            {
                                const uint4* _vptr_44 = reinterpret_cast<const uint4*>(K + ((((row_end_5 > brow_5 + 64 + rgrp_2) ? brow_5 + 64 + rgrp_2 : row_end_5 - 1) * num_heads + head_0_5) * 128 + col_2) + 0);
                                uint4 _vld_44[1];
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
                                    _vld_44[_blk] = _vptr_44[_blk];
                                    uint32_t* _vpairs_44 = reinterpret_cast<uint32_t*>(&_vld_44[_blk]);
                                    #pragma unroll
                                    for (int _pair = 0; _pair < 4; _pair++) {
                                        (&vec_load_4_5[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_44[_pair]) << 16);
                                        (&vec_load_4_5[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_44[_pair]) & 0xffff0000u);
                                    }
                                }
                            }
                            float vec_load_5_5[8];
                            {
                                const uint4* _vptr_45 = reinterpret_cast<const uint4*>(K + ((((row_end_5 > brow_5 + 80 + rgrp_2) ? brow_5 + 80 + rgrp_2 : row_end_5 - 1) * num_heads + head_0_5) * 128 + col_2) + 0);
                                uint4 _vld_45[1];
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
                                    _vld_45[_blk] = _vptr_45[_blk];
                                    uint32_t* _vpairs_45 = reinterpret_cast<uint32_t*>(&_vld_45[_blk]);
                                    #pragma unroll
                                    for (int _pair = 0; _pair < 4; _pair++) {
                                        (&vec_load_5_5[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_45[_pair]) << 16);
                                        (&vec_load_5_5[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_45[_pair]) & 0xffff0000u);
                                    }
                                }
                            }
                            float vec_load_6_5[8];
                            {
                                const uint4* _vptr_46 = reinterpret_cast<const uint4*>(K + ((((row_end_5 > brow_5 + 96 + rgrp_2) ? brow_5 + 96 + rgrp_2 : row_end_5 - 1) * num_heads + head_0_5) * 128 + col_2) + 0);
                                uint4 _vld_46[1];
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
                                    _vld_46[_blk] = _vptr_46[_blk];
                                    uint32_t* _vpairs_46 = reinterpret_cast<uint32_t*>(&_vld_46[_blk]);
                                    #pragma unroll
                                    for (int _pair = 0; _pair < 4; _pair++) {
                                        (&vec_load_6_5[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_46[_pair]) << 16);
                                        (&vec_load_6_5[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_46[_pair]) & 0xffff0000u);
                                    }
                                }
                            }
                            float vec_load_7_5[8];
                            {
                                const uint4* _vptr_47 = reinterpret_cast<const uint4*>(K + ((((row_end_5 > brow_5 + 112 + rgrp_2) ? brow_5 + 112 + rgrp_2 : row_end_5 - 1) * num_heads + head_0_5) * 128 + col_2) + 0);
                                uint4 _vld_47[1];
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
                                    _vld_47[_blk] = _vptr_47[_blk];
                                    uint32_t* _vpairs_47 = reinterpret_cast<uint32_t*>(&_vld_47[_blk]);
                                    #pragma unroll
                                    for (int _pair = 0; _pair < 4; _pair++) {
                                        (&vec_load_7_5[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_47[_pair]) << 16);
                                        (&vec_load_7_5[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_47[_pair]) & 0xffff0000u);
                                    }
                                }
                            }
                            kf8_2[0] = vec_load_14[0];
                            kf8_2[1] = vec_load_14[1];
                            kf8_2[2] = vec_load_14[2];
                            kf8_2[3] = vec_load_14[3];
                            kf8_2[4] = vec_load_14[4];
                            kf8_2[5] = vec_load_14[5];
                            kf8_2[6] = vec_load_14[6];
                            kf8_2[7] = vec_load_14[7];
                            ksum_2[0] = ksum_2[0] + ((row_end_5 > brow_5 + rgrp_2) ? kf8_2[0] : 0.0f);
                            ksum_2[1] = ksum_2[1] + ((row_end_5 > brow_5 + rgrp_2) ? kf8_2[1] : 0.0f);
                            ksum_2[2] = ksum_2[2] + ((row_end_5 > brow_5 + rgrp_2) ? kf8_2[2] : 0.0f);
                            ksum_2[3] = ksum_2[3] + ((row_end_5 > brow_5 + rgrp_2) ? kf8_2[3] : 0.0f);
                            ksum_2[4] = ksum_2[4] + ((row_end_5 > brow_5 + rgrp_2) ? kf8_2[4] : 0.0f);
                            ksum_2[5] = ksum_2[5] + ((row_end_5 > brow_5 + rgrp_2) ? kf8_2[5] : 0.0f);
                            ksum_2[6] = ksum_2[6] + ((row_end_5 > brow_5 + rgrp_2) ? kf8_2[6] : 0.0f);
                            ksum_2[7] = ksum_2[7] + ((row_end_5 > brow_5 + rgrp_2) ? kf8_2[7] : 0.0f);
                            kf8_2[0] = vec_load_1_5[0];
                            kf8_2[1] = vec_load_1_5[1];
                            kf8_2[2] = vec_load_1_5[2];
                            kf8_2[3] = vec_load_1_5[3];
                            kf8_2[4] = vec_load_1_5[4];
                            kf8_2[5] = vec_load_1_5[5];
                            kf8_2[6] = vec_load_1_5[6];
                            kf8_2[7] = vec_load_1_5[7];
                            ksum_2[0] = ksum_2[0] + ((row_end_5 > brow_5 + 16 + rgrp_2) ? kf8_2[0] : 0.0f);
                            ksum_2[1] = ksum_2[1] + ((row_end_5 > brow_5 + 16 + rgrp_2) ? kf8_2[1] : 0.0f);
                            ksum_2[2] = ksum_2[2] + ((row_end_5 > brow_5 + 16 + rgrp_2) ? kf8_2[2] : 0.0f);
                            ksum_2[3] = ksum_2[3] + ((row_end_5 > brow_5 + 16 + rgrp_2) ? kf8_2[3] : 0.0f);
                            ksum_2[4] = ksum_2[4] + ((row_end_5 > brow_5 + 16 + rgrp_2) ? kf8_2[4] : 0.0f);
                            ksum_2[5] = ksum_2[5] + ((row_end_5 > brow_5 + 16 + rgrp_2) ? kf8_2[5] : 0.0f);
                            ksum_2[6] = ksum_2[6] + ((row_end_5 > brow_5 + 16 + rgrp_2) ? kf8_2[6] : 0.0f);
                            ksum_2[7] = ksum_2[7] + ((row_end_5 > brow_5 + 16 + rgrp_2) ? kf8_2[7] : 0.0f);
                            kf8_2[0] = vec_load_2_5[0];
                            kf8_2[1] = vec_load_2_5[1];
                            kf8_2[2] = vec_load_2_5[2];
                            kf8_2[3] = vec_load_2_5[3];
                            kf8_2[4] = vec_load_2_5[4];
                            kf8_2[5] = vec_load_2_5[5];
                            kf8_2[6] = vec_load_2_5[6];
                            kf8_2[7] = vec_load_2_5[7];
                            ksum_2[0] = ksum_2[0] + ((row_end_5 > brow_5 + 32 + rgrp_2) ? kf8_2[0] : 0.0f);
                            ksum_2[1] = ksum_2[1] + ((row_end_5 > brow_5 + 32 + rgrp_2) ? kf8_2[1] : 0.0f);
                            ksum_2[2] = ksum_2[2] + ((row_end_5 > brow_5 + 32 + rgrp_2) ? kf8_2[2] : 0.0f);
                            ksum_2[3] = ksum_2[3] + ((row_end_5 > brow_5 + 32 + rgrp_2) ? kf8_2[3] : 0.0f);
                            ksum_2[4] = ksum_2[4] + ((row_end_5 > brow_5 + 32 + rgrp_2) ? kf8_2[4] : 0.0f);
                            ksum_2[5] = ksum_2[5] + ((row_end_5 > brow_5 + 32 + rgrp_2) ? kf8_2[5] : 0.0f);
                            ksum_2[6] = ksum_2[6] + ((row_end_5 > brow_5 + 32 + rgrp_2) ? kf8_2[6] : 0.0f);
                            ksum_2[7] = ksum_2[7] + ((row_end_5 > brow_5 + 32 + rgrp_2) ? kf8_2[7] : 0.0f);
                            kf8_2[0] = vec_load_3_5[0];
                            kf8_2[1] = vec_load_3_5[1];
                            kf8_2[2] = vec_load_3_5[2];
                            kf8_2[3] = vec_load_3_5[3];
                            kf8_2[4] = vec_load_3_5[4];
                            kf8_2[5] = vec_load_3_5[5];
                            kf8_2[6] = vec_load_3_5[6];
                            kf8_2[7] = vec_load_3_5[7];
                            ksum_2[0] = ksum_2[0] + ((row_end_5 > brow_5 + 48 + rgrp_2) ? kf8_2[0] : 0.0f);
                            ksum_2[1] = ksum_2[1] + ((row_end_5 > brow_5 + 48 + rgrp_2) ? kf8_2[1] : 0.0f);
                            ksum_2[2] = ksum_2[2] + ((row_end_5 > brow_5 + 48 + rgrp_2) ? kf8_2[2] : 0.0f);
                            ksum_2[3] = ksum_2[3] + ((row_end_5 > brow_5 + 48 + rgrp_2) ? kf8_2[3] : 0.0f);
                            ksum_2[4] = ksum_2[4] + ((row_end_5 > brow_5 + 48 + rgrp_2) ? kf8_2[4] : 0.0f);
                            ksum_2[5] = ksum_2[5] + ((row_end_5 > brow_5 + 48 + rgrp_2) ? kf8_2[5] : 0.0f);
                            ksum_2[6] = ksum_2[6] + ((row_end_5 > brow_5 + 48 + rgrp_2) ? kf8_2[6] : 0.0f);
                            ksum_2[7] = ksum_2[7] + ((row_end_5 > brow_5 + 48 + rgrp_2) ? kf8_2[7] : 0.0f);
                            kf8_2[0] = vec_load_4_5[0];
                            kf8_2[1] = vec_load_4_5[1];
                            kf8_2[2] = vec_load_4_5[2];
                            kf8_2[3] = vec_load_4_5[3];
                            kf8_2[4] = vec_load_4_5[4];
                            kf8_2[5] = vec_load_4_5[5];
                            kf8_2[6] = vec_load_4_5[6];
                            kf8_2[7] = vec_load_4_5[7];
                            ksum_2[0] = ksum_2[0] + ((row_end_5 > brow_5 + 64 + rgrp_2) ? kf8_2[0] : 0.0f);
                            ksum_2[1] = ksum_2[1] + ((row_end_5 > brow_5 + 64 + rgrp_2) ? kf8_2[1] : 0.0f);
                            ksum_2[2] = ksum_2[2] + ((row_end_5 > brow_5 + 64 + rgrp_2) ? kf8_2[2] : 0.0f);
                            ksum_2[3] = ksum_2[3] + ((row_end_5 > brow_5 + 64 + rgrp_2) ? kf8_2[3] : 0.0f);
                            ksum_2[4] = ksum_2[4] + ((row_end_5 > brow_5 + 64 + rgrp_2) ? kf8_2[4] : 0.0f);
                            ksum_2[5] = ksum_2[5] + ((row_end_5 > brow_5 + 64 + rgrp_2) ? kf8_2[5] : 0.0f);
                            ksum_2[6] = ksum_2[6] + ((row_end_5 > brow_5 + 64 + rgrp_2) ? kf8_2[6] : 0.0f);
                            ksum_2[7] = ksum_2[7] + ((row_end_5 > brow_5 + 64 + rgrp_2) ? kf8_2[7] : 0.0f);
                            kf8_2[0] = vec_load_5_5[0];
                            kf8_2[1] = vec_load_5_5[1];
                            kf8_2[2] = vec_load_5_5[2];
                            kf8_2[3] = vec_load_5_5[3];
                            kf8_2[4] = vec_load_5_5[4];
                            kf8_2[5] = vec_load_5_5[5];
                            kf8_2[6] = vec_load_5_5[6];
                            kf8_2[7] = vec_load_5_5[7];
                            ksum_2[0] = ksum_2[0] + ((row_end_5 > brow_5 + 80 + rgrp_2) ? kf8_2[0] : 0.0f);
                            ksum_2[1] = ksum_2[1] + ((row_end_5 > brow_5 + 80 + rgrp_2) ? kf8_2[1] : 0.0f);
                            ksum_2[2] = ksum_2[2] + ((row_end_5 > brow_5 + 80 + rgrp_2) ? kf8_2[2] : 0.0f);
                            ksum_2[3] = ksum_2[3] + ((row_end_5 > brow_5 + 80 + rgrp_2) ? kf8_2[3] : 0.0f);
                            ksum_2[4] = ksum_2[4] + ((row_end_5 > brow_5 + 80 + rgrp_2) ? kf8_2[4] : 0.0f);
                            ksum_2[5] = ksum_2[5] + ((row_end_5 > brow_5 + 80 + rgrp_2) ? kf8_2[5] : 0.0f);
                            ksum_2[6] = ksum_2[6] + ((row_end_5 > brow_5 + 80 + rgrp_2) ? kf8_2[6] : 0.0f);
                            ksum_2[7] = ksum_2[7] + ((row_end_5 > brow_5 + 80 + rgrp_2) ? kf8_2[7] : 0.0f);
                            kf8_2[0] = vec_load_6_5[0];
                            kf8_2[1] = vec_load_6_5[1];
                            kf8_2[2] = vec_load_6_5[2];
                            kf8_2[3] = vec_load_6_5[3];
                            kf8_2[4] = vec_load_6_5[4];
                            kf8_2[5] = vec_load_6_5[5];
                            kf8_2[6] = vec_load_6_5[6];
                            kf8_2[7] = vec_load_6_5[7];
                            ksum_2[0] = ksum_2[0] + ((row_end_5 > brow_5 + 96 + rgrp_2) ? kf8_2[0] : 0.0f);
                            ksum_2[1] = ksum_2[1] + ((row_end_5 > brow_5 + 96 + rgrp_2) ? kf8_2[1] : 0.0f);
                            ksum_2[2] = ksum_2[2] + ((row_end_5 > brow_5 + 96 + rgrp_2) ? kf8_2[2] : 0.0f);
                            ksum_2[3] = ksum_2[3] + ((row_end_5 > brow_5 + 96 + rgrp_2) ? kf8_2[3] : 0.0f);
                            ksum_2[4] = ksum_2[4] + ((row_end_5 > brow_5 + 96 + rgrp_2) ? kf8_2[4] : 0.0f);
                            ksum_2[5] = ksum_2[5] + ((row_end_5 > brow_5 + 96 + rgrp_2) ? kf8_2[5] : 0.0f);
                            ksum_2[6] = ksum_2[6] + ((row_end_5 > brow_5 + 96 + rgrp_2) ? kf8_2[6] : 0.0f);
                            ksum_2[7] = ksum_2[7] + ((row_end_5 > brow_5 + 96 + rgrp_2) ? kf8_2[7] : 0.0f);
                            kf8_2[0] = vec_load_7_5[0];
                            kf8_2[1] = vec_load_7_5[1];
                            kf8_2[2] = vec_load_7_5[2];
                            kf8_2[3] = vec_load_7_5[3];
                            kf8_2[4] = vec_load_7_5[4];
                            kf8_2[5] = vec_load_7_5[5];
                            kf8_2[6] = vec_load_7_5[6];
                            kf8_2[7] = vec_load_7_5[7];
                            ksum_2[0] = ksum_2[0] + ((row_end_5 > brow_5 + 112 + rgrp_2) ? kf8_2[0] : 0.0f);
                            ksum_2[1] = ksum_2[1] + ((row_end_5 > brow_5 + 112 + rgrp_2) ? kf8_2[1] : 0.0f);
                            ksum_2[2] = ksum_2[2] + ((row_end_5 > brow_5 + 112 + rgrp_2) ? kf8_2[2] : 0.0f);
                            ksum_2[3] = ksum_2[3] + ((row_end_5 > brow_5 + 112 + rgrp_2) ? kf8_2[3] : 0.0f);
                            ksum_2[4] = ksum_2[4] + ((row_end_5 > brow_5 + 112 + rgrp_2) ? kf8_2[4] : 0.0f);
                            ksum_2[5] = ksum_2[5] + ((row_end_5 > brow_5 + 112 + rgrp_2) ? kf8_2[5] : 0.0f);
                            ksum_2[6] = ksum_2[6] + ((row_end_5 > brow_5 + 112 + rgrp_2) ? kf8_2[6] : 0.0f);
                            ksum_2[7] = ksum_2[7] + ((row_end_5 > brow_5 + 112 + rgrp_2) ? kf8_2[7] : 0.0f);
                        }
                        for (int e_16 = 0; e_16 < 8; e_16++) {
                            red_sum[4096 + rgrp_2 * 128 + col_2 + e_16] = ksum_2[e_16];
                        }
                        asm volatile("bar.sync 3, 256;" ::: "memory");
                        int direct_2 = ((t1_2 - t0_5 == 1) ? 1 : 0);
                        if (direct_2 == 0) {
                            if (wtid_2 < 128) {
                                float total_2 = 0.0f;
                                for (int g_7 = 0; g_7 < 16; g_7++) {
                                    total_2 = total_2 + red_sum[4096 + g_7 * 128 + wtid_2];
                                }
                                *(reinterpret_cast<float*>(partials + (((ri_2 - num_tiles - num_tblk_2) * num_heads + head_2) * 128 + wtid_2)) + (0)) = total_2;
                            }
                            asm volatile("bar.sync 3, 256;" ::: "memory");
                            if (wtid_2 == 0) {
                                unsigned int old_4;
                                asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], %2;"
                                    : "=r"(old_4) : "l"(&counters[seg_7 * num_heads + head_0_5]), "r"(static_cast<uint32_t>(1)) : "memory");
                                int last_2 = (((int)old_4 == t1_2 - t0_5 - 1) ? 1 : 0);
                                flag[8] = last_2;
                                if (last_2 == 1) {
                                    *(reinterpret_cast<unsigned int*>(counters + (seg_7 * num_heads + head_0_5)) + (0)) = (unsigned int)0;
                                }
                            }
                            asm volatile("bar.sync 3, 256;" ::: "memory");
                        }
                        int is_last_2 = ((direct_2 == 0) ? flag[8] : 1);
                        if (is_last_2 == 1) {
                            if (wtid_2 < 128) {
                                float acc_a_2[1];
                                float fold_v_2[4];
                                acc_a_2[0] = 0.0f;
                                if (direct_2 == 1) {
                                    for (int g_8 = 0; g_8 < 16; g_8++) {
                                        acc_a_2[0] = acc_a_2[0] + red_sum[4096 + g_8 * 128 + wtid_2];
                                    }
                                } else {
                                    #pragma unroll 1
                                    for (int tt_2 = t0_5; tt_2 < t1_2; tt_2 += 4) {
                                        for (int j_4 = 0; j_4 < 4; j_4++) {
                                            int ttj_2 = ((t1_2 > tt_2 + j_4) ? tt_2 + j_4 : t1_2 - 1);
                                            fold_v_2[j_4] = partials[(ttj_2 * num_heads + head_0_5) * 128 + wtid_2];
                                        }
                                        for (int j_5 = 0; j_5 < 4; j_5++) {
                                            float fv_2 = ((t1_2 > tt_2 + j_5) ? fold_v_2[j_5] : 0.0f);
                                            acc_a_2[0] = acc_a_2[0] + fv_2;
                                        }
                                    }
                                }
                                if (length_2 > 0) {
                                    float lenf_2 = (float)length_2;
                                    float _fdiv_48_2 = __fdiv_rn(acc_a_2[0], lenf_2);
                                    *(reinterpret_cast<float*>(mean_k + ((seg_7 * num_heads + head_0_5) * 128 + wtid_2)) + (0)) = _fdiv_48_2;
                                }
                            }
                        }
                        asm volatile("bar.sync 3, 256;" ::: "memory");
                        if (wtid_2 == 0) {
                            asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(hctr) + (num_heads + head_2))), "r"(static_cast<unsigned int>(1)) : "memory");
                        }
                    }
                } else {
                    if (wtid_2 == 0) {
                        {
                        unsigned int _acquire_observed;
                        do {
                        asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(_acquire_observed) : "l"((reinterpret_cast<unsigned int*>(hctr) + (num_heads + head_2))) : "memory");
                        } while (static_cast<unsigned int>(_acquire_observed - static_cast<unsigned int>(num_tiles)) >= static_cast<unsigned int>(1));
                        }
                    }
                    asm volatile("bar.sync 3, 256;" ::: "memory");
                    int tok_5 = ri_2 * 32 + trow_2;
                    int valid_5 = ((tok_5 < total_tokens) ? 1 : 0);
                    int tok_c_5 = ((tok_5 < total_tokens) ? tok_5 : total_tokens - 1);
                    float vec_load_15[16];
                    {
                        const uint4* _vptr_49 = reinterpret_cast<const uint4*>(K + ((tok_c_5 * num_heads + head_2) * 128 + col16_2) + 0);
                        uint4 _vld_49[2];
                        #pragma unroll
                        for (int _blk = 0; _blk < 2; _blk++) {
                            _vld_49[_blk] = _vptr_49[_blk];
                            uint32_t* _vpairs_49 = reinterpret_cast<uint32_t*>(&_vld_49[_blk]);
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
                                (&vec_load_15[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_49[_pair]) << 16);
                                (&vec_load_15[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_49[_pair]) & 0xffff0000u);
                            }
                        }
                    }
                    int seg_8 = tok_seg[tok_c_5];
                    int mbase_2 = (seg_8 * num_heads + head_2) * 128 + col16_2;
                    float kc_2[16];
                    for (int quad_5 = 0; quad_5 < 4; quad_5++) {
                        float _vec_load_30_2[4];
                        {
                            float4 _v4 = *reinterpret_cast<const float4*>(mean_k + (mbase_2 + quad_5 * 4) + 0);
                            _vec_load_30_2[0 + 0] = _v4.x;
                            _vec_load_30_2[0 + 1] = _v4.y;
                            _vec_load_30_2[0 + 2] = _v4.z;
                            _vec_load_30_2[0 + 3] = _v4.w;
                        }
                        for (int e_17 = 0; e_17 < 4; e_17++) {
                            float kf_2 = vec_load_15[quad_5 * 4 + e_17];
                            kc_2[quad_5 * 4 + e_17] = ((valid_5 == 1) ? kf_2 - _vec_load_30_2[e_17] : 0.0f);
                        }
                    }
                    unsigned int kw_2[2];
                    float ksf_a_2[1];
                    float max_1_5 = fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fabsf(kc_2[0]), fabsf(kc_2[1])), fabsf(kc_2[2])), fabsf(kc_2[3])), fabsf(kc_2[4])), fabsf(kc_2[5])), fabsf(kc_2[6])), fabsf(kc_2[7])), fabsf(kc_2[8])), fabsf(kc_2[9])), fabsf(kc_2[10])), fabsf(kc_2[11])), fabsf(kc_2[12])), fabsf(kc_2[13])), fabsf(kc_2[14])), fabsf(kc_2[15]));
                    uint16_t _cvt_satfinite_e4m3x2_f32_51_2;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_51_2) : "f"(0.0f), "f"(max_1_5 * 0.16666667f));
                    uint16_t _e4m3x2_decode_52 = (uint16_t)((unsigned int)_cvt_satfinite_e4m3x2_f32_51_2 & 0xFFu);
                    uint32_t _f16x2_decode_52;
                    float fp8_decode_5;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_decode_52) : "h"(_e4m3x2_decode_52));
                    uint16_t _f16_decode_52 = (uint16_t)_f16x2_decode_52;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(fp8_decode_5) : "h"(_f16_decode_52));
                    float fdiv_rn_5 = __fdiv_rn(1.0f, fp8_decode_5);
                    kc_2[0] = kc_2[0] * ((fp8_decode_5 != 0.0f) ? fdiv_rn_5 : 0.0f);
                    kc_2[1] = kc_2[1] * ((fp8_decode_5 != 0.0f) ? fdiv_rn_5 : 0.0f);
                    kc_2[2] = kc_2[2] * ((fp8_decode_5 != 0.0f) ? fdiv_rn_5 : 0.0f);
                    kc_2[3] = kc_2[3] * ((fp8_decode_5 != 0.0f) ? fdiv_rn_5 : 0.0f);
                    kc_2[4] = kc_2[4] * ((fp8_decode_5 != 0.0f) ? fdiv_rn_5 : 0.0f);
                    kc_2[5] = kc_2[5] * ((fp8_decode_5 != 0.0f) ? fdiv_rn_5 : 0.0f);
                    kc_2[6] = kc_2[6] * ((fp8_decode_5 != 0.0f) ? fdiv_rn_5 : 0.0f);
                    kc_2[7] = kc_2[7] * ((fp8_decode_5 != 0.0f) ? fdiv_rn_5 : 0.0f);
                    kc_2[8] = kc_2[8] * ((fp8_decode_5 != 0.0f) ? fdiv_rn_5 : 0.0f);
                    kc_2[9] = kc_2[9] * ((fp8_decode_5 != 0.0f) ? fdiv_rn_5 : 0.0f);
                    kc_2[10] = kc_2[10] * ((fp8_decode_5 != 0.0f) ? fdiv_rn_5 : 0.0f);
                    kc_2[11] = kc_2[11] * ((fp8_decode_5 != 0.0f) ? fdiv_rn_5 : 0.0f);
                    kc_2[12] = kc_2[12] * ((fp8_decode_5 != 0.0f) ? fdiv_rn_5 : 0.0f);
                    kc_2[13] = kc_2[13] * ((fp8_decode_5 != 0.0f) ? fdiv_rn_5 : 0.0f);
                    kc_2[14] = kc_2[14] * ((fp8_decode_5 != 0.0f) ? fdiv_rn_5 : 0.0f);
                    kc_2[15] = kc_2[15] * ((fp8_decode_5 != 0.0f) ? fdiv_rn_5 : 0.0f);
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(kw_2[0]) : "f"(kc_2[0]), "f"(kc_2[1]), "f"(kc_2[2]), "f"(kc_2[3]), "f"(kc_2[4]), "f"(kc_2[5]), "f"(kc_2[6]), "f"(kc_2[7]));
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(kw_2[1]) : "f"(kc_2[8]), "f"(kc_2[9]), "f"(kc_2[10]), "f"(kc_2[11]), "f"(kc_2[12]), "f"(kc_2[13]), "f"(kc_2[14]), "f"(kc_2[15]));
                    ksf_a_2[0] = max_1_5 * 0.16666667f;
                    unsigned int kstore_2[4];
                    unsigned int _shfl_xor_sync_53_2 = __shfl_xor_sync(0xFFFFFFFF, kw_2[0], 1);
                    kstore_2[0] = _shfl_xor_sync_53_2;
                    unsigned int _shfl_xor_sync_54_2 = __shfl_xor_sync(0xFFFFFFFF, kw_2[1], 1);
                    kstore_2[1] = _shfl_xor_sync_54_2;
                    kstore_2[2] = kw_2[0];
                    kstore_2[3] = kw_2[1];
                    uint16_t _cvt_satfinite_e4m3x2_f32_55_2;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_55_2) : "f"(0.0f), "f"(ksf_a_2[0]));
                    *reinterpret_cast<unsigned char*>(k_sf + (((head_2 * num_kblocks + (tok_5 >> 7)) * 128 + (tok_5 & 127)) * 8 + sub_2)) = (unsigned char)((unsigned int)(unsigned char)(_cvt_satfinite_e4m3x2_f32_55_2 & 127));
                    if (valid_5 == 1) {
                        if ((sub_2 & 1) == 1) {
                            reinterpret_cast<int4*>(k4 + (((tok_5 * num_heads + head_2) * 64 + (sub_2 - 1) * 8) / 4))[0] = reinterpret_cast<int4*>(kstore_2)[0];
                        }
                    }
                }
                if (wtid_2 == 0) {
                    item_cell[8 + (it_2 + 1 & 1)] = (int)nxt_r_2[0];
                }
                asm volatile("bar.sync 3, 256;" ::: "memory");
                if (wtid_2 == 0) {
                    if (is_d_2 == 1) {
                        unsigned int dold_2;
                        asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], %2;"
                            : "=r"(dold_2) : "l"(&hctr[2 * num_heads + head_2]), "r"(static_cast<uint32_t>(1)) : "memory");
                        if ((int)dold_2 == num_tblk_2 - 1) {
                            *(reinterpret_cast<unsigned int*>(hctr + (num_heads + head_2)) + (0)) = (unsigned int)0;
                            *(reinterpret_cast<unsigned int*>(hctr + (2 * num_heads + head_2)) + (0)) = (unsigned int)0;
                        }
                    }
                    unsigned int old_5;
                    asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], %2;"
                        : "=r"(old_5) : "l"(&hctr[head_2]), "r"(static_cast<uint32_t>(1)) : "memory");
                    if ((int)old_5 == per_2 - 1) {
                        *(reinterpret_cast<unsigned int*>(hctr + head_2) + (0)) = (unsigned int)0;
                        asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(hctr) + (3 * num_heads + head_2))), "r"(static_cast<unsigned int>(1)) : "memory");
                    }
                }
            }
            asm volatile("barrier.sync 4, 768;" ::: "memory");
        }
    }
}

} // extern "C"
