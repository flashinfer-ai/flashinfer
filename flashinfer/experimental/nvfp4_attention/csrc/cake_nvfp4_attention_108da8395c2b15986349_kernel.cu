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
#define SMEM_RED_QMAX_OFF 16384
#define SMEM_RED_QMAX_STAGE_BYTES 8192
#define SMEM_RED_QMAX_STRIDE 8192
#define SMEM_RED_QMIN_OFF 24576
#define SMEM_RED_QMIN_STAGE_BYTES 8192
#define SMEM_RED_QMIN_STRIDE 8192
#define SMEM_RED_KMAX_OFF 32768
#define SMEM_RED_KMAX_STAGE_BYTES 8192
#define SMEM_RED_KMAX_STRIDE 8192
#define SMEM_RED_KMIN_OFF 40960
#define SMEM_RED_KMIN_STAGE_BYTES 8192
#define SMEM_RED_KMIN_STRIDE 8192
#define SMEM_MEAN_S_OFF 49152
#define SMEM_MEAN_S_STAGE_BYTES 512
#define SMEM_MEAN_S_STRIDE 512
#define SMEM_RED_SMALL_OFF 49664
#define SMEM_RED_SMALL_STAGE_BYTES 32
#define SMEM_RED_SMALL_STRIDE 32
#define SMEM_FLAG_OFF 49696
#define SMEM_FLAG_STAGE_BYTES 16
#define SMEM_FLAG_STRIDE 16
#define SMEM_TOTAL 49792
#define THREADS 256
#define LAUNCH_MIN_BLOCKS 2

#include <math_constants.h>

extern "C" {

__global__ __launch_bounds__(256, LAUNCH_MIN_BLOCKS) void
kernel_cake_nvfp4_attention_108da8395c2b15986349(__nv_bfloat16* __restrict__ Q, __nv_bfloat16* __restrict__ K, float* __restrict__ partials, unsigned int* __restrict__ counters, float* __restrict__ mean_k, float* __restrict__ q_mean, float* __restrict__ k_minmax, float* __restrict__ q_amax, float* __restrict__ qm_amax, float* __restrict__ k_amax, int seq_len, int num_heads, int num_qblocks, int num_kblocks, int num_items, long long q_stride_b, long long q_stride_h, long long q_stride_s, long long k_stride_b, long long k_stride_h, long long k_stride_s)
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
    float* red_qmax = reinterpret_cast<float*>(smem_raw + 16384);
    const int red_qmax_addr = smem + 16384;
    float* red_qmin = reinterpret_cast<float*>(smem_raw + 24576);
    const int red_qmin_addr = smem + 24576;
    float* red_kmax = reinterpret_cast<float*>(smem_raw + 32768);
    const int red_kmax_addr = smem + 32768;
    float* red_kmin = reinterpret_cast<float*>(smem_raw + 40960);
    const int red_kmin_addr = smem + 40960;
    float* mean_s = reinterpret_cast<float*>(smem_raw + 49152);
    const int mean_s_addr = smem + 49152;
    float* red_small = reinterpret_cast<float*>(smem_raw + 49664);
    const int red_small_addr = smem + 49664;
    int* flag = reinterpret_cast<int*>(smem_raw + 49696);
    const int flag_addr = smem + 49696;

    // === Task calls (declaration order) ===
    int col = (tid & 15) * 8;
    int rgrp = tid >> 4;
    unsigned long long qsb = (unsigned long long)q_stride_b;
    unsigned long long qsh = (unsigned long long)q_stride_h;
    unsigned long long qss = (unsigned long long)q_stride_s;
    unsigned long long ksb = (unsigned long long)k_stride_b;
    unsigned long long ksh = (unsigned long long)k_stride_h;
    unsigned long long kss = (unsigned long long)k_stride_s;
    asm volatile("griddepcontrol.wait;" ::: "memory");
    #pragma unroll 1
    for (int work = bid; work < num_items; work += num_bids) {
        int tq = work / num_heads;
        int head = work % num_heads;
        int batch = tq / num_qblocks;
        int tile = tq % num_qblocks;
        int row_begin = tile * 128;
        int row_end = ((row_begin + 128 < seq_len) ? row_begin + 128 : seq_len);
        unsigned long long b64 = (unsigned long long)batch;
        unsigned long long h64 = (unsigned long long)head;
        unsigned long long qbase = b64 * qsb + h64 * qsh;
        unsigned long long kbase = b64 * ksb + h64 * ksh;
        float ksum[8];
        float qsum[8];
        float qmax[8];
        float qmin[8];
        float kmax[8];
        float kmin[8];
        float qf8[8];
        float kf8[8];
        for (int e = 0; e < 8; e++) {
            ksum[e] = 0.0f;
            qsum[e] = 0.0f;
            qmax[e] = -CUDART_INF_F;
            qmin[e] = CUDART_INF_F;
            kmax[e] = -CUDART_INF_F;
            kmin[e] = CUDART_INF_F;
        }
        float _vec_load_0[8];
        {
            const uint4* _vptr_0 = reinterpret_cast<const uint4*>(Q + (qbase + (unsigned long long)(((row_end > row_begin + rgrp) ? row_begin + rgrp : row_end - 1)) * qss + (unsigned long long)col) + 0);
            uint4 _vld_0[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vld_0[_blk] = _vptr_0[_blk];
                uint32_t* _vpairs_0 = reinterpret_cast<uint32_t*>(&_vld_0[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_vec_load_0[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_0[_pair]) << 16);
                    (&_vec_load_0[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_0[_pair]) & 0xffff0000u);
                }
            }
        }
        float _vec_load_1[8];
        {
            const uint4* _vptr_1 = reinterpret_cast<const uint4*>(K + (kbase + (unsigned long long)(((row_end > row_begin + rgrp) ? row_begin + rgrp : row_end - 1)) * kss + (unsigned long long)col) + 0);
            uint4 _vld_1[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vld_1[_blk] = _vptr_1[_blk];
                uint32_t* _vpairs_1 = reinterpret_cast<uint32_t*>(&_vld_1[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_vec_load_1[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_1[_pair]) << 16);
                    (&_vec_load_1[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_1[_pair]) & 0xffff0000u);
                }
            }
        }
        float _vec_load_2[8];
        {
            const uint4* _vptr_2 = reinterpret_cast<const uint4*>(Q + (qbase + (unsigned long long)(((row_end > row_begin + 16 + rgrp) ? row_begin + 16 + rgrp : row_end - 1)) * qss + (unsigned long long)col) + 0);
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
            const uint4* _vptr_3 = reinterpret_cast<const uint4*>(K + (kbase + (unsigned long long)(((row_end > row_begin + 16 + rgrp) ? row_begin + 16 + rgrp : row_end - 1)) * kss + (unsigned long long)col) + 0);
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
            const uint4* _vptr_4 = reinterpret_cast<const uint4*>(Q + (qbase + (unsigned long long)(((row_end > row_begin + 32 + rgrp) ? row_begin + 32 + rgrp : row_end - 1)) * qss + (unsigned long long)col) + 0);
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
            const uint4* _vptr_5 = reinterpret_cast<const uint4*>(K + (kbase + (unsigned long long)(((row_end > row_begin + 32 + rgrp) ? row_begin + 32 + rgrp : row_end - 1)) * kss + (unsigned long long)col) + 0);
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
            const uint4* _vptr_6 = reinterpret_cast<const uint4*>(Q + (qbase + (unsigned long long)(((row_end > row_begin + 48 + rgrp) ? row_begin + 48 + rgrp : row_end - 1)) * qss + (unsigned long long)col) + 0);
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
            const uint4* _vptr_7 = reinterpret_cast<const uint4*>(K + (kbase + (unsigned long long)(((row_end > row_begin + 48 + rgrp) ? row_begin + 48 + rgrp : row_end - 1)) * kss + (unsigned long long)col) + 0);
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
            const uint4* _vptr_8 = reinterpret_cast<const uint4*>(Q + (qbase + (unsigned long long)(((row_end > row_begin + 64 + rgrp) ? row_begin + 64 + rgrp : row_end - 1)) * qss + (unsigned long long)col) + 0);
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
            const uint4* _vptr_9 = reinterpret_cast<const uint4*>(K + (kbase + (unsigned long long)(((row_end > row_begin + 64 + rgrp) ? row_begin + 64 + rgrp : row_end - 1)) * kss + (unsigned long long)col) + 0);
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
            const uint4* _vptr_10 = reinterpret_cast<const uint4*>(Q + (qbase + (unsigned long long)(((row_end > row_begin + 80 + rgrp) ? row_begin + 80 + rgrp : row_end - 1)) * qss + (unsigned long long)col) + 0);
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
            const uint4* _vptr_11 = reinterpret_cast<const uint4*>(K + (kbase + (unsigned long long)(((row_end > row_begin + 80 + rgrp) ? row_begin + 80 + rgrp : row_end - 1)) * kss + (unsigned long long)col) + 0);
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
            const uint4* _vptr_12 = reinterpret_cast<const uint4*>(Q + (qbase + (unsigned long long)(((row_end > row_begin + 96 + rgrp) ? row_begin + 96 + rgrp : row_end - 1)) * qss + (unsigned long long)col) + 0);
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
            const uint4* _vptr_13 = reinterpret_cast<const uint4*>(K + (kbase + (unsigned long long)(((row_end > row_begin + 96 + rgrp) ? row_begin + 96 + rgrp : row_end - 1)) * kss + (unsigned long long)col) + 0);
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
            const uint4* _vptr_14 = reinterpret_cast<const uint4*>(Q + (qbase + (unsigned long long)(((row_end > row_begin + 112 + rgrp) ? row_begin + 112 + rgrp : row_end - 1)) * qss + (unsigned long long)col) + 0);
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
            const uint4* _vptr_15 = reinterpret_cast<const uint4*>(K + (kbase + (unsigned long long)(((row_end > row_begin + 112 + rgrp) ? row_begin + 112 + rgrp : row_end - 1)) * kss + (unsigned long long)col) + 0);
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
        qf8[0] = _vec_load_0[0];
        kf8[0] = _vec_load_1[0];
        qf8[1] = _vec_load_0[1];
        kf8[1] = _vec_load_1[1];
        qf8[2] = _vec_load_0[2];
        kf8[2] = _vec_load_1[2];
        qf8[3] = _vec_load_0[3];
        kf8[3] = _vec_load_1[3];
        qf8[4] = _vec_load_0[4];
        kf8[4] = _vec_load_1[4];
        qf8[5] = _vec_load_0[5];
        kf8[5] = _vec_load_1[5];
        qf8[6] = _vec_load_0[6];
        kf8[6] = _vec_load_1[6];
        qf8[7] = _vec_load_0[7];
        kf8[7] = _vec_load_1[7];
        ksum[0] = ksum[0] + ((row_end > row_begin + rgrp) ? kf8[0] : 0.0f);
        qsum[0] = qsum[0] + ((row_end > row_begin + rgrp) ? qf8[0] : 0.0f);
        ksum[1] = ksum[1] + ((row_end > row_begin + rgrp) ? kf8[1] : 0.0f);
        qsum[1] = qsum[1] + ((row_end > row_begin + rgrp) ? qf8[1] : 0.0f);
        ksum[2] = ksum[2] + ((row_end > row_begin + rgrp) ? kf8[2] : 0.0f);
        qsum[2] = qsum[2] + ((row_end > row_begin + rgrp) ? qf8[2] : 0.0f);
        ksum[3] = ksum[3] + ((row_end > row_begin + rgrp) ? kf8[3] : 0.0f);
        qsum[3] = qsum[3] + ((row_end > row_begin + rgrp) ? qf8[3] : 0.0f);
        ksum[4] = ksum[4] + ((row_end > row_begin + rgrp) ? kf8[4] : 0.0f);
        qsum[4] = qsum[4] + ((row_end > row_begin + rgrp) ? qf8[4] : 0.0f);
        ksum[5] = ksum[5] + ((row_end > row_begin + rgrp) ? kf8[5] : 0.0f);
        qsum[5] = qsum[5] + ((row_end > row_begin + rgrp) ? qf8[5] : 0.0f);
        ksum[6] = ksum[6] + ((row_end > row_begin + rgrp) ? kf8[6] : 0.0f);
        qsum[6] = qsum[6] + ((row_end > row_begin + rgrp) ? qf8[6] : 0.0f);
        ksum[7] = ksum[7] + ((row_end > row_begin + rgrp) ? kf8[7] : 0.0f);
        qsum[7] = qsum[7] + ((row_end > row_begin + rgrp) ? qf8[7] : 0.0f);
        qmax[0] = fmaxf(qmax[0], ((row_end > row_begin + rgrp) ? qf8[0] : -CUDART_INF_F));
        qmin[0] = fminf(qmin[0], ((row_end > row_begin + rgrp) ? qf8[0] : CUDART_INF_F));
        kmax[0] = fmaxf(kmax[0], ((row_end > row_begin + rgrp) ? kf8[0] : -CUDART_INF_F));
        kmin[0] = fminf(kmin[0], ((row_end > row_begin + rgrp) ? kf8[0] : CUDART_INF_F));
        qmax[1] = fmaxf(qmax[1], ((row_end > row_begin + rgrp) ? qf8[1] : -CUDART_INF_F));
        qmin[1] = fminf(qmin[1], ((row_end > row_begin + rgrp) ? qf8[1] : CUDART_INF_F));
        kmax[1] = fmaxf(kmax[1], ((row_end > row_begin + rgrp) ? kf8[1] : -CUDART_INF_F));
        kmin[1] = fminf(kmin[1], ((row_end > row_begin + rgrp) ? kf8[1] : CUDART_INF_F));
        qmax[2] = fmaxf(qmax[2], ((row_end > row_begin + rgrp) ? qf8[2] : -CUDART_INF_F));
        qmin[2] = fminf(qmin[2], ((row_end > row_begin + rgrp) ? qf8[2] : CUDART_INF_F));
        kmax[2] = fmaxf(kmax[2], ((row_end > row_begin + rgrp) ? kf8[2] : -CUDART_INF_F));
        kmin[2] = fminf(kmin[2], ((row_end > row_begin + rgrp) ? kf8[2] : CUDART_INF_F));
        qmax[3] = fmaxf(qmax[3], ((row_end > row_begin + rgrp) ? qf8[3] : -CUDART_INF_F));
        qmin[3] = fminf(qmin[3], ((row_end > row_begin + rgrp) ? qf8[3] : CUDART_INF_F));
        kmax[3] = fmaxf(kmax[3], ((row_end > row_begin + rgrp) ? kf8[3] : -CUDART_INF_F));
        kmin[3] = fminf(kmin[3], ((row_end > row_begin + rgrp) ? kf8[3] : CUDART_INF_F));
        qmax[4] = fmaxf(qmax[4], ((row_end > row_begin + rgrp) ? qf8[4] : -CUDART_INF_F));
        qmin[4] = fminf(qmin[4], ((row_end > row_begin + rgrp) ? qf8[4] : CUDART_INF_F));
        kmax[4] = fmaxf(kmax[4], ((row_end > row_begin + rgrp) ? kf8[4] : -CUDART_INF_F));
        kmin[4] = fminf(kmin[4], ((row_end > row_begin + rgrp) ? kf8[4] : CUDART_INF_F));
        qmax[5] = fmaxf(qmax[5], ((row_end > row_begin + rgrp) ? qf8[5] : -CUDART_INF_F));
        qmin[5] = fminf(qmin[5], ((row_end > row_begin + rgrp) ? qf8[5] : CUDART_INF_F));
        kmax[5] = fmaxf(kmax[5], ((row_end > row_begin + rgrp) ? kf8[5] : -CUDART_INF_F));
        kmin[5] = fminf(kmin[5], ((row_end > row_begin + rgrp) ? kf8[5] : CUDART_INF_F));
        qmax[6] = fmaxf(qmax[6], ((row_end > row_begin + rgrp) ? qf8[6] : -CUDART_INF_F));
        qmin[6] = fminf(qmin[6], ((row_end > row_begin + rgrp) ? qf8[6] : CUDART_INF_F));
        kmax[6] = fmaxf(kmax[6], ((row_end > row_begin + rgrp) ? kf8[6] : -CUDART_INF_F));
        kmin[6] = fminf(kmin[6], ((row_end > row_begin + rgrp) ? kf8[6] : CUDART_INF_F));
        qmax[7] = fmaxf(qmax[7], ((row_end > row_begin + rgrp) ? qf8[7] : -CUDART_INF_F));
        qmin[7] = fminf(qmin[7], ((row_end > row_begin + rgrp) ? qf8[7] : CUDART_INF_F));
        kmax[7] = fmaxf(kmax[7], ((row_end > row_begin + rgrp) ? kf8[7] : -CUDART_INF_F));
        kmin[7] = fminf(kmin[7], ((row_end > row_begin + rgrp) ? kf8[7] : CUDART_INF_F));
        qf8[0] = _vec_load_2[0];
        kf8[0] = _vec_load_3[0];
        qf8[1] = _vec_load_2[1];
        kf8[1] = _vec_load_3[1];
        qf8[2] = _vec_load_2[2];
        kf8[2] = _vec_load_3[2];
        qf8[3] = _vec_load_2[3];
        kf8[3] = _vec_load_3[3];
        qf8[4] = _vec_load_2[4];
        kf8[4] = _vec_load_3[4];
        qf8[5] = _vec_load_2[5];
        kf8[5] = _vec_load_3[5];
        qf8[6] = _vec_load_2[6];
        kf8[6] = _vec_load_3[6];
        qf8[7] = _vec_load_2[7];
        kf8[7] = _vec_load_3[7];
        ksum[0] = ksum[0] + ((row_end > row_begin + 16 + rgrp) ? kf8[0] : 0.0f);
        qsum[0] = qsum[0] + ((row_end > row_begin + 16 + rgrp) ? qf8[0] : 0.0f);
        ksum[1] = ksum[1] + ((row_end > row_begin + 16 + rgrp) ? kf8[1] : 0.0f);
        qsum[1] = qsum[1] + ((row_end > row_begin + 16 + rgrp) ? qf8[1] : 0.0f);
        ksum[2] = ksum[2] + ((row_end > row_begin + 16 + rgrp) ? kf8[2] : 0.0f);
        qsum[2] = qsum[2] + ((row_end > row_begin + 16 + rgrp) ? qf8[2] : 0.0f);
        ksum[3] = ksum[3] + ((row_end > row_begin + 16 + rgrp) ? kf8[3] : 0.0f);
        qsum[3] = qsum[3] + ((row_end > row_begin + 16 + rgrp) ? qf8[3] : 0.0f);
        ksum[4] = ksum[4] + ((row_end > row_begin + 16 + rgrp) ? kf8[4] : 0.0f);
        qsum[4] = qsum[4] + ((row_end > row_begin + 16 + rgrp) ? qf8[4] : 0.0f);
        ksum[5] = ksum[5] + ((row_end > row_begin + 16 + rgrp) ? kf8[5] : 0.0f);
        qsum[5] = qsum[5] + ((row_end > row_begin + 16 + rgrp) ? qf8[5] : 0.0f);
        ksum[6] = ksum[6] + ((row_end > row_begin + 16 + rgrp) ? kf8[6] : 0.0f);
        qsum[6] = qsum[6] + ((row_end > row_begin + 16 + rgrp) ? qf8[6] : 0.0f);
        ksum[7] = ksum[7] + ((row_end > row_begin + 16 + rgrp) ? kf8[7] : 0.0f);
        qsum[7] = qsum[7] + ((row_end > row_begin + 16 + rgrp) ? qf8[7] : 0.0f);
        qmax[0] = fmaxf(qmax[0], ((row_end > row_begin + 16 + rgrp) ? qf8[0] : -CUDART_INF_F));
        qmin[0] = fminf(qmin[0], ((row_end > row_begin + 16 + rgrp) ? qf8[0] : CUDART_INF_F));
        kmax[0] = fmaxf(kmax[0], ((row_end > row_begin + 16 + rgrp) ? kf8[0] : -CUDART_INF_F));
        kmin[0] = fminf(kmin[0], ((row_end > row_begin + 16 + rgrp) ? kf8[0] : CUDART_INF_F));
        qmax[1] = fmaxf(qmax[1], ((row_end > row_begin + 16 + rgrp) ? qf8[1] : -CUDART_INF_F));
        qmin[1] = fminf(qmin[1], ((row_end > row_begin + 16 + rgrp) ? qf8[1] : CUDART_INF_F));
        kmax[1] = fmaxf(kmax[1], ((row_end > row_begin + 16 + rgrp) ? kf8[1] : -CUDART_INF_F));
        kmin[1] = fminf(kmin[1], ((row_end > row_begin + 16 + rgrp) ? kf8[1] : CUDART_INF_F));
        qmax[2] = fmaxf(qmax[2], ((row_end > row_begin + 16 + rgrp) ? qf8[2] : -CUDART_INF_F));
        qmin[2] = fminf(qmin[2], ((row_end > row_begin + 16 + rgrp) ? qf8[2] : CUDART_INF_F));
        kmax[2] = fmaxf(kmax[2], ((row_end > row_begin + 16 + rgrp) ? kf8[2] : -CUDART_INF_F));
        kmin[2] = fminf(kmin[2], ((row_end > row_begin + 16 + rgrp) ? kf8[2] : CUDART_INF_F));
        qmax[3] = fmaxf(qmax[3], ((row_end > row_begin + 16 + rgrp) ? qf8[3] : -CUDART_INF_F));
        qmin[3] = fminf(qmin[3], ((row_end > row_begin + 16 + rgrp) ? qf8[3] : CUDART_INF_F));
        kmax[3] = fmaxf(kmax[3], ((row_end > row_begin + 16 + rgrp) ? kf8[3] : -CUDART_INF_F));
        kmin[3] = fminf(kmin[3], ((row_end > row_begin + 16 + rgrp) ? kf8[3] : CUDART_INF_F));
        qmax[4] = fmaxf(qmax[4], ((row_end > row_begin + 16 + rgrp) ? qf8[4] : -CUDART_INF_F));
        qmin[4] = fminf(qmin[4], ((row_end > row_begin + 16 + rgrp) ? qf8[4] : CUDART_INF_F));
        kmax[4] = fmaxf(kmax[4], ((row_end > row_begin + 16 + rgrp) ? kf8[4] : -CUDART_INF_F));
        kmin[4] = fminf(kmin[4], ((row_end > row_begin + 16 + rgrp) ? kf8[4] : CUDART_INF_F));
        qmax[5] = fmaxf(qmax[5], ((row_end > row_begin + 16 + rgrp) ? qf8[5] : -CUDART_INF_F));
        qmin[5] = fminf(qmin[5], ((row_end > row_begin + 16 + rgrp) ? qf8[5] : CUDART_INF_F));
        kmax[5] = fmaxf(kmax[5], ((row_end > row_begin + 16 + rgrp) ? kf8[5] : -CUDART_INF_F));
        kmin[5] = fminf(kmin[5], ((row_end > row_begin + 16 + rgrp) ? kf8[5] : CUDART_INF_F));
        qmax[6] = fmaxf(qmax[6], ((row_end > row_begin + 16 + rgrp) ? qf8[6] : -CUDART_INF_F));
        qmin[6] = fminf(qmin[6], ((row_end > row_begin + 16 + rgrp) ? qf8[6] : CUDART_INF_F));
        kmax[6] = fmaxf(kmax[6], ((row_end > row_begin + 16 + rgrp) ? kf8[6] : -CUDART_INF_F));
        kmin[6] = fminf(kmin[6], ((row_end > row_begin + 16 + rgrp) ? kf8[6] : CUDART_INF_F));
        qmax[7] = fmaxf(qmax[7], ((row_end > row_begin + 16 + rgrp) ? qf8[7] : -CUDART_INF_F));
        qmin[7] = fminf(qmin[7], ((row_end > row_begin + 16 + rgrp) ? qf8[7] : CUDART_INF_F));
        kmax[7] = fmaxf(kmax[7], ((row_end > row_begin + 16 + rgrp) ? kf8[7] : -CUDART_INF_F));
        kmin[7] = fminf(kmin[7], ((row_end > row_begin + 16 + rgrp) ? kf8[7] : CUDART_INF_F));
        qf8[0] = _vec_load_4[0];
        kf8[0] = _vec_load_5[0];
        qf8[1] = _vec_load_4[1];
        kf8[1] = _vec_load_5[1];
        qf8[2] = _vec_load_4[2];
        kf8[2] = _vec_load_5[2];
        qf8[3] = _vec_load_4[3];
        kf8[3] = _vec_load_5[3];
        qf8[4] = _vec_load_4[4];
        kf8[4] = _vec_load_5[4];
        qf8[5] = _vec_load_4[5];
        kf8[5] = _vec_load_5[5];
        qf8[6] = _vec_load_4[6];
        kf8[6] = _vec_load_5[6];
        qf8[7] = _vec_load_4[7];
        kf8[7] = _vec_load_5[7];
        ksum[0] = ksum[0] + ((row_end > row_begin + 32 + rgrp) ? kf8[0] : 0.0f);
        qsum[0] = qsum[0] + ((row_end > row_begin + 32 + rgrp) ? qf8[0] : 0.0f);
        ksum[1] = ksum[1] + ((row_end > row_begin + 32 + rgrp) ? kf8[1] : 0.0f);
        qsum[1] = qsum[1] + ((row_end > row_begin + 32 + rgrp) ? qf8[1] : 0.0f);
        ksum[2] = ksum[2] + ((row_end > row_begin + 32 + rgrp) ? kf8[2] : 0.0f);
        qsum[2] = qsum[2] + ((row_end > row_begin + 32 + rgrp) ? qf8[2] : 0.0f);
        ksum[3] = ksum[3] + ((row_end > row_begin + 32 + rgrp) ? kf8[3] : 0.0f);
        qsum[3] = qsum[3] + ((row_end > row_begin + 32 + rgrp) ? qf8[3] : 0.0f);
        ksum[4] = ksum[4] + ((row_end > row_begin + 32 + rgrp) ? kf8[4] : 0.0f);
        qsum[4] = qsum[4] + ((row_end > row_begin + 32 + rgrp) ? qf8[4] : 0.0f);
        ksum[5] = ksum[5] + ((row_end > row_begin + 32 + rgrp) ? kf8[5] : 0.0f);
        qsum[5] = qsum[5] + ((row_end > row_begin + 32 + rgrp) ? qf8[5] : 0.0f);
        ksum[6] = ksum[6] + ((row_end > row_begin + 32 + rgrp) ? kf8[6] : 0.0f);
        qsum[6] = qsum[6] + ((row_end > row_begin + 32 + rgrp) ? qf8[6] : 0.0f);
        ksum[7] = ksum[7] + ((row_end > row_begin + 32 + rgrp) ? kf8[7] : 0.0f);
        qsum[7] = qsum[7] + ((row_end > row_begin + 32 + rgrp) ? qf8[7] : 0.0f);
        qmax[0] = fmaxf(qmax[0], ((row_end > row_begin + 32 + rgrp) ? qf8[0] : -CUDART_INF_F));
        qmin[0] = fminf(qmin[0], ((row_end > row_begin + 32 + rgrp) ? qf8[0] : CUDART_INF_F));
        kmax[0] = fmaxf(kmax[0], ((row_end > row_begin + 32 + rgrp) ? kf8[0] : -CUDART_INF_F));
        kmin[0] = fminf(kmin[0], ((row_end > row_begin + 32 + rgrp) ? kf8[0] : CUDART_INF_F));
        qmax[1] = fmaxf(qmax[1], ((row_end > row_begin + 32 + rgrp) ? qf8[1] : -CUDART_INF_F));
        qmin[1] = fminf(qmin[1], ((row_end > row_begin + 32 + rgrp) ? qf8[1] : CUDART_INF_F));
        kmax[1] = fmaxf(kmax[1], ((row_end > row_begin + 32 + rgrp) ? kf8[1] : -CUDART_INF_F));
        kmin[1] = fminf(kmin[1], ((row_end > row_begin + 32 + rgrp) ? kf8[1] : CUDART_INF_F));
        qmax[2] = fmaxf(qmax[2], ((row_end > row_begin + 32 + rgrp) ? qf8[2] : -CUDART_INF_F));
        qmin[2] = fminf(qmin[2], ((row_end > row_begin + 32 + rgrp) ? qf8[2] : CUDART_INF_F));
        kmax[2] = fmaxf(kmax[2], ((row_end > row_begin + 32 + rgrp) ? kf8[2] : -CUDART_INF_F));
        kmin[2] = fminf(kmin[2], ((row_end > row_begin + 32 + rgrp) ? kf8[2] : CUDART_INF_F));
        qmax[3] = fmaxf(qmax[3], ((row_end > row_begin + 32 + rgrp) ? qf8[3] : -CUDART_INF_F));
        qmin[3] = fminf(qmin[3], ((row_end > row_begin + 32 + rgrp) ? qf8[3] : CUDART_INF_F));
        kmax[3] = fmaxf(kmax[3], ((row_end > row_begin + 32 + rgrp) ? kf8[3] : -CUDART_INF_F));
        kmin[3] = fminf(kmin[3], ((row_end > row_begin + 32 + rgrp) ? kf8[3] : CUDART_INF_F));
        qmax[4] = fmaxf(qmax[4], ((row_end > row_begin + 32 + rgrp) ? qf8[4] : -CUDART_INF_F));
        qmin[4] = fminf(qmin[4], ((row_end > row_begin + 32 + rgrp) ? qf8[4] : CUDART_INF_F));
        kmax[4] = fmaxf(kmax[4], ((row_end > row_begin + 32 + rgrp) ? kf8[4] : -CUDART_INF_F));
        kmin[4] = fminf(kmin[4], ((row_end > row_begin + 32 + rgrp) ? kf8[4] : CUDART_INF_F));
        qmax[5] = fmaxf(qmax[5], ((row_end > row_begin + 32 + rgrp) ? qf8[5] : -CUDART_INF_F));
        qmin[5] = fminf(qmin[5], ((row_end > row_begin + 32 + rgrp) ? qf8[5] : CUDART_INF_F));
        kmax[5] = fmaxf(kmax[5], ((row_end > row_begin + 32 + rgrp) ? kf8[5] : -CUDART_INF_F));
        kmin[5] = fminf(kmin[5], ((row_end > row_begin + 32 + rgrp) ? kf8[5] : CUDART_INF_F));
        qmax[6] = fmaxf(qmax[6], ((row_end > row_begin + 32 + rgrp) ? qf8[6] : -CUDART_INF_F));
        qmin[6] = fminf(qmin[6], ((row_end > row_begin + 32 + rgrp) ? qf8[6] : CUDART_INF_F));
        kmax[6] = fmaxf(kmax[6], ((row_end > row_begin + 32 + rgrp) ? kf8[6] : -CUDART_INF_F));
        kmin[6] = fminf(kmin[6], ((row_end > row_begin + 32 + rgrp) ? kf8[6] : CUDART_INF_F));
        qmax[7] = fmaxf(qmax[7], ((row_end > row_begin + 32 + rgrp) ? qf8[7] : -CUDART_INF_F));
        qmin[7] = fminf(qmin[7], ((row_end > row_begin + 32 + rgrp) ? qf8[7] : CUDART_INF_F));
        kmax[7] = fmaxf(kmax[7], ((row_end > row_begin + 32 + rgrp) ? kf8[7] : -CUDART_INF_F));
        kmin[7] = fminf(kmin[7], ((row_end > row_begin + 32 + rgrp) ? kf8[7] : CUDART_INF_F));
        qf8[0] = _vec_load_6[0];
        kf8[0] = _vec_load_7[0];
        qf8[1] = _vec_load_6[1];
        kf8[1] = _vec_load_7[1];
        qf8[2] = _vec_load_6[2];
        kf8[2] = _vec_load_7[2];
        qf8[3] = _vec_load_6[3];
        kf8[3] = _vec_load_7[3];
        qf8[4] = _vec_load_6[4];
        kf8[4] = _vec_load_7[4];
        qf8[5] = _vec_load_6[5];
        kf8[5] = _vec_load_7[5];
        qf8[6] = _vec_load_6[6];
        kf8[6] = _vec_load_7[6];
        qf8[7] = _vec_load_6[7];
        kf8[7] = _vec_load_7[7];
        ksum[0] = ksum[0] + ((row_end > row_begin + 48 + rgrp) ? kf8[0] : 0.0f);
        qsum[0] = qsum[0] + ((row_end > row_begin + 48 + rgrp) ? qf8[0] : 0.0f);
        ksum[1] = ksum[1] + ((row_end > row_begin + 48 + rgrp) ? kf8[1] : 0.0f);
        qsum[1] = qsum[1] + ((row_end > row_begin + 48 + rgrp) ? qf8[1] : 0.0f);
        ksum[2] = ksum[2] + ((row_end > row_begin + 48 + rgrp) ? kf8[2] : 0.0f);
        qsum[2] = qsum[2] + ((row_end > row_begin + 48 + rgrp) ? qf8[2] : 0.0f);
        ksum[3] = ksum[3] + ((row_end > row_begin + 48 + rgrp) ? kf8[3] : 0.0f);
        qsum[3] = qsum[3] + ((row_end > row_begin + 48 + rgrp) ? qf8[3] : 0.0f);
        ksum[4] = ksum[4] + ((row_end > row_begin + 48 + rgrp) ? kf8[4] : 0.0f);
        qsum[4] = qsum[4] + ((row_end > row_begin + 48 + rgrp) ? qf8[4] : 0.0f);
        ksum[5] = ksum[5] + ((row_end > row_begin + 48 + rgrp) ? kf8[5] : 0.0f);
        qsum[5] = qsum[5] + ((row_end > row_begin + 48 + rgrp) ? qf8[5] : 0.0f);
        ksum[6] = ksum[6] + ((row_end > row_begin + 48 + rgrp) ? kf8[6] : 0.0f);
        qsum[6] = qsum[6] + ((row_end > row_begin + 48 + rgrp) ? qf8[6] : 0.0f);
        ksum[7] = ksum[7] + ((row_end > row_begin + 48 + rgrp) ? kf8[7] : 0.0f);
        qsum[7] = qsum[7] + ((row_end > row_begin + 48 + rgrp) ? qf8[7] : 0.0f);
        qmax[0] = fmaxf(qmax[0], ((row_end > row_begin + 48 + rgrp) ? qf8[0] : -CUDART_INF_F));
        qmin[0] = fminf(qmin[0], ((row_end > row_begin + 48 + rgrp) ? qf8[0] : CUDART_INF_F));
        kmax[0] = fmaxf(kmax[0], ((row_end > row_begin + 48 + rgrp) ? kf8[0] : -CUDART_INF_F));
        kmin[0] = fminf(kmin[0], ((row_end > row_begin + 48 + rgrp) ? kf8[0] : CUDART_INF_F));
        qmax[1] = fmaxf(qmax[1], ((row_end > row_begin + 48 + rgrp) ? qf8[1] : -CUDART_INF_F));
        qmin[1] = fminf(qmin[1], ((row_end > row_begin + 48 + rgrp) ? qf8[1] : CUDART_INF_F));
        kmax[1] = fmaxf(kmax[1], ((row_end > row_begin + 48 + rgrp) ? kf8[1] : -CUDART_INF_F));
        kmin[1] = fminf(kmin[1], ((row_end > row_begin + 48 + rgrp) ? kf8[1] : CUDART_INF_F));
        qmax[2] = fmaxf(qmax[2], ((row_end > row_begin + 48 + rgrp) ? qf8[2] : -CUDART_INF_F));
        qmin[2] = fminf(qmin[2], ((row_end > row_begin + 48 + rgrp) ? qf8[2] : CUDART_INF_F));
        kmax[2] = fmaxf(kmax[2], ((row_end > row_begin + 48 + rgrp) ? kf8[2] : -CUDART_INF_F));
        kmin[2] = fminf(kmin[2], ((row_end > row_begin + 48 + rgrp) ? kf8[2] : CUDART_INF_F));
        qmax[3] = fmaxf(qmax[3], ((row_end > row_begin + 48 + rgrp) ? qf8[3] : -CUDART_INF_F));
        qmin[3] = fminf(qmin[3], ((row_end > row_begin + 48 + rgrp) ? qf8[3] : CUDART_INF_F));
        kmax[3] = fmaxf(kmax[3], ((row_end > row_begin + 48 + rgrp) ? kf8[3] : -CUDART_INF_F));
        kmin[3] = fminf(kmin[3], ((row_end > row_begin + 48 + rgrp) ? kf8[3] : CUDART_INF_F));
        qmax[4] = fmaxf(qmax[4], ((row_end > row_begin + 48 + rgrp) ? qf8[4] : -CUDART_INF_F));
        qmin[4] = fminf(qmin[4], ((row_end > row_begin + 48 + rgrp) ? qf8[4] : CUDART_INF_F));
        kmax[4] = fmaxf(kmax[4], ((row_end > row_begin + 48 + rgrp) ? kf8[4] : -CUDART_INF_F));
        kmin[4] = fminf(kmin[4], ((row_end > row_begin + 48 + rgrp) ? kf8[4] : CUDART_INF_F));
        qmax[5] = fmaxf(qmax[5], ((row_end > row_begin + 48 + rgrp) ? qf8[5] : -CUDART_INF_F));
        qmin[5] = fminf(qmin[5], ((row_end > row_begin + 48 + rgrp) ? qf8[5] : CUDART_INF_F));
        kmax[5] = fmaxf(kmax[5], ((row_end > row_begin + 48 + rgrp) ? kf8[5] : -CUDART_INF_F));
        kmin[5] = fminf(kmin[5], ((row_end > row_begin + 48 + rgrp) ? kf8[5] : CUDART_INF_F));
        qmax[6] = fmaxf(qmax[6], ((row_end > row_begin + 48 + rgrp) ? qf8[6] : -CUDART_INF_F));
        qmin[6] = fminf(qmin[6], ((row_end > row_begin + 48 + rgrp) ? qf8[6] : CUDART_INF_F));
        kmax[6] = fmaxf(kmax[6], ((row_end > row_begin + 48 + rgrp) ? kf8[6] : -CUDART_INF_F));
        kmin[6] = fminf(kmin[6], ((row_end > row_begin + 48 + rgrp) ? kf8[6] : CUDART_INF_F));
        qmax[7] = fmaxf(qmax[7], ((row_end > row_begin + 48 + rgrp) ? qf8[7] : -CUDART_INF_F));
        qmin[7] = fminf(qmin[7], ((row_end > row_begin + 48 + rgrp) ? qf8[7] : CUDART_INF_F));
        kmax[7] = fmaxf(kmax[7], ((row_end > row_begin + 48 + rgrp) ? kf8[7] : -CUDART_INF_F));
        kmin[7] = fminf(kmin[7], ((row_end > row_begin + 48 + rgrp) ? kf8[7] : CUDART_INF_F));
        qf8[0] = _vec_load_8[0];
        kf8[0] = _vec_load_9[0];
        qf8[1] = _vec_load_8[1];
        kf8[1] = _vec_load_9[1];
        qf8[2] = _vec_load_8[2];
        kf8[2] = _vec_load_9[2];
        qf8[3] = _vec_load_8[3];
        kf8[3] = _vec_load_9[3];
        qf8[4] = _vec_load_8[4];
        kf8[4] = _vec_load_9[4];
        qf8[5] = _vec_load_8[5];
        kf8[5] = _vec_load_9[5];
        qf8[6] = _vec_load_8[6];
        kf8[6] = _vec_load_9[6];
        qf8[7] = _vec_load_8[7];
        kf8[7] = _vec_load_9[7];
        ksum[0] = ksum[0] + ((row_end > row_begin + 64 + rgrp) ? kf8[0] : 0.0f);
        qsum[0] = qsum[0] + ((row_end > row_begin + 64 + rgrp) ? qf8[0] : 0.0f);
        ksum[1] = ksum[1] + ((row_end > row_begin + 64 + rgrp) ? kf8[1] : 0.0f);
        qsum[1] = qsum[1] + ((row_end > row_begin + 64 + rgrp) ? qf8[1] : 0.0f);
        ksum[2] = ksum[2] + ((row_end > row_begin + 64 + rgrp) ? kf8[2] : 0.0f);
        qsum[2] = qsum[2] + ((row_end > row_begin + 64 + rgrp) ? qf8[2] : 0.0f);
        ksum[3] = ksum[3] + ((row_end > row_begin + 64 + rgrp) ? kf8[3] : 0.0f);
        qsum[3] = qsum[3] + ((row_end > row_begin + 64 + rgrp) ? qf8[3] : 0.0f);
        ksum[4] = ksum[4] + ((row_end > row_begin + 64 + rgrp) ? kf8[4] : 0.0f);
        qsum[4] = qsum[4] + ((row_end > row_begin + 64 + rgrp) ? qf8[4] : 0.0f);
        ksum[5] = ksum[5] + ((row_end > row_begin + 64 + rgrp) ? kf8[5] : 0.0f);
        qsum[5] = qsum[5] + ((row_end > row_begin + 64 + rgrp) ? qf8[5] : 0.0f);
        ksum[6] = ksum[6] + ((row_end > row_begin + 64 + rgrp) ? kf8[6] : 0.0f);
        qsum[6] = qsum[6] + ((row_end > row_begin + 64 + rgrp) ? qf8[6] : 0.0f);
        ksum[7] = ksum[7] + ((row_end > row_begin + 64 + rgrp) ? kf8[7] : 0.0f);
        qsum[7] = qsum[7] + ((row_end > row_begin + 64 + rgrp) ? qf8[7] : 0.0f);
        qmax[0] = fmaxf(qmax[0], ((row_end > row_begin + 64 + rgrp) ? qf8[0] : -CUDART_INF_F));
        qmin[0] = fminf(qmin[0], ((row_end > row_begin + 64 + rgrp) ? qf8[0] : CUDART_INF_F));
        kmax[0] = fmaxf(kmax[0], ((row_end > row_begin + 64 + rgrp) ? kf8[0] : -CUDART_INF_F));
        kmin[0] = fminf(kmin[0], ((row_end > row_begin + 64 + rgrp) ? kf8[0] : CUDART_INF_F));
        qmax[1] = fmaxf(qmax[1], ((row_end > row_begin + 64 + rgrp) ? qf8[1] : -CUDART_INF_F));
        qmin[1] = fminf(qmin[1], ((row_end > row_begin + 64 + rgrp) ? qf8[1] : CUDART_INF_F));
        kmax[1] = fmaxf(kmax[1], ((row_end > row_begin + 64 + rgrp) ? kf8[1] : -CUDART_INF_F));
        kmin[1] = fminf(kmin[1], ((row_end > row_begin + 64 + rgrp) ? kf8[1] : CUDART_INF_F));
        qmax[2] = fmaxf(qmax[2], ((row_end > row_begin + 64 + rgrp) ? qf8[2] : -CUDART_INF_F));
        qmin[2] = fminf(qmin[2], ((row_end > row_begin + 64 + rgrp) ? qf8[2] : CUDART_INF_F));
        kmax[2] = fmaxf(kmax[2], ((row_end > row_begin + 64 + rgrp) ? kf8[2] : -CUDART_INF_F));
        kmin[2] = fminf(kmin[2], ((row_end > row_begin + 64 + rgrp) ? kf8[2] : CUDART_INF_F));
        qmax[3] = fmaxf(qmax[3], ((row_end > row_begin + 64 + rgrp) ? qf8[3] : -CUDART_INF_F));
        qmin[3] = fminf(qmin[3], ((row_end > row_begin + 64 + rgrp) ? qf8[3] : CUDART_INF_F));
        kmax[3] = fmaxf(kmax[3], ((row_end > row_begin + 64 + rgrp) ? kf8[3] : -CUDART_INF_F));
        kmin[3] = fminf(kmin[3], ((row_end > row_begin + 64 + rgrp) ? kf8[3] : CUDART_INF_F));
        qmax[4] = fmaxf(qmax[4], ((row_end > row_begin + 64 + rgrp) ? qf8[4] : -CUDART_INF_F));
        qmin[4] = fminf(qmin[4], ((row_end > row_begin + 64 + rgrp) ? qf8[4] : CUDART_INF_F));
        kmax[4] = fmaxf(kmax[4], ((row_end > row_begin + 64 + rgrp) ? kf8[4] : -CUDART_INF_F));
        kmin[4] = fminf(kmin[4], ((row_end > row_begin + 64 + rgrp) ? kf8[4] : CUDART_INF_F));
        qmax[5] = fmaxf(qmax[5], ((row_end > row_begin + 64 + rgrp) ? qf8[5] : -CUDART_INF_F));
        qmin[5] = fminf(qmin[5], ((row_end > row_begin + 64 + rgrp) ? qf8[5] : CUDART_INF_F));
        kmax[5] = fmaxf(kmax[5], ((row_end > row_begin + 64 + rgrp) ? kf8[5] : -CUDART_INF_F));
        kmin[5] = fminf(kmin[5], ((row_end > row_begin + 64 + rgrp) ? kf8[5] : CUDART_INF_F));
        qmax[6] = fmaxf(qmax[6], ((row_end > row_begin + 64 + rgrp) ? qf8[6] : -CUDART_INF_F));
        qmin[6] = fminf(qmin[6], ((row_end > row_begin + 64 + rgrp) ? qf8[6] : CUDART_INF_F));
        kmax[6] = fmaxf(kmax[6], ((row_end > row_begin + 64 + rgrp) ? kf8[6] : -CUDART_INF_F));
        kmin[6] = fminf(kmin[6], ((row_end > row_begin + 64 + rgrp) ? kf8[6] : CUDART_INF_F));
        qmax[7] = fmaxf(qmax[7], ((row_end > row_begin + 64 + rgrp) ? qf8[7] : -CUDART_INF_F));
        qmin[7] = fminf(qmin[7], ((row_end > row_begin + 64 + rgrp) ? qf8[7] : CUDART_INF_F));
        kmax[7] = fmaxf(kmax[7], ((row_end > row_begin + 64 + rgrp) ? kf8[7] : -CUDART_INF_F));
        kmin[7] = fminf(kmin[7], ((row_end > row_begin + 64 + rgrp) ? kf8[7] : CUDART_INF_F));
        qf8[0] = _vec_load_10[0];
        kf8[0] = _vec_load_11[0];
        qf8[1] = _vec_load_10[1];
        kf8[1] = _vec_load_11[1];
        qf8[2] = _vec_load_10[2];
        kf8[2] = _vec_load_11[2];
        qf8[3] = _vec_load_10[3];
        kf8[3] = _vec_load_11[3];
        qf8[4] = _vec_load_10[4];
        kf8[4] = _vec_load_11[4];
        qf8[5] = _vec_load_10[5];
        kf8[5] = _vec_load_11[5];
        qf8[6] = _vec_load_10[6];
        kf8[6] = _vec_load_11[6];
        qf8[7] = _vec_load_10[7];
        kf8[7] = _vec_load_11[7];
        ksum[0] = ksum[0] + ((row_end > row_begin + 80 + rgrp) ? kf8[0] : 0.0f);
        qsum[0] = qsum[0] + ((row_end > row_begin + 80 + rgrp) ? qf8[0] : 0.0f);
        ksum[1] = ksum[1] + ((row_end > row_begin + 80 + rgrp) ? kf8[1] : 0.0f);
        qsum[1] = qsum[1] + ((row_end > row_begin + 80 + rgrp) ? qf8[1] : 0.0f);
        ksum[2] = ksum[2] + ((row_end > row_begin + 80 + rgrp) ? kf8[2] : 0.0f);
        qsum[2] = qsum[2] + ((row_end > row_begin + 80 + rgrp) ? qf8[2] : 0.0f);
        ksum[3] = ksum[3] + ((row_end > row_begin + 80 + rgrp) ? kf8[3] : 0.0f);
        qsum[3] = qsum[3] + ((row_end > row_begin + 80 + rgrp) ? qf8[3] : 0.0f);
        ksum[4] = ksum[4] + ((row_end > row_begin + 80 + rgrp) ? kf8[4] : 0.0f);
        qsum[4] = qsum[4] + ((row_end > row_begin + 80 + rgrp) ? qf8[4] : 0.0f);
        ksum[5] = ksum[5] + ((row_end > row_begin + 80 + rgrp) ? kf8[5] : 0.0f);
        qsum[5] = qsum[5] + ((row_end > row_begin + 80 + rgrp) ? qf8[5] : 0.0f);
        ksum[6] = ksum[6] + ((row_end > row_begin + 80 + rgrp) ? kf8[6] : 0.0f);
        qsum[6] = qsum[6] + ((row_end > row_begin + 80 + rgrp) ? qf8[6] : 0.0f);
        ksum[7] = ksum[7] + ((row_end > row_begin + 80 + rgrp) ? kf8[7] : 0.0f);
        qsum[7] = qsum[7] + ((row_end > row_begin + 80 + rgrp) ? qf8[7] : 0.0f);
        qmax[0] = fmaxf(qmax[0], ((row_end > row_begin + 80 + rgrp) ? qf8[0] : -CUDART_INF_F));
        qmin[0] = fminf(qmin[0], ((row_end > row_begin + 80 + rgrp) ? qf8[0] : CUDART_INF_F));
        kmax[0] = fmaxf(kmax[0], ((row_end > row_begin + 80 + rgrp) ? kf8[0] : -CUDART_INF_F));
        kmin[0] = fminf(kmin[0], ((row_end > row_begin + 80 + rgrp) ? kf8[0] : CUDART_INF_F));
        qmax[1] = fmaxf(qmax[1], ((row_end > row_begin + 80 + rgrp) ? qf8[1] : -CUDART_INF_F));
        qmin[1] = fminf(qmin[1], ((row_end > row_begin + 80 + rgrp) ? qf8[1] : CUDART_INF_F));
        kmax[1] = fmaxf(kmax[1], ((row_end > row_begin + 80 + rgrp) ? kf8[1] : -CUDART_INF_F));
        kmin[1] = fminf(kmin[1], ((row_end > row_begin + 80 + rgrp) ? kf8[1] : CUDART_INF_F));
        qmax[2] = fmaxf(qmax[2], ((row_end > row_begin + 80 + rgrp) ? qf8[2] : -CUDART_INF_F));
        qmin[2] = fminf(qmin[2], ((row_end > row_begin + 80 + rgrp) ? qf8[2] : CUDART_INF_F));
        kmax[2] = fmaxf(kmax[2], ((row_end > row_begin + 80 + rgrp) ? kf8[2] : -CUDART_INF_F));
        kmin[2] = fminf(kmin[2], ((row_end > row_begin + 80 + rgrp) ? kf8[2] : CUDART_INF_F));
        qmax[3] = fmaxf(qmax[3], ((row_end > row_begin + 80 + rgrp) ? qf8[3] : -CUDART_INF_F));
        qmin[3] = fminf(qmin[3], ((row_end > row_begin + 80 + rgrp) ? qf8[3] : CUDART_INF_F));
        kmax[3] = fmaxf(kmax[3], ((row_end > row_begin + 80 + rgrp) ? kf8[3] : -CUDART_INF_F));
        kmin[3] = fminf(kmin[3], ((row_end > row_begin + 80 + rgrp) ? kf8[3] : CUDART_INF_F));
        qmax[4] = fmaxf(qmax[4], ((row_end > row_begin + 80 + rgrp) ? qf8[4] : -CUDART_INF_F));
        qmin[4] = fminf(qmin[4], ((row_end > row_begin + 80 + rgrp) ? qf8[4] : CUDART_INF_F));
        kmax[4] = fmaxf(kmax[4], ((row_end > row_begin + 80 + rgrp) ? kf8[4] : -CUDART_INF_F));
        kmin[4] = fminf(kmin[4], ((row_end > row_begin + 80 + rgrp) ? kf8[4] : CUDART_INF_F));
        qmax[5] = fmaxf(qmax[5], ((row_end > row_begin + 80 + rgrp) ? qf8[5] : -CUDART_INF_F));
        qmin[5] = fminf(qmin[5], ((row_end > row_begin + 80 + rgrp) ? qf8[5] : CUDART_INF_F));
        kmax[5] = fmaxf(kmax[5], ((row_end > row_begin + 80 + rgrp) ? kf8[5] : -CUDART_INF_F));
        kmin[5] = fminf(kmin[5], ((row_end > row_begin + 80 + rgrp) ? kf8[5] : CUDART_INF_F));
        qmax[6] = fmaxf(qmax[6], ((row_end > row_begin + 80 + rgrp) ? qf8[6] : -CUDART_INF_F));
        qmin[6] = fminf(qmin[6], ((row_end > row_begin + 80 + rgrp) ? qf8[6] : CUDART_INF_F));
        kmax[6] = fmaxf(kmax[6], ((row_end > row_begin + 80 + rgrp) ? kf8[6] : -CUDART_INF_F));
        kmin[6] = fminf(kmin[6], ((row_end > row_begin + 80 + rgrp) ? kf8[6] : CUDART_INF_F));
        qmax[7] = fmaxf(qmax[7], ((row_end > row_begin + 80 + rgrp) ? qf8[7] : -CUDART_INF_F));
        qmin[7] = fminf(qmin[7], ((row_end > row_begin + 80 + rgrp) ? qf8[7] : CUDART_INF_F));
        kmax[7] = fmaxf(kmax[7], ((row_end > row_begin + 80 + rgrp) ? kf8[7] : -CUDART_INF_F));
        kmin[7] = fminf(kmin[7], ((row_end > row_begin + 80 + rgrp) ? kf8[7] : CUDART_INF_F));
        qf8[0] = _vec_load_12[0];
        kf8[0] = _vec_load_13[0];
        qf8[1] = _vec_load_12[1];
        kf8[1] = _vec_load_13[1];
        qf8[2] = _vec_load_12[2];
        kf8[2] = _vec_load_13[2];
        qf8[3] = _vec_load_12[3];
        kf8[3] = _vec_load_13[3];
        qf8[4] = _vec_load_12[4];
        kf8[4] = _vec_load_13[4];
        qf8[5] = _vec_load_12[5];
        kf8[5] = _vec_load_13[5];
        qf8[6] = _vec_load_12[6];
        kf8[6] = _vec_load_13[6];
        qf8[7] = _vec_load_12[7];
        kf8[7] = _vec_load_13[7];
        ksum[0] = ksum[0] + ((row_end > row_begin + 96 + rgrp) ? kf8[0] : 0.0f);
        qsum[0] = qsum[0] + ((row_end > row_begin + 96 + rgrp) ? qf8[0] : 0.0f);
        ksum[1] = ksum[1] + ((row_end > row_begin + 96 + rgrp) ? kf8[1] : 0.0f);
        qsum[1] = qsum[1] + ((row_end > row_begin + 96 + rgrp) ? qf8[1] : 0.0f);
        ksum[2] = ksum[2] + ((row_end > row_begin + 96 + rgrp) ? kf8[2] : 0.0f);
        qsum[2] = qsum[2] + ((row_end > row_begin + 96 + rgrp) ? qf8[2] : 0.0f);
        ksum[3] = ksum[3] + ((row_end > row_begin + 96 + rgrp) ? kf8[3] : 0.0f);
        qsum[3] = qsum[3] + ((row_end > row_begin + 96 + rgrp) ? qf8[3] : 0.0f);
        ksum[4] = ksum[4] + ((row_end > row_begin + 96 + rgrp) ? kf8[4] : 0.0f);
        qsum[4] = qsum[4] + ((row_end > row_begin + 96 + rgrp) ? qf8[4] : 0.0f);
        ksum[5] = ksum[5] + ((row_end > row_begin + 96 + rgrp) ? kf8[5] : 0.0f);
        qsum[5] = qsum[5] + ((row_end > row_begin + 96 + rgrp) ? qf8[5] : 0.0f);
        ksum[6] = ksum[6] + ((row_end > row_begin + 96 + rgrp) ? kf8[6] : 0.0f);
        qsum[6] = qsum[6] + ((row_end > row_begin + 96 + rgrp) ? qf8[6] : 0.0f);
        ksum[7] = ksum[7] + ((row_end > row_begin + 96 + rgrp) ? kf8[7] : 0.0f);
        qsum[7] = qsum[7] + ((row_end > row_begin + 96 + rgrp) ? qf8[7] : 0.0f);
        qmax[0] = fmaxf(qmax[0], ((row_end > row_begin + 96 + rgrp) ? qf8[0] : -CUDART_INF_F));
        qmin[0] = fminf(qmin[0], ((row_end > row_begin + 96 + rgrp) ? qf8[0] : CUDART_INF_F));
        kmax[0] = fmaxf(kmax[0], ((row_end > row_begin + 96 + rgrp) ? kf8[0] : -CUDART_INF_F));
        kmin[0] = fminf(kmin[0], ((row_end > row_begin + 96 + rgrp) ? kf8[0] : CUDART_INF_F));
        qmax[1] = fmaxf(qmax[1], ((row_end > row_begin + 96 + rgrp) ? qf8[1] : -CUDART_INF_F));
        qmin[1] = fminf(qmin[1], ((row_end > row_begin + 96 + rgrp) ? qf8[1] : CUDART_INF_F));
        kmax[1] = fmaxf(kmax[1], ((row_end > row_begin + 96 + rgrp) ? kf8[1] : -CUDART_INF_F));
        kmin[1] = fminf(kmin[1], ((row_end > row_begin + 96 + rgrp) ? kf8[1] : CUDART_INF_F));
        qmax[2] = fmaxf(qmax[2], ((row_end > row_begin + 96 + rgrp) ? qf8[2] : -CUDART_INF_F));
        qmin[2] = fminf(qmin[2], ((row_end > row_begin + 96 + rgrp) ? qf8[2] : CUDART_INF_F));
        kmax[2] = fmaxf(kmax[2], ((row_end > row_begin + 96 + rgrp) ? kf8[2] : -CUDART_INF_F));
        kmin[2] = fminf(kmin[2], ((row_end > row_begin + 96 + rgrp) ? kf8[2] : CUDART_INF_F));
        qmax[3] = fmaxf(qmax[3], ((row_end > row_begin + 96 + rgrp) ? qf8[3] : -CUDART_INF_F));
        qmin[3] = fminf(qmin[3], ((row_end > row_begin + 96 + rgrp) ? qf8[3] : CUDART_INF_F));
        kmax[3] = fmaxf(kmax[3], ((row_end > row_begin + 96 + rgrp) ? kf8[3] : -CUDART_INF_F));
        kmin[3] = fminf(kmin[3], ((row_end > row_begin + 96 + rgrp) ? kf8[3] : CUDART_INF_F));
        qmax[4] = fmaxf(qmax[4], ((row_end > row_begin + 96 + rgrp) ? qf8[4] : -CUDART_INF_F));
        qmin[4] = fminf(qmin[4], ((row_end > row_begin + 96 + rgrp) ? qf8[4] : CUDART_INF_F));
        kmax[4] = fmaxf(kmax[4], ((row_end > row_begin + 96 + rgrp) ? kf8[4] : -CUDART_INF_F));
        kmin[4] = fminf(kmin[4], ((row_end > row_begin + 96 + rgrp) ? kf8[4] : CUDART_INF_F));
        qmax[5] = fmaxf(qmax[5], ((row_end > row_begin + 96 + rgrp) ? qf8[5] : -CUDART_INF_F));
        qmin[5] = fminf(qmin[5], ((row_end > row_begin + 96 + rgrp) ? qf8[5] : CUDART_INF_F));
        kmax[5] = fmaxf(kmax[5], ((row_end > row_begin + 96 + rgrp) ? kf8[5] : -CUDART_INF_F));
        kmin[5] = fminf(kmin[5], ((row_end > row_begin + 96 + rgrp) ? kf8[5] : CUDART_INF_F));
        qmax[6] = fmaxf(qmax[6], ((row_end > row_begin + 96 + rgrp) ? qf8[6] : -CUDART_INF_F));
        qmin[6] = fminf(qmin[6], ((row_end > row_begin + 96 + rgrp) ? qf8[6] : CUDART_INF_F));
        kmax[6] = fmaxf(kmax[6], ((row_end > row_begin + 96 + rgrp) ? kf8[6] : -CUDART_INF_F));
        kmin[6] = fminf(kmin[6], ((row_end > row_begin + 96 + rgrp) ? kf8[6] : CUDART_INF_F));
        qmax[7] = fmaxf(qmax[7], ((row_end > row_begin + 96 + rgrp) ? qf8[7] : -CUDART_INF_F));
        qmin[7] = fminf(qmin[7], ((row_end > row_begin + 96 + rgrp) ? qf8[7] : CUDART_INF_F));
        kmax[7] = fmaxf(kmax[7], ((row_end > row_begin + 96 + rgrp) ? kf8[7] : -CUDART_INF_F));
        kmin[7] = fminf(kmin[7], ((row_end > row_begin + 96 + rgrp) ? kf8[7] : CUDART_INF_F));
        qf8[0] = _vec_load_14[0];
        kf8[0] = _vec_load_15[0];
        qf8[1] = _vec_load_14[1];
        kf8[1] = _vec_load_15[1];
        qf8[2] = _vec_load_14[2];
        kf8[2] = _vec_load_15[2];
        qf8[3] = _vec_load_14[3];
        kf8[3] = _vec_load_15[3];
        qf8[4] = _vec_load_14[4];
        kf8[4] = _vec_load_15[4];
        qf8[5] = _vec_load_14[5];
        kf8[5] = _vec_load_15[5];
        qf8[6] = _vec_load_14[6];
        kf8[6] = _vec_load_15[6];
        qf8[7] = _vec_load_14[7];
        kf8[7] = _vec_load_15[7];
        ksum[0] = ksum[0] + ((row_end > row_begin + 112 + rgrp) ? kf8[0] : 0.0f);
        qsum[0] = qsum[0] + ((row_end > row_begin + 112 + rgrp) ? qf8[0] : 0.0f);
        ksum[1] = ksum[1] + ((row_end > row_begin + 112 + rgrp) ? kf8[1] : 0.0f);
        qsum[1] = qsum[1] + ((row_end > row_begin + 112 + rgrp) ? qf8[1] : 0.0f);
        ksum[2] = ksum[2] + ((row_end > row_begin + 112 + rgrp) ? kf8[2] : 0.0f);
        qsum[2] = qsum[2] + ((row_end > row_begin + 112 + rgrp) ? qf8[2] : 0.0f);
        ksum[3] = ksum[3] + ((row_end > row_begin + 112 + rgrp) ? kf8[3] : 0.0f);
        qsum[3] = qsum[3] + ((row_end > row_begin + 112 + rgrp) ? qf8[3] : 0.0f);
        ksum[4] = ksum[4] + ((row_end > row_begin + 112 + rgrp) ? kf8[4] : 0.0f);
        qsum[4] = qsum[4] + ((row_end > row_begin + 112 + rgrp) ? qf8[4] : 0.0f);
        ksum[5] = ksum[5] + ((row_end > row_begin + 112 + rgrp) ? kf8[5] : 0.0f);
        qsum[5] = qsum[5] + ((row_end > row_begin + 112 + rgrp) ? qf8[5] : 0.0f);
        ksum[6] = ksum[6] + ((row_end > row_begin + 112 + rgrp) ? kf8[6] : 0.0f);
        qsum[6] = qsum[6] + ((row_end > row_begin + 112 + rgrp) ? qf8[6] : 0.0f);
        ksum[7] = ksum[7] + ((row_end > row_begin + 112 + rgrp) ? kf8[7] : 0.0f);
        qsum[7] = qsum[7] + ((row_end > row_begin + 112 + rgrp) ? qf8[7] : 0.0f);
        qmax[0] = fmaxf(qmax[0], ((row_end > row_begin + 112 + rgrp) ? qf8[0] : -CUDART_INF_F));
        qmin[0] = fminf(qmin[0], ((row_end > row_begin + 112 + rgrp) ? qf8[0] : CUDART_INF_F));
        kmax[0] = fmaxf(kmax[0], ((row_end > row_begin + 112 + rgrp) ? kf8[0] : -CUDART_INF_F));
        kmin[0] = fminf(kmin[0], ((row_end > row_begin + 112 + rgrp) ? kf8[0] : CUDART_INF_F));
        qmax[1] = fmaxf(qmax[1], ((row_end > row_begin + 112 + rgrp) ? qf8[1] : -CUDART_INF_F));
        qmin[1] = fminf(qmin[1], ((row_end > row_begin + 112 + rgrp) ? qf8[1] : CUDART_INF_F));
        kmax[1] = fmaxf(kmax[1], ((row_end > row_begin + 112 + rgrp) ? kf8[1] : -CUDART_INF_F));
        kmin[1] = fminf(kmin[1], ((row_end > row_begin + 112 + rgrp) ? kf8[1] : CUDART_INF_F));
        qmax[2] = fmaxf(qmax[2], ((row_end > row_begin + 112 + rgrp) ? qf8[2] : -CUDART_INF_F));
        qmin[2] = fminf(qmin[2], ((row_end > row_begin + 112 + rgrp) ? qf8[2] : CUDART_INF_F));
        kmax[2] = fmaxf(kmax[2], ((row_end > row_begin + 112 + rgrp) ? kf8[2] : -CUDART_INF_F));
        kmin[2] = fminf(kmin[2], ((row_end > row_begin + 112 + rgrp) ? kf8[2] : CUDART_INF_F));
        qmax[3] = fmaxf(qmax[3], ((row_end > row_begin + 112 + rgrp) ? qf8[3] : -CUDART_INF_F));
        qmin[3] = fminf(qmin[3], ((row_end > row_begin + 112 + rgrp) ? qf8[3] : CUDART_INF_F));
        kmax[3] = fmaxf(kmax[3], ((row_end > row_begin + 112 + rgrp) ? kf8[3] : -CUDART_INF_F));
        kmin[3] = fminf(kmin[3], ((row_end > row_begin + 112 + rgrp) ? kf8[3] : CUDART_INF_F));
        qmax[4] = fmaxf(qmax[4], ((row_end > row_begin + 112 + rgrp) ? qf8[4] : -CUDART_INF_F));
        qmin[4] = fminf(qmin[4], ((row_end > row_begin + 112 + rgrp) ? qf8[4] : CUDART_INF_F));
        kmax[4] = fmaxf(kmax[4], ((row_end > row_begin + 112 + rgrp) ? kf8[4] : -CUDART_INF_F));
        kmin[4] = fminf(kmin[4], ((row_end > row_begin + 112 + rgrp) ? kf8[4] : CUDART_INF_F));
        qmax[5] = fmaxf(qmax[5], ((row_end > row_begin + 112 + rgrp) ? qf8[5] : -CUDART_INF_F));
        qmin[5] = fminf(qmin[5], ((row_end > row_begin + 112 + rgrp) ? qf8[5] : CUDART_INF_F));
        kmax[5] = fmaxf(kmax[5], ((row_end > row_begin + 112 + rgrp) ? kf8[5] : -CUDART_INF_F));
        kmin[5] = fminf(kmin[5], ((row_end > row_begin + 112 + rgrp) ? kf8[5] : CUDART_INF_F));
        qmax[6] = fmaxf(qmax[6], ((row_end > row_begin + 112 + rgrp) ? qf8[6] : -CUDART_INF_F));
        qmin[6] = fminf(qmin[6], ((row_end > row_begin + 112 + rgrp) ? qf8[6] : CUDART_INF_F));
        kmax[6] = fmaxf(kmax[6], ((row_end > row_begin + 112 + rgrp) ? kf8[6] : -CUDART_INF_F));
        kmin[6] = fminf(kmin[6], ((row_end > row_begin + 112 + rgrp) ? kf8[6] : CUDART_INF_F));
        qmax[7] = fmaxf(qmax[7], ((row_end > row_begin + 112 + rgrp) ? qf8[7] : -CUDART_INF_F));
        qmin[7] = fminf(qmin[7], ((row_end > row_begin + 112 + rgrp) ? qf8[7] : CUDART_INF_F));
        kmax[7] = fmaxf(kmax[7], ((row_end > row_begin + 112 + rgrp) ? kf8[7] : -CUDART_INF_F));
        kmin[7] = fminf(kmin[7], ((row_end > row_begin + 112 + rgrp) ? kf8[7] : CUDART_INF_F));
        for (int e_1 = 0; e_1 < 8; e_1++) {
            red_q[rgrp * 128 + col + e_1] = qsum[e_1];
            red_sum[rgrp * 128 + col + e_1] = ksum[e_1];
            red_qmax[rgrp * 128 + col + e_1] = qmax[e_1];
            red_qmin[rgrp * 128 + col + e_1] = qmin[e_1];
            red_kmax[rgrp * 128 + col + e_1] = kmax[e_1];
            red_kmin[rgrp * 128 + col + e_1] = kmin[e_1];
        }
        __syncthreads();
        int direct = ((num_qblocks == 1) ? 1 : 0);
        if (tid < 128) {
            float qtot = 0.0f;
            float qmx = -CUDART_INF_F;
            float qmn = CUDART_INF_F;
            float kmx = -CUDART_INF_F;
            float kmn = CUDART_INF_F;
            for (int g = 0; g < 16; g++) {
                qtot = qtot + red_q[g * 128 + tid];
                qmx = fmaxf(qmx, red_qmax[g * 128 + tid]);
                qmn = fminf(qmn, red_qmin[g * 128 + tid]);
                kmx = fmaxf(kmx, red_kmax[g * 128 + tid]);
                kmn = fminf(kmn, red_kmin[g * 128 + tid]);
            }
            int rows = row_end - row_begin;
            float qmean = __fdiv_rn(qtot, (float)rows);
            *(reinterpret_cast<float*>(q_mean + ((tq * num_heads + head) * 128 + tid)) + (0)) = qmean;
            *(reinterpret_cast<float*>(k_minmax + (work * 256 + tid)) + (0)) = kmx;
            *(reinterpret_cast<float*>(k_minmax + (work * 256 + 128 + tid)) + (0)) = kmn;
            float qdev = fmaxf(fabsf(qmx - qmean), fabsf(qmn - qmean));
            float qmabs = fabsf(qmean);
            float _shfl_xor_sync_16 = __shfl_xor_sync(0xFFFFFFFF, qdev, 16);
            float _max_0 = fmaxf(qdev, _shfl_xor_sync_16);
            float _shfl_xor_sync_17 = __shfl_xor_sync(0xFFFFFFFF, _max_0, 8);
            float _max_1 = fmaxf(_max_0, _shfl_xor_sync_17);
            float _shfl_xor_sync_18 = __shfl_xor_sync(0xFFFFFFFF, _max_1, 4);
            float _max_2 = fmaxf(_max_1, _shfl_xor_sync_18);
            float _shfl_xor_sync_19 = __shfl_xor_sync(0xFFFFFFFF, _max_2, 2);
            float _max_3 = fmaxf(_max_2, _shfl_xor_sync_19);
            float _shfl_xor_sync_20 = __shfl_xor_sync(0xFFFFFFFF, _max_3, 1);
            qdev = fmaxf(_max_3, _shfl_xor_sync_20);
            float _shfl_xor_sync_21 = __shfl_xor_sync(0xFFFFFFFF, qmabs, 16);
            float _max_4 = fmaxf(qmabs, _shfl_xor_sync_21);
            float _shfl_xor_sync_22 = __shfl_xor_sync(0xFFFFFFFF, _max_4, 8);
            float _max_5 = fmaxf(_max_4, _shfl_xor_sync_22);
            float _shfl_xor_sync_23 = __shfl_xor_sync(0xFFFFFFFF, _max_5, 4);
            float _max_6 = fmaxf(_max_5, _shfl_xor_sync_23);
            float _shfl_xor_sync_24 = __shfl_xor_sync(0xFFFFFFFF, _max_6, 2);
            float _max_7 = fmaxf(_max_6, _shfl_xor_sync_24);
            float _shfl_xor_sync_25 = __shfl_xor_sync(0xFFFFFFFF, _max_7, 1);
            qmabs = fmaxf(_max_7, _shfl_xor_sync_25);
            if (lane == 0) {
                red_small[warp] = qdev;
                red_small[4 + warp] = qmabs;
            }
            if (direct == 0) {
                float total = 0.0f;
                for (int g_1 = 0; g_1 < 16; g_1++) {
                    total = total + red_sum[g_1 * 128 + tid];
                }
                *(reinterpret_cast<float*>(partials + (work * 128 + tid)) + (0)) = total;
            }
        }
        __syncthreads();
        if (tid == 0) {
            float qa = fmaxf(fmaxf(red_small[0], red_small[1]), fmaxf(red_small[2], red_small[3]));
            float qma = fmaxf(fmaxf(red_small[4], red_small[5]), fmaxf(red_small[6], red_small[7]));
            *(reinterpret_cast<float*>(q_amax + (tq * num_heads + head)) + (0)) = fmaxf(qa, 1e-12f);
            *(reinterpret_cast<float*>(qm_amax + (tq * num_heads + head)) + (0)) = fmaxf(qma, 1e-12f);
            if (direct == 0) {
                unsigned int old;
                asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], %2;"
                    : "=r"(old) : "l"(&counters[batch * num_heads + head]), "r"(static_cast<uint32_t>(1)) : "memory");
                int last = (((int)old == num_qblocks - 1) ? 1 : 0);
                flag[0] = last;
                if (last == 1) {
                    *(reinterpret_cast<unsigned int*>(counters + (batch * num_heads + head)) + (0)) = (unsigned int)0;
                }
            } else {
                flag[0] = 1;
            }
        }
        __syncthreads();
        int is_last = flag[0];
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
                    int t0 = batch * num_qblocks;
                    int t1 = t0 + num_qblocks;
                    #pragma unroll 1
                    for (int tt = t0; tt < t1; tt += 4) {
                        for (int j = 0; j < 4; j++) {
                            int ttj = ((t1 > tt + j) ? tt + j : t1 - 1);
                            fold_v[j] = partials[(ttj * num_heads + head) * 128 + tid];
                        }
                        for (int j_1 = 0; j_1 < 4; j_1++) {
                            float fv = ((t1 > tt + j_1) ? fold_v[j_1] : 0.0f);
                            acc_a[0] = acc_a[0] + fv;
                        }
                    }
                }
                float lenf = (float)seq_len;
                float mk = __fdiv_rn(acc_a[0], lenf);
                *(reinterpret_cast<float*>(mean_k + ((batch * num_heads + head) * 128 + tid)) + (0)) = mk;
                mean_s[tid] = mk;
            }
            __syncthreads();
            int kt0 = batch * num_qblocks;
            int kt1 = kt0 + num_qblocks;
            float kdev4[4];
            #pragma unroll 1
            for (int tt_1 = kt0 + warp; tt_1 < kt1; tt_1 += 8) {
                int kbase_w = (tt_1 * num_heads + head) * 256;
                for (int j_2 = 0; j_2 < 4; j_2++) {
                    int ch = lane + j_2 * 32;
                    float mkc = mean_s[ch];
                    kdev4[j_2] = fmaxf(fabsf(k_minmax[kbase_w + ch] - mkc), fabsf(k_minmax[kbase_w + 128 + ch] - mkc));
                }
                float kdev = fmaxf(fmaxf(kdev4[0], kdev4[1]), fmaxf(kdev4[2], kdev4[3]));
                float _shfl_xor_sync_26 = __shfl_xor_sync(0xFFFFFFFF, kdev, 16);
                float _max_8 = fmaxf(kdev, _shfl_xor_sync_26);
                float _shfl_xor_sync_27 = __shfl_xor_sync(0xFFFFFFFF, _max_8, 8);
                float _max_9 = fmaxf(_max_8, _shfl_xor_sync_27);
                float _shfl_xor_sync_28 = __shfl_xor_sync(0xFFFFFFFF, _max_9, 4);
                float _max_10 = fmaxf(_max_9, _shfl_xor_sync_28);
                float _shfl_xor_sync_29 = __shfl_xor_sync(0xFFFFFFFF, _max_10, 2);
                float _max_11 = fmaxf(_max_10, _shfl_xor_sync_29);
                float _shfl_xor_sync_30 = __shfl_xor_sync(0xFFFFFFFF, _max_11, 1);
                kdev = fmaxf(_max_11, _shfl_xor_sync_30);
                if (lane == 0) {
                    *(reinterpret_cast<float*>(k_amax + (head * num_kblocks + tt_1)) + (0)) = fmaxf(kdev, 1e-12f);
                }
            }
        }
        __syncthreads();
    }
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
}

} // extern "C"
