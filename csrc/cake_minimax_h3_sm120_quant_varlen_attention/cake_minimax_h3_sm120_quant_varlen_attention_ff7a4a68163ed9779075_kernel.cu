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
#define SMEM_RED_MAX_OFF 8192
#define SMEM_RED_MAX_STAGE_BYTES 8192
#define SMEM_RED_MAX_STRIDE 8192
#define SMEM_FLAG_OFF 16384
#define SMEM_FLAG_STAGE_BYTES 4
#define SMEM_FLAG_STRIDE 4
#define SMEM_TOTAL 16512
#define THREADS 256
#define LAUNCH_MIN_BLOCKS 2

#include <math_constants.h>

extern "C" {

__global__ __launch_bounds__(256, LAUNCH_MIN_BLOCKS) void
kernel_cake_minimax_h3_sm120_quant_varlen_attention_ff7a4a68163ed9779075(__nv_bfloat16* __restrict__ K, __nv_bfloat16* __restrict__ V, int* __restrict__ tile_table, int* __restrict__ seg_begin, int* __restrict__ seg_len, int* __restrict__ seg_tile_begin, float* __restrict__ partials, unsigned int* __restrict__ counters, float* __restrict__ mean_k, float* __restrict__ v_scale, int num_tiles, int num_heads)
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
    int* flag = reinterpret_cast<int*>(smem_raw + 16384);
    const int flag_addr = smem + 16384;

    // === Task calls (declaration order) ===
    int col = (tid & 15) * 8;
    int rgrp = tid >> 4;
    #pragma unroll 1
    for (int work = bid; work < num_tiles * num_heads; work += num_bids) {
        int tile = work / num_heads;
        int head = work % num_heads;
        int seg = tile_table[2 * tile];
        int tile_in_seg = tile_table[2 * tile + 1];
        int sb = seg_begin[seg];
        int length = seg_len[seg];
        int t0 = seg_tile_begin[seg];
        int t1 = seg_tile_begin[seg + 1];
        int row_begin = sb + tile_in_seg * 128;
        int seg_end = sb + length;
        int row_end = ((seg_end > row_begin + 128) ? row_begin + 128 : seg_end);
        float ksum[8];
        float vmax[8];
        for (int e = 0; e < 8; e++) {
            ksum[e] = 0.0f;
            vmax[e] = 0.0f;
        }
        #pragma unroll 1
        for (int row = row_begin + rgrp; row < row_end; row += 16) {
            int base = (row * num_heads + head) * 128 + col;
            float _vec_load_0[8];
            {
                const uint4* _vptr_0 = reinterpret_cast<const uint4*>(K + base + 0);
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
                const uint4* _vptr_1 = reinterpret_cast<const uint4*>(V + base + 0);
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
            for (int e_1 = 0; e_1 < 8; e_1++) {
                float kf = _vec_load_0[e_1];
                float vf = _vec_load_1[e_1];
                ksum[e_1] = ksum[e_1] + kf;
                vmax[e_1] = fmaxf(vmax[e_1], fabsf(vf));
            }
        }
        for (int e_2 = 0; e_2 < 8; e_2++) {
            red_sum[rgrp * 128 + col + e_2] = ksum[e_2];
            red_max[rgrp * 128 + col + e_2] = vmax[e_2];
        }
        __syncthreads();
        int single = ((t1 - t0 == 1) ? 1 : 0);
        if (tid < 128) {
            float total = 0.0f;
            float vmx = 0.0f;
            for (int g = 0; g < 16; g++) {
                total = total + red_sum[g * 128 + tid];
                vmx = fmaxf(vmx, red_max[g * 128 + tid]);
            }
            if (single == 1) {
                float msum = 0.0f;
                float mmax = 0.0f;
                msum = msum + total;
                mmax = fmaxf(mmax, vmx);
                float _fdiv_2 = __fdiv_rn(msum, (float)length);
                *(reinterpret_cast<float*>(mean_k + ((seg * num_heads + head) * 128 + tid)) + (0)) = _fdiv_2;
                float _fdiv_3 = __fdiv_rn(fmaxf(mmax, 1e-12f), 448.0f);
                *(reinterpret_cast<float*>(v_scale + ((seg * num_heads + head) * 128 + tid)) + (0)) = _fdiv_3;
            } else {
                *(reinterpret_cast<float*>(partials + (work * 256 + tid)) + (0)) = total;
                *(reinterpret_cast<float*>(partials + (work * 256 + 128 + tid)) + (0)) = vmx;
            }
        }
        if (single == 0) {
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
            if (flag[0] == 1) {
                int chan = tid & 127;
                int is_max = tid >> 7;
                float acc[1];
                float fold_v[8];
                acc[0] = 0.0f;
                #pragma unroll 1
                for (int tt = t0; tt < t1; tt += 8) {
                    for (int j = 0; j < 8; j++) {
                        int ttj = ((t1 > tt + j) ? tt + j : t1 - 1);
                        fold_v[j] = partials[(ttj * num_heads + head) * 256 + is_max * 128 + chan];
                    }
                    for (int j_1 = 0; j_1 < 8; j_1++) {
                        if (t1 > tt + j_1) {
                            acc[0] = ((is_max == 0) ? acc[0] + fold_v[j_1] : fmaxf(acc[0], fold_v[j_1]));
                        }
                    }
                }
                if (is_max == 0) {
                    float _fdiv_4 = __fdiv_rn(acc[0], (float)length);
                    *(reinterpret_cast<float*>(mean_k + ((seg * num_heads + head) * 128 + chan)) + (0)) = _fdiv_4;
                } else {
                    float _fdiv_5 = __fdiv_rn(fmaxf(acc[0], 1e-12f), 448.0f);
                    *(reinterpret_cast<float*>(v_scale + ((seg * num_heads + head) * 128 + chan)) + (0)) = _fdiv_5;
                }
            }
        }
        __syncthreads();
    }
}

} // extern "C"
