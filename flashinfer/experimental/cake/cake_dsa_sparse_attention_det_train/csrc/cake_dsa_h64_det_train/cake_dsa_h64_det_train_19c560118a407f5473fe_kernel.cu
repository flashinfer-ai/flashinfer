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

#define LAUNCH_MIN_BLOCKS 1

#include <math_constants.h>

extern "C" {

__global__ __launch_bounds__(128, LAUNCH_MIN_BLOCKS) void
kernel_cake_dsa_h64_det_train_19c560118a407f5473fe(int* __restrict__ indices, int* __restrict__ topk_length, int* __restrict__ key_scratch, int* __restrict__ pass_counts, int num_tokens, int topk, int idx_stride, int indices_offset, int has_topk_length, int token_base, int token_step, int pass_lo, int pass_hi)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // === Task calls (declaration order) ===
    int rowi = blockIdx.x * 4 + warp;
    int lane_0 = lane;
    if (rowi < num_tokens) {
        int token = token_base + token_step * rowi;
        int active = topk;
        if (has_topk_length != 0) {
            int _max_1 = ((topk_length[token]) > (0) ? (topk_length[token]) : (0));
            int _min_0 = ((_max_1) < (topk) ? (_max_1) : (topk));
            active = _min_0;
        }
        long long row_base = (long long)indices_offset + (long long)token * (long long)idx_stride;
        long long out_base = (long long)rowi * (long long)topk;
        int count = 0;
        #pragma unroll 1
        for (int cblk = 0; cblk < (active + 255) / 256; cblk++) {
            int cpos = cblk * 256 + lane_0 * 8;
            int kv8[8];
            int cwhole = (int)(active >= cblk * 256 + 256 && ((idx_stride | indices_offset) & 7) == 0);
            if (cwhole != 0) {
                int _vec_load_0[8];
                {
                    uint32_t _iv_2_0;
                    uint32_t _iv_2_1;
                    uint32_t _iv_2_2;
                    uint32_t _iv_2_3;
                    uint32_t _iv_2_4;
                    uint32_t _iv_2_5;
                    uint32_t _iv_2_6;
                    uint32_t _iv_2_7;
                    asm volatile("ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];" : "=r"(_iv_2_0), "=r"(_iv_2_1), "=r"(_iv_2_2), "=r"(_iv_2_3), "=r"(_iv_2_4), "=r"(_iv_2_5), "=r"(_iv_2_6), "=r"(_iv_2_7) : "l"((const void*)(indices + (row_base + (long long)cpos) + (0))) : "memory");
                    _vec_load_0[0 + 0] = (int32_t)_iv_2_0;
                    _vec_load_0[0 + 1] = (int32_t)_iv_2_1;
                    _vec_load_0[0 + 2] = (int32_t)_iv_2_2;
                    _vec_load_0[0 + 3] = (int32_t)_iv_2_3;
                    _vec_load_0[0 + 4] = (int32_t)_iv_2_4;
                    _vec_load_0[0 + 5] = (int32_t)_iv_2_5;
                    _vec_load_0[0 + 6] = (int32_t)_iv_2_6;
                    _vec_load_0[0 + 7] = (int32_t)_iv_2_7;
                }
                kv8[0] = _vec_load_0[0];
                kv8[1] = _vec_load_0[1];
                kv8[2] = _vec_load_0[2];
                kv8[3] = _vec_load_0[3];
                kv8[4] = _vec_load_0[4];
                kv8[5] = _vec_load_0[5];
                kv8[6] = _vec_load_0[6];
                kv8[7] = _vec_load_0[7];
            } else {
                int sv_7_0 = -1;
                if (active > cpos) {
                    sv_7_0 = indices[row_base + (long long)cpos];
                }
                kv8[0] = sv_7_0;
                int sv_7_1 = -1;
                if (active > cpos + 1) {
                    sv_7_1 = indices[row_base + (long long)(cpos + 1)];
                }
                kv8[1] = sv_7_1;
                int sv_7_2 = -1;
                if (active > cpos + 2) {
                    sv_7_2 = indices[row_base + (long long)(cpos + 2)];
                }
                kv8[2] = sv_7_2;
                int sv_7_3 = -1;
                if (active > cpos + 3) {
                    sv_7_3 = indices[row_base + (long long)(cpos + 3)];
                }
                kv8[3] = sv_7_3;
                int sv_7_4 = -1;
                if (active > cpos + 4) {
                    sv_7_4 = indices[row_base + (long long)(cpos + 4)];
                }
                kv8[4] = sv_7_4;
                int sv_7_5 = -1;
                if (active > cpos + 5) {
                    sv_7_5 = indices[row_base + (long long)(cpos + 5)];
                }
                kv8[5] = sv_7_5;
                int sv_7_6 = -1;
                if (active > cpos + 6) {
                    sv_7_6 = indices[row_base + (long long)(cpos + 6)];
                }
                kv8[6] = sv_7_6;
                int sv_7_7 = -1;
                if (active > cpos + 7) {
                    sv_7_7 = indices[row_base + (long long)(cpos + 7)];
                }
                kv8[7] = sv_7_7;
            }
            unsigned int hit = 0;
            if (kv8[0] >= pass_lo && kv8[0] < pass_hi) {
                hit = hit + 1;
            }
            if (kv8[1] >= pass_lo && kv8[1] < pass_hi) {
                hit = hit + 1;
            }
            if (kv8[2] >= pass_lo && kv8[2] < pass_hi) {
                hit = hit + 1;
            }
            if (kv8[3] >= pass_lo && kv8[3] < pass_hi) {
                hit = hit + 1;
            }
            if (kv8[4] >= pass_lo && kv8[4] < pass_hi) {
                hit = hit + 1;
            }
            if (kv8[5] >= pass_lo && kv8[5] < pass_hi) {
                hit = hit + 1;
            }
            if (kv8[6] >= pass_lo && kv8[6] < pass_hi) {
                hit = hit + 1;
            }
            if (kv8[7] >= pass_lo && kv8[7] < pass_hi) {
                hit = hit + 1;
            }
            uint32_t incl = hit;
            { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, incl, 1, 32); if (lane >= 1) { incl += peer; } }
            { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, incl, 2, 32); if (lane >= 2) { incl += peer; } }
            { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, incl, 4, 32); if (lane >= 4) { incl += peer; } }
            { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, incl, 8, 32); if (lane >= 8) { incl += peer; } }
            { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, incl, 16, 32); if (lane >= 16) { incl += peer; } }
            int wpos = count + (int)(incl - hit);
            if (kv8[0] >= pass_lo && kv8[0] < pass_hi) {
                key_scratch[out_base + (long long)wpos] = kv8[0];
                wpos = wpos + 1;
            }
            if (kv8[1] >= pass_lo && kv8[1] < pass_hi) {
                key_scratch[out_base + (long long)wpos] = kv8[1];
                wpos = wpos + 1;
            }
            if (kv8[2] >= pass_lo && kv8[2] < pass_hi) {
                key_scratch[out_base + (long long)wpos] = kv8[2];
                wpos = wpos + 1;
            }
            if (kv8[3] >= pass_lo && kv8[3] < pass_hi) {
                key_scratch[out_base + (long long)wpos] = kv8[3];
                wpos = wpos + 1;
            }
            if (kv8[4] >= pass_lo && kv8[4] < pass_hi) {
                key_scratch[out_base + (long long)wpos] = kv8[4];
                wpos = wpos + 1;
            }
            if (kv8[5] >= pass_lo && kv8[5] < pass_hi) {
                key_scratch[out_base + (long long)wpos] = kv8[5];
                wpos = wpos + 1;
            }
            if (kv8[6] >= pass_lo && kv8[6] < pass_hi) {
                key_scratch[out_base + (long long)wpos] = kv8[6];
                wpos = wpos + 1;
            }
            if (kv8[7] >= pass_lo && kv8[7] < pass_hi) {
                key_scratch[out_base + (long long)wpos] = kv8[7];
                wpos = wpos + 1;
            }
            unsigned int _shfl_sync_3 = __shfl_sync(0xFFFFFFFF, incl, 31);
            count = count + (int)_shfl_sync_3;
        }
        if (lane_0 == 0) {
            pass_counts[rowi] = count;
        }
    }
}

} // extern "C"
