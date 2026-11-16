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

__global__ __launch_bounds__(256, LAUNCH_MIN_BLOCKS) void
kernel_cake_dsa_h64_det_nope_train_9c5305b6ec6841752260(unsigned long long* __restrict__ cnt64, unsigned int* __restrict__ row_pairs, int* __restrict__ e_lat, int* dkv_acc, int* __restrict__ redo_mark, int* __restrict__ redo_rows, unsigned int* __restrict__ det_globals, int num_rows, int dkv_stride, int capacity, int min_bump, int bump_margin)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // === Task calls (declaration order) ===
    int row = blockIdx.x * 8 + warp;
    int lane_1 = lane;
    if (row < num_rows) {
        int word0 = e_lat[row];
        int e = word0 << 16 >> 16;
        int flagged = 0;
        int nonfinite = 0;
        unsigned long long worst = 0ull;
        if (e != 127) {
            unsigned long long cq_0 = cnt64[(long long)row * 4ll];
            if (cq_0 >> 63ull != 0ull) {
                nonfinite = 1;
            }
            unsigned long long _max_0 = ((worst) > (cq_0 & 9223372036854775807ull) ? (worst) : (cq_0 & 9223372036854775807ull));
            worst = _max_0;
            unsigned long long cq_1 = cnt64[(long long)row * 4ll + 1ll];
            if (cq_1 >> 63ull != 0ull) {
                nonfinite = 1;
            }
            unsigned long long _max_1 = ((worst) > (cq_1 & 9223372036854775807ull) ? (worst) : (cq_1 & 9223372036854775807ull));
            worst = _max_1;
            unsigned long long cq_2 = cnt64[(long long)row * 4ll + 2ll];
            if (cq_2 >> 63ull != 0ull) {
                nonfinite = 1;
            }
            unsigned long long _max_2 = ((worst) > (cq_2 & 9223372036854775807ull) ? (worst) : (cq_2 & 9223372036854775807ull));
            worst = _max_2;
            unsigned long long cq_3 = cnt64[(long long)row * 4ll + 3ll];
            if (cq_3 >> 63ull != 0ull) {
                nonfinite = 1;
            }
            unsigned long long _max_3 = ((worst) > (cq_3 & 9223372036854775807ull) ? (worst) : (cq_3 & 9223372036854775807ull));
            worst = _max_3;
            unsigned long long slack = (unsigned long long)row_pairs[row] << (unsigned long long)(word0 >> 16);
            worst = worst + slack;
            if (nonfinite == 0 && worst >= 2147483648ull) {
                flagged = 1;
            }
        }
        __syncwarp();
        if (flagged != 0) {
            unsigned int hi = (unsigned int)(worst >> 32ull);
            unsigned int lo = (unsigned int)(worst & 4294967295ull);
            int _count_leading_zeros_4 = __clz(lo);
            int bits = 32 - _count_leading_zeros_4;
            if (hi != 0) {
                int _count_leading_zeros_5 = __clz(hi);
                bits = 64 - _count_leading_zeros_5;
            }
            int bump = ((bits - 30 + bump_margin) > (min_bump) ? (bits - 30 + bump_margin) : (min_bump));
            if (lane_1 == 0) {
                int _min_6 = ((e + bump) < (126) ? (e + bump) : (126));
                e_lat[row] = _min_6 & 65535 | word0 >> 16 << 16;
                unsigned int slot = atomicAdd(&det_globals[8], 1);
                if ((int)slot < capacity) {
                    redo_rows[(int)slot] = row;
                }
            }
            int zero4[4];
            zero4[0] = 0;
            zero4[1] = 0;
            zero4[2] = 0;
            zero4[3] = 0;
            long long lbase = (long long)row * (long long)dkv_stride;
            {
                int4 _iv4 = make_int4(zero4[0 + 0], zero4[0 + 1], zero4[0 + 2], zero4[0 + 3]);
                *reinterpret_cast<int4*>(dkv_acc + lbase + (long long)(lane_1 * 4)) = _iv4;
            }
            {
                int4 _iv4 = make_int4(zero4[0 + 0], zero4[0 + 1], zero4[0 + 2], zero4[0 + 3]);
                *reinterpret_cast<int4*>(dkv_acc + lbase + (long long)(128 + lane_1 * 4)) = _iv4;
            }
            {
                int4 _iv4 = make_int4(zero4[0 + 0], zero4[0 + 1], zero4[0 + 2], zero4[0 + 3]);
                *reinterpret_cast<int4*>(dkv_acc + lbase + (long long)(256 + lane_1 * 4)) = _iv4;
            }
            {
                int4 _iv4 = make_int4(zero4[0 + 0], zero4[0 + 1], zero4[0 + 2], zero4[0 + 3]);
                *reinterpret_cast<int4*>(dkv_acc + lbase + (long long)(384 + lane_1 * 4)) = _iv4;
            }
            if (lane_1 < 4) {
                cnt64[(long long)row * 4ll + (long long)lane_1] = 0ull;
            }
        }
        if (lane_1 == 0) {
            redo_mark[row] = flagged;
            if (nonfinite != 0) {
                asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(det_globals) + (9))), "r"(static_cast<unsigned int>(1)) : "memory");
            }
        }
    }
}

} // extern "C"
