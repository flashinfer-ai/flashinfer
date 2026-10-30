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

extern "C" {

__global__ __launch_bounds__(128) void
kernel_cake_lm_head_loss_c9dbb1bda1359978bdcc(__nv_bfloat16* __restrict__ rows, __nv_bfloat16* __restrict__ scale, long long* __restrict__ keys, int* __restrict__ offsets, __nv_bfloat16* __restrict__ out, float* __restrict__ partial, int ld_rows, int H, int V, int M, int T, int n_keys, int split, int span)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // === Task calls (declaration order) ===
    int col = blockIdx.y * 1024 + warp * 256 + lane * 8;
    int live = ((col < H) ? 1 : 0);
    int lo = blockIdx.x * span;
    if (lo < n_keys) {
        int _min_0 = ((span) < (n_keys - lo) ? (span) : (n_keys - lo));
        int hi = lo + _min_0;
        int _min_1 = ((split) < (n_keys - hi) ? (split) : (n_keys - hi));
        int limit = hi + _min_1;
        int start = lo;
        int prev_row = -1;
        int run = 0;
        if (lo > 0) {
            int first_row = (int)(keys[lo] >> 32);
            prev_row = (int)(keys[lo - 1] >> 32);
            if (prev_row == first_row) {
                int seg = 0;
                int left = lo;
                int _count_leading_zeros_2 = __clz(lo);
                #pragma unroll 1
                for (int _halving = 0; _halving < 32 - _count_leading_zeros_2; _halving++) {
                    int step = left >> 1;
                    if ((int)(keys[seg + step] >> 32) < first_row) {
                        seg = seg + step + 1;
                        left = left - step - 1;
                    } else {
                        left = step;
                    }
                }
                run = (lo - 1 - seg) % split;
                if (run + 1 != split) {
                    if ((int)(keys[hi - 1] >> 32) == first_row) {
                        start = lo + (split - 1 - run);
                        run = split - 1;
                    }
                }
            }
        }
        float acc[8];
        acc[0] = 0.0f;
        acc[1] = 0.0f;
        acc[2] = 0.0f;
        acc[3] = 0.0f;
        acc[4] = 0.0f;
        acc[5] = 0.0f;
        acc[6] = 0.0f;
        acc[7] = 0.0f;
        int active = 0;
        int done = 0;
        int head = 0;
        int later_part = 0;
        int open_row = 0;
        if (start >= hi) {
            done = 1;
        }
        #pragma unroll 1
        for (int base = start; base < limit; base += 32) {
            if (done != 0) {
                break;
            }
            int count = ((limit - base) < (32) ? (limit - base) : (32));
            int _min_3 = ((lane) < (count - 1) ? (lane) : (count - 1));
            int pos_l = base + _min_3;
            int pair_l = (((int)(keys[pos_l] & 2147483647)) < (M - 1) ? ((int)(keys[pos_l] & 2147483647)) : (M - 1));
            int token_lo = 0;
            int token_width = T;
            int _count_leading_zeros_4 = __clz(T - 1);
            #pragma unroll 1
            for (int _token_halving = 0; _token_halving < 32 - _count_leading_zeros_4; _token_halving++) {
                int token_half = token_width >> 1;
                if (offsets[token_lo + token_half] <= pair_l) {
                    token_lo = token_lo + token_half;
                }
                token_width = token_width - token_half;
            }
            int token_l = token_lo;
            float scale_l = (float)scale[pair_l];
            #pragma unroll 1
            for (int g = 0; g < count; g += 8) {
                int _min_5 = ((g) < (31) ? (g) : (31));
                int token_u_0 = __shfl_sync(0xFFFFFFFF, token_l, _min_5);
                unsigned int words[4];
                if (g < count && live != 0) {
                    {
                        const uint4* _vptr_6 = reinterpret_cast<const uint4*>(rows + (unsigned long long)token_u_0 * (unsigned long long)ld_rows + (unsigned long long)col);
                        uint4* _vdst_6 = reinterpret_cast<uint4*>(&words[0]);
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            _vdst_6[_blk] = _vptr_6[_blk];
                        }
                    }
                }
                int _min_7 = ((g) < (count - 1) ? (g) : (count - 1));
                int row_u_0 = (int)(keys[base + _min_7] >> 32);
                int _min_8 = ((g + 1) < (31) ? (g + 1) : (31));
                int token_u_1 = __shfl_sync(0xFFFFFFFF, token_l, _min_8);
                unsigned int words_0[4];
                if (g + 1 < count && live != 0) {
                    {
                        const uint4* _vptr_9 = reinterpret_cast<const uint4*>(rows + (unsigned long long)token_u_1 * (unsigned long long)ld_rows + (unsigned long long)col);
                        uint4* _vdst_9 = reinterpret_cast<uint4*>(&words_0[0]);
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            _vdst_9[_blk] = _vptr_9[_blk];
                        }
                    }
                }
                int _min_10 = ((g + 1) < (count - 1) ? (g + 1) : (count - 1));
                int row_u_1 = (int)(keys[base + _min_10] >> 32);
                int _min_11 = ((g + 2) < (31) ? (g + 2) : (31));
                int token_u_2 = __shfl_sync(0xFFFFFFFF, token_l, _min_11);
                unsigned int words_1[4];
                if (g + 2 < count && live != 0) {
                    {
                        const uint4* _vptr_12 = reinterpret_cast<const uint4*>(rows + (unsigned long long)token_u_2 * (unsigned long long)ld_rows + (unsigned long long)col);
                        uint4* _vdst_12 = reinterpret_cast<uint4*>(&words_1[0]);
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            _vdst_12[_blk] = _vptr_12[_blk];
                        }
                    }
                }
                int _min_13 = ((g + 2) < (count - 1) ? (g + 2) : (count - 1));
                int row_u_2 = (int)(keys[base + _min_13] >> 32);
                int _min_14 = ((g + 3) < (31) ? (g + 3) : (31));
                int token_u_3 = __shfl_sync(0xFFFFFFFF, token_l, _min_14);
                unsigned int words_2[4];
                if (g + 3 < count && live != 0) {
                    {
                        const uint4* _vptr_15 = reinterpret_cast<const uint4*>(rows + (unsigned long long)token_u_3 * (unsigned long long)ld_rows + (unsigned long long)col);
                        uint4* _vdst_15 = reinterpret_cast<uint4*>(&words_2[0]);
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            _vdst_15[_blk] = _vptr_15[_blk];
                        }
                    }
                }
                int _min_16 = ((g + 3) < (count - 1) ? (g + 3) : (count - 1));
                int row_u_3 = (int)(keys[base + _min_16] >> 32);
                int _min_17 = ((g + 4) < (31) ? (g + 4) : (31));
                int token_u_4 = __shfl_sync(0xFFFFFFFF, token_l, _min_17);
                unsigned int words_3[4];
                if (g + 4 < count && live != 0) {
                    {
                        const uint4* _vptr_18 = reinterpret_cast<const uint4*>(rows + (unsigned long long)token_u_4 * (unsigned long long)ld_rows + (unsigned long long)col);
                        uint4* _vdst_18 = reinterpret_cast<uint4*>(&words_3[0]);
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            _vdst_18[_blk] = _vptr_18[_blk];
                        }
                    }
                }
                int _min_19 = ((g + 4) < (count - 1) ? (g + 4) : (count - 1));
                int row_u_4 = (int)(keys[base + _min_19] >> 32);
                int _min_20 = ((g + 5) < (31) ? (g + 5) : (31));
                int token_u_5 = __shfl_sync(0xFFFFFFFF, token_l, _min_20);
                unsigned int words_4[4];
                if (g + 5 < count && live != 0) {
                    {
                        const uint4* _vptr_21 = reinterpret_cast<const uint4*>(rows + (unsigned long long)token_u_5 * (unsigned long long)ld_rows + (unsigned long long)col);
                        uint4* _vdst_21 = reinterpret_cast<uint4*>(&words_4[0]);
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            _vdst_21[_blk] = _vptr_21[_blk];
                        }
                    }
                }
                int _min_22 = ((g + 5) < (count - 1) ? (g + 5) : (count - 1));
                int row_u_5 = (int)(keys[base + _min_22] >> 32);
                int _min_23 = ((g + 6) < (31) ? (g + 6) : (31));
                int token_u_6 = __shfl_sync(0xFFFFFFFF, token_l, _min_23);
                unsigned int words_5[4];
                if (g + 6 < count && live != 0) {
                    {
                        const uint4* _vptr_24 = reinterpret_cast<const uint4*>(rows + (unsigned long long)token_u_6 * (unsigned long long)ld_rows + (unsigned long long)col);
                        uint4* _vdst_24 = reinterpret_cast<uint4*>(&words_5[0]);
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            _vdst_24[_blk] = _vptr_24[_blk];
                        }
                    }
                }
                int _min_25 = ((g + 6) < (count - 1) ? (g + 6) : (count - 1));
                int row_u_6 = (int)(keys[base + _min_25] >> 32);
                int _min_26 = ((g + 7) < (31) ? (g + 7) : (31));
                int token_u_7 = __shfl_sync(0xFFFFFFFF, token_l, _min_26);
                unsigned int words_6[4];
                if (g + 7 < count && live != 0) {
                    {
                        const uint4* _vptr_27 = reinterpret_cast<const uint4*>(rows + (unsigned long long)token_u_7 * (unsigned long long)ld_rows + (unsigned long long)col);
                        uint4* _vdst_27 = reinterpret_cast<uint4*>(&words_6[0]);
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            _vdst_27[_blk] = _vptr_27[_blk];
                        }
                    }
                }
                int _min_28 = ((g + 7) < (count - 1) ? (g + 7) : (count - 1));
                int row_u_7 = (int)(keys[base + _min_28] >> 32);
                int _min_29 = ((g) < (31) ? (g) : (31));
                float scale_u_0 = __shfl_sync(0xFFFFFFFF, scale_l, _min_29);
                if (g < count) {
                    if (done == 0) {
                        int same_0 = ((row_u_0 == prev_row) ? 1 : 0);
                        run = ((same_0 != 0) ? ((run + 1 == split) ? 0 : run + 1) : 0);
                        if (run == 0) {
                            if (active != 0) {
                                if (live != 0) {
                                    if (later_part != 0 || 1 - same_0 == 0) {
                                        int slot = 2 * (head / split) + (1 - (1 - same_0));
                                        {
                                            float4 _v4 = make_float4(acc[0 + 0], acc[0 + 1], acc[0 + 2], acc[0 + 3]);
                                            *reinterpret_cast<float4*>((partial + ((unsigned long long)slot * (unsigned long long)H + (unsigned long long)col)) + 0) = _v4;
                                        }
                                        {
                                            float4 _v4 = make_float4(acc[4 + 0], acc[4 + 1], acc[4 + 2], acc[4 + 3]);
                                            *reinterpret_cast<float4*>((partial + ((unsigned long long)slot * (unsigned long long)H + (unsigned long long)col + 4)) + 0) = _v4;
                                        }
                                    } else {
                                        unsigned int packed[4];
                                        #pragma unroll
                                        for (int _lp = 0; _lp < 4; _lp++) {
                                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc[_lp*2 + 0], acc[_lp*2+1 + 0]));
                                            packed[_lp] = *(uint32_t*)&_bf2;
                                        }
                                        reinterpret_cast<int4*>(out + ((unsigned long long)open_row * (unsigned long long)H + (unsigned long long)col))[0] = reinterpret_cast<int4*>(packed)[0];
                                    }
                                }
                            }
                            int ends_0 = ((base + g >= hi) ? 1 : 0);
                            ends_0 = ((row_u_0 < 0) ? 1 : ends_0);
                            ends_0 = ((row_u_0 >= V) ? 1 : ends_0);
                            if (ends_0 != 0) {
                                done = 1;
                                active = 0;
                            } else {
                                active = 1;
                                head = base + g;
                                later_part = same_0;
                                open_row = row_u_0;
                                acc[0] = 0.0f;
                                acc[1] = 0.0f;
                                acc[2] = 0.0f;
                                acc[3] = 0.0f;
                                acc[4] = 0.0f;
                                acc[5] = 0.0f;
                                acc[6] = 0.0f;
                                acc[7] = 0.0f;
                            }
                        }
                        if (active != 0) {
                            acc[0] = __fmaf_rn(scale_u_0, __uint_as_float(words[0] << 16), acc[0]);
                            acc[1] = __fmaf_rn(scale_u_0, __uint_as_float(words[0] & 4294901760u), acc[1]);
                            acc[2] = __fmaf_rn(scale_u_0, __uint_as_float(words[1] << 16), acc[2]);
                            acc[3] = __fmaf_rn(scale_u_0, __uint_as_float(words[1] & 4294901760u), acc[3]);
                            acc[4] = __fmaf_rn(scale_u_0, __uint_as_float(words[2] << 16), acc[4]);
                            acc[5] = __fmaf_rn(scale_u_0, __uint_as_float(words[2] & 4294901760u), acc[5]);
                            acc[6] = __fmaf_rn(scale_u_0, __uint_as_float(words[3] << 16), acc[6]);
                            acc[7] = __fmaf_rn(scale_u_0, __uint_as_float(words[3] & 4294901760u), acc[7]);
                        }
                        prev_row = row_u_0;
                    }
                }
                int _min_30 = ((g + 1) < (31) ? (g + 1) : (31));
                float scale_u_1 = __shfl_sync(0xFFFFFFFF, scale_l, _min_30);
                if (g + 1 < count) {
                    if (done == 0) {
                        int same_1 = ((row_u_1 == prev_row) ? 1 : 0);
                        run = ((same_1 != 0) ? ((run + 1 == split) ? 0 : run + 1) : 0);
                        if (run == 0) {
                            if (active != 0) {
                                if (live != 0) {
                                    if (later_part != 0 || 1 - same_1 == 0) {
                                        int slot_1 = 2 * (head / split) + (1 - (1 - same_1));
                                        {
                                            float4 _v4 = make_float4(acc[0 + 0], acc[0 + 1], acc[0 + 2], acc[0 + 3]);
                                            *reinterpret_cast<float4*>((partial + ((unsigned long long)slot_1 * (unsigned long long)H + (unsigned long long)col)) + 0) = _v4;
                                        }
                                        {
                                            float4 _v4 = make_float4(acc[4 + 0], acc[4 + 1], acc[4 + 2], acc[4 + 3]);
                                            *reinterpret_cast<float4*>((partial + ((unsigned long long)slot_1 * (unsigned long long)H + (unsigned long long)col + 4)) + 0) = _v4;
                                        }
                                    } else {
                                        unsigned int packed_1[4];
                                        #pragma unroll
                                        for (int _lp = 0; _lp < 4; _lp++) {
                                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc[_lp*2 + 0], acc[_lp*2+1 + 0]));
                                            packed_1[_lp] = *(uint32_t*)&_bf2;
                                        }
                                        reinterpret_cast<int4*>(out + ((unsigned long long)open_row * (unsigned long long)H + (unsigned long long)col))[0] = reinterpret_cast<int4*>(packed_1)[0];
                                    }
                                }
                            }
                            int ends_1 = ((base + g + 1 >= hi) ? 1 : 0);
                            ends_1 = ((row_u_1 < 0) ? 1 : ends_1);
                            ends_1 = ((row_u_1 >= V) ? 1 : ends_1);
                            if (ends_1 != 0) {
                                done = 1;
                                active = 0;
                            } else {
                                active = 1;
                                head = base + g + 1;
                                later_part = same_1;
                                open_row = row_u_1;
                                acc[0] = 0.0f;
                                acc[1] = 0.0f;
                                acc[2] = 0.0f;
                                acc[3] = 0.0f;
                                acc[4] = 0.0f;
                                acc[5] = 0.0f;
                                acc[6] = 0.0f;
                                acc[7] = 0.0f;
                            }
                        }
                        if (active != 0) {
                            acc[0] = __fmaf_rn(scale_u_1, __uint_as_float(words_0[0] << 16), acc[0]);
                            acc[1] = __fmaf_rn(scale_u_1, __uint_as_float(words_0[0] & 4294901760u), acc[1]);
                            acc[2] = __fmaf_rn(scale_u_1, __uint_as_float(words_0[1] << 16), acc[2]);
                            acc[3] = __fmaf_rn(scale_u_1, __uint_as_float(words_0[1] & 4294901760u), acc[3]);
                            acc[4] = __fmaf_rn(scale_u_1, __uint_as_float(words_0[2] << 16), acc[4]);
                            acc[5] = __fmaf_rn(scale_u_1, __uint_as_float(words_0[2] & 4294901760u), acc[5]);
                            acc[6] = __fmaf_rn(scale_u_1, __uint_as_float(words_0[3] << 16), acc[6]);
                            acc[7] = __fmaf_rn(scale_u_1, __uint_as_float(words_0[3] & 4294901760u), acc[7]);
                        }
                        prev_row = row_u_1;
                    }
                }
                int _min_31 = ((g + 2) < (31) ? (g + 2) : (31));
                float scale_u_2 = __shfl_sync(0xFFFFFFFF, scale_l, _min_31);
                if (g + 2 < count) {
                    if (done == 0) {
                        int same_2 = ((row_u_2 == prev_row) ? 1 : 0);
                        run = ((same_2 != 0) ? ((run + 1 == split) ? 0 : run + 1) : 0);
                        if (run == 0) {
                            if (active != 0) {
                                if (live != 0) {
                                    if (later_part != 0 || 1 - same_2 == 0) {
                                        int slot_2 = 2 * (head / split) + (1 - (1 - same_2));
                                        {
                                            float4 _v4 = make_float4(acc[0 + 0], acc[0 + 1], acc[0 + 2], acc[0 + 3]);
                                            *reinterpret_cast<float4*>((partial + ((unsigned long long)slot_2 * (unsigned long long)H + (unsigned long long)col)) + 0) = _v4;
                                        }
                                        {
                                            float4 _v4 = make_float4(acc[4 + 0], acc[4 + 1], acc[4 + 2], acc[4 + 3]);
                                            *reinterpret_cast<float4*>((partial + ((unsigned long long)slot_2 * (unsigned long long)H + (unsigned long long)col + 4)) + 0) = _v4;
                                        }
                                    } else {
                                        unsigned int packed_2[4];
                                        #pragma unroll
                                        for (int _lp = 0; _lp < 4; _lp++) {
                                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc[_lp*2 + 0], acc[_lp*2+1 + 0]));
                                            packed_2[_lp] = *(uint32_t*)&_bf2;
                                        }
                                        reinterpret_cast<int4*>(out + ((unsigned long long)open_row * (unsigned long long)H + (unsigned long long)col))[0] = reinterpret_cast<int4*>(packed_2)[0];
                                    }
                                }
                            }
                            int ends_2 = ((base + g + 2 >= hi) ? 1 : 0);
                            ends_2 = ((row_u_2 < 0) ? 1 : ends_2);
                            ends_2 = ((row_u_2 >= V) ? 1 : ends_2);
                            if (ends_2 != 0) {
                                done = 1;
                                active = 0;
                            } else {
                                active = 1;
                                head = base + g + 2;
                                later_part = same_2;
                                open_row = row_u_2;
                                acc[0] = 0.0f;
                                acc[1] = 0.0f;
                                acc[2] = 0.0f;
                                acc[3] = 0.0f;
                                acc[4] = 0.0f;
                                acc[5] = 0.0f;
                                acc[6] = 0.0f;
                                acc[7] = 0.0f;
                            }
                        }
                        if (active != 0) {
                            acc[0] = __fmaf_rn(scale_u_2, __uint_as_float(words_1[0] << 16), acc[0]);
                            acc[1] = __fmaf_rn(scale_u_2, __uint_as_float(words_1[0] & 4294901760u), acc[1]);
                            acc[2] = __fmaf_rn(scale_u_2, __uint_as_float(words_1[1] << 16), acc[2]);
                            acc[3] = __fmaf_rn(scale_u_2, __uint_as_float(words_1[1] & 4294901760u), acc[3]);
                            acc[4] = __fmaf_rn(scale_u_2, __uint_as_float(words_1[2] << 16), acc[4]);
                            acc[5] = __fmaf_rn(scale_u_2, __uint_as_float(words_1[2] & 4294901760u), acc[5]);
                            acc[6] = __fmaf_rn(scale_u_2, __uint_as_float(words_1[3] << 16), acc[6]);
                            acc[7] = __fmaf_rn(scale_u_2, __uint_as_float(words_1[3] & 4294901760u), acc[7]);
                        }
                        prev_row = row_u_2;
                    }
                }
                int _min_32 = ((g + 3) < (31) ? (g + 3) : (31));
                float scale_u_3 = __shfl_sync(0xFFFFFFFF, scale_l, _min_32);
                if (g + 3 < count) {
                    if (done == 0) {
                        int same_3 = ((row_u_3 == prev_row) ? 1 : 0);
                        run = ((same_3 != 0) ? ((run + 1 == split) ? 0 : run + 1) : 0);
                        if (run == 0) {
                            if (active != 0) {
                                if (live != 0) {
                                    if (later_part != 0 || 1 - same_3 == 0) {
                                        int slot_3 = 2 * (head / split) + (1 - (1 - same_3));
                                        {
                                            float4 _v4 = make_float4(acc[0 + 0], acc[0 + 1], acc[0 + 2], acc[0 + 3]);
                                            *reinterpret_cast<float4*>((partial + ((unsigned long long)slot_3 * (unsigned long long)H + (unsigned long long)col)) + 0) = _v4;
                                        }
                                        {
                                            float4 _v4 = make_float4(acc[4 + 0], acc[4 + 1], acc[4 + 2], acc[4 + 3]);
                                            *reinterpret_cast<float4*>((partial + ((unsigned long long)slot_3 * (unsigned long long)H + (unsigned long long)col + 4)) + 0) = _v4;
                                        }
                                    } else {
                                        unsigned int packed_3[4];
                                        #pragma unroll
                                        for (int _lp = 0; _lp < 4; _lp++) {
                                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc[_lp*2 + 0], acc[_lp*2+1 + 0]));
                                            packed_3[_lp] = *(uint32_t*)&_bf2;
                                        }
                                        reinterpret_cast<int4*>(out + ((unsigned long long)open_row * (unsigned long long)H + (unsigned long long)col))[0] = reinterpret_cast<int4*>(packed_3)[0];
                                    }
                                }
                            }
                            int ends_3 = ((base + g + 3 >= hi) ? 1 : 0);
                            ends_3 = ((row_u_3 < 0) ? 1 : ends_3);
                            ends_3 = ((row_u_3 >= V) ? 1 : ends_3);
                            if (ends_3 != 0) {
                                done = 1;
                                active = 0;
                            } else {
                                active = 1;
                                head = base + g + 3;
                                later_part = same_3;
                                open_row = row_u_3;
                                acc[0] = 0.0f;
                                acc[1] = 0.0f;
                                acc[2] = 0.0f;
                                acc[3] = 0.0f;
                                acc[4] = 0.0f;
                                acc[5] = 0.0f;
                                acc[6] = 0.0f;
                                acc[7] = 0.0f;
                            }
                        }
                        if (active != 0) {
                            acc[0] = __fmaf_rn(scale_u_3, __uint_as_float(words_2[0] << 16), acc[0]);
                            acc[1] = __fmaf_rn(scale_u_3, __uint_as_float(words_2[0] & 4294901760u), acc[1]);
                            acc[2] = __fmaf_rn(scale_u_3, __uint_as_float(words_2[1] << 16), acc[2]);
                            acc[3] = __fmaf_rn(scale_u_3, __uint_as_float(words_2[1] & 4294901760u), acc[3]);
                            acc[4] = __fmaf_rn(scale_u_3, __uint_as_float(words_2[2] << 16), acc[4]);
                            acc[5] = __fmaf_rn(scale_u_3, __uint_as_float(words_2[2] & 4294901760u), acc[5]);
                            acc[6] = __fmaf_rn(scale_u_3, __uint_as_float(words_2[3] << 16), acc[6]);
                            acc[7] = __fmaf_rn(scale_u_3, __uint_as_float(words_2[3] & 4294901760u), acc[7]);
                        }
                        prev_row = row_u_3;
                    }
                }
                int _min_33 = ((g + 4) < (31) ? (g + 4) : (31));
                float scale_u_4 = __shfl_sync(0xFFFFFFFF, scale_l, _min_33);
                if (g + 4 < count) {
                    if (done == 0) {
                        int same_4 = ((row_u_4 == prev_row) ? 1 : 0);
                        run = ((same_4 != 0) ? ((run + 1 == split) ? 0 : run + 1) : 0);
                        if (run == 0) {
                            if (active != 0) {
                                if (live != 0) {
                                    if (later_part != 0 || 1 - same_4 == 0) {
                                        int slot_4 = 2 * (head / split) + (1 - (1 - same_4));
                                        {
                                            float4 _v4 = make_float4(acc[0 + 0], acc[0 + 1], acc[0 + 2], acc[0 + 3]);
                                            *reinterpret_cast<float4*>((partial + ((unsigned long long)slot_4 * (unsigned long long)H + (unsigned long long)col)) + 0) = _v4;
                                        }
                                        {
                                            float4 _v4 = make_float4(acc[4 + 0], acc[4 + 1], acc[4 + 2], acc[4 + 3]);
                                            *reinterpret_cast<float4*>((partial + ((unsigned long long)slot_4 * (unsigned long long)H + (unsigned long long)col + 4)) + 0) = _v4;
                                        }
                                    } else {
                                        unsigned int packed_4[4];
                                        #pragma unroll
                                        for (int _lp = 0; _lp < 4; _lp++) {
                                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc[_lp*2 + 0], acc[_lp*2+1 + 0]));
                                            packed_4[_lp] = *(uint32_t*)&_bf2;
                                        }
                                        reinterpret_cast<int4*>(out + ((unsigned long long)open_row * (unsigned long long)H + (unsigned long long)col))[0] = reinterpret_cast<int4*>(packed_4)[0];
                                    }
                                }
                            }
                            int ends_4 = ((base + g + 4 >= hi) ? 1 : 0);
                            ends_4 = ((row_u_4 < 0) ? 1 : ends_4);
                            ends_4 = ((row_u_4 >= V) ? 1 : ends_4);
                            if (ends_4 != 0) {
                                done = 1;
                                active = 0;
                            } else {
                                active = 1;
                                head = base + g + 4;
                                later_part = same_4;
                                open_row = row_u_4;
                                acc[0] = 0.0f;
                                acc[1] = 0.0f;
                                acc[2] = 0.0f;
                                acc[3] = 0.0f;
                                acc[4] = 0.0f;
                                acc[5] = 0.0f;
                                acc[6] = 0.0f;
                                acc[7] = 0.0f;
                            }
                        }
                        if (active != 0) {
                            acc[0] = __fmaf_rn(scale_u_4, __uint_as_float(words_3[0] << 16), acc[0]);
                            acc[1] = __fmaf_rn(scale_u_4, __uint_as_float(words_3[0] & 4294901760u), acc[1]);
                            acc[2] = __fmaf_rn(scale_u_4, __uint_as_float(words_3[1] << 16), acc[2]);
                            acc[3] = __fmaf_rn(scale_u_4, __uint_as_float(words_3[1] & 4294901760u), acc[3]);
                            acc[4] = __fmaf_rn(scale_u_4, __uint_as_float(words_3[2] << 16), acc[4]);
                            acc[5] = __fmaf_rn(scale_u_4, __uint_as_float(words_3[2] & 4294901760u), acc[5]);
                            acc[6] = __fmaf_rn(scale_u_4, __uint_as_float(words_3[3] << 16), acc[6]);
                            acc[7] = __fmaf_rn(scale_u_4, __uint_as_float(words_3[3] & 4294901760u), acc[7]);
                        }
                        prev_row = row_u_4;
                    }
                }
                int _min_34 = ((g + 5) < (31) ? (g + 5) : (31));
                float scale_u_5 = __shfl_sync(0xFFFFFFFF, scale_l, _min_34);
                if (g + 5 < count) {
                    if (done == 0) {
                        int same_5 = ((row_u_5 == prev_row) ? 1 : 0);
                        run = ((same_5 != 0) ? ((run + 1 == split) ? 0 : run + 1) : 0);
                        if (run == 0) {
                            if (active != 0) {
                                if (live != 0) {
                                    if (later_part != 0 || 1 - same_5 == 0) {
                                        int slot_5 = 2 * (head / split) + (1 - (1 - same_5));
                                        {
                                            float4 _v4 = make_float4(acc[0 + 0], acc[0 + 1], acc[0 + 2], acc[0 + 3]);
                                            *reinterpret_cast<float4*>((partial + ((unsigned long long)slot_5 * (unsigned long long)H + (unsigned long long)col)) + 0) = _v4;
                                        }
                                        {
                                            float4 _v4 = make_float4(acc[4 + 0], acc[4 + 1], acc[4 + 2], acc[4 + 3]);
                                            *reinterpret_cast<float4*>((partial + ((unsigned long long)slot_5 * (unsigned long long)H + (unsigned long long)col + 4)) + 0) = _v4;
                                        }
                                    } else {
                                        unsigned int packed_5[4];
                                        #pragma unroll
                                        for (int _lp = 0; _lp < 4; _lp++) {
                                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc[_lp*2 + 0], acc[_lp*2+1 + 0]));
                                            packed_5[_lp] = *(uint32_t*)&_bf2;
                                        }
                                        reinterpret_cast<int4*>(out + ((unsigned long long)open_row * (unsigned long long)H + (unsigned long long)col))[0] = reinterpret_cast<int4*>(packed_5)[0];
                                    }
                                }
                            }
                            int ends_5 = ((base + g + 5 >= hi) ? 1 : 0);
                            ends_5 = ((row_u_5 < 0) ? 1 : ends_5);
                            ends_5 = ((row_u_5 >= V) ? 1 : ends_5);
                            if (ends_5 != 0) {
                                done = 1;
                                active = 0;
                            } else {
                                active = 1;
                                head = base + g + 5;
                                later_part = same_5;
                                open_row = row_u_5;
                                acc[0] = 0.0f;
                                acc[1] = 0.0f;
                                acc[2] = 0.0f;
                                acc[3] = 0.0f;
                                acc[4] = 0.0f;
                                acc[5] = 0.0f;
                                acc[6] = 0.0f;
                                acc[7] = 0.0f;
                            }
                        }
                        if (active != 0) {
                            acc[0] = __fmaf_rn(scale_u_5, __uint_as_float(words_4[0] << 16), acc[0]);
                            acc[1] = __fmaf_rn(scale_u_5, __uint_as_float(words_4[0] & 4294901760u), acc[1]);
                            acc[2] = __fmaf_rn(scale_u_5, __uint_as_float(words_4[1] << 16), acc[2]);
                            acc[3] = __fmaf_rn(scale_u_5, __uint_as_float(words_4[1] & 4294901760u), acc[3]);
                            acc[4] = __fmaf_rn(scale_u_5, __uint_as_float(words_4[2] << 16), acc[4]);
                            acc[5] = __fmaf_rn(scale_u_5, __uint_as_float(words_4[2] & 4294901760u), acc[5]);
                            acc[6] = __fmaf_rn(scale_u_5, __uint_as_float(words_4[3] << 16), acc[6]);
                            acc[7] = __fmaf_rn(scale_u_5, __uint_as_float(words_4[3] & 4294901760u), acc[7]);
                        }
                        prev_row = row_u_5;
                    }
                }
                int _min_35 = ((g + 6) < (31) ? (g + 6) : (31));
                float scale_u_6 = __shfl_sync(0xFFFFFFFF, scale_l, _min_35);
                if (g + 6 < count) {
                    if (done == 0) {
                        int same_6 = ((row_u_6 == prev_row) ? 1 : 0);
                        run = ((same_6 != 0) ? ((run + 1 == split) ? 0 : run + 1) : 0);
                        if (run == 0) {
                            if (active != 0) {
                                if (live != 0) {
                                    if (later_part != 0 || 1 - same_6 == 0) {
                                        int slot_6 = 2 * (head / split) + (1 - (1 - same_6));
                                        {
                                            float4 _v4 = make_float4(acc[0 + 0], acc[0 + 1], acc[0 + 2], acc[0 + 3]);
                                            *reinterpret_cast<float4*>((partial + ((unsigned long long)slot_6 * (unsigned long long)H + (unsigned long long)col)) + 0) = _v4;
                                        }
                                        {
                                            float4 _v4 = make_float4(acc[4 + 0], acc[4 + 1], acc[4 + 2], acc[4 + 3]);
                                            *reinterpret_cast<float4*>((partial + ((unsigned long long)slot_6 * (unsigned long long)H + (unsigned long long)col + 4)) + 0) = _v4;
                                        }
                                    } else {
                                        unsigned int packed_6[4];
                                        #pragma unroll
                                        for (int _lp = 0; _lp < 4; _lp++) {
                                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc[_lp*2 + 0], acc[_lp*2+1 + 0]));
                                            packed_6[_lp] = *(uint32_t*)&_bf2;
                                        }
                                        reinterpret_cast<int4*>(out + ((unsigned long long)open_row * (unsigned long long)H + (unsigned long long)col))[0] = reinterpret_cast<int4*>(packed_6)[0];
                                    }
                                }
                            }
                            int ends_6 = ((base + g + 6 >= hi) ? 1 : 0);
                            ends_6 = ((row_u_6 < 0) ? 1 : ends_6);
                            ends_6 = ((row_u_6 >= V) ? 1 : ends_6);
                            if (ends_6 != 0) {
                                done = 1;
                                active = 0;
                            } else {
                                active = 1;
                                head = base + g + 6;
                                later_part = same_6;
                                open_row = row_u_6;
                                acc[0] = 0.0f;
                                acc[1] = 0.0f;
                                acc[2] = 0.0f;
                                acc[3] = 0.0f;
                                acc[4] = 0.0f;
                                acc[5] = 0.0f;
                                acc[6] = 0.0f;
                                acc[7] = 0.0f;
                            }
                        }
                        if (active != 0) {
                            acc[0] = __fmaf_rn(scale_u_6, __uint_as_float(words_5[0] << 16), acc[0]);
                            acc[1] = __fmaf_rn(scale_u_6, __uint_as_float(words_5[0] & 4294901760u), acc[1]);
                            acc[2] = __fmaf_rn(scale_u_6, __uint_as_float(words_5[1] << 16), acc[2]);
                            acc[3] = __fmaf_rn(scale_u_6, __uint_as_float(words_5[1] & 4294901760u), acc[3]);
                            acc[4] = __fmaf_rn(scale_u_6, __uint_as_float(words_5[2] << 16), acc[4]);
                            acc[5] = __fmaf_rn(scale_u_6, __uint_as_float(words_5[2] & 4294901760u), acc[5]);
                            acc[6] = __fmaf_rn(scale_u_6, __uint_as_float(words_5[3] << 16), acc[6]);
                            acc[7] = __fmaf_rn(scale_u_6, __uint_as_float(words_5[3] & 4294901760u), acc[7]);
                        }
                        prev_row = row_u_6;
                    }
                }
                int _min_36 = ((g + 7) < (31) ? (g + 7) : (31));
                float scale_u_7 = __shfl_sync(0xFFFFFFFF, scale_l, _min_36);
                if (g + 7 < count) {
                    if (done == 0) {
                        int same_7 = ((row_u_7 == prev_row) ? 1 : 0);
                        run = ((same_7 != 0) ? ((run + 1 == split) ? 0 : run + 1) : 0);
                        if (run == 0) {
                            if (active != 0) {
                                if (live != 0) {
                                    if (later_part != 0 || 1 - same_7 == 0) {
                                        int slot_7 = 2 * (head / split) + (1 - (1 - same_7));
                                        {
                                            float4 _v4 = make_float4(acc[0 + 0], acc[0 + 1], acc[0 + 2], acc[0 + 3]);
                                            *reinterpret_cast<float4*>((partial + ((unsigned long long)slot_7 * (unsigned long long)H + (unsigned long long)col)) + 0) = _v4;
                                        }
                                        {
                                            float4 _v4 = make_float4(acc[4 + 0], acc[4 + 1], acc[4 + 2], acc[4 + 3]);
                                            *reinterpret_cast<float4*>((partial + ((unsigned long long)slot_7 * (unsigned long long)H + (unsigned long long)col + 4)) + 0) = _v4;
                                        }
                                    } else {
                                        unsigned int packed_7[4];
                                        #pragma unroll
                                        for (int _lp = 0; _lp < 4; _lp++) {
                                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc[_lp*2 + 0], acc[_lp*2+1 + 0]));
                                            packed_7[_lp] = *(uint32_t*)&_bf2;
                                        }
                                        reinterpret_cast<int4*>(out + ((unsigned long long)open_row * (unsigned long long)H + (unsigned long long)col))[0] = reinterpret_cast<int4*>(packed_7)[0];
                                    }
                                }
                            }
                            int ends_7 = ((base + g + 7 >= hi) ? 1 : 0);
                            ends_7 = ((row_u_7 < 0) ? 1 : ends_7);
                            ends_7 = ((row_u_7 >= V) ? 1 : ends_7);
                            if (ends_7 != 0) {
                                done = 1;
                                active = 0;
                            } else {
                                active = 1;
                                head = base + g + 7;
                                later_part = same_7;
                                open_row = row_u_7;
                                acc[0] = 0.0f;
                                acc[1] = 0.0f;
                                acc[2] = 0.0f;
                                acc[3] = 0.0f;
                                acc[4] = 0.0f;
                                acc[5] = 0.0f;
                                acc[6] = 0.0f;
                                acc[7] = 0.0f;
                            }
                        }
                        if (active != 0) {
                            acc[0] = __fmaf_rn(scale_u_7, __uint_as_float(words_6[0] << 16), acc[0]);
                            acc[1] = __fmaf_rn(scale_u_7, __uint_as_float(words_6[0] & 4294901760u), acc[1]);
                            acc[2] = __fmaf_rn(scale_u_7, __uint_as_float(words_6[1] << 16), acc[2]);
                            acc[3] = __fmaf_rn(scale_u_7, __uint_as_float(words_6[1] & 4294901760u), acc[3]);
                            acc[4] = __fmaf_rn(scale_u_7, __uint_as_float(words_6[2] << 16), acc[4]);
                            acc[5] = __fmaf_rn(scale_u_7, __uint_as_float(words_6[2] & 4294901760u), acc[5]);
                            acc[6] = __fmaf_rn(scale_u_7, __uint_as_float(words_6[3] << 16), acc[6]);
                            acc[7] = __fmaf_rn(scale_u_7, __uint_as_float(words_6[3] & 4294901760u), acc[7]);
                        }
                        prev_row = row_u_7;
                    }
                }
            }
            if (done != 0) {
                break;
            }
        }
        if (active != 0) {
            if (live != 0) {
                if (later_part != 0 || 0) {
                    int slot_8 = 2 * (head / split);
                    {
                        float4 _v4 = make_float4(acc[0 + 0], acc[0 + 1], acc[0 + 2], acc[0 + 3]);
                        *reinterpret_cast<float4*>((partial + ((unsigned long long)slot_8 * (unsigned long long)H + (unsigned long long)col)) + 0) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(acc[4 + 0], acc[4 + 1], acc[4 + 2], acc[4 + 3]);
                        *reinterpret_cast<float4*>((partial + ((unsigned long long)slot_8 * (unsigned long long)H + (unsigned long long)col + 4)) + 0) = _v4;
                    }
                } else {
                    unsigned int packed_8[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc[_lp*2 + 0], acc[_lp*2+1 + 0]));
                        packed_8[_lp] = *(uint32_t*)&_bf2;
                    }
                    reinterpret_cast<int4*>(out + ((unsigned long long)open_row * (unsigned long long)H + (unsigned long long)col))[0] = reinterpret_cast<int4*>(packed_8)[0];
                }
            }
        }
    }
}

} // extern "C"
