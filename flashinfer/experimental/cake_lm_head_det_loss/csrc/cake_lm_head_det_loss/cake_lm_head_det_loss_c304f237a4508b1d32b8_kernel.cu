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
kernel_cake_lm_head_det_loss_c304f237a4508b1d32b8(__nv_bfloat16* __restrict__ rows, __nv_bfloat16* __restrict__ scale, int* __restrict__ ids, int* __restrict__ offsets, __nv_bfloat16* __restrict__ out, int ld_rows, int H, int V, int M, int T, int min_pairs)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // === Task calls (declaration order) ===
    int t = blockIdx.x;
    int col = blockIdx.y * 1024 + warp * 256 + lane * 8;
    int live = ((col < H) ? 1 : 0);
    if (t < T) {
        int _max_0 = ((offsets[t]) > (0) ? (offsets[t]) : (0));
        int start = ((_max_0) < (M) ? (_max_0) : (M));
        int _max_1 = ((offsets[t + 1]) > (start) ? (offsets[t + 1]) : (start));
        int end = ((_max_1) < (M) ? (_max_1) : (M));
        float acc[8];
        acc[0] = 0.0f;
        acc[1] = 0.0f;
        acc[2] = 0.0f;
        acc[3] = 0.0f;
        acc[4] = 0.0f;
        acc[5] = 0.0f;
        acc[6] = 0.0f;
        acc[7] = 0.0f;
        if (end - start >= min_pairs) {
            #pragma unroll 1
            for (int base = start; base < end; base += 32) {
                int row_l = 0;
                float scale_l = 0.0f;
                if (base + lane < end) {
                    int _max_3 = ((ids[base + lane]) > (0) ? (ids[base + lane]) : (0));
                    int _min_2 = ((_max_3) < (V - 1) ? (_max_3) : (V - 1));
                    row_l = _min_2;
                    scale_l = (float)scale[base + lane];
                }
                int count = ((end - base) < (32) ? (end - base) : (32));
                #pragma unroll 1
                for (int g = 0; g < count; g += 8) {
                    int _min_4 = ((g) < (31) ? (g) : (31));
                    int row_u_0 = __shfl_sync(0xFFFFFFFF, row_l, _min_4);
                    unsigned int words[4];
                    if (g < count && live != 0) {
                        {
                            const uint4* _vptr_5 = reinterpret_cast<const uint4*>(rows + (unsigned long long)row_u_0 * (unsigned long long)ld_rows + (unsigned long long)col);
                            uint4* _vdst_5 = reinterpret_cast<uint4*>(&words[0]);
                            #pragma unroll
                            for (int _blk = 0; _blk < 1; _blk++) {
                                _vdst_5[_blk] = _vptr_5[_blk];
                            }
                        }
                    }
                    int _min_6 = ((g + 1) < (31) ? (g + 1) : (31));
                    int row_u_1 = __shfl_sync(0xFFFFFFFF, row_l, _min_6);
                    unsigned int words_0[4];
                    if (g + 1 < count && live != 0) {
                        {
                            const uint4* _vptr_7 = reinterpret_cast<const uint4*>(rows + (unsigned long long)row_u_1 * (unsigned long long)ld_rows + (unsigned long long)col);
                            uint4* _vdst_7 = reinterpret_cast<uint4*>(&words_0[0]);
                            #pragma unroll
                            for (int _blk = 0; _blk < 1; _blk++) {
                                _vdst_7[_blk] = _vptr_7[_blk];
                            }
                        }
                    }
                    int _min_8 = ((g + 2) < (31) ? (g + 2) : (31));
                    int row_u_2 = __shfl_sync(0xFFFFFFFF, row_l, _min_8);
                    unsigned int words_1[4];
                    if (g + 2 < count && live != 0) {
                        {
                            const uint4* _vptr_9 = reinterpret_cast<const uint4*>(rows + (unsigned long long)row_u_2 * (unsigned long long)ld_rows + (unsigned long long)col);
                            uint4* _vdst_9 = reinterpret_cast<uint4*>(&words_1[0]);
                            #pragma unroll
                            for (int _blk = 0; _blk < 1; _blk++) {
                                _vdst_9[_blk] = _vptr_9[_blk];
                            }
                        }
                    }
                    int _min_10 = ((g + 3) < (31) ? (g + 3) : (31));
                    int row_u_3 = __shfl_sync(0xFFFFFFFF, row_l, _min_10);
                    unsigned int words_2[4];
                    if (g + 3 < count && live != 0) {
                        {
                            const uint4* _vptr_11 = reinterpret_cast<const uint4*>(rows + (unsigned long long)row_u_3 * (unsigned long long)ld_rows + (unsigned long long)col);
                            uint4* _vdst_11 = reinterpret_cast<uint4*>(&words_2[0]);
                            #pragma unroll
                            for (int _blk = 0; _blk < 1; _blk++) {
                                _vdst_11[_blk] = _vptr_11[_blk];
                            }
                        }
                    }
                    int _min_12 = ((g + 4) < (31) ? (g + 4) : (31));
                    int row_u_4 = __shfl_sync(0xFFFFFFFF, row_l, _min_12);
                    unsigned int words_3[4];
                    if (g + 4 < count && live != 0) {
                        {
                            const uint4* _vptr_13 = reinterpret_cast<const uint4*>(rows + (unsigned long long)row_u_4 * (unsigned long long)ld_rows + (unsigned long long)col);
                            uint4* _vdst_13 = reinterpret_cast<uint4*>(&words_3[0]);
                            #pragma unroll
                            for (int _blk = 0; _blk < 1; _blk++) {
                                _vdst_13[_blk] = _vptr_13[_blk];
                            }
                        }
                    }
                    int _min_14 = ((g + 5) < (31) ? (g + 5) : (31));
                    int row_u_5 = __shfl_sync(0xFFFFFFFF, row_l, _min_14);
                    unsigned int words_4[4];
                    if (g + 5 < count && live != 0) {
                        {
                            const uint4* _vptr_15 = reinterpret_cast<const uint4*>(rows + (unsigned long long)row_u_5 * (unsigned long long)ld_rows + (unsigned long long)col);
                            uint4* _vdst_15 = reinterpret_cast<uint4*>(&words_4[0]);
                            #pragma unroll
                            for (int _blk = 0; _blk < 1; _blk++) {
                                _vdst_15[_blk] = _vptr_15[_blk];
                            }
                        }
                    }
                    int _min_16 = ((g + 6) < (31) ? (g + 6) : (31));
                    int row_u_6 = __shfl_sync(0xFFFFFFFF, row_l, _min_16);
                    unsigned int words_5[4];
                    if (g + 6 < count && live != 0) {
                        {
                            const uint4* _vptr_17 = reinterpret_cast<const uint4*>(rows + (unsigned long long)row_u_6 * (unsigned long long)ld_rows + (unsigned long long)col);
                            uint4* _vdst_17 = reinterpret_cast<uint4*>(&words_5[0]);
                            #pragma unroll
                            for (int _blk = 0; _blk < 1; _blk++) {
                                _vdst_17[_blk] = _vptr_17[_blk];
                            }
                        }
                    }
                    int _min_18 = ((g + 7) < (31) ? (g + 7) : (31));
                    int row_u_7 = __shfl_sync(0xFFFFFFFF, row_l, _min_18);
                    unsigned int words_6[4];
                    if (g + 7 < count && live != 0) {
                        {
                            const uint4* _vptr_19 = reinterpret_cast<const uint4*>(rows + (unsigned long long)row_u_7 * (unsigned long long)ld_rows + (unsigned long long)col);
                            uint4* _vdst_19 = reinterpret_cast<uint4*>(&words_6[0]);
                            #pragma unroll
                            for (int _blk = 0; _blk < 1; _blk++) {
                                _vdst_19[_blk] = _vptr_19[_blk];
                            }
                        }
                    }
                    int _min_20 = ((g) < (31) ? (g) : (31));
                    float scale_u_0 = __shfl_sync(0xFFFFFFFF, scale_l, _min_20);
                    if (g < count) {
                        acc[0] = __fmaf_rn(scale_u_0, __uint_as_float(words[0] << 16), acc[0]);
                        acc[1] = __fmaf_rn(scale_u_0, __uint_as_float(words[0] & 4294901760u), acc[1]);
                        acc[2] = __fmaf_rn(scale_u_0, __uint_as_float(words[1] << 16), acc[2]);
                        acc[3] = __fmaf_rn(scale_u_0, __uint_as_float(words[1] & 4294901760u), acc[3]);
                        acc[4] = __fmaf_rn(scale_u_0, __uint_as_float(words[2] << 16), acc[4]);
                        acc[5] = __fmaf_rn(scale_u_0, __uint_as_float(words[2] & 4294901760u), acc[5]);
                        acc[6] = __fmaf_rn(scale_u_0, __uint_as_float(words[3] << 16), acc[6]);
                        acc[7] = __fmaf_rn(scale_u_0, __uint_as_float(words[3] & 4294901760u), acc[7]);
                    }
                    int _min_21 = ((g + 1) < (31) ? (g + 1) : (31));
                    float scale_u_1 = __shfl_sync(0xFFFFFFFF, scale_l, _min_21);
                    if (g + 1 < count) {
                        acc[0] = __fmaf_rn(scale_u_1, __uint_as_float(words_0[0] << 16), acc[0]);
                        acc[1] = __fmaf_rn(scale_u_1, __uint_as_float(words_0[0] & 4294901760u), acc[1]);
                        acc[2] = __fmaf_rn(scale_u_1, __uint_as_float(words_0[1] << 16), acc[2]);
                        acc[3] = __fmaf_rn(scale_u_1, __uint_as_float(words_0[1] & 4294901760u), acc[3]);
                        acc[4] = __fmaf_rn(scale_u_1, __uint_as_float(words_0[2] << 16), acc[4]);
                        acc[5] = __fmaf_rn(scale_u_1, __uint_as_float(words_0[2] & 4294901760u), acc[5]);
                        acc[6] = __fmaf_rn(scale_u_1, __uint_as_float(words_0[3] << 16), acc[6]);
                        acc[7] = __fmaf_rn(scale_u_1, __uint_as_float(words_0[3] & 4294901760u), acc[7]);
                    }
                    int _min_22 = ((g + 2) < (31) ? (g + 2) : (31));
                    float scale_u_2 = __shfl_sync(0xFFFFFFFF, scale_l, _min_22);
                    if (g + 2 < count) {
                        acc[0] = __fmaf_rn(scale_u_2, __uint_as_float(words_1[0] << 16), acc[0]);
                        acc[1] = __fmaf_rn(scale_u_2, __uint_as_float(words_1[0] & 4294901760u), acc[1]);
                        acc[2] = __fmaf_rn(scale_u_2, __uint_as_float(words_1[1] << 16), acc[2]);
                        acc[3] = __fmaf_rn(scale_u_2, __uint_as_float(words_1[1] & 4294901760u), acc[3]);
                        acc[4] = __fmaf_rn(scale_u_2, __uint_as_float(words_1[2] << 16), acc[4]);
                        acc[5] = __fmaf_rn(scale_u_2, __uint_as_float(words_1[2] & 4294901760u), acc[5]);
                        acc[6] = __fmaf_rn(scale_u_2, __uint_as_float(words_1[3] << 16), acc[6]);
                        acc[7] = __fmaf_rn(scale_u_2, __uint_as_float(words_1[3] & 4294901760u), acc[7]);
                    }
                    int _min_23 = ((g + 3) < (31) ? (g + 3) : (31));
                    float scale_u_3 = __shfl_sync(0xFFFFFFFF, scale_l, _min_23);
                    if (g + 3 < count) {
                        acc[0] = __fmaf_rn(scale_u_3, __uint_as_float(words_2[0] << 16), acc[0]);
                        acc[1] = __fmaf_rn(scale_u_3, __uint_as_float(words_2[0] & 4294901760u), acc[1]);
                        acc[2] = __fmaf_rn(scale_u_3, __uint_as_float(words_2[1] << 16), acc[2]);
                        acc[3] = __fmaf_rn(scale_u_3, __uint_as_float(words_2[1] & 4294901760u), acc[3]);
                        acc[4] = __fmaf_rn(scale_u_3, __uint_as_float(words_2[2] << 16), acc[4]);
                        acc[5] = __fmaf_rn(scale_u_3, __uint_as_float(words_2[2] & 4294901760u), acc[5]);
                        acc[6] = __fmaf_rn(scale_u_3, __uint_as_float(words_2[3] << 16), acc[6]);
                        acc[7] = __fmaf_rn(scale_u_3, __uint_as_float(words_2[3] & 4294901760u), acc[7]);
                    }
                    int _min_24 = ((g + 4) < (31) ? (g + 4) : (31));
                    float scale_u_4 = __shfl_sync(0xFFFFFFFF, scale_l, _min_24);
                    if (g + 4 < count) {
                        acc[0] = __fmaf_rn(scale_u_4, __uint_as_float(words_3[0] << 16), acc[0]);
                        acc[1] = __fmaf_rn(scale_u_4, __uint_as_float(words_3[0] & 4294901760u), acc[1]);
                        acc[2] = __fmaf_rn(scale_u_4, __uint_as_float(words_3[1] << 16), acc[2]);
                        acc[3] = __fmaf_rn(scale_u_4, __uint_as_float(words_3[1] & 4294901760u), acc[3]);
                        acc[4] = __fmaf_rn(scale_u_4, __uint_as_float(words_3[2] << 16), acc[4]);
                        acc[5] = __fmaf_rn(scale_u_4, __uint_as_float(words_3[2] & 4294901760u), acc[5]);
                        acc[6] = __fmaf_rn(scale_u_4, __uint_as_float(words_3[3] << 16), acc[6]);
                        acc[7] = __fmaf_rn(scale_u_4, __uint_as_float(words_3[3] & 4294901760u), acc[7]);
                    }
                    int _min_25 = ((g + 5) < (31) ? (g + 5) : (31));
                    float scale_u_5 = __shfl_sync(0xFFFFFFFF, scale_l, _min_25);
                    if (g + 5 < count) {
                        acc[0] = __fmaf_rn(scale_u_5, __uint_as_float(words_4[0] << 16), acc[0]);
                        acc[1] = __fmaf_rn(scale_u_5, __uint_as_float(words_4[0] & 4294901760u), acc[1]);
                        acc[2] = __fmaf_rn(scale_u_5, __uint_as_float(words_4[1] << 16), acc[2]);
                        acc[3] = __fmaf_rn(scale_u_5, __uint_as_float(words_4[1] & 4294901760u), acc[3]);
                        acc[4] = __fmaf_rn(scale_u_5, __uint_as_float(words_4[2] << 16), acc[4]);
                        acc[5] = __fmaf_rn(scale_u_5, __uint_as_float(words_4[2] & 4294901760u), acc[5]);
                        acc[6] = __fmaf_rn(scale_u_5, __uint_as_float(words_4[3] << 16), acc[6]);
                        acc[7] = __fmaf_rn(scale_u_5, __uint_as_float(words_4[3] & 4294901760u), acc[7]);
                    }
                    int _min_26 = ((g + 6) < (31) ? (g + 6) : (31));
                    float scale_u_6 = __shfl_sync(0xFFFFFFFF, scale_l, _min_26);
                    if (g + 6 < count) {
                        acc[0] = __fmaf_rn(scale_u_6, __uint_as_float(words_5[0] << 16), acc[0]);
                        acc[1] = __fmaf_rn(scale_u_6, __uint_as_float(words_5[0] & 4294901760u), acc[1]);
                        acc[2] = __fmaf_rn(scale_u_6, __uint_as_float(words_5[1] << 16), acc[2]);
                        acc[3] = __fmaf_rn(scale_u_6, __uint_as_float(words_5[1] & 4294901760u), acc[3]);
                        acc[4] = __fmaf_rn(scale_u_6, __uint_as_float(words_5[2] << 16), acc[4]);
                        acc[5] = __fmaf_rn(scale_u_6, __uint_as_float(words_5[2] & 4294901760u), acc[5]);
                        acc[6] = __fmaf_rn(scale_u_6, __uint_as_float(words_5[3] << 16), acc[6]);
                        acc[7] = __fmaf_rn(scale_u_6, __uint_as_float(words_5[3] & 4294901760u), acc[7]);
                    }
                    int _min_27 = ((g + 7) < (31) ? (g + 7) : (31));
                    float scale_u_7 = __shfl_sync(0xFFFFFFFF, scale_l, _min_27);
                    if (g + 7 < count) {
                        acc[0] = __fmaf_rn(scale_u_7, __uint_as_float(words_6[0] << 16), acc[0]);
                        acc[1] = __fmaf_rn(scale_u_7, __uint_as_float(words_6[0] & 4294901760u), acc[1]);
                        acc[2] = __fmaf_rn(scale_u_7, __uint_as_float(words_6[1] << 16), acc[2]);
                        acc[3] = __fmaf_rn(scale_u_7, __uint_as_float(words_6[1] & 4294901760u), acc[3]);
                        acc[4] = __fmaf_rn(scale_u_7, __uint_as_float(words_6[2] << 16), acc[4]);
                        acc[5] = __fmaf_rn(scale_u_7, __uint_as_float(words_6[2] & 4294901760u), acc[5]);
                        acc[6] = __fmaf_rn(scale_u_7, __uint_as_float(words_6[3] << 16), acc[6]);
                        acc[7] = __fmaf_rn(scale_u_7, __uint_as_float(words_6[3] & 4294901760u), acc[7]);
                    }
                }
            }
        }
        if (live != 0) {
            unsigned int packed[4];
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc[_lp*2 + 0], acc[_lp*2+1 + 0]));
                packed[_lp] = *(uint32_t*)&_bf2;
            }
            reinterpret_cast<int4*>(out + ((unsigned long long)t * (unsigned long long)H + (unsigned long long)col))[0] = reinterpret_cast<int4*>(packed)[0];
        }
    }
}

} // extern "C"
