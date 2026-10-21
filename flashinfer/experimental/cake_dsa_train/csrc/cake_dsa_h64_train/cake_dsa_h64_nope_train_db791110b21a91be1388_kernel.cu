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
#define LAUNCH_MIN_BLOCKS 1

#include <math_constants.h>

extern "C" {

__global__ __launch_bounds__(128, LAUNCH_MIN_BLOCKS) void
kernel_cake_dsa_h64_nope_train_db791110b21a91be1388(int* __restrict__ indices, int* __restrict__ cu_seqlens_q, int* __restrict__ cu_seqlens_k, int* __restrict__ out, int* __restrict__ topk_length, int num_rows, int topk, int idx_stride, int indices_offset, int num_docs, int search_iters, int causal, int has_docs, int flat_bound, int write_out)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // === Task calls (dependency order) ===
    int t = blockIdx.x * 4 + warp;
    int lane_0 = lane;
    if (t < num_rows) {
        int key_base = 0;
        int bound = flat_bound;
        if (has_docs != 0) {
            int lo = 0;
            int hi = num_docs;
            #pragma unroll 1
            for (int it = 0; it < search_iters; it++) {
                int mid = (lo + hi) / 2;
                int _vec_load_0[1];
                {
                    _vec_load_0[0] = *reinterpret_cast<const int*>(cu_seqlens_q + (long long)mid + 0);
                }
                int go_lo = (int)(mid > lo && _vec_load_0[0] <= t);
                int go_hi = (int)(mid > lo && _vec_load_0[0] > t);
                lo = ((go_lo != 0) ? mid : lo);
                hi = ((go_hi != 0) ? mid : hi);
            }
            int _vec_load_1[1];
            {
                _vec_load_1[0] = *reinterpret_cast<const int*>(cu_seqlens_k + (long long)lo + 0);
            }
            int _vec_load_2[1];
            {
                _vec_load_2[0] = *reinterpret_cast<const int*>(cu_seqlens_k + (long long)(lo + 1) + 0);
            }
            key_base = _vec_load_1[0];
            int k_end = _vec_load_2[0];
            bound = k_end - key_base - 1;
            if (causal != 0) {
                int _vec_load_3[1];
                {
                    _vec_load_3[0] = *reinterpret_cast<const int*>(cu_seqlens_q + (long long)(lo + 1) + 0);
                }
                bound = k_end - _vec_load_3[0] + t - key_base;
            }
        }
        long long row_base = (long long)indices_offset + (long long)t * (long long)idx_stride;
        long long out_base = (long long)t * (long long)topk;
        int aligned = (int)(((idx_stride | indices_offset | topk) & 3) == 0);
        int last = 0;
        #pragma unroll 1
        for (int sblk = 0; sblk < (topk + 255) / 256; sblk++) {
            int spos = sblk * 256 + lane_0 * 8;
            int vals[8];
            int whole = (int)(spos + 8 <= topk) & aligned;
            if (whole != 0) {
                int _vec_load_4[4];
                {
                    const int4* _ivptr_0 = reinterpret_cast<const int4*>(indices + (row_base + (long long)spos) + 0);
                    int4 _ivld_0;
                    _ivld_0 = *_ivptr_0;
                    _vec_load_4[0 + 0] = _ivld_0.x;
                    _vec_load_4[0 + 1] = _ivld_0.y;
                    _vec_load_4[0 + 2] = _ivld_0.z;
                    _vec_load_4[0 + 3] = _ivld_0.w;
                }
                int _vec_load_5[4];
                {
                    const int4* _ivptr_1 = reinterpret_cast<const int4*>(indices + (row_base + (long long)spos + 4) + 0);
                    int4 _ivld_1;
                    _ivld_1 = *_ivptr_1;
                    _vec_load_5[0 + 0] = _ivld_1.x;
                    _vec_load_5[0 + 1] = _ivld_1.y;
                    _vec_load_5[0 + 2] = _ivld_1.z;
                    _vec_load_5[0 + 3] = _ivld_1.w;
                }
                vals[0] = _vec_load_4[0];
                vals[4] = _vec_load_5[0];
                vals[1] = _vec_load_4[1];
                vals[5] = _vec_load_5[1];
                vals[2] = _vec_load_4[2];
                vals[6] = _vec_load_5[2];
                vals[3] = _vec_load_4[3];
                vals[7] = _vec_load_5[3];
            } else {
                int sv = -1;
                if (spos < topk) {
                    sv = indices[row_base + (long long)spos];
                }
                vals[0] = sv;
                int sv_0 = -1;
                if (spos + 1 < topk) {
                    sv_0 = indices[row_base + (long long)(spos + 1)];
                }
                vals[1] = sv_0;
                int sv_1 = -1;
                if (spos + 2 < topk) {
                    sv_1 = indices[row_base + (long long)(spos + 2)];
                }
                vals[2] = sv_1;
                int sv_2 = -1;
                if (spos + 3 < topk) {
                    sv_2 = indices[row_base + (long long)(spos + 3)];
                }
                vals[3] = sv_2;
                int sv_3 = -1;
                if (spos + 4 < topk) {
                    sv_3 = indices[row_base + (long long)(spos + 4)];
                }
                vals[4] = sv_3;
                int sv_4 = -1;
                if (spos + 5 < topk) {
                    sv_4 = indices[row_base + (long long)(spos + 5)];
                }
                vals[5] = sv_4;
                int sv_5 = -1;
                if (spos + 6 < topk) {
                    sv_5 = indices[row_base + (long long)(spos + 6)];
                }
                vals[6] = sv_5;
                int sv_6 = -1;
                if (spos + 7 < topk) {
                    sv_6 = indices[row_base + (long long)(spos + 7)];
                }
                vals[7] = sv_6;
            }
            int outv[8];
            int v = vals[0];
            int ok = (int)(v >= 0 && v <= bound);
            outv[0] = -1;
            if (ok != 0) {
                outv[0] = v + key_base;
                last = spos + 1;
            }
            int v_0 = vals[1];
            int ok_1 = (int)(v_0 >= 0 && v_0 <= bound);
            outv[1] = -1;
            if (ok_1 != 0) {
                outv[1] = v_0 + key_base;
                last = spos + 1 + 1;
            }
            int v_2 = vals[2];
            int ok_3 = (int)(v_2 >= 0 && v_2 <= bound);
            outv[2] = -1;
            if (ok_3 != 0) {
                outv[2] = v_2 + key_base;
                last = spos + 2 + 1;
            }
            int v_4 = vals[3];
            int ok_5 = (int)(v_4 >= 0 && v_4 <= bound);
            outv[3] = -1;
            if (ok_5 != 0) {
                outv[3] = v_4 + key_base;
                last = spos + 3 + 1;
            }
            int v_6 = vals[4];
            int ok_7 = (int)(v_6 >= 0 && v_6 <= bound);
            outv[4] = -1;
            if (ok_7 != 0) {
                outv[4] = v_6 + key_base;
                last = spos + 4 + 1;
            }
            int v_8 = vals[5];
            int ok_9 = (int)(v_8 >= 0 && v_8 <= bound);
            outv[5] = -1;
            if (ok_9 != 0) {
                outv[5] = v_8 + key_base;
                last = spos + 5 + 1;
            }
            int v_10 = vals[6];
            int ok_11 = (int)(v_10 >= 0 && v_10 <= bound);
            outv[6] = -1;
            if (ok_11 != 0) {
                outv[6] = v_10 + key_base;
                last = spos + 6 + 1;
            }
            int v_12 = vals[7];
            int ok_13 = (int)(v_12 >= 0 && v_12 <= bound);
            outv[7] = -1;
            if (ok_13 != 0) {
                outv[7] = v_12 + key_base;
                last = spos + 7 + 1;
            }
            if (write_out != 0) {
                if (whole != 0) {
                    {
                        int4 _iv4 = make_int4(outv[0 + 0], outv[0 + 1], outv[0 + 2], outv[0 + 3]);
                        *reinterpret_cast<int4*>(out + out_base + (long long)spos) = _iv4;
                    }
                    {
                        int4 _iv4 = make_int4(outv[4 + 0], outv[4 + 1], outv[4 + 2], outv[4 + 3]);
                        *reinterpret_cast<int4*>(out + out_base + (long long)spos + 4) = _iv4;
                    }
                } else {
                    if (spos < topk) {
                        out[out_base + (long long)spos] = outv[0];
                    }
                    if (spos + 1 < topk) {
                        out[out_base + (long long)(spos + 1)] = outv[1];
                    }
                    if (spos + 2 < topk) {
                        out[out_base + (long long)(spos + 2)] = outv[2];
                    }
                    if (spos + 3 < topk) {
                        out[out_base + (long long)(spos + 3)] = outv[3];
                    }
                    if (spos + 4 < topk) {
                        out[out_base + (long long)(spos + 4)] = outv[4];
                    }
                    if (spos + 5 < topk) {
                        out[out_base + (long long)(spos + 5)] = outv[5];
                    }
                    if (spos + 6 < topk) {
                        out[out_base + (long long)(spos + 6)] = outv[6];
                    }
                    if (spos + 7 < topk) {
                        out[out_base + (long long)(spos + 7)] = outv[7];
                    }
                }
            }
        }
        int _warp_redux_i32_0;
        asm volatile("redux.sync.max.s32 %0, %1, 0xffffffff;" : "=r"(_warp_redux_i32_0) : "r"(last));
        int row_len = _warp_redux_i32_0;
        if (lane_0 == 0) {
            topk_length[(long long)t] = row_len;
        }
    }
}

} // extern "C"

