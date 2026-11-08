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


__device__ __forceinline__ unsigned int __as_u32(float v) {
    unsigned int u;
    asm("mov.b32 %0, %1;" : "=r"(u) : "f"(v));
    return u;
}
__device__ __forceinline__ unsigned int __as_u32(__nv_bfloat162 v) {
    return *reinterpret_cast<const unsigned int*>(&v);
}
__device__ __forceinline__ unsigned int __as_u32(unsigned int v) { return v; }
__device__ __forceinline__ unsigned int __as_u32(int v) {
    unsigned int u;
    asm("mov.b32 %0, %1;" : "=r"(u) : "r"(v));
    return u;
}

extern "C" {

__global__ __launch_bounds__(256, LAUNCH_MIN_BLOCKS) void
kernel_cake_dsa_h64_det_nope_train_aedc5c845a7175b21782(int* __restrict__ indices, int* __restrict__ topk_length, int* __restrict__ dkv_dst_map, float* __restrict__ knorm, float* __restrict__ qstat, unsigned int* __restrict__ det_globals, unsigned long long* __restrict__ bound, int num_queries, int num_kv, int num_rows, int topk, int idx_stride, int indices_offset, int has_topk_length, int dkv_has_map, int qstat_stride, int bound_stride, int part_stride)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // === Task calls (declaration order) ===
    int token = blockIdx.x * 8 + warp;
    int lane_1 = lane;
    if (token < num_queries) {
        float scales[1];
        float terms[2];
        float a_max = __uint_as_float(det_globals[0]);
        float b_max = __uint_as_float(det_globals[1]);
        float k_max = __uint_as_float(det_globals[4]);
        float top = __fmaf_rn(b_max, k_max, a_max);
        int exponent = (int)(__as_u32(top) >> 23 & 255) - 127;
        int _max_1 = ((exponent) > (-86) ? (exponent) : (-86));
        int _min_0 = ((_max_1) < (127) ? (_max_1) : (127));
        exponent = _min_0;
        scales[0] = __uint_as_float((unsigned int)(39 - exponent + 127) << 23);
        terms[0] = qstat[(long long)token * (long long)qstat_stride];
        terms[1] = qstat[(long long)token * (long long)qstat_stride + 1ll];
        int active = topk;
        if (has_topk_length != 0) {
            int _max_3 = ((topk_length[token]) > (0) ? (topk_length[token]) : (0));
            int _min_2 = ((_max_3) < (topk) ? (_max_3) : (topk));
            active = _min_2;
        }
        long long row_base = (long long)indices_offset + (long long)token * (long long)idx_stride;
        int aligned4 = (int)(((idx_stride | indices_offset) & 3) == 0);
        #pragma unroll 1
        for (int blk = 0; blk < (active + 127) / 128; blk++) {
            int pos = blk * 128 + lane_1 * 4;
            int keys[4];
            if (aligned4 != 0 && pos + 4 <= active) {
                {
                    const int4* _ivptr_4 = reinterpret_cast<const int4*>(indices + (row_base + (long long)pos) + 0);
                    int4 _ivld_4;
                    _ivld_4 = *_ivptr_4;
                    keys[0 + 0] = _ivld_4.x;
                    keys[0 + 1] = _ivld_4.y;
                    keys[0 + 2] = _ivld_4.z;
                    keys[0 + 3] = _ivld_4.w;
                }
            } else {
                int key_s_0 = -1;
                if (pos < active) {
                    key_s_0 = indices[row_base + (long long)pos];
                }
                keys[0] = key_s_0;
                int key_s_1 = -1;
                if (pos + 1 < active) {
                    key_s_1 = indices[row_base + (long long)(pos + 1)];
                }
                keys[1] = key_s_1;
                int key_s_2 = -1;
                if (pos + 2 < active) {
                    key_s_2 = indices[row_base + (long long)(pos + 2)];
                }
                keys[2] = key_s_2;
                int key_s_3 = -1;
                if (pos + 3 < active) {
                    key_s_3 = indices[row_base + (long long)(pos + 3)];
                }
                keys[3] = key_s_3;
            }
            int key_0 = keys[0];
            if (key_0 >= 0 && key_0 < num_kv) {
                int row_0 = key_0;
                if (dkv_has_map != 0) {
                    row_0 = dkv_dst_map[(long long)key_0];
                }
                if (row_0 >= 0 && row_0 < num_rows) {
                    float kn_0 = knorm[(long long)key_0];
                    float term_0 = __fmaf_rn(terms[1], kn_0, terms[0]);
                    unsigned long long fixed_0 = 1ull;
                    if ((__as_u32(term_0) & 2147483647) < 2139095040) {
                        float scaled_0;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(scaled_0) : "f"(term_0), "f"(scales[0]));
                        unsigned long long whole_0 = (unsigned long long)scaled_0;
                        if ((float)whole_0 < scaled_0) {
                            whole_0 = whole_0 + 1ull;
                        }
                        unsigned long long _max_5 = ((whole_0) > (1ull) ? (whole_0) : (1ull));
                        fixed_0 = _max_5;
                    }
                    asm volatile("red.relaxed.gpu.global.add.u64 [%0], %1;" :: "l"((reinterpret_cast<unsigned long long*>(bound) + ((long long)row_0 * (long long)bound_stride))), "l"(static_cast<unsigned long long>(fixed_0)) : "memory");
                    asm volatile("red.relaxed.gpu.global.add.u64 [%0], %1;" :: "l"((reinterpret_cast<unsigned long long*>(bound) + ((long long)row_0 * (long long)bound_stride + (long long)part_stride))), "l"(static_cast<unsigned long long>(1ull)) : "memory");
                }
            }
            int key_1 = keys[1];
            if (key_1 >= 0 && key_1 < num_kv) {
                int row_1 = key_1;
                if (dkv_has_map != 0) {
                    row_1 = dkv_dst_map[(long long)key_1];
                }
                if (row_1 >= 0 && row_1 < num_rows) {
                    float kn_1 = knorm[(long long)key_1];
                    float term_1 = __fmaf_rn(terms[1], kn_1, terms[0]);
                    unsigned long long fixed_1 = 1ull;
                    if ((__as_u32(term_1) & 2147483647) < 2139095040) {
                        float scaled_1;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(scaled_1) : "f"(term_1), "f"(scales[0]));
                        unsigned long long whole_1 = (unsigned long long)scaled_1;
                        if ((float)whole_1 < scaled_1) {
                            whole_1 = whole_1 + 1ull;
                        }
                        unsigned long long _max_6 = ((whole_1) > (1ull) ? (whole_1) : (1ull));
                        fixed_1 = _max_6;
                    }
                    asm volatile("red.relaxed.gpu.global.add.u64 [%0], %1;" :: "l"((reinterpret_cast<unsigned long long*>(bound) + ((long long)row_1 * (long long)bound_stride))), "l"(static_cast<unsigned long long>(fixed_1)) : "memory");
                    asm volatile("red.relaxed.gpu.global.add.u64 [%0], %1;" :: "l"((reinterpret_cast<unsigned long long*>(bound) + ((long long)row_1 * (long long)bound_stride + (long long)part_stride))), "l"(static_cast<unsigned long long>(1ull)) : "memory");
                }
            }
            int key_2 = keys[2];
            if (key_2 >= 0 && key_2 < num_kv) {
                int row_2 = key_2;
                if (dkv_has_map != 0) {
                    row_2 = dkv_dst_map[(long long)key_2];
                }
                if (row_2 >= 0 && row_2 < num_rows) {
                    float kn_2 = knorm[(long long)key_2];
                    float term_2 = __fmaf_rn(terms[1], kn_2, terms[0]);
                    unsigned long long fixed_2 = 1ull;
                    if ((__as_u32(term_2) & 2147483647) < 2139095040) {
                        float scaled_2;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(scaled_2) : "f"(term_2), "f"(scales[0]));
                        unsigned long long whole_2 = (unsigned long long)scaled_2;
                        if ((float)whole_2 < scaled_2) {
                            whole_2 = whole_2 + 1ull;
                        }
                        unsigned long long _max_7 = ((whole_2) > (1ull) ? (whole_2) : (1ull));
                        fixed_2 = _max_7;
                    }
                    asm volatile("red.relaxed.gpu.global.add.u64 [%0], %1;" :: "l"((reinterpret_cast<unsigned long long*>(bound) + ((long long)row_2 * (long long)bound_stride))), "l"(static_cast<unsigned long long>(fixed_2)) : "memory");
                    asm volatile("red.relaxed.gpu.global.add.u64 [%0], %1;" :: "l"((reinterpret_cast<unsigned long long*>(bound) + ((long long)row_2 * (long long)bound_stride + (long long)part_stride))), "l"(static_cast<unsigned long long>(1ull)) : "memory");
                }
            }
            int key_3 = keys[3];
            if (key_3 >= 0 && key_3 < num_kv) {
                int row_3 = key_3;
                if (dkv_has_map != 0) {
                    row_3 = dkv_dst_map[(long long)key_3];
                }
                if (row_3 >= 0 && row_3 < num_rows) {
                    float kn_3 = knorm[(long long)key_3];
                    float term_3 = __fmaf_rn(terms[1], kn_3, terms[0]);
                    unsigned long long fixed_3 = 1ull;
                    if ((__as_u32(term_3) & 2147483647) < 2139095040) {
                        float scaled_3;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(scaled_3) : "f"(term_3), "f"(scales[0]));
                        unsigned long long whole_3 = (unsigned long long)scaled_3;
                        if ((float)whole_3 < scaled_3) {
                            whole_3 = whole_3 + 1ull;
                        }
                        unsigned long long _max_8 = ((whole_3) > (1ull) ? (whole_3) : (1ull));
                        fixed_3 = _max_8;
                    }
                    asm volatile("red.relaxed.gpu.global.add.u64 [%0], %1;" :: "l"((reinterpret_cast<unsigned long long*>(bound) + ((long long)row_3 * (long long)bound_stride))), "l"(static_cast<unsigned long long>(fixed_3)) : "memory");
                    asm volatile("red.relaxed.gpu.global.add.u64 [%0], %1;" :: "l"((reinterpret_cast<unsigned long long*>(bound) + ((long long)row_3 * (long long)bound_stride + (long long)part_stride))), "l"(static_cast<unsigned long long>(1ull)) : "memory");
                }
            }
        }
    }
}

} // extern "C"
