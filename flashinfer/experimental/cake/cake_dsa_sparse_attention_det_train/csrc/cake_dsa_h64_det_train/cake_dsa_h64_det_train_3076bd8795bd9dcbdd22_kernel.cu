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
kernel_cake_dsa_h64_det_train_3076bd8795bd9dcbdd22(int* __restrict__ indices, int* __restrict__ topk_length, int* __restrict__ dkv_dst_map, float* __restrict__ knorm, float* __restrict__ qstat, unsigned int* __restrict__ det_globals, unsigned long long* __restrict__ bound, int num_queries, int num_kv, int num_rows, int topk, int idx_stride, int indices_offset, int has_topk_length, int dkv_has_map, int qstat_stride, int bound_stride, int part_stride)
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
        int packed = 0;
        packed = (int)((long long)num_queries * (long long)topk < 134217728ll);
        if (token == 0 && lane_1 == 0) {
            det_globals[10] = (unsigned int)packed;
        }
        float scales[3];
        int bits_w_0 = 40;
        float a_max = __uint_as_float(det_globals[0]);
        float b_max = __uint_as_float(det_globals[1]);
        float k_max = __uint_as_float(det_globals[4]);
        float top = __fmaf_rn(b_max, k_max, a_max);
        int exponent = (int)(__as_u32(top) >> 23 & 255) - 127;
        int _max_1 = ((exponent) > (-86) ? (exponent) : (-86));
        int _min_0 = ((_max_1) < (127) ? (_max_1) : (127));
        exponent = _min_0;
        scales[0] = __uint_as_float((unsigned int)(bits_w_0 - 1 - exponent + 127) << 23);
        int bits_w_1 = 40;
        float a_max_0 = __uint_as_float(det_globals[2]);
        float b_max_0 = __uint_as_float(det_globals[3]);
        float k_max_0 = __uint_as_float(det_globals[4]);
        float top_0 = __fmaf_rn(b_max_0, k_max_0, a_max_0);
        int exponent_0 = (int)(__as_u32(top_0) >> 23 & 255) - 127;
        int _max_3 = ((exponent_0) > (-86) ? (exponent_0) : (-86));
        int _min_2 = ((_max_3) < (127) ? (_max_3) : (127));
        exponent_0 = _min_2;
        scales[1] = __uint_as_float((unsigned int)(bits_w_1 - 1 - exponent_0 + 127) << 23);
        int bits_w_2 = 40;
        bits_w_2 = 10;
        float a_max_1 = __uint_as_float(det_globals[0]);
        float b_max_1 = __uint_as_float(det_globals[1]);
        float k_max_1 = __uint_as_float(det_globals[4]);
        float top_1 = __fmaf_rn(b_max_1, k_max_1, a_max_1);
        int exponent_1 = (int)(__as_u32(top_1) >> 23 & 255) - 127;
        int _max_5 = ((exponent_1) > (-86) ? (exponent_1) : (-86));
        int _min_4 = ((_max_5) < (127) ? (_max_5) : (127));
        exponent_1 = _min_4;
        scales[2] = __uint_as_float((unsigned int)(bits_w_2 - 1 - exponent_1 + 127) << 23);
        float terms[4];
        terms[0] = qstat[(long long)token * (long long)qstat_stride];
        terms[1] = qstat[(long long)token * (long long)qstat_stride + 1ll];
        terms[2] = qstat[(long long)token * (long long)qstat_stride + 2ll];
        terms[3] = qstat[(long long)token * (long long)qstat_stride + 3ll];
        int active = topk;
        if (has_topk_length != 0) {
            int _max_7 = ((topk_length[token]) > (0) ? (topk_length[token]) : (0));
            int _min_6 = ((_max_7) < (topk) ? (_max_7) : (topk));
            active = _min_6;
        }
        long long row_base = (long long)indices_offset + (long long)token * (long long)idx_stride;
        int aligned4 = (int)(((idx_stride | indices_offset) & 3) == 0);
        #pragma unroll 1
        for (int blk = 0; blk < (active + 127) / 128; blk++) {
            int pos = blk * 128 + lane_1 * 4;
            int keys[4];
            if (aligned4 != 0 && pos + 4 <= active) {
                {
                    const int4* _ivptr_8 = reinterpret_cast<const int4*>(indices + (row_base + (long long)pos) + 0);
                    int4 _ivld_8;
                    _ivld_8 = *_ivptr_8;
                    keys[0 + 0] = _ivld_8.x;
                    keys[0 + 1] = _ivld_8.y;
                    keys[0 + 2] = _ivld_8.z;
                    keys[0 + 3] = _ivld_8.w;
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
                    long long base_0 = (long long)row_0 * (long long)bound_stride;
                    float term_0 = __fmaf_rn(terms[1], kn_0, terms[0]);
                    int finite_0 = (int)((__as_u32(term_0) & 2147483647) < 2139095040);
                    if (packed != 0) {
                        unsigned int coarse_0 = 0;
                        if (finite_0 != 0) {
                            float scaled_c_0;
                            asm("mul.rn.f32 %0, %1, %2;" : "=f"(scaled_c_0) : "f"(term_0), "f"(scales[2]));
                            unsigned int whole_c_0 = (unsigned int)scaled_c_0;
                            if ((float)whole_c_0 < scaled_c_0) {
                                whole_c_0 = whole_c_0 + 1;
                            }
                            coarse_0 = whole_c_0;
                        }
                        if (coarse_0 >= 64) {
                            asm volatile("red.relaxed.gpu.global.add.u64 [%0], %1;" :: "l"((reinterpret_cast<unsigned long long*>(bound) + (base_0))), "l"(static_cast<unsigned long long>(((unsigned long long)coarse_0 << 27ull) + 1ull)) : "memory");
                        } else {
                            asm volatile("red.relaxed.gpu.global.add.u64 [%0], %1;" :: "l"((reinterpret_cast<unsigned long long*>(bound) + (base_0))), "l"(static_cast<unsigned long long>(1ull)) : "memory");
                            unsigned long long ft_fixed = 1ull;
                            if (finite_0 != 0) {
                                float ft_scaled;
                                asm("mul.rn.f32 %0, %1, %2;" : "=f"(ft_scaled) : "f"(term_0), "f"(scales[0]));
                                unsigned long long ft_whole = (unsigned long long)ft_scaled;
                                if ((float)ft_whole < ft_scaled) {
                                    ft_whole = ft_whole + 1ull;
                                }
                                unsigned long long _max_9 = ((ft_whole) > (1ull) ? (ft_whole) : (1ull));
                                ft_fixed = _max_9;
                            }
                            asm volatile("red.relaxed.gpu.global.add.u64 [%0], %1;" :: "l"((reinterpret_cast<unsigned long long*>(bound) + (base_0 + (long long)part_stride * 2ll))), "l"(static_cast<unsigned long long>(ft_fixed)) : "memory");
                        }
                    } else {
                        unsigned long long ft_fixed_1 = 1ull;
                        if (finite_0 != 0) {
                            float ft_scaled_1;
                            asm("mul.rn.f32 %0, %1, %2;" : "=f"(ft_scaled_1) : "f"(term_0), "f"(scales[0]));
                            unsigned long long ft_whole_1 = (unsigned long long)ft_scaled_1;
                            if ((float)ft_whole_1 < ft_scaled_1) {
                                ft_whole_1 = ft_whole_1 + 1ull;
                            }
                            unsigned long long _max_10 = ((ft_whole_1) > (1ull) ? (ft_whole_1) : (1ull));
                            ft_fixed_1 = _max_10;
                        }
                        asm volatile("red.relaxed.gpu.global.add.u64 [%0], %1;" :: "l"((reinterpret_cast<unsigned long long*>(bound) + (base_0))), "l"(static_cast<unsigned long long>(ft_fixed_1)) : "memory");
                    }
                    float term_1 = __fmaf_rn(terms[3], kn_0, terms[2]);
                    int finite_1 = (int)((__as_u32(term_1) & 2147483647) < 2139095040);
                    unsigned long long ft_fixed_2 = 1ull;
                    if (finite_1 != 0) {
                        float ft_scaled_2;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(ft_scaled_2) : "f"(term_1), "f"(scales[1]));
                        unsigned long long ft_whole_2 = (unsigned long long)ft_scaled_2;
                        if ((float)ft_whole_2 < ft_scaled_2) {
                            ft_whole_2 = ft_whole_2 + 1ull;
                        }
                        unsigned long long _max_11 = ((ft_whole_2) > (1ull) ? (ft_whole_2) : (1ull));
                        ft_fixed_2 = _max_11;
                    }
                    asm volatile("red.relaxed.gpu.global.add.u64 [%0], %1;" :: "l"((reinterpret_cast<unsigned long long*>(bound) + (base_0 + (long long)part_stride))), "l"(static_cast<unsigned long long>(ft_fixed_2)) : "memory");
                    if (packed == 0) {
                        asm volatile("red.relaxed.gpu.global.add.u64 [%0], %1;" :: "l"((reinterpret_cast<unsigned long long*>(bound) + (base_0 + (long long)part_stride * 2ll))), "l"(static_cast<unsigned long long>(1ull)) : "memory");
                    }
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
                    long long base_1 = (long long)row_1 * (long long)bound_stride;
                    float term_2 = __fmaf_rn(terms[1], kn_1, terms[0]);
                    int finite_2 = (int)((__as_u32(term_2) & 2147483647) < 2139095040);
                    if (packed != 0) {
                        unsigned int coarse_1 = 0;
                        if (finite_2 != 0) {
                            float scaled_c_1;
                            asm("mul.rn.f32 %0, %1, %2;" : "=f"(scaled_c_1) : "f"(term_2), "f"(scales[2]));
                            unsigned int whole_c_1 = (unsigned int)scaled_c_1;
                            if ((float)whole_c_1 < scaled_c_1) {
                                whole_c_1 = whole_c_1 + 1;
                            }
                            coarse_1 = whole_c_1;
                        }
                        if (coarse_1 >= 64) {
                            asm volatile("red.relaxed.gpu.global.add.u64 [%0], %1;" :: "l"((reinterpret_cast<unsigned long long*>(bound) + (base_1))), "l"(static_cast<unsigned long long>(((unsigned long long)coarse_1 << 27ull) + 1ull)) : "memory");
                        } else {
                            asm volatile("red.relaxed.gpu.global.add.u64 [%0], %1;" :: "l"((reinterpret_cast<unsigned long long*>(bound) + (base_1))), "l"(static_cast<unsigned long long>(1ull)) : "memory");
                            unsigned long long ft_fixed_3 = 1ull;
                            if (finite_2 != 0) {
                                float ft_scaled_3;
                                asm("mul.rn.f32 %0, %1, %2;" : "=f"(ft_scaled_3) : "f"(term_2), "f"(scales[0]));
                                unsigned long long ft_whole_3 = (unsigned long long)ft_scaled_3;
                                if ((float)ft_whole_3 < ft_scaled_3) {
                                    ft_whole_3 = ft_whole_3 + 1ull;
                                }
                                unsigned long long _max_12 = ((ft_whole_3) > (1ull) ? (ft_whole_3) : (1ull));
                                ft_fixed_3 = _max_12;
                            }
                            asm volatile("red.relaxed.gpu.global.add.u64 [%0], %1;" :: "l"((reinterpret_cast<unsigned long long*>(bound) + (base_1 + (long long)part_stride * 2ll))), "l"(static_cast<unsigned long long>(ft_fixed_3)) : "memory");
                        }
                    } else {
                        unsigned long long ft_fixed_4 = 1ull;
                        if (finite_2 != 0) {
                            float ft_scaled_4;
                            asm("mul.rn.f32 %0, %1, %2;" : "=f"(ft_scaled_4) : "f"(term_2), "f"(scales[0]));
                            unsigned long long ft_whole_4 = (unsigned long long)ft_scaled_4;
                            if ((float)ft_whole_4 < ft_scaled_4) {
                                ft_whole_4 = ft_whole_4 + 1ull;
                            }
                            unsigned long long _max_13 = ((ft_whole_4) > (1ull) ? (ft_whole_4) : (1ull));
                            ft_fixed_4 = _max_13;
                        }
                        asm volatile("red.relaxed.gpu.global.add.u64 [%0], %1;" :: "l"((reinterpret_cast<unsigned long long*>(bound) + (base_1))), "l"(static_cast<unsigned long long>(ft_fixed_4)) : "memory");
                    }
                    float term_3 = __fmaf_rn(terms[3], kn_1, terms[2]);
                    int finite_3 = (int)((__as_u32(term_3) & 2147483647) < 2139095040);
                    unsigned long long ft_fixed_5 = 1ull;
                    if (finite_3 != 0) {
                        float ft_scaled_5;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(ft_scaled_5) : "f"(term_3), "f"(scales[1]));
                        unsigned long long ft_whole_5 = (unsigned long long)ft_scaled_5;
                        if ((float)ft_whole_5 < ft_scaled_5) {
                            ft_whole_5 = ft_whole_5 + 1ull;
                        }
                        unsigned long long _max_14 = ((ft_whole_5) > (1ull) ? (ft_whole_5) : (1ull));
                        ft_fixed_5 = _max_14;
                    }
                    asm volatile("red.relaxed.gpu.global.add.u64 [%0], %1;" :: "l"((reinterpret_cast<unsigned long long*>(bound) + (base_1 + (long long)part_stride))), "l"(static_cast<unsigned long long>(ft_fixed_5)) : "memory");
                    if (packed == 0) {
                        asm volatile("red.relaxed.gpu.global.add.u64 [%0], %1;" :: "l"((reinterpret_cast<unsigned long long*>(bound) + (base_1 + (long long)part_stride * 2ll))), "l"(static_cast<unsigned long long>(1ull)) : "memory");
                    }
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
                    long long base_2 = (long long)row_2 * (long long)bound_stride;
                    float term_4 = __fmaf_rn(terms[1], kn_2, terms[0]);
                    int finite_4 = (int)((__as_u32(term_4) & 2147483647) < 2139095040);
                    if (packed != 0) {
                        unsigned int coarse_2 = 0;
                        if (finite_4 != 0) {
                            float scaled_c_2;
                            asm("mul.rn.f32 %0, %1, %2;" : "=f"(scaled_c_2) : "f"(term_4), "f"(scales[2]));
                            unsigned int whole_c_2 = (unsigned int)scaled_c_2;
                            if ((float)whole_c_2 < scaled_c_2) {
                                whole_c_2 = whole_c_2 + 1;
                            }
                            coarse_2 = whole_c_2;
                        }
                        if (coarse_2 >= 64) {
                            asm volatile("red.relaxed.gpu.global.add.u64 [%0], %1;" :: "l"((reinterpret_cast<unsigned long long*>(bound) + (base_2))), "l"(static_cast<unsigned long long>(((unsigned long long)coarse_2 << 27ull) + 1ull)) : "memory");
                        } else {
                            asm volatile("red.relaxed.gpu.global.add.u64 [%0], %1;" :: "l"((reinterpret_cast<unsigned long long*>(bound) + (base_2))), "l"(static_cast<unsigned long long>(1ull)) : "memory");
                            unsigned long long ft_fixed_6 = 1ull;
                            if (finite_4 != 0) {
                                float ft_scaled_6;
                                asm("mul.rn.f32 %0, %1, %2;" : "=f"(ft_scaled_6) : "f"(term_4), "f"(scales[0]));
                                unsigned long long ft_whole_6 = (unsigned long long)ft_scaled_6;
                                if ((float)ft_whole_6 < ft_scaled_6) {
                                    ft_whole_6 = ft_whole_6 + 1ull;
                                }
                                unsigned long long _max_15 = ((ft_whole_6) > (1ull) ? (ft_whole_6) : (1ull));
                                ft_fixed_6 = _max_15;
                            }
                            asm volatile("red.relaxed.gpu.global.add.u64 [%0], %1;" :: "l"((reinterpret_cast<unsigned long long*>(bound) + (base_2 + (long long)part_stride * 2ll))), "l"(static_cast<unsigned long long>(ft_fixed_6)) : "memory");
                        }
                    } else {
                        unsigned long long ft_fixed_7 = 1ull;
                        if (finite_4 != 0) {
                            float ft_scaled_7;
                            asm("mul.rn.f32 %0, %1, %2;" : "=f"(ft_scaled_7) : "f"(term_4), "f"(scales[0]));
                            unsigned long long ft_whole_7 = (unsigned long long)ft_scaled_7;
                            if ((float)ft_whole_7 < ft_scaled_7) {
                                ft_whole_7 = ft_whole_7 + 1ull;
                            }
                            unsigned long long _max_16 = ((ft_whole_7) > (1ull) ? (ft_whole_7) : (1ull));
                            ft_fixed_7 = _max_16;
                        }
                        asm volatile("red.relaxed.gpu.global.add.u64 [%0], %1;" :: "l"((reinterpret_cast<unsigned long long*>(bound) + (base_2))), "l"(static_cast<unsigned long long>(ft_fixed_7)) : "memory");
                    }
                    float term_5 = __fmaf_rn(terms[3], kn_2, terms[2]);
                    int finite_5 = (int)((__as_u32(term_5) & 2147483647) < 2139095040);
                    unsigned long long ft_fixed_8 = 1ull;
                    if (finite_5 != 0) {
                        float ft_scaled_8;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(ft_scaled_8) : "f"(term_5), "f"(scales[1]));
                        unsigned long long ft_whole_8 = (unsigned long long)ft_scaled_8;
                        if ((float)ft_whole_8 < ft_scaled_8) {
                            ft_whole_8 = ft_whole_8 + 1ull;
                        }
                        unsigned long long _max_17 = ((ft_whole_8) > (1ull) ? (ft_whole_8) : (1ull));
                        ft_fixed_8 = _max_17;
                    }
                    asm volatile("red.relaxed.gpu.global.add.u64 [%0], %1;" :: "l"((reinterpret_cast<unsigned long long*>(bound) + (base_2 + (long long)part_stride))), "l"(static_cast<unsigned long long>(ft_fixed_8)) : "memory");
                    if (packed == 0) {
                        asm volatile("red.relaxed.gpu.global.add.u64 [%0], %1;" :: "l"((reinterpret_cast<unsigned long long*>(bound) + (base_2 + (long long)part_stride * 2ll))), "l"(static_cast<unsigned long long>(1ull)) : "memory");
                    }
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
                    long long base_3 = (long long)row_3 * (long long)bound_stride;
                    float term_6 = __fmaf_rn(terms[1], kn_3, terms[0]);
                    int finite_6 = (int)((__as_u32(term_6) & 2147483647) < 2139095040);
                    if (packed != 0) {
                        unsigned int coarse_3 = 0;
                        if (finite_6 != 0) {
                            float scaled_c_3;
                            asm("mul.rn.f32 %0, %1, %2;" : "=f"(scaled_c_3) : "f"(term_6), "f"(scales[2]));
                            unsigned int whole_c_3 = (unsigned int)scaled_c_3;
                            if ((float)whole_c_3 < scaled_c_3) {
                                whole_c_3 = whole_c_3 + 1;
                            }
                            coarse_3 = whole_c_3;
                        }
                        if (coarse_3 >= 64) {
                            asm volatile("red.relaxed.gpu.global.add.u64 [%0], %1;" :: "l"((reinterpret_cast<unsigned long long*>(bound) + (base_3))), "l"(static_cast<unsigned long long>(((unsigned long long)coarse_3 << 27ull) + 1ull)) : "memory");
                        } else {
                            asm volatile("red.relaxed.gpu.global.add.u64 [%0], %1;" :: "l"((reinterpret_cast<unsigned long long*>(bound) + (base_3))), "l"(static_cast<unsigned long long>(1ull)) : "memory");
                            unsigned long long ft_fixed_9 = 1ull;
                            if (finite_6 != 0) {
                                float ft_scaled_9;
                                asm("mul.rn.f32 %0, %1, %2;" : "=f"(ft_scaled_9) : "f"(term_6), "f"(scales[0]));
                                unsigned long long ft_whole_9 = (unsigned long long)ft_scaled_9;
                                if ((float)ft_whole_9 < ft_scaled_9) {
                                    ft_whole_9 = ft_whole_9 + 1ull;
                                }
                                unsigned long long _max_18 = ((ft_whole_9) > (1ull) ? (ft_whole_9) : (1ull));
                                ft_fixed_9 = _max_18;
                            }
                            asm volatile("red.relaxed.gpu.global.add.u64 [%0], %1;" :: "l"((reinterpret_cast<unsigned long long*>(bound) + (base_3 + (long long)part_stride * 2ll))), "l"(static_cast<unsigned long long>(ft_fixed_9)) : "memory");
                        }
                    } else {
                        unsigned long long ft_fixed_10 = 1ull;
                        if (finite_6 != 0) {
                            float ft_scaled_10;
                            asm("mul.rn.f32 %0, %1, %2;" : "=f"(ft_scaled_10) : "f"(term_6), "f"(scales[0]));
                            unsigned long long ft_whole_10 = (unsigned long long)ft_scaled_10;
                            if ((float)ft_whole_10 < ft_scaled_10) {
                                ft_whole_10 = ft_whole_10 + 1ull;
                            }
                            unsigned long long _max_19 = ((ft_whole_10) > (1ull) ? (ft_whole_10) : (1ull));
                            ft_fixed_10 = _max_19;
                        }
                        asm volatile("red.relaxed.gpu.global.add.u64 [%0], %1;" :: "l"((reinterpret_cast<unsigned long long*>(bound) + (base_3))), "l"(static_cast<unsigned long long>(ft_fixed_10)) : "memory");
                    }
                    float term_7 = __fmaf_rn(terms[3], kn_3, terms[2]);
                    int finite_7 = (int)((__as_u32(term_7) & 2147483647) < 2139095040);
                    unsigned long long ft_fixed_11 = 1ull;
                    if (finite_7 != 0) {
                        float ft_scaled_11;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(ft_scaled_11) : "f"(term_7), "f"(scales[1]));
                        unsigned long long ft_whole_11 = (unsigned long long)ft_scaled_11;
                        if ((float)ft_whole_11 < ft_scaled_11) {
                            ft_whole_11 = ft_whole_11 + 1ull;
                        }
                        unsigned long long _max_20 = ((ft_whole_11) > (1ull) ? (ft_whole_11) : (1ull));
                        ft_fixed_11 = _max_20;
                    }
                    asm volatile("red.relaxed.gpu.global.add.u64 [%0], %1;" :: "l"((reinterpret_cast<unsigned long long*>(bound) + (base_3 + (long long)part_stride))), "l"(static_cast<unsigned long long>(ft_fixed_11)) : "memory");
                    if (packed == 0) {
                        asm volatile("red.relaxed.gpu.global.add.u64 [%0], %1;" :: "l"((reinterpret_cast<unsigned long long*>(bound) + (base_3 + (long long)part_stride * 2ll))), "l"(static_cast<unsigned long long>(1ull)) : "memory");
                    }
                }
            }
        }
    }
}

} // extern "C"
