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

#define LAUNCH_MIN_BLOCKS 5

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
kernel_cake_dsa_h64_det_nope_train_cd3686becf1223e8ca1b(int* __restrict__ indices, int* __restrict__ topk_length, int* __restrict__ dkv_dst_map, float* __restrict__ knorm, float* __restrict__ qstat, unsigned int* __restrict__ det_globals, unsigned long long* __restrict__ bound, int num_queries, int num_kv, int num_rows, int topk, int idx_stride, int indices_offset, int has_topk_length, int dkv_has_map, int qstat_stride, int bound_stride, int part_stride)
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
        packed = (int)((long long)num_queries * (long long)topk < 67108864ll);
        if (token == 0 && lane_1 == 0) {
            det_globals[10] = (unsigned int)packed;
        }
        float scales[2];
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
        bits_w_1 = 12;
        float a_max_0 = __uint_as_float(det_globals[0]);
        float b_max_0 = __uint_as_float(det_globals[1]);
        float k_max_0 = __uint_as_float(det_globals[4]);
        float top_0 = __fmaf_rn(b_max_0, k_max_0, a_max_0);
        int exponent_0 = (int)(__as_u32(top_0) >> 23 & 255) - 127;
        int _max_3 = ((exponent_0) > (-86) ? (exponent_0) : (-86));
        int _min_2 = ((_max_3) < (127) ? (_max_3) : (127));
        exponent_0 = _min_2;
        scales[1] = __uint_as_float((unsigned int)(bits_w_1 - 1 - exponent_0 + 127) << 23);
        float terms[2];
        terms[0] = qstat[(long long)token * (long long)qstat_stride];
        terms[1] = qstat[(long long)token * (long long)qstat_stride + 1ll];
        int active = topk;
        if (has_topk_length != 0) {
            int _max_5 = ((topk_length[token]) > (0) ? (topk_length[token]) : (0));
            int _min_4 = ((_max_5) < (topk) ? (_max_5) : (topk));
            active = _min_4;
        }
        long long row_base = (long long)indices_offset + (long long)token * (long long)idx_stride;
        int aligned4 = (int)(((idx_stride | indices_offset) & 3) == 0);
        int blocks = (active + 127) / 128;
        int rot = 0;
        if (blocks > 0) {
            rot = token % blocks;
        }
        int nblk = (active + 127) / 128;
        int cur[8];
        int nxt[8];
        int pos0_0 = lane_1 * 4;
        int rb0_0 = rot;
        if (rb0_0 >= blocks) {
            rb0_0 = rb0_0 - blocks;
        }
        pos0_0 = rb0_0 * 128 + lane_1 * 4;
        if (0 < nblk && aligned4 != 0 && pos0_0 + 4 <= active) {
            {
                const int4* _ivptr_6 = reinterpret_cast<const int4*>(indices + (row_base + (long long)pos0_0) + 0);
                int4 _ivld_6;
                _ivld_6 = *_ivptr_6;
                nxt[0 + 0] = _ivld_6.x;
                nxt[0 + 1] = _ivld_6.y;
                nxt[0 + 2] = _ivld_6.z;
                nxt[0 + 3] = _ivld_6.w;
            }
        } else {
            int key0_0 = -1;
            if (0 < nblk && pos0_0 < active) {
                key0_0 = indices[row_base + (long long)pos0_0];
            }
            nxt[0] = key0_0;
            int key0_1 = -1;
            if (0 < nblk && pos0_0 + 1 < active) {
                key0_1 = indices[row_base + (long long)(pos0_0 + 1)];
            }
            nxt[1] = key0_1;
            int key0_2 = -1;
            if (0 < nblk && pos0_0 + 2 < active) {
                key0_2 = indices[row_base + (long long)(pos0_0 + 2)];
            }
            nxt[2] = key0_2;
            int key0_3 = -1;
            if (0 < nblk && pos0_0 + 3 < active) {
                key0_3 = indices[row_base + (long long)(pos0_0 + 3)];
            }
            nxt[3] = key0_3;
        }
        int pos0_1 = 128 + lane_1 * 4;
        int rb0_1 = 1 + rot;
        if (rb0_1 >= blocks) {
            rb0_1 = rb0_1 - blocks;
        }
        pos0_1 = rb0_1 * 128 + lane_1 * 4;
        if (1 < nblk && aligned4 != 0 && pos0_1 + 4 <= active) {
            {
                const int4* _ivptr_7 = reinterpret_cast<const int4*>(indices + (row_base + (long long)pos0_1) + 0);
                int4 _ivld_7;
                _ivld_7 = *_ivptr_7;
                nxt[4 + 0] = _ivld_7.x;
                nxt[4 + 1] = _ivld_7.y;
                nxt[4 + 2] = _ivld_7.z;
                nxt[4 + 3] = _ivld_7.w;
            }
        } else {
            int key0_4 = -1;
            if (1 < nblk && pos0_1 < active) {
                key0_4 = indices[row_base + (long long)pos0_1];
            }
            nxt[4] = key0_4;
            int key0_5 = -1;
            if (1 < nblk && pos0_1 + 1 < active) {
                key0_5 = indices[row_base + (long long)(pos0_1 + 1)];
            }
            nxt[5] = key0_5;
            int key0_6 = -1;
            if (1 < nblk && pos0_1 + 2 < active) {
                key0_6 = indices[row_base + (long long)(pos0_1 + 2)];
            }
            nxt[6] = key0_6;
            int key0_7 = -1;
            if (1 < nblk && pos0_1 + 3 < active) {
                key0_7 = indices[row_base + (long long)(pos0_1 + 3)];
            }
            nxt[7] = key0_7;
        }
        #pragma unroll 1
        for (int it = 0; it < (nblk + 1) / 2; it++) {
            cur[0] = nxt[0];
            cur[1] = nxt[1];
            cur[2] = nxt[2];
            cur[3] = nxt[3];
            cur[4] = nxt[4];
            cur[5] = nxt[5];
            cur[6] = nxt[6];
            cur[7] = nxt[7];
            int b1_0 = 2 * it + 2;
            int pos1_0 = b1_0 * 128 + lane_1 * 4;
            int rb1_0 = b1_0 + rot;
            if (rb1_0 >= blocks) {
                rb1_0 = rb1_0 - blocks;
            }
            pos1_0 = rb1_0 * 128 + lane_1 * 4;
            if (b1_0 < nblk && aligned4 != 0 && pos1_0 + 4 <= active) {
                {
                    const int4* _ivptr_8 = reinterpret_cast<const int4*>(indices + (row_base + (long long)pos1_0) + 0);
                    int4 _ivld_8;
                    _ivld_8 = *_ivptr_8;
                    nxt[0 + 0] = _ivld_8.x;
                    nxt[0 + 1] = _ivld_8.y;
                    nxt[0 + 2] = _ivld_8.z;
                    nxt[0 + 3] = _ivld_8.w;
                }
            } else {
                int key3_0 = -1;
                if (b1_0 < nblk && pos1_0 < active) {
                    key3_0 = indices[row_base + (long long)pos1_0];
                }
                nxt[0] = key3_0;
                int key3_1 = -1;
                if (b1_0 < nblk && pos1_0 + 1 < active) {
                    key3_1 = indices[row_base + (long long)(pos1_0 + 1)];
                }
                nxt[1] = key3_1;
                int key3_2 = -1;
                if (b1_0 < nblk && pos1_0 + 2 < active) {
                    key3_2 = indices[row_base + (long long)(pos1_0 + 2)];
                }
                nxt[2] = key3_2;
                int key3_3 = -1;
                if (b1_0 < nblk && pos1_0 + 3 < active) {
                    key3_3 = indices[row_base + (long long)(pos1_0 + 3)];
                }
                nxt[3] = key3_3;
            }
            int b1_1 = 2 * it + 2 + 1;
            int pos1_1 = b1_1 * 128 + lane_1 * 4;
            int rb1_1 = b1_1 + rot;
            if (rb1_1 >= blocks) {
                rb1_1 = rb1_1 - blocks;
            }
            pos1_1 = rb1_1 * 128 + lane_1 * 4;
            if (b1_1 < nblk && aligned4 != 0 && pos1_1 + 4 <= active) {
                {
                    const int4* _ivptr_9 = reinterpret_cast<const int4*>(indices + (row_base + (long long)pos1_1) + 0);
                    int4 _ivld_9;
                    _ivld_9 = *_ivptr_9;
                    nxt[4 + 0] = _ivld_9.x;
                    nxt[4 + 1] = _ivld_9.y;
                    nxt[4 + 2] = _ivld_9.z;
                    nxt[4 + 3] = _ivld_9.w;
                }
            } else {
                int key3_4 = -1;
                if (b1_1 < nblk && pos1_1 < active) {
                    key3_4 = indices[row_base + (long long)pos1_1];
                }
                nxt[4] = key3_4;
                int key3_5 = -1;
                if (b1_1 < nblk && pos1_1 + 1 < active) {
                    key3_5 = indices[row_base + (long long)(pos1_1 + 1)];
                }
                nxt[5] = key3_5;
                int key3_6 = -1;
                if (b1_1 < nblk && pos1_1 + 2 < active) {
                    key3_6 = indices[row_base + (long long)(pos1_1 + 2)];
                }
                nxt[6] = key3_6;
                int key3_7 = -1;
                if (b1_1 < nblk && pos1_1 + 3 < active) {
                    key3_7 = indices[row_base + (long long)(pos1_1 + 3)];
                }
                nxt[7] = key3_7;
            }
            int rows8[8];
            float kns8[8];
            int key4_0 = cur[0];
            int row4_0 = -1;
            float kn4_0 = 0.0f;
            if (key4_0 >= 0 && key4_0 < num_kv) {
                row4_0 = key4_0;
                if (dkv_has_map != 0) {
                    row4_0 = dkv_dst_map[(long long)key4_0];
                }
                kn4_0 = knorm[(long long)key4_0];
            }
            rows8[0] = row4_0;
            kns8[0] = kn4_0;
            int key4_1 = cur[1];
            int row4_1 = -1;
            float kn4_1 = 0.0f;
            if (key4_1 >= 0 && key4_1 < num_kv) {
                row4_1 = key4_1;
                if (dkv_has_map != 0) {
                    row4_1 = dkv_dst_map[(long long)key4_1];
                }
                kn4_1 = knorm[(long long)key4_1];
            }
            rows8[1] = row4_1;
            kns8[1] = kn4_1;
            int key4_2 = cur[2];
            int row4_2 = -1;
            float kn4_2 = 0.0f;
            if (key4_2 >= 0 && key4_2 < num_kv) {
                row4_2 = key4_2;
                if (dkv_has_map != 0) {
                    row4_2 = dkv_dst_map[(long long)key4_2];
                }
                kn4_2 = knorm[(long long)key4_2];
            }
            rows8[2] = row4_2;
            kns8[2] = kn4_2;
            int key4_3 = cur[3];
            int row4_3 = -1;
            float kn4_3 = 0.0f;
            if (key4_3 >= 0 && key4_3 < num_kv) {
                row4_3 = key4_3;
                if (dkv_has_map != 0) {
                    row4_3 = dkv_dst_map[(long long)key4_3];
                }
                kn4_3 = knorm[(long long)key4_3];
            }
            rows8[3] = row4_3;
            kns8[3] = kn4_3;
            int key4_4 = cur[4];
            int row4_4 = -1;
            float kn4_4 = 0.0f;
            if (key4_4 >= 0 && key4_4 < num_kv) {
                row4_4 = key4_4;
                if (dkv_has_map != 0) {
                    row4_4 = dkv_dst_map[(long long)key4_4];
                }
                kn4_4 = knorm[(long long)key4_4];
            }
            rows8[4] = row4_4;
            kns8[4] = kn4_4;
            int key4_5 = cur[5];
            int row4_5 = -1;
            float kn4_5 = 0.0f;
            if (key4_5 >= 0 && key4_5 < num_kv) {
                row4_5 = key4_5;
                if (dkv_has_map != 0) {
                    row4_5 = dkv_dst_map[(long long)key4_5];
                }
                kn4_5 = knorm[(long long)key4_5];
            }
            rows8[5] = row4_5;
            kns8[5] = kn4_5;
            int key4_6 = cur[6];
            int row4_6 = -1;
            float kn4_6 = 0.0f;
            if (key4_6 >= 0 && key4_6 < num_kv) {
                row4_6 = key4_6;
                if (dkv_has_map != 0) {
                    row4_6 = dkv_dst_map[(long long)key4_6];
                }
                kn4_6 = knorm[(long long)key4_6];
            }
            rows8[6] = row4_6;
            kns8[6] = kn4_6;
            int key4_7 = cur[7];
            int row4_7 = -1;
            float kn4_7 = 0.0f;
            if (key4_7 >= 0 && key4_7 < num_kv) {
                row4_7 = key4_7;
                if (dkv_has_map != 0) {
                    row4_7 = dkv_dst_map[(long long)key4_7];
                }
                kn4_7 = knorm[(long long)key4_7];
            }
            rows8[7] = row4_7;
            kns8[7] = kn4_7;
            int row5_0 = rows8[0];
            if (row5_0 >= 0 && row5_0 < num_rows) {
                float kn5_0 = kns8[0];
                long long base5_0 = (long long)row5_0 * (long long)bound_stride;
                float term5_0 = __fmaf_rn(terms[1], kn5_0, terms[0]);
                int finite5_0 = (int)((__as_u32(term5_0) & 2147483647) < 2139095040);
                if (packed != 0) {
                    unsigned int coarse5_0 = 0;
                    if (finite5_0 != 0) {
                        float ct_scaled;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(ct_scaled) : "f"(term5_0), "f"(scales[1]));
                        float ct_ceil = ct_scaled;
                        if (ct_scaled < 8.388608e+06f) {
                            float _sub_10;
                            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_10) : "f"(__fmaf_ru(term5_0, scales[1], 8.388608e+06f)), "f"(8.388608e+06f));
                            ct_ceil = _sub_10;
                        }
                        coarse5_0 = (unsigned int)ct_ceil;
                    }
                    if (coarse5_0 >= 64) {
                        asm volatile("red.relaxed.gpu.global.add.u64 [%0], %1;" :: "l"((reinterpret_cast<unsigned long long*>(bound) + (base5_0))), "l"(static_cast<unsigned long long>(((unsigned long long)coarse5_0 << 26ull) + 1ull)) : "memory");
                    } else {
                        asm volatile("red.relaxed.gpu.global.add.u64 [%0], %1;" :: "l"((reinterpret_cast<unsigned long long*>(bound) + (base5_0))), "l"(static_cast<unsigned long long>(1ull)) : "memory");
                        unsigned long long fc_fixed = 1ull;
                        if (finite5_0 != 0) {
                            float ct_scaled_1;
                            asm("mul.rn.f32 %0, %1, %2;" : "=f"(ct_scaled_1) : "f"(term5_0), "f"(scales[0]));
                            float ct_ceil_1 = ct_scaled_1;
                            if (ct_scaled_1 < 8.388608e+06f) {
                                float _sub_11;
                                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_11) : "f"(__fmaf_ru(term5_0, scales[0], 8.388608e+06f)), "f"(8.388608e+06f));
                                ct_ceil_1 = _sub_11;
                            }
                            unsigned long long _max_12 = (((unsigned long long)ct_ceil_1) > (1ull) ? ((unsigned long long)ct_ceil_1) : (1ull));
                            fc_fixed = _max_12;
                        }
                        asm volatile("red.relaxed.gpu.global.add.u64 [%0], %1;" :: "l"((reinterpret_cast<unsigned long long*>(bound) + (base5_0 + (long long)part_stride))), "l"(static_cast<unsigned long long>(fc_fixed)) : "memory");
                    }
                } else {
                    unsigned long long fc_fixed_1 = 1ull;
                    if (finite5_0 != 0) {
                        float ct_scaled_2;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(ct_scaled_2) : "f"(term5_0), "f"(scales[0]));
                        float ct_ceil_2 = ct_scaled_2;
                        if (ct_scaled_2 < 8.388608e+06f) {
                            float _sub_13;
                            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_13) : "f"(__fmaf_ru(term5_0, scales[0], 8.388608e+06f)), "f"(8.388608e+06f));
                            ct_ceil_2 = _sub_13;
                        }
                        unsigned long long _max_14 = (((unsigned long long)ct_ceil_2) > (1ull) ? ((unsigned long long)ct_ceil_2) : (1ull));
                        fc_fixed_1 = _max_14;
                    }
                    asm volatile("red.relaxed.gpu.global.add.u64 [%0], %1;" :: "l"((reinterpret_cast<unsigned long long*>(bound) + (base5_0))), "l"(static_cast<unsigned long long>(fc_fixed_1)) : "memory");
                }
                if (packed == 0) {
                    asm volatile("red.relaxed.gpu.global.add.u64 [%0], %1;" :: "l"((reinterpret_cast<unsigned long long*>(bound) + (base5_0 + (long long)part_stride))), "l"(static_cast<unsigned long long>(1ull)) : "memory");
                }
            }
            int row5_1 = rows8[1];
            if (row5_1 >= 0 && row5_1 < num_rows) {
                float kn5_1 = kns8[1];
                long long base5_1 = (long long)row5_1 * (long long)bound_stride;
                float term5_1 = __fmaf_rn(terms[1], kn5_1, terms[0]);
                int finite5_1 = (int)((__as_u32(term5_1) & 2147483647) < 2139095040);
                if (packed != 0) {
                    unsigned int coarse5_1 = 0;
                    if (finite5_1 != 0) {
                        float ct_scaled_3;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(ct_scaled_3) : "f"(term5_1), "f"(scales[1]));
                        float ct_ceil_3 = ct_scaled_3;
                        if (ct_scaled_3 < 8.388608e+06f) {
                            float _sub_15;
                            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_15) : "f"(__fmaf_ru(term5_1, scales[1], 8.388608e+06f)), "f"(8.388608e+06f));
                            ct_ceil_3 = _sub_15;
                        }
                        coarse5_1 = (unsigned int)ct_ceil_3;
                    }
                    if (coarse5_1 >= 64) {
                        asm volatile("red.relaxed.gpu.global.add.u64 [%0], %1;" :: "l"((reinterpret_cast<unsigned long long*>(bound) + (base5_1))), "l"(static_cast<unsigned long long>(((unsigned long long)coarse5_1 << 26ull) + 1ull)) : "memory");
                    } else {
                        asm volatile("red.relaxed.gpu.global.add.u64 [%0], %1;" :: "l"((reinterpret_cast<unsigned long long*>(bound) + (base5_1))), "l"(static_cast<unsigned long long>(1ull)) : "memory");
                        unsigned long long fc_fixed_2 = 1ull;
                        if (finite5_1 != 0) {
                            float ct_scaled_4;
                            asm("mul.rn.f32 %0, %1, %2;" : "=f"(ct_scaled_4) : "f"(term5_1), "f"(scales[0]));
                            float ct_ceil_4 = ct_scaled_4;
                            if (ct_scaled_4 < 8.388608e+06f) {
                                float _sub_16;
                                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_16) : "f"(__fmaf_ru(term5_1, scales[0], 8.388608e+06f)), "f"(8.388608e+06f));
                                ct_ceil_4 = _sub_16;
                            }
                            unsigned long long _max_17 = (((unsigned long long)ct_ceil_4) > (1ull) ? ((unsigned long long)ct_ceil_4) : (1ull));
                            fc_fixed_2 = _max_17;
                        }
                        asm volatile("red.relaxed.gpu.global.add.u64 [%0], %1;" :: "l"((reinterpret_cast<unsigned long long*>(bound) + (base5_1 + (long long)part_stride))), "l"(static_cast<unsigned long long>(fc_fixed_2)) : "memory");
                    }
                } else {
                    unsigned long long fc_fixed_3 = 1ull;
                    if (finite5_1 != 0) {
                        float ct_scaled_5;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(ct_scaled_5) : "f"(term5_1), "f"(scales[0]));
                        float ct_ceil_5 = ct_scaled_5;
                        if (ct_scaled_5 < 8.388608e+06f) {
                            float _sub_18;
                            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_18) : "f"(__fmaf_ru(term5_1, scales[0], 8.388608e+06f)), "f"(8.388608e+06f));
                            ct_ceil_5 = _sub_18;
                        }
                        unsigned long long _max_19 = (((unsigned long long)ct_ceil_5) > (1ull) ? ((unsigned long long)ct_ceil_5) : (1ull));
                        fc_fixed_3 = _max_19;
                    }
                    asm volatile("red.relaxed.gpu.global.add.u64 [%0], %1;" :: "l"((reinterpret_cast<unsigned long long*>(bound) + (base5_1))), "l"(static_cast<unsigned long long>(fc_fixed_3)) : "memory");
                }
                if (packed == 0) {
                    asm volatile("red.relaxed.gpu.global.add.u64 [%0], %1;" :: "l"((reinterpret_cast<unsigned long long*>(bound) + (base5_1 + (long long)part_stride))), "l"(static_cast<unsigned long long>(1ull)) : "memory");
                }
            }
            int row5_2 = rows8[2];
            if (row5_2 >= 0 && row5_2 < num_rows) {
                float kn5_2 = kns8[2];
                long long base5_2 = (long long)row5_2 * (long long)bound_stride;
                float term5_2 = __fmaf_rn(terms[1], kn5_2, terms[0]);
                int finite5_2 = (int)((__as_u32(term5_2) & 2147483647) < 2139095040);
                if (packed != 0) {
                    unsigned int coarse5_2 = 0;
                    if (finite5_2 != 0) {
                        float ct_scaled_6;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(ct_scaled_6) : "f"(term5_2), "f"(scales[1]));
                        float ct_ceil_6 = ct_scaled_6;
                        if (ct_scaled_6 < 8.388608e+06f) {
                            float _sub_20;
                            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_20) : "f"(__fmaf_ru(term5_2, scales[1], 8.388608e+06f)), "f"(8.388608e+06f));
                            ct_ceil_6 = _sub_20;
                        }
                        coarse5_2 = (unsigned int)ct_ceil_6;
                    }
                    if (coarse5_2 >= 64) {
                        asm volatile("red.relaxed.gpu.global.add.u64 [%0], %1;" :: "l"((reinterpret_cast<unsigned long long*>(bound) + (base5_2))), "l"(static_cast<unsigned long long>(((unsigned long long)coarse5_2 << 26ull) + 1ull)) : "memory");
                    } else {
                        asm volatile("red.relaxed.gpu.global.add.u64 [%0], %1;" :: "l"((reinterpret_cast<unsigned long long*>(bound) + (base5_2))), "l"(static_cast<unsigned long long>(1ull)) : "memory");
                        unsigned long long fc_fixed_4 = 1ull;
                        if (finite5_2 != 0) {
                            float ct_scaled_7;
                            asm("mul.rn.f32 %0, %1, %2;" : "=f"(ct_scaled_7) : "f"(term5_2), "f"(scales[0]));
                            float ct_ceil_7 = ct_scaled_7;
                            if (ct_scaled_7 < 8.388608e+06f) {
                                float _sub_21;
                                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_21) : "f"(__fmaf_ru(term5_2, scales[0], 8.388608e+06f)), "f"(8.388608e+06f));
                                ct_ceil_7 = _sub_21;
                            }
                            unsigned long long _max_22 = (((unsigned long long)ct_ceil_7) > (1ull) ? ((unsigned long long)ct_ceil_7) : (1ull));
                            fc_fixed_4 = _max_22;
                        }
                        asm volatile("red.relaxed.gpu.global.add.u64 [%0], %1;" :: "l"((reinterpret_cast<unsigned long long*>(bound) + (base5_2 + (long long)part_stride))), "l"(static_cast<unsigned long long>(fc_fixed_4)) : "memory");
                    }
                } else {
                    unsigned long long fc_fixed_5 = 1ull;
                    if (finite5_2 != 0) {
                        float ct_scaled_8;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(ct_scaled_8) : "f"(term5_2), "f"(scales[0]));
                        float ct_ceil_8 = ct_scaled_8;
                        if (ct_scaled_8 < 8.388608e+06f) {
                            float _sub_23;
                            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_23) : "f"(__fmaf_ru(term5_2, scales[0], 8.388608e+06f)), "f"(8.388608e+06f));
                            ct_ceil_8 = _sub_23;
                        }
                        unsigned long long _max_24 = (((unsigned long long)ct_ceil_8) > (1ull) ? ((unsigned long long)ct_ceil_8) : (1ull));
                        fc_fixed_5 = _max_24;
                    }
                    asm volatile("red.relaxed.gpu.global.add.u64 [%0], %1;" :: "l"((reinterpret_cast<unsigned long long*>(bound) + (base5_2))), "l"(static_cast<unsigned long long>(fc_fixed_5)) : "memory");
                }
                if (packed == 0) {
                    asm volatile("red.relaxed.gpu.global.add.u64 [%0], %1;" :: "l"((reinterpret_cast<unsigned long long*>(bound) + (base5_2 + (long long)part_stride))), "l"(static_cast<unsigned long long>(1ull)) : "memory");
                }
            }
            int row5_3 = rows8[3];
            if (row5_3 >= 0 && row5_3 < num_rows) {
                float kn5_3 = kns8[3];
                long long base5_3 = (long long)row5_3 * (long long)bound_stride;
                float term5_3 = __fmaf_rn(terms[1], kn5_3, terms[0]);
                int finite5_3 = (int)((__as_u32(term5_3) & 2147483647) < 2139095040);
                if (packed != 0) {
                    unsigned int coarse5_3 = 0;
                    if (finite5_3 != 0) {
                        float ct_scaled_9;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(ct_scaled_9) : "f"(term5_3), "f"(scales[1]));
                        float ct_ceil_9 = ct_scaled_9;
                        if (ct_scaled_9 < 8.388608e+06f) {
                            float _sub_25;
                            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_25) : "f"(__fmaf_ru(term5_3, scales[1], 8.388608e+06f)), "f"(8.388608e+06f));
                            ct_ceil_9 = _sub_25;
                        }
                        coarse5_3 = (unsigned int)ct_ceil_9;
                    }
                    if (coarse5_3 >= 64) {
                        asm volatile("red.relaxed.gpu.global.add.u64 [%0], %1;" :: "l"((reinterpret_cast<unsigned long long*>(bound) + (base5_3))), "l"(static_cast<unsigned long long>(((unsigned long long)coarse5_3 << 26ull) + 1ull)) : "memory");
                    } else {
                        asm volatile("red.relaxed.gpu.global.add.u64 [%0], %1;" :: "l"((reinterpret_cast<unsigned long long*>(bound) + (base5_3))), "l"(static_cast<unsigned long long>(1ull)) : "memory");
                        unsigned long long fc_fixed_6 = 1ull;
                        if (finite5_3 != 0) {
                            float ct_scaled_10;
                            asm("mul.rn.f32 %0, %1, %2;" : "=f"(ct_scaled_10) : "f"(term5_3), "f"(scales[0]));
                            float ct_ceil_10 = ct_scaled_10;
                            if (ct_scaled_10 < 8.388608e+06f) {
                                float _sub_26;
                                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_26) : "f"(__fmaf_ru(term5_3, scales[0], 8.388608e+06f)), "f"(8.388608e+06f));
                                ct_ceil_10 = _sub_26;
                            }
                            unsigned long long _max_27 = (((unsigned long long)ct_ceil_10) > (1ull) ? ((unsigned long long)ct_ceil_10) : (1ull));
                            fc_fixed_6 = _max_27;
                        }
                        asm volatile("red.relaxed.gpu.global.add.u64 [%0], %1;" :: "l"((reinterpret_cast<unsigned long long*>(bound) + (base5_3 + (long long)part_stride))), "l"(static_cast<unsigned long long>(fc_fixed_6)) : "memory");
                    }
                } else {
                    unsigned long long fc_fixed_7 = 1ull;
                    if (finite5_3 != 0) {
                        float ct_scaled_11;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(ct_scaled_11) : "f"(term5_3), "f"(scales[0]));
                        float ct_ceil_11 = ct_scaled_11;
                        if (ct_scaled_11 < 8.388608e+06f) {
                            float _sub_28;
                            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_28) : "f"(__fmaf_ru(term5_3, scales[0], 8.388608e+06f)), "f"(8.388608e+06f));
                            ct_ceil_11 = _sub_28;
                        }
                        unsigned long long _max_29 = (((unsigned long long)ct_ceil_11) > (1ull) ? ((unsigned long long)ct_ceil_11) : (1ull));
                        fc_fixed_7 = _max_29;
                    }
                    asm volatile("red.relaxed.gpu.global.add.u64 [%0], %1;" :: "l"((reinterpret_cast<unsigned long long*>(bound) + (base5_3))), "l"(static_cast<unsigned long long>(fc_fixed_7)) : "memory");
                }
                if (packed == 0) {
                    asm volatile("red.relaxed.gpu.global.add.u64 [%0], %1;" :: "l"((reinterpret_cast<unsigned long long*>(bound) + (base5_3 + (long long)part_stride))), "l"(static_cast<unsigned long long>(1ull)) : "memory");
                }
            }
            int row5_4 = rows8[4];
            if (row5_4 >= 0 && row5_4 < num_rows) {
                float kn5_4 = kns8[4];
                long long base5_4 = (long long)row5_4 * (long long)bound_stride;
                float term5_4 = __fmaf_rn(terms[1], kn5_4, terms[0]);
                int finite5_4 = (int)((__as_u32(term5_4) & 2147483647) < 2139095040);
                if (packed != 0) {
                    unsigned int coarse5_4 = 0;
                    if (finite5_4 != 0) {
                        float ct_scaled_12;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(ct_scaled_12) : "f"(term5_4), "f"(scales[1]));
                        float ct_ceil_12 = ct_scaled_12;
                        if (ct_scaled_12 < 8.388608e+06f) {
                            float _sub_30;
                            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_30) : "f"(__fmaf_ru(term5_4, scales[1], 8.388608e+06f)), "f"(8.388608e+06f));
                            ct_ceil_12 = _sub_30;
                        }
                        coarse5_4 = (unsigned int)ct_ceil_12;
                    }
                    if (coarse5_4 >= 64) {
                        asm volatile("red.relaxed.gpu.global.add.u64 [%0], %1;" :: "l"((reinterpret_cast<unsigned long long*>(bound) + (base5_4))), "l"(static_cast<unsigned long long>(((unsigned long long)coarse5_4 << 26ull) + 1ull)) : "memory");
                    } else {
                        asm volatile("red.relaxed.gpu.global.add.u64 [%0], %1;" :: "l"((reinterpret_cast<unsigned long long*>(bound) + (base5_4))), "l"(static_cast<unsigned long long>(1ull)) : "memory");
                        unsigned long long fc_fixed_8 = 1ull;
                        if (finite5_4 != 0) {
                            float ct_scaled_13;
                            asm("mul.rn.f32 %0, %1, %2;" : "=f"(ct_scaled_13) : "f"(term5_4), "f"(scales[0]));
                            float ct_ceil_13 = ct_scaled_13;
                            if (ct_scaled_13 < 8.388608e+06f) {
                                float _sub_31;
                                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_31) : "f"(__fmaf_ru(term5_4, scales[0], 8.388608e+06f)), "f"(8.388608e+06f));
                                ct_ceil_13 = _sub_31;
                            }
                            unsigned long long _max_32 = (((unsigned long long)ct_ceil_13) > (1ull) ? ((unsigned long long)ct_ceil_13) : (1ull));
                            fc_fixed_8 = _max_32;
                        }
                        asm volatile("red.relaxed.gpu.global.add.u64 [%0], %1;" :: "l"((reinterpret_cast<unsigned long long*>(bound) + (base5_4 + (long long)part_stride))), "l"(static_cast<unsigned long long>(fc_fixed_8)) : "memory");
                    }
                } else {
                    unsigned long long fc_fixed_9 = 1ull;
                    if (finite5_4 != 0) {
                        float ct_scaled_14;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(ct_scaled_14) : "f"(term5_4), "f"(scales[0]));
                        float ct_ceil_14 = ct_scaled_14;
                        if (ct_scaled_14 < 8.388608e+06f) {
                            float _sub_33;
                            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_33) : "f"(__fmaf_ru(term5_4, scales[0], 8.388608e+06f)), "f"(8.388608e+06f));
                            ct_ceil_14 = _sub_33;
                        }
                        unsigned long long _max_34 = (((unsigned long long)ct_ceil_14) > (1ull) ? ((unsigned long long)ct_ceil_14) : (1ull));
                        fc_fixed_9 = _max_34;
                    }
                    asm volatile("red.relaxed.gpu.global.add.u64 [%0], %1;" :: "l"((reinterpret_cast<unsigned long long*>(bound) + (base5_4))), "l"(static_cast<unsigned long long>(fc_fixed_9)) : "memory");
                }
                if (packed == 0) {
                    asm volatile("red.relaxed.gpu.global.add.u64 [%0], %1;" :: "l"((reinterpret_cast<unsigned long long*>(bound) + (base5_4 + (long long)part_stride))), "l"(static_cast<unsigned long long>(1ull)) : "memory");
                }
            }
            int row5_5 = rows8[5];
            if (row5_5 >= 0 && row5_5 < num_rows) {
                float kn5_5 = kns8[5];
                long long base5_5 = (long long)row5_5 * (long long)bound_stride;
                float term5_5 = __fmaf_rn(terms[1], kn5_5, terms[0]);
                int finite5_5 = (int)((__as_u32(term5_5) & 2147483647) < 2139095040);
                if (packed != 0) {
                    unsigned int coarse5_5 = 0;
                    if (finite5_5 != 0) {
                        float ct_scaled_15;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(ct_scaled_15) : "f"(term5_5), "f"(scales[1]));
                        float ct_ceil_15 = ct_scaled_15;
                        if (ct_scaled_15 < 8.388608e+06f) {
                            float _sub_35;
                            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_35) : "f"(__fmaf_ru(term5_5, scales[1], 8.388608e+06f)), "f"(8.388608e+06f));
                            ct_ceil_15 = _sub_35;
                        }
                        coarse5_5 = (unsigned int)ct_ceil_15;
                    }
                    if (coarse5_5 >= 64) {
                        asm volatile("red.relaxed.gpu.global.add.u64 [%0], %1;" :: "l"((reinterpret_cast<unsigned long long*>(bound) + (base5_5))), "l"(static_cast<unsigned long long>(((unsigned long long)coarse5_5 << 26ull) + 1ull)) : "memory");
                    } else {
                        asm volatile("red.relaxed.gpu.global.add.u64 [%0], %1;" :: "l"((reinterpret_cast<unsigned long long*>(bound) + (base5_5))), "l"(static_cast<unsigned long long>(1ull)) : "memory");
                        unsigned long long fc_fixed_10 = 1ull;
                        if (finite5_5 != 0) {
                            float ct_scaled_16;
                            asm("mul.rn.f32 %0, %1, %2;" : "=f"(ct_scaled_16) : "f"(term5_5), "f"(scales[0]));
                            float ct_ceil_16 = ct_scaled_16;
                            if (ct_scaled_16 < 8.388608e+06f) {
                                float _sub_36;
                                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_36) : "f"(__fmaf_ru(term5_5, scales[0], 8.388608e+06f)), "f"(8.388608e+06f));
                                ct_ceil_16 = _sub_36;
                            }
                            unsigned long long _max_37 = (((unsigned long long)ct_ceil_16) > (1ull) ? ((unsigned long long)ct_ceil_16) : (1ull));
                            fc_fixed_10 = _max_37;
                        }
                        asm volatile("red.relaxed.gpu.global.add.u64 [%0], %1;" :: "l"((reinterpret_cast<unsigned long long*>(bound) + (base5_5 + (long long)part_stride))), "l"(static_cast<unsigned long long>(fc_fixed_10)) : "memory");
                    }
                } else {
                    unsigned long long fc_fixed_11 = 1ull;
                    if (finite5_5 != 0) {
                        float ct_scaled_17;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(ct_scaled_17) : "f"(term5_5), "f"(scales[0]));
                        float ct_ceil_17 = ct_scaled_17;
                        if (ct_scaled_17 < 8.388608e+06f) {
                            float _sub_38;
                            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_38) : "f"(__fmaf_ru(term5_5, scales[0], 8.388608e+06f)), "f"(8.388608e+06f));
                            ct_ceil_17 = _sub_38;
                        }
                        unsigned long long _max_39 = (((unsigned long long)ct_ceil_17) > (1ull) ? ((unsigned long long)ct_ceil_17) : (1ull));
                        fc_fixed_11 = _max_39;
                    }
                    asm volatile("red.relaxed.gpu.global.add.u64 [%0], %1;" :: "l"((reinterpret_cast<unsigned long long*>(bound) + (base5_5))), "l"(static_cast<unsigned long long>(fc_fixed_11)) : "memory");
                }
                if (packed == 0) {
                    asm volatile("red.relaxed.gpu.global.add.u64 [%0], %1;" :: "l"((reinterpret_cast<unsigned long long*>(bound) + (base5_5 + (long long)part_stride))), "l"(static_cast<unsigned long long>(1ull)) : "memory");
                }
            }
            int row5_6 = rows8[6];
            if (row5_6 >= 0 && row5_6 < num_rows) {
                float kn5_6 = kns8[6];
                long long base5_6 = (long long)row5_6 * (long long)bound_stride;
                float term5_6 = __fmaf_rn(terms[1], kn5_6, terms[0]);
                int finite5_6 = (int)((__as_u32(term5_6) & 2147483647) < 2139095040);
                if (packed != 0) {
                    unsigned int coarse5_6 = 0;
                    if (finite5_6 != 0) {
                        float ct_scaled_18;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(ct_scaled_18) : "f"(term5_6), "f"(scales[1]));
                        float ct_ceil_18 = ct_scaled_18;
                        if (ct_scaled_18 < 8.388608e+06f) {
                            float _sub_40;
                            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_40) : "f"(__fmaf_ru(term5_6, scales[1], 8.388608e+06f)), "f"(8.388608e+06f));
                            ct_ceil_18 = _sub_40;
                        }
                        coarse5_6 = (unsigned int)ct_ceil_18;
                    }
                    if (coarse5_6 >= 64) {
                        asm volatile("red.relaxed.gpu.global.add.u64 [%0], %1;" :: "l"((reinterpret_cast<unsigned long long*>(bound) + (base5_6))), "l"(static_cast<unsigned long long>(((unsigned long long)coarse5_6 << 26ull) + 1ull)) : "memory");
                    } else {
                        asm volatile("red.relaxed.gpu.global.add.u64 [%0], %1;" :: "l"((reinterpret_cast<unsigned long long*>(bound) + (base5_6))), "l"(static_cast<unsigned long long>(1ull)) : "memory");
                        unsigned long long fc_fixed_12 = 1ull;
                        if (finite5_6 != 0) {
                            float ct_scaled_19;
                            asm("mul.rn.f32 %0, %1, %2;" : "=f"(ct_scaled_19) : "f"(term5_6), "f"(scales[0]));
                            float ct_ceil_19 = ct_scaled_19;
                            if (ct_scaled_19 < 8.388608e+06f) {
                                float _sub_41;
                                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_41) : "f"(__fmaf_ru(term5_6, scales[0], 8.388608e+06f)), "f"(8.388608e+06f));
                                ct_ceil_19 = _sub_41;
                            }
                            unsigned long long _max_42 = (((unsigned long long)ct_ceil_19) > (1ull) ? ((unsigned long long)ct_ceil_19) : (1ull));
                            fc_fixed_12 = _max_42;
                        }
                        asm volatile("red.relaxed.gpu.global.add.u64 [%0], %1;" :: "l"((reinterpret_cast<unsigned long long*>(bound) + (base5_6 + (long long)part_stride))), "l"(static_cast<unsigned long long>(fc_fixed_12)) : "memory");
                    }
                } else {
                    unsigned long long fc_fixed_13 = 1ull;
                    if (finite5_6 != 0) {
                        float ct_scaled_20;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(ct_scaled_20) : "f"(term5_6), "f"(scales[0]));
                        float ct_ceil_20 = ct_scaled_20;
                        if (ct_scaled_20 < 8.388608e+06f) {
                            float _sub_43;
                            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_43) : "f"(__fmaf_ru(term5_6, scales[0], 8.388608e+06f)), "f"(8.388608e+06f));
                            ct_ceil_20 = _sub_43;
                        }
                        unsigned long long _max_44 = (((unsigned long long)ct_ceil_20) > (1ull) ? ((unsigned long long)ct_ceil_20) : (1ull));
                        fc_fixed_13 = _max_44;
                    }
                    asm volatile("red.relaxed.gpu.global.add.u64 [%0], %1;" :: "l"((reinterpret_cast<unsigned long long*>(bound) + (base5_6))), "l"(static_cast<unsigned long long>(fc_fixed_13)) : "memory");
                }
                if (packed == 0) {
                    asm volatile("red.relaxed.gpu.global.add.u64 [%0], %1;" :: "l"((reinterpret_cast<unsigned long long*>(bound) + (base5_6 + (long long)part_stride))), "l"(static_cast<unsigned long long>(1ull)) : "memory");
                }
            }
            int row5_7 = rows8[7];
            if (row5_7 >= 0 && row5_7 < num_rows) {
                float kn5_7 = kns8[7];
                long long base5_7 = (long long)row5_7 * (long long)bound_stride;
                float term5_7 = __fmaf_rn(terms[1], kn5_7, terms[0]);
                int finite5_7 = (int)((__as_u32(term5_7) & 2147483647) < 2139095040);
                if (packed != 0) {
                    unsigned int coarse5_7 = 0;
                    if (finite5_7 != 0) {
                        float ct_scaled_21;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(ct_scaled_21) : "f"(term5_7), "f"(scales[1]));
                        float ct_ceil_21 = ct_scaled_21;
                        if (ct_scaled_21 < 8.388608e+06f) {
                            float _sub_45;
                            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_45) : "f"(__fmaf_ru(term5_7, scales[1], 8.388608e+06f)), "f"(8.388608e+06f));
                            ct_ceil_21 = _sub_45;
                        }
                        coarse5_7 = (unsigned int)ct_ceil_21;
                    }
                    if (coarse5_7 >= 64) {
                        asm volatile("red.relaxed.gpu.global.add.u64 [%0], %1;" :: "l"((reinterpret_cast<unsigned long long*>(bound) + (base5_7))), "l"(static_cast<unsigned long long>(((unsigned long long)coarse5_7 << 26ull) + 1ull)) : "memory");
                    } else {
                        asm volatile("red.relaxed.gpu.global.add.u64 [%0], %1;" :: "l"((reinterpret_cast<unsigned long long*>(bound) + (base5_7))), "l"(static_cast<unsigned long long>(1ull)) : "memory");
                        unsigned long long fc_fixed_14 = 1ull;
                        if (finite5_7 != 0) {
                            float ct_scaled_22;
                            asm("mul.rn.f32 %0, %1, %2;" : "=f"(ct_scaled_22) : "f"(term5_7), "f"(scales[0]));
                            float ct_ceil_22 = ct_scaled_22;
                            if (ct_scaled_22 < 8.388608e+06f) {
                                float _sub_46;
                                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_46) : "f"(__fmaf_ru(term5_7, scales[0], 8.388608e+06f)), "f"(8.388608e+06f));
                                ct_ceil_22 = _sub_46;
                            }
                            unsigned long long _max_47 = (((unsigned long long)ct_ceil_22) > (1ull) ? ((unsigned long long)ct_ceil_22) : (1ull));
                            fc_fixed_14 = _max_47;
                        }
                        asm volatile("red.relaxed.gpu.global.add.u64 [%0], %1;" :: "l"((reinterpret_cast<unsigned long long*>(bound) + (base5_7 + (long long)part_stride))), "l"(static_cast<unsigned long long>(fc_fixed_14)) : "memory");
                    }
                } else {
                    unsigned long long fc_fixed_15 = 1ull;
                    if (finite5_7 != 0) {
                        float ct_scaled_23;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(ct_scaled_23) : "f"(term5_7), "f"(scales[0]));
                        float ct_ceil_23 = ct_scaled_23;
                        if (ct_scaled_23 < 8.388608e+06f) {
                            float _sub_48;
                            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_48) : "f"(__fmaf_ru(term5_7, scales[0], 8.388608e+06f)), "f"(8.388608e+06f));
                            ct_ceil_23 = _sub_48;
                        }
                        unsigned long long _max_49 = (((unsigned long long)ct_ceil_23) > (1ull) ? ((unsigned long long)ct_ceil_23) : (1ull));
                        fc_fixed_15 = _max_49;
                    }
                    asm volatile("red.relaxed.gpu.global.add.u64 [%0], %1;" :: "l"((reinterpret_cast<unsigned long long*>(bound) + (base5_7))), "l"(static_cast<unsigned long long>(fc_fixed_15)) : "memory");
                }
                if (packed == 0) {
                    asm volatile("red.relaxed.gpu.global.add.u64 [%0], %1;" :: "l"((reinterpret_cast<unsigned long long*>(bound) + (base5_7 + (long long)part_stride))), "l"(static_cast<unsigned long long>(1ull)) : "memory");
                }
            }
        }
    }
}

} // extern "C"
