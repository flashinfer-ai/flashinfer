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
#define NUM_Q_HEADS 64
#define NUM_KV_HEADS 8
#define WARPS_PER_ROW 8
#define V_SUB_BASE 0
#define CLEAR_UNIT_CHUNKS 256
#define L_SKIP_INVALID 0
#define L_WEIGHT_SELECT 0
#define L_DIRECT_LOOKUP 0
#define L_HITLANE_PAGE 0
#define L_ZERO_HINT 0
#define L_DEP_GATE 0
#define L_ROW_CLEAR 0
#define CLEAR_UNITS_PER_WARP 1
#define P_NO_CLEAR 0
#define P_CLEAR_ONLY 0
#define P_NO_LOOKUP 0
#define P_NO_NORM 0
#define P_NO_ROPE 0
#define P_NO_KV 0
#define P_NO_Q 0

#include <math_constants.h>

extern "C" {

__global__ __launch_bounds__(128) void
kernel_cake_fused_qk_rope_append_43c1aaca2d00f74d875f(__nv_bfloat16* __restrict__ qkv, float* __restrict__ cos_sin, int* __restrict__ q_indptr, int* __restrict__ seq_lens, int* __restrict__ page_table, __nv_bfloat16* __restrict__ k_cache, __nv_bfloat16* __restrict__ v_cache, __nv_bfloat16* __restrict__ out_q, __nv_bfloat16* __restrict__ out_k, __nv_bfloat16* __restrict__ out_v, float* __restrict__ q_norm_weight, float* __restrict__ k_norm_weight, int num_rows, int num_requests, int max_pages, int page_size, long long k_page_stride, long long v_page_stride, int num_row_ctas, int clear_units_per_request, int qk_norm_policy, float eps, int use_out_k, int use_out_v, int clear_last_page)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // === Task calls (dependency order) ===
    int lane_0 = lane;
    int warp_1 = warp;
    int hsel = lane_0 / 8;
    int d0 = lane_0 % 8 * 8;
    if (bid >= num_row_ctas) {
        {
            int cid = (bid - num_row_ctas) * 4 + warp_1;
            if (cid < num_requests * clear_units_per_request && clear_last_page != 0) {
                int req_c = cid / clear_units_per_request;
                int u = cid % clear_units_per_request;
                int cache_sel = u % 2;
                int unit_begin = u / 2 * CLEAR_UNIT_CHUNKS;
                int seq_len_c = seq_lens[req_c];
                if (seq_len_c > 0) {
                    int last_c = seq_len_c - 1;
                    int start_chunk = (last_c % page_size + 1) * (NUM_KV_HEADS * 128 / 8);
                    int end_chunk = page_size * (NUM_KV_HEADS * 128 / 8);
                    if (start_chunk < unit_begin + CLEAR_UNIT_CHUNKS && unit_begin < end_chunk) {
                        int page_c32 = page_table[req_c * max_pages + last_c / page_size];
                        long long page_c = (long long)page_c32;
                        float zeros[8];
                        #pragma unroll
                        for (int j = 0; j < 8; j++) {
                            zeros[j] = 0.0f;
                        }
                        int c = 0;
                        long long off_c = 0;
                        long long base_c = 0;
                        if (cache_sel == 0) {
                            base_c = page_c * k_page_stride;
                            #pragma unroll
                            for (int k = 0; k < CLEAR_UNIT_CHUNKS / 32; k++) {
                                c = unit_begin + lane_0 + 32 * k;
                                if (c >= start_chunk && c < end_chunk) {
                                    off_c = (long long)c * 8;
                                    {
                                        {
                                            __nv_bfloat162 _pk[4];
                                            _pk[0] = __floats2bfloat162_rn(zeros[0 + 0], zeros[0 + 1]);
                                            _pk[1] = __floats2bfloat162_rn(zeros[0 + 2], zeros[0 + 3]);
                                            _pk[2] = __floats2bfloat162_rn(zeros[0 + 4], zeros[0 + 5]);
                                            _pk[3] = __floats2bfloat162_rn(zeros[0 + 6], zeros[0 + 7]);
                                            *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(k_cache + (base_c + off_c)))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                                        }
                                    }
                                }
                            }
                        } else {
                            base_c = page_c * v_page_stride;
                            #pragma unroll
                            for (int k_1 = 0; k_1 < CLEAR_UNIT_CHUNKS / 32; k_1++) {
                                c = unit_begin + lane_0 + 32 * k_1;
                                if (c >= start_chunk && c < end_chunk) {
                                    off_c = (long long)c * 8;
                                    {
                                        {
                                            __nv_bfloat162 _pk[4];
                                            _pk[0] = __floats2bfloat162_rn(zeros[0 + 0], zeros[0 + 1]);
                                            _pk[1] = __floats2bfloat162_rn(zeros[0 + 2], zeros[0 + 3]);
                                            _pk[2] = __floats2bfloat162_rn(zeros[0 + 4], zeros[0 + 5]);
                                            _pk[3] = __floats2bfloat162_rn(zeros[0 + 6], zeros[0 + 7]);
                                            *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(v_cache + (base_c + off_c)))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    } else {
        int row = bid / ((WARPS_PER_ROW + 3) / 4) * ((4 + WARPS_PER_ROW - 1) / WARPS_PER_ROW) + warp_1 / WARPS_PER_ROW;
        int sub = bid % ((WARPS_PER_ROW + 3) / 4) * 4 + warp_1 % WARPS_PER_ROW;
        if (row < num_rows && P_CLEAR_ONLY == 0) {
            long long row_base = (long long)row * (long long)((NUM_Q_HEADS + 2 * NUM_KV_HEADS) * 128);
            float xs[((NUM_Q_HEADS + NUM_KV_HEADS + 4 - 1) / 4 + WARPS_PER_ROW - 1) / WARPS_PER_ROW * 2 * 8];
            int head_of[((NUM_Q_HEADS + NUM_KV_HEADS + 4 - 1) / 4 + WARPS_PER_ROW - 1) / WARPS_PER_ROW];
            int gok[((NUM_Q_HEADS + NUM_KV_HEADS + 4 - 1) / 4 + WARPS_PER_ROW - 1) / WARPS_PER_ROW];
            int src_h = 0;
            #pragma unroll
            for (int g = 0; g < ((NUM_Q_HEADS + NUM_KV_HEADS + 4 - 1) / 4 + WARPS_PER_ROW - 1) / WARPS_PER_ROW; g++) {
                head_of[g] = 4 * (sub + WARPS_PER_ROW * g) + hsel;
                src_h = ((head_of[g] < NUM_Q_HEADS + NUM_KV_HEADS) ? head_of[g] : NUM_Q_HEADS + NUM_KV_HEADS - 1);
                {
                    {
                        const uint4* _vptr_0 = reinterpret_cast<const uint4*>(qkv + (row_base + (long long)(src_h * 128 + d0)) + 0);
                        uint4 _vld_0[1];
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            asm volatile("ld.global.nc.v4.b32 {%0, %1, %2, %3}, [%4];"
                                : "=r"(_vld_0[_blk].x), "=r"(_vld_0[_blk].y), "=r"(_vld_0[_blk].z), "=r"(_vld_0[_blk].w) : "l"((const void*)(_vptr_0 + _blk)) : "memory");
                            uint32_t* _vpairs_0 = reinterpret_cast<uint32_t*>(&_vld_0[_blk]);
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
                                asm volatile(
                                    "{\n\t"
                                    "shl.b32 %0, %2, 16;\n\t"
                                    "and.b32 %1, %2, 0xffff0000;\n\t"
                                    "}\n"
                                    : "=f"((&xs[g * 2 * 8 + _blk * 8 + _pair * 2])[0]), "=f"((&xs[g * 2 * 8 + _blk * 8 + _pair * 2])[1])
                                    : "r"(_vpairs_0[_pair]));
                            }
                        }
                    }
                    {
                        const uint4* _vptr_1 = reinterpret_cast<const uint4*>(qkv + (row_base + (long long)(src_h * 128 + d0 + 64)) + 0);
                        uint4 _vld_1[1];
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            asm volatile("ld.global.nc.v4.b32 {%0, %1, %2, %3}, [%4];"
                                : "=r"(_vld_1[_blk].x), "=r"(_vld_1[_blk].y), "=r"(_vld_1[_blk].z), "=r"(_vld_1[_blk].w) : "l"((const void*)(_vptr_1 + _blk)) : "memory");
                            uint32_t* _vpairs_1 = reinterpret_cast<uint32_t*>(&_vld_1[_blk]);
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
                                asm volatile(
                                    "{\n\t"
                                    "shl.b32 %0, %2, 16;\n\t"
                                    "and.b32 %1, %2, 0xffff0000;\n\t"
                                    "}\n"
                                    : "=f"((&xs[g * 2 * 8 + 8 + _blk * 8 + _pair * 2])[0]), "=f"((&xs[g * 2 * 8 + 8 + _blk * 8 + _pair * 2])[1])
                                    : "r"(_vpairs_1[_pair]));
                            }
                        }
                    }
                }
            }
            int vbits[((NUM_KV_HEADS + 4 - 1) / 4 + WARPS_PER_ROW - 1) / WARPS_PER_ROW * 8];
            int vhead_of[((NUM_KV_HEADS + 4 - 1) / 4 + WARPS_PER_ROW - 1) / WARPS_PER_ROW];
            int vok[((NUM_KV_HEADS + 4 - 1) / 4 + WARPS_PER_ROW - 1) / WARPS_PER_ROW];
            int vfirst = 0;
            int src_v = 0;
            #pragma unroll
            for (int g_1 = 0; g_1 < ((NUM_KV_HEADS + 4 - 1) / 4 + WARPS_PER_ROW - 1) / WARPS_PER_ROW; g_1++) {
                {
                    vhead_of[g_1] = 4 * (sub + WARPS_PER_ROW * g_1) + hsel;
                    vok[g_1] = ((4 * (sub + WARPS_PER_ROW * g_1) < NUM_KV_HEADS) ? 1 : 0);
                    src_v = ((vhead_of[g_1] < NUM_KV_HEADS) ? vhead_of[g_1] : NUM_KV_HEADS - 1);
                }
                {
                    {
                        const int4* _ivptr_2 = reinterpret_cast<const int4*>(qkv + (row_base + (long long)((NUM_Q_HEADS + NUM_KV_HEADS + src_v) * 128 + d0)) + 0);
                        int4 _ivld_2;
                        asm volatile("ld.global.nc.v4.b32 {%0, %1, %2, %3}, [%4];"
                            : "=r"(_ivld_2.x), "=r"(_ivld_2.y), "=r"(_ivld_2.z), "=r"(_ivld_2.w) : "l"((const void*)(_ivptr_2)) : "memory");
                        vbits[g_1 * 8 + 0] = _ivld_2.x;
                        vbits[g_1 * 8 + 1] = _ivld_2.y;
                        vbits[g_1 * 8 + 2] = _ivld_2.z;
                        vbits[g_1 * 8 + 3] = _ivld_2.w;
                    }
                    {
                        const int4* _ivptr_3 = reinterpret_cast<const int4*>(qkv + (row_base + (long long)((NUM_Q_HEADS + NUM_KV_HEADS + src_v) * 128 + d0 + 64)) + 0);
                        int4 _ivld_3;
                        asm volatile("ld.global.nc.v4.b32 {%0, %1, %2, %3}, [%4];"
                            : "=r"(_ivld_3.x), "=r"(_ivld_3.y), "=r"(_ivld_3.z), "=r"(_ivld_3.w) : "l"((const void*)(_ivptr_3)) : "memory");
                        vbits[g_1 * 8 + 4 + 0] = _ivld_3.x;
                        vbits[g_1 * 8 + 4 + 1] = _ivld_3.y;
                        vbits[g_1 * 8 + 4 + 2] = _ivld_3.z;
                        vbits[g_1 * 8 + 4 + 3] = _ivld_3.w;
                    }
                }
            }
            float qw[16];
            float kw[16];
            int has_q = 1;
            int has_k = 1;
            if (qk_norm_policy != 0) {
                {
                    {
                        unsigned _v4_4_0;
                        unsigned _v4_4_1;
                        unsigned _v4_4_2;
                        unsigned _v4_4_3;
                        asm volatile("ld.global.nc.v4.b32 {%0, %1, %2, %3}, [%4];"
                            : "=r"(_v4_4_0), "=r"(_v4_4_1), "=r"(_v4_4_2), "=r"(_v4_4_3) : "l"((const void*)(q_norm_weight + d0 + (0))) : "memory");
                        qw[0 + 0] = __uint_as_float(_v4_4_0);
                        qw[0 + 1] = __uint_as_float(_v4_4_1);
                        qw[0 + 2] = __uint_as_float(_v4_4_2);
                        qw[0 + 3] = __uint_as_float(_v4_4_3);
                    }
                    {
                        unsigned _v4_5_0;
                        unsigned _v4_5_1;
                        unsigned _v4_5_2;
                        unsigned _v4_5_3;
                        asm volatile("ld.global.nc.v4.b32 {%0, %1, %2, %3}, [%4];"
                            : "=r"(_v4_5_0), "=r"(_v4_5_1), "=r"(_v4_5_2), "=r"(_v4_5_3) : "l"((const void*)(q_norm_weight + (d0 + 4) + (0))) : "memory");
                        qw[4 + 0] = __uint_as_float(_v4_5_0);
                        qw[4 + 1] = __uint_as_float(_v4_5_1);
                        qw[4 + 2] = __uint_as_float(_v4_5_2);
                        qw[4 + 3] = __uint_as_float(_v4_5_3);
                    }
                    {
                        unsigned _v4_6_0;
                        unsigned _v4_6_1;
                        unsigned _v4_6_2;
                        unsigned _v4_6_3;
                        asm volatile("ld.global.nc.v4.b32 {%0, %1, %2, %3}, [%4];"
                            : "=r"(_v4_6_0), "=r"(_v4_6_1), "=r"(_v4_6_2), "=r"(_v4_6_3) : "l"((const void*)(q_norm_weight + (d0 + 64) + (0))) : "memory");
                        qw[8 + 0] = __uint_as_float(_v4_6_0);
                        qw[8 + 1] = __uint_as_float(_v4_6_1);
                        qw[8 + 2] = __uint_as_float(_v4_6_2);
                        qw[8 + 3] = __uint_as_float(_v4_6_3);
                    }
                    {
                        unsigned _v4_7_0;
                        unsigned _v4_7_1;
                        unsigned _v4_7_2;
                        unsigned _v4_7_3;
                        asm volatile("ld.global.nc.v4.b32 {%0, %1, %2, %3}, [%4];"
                            : "=r"(_v4_7_0), "=r"(_v4_7_1), "=r"(_v4_7_2), "=r"(_v4_7_3) : "l"((const void*)(q_norm_weight + (d0 + 64 + 4) + (0))) : "memory");
                        qw[12 + 0] = __uint_as_float(_v4_7_0);
                        qw[12 + 1] = __uint_as_float(_v4_7_1);
                        qw[12 + 2] = __uint_as_float(_v4_7_2);
                        qw[12 + 3] = __uint_as_float(_v4_7_3);
                    }
                    {
                        unsigned _v4_8_0;
                        unsigned _v4_8_1;
                        unsigned _v4_8_2;
                        unsigned _v4_8_3;
                        asm volatile("ld.global.nc.v4.b32 {%0, %1, %2, %3}, [%4];"
                            : "=r"(_v4_8_0), "=r"(_v4_8_1), "=r"(_v4_8_2), "=r"(_v4_8_3) : "l"((const void*)(k_norm_weight + d0 + (0))) : "memory");
                        kw[0 + 0] = __uint_as_float(_v4_8_0);
                        kw[0 + 1] = __uint_as_float(_v4_8_1);
                        kw[0 + 2] = __uint_as_float(_v4_8_2);
                        kw[0 + 3] = __uint_as_float(_v4_8_3);
                    }
                    {
                        unsigned _v4_9_0;
                        unsigned _v4_9_1;
                        unsigned _v4_9_2;
                        unsigned _v4_9_3;
                        asm volatile("ld.global.nc.v4.b32 {%0, %1, %2, %3}, [%4];"
                            : "=r"(_v4_9_0), "=r"(_v4_9_1), "=r"(_v4_9_2), "=r"(_v4_9_3) : "l"((const void*)(k_norm_weight + (d0 + 4) + (0))) : "memory");
                        kw[4 + 0] = __uint_as_float(_v4_9_0);
                        kw[4 + 1] = __uint_as_float(_v4_9_1);
                        kw[4 + 2] = __uint_as_float(_v4_9_2);
                        kw[4 + 3] = __uint_as_float(_v4_9_3);
                    }
                    {
                        unsigned _v4_10_0;
                        unsigned _v4_10_1;
                        unsigned _v4_10_2;
                        unsigned _v4_10_3;
                        asm volatile("ld.global.nc.v4.b32 {%0, %1, %2, %3}, [%4];"
                            : "=r"(_v4_10_0), "=r"(_v4_10_1), "=r"(_v4_10_2), "=r"(_v4_10_3) : "l"((const void*)(k_norm_weight + (d0 + 64) + (0))) : "memory");
                        kw[8 + 0] = __uint_as_float(_v4_10_0);
                        kw[8 + 1] = __uint_as_float(_v4_10_1);
                        kw[8 + 2] = __uint_as_float(_v4_10_2);
                        kw[8 + 3] = __uint_as_float(_v4_10_3);
                    }
                    {
                        unsigned _v4_11_0;
                        unsigned _v4_11_1;
                        unsigned _v4_11_2;
                        unsigned _v4_11_3;
                        asm volatile("ld.global.nc.v4.b32 {%0, %1, %2, %3}, [%4];"
                            : "=r"(_v4_11_0), "=r"(_v4_11_1), "=r"(_v4_11_2), "=r"(_v4_11_3) : "l"((const void*)(k_norm_weight + (d0 + 64 + 4) + (0))) : "memory");
                        kw[12 + 0] = __uint_as_float(_v4_11_0);
                        kw[12 + 1] = __uint_as_float(_v4_11_1);
                        kw[12 + 2] = __uint_as_float(_v4_11_2);
                        kw[12 + 3] = __uint_as_float(_v4_11_3);
                    }
                }
            }
            int req = -1;
            int pos = -1;
            int le_req = 0;
            int lb[8];
            int le[8];
            int ll[8];
            int bk = 0;
            int g_req = 0;
            int g_b = 0;
            int g_e = 0;
            int g_l = 0;
            {
                #pragma unroll
                for (int k_2 = 0; k_2 < 8; k_2++) {
                    lb[k_2] = 0;
                    le[k_2] = 0;
                    ll[k_2] = 0;
                    bk = lane_0 + 32 * k_2;
                    if (bk < num_requests) {
                        lb[k_2] = q_indptr[bk];
                        le[k_2] = q_indptr[bk + 1];
                        ll[k_2] = seq_lens[bk];
                    }
                }
            }
            unsigned int dep = 0;
            unsigned int dep_self = 0;
            unsigned int dep_zero = 0;
            int row_g = row;
            int page32 = 0;
            int have_page = 0;
            int hit = 0;
            unsigned int hit_mask = 0;
            int src_lane = 0;
            int pl = -1;
            int pg = 0;
            #pragma unroll
            for (int k_3 = 0; k_3 < 8; k_3++) {
                if (req < 0 && 32 * k_3 < num_requests) {
                    bk = lane_0 + 32 * k_3;
                    hit = 0;
                    {
                        if (bk < num_requests) {
                            if (lb[k_3] <= row_g && row_g < le[k_3]) {
                                hit = 1;
                            }
                        }
                        unsigned int _vote_1 = __ballot_sync(0xFFFFFFFF, hit != 0);
                        hit_mask = _vote_1;
                        if (hit_mask != 0) {
                            int _ffs_1 = __ffs(hit_mask);
                            src_lane = _ffs_1 - 1;
                            int _shfl_5 = __shfl_sync(0xFFFFFFFF, bk, src_lane);
                            req = _shfl_5;
                            int _shfl_6 = __shfl_sync(0xFFFFFFFF, row + ll[k_3] - le[k_3], src_lane);
                            pos = _shfl_6;
                        }
                    }
                }
            }
            if (req < 0) {
                #pragma unroll 1
                for (int base = 256; base < num_requests; base += 32) {
                    int b = base + lane_0;
                    hit = 0;
                    int b_begin = 0;
                    int b_end = 0;
                    int b_len = 0;
                    if (b < num_requests) {
                        b_begin = q_indptr[b];
                        b_end = q_indptr[b + 1];
                        b_len = seq_lens[b];
                        if (b_begin <= row_g && row_g < b_end) {
                            hit = 1;
                        }
                    }
                    unsigned int _vote_2 = __ballot_sync(0xFFFFFFFF, hit != 0);
                    hit_mask = _vote_2;
                    if (hit_mask != 0) {
                        int _ffs_2 = __ffs(hit_mask);
                        src_lane = _ffs_2 - 1;
                        int _shfl_8 = __shfl_sync(0xFFFFFFFF, b, src_lane);
                        req = _shfl_8;
                        int _shfl_9 = __shfl_sync(0xFFFFFFFF, row + b_len - b_end, src_lane);
                        pos = _shfl_9;
                        break;
                    }
                }
            }
            if (req >= 0 && pos >= 0) {
                {
                    page32 = page_table[req * max_pages + pos / page_size];
                }
                long long page = (long long)page32;
                int slot = pos % page_size;
                long long k_row_base = page * k_page_stride + (long long)(slot * (NUM_KV_HEADS * 128));
                long long v_row_base = page * v_page_stride + (long long)(slot * (NUM_KV_HEADS * 128));
                long long cs_base = (long long)pos * 128;
                float cosv[8];
                float sinv[8];
                {
                    unsigned _v4_12_0;
                    unsigned _v4_12_1;
                    unsigned _v4_12_2;
                    unsigned _v4_12_3;
                    asm volatile("ld.global.nc.v4.b32 {%0, %1, %2, %3}, [%4];"
                        : "=r"(_v4_12_0), "=r"(_v4_12_1), "=r"(_v4_12_2), "=r"(_v4_12_3) : "l"((const void*)(cos_sin + (cs_base + (long long)d0) + (0))) : "memory");
                    cosv[0 + 0] = __uint_as_float(_v4_12_0);
                    cosv[0 + 1] = __uint_as_float(_v4_12_1);
                    cosv[0 + 2] = __uint_as_float(_v4_12_2);
                    cosv[0 + 3] = __uint_as_float(_v4_12_3);
                }
                {
                    unsigned _v4_13_0;
                    unsigned _v4_13_1;
                    unsigned _v4_13_2;
                    unsigned _v4_13_3;
                    asm volatile("ld.global.nc.v4.b32 {%0, %1, %2, %3}, [%4];"
                        : "=r"(_v4_13_0), "=r"(_v4_13_1), "=r"(_v4_13_2), "=r"(_v4_13_3) : "l"((const void*)(cos_sin + (cs_base + (long long)d0 + 4) + (0))) : "memory");
                    cosv[4 + 0] = __uint_as_float(_v4_13_0);
                    cosv[4 + 1] = __uint_as_float(_v4_13_1);
                    cosv[4 + 2] = __uint_as_float(_v4_13_2);
                    cosv[4 + 3] = __uint_as_float(_v4_13_3);
                }
                {
                    unsigned _v4_14_0;
                    unsigned _v4_14_1;
                    unsigned _v4_14_2;
                    unsigned _v4_14_3;
                    asm volatile("ld.global.nc.v4.b32 {%0, %1, %2, %3}, [%4];"
                        : "=r"(_v4_14_0), "=r"(_v4_14_1), "=r"(_v4_14_2), "=r"(_v4_14_3) : "l"((const void*)(cos_sin + (cs_base + (long long)(d0 + 64)) + (0))) : "memory");
                    sinv[0 + 0] = __uint_as_float(_v4_14_0);
                    sinv[0 + 1] = __uint_as_float(_v4_14_1);
                    sinv[0 + 2] = __uint_as_float(_v4_14_2);
                    sinv[0 + 3] = __uint_as_float(_v4_14_3);
                }
                {
                    unsigned _v4_15_0;
                    unsigned _v4_15_1;
                    unsigned _v4_15_2;
                    unsigned _v4_15_3;
                    asm volatile("ld.global.nc.v4.b32 {%0, %1, %2, %3}, [%4];"
                        : "=r"(_v4_15_0), "=r"(_v4_15_1), "=r"(_v4_15_2), "=r"(_v4_15_3) : "l"((const void*)(cos_sin + (cs_base + (long long)(d0 + 64) + 4) + (0))) : "memory");
                    sinv[4 + 0] = __uint_as_float(_v4_15_0);
                    sinv[4 + 1] = __uint_as_float(_v4_15_1);
                    sinv[4 + 2] = __uint_as_float(_v4_15_2);
                    sinv[4 + 3] = __uint_as_float(_v4_15_3);
                }
                float ss = 0.0f;
                float inv = 0.0f;
                float wj = 0.0f;
                float x1 = 0.0f;
                float x2 = 0.0f;
                int head = 0;
                int is_q = 0;
                int kvh = 0;
                long long q_dst = 0;
                long long k_dst = 0;
                int do_g = 1;
                #pragma unroll
                for (int g_2 = 0; g_2 < ((NUM_Q_HEADS + NUM_KV_HEADS + 4 - 1) / 4 + WARPS_PER_ROW - 1) / WARPS_PER_ROW; g_2++) {
                    do_g = ((L_SKIP_INVALID == 1) ? gok[g_2] : 1);
                    if (do_g != 0) {
                        head = head_of[g_2];
                        is_q = ((head < NUM_Q_HEADS) ? 1 : 0);
                        if (P_NO_NORM == 0 && qk_norm_policy == 2) {
                            ss = 0.0f;
                            #pragma unroll
                            for (int j_1 = 0; j_1 < 16; j_1++) {
                                ss += xs[g_2 * 2 * 8 + j_1] * xs[g_2 * 2 * 8 + j_1];
                            }
                            float _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, ss, 4);
                            ss += _shfl_xor_0;
                            float _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, ss, 2);
                            ss += _shfl_xor_1;
                            float _shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, ss, 1);
                            ss += _shfl_xor_2;
                            float _rsqrt_0 = rsqrtf(ss * 0.0078125f + eps);
                            inv = _rsqrt_0;
                            #pragma unroll
                            for (int j_2 = 0; j_2 < 16; j_2++) {
                                wj = ((is_q != 0) ? qw[j_2] : kw[j_2]);
                                xs[g_2 * 2 * 8 + j_2] = xs[g_2 * 2 * 8 + j_2] * (inv * wj);
                            }
                        }
                        {
                            #pragma unroll
                            for (int j_3 = 0; j_3 < 8; j_3++) {
                                x1 = xs[g_2 * 2 * 8 + j_3];
                                x2 = xs[g_2 * 2 * 8 + 8 + j_3];
                                float _fma_0 = __fmaf_rn(x1, cosv[j_3], -(x2 * sinv[j_3]));
                                xs[g_2 * 2 * 8 + j_3] = _fma_0;
                                float _fma_1 = __fmaf_rn(x2, cosv[j_3], x1 * sinv[j_3]);
                                xs[g_2 * 2 * 8 + 8 + j_3] = _fma_1;
                            }
                        }
                        if (P_NO_NORM == 0 && qk_norm_policy == 1) {
                            ss = 0.0f;
                            #pragma unroll
                            for (int j_4 = 0; j_4 < 16; j_4++) {
                                ss += xs[g_2 * 2 * 8 + j_4] * xs[g_2 * 2 * 8 + j_4];
                            }
                            float _shfl_xor_3 = __shfl_xor_sync(0xFFFFFFFF, ss, 4);
                            ss += _shfl_xor_3;
                            float _shfl_xor_4 = __shfl_xor_sync(0xFFFFFFFF, ss, 2);
                            ss += _shfl_xor_4;
                            float _shfl_xor_5 = __shfl_xor_sync(0xFFFFFFFF, ss, 1);
                            ss += _shfl_xor_5;
                            float _rsqrt_1 = rsqrtf(ss * 0.0078125f + eps);
                            inv = _rsqrt_1;
                            #pragma unroll
                            for (int j_5 = 0; j_5 < 16; j_5++) {
                                wj = ((is_q != 0) ? qw[j_5] : kw[j_5]);
                                xs[g_2 * 2 * 8 + j_5] = xs[g_2 * 2 * 8 + j_5] * (inv * wj);
                            }
                        }
                        if (head < NUM_Q_HEADS) {
                            {
                                q_dst = (long long)row * (long long)(NUM_Q_HEADS * 128) + (long long)(head * 128 + d0);
                                {
                                    __nv_bfloat162 _pk[4];
                                    _pk[0] = __floats2bfloat162_rn(xs[g_2 * 2 * 8 + 0], xs[g_2 * 2 * 8 + 1]);
                                    _pk[1] = __floats2bfloat162_rn(xs[g_2 * 2 * 8 + 2], xs[g_2 * 2 * 8 + 3]);
                                    _pk[2] = __floats2bfloat162_rn(xs[g_2 * 2 * 8 + 4], xs[g_2 * 2 * 8 + 5]);
                                    _pk[3] = __floats2bfloat162_rn(xs[g_2 * 2 * 8 + 6], xs[g_2 * 2 * 8 + 7]);
                                    *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(out_q + q_dst))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                                }
                                {
                                    __nv_bfloat162 _pk[4];
                                    _pk[0] = __floats2bfloat162_rn(xs[g_2 * 2 * 8 + 8 + 0], xs[g_2 * 2 * 8 + 8 + 1]);
                                    _pk[1] = __floats2bfloat162_rn(xs[g_2 * 2 * 8 + 8 + 2], xs[g_2 * 2 * 8 + 8 + 3]);
                                    _pk[2] = __floats2bfloat162_rn(xs[g_2 * 2 * 8 + 8 + 4], xs[g_2 * 2 * 8 + 8 + 5]);
                                    _pk[3] = __floats2bfloat162_rn(xs[g_2 * 2 * 8 + 8 + 6], xs[g_2 * 2 * 8 + 8 + 7]);
                                    *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(out_q + (q_dst + 64)))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                                }
                            }
                        } else if (head < NUM_Q_HEADS + NUM_KV_HEADS) {
                            {
                                kvh = head - NUM_Q_HEADS;
                                if (use_out_k != 0) {
                                    k_dst = (long long)row * (long long)(NUM_KV_HEADS * 128) + (long long)(kvh * 128 + d0);
                                    {
                                        __nv_bfloat162 _pk[4];
                                        _pk[0] = __floats2bfloat162_rn(xs[g_2 * 2 * 8 + 0], xs[g_2 * 2 * 8 + 1]);
                                        _pk[1] = __floats2bfloat162_rn(xs[g_2 * 2 * 8 + 2], xs[g_2 * 2 * 8 + 3]);
                                        _pk[2] = __floats2bfloat162_rn(xs[g_2 * 2 * 8 + 4], xs[g_2 * 2 * 8 + 5]);
                                        _pk[3] = __floats2bfloat162_rn(xs[g_2 * 2 * 8 + 6], xs[g_2 * 2 * 8 + 7]);
                                        *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(out_k + k_dst))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                                    }
                                    {
                                        __nv_bfloat162 _pk[4];
                                        _pk[0] = __floats2bfloat162_rn(xs[g_2 * 2 * 8 + 8 + 0], xs[g_2 * 2 * 8 + 8 + 1]);
                                        _pk[1] = __floats2bfloat162_rn(xs[g_2 * 2 * 8 + 8 + 2], xs[g_2 * 2 * 8 + 8 + 3]);
                                        _pk[2] = __floats2bfloat162_rn(xs[g_2 * 2 * 8 + 8 + 4], xs[g_2 * 2 * 8 + 8 + 5]);
                                        _pk[3] = __floats2bfloat162_rn(xs[g_2 * 2 * 8 + 8 + 6], xs[g_2 * 2 * 8 + 8 + 7]);
                                        *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(out_k + (k_dst + 64)))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                                    }
                                } else {
                                    k_dst = k_row_base + (long long)(kvh * 128 + d0);
                                    {
                                        __nv_bfloat162 _pk[4];
                                        _pk[0] = __floats2bfloat162_rn(xs[g_2 * 2 * 8 + 0], xs[g_2 * 2 * 8 + 1]);
                                        _pk[1] = __floats2bfloat162_rn(xs[g_2 * 2 * 8 + 2], xs[g_2 * 2 * 8 + 3]);
                                        _pk[2] = __floats2bfloat162_rn(xs[g_2 * 2 * 8 + 4], xs[g_2 * 2 * 8 + 5]);
                                        _pk[3] = __floats2bfloat162_rn(xs[g_2 * 2 * 8 + 6], xs[g_2 * 2 * 8 + 7]);
                                        *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(k_cache + k_dst))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                                    }
                                    {
                                        __nv_bfloat162 _pk[4];
                                        _pk[0] = __floats2bfloat162_rn(xs[g_2 * 2 * 8 + 8 + 0], xs[g_2 * 2 * 8 + 8 + 1]);
                                        _pk[1] = __floats2bfloat162_rn(xs[g_2 * 2 * 8 + 8 + 2], xs[g_2 * 2 * 8 + 8 + 3]);
                                        _pk[2] = __floats2bfloat162_rn(xs[g_2 * 2 * 8 + 8 + 4], xs[g_2 * 2 * 8 + 8 + 5]);
                                        _pk[3] = __floats2bfloat162_rn(xs[g_2 * 2 * 8 + 8 + 6], xs[g_2 * 2 * 8 + 8 + 7]);
                                        *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(k_cache + (k_dst + 64)))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                                    }
                                }
                            }
                        }
                    }
                }
                int vh = 0;
                long long v_dst = 0;
                int do_v = 1;
                int v_ok = 0;
                #pragma unroll
                for (int g_3 = 0; g_3 < ((NUM_KV_HEADS + 4 - 1) / 4 + WARPS_PER_ROW - 1) / WARPS_PER_ROW; g_3++) {
                    do_v = ((L_SKIP_INVALID == 1) ? vok[g_3] : 1);
                    if (do_v != 0) {
                        vh = vhead_of[g_3];
                        {
                            v_ok = ((vh < NUM_KV_HEADS) ? 1 : 0);
                        }
                        if (P_NO_KV == 0 && v_ok != 0) {
                            if (use_out_v != 0) {
                                v_dst = (long long)row * (long long)(NUM_KV_HEADS * 128) + (long long)(vh * 128 + d0);
                                {
                                    int4 _iv4 = make_int4(vbits[g_3 * 8 + 0], vbits[g_3 * 8 + 1], vbits[g_3 * 8 + 2], vbits[g_3 * 8 + 3]);
                                    *reinterpret_cast<int4*>(out_v + v_dst + 0) = _iv4;
                                }
                                {
                                    int4 _iv4 = make_int4(vbits[g_3 * 8 + 4 + 0], vbits[g_3 * 8 + 4 + 1], vbits[g_3 * 8 + 4 + 2], vbits[g_3 * 8 + 4 + 3]);
                                    *reinterpret_cast<int4*>(out_v + (v_dst + 64) + 0) = _iv4;
                                }
                            } else {
                                v_dst = v_row_base + (long long)(vh * 128 + d0);
                                {
                                    int4 _iv4 = make_int4(vbits[g_3 * 8 + 0], vbits[g_3 * 8 + 1], vbits[g_3 * 8 + 2], vbits[g_3 * 8 + 3]);
                                    *reinterpret_cast<int4*>(v_cache + v_dst + 0) = _iv4;
                                }
                                {
                                    int4 _iv4 = make_int4(vbits[g_3 * 8 + 4 + 0], vbits[g_3 * 8 + 4 + 1], vbits[g_3 * 8 + 4 + 2], vbits[g_3 * 8 + 4 + 3]);
                                    *reinterpret_cast<int4*>(v_cache + (v_dst + 64) + 0) = _iv4;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

} // extern "C"
