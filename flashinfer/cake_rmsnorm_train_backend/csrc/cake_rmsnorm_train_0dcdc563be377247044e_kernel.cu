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
#define SMEM_DOT_SLOTS_OFF 0
#define SMEM_DOT_SLOTS_STAGE_BYTES 96
#define SMEM_DOT_SLOTS_STRIDE 96
#define SMEM_FLAGS_OFF 128
#define SMEM_FLAGS_STAGE_BYTES 384
#define SMEM_FLAGS_STRIDE 384
#define SMEM_RED_FOLD_OFF 512
#define SMEM_RED_FOLD_STAGE_BYTES 6144
#define SMEM_RED_FOLD_STRIDE 6144
#define SMEM_ROW_FOLD_OFF 6656
#define SMEM_ROW_FOLD_STAGE_BYTES 128
#define SMEM_ROW_FOLD_STRIDE 128
#define SMEM_COMP_SMEM_OFF 6784
#define SMEM_COMP_SMEM_STAGE_BYTES 128
#define SMEM_COMP_SMEM_STRIDE 128
#define SMEM_W_SM_OFF 6912
#define SMEM_W_SM_STAGE_BYTES 16
#define SMEM_W_SM_STRIDE 16
#define SMEM_PART_SM_OFF 7040
#define SMEM_PART_SM_STAGE_BYTES 16
#define SMEM_PART_SM_STRIDE 16
#define SMEM_TOTAL 7168
#define THREADS 384
#define LAUNCH_MIN_BLOCKS 1

#include <math_constants.h>

extern "C" {

__global__ __launch_bounds__(384, LAUNCH_MIN_BLOCKS) void
kernel_cake_rmsnorm_train_0dcdc563be377247044e(__nv_bfloat16* __restrict__ g, __nv_bfloat16* __restrict__ x, __nv_bfloat16* __restrict__ w, float* __restrict__ r, __nv_bfloat16* __restrict__ dx, float* __restrict__ dw, float* __restrict__ partial, unsigned int* __restrict__ counters, __nv_bfloat16* __restrict__ g_h, int T, int n_chunks, int rows_per_chunk, long long g_stride_p0, long long x_stride_p0, long long g_h_stride_p0)
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
    float* dot_slots = reinterpret_cast<float*>(smem_raw + 0);
    const int dot_slots_addr = smem + 0;
    int* flags = reinterpret_cast<int*>(smem_raw + 128);
    const int flags_addr = smem + 128;
    float* red_fold = reinterpret_cast<float*>(smem_raw + 512);
    const int red_fold_addr = smem + 512;
    float* row_fold = reinterpret_cast<float*>(smem_raw + 6656);
    const int row_fold_addr = smem + 6656;
    float* comp_smem = reinterpret_cast<float*>(smem_raw + 6784);
    const int comp_smem_addr = smem + 6784;
    unsigned int* w_sm = reinterpret_cast<unsigned int*>(smem_raw + 6912);
    const int w_sm_addr = smem + 6912;
    float* part_sm = reinterpret_cast<float*>(smem_raw + 7040);
    const int part_sm_addr = smem + 7040;

    // === Task calls (declaration order) ===
    asm volatile("griddepcontrol.wait;" ::: "memory");
    unsigned int nxt = 0;
    if (tid == 0) {
        unsigned int _atomic_fetch_add_0 = atomicAdd(&counters[0], 1);
        nxt = _atomic_fetch_add_0;
    }
#if __CUDA_ARCH__ == 1070
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
#endif
    long long g_stride = g_stride_p0;
    long long x_stride = x_stride_p0;
    long long gh_stride = g_h_stride_p0;
    int c = 0;
    int rg = warp / 12;
    int tir = tid - rg * 384;
    int row_begin = 0;
    int row_end = 0;
    int n_rows = 0;
    int n_iters = 0;
    int last_row = 0;
    float wc[16];
    unsigned int ww[8];
    #pragma unroll
    for (int v = 0; v < 2; v++) {
        int wcolp = (v * 384 + tir) * 8;
        {
            const uint4* _vptr_1 = reinterpret_cast<const uint4*>(w + (unsigned long long)wcolp);
            uint4* _vdst_1 = reinterpret_cast<uint4*>(&ww[v * 4]);
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vdst_1[_blk] = _vptr_1[_blk];
            }
        }
    }
    float acc[16];
    float comp[16];
    unsigned int compw[8];
    if (tid == 0) {
        flags[1] = (int)nxt;
    }
    __syncthreads();
    c = flags[1];
    #pragma unroll 1
    for (int claim_it = 0; claim_it < n_chunks; claim_it++) {
        if (c >= n_chunks) {
            break;
        }
        if (tid == 0) {
            unsigned int _atomic_fetch_add_2 = atomicAdd(&counters[0], 1);
            nxt = _atomic_fetch_add_2;
        }
        int stages = T;
        int sl = rows_per_chunk;
        if (sl > stages) {
            sl = stages;
        }
        if (sl < 1) {
            sl = 1;
        }
        int share = (stages + sl - 1) / sl;
        int tail = (share + 11) / 12;
        if (tail < 1) {
            tail = 1;
        }
        int rem = stages;
        int size = (share + 1) / 2;
        int first = 0;
        int base = 0;
        int rows = 0;
        int halving = 1;
        int begin_it = 0;
        #pragma unroll 1
        for (int k = 0; k < 5; k++) {
            if (halving != 0) {
                if (size > tail) {
                    if (rem >= sl * size) {
                        if (c < first + sl) {
                            rows = size;
                            begin_it = base + (c - first) * size;
                            halving = 0;
                        } else {
                            first = first + sl;
                            base = base + sl * size;
                            rem = rem - sl * size;
                            size = (size + 1) / 2;
                        }
                    } else {
                        halving = 0;
                    }
                } else {
                    halving = 0;
                }
            }
        }
        if (rows == 0) {
            int quot = rem / sl;
            int extra = rem - quot * sl;
            if (c < first + extra) {
                rows = quot + 1;
                begin_it = base + (c - first) * (quot + 1);
            } else if (c < first + sl) {
                rows = quot;
                begin_it = base + extra * (quot + 1) + (c - first - extra) * quot;
            } else {
                begin_it = stages;
            }
        }
        row_begin = begin_it;
        row_end = row_begin + rows;
        if (row_begin > T) {
            row_begin = T;
        }
        if (row_end > T) {
            row_end = T;
        }
        n_rows = row_end - row_begin;
        n_iters = n_rows;
        last_row = row_end - 1;
        #pragma unroll
        for (int j = 0; j < 16; j++) {
            acc[j] = 0.0f;
            comp[j] = 0.0f;
        }
        unsigned int gwP0[8];
        unsigned int xwP0[8];
        unsigned int ghwP0[8];
        float rrP0[1];
        unsigned int gwP1[8];
        unsigned int xwP1[8];
        unsigned int ghwP1[8];
        float rrP1[1];
        unsigned int gwP2[8];
        unsigned int xwP2[8];
        unsigned int ghwP2[8];
        float rrP2[1];
        unsigned int gwP3[8];
        unsigned int xwP3[8];
        unsigned int ghwP3[8];
        float rrP3[1];
        if (n_iters > 0) {
            int rowp0 = ((row_begin + rg) < (last_row) ? (row_begin + rg) : (last_row));
            unsigned long long g_base = (unsigned long long)rowp0 * (unsigned long long)g_stride;
            unsigned long long x_base = (unsigned long long)rowp0 * (unsigned long long)x_stride;
            rrP0[0] = r[rowp0];
            #pragma unroll
            for (int v_1 = 0; v_1 < 2; v_1++) {
                int coln = (v_1 * 384 + tir) * 8;
                {
                    const uint4* _vptr_3 = reinterpret_cast<const uint4*>(g + g_base + (unsigned long long)coln);
                    uint4* _vdst_3 = reinterpret_cast<uint4*>(&gwP0[v_1 * 4]);
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                            : "=r"(_vdst_3[_blk].x), "=r"(_vdst_3[_blk].y), "=r"(_vdst_3[_blk].z), "=r"(_vdst_3[_blk].w) : "l"((const void*)(_vptr_3 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                    }
                }
                {
                    const uint4* _vptr_4 = reinterpret_cast<const uint4*>(x + x_base + (unsigned long long)coln);
                    uint4* _vdst_4 = reinterpret_cast<uint4*>(&xwP0[v_1 * 4]);
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                            : "=r"(_vdst_4[_blk].x), "=r"(_vdst_4[_blk].y), "=r"(_vdst_4[_blk].z), "=r"(_vdst_4[_blk].w) : "l"((const void*)(_vptr_4 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                    }
                }
            }
            unsigned long long gh_base2 = (unsigned long long)rowp0 * (unsigned long long)gh_stride;
            #pragma unroll
            for (int v2 = 0; v2 < 2; v2++) {
                int coln2 = (v2 * 384 + tir) * 8;
                {
                    const uint4* _vptr_5 = reinterpret_cast<const uint4*>(g_h + gh_base2 + (unsigned long long)coln2);
                    uint4* _vdst_5 = reinterpret_cast<uint4*>(&ghwP0[v2 * 4]);
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                            : "=r"(_vdst_5[_blk].x), "=r"(_vdst_5[_blk].y), "=r"(_vdst_5[_blk].z), "=r"(_vdst_5[_blk].w) : "l"((const void*)(_vptr_5 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                    }
                }
            }
            if (n_iters > 1) {
                int rowp1 = ((row_begin + rg + 1) < (last_row) ? (row_begin + rg + 1) : (last_row));
                unsigned long long g_base_0 = (unsigned long long)rowp1 * (unsigned long long)g_stride;
                unsigned long long x_base_1 = (unsigned long long)rowp1 * (unsigned long long)x_stride;
                rrP1[0] = r[rowp1];
                #pragma unroll
                for (int v_2 = 0; v_2 < 2; v_2++) {
                    int coln_1 = (v_2 * 384 + tir) * 8;
                    {
                        const uint4* _vptr_6 = reinterpret_cast<const uint4*>(g + g_base_0 + (unsigned long long)coln_1);
                        uint4* _vdst_6 = reinterpret_cast<uint4*>(&gwP1[v_2 * 4]);
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                : "=r"(_vdst_6[_blk].x), "=r"(_vdst_6[_blk].y), "=r"(_vdst_6[_blk].z), "=r"(_vdst_6[_blk].w) : "l"((const void*)(_vptr_6 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                        }
                    }
                    {
                        const uint4* _vptr_7 = reinterpret_cast<const uint4*>(x + x_base_1 + (unsigned long long)coln_1);
                        uint4* _vdst_7 = reinterpret_cast<uint4*>(&xwP1[v_2 * 4]);
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                : "=r"(_vdst_7[_blk].x), "=r"(_vdst_7[_blk].y), "=r"(_vdst_7[_blk].z), "=r"(_vdst_7[_blk].w) : "l"((const void*)(_vptr_7 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                        }
                    }
                }
                unsigned long long gh_base2_2 = (unsigned long long)rowp1 * (unsigned long long)gh_stride;
                #pragma unroll
                for (int v2_1 = 0; v2_1 < 2; v2_1++) {
                    int coln2_1 = (v2_1 * 384 + tir) * 8;
                    {
                        const uint4* _vptr_8 = reinterpret_cast<const uint4*>(g_h + gh_base2_2 + (unsigned long long)coln2_1);
                        uint4* _vdst_8 = reinterpret_cast<uint4*>(&ghwP1[v2_1 * 4]);
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                : "=r"(_vdst_8[_blk].x), "=r"(_vdst_8[_blk].y), "=r"(_vdst_8[_blk].z), "=r"(_vdst_8[_blk].w) : "l"((const void*)(_vptr_8 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                        }
                    }
                }
            }
            int nfree = n_iters - 4;
            int nfree_c = ((nfree) > (0) ? (nfree) : (0));
            int full_end = (nfree_c + 2) / 3 * 3;
            #pragma unroll 1
            for (int it = 0; it < full_end; it += 3) {
                int rowf = row_begin + it + rg;
                int rowf_ld = ((rowf + 2) < (last_row) ? (rowf + 2) : (last_row));
                unsigned long long g_base_1 = (unsigned long long)rowf_ld * (unsigned long long)g_stride;
                unsigned long long x_base_0 = (unsigned long long)rowf_ld * (unsigned long long)x_stride;
                rrP2[0] = r[rowf_ld];
                #pragma unroll
                for (int v_3 = 0; v_3 < 2; v_3++) {
                    int coln_2 = (v_3 * 384 + tir) * 8;
                    {
                        const uint4* _vptr_9 = reinterpret_cast<const uint4*>(g + g_base_1 + (unsigned long long)coln_2);
                        uint4* _vdst_9 = reinterpret_cast<uint4*>(&gwP2[v_3 * 4]);
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                : "=r"(_vdst_9[_blk].x), "=r"(_vdst_9[_blk].y), "=r"(_vdst_9[_blk].z), "=r"(_vdst_9[_blk].w) : "l"((const void*)(_vptr_9 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                        }
                    }
                    {
                        const uint4* _vptr_10 = reinterpret_cast<const uint4*>(x + x_base_0 + (unsigned long long)coln_2);
                        uint4* _vdst_10 = reinterpret_cast<uint4*>(&xwP2[v_3 * 4]);
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                : "=r"(_vdst_10[_blk].x), "=r"(_vdst_10[_blk].y), "=r"(_vdst_10[_blk].z), "=r"(_vdst_10[_blk].w) : "l"((const void*)(_vptr_10 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                        }
                    }
                }
                unsigned long long gh_base2_0 = (unsigned long long)rowf_ld * (unsigned long long)gh_stride;
                #pragma unroll
                for (int v2_2 = 0; v2_2 < 2; v2_2++) {
                    int coln2_2 = (v2_2 * 384 + tir) * 8;
                    {
                        const uint4* _vptr_11 = reinterpret_cast<const uint4*>(g_h + gh_base2_0 + (unsigned long long)coln2_2);
                        uint4* _vdst_11 = reinterpret_cast<uint4*>(&ghwP2[v2_2 * 4]);
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                : "=r"(_vdst_11[_blk].x), "=r"(_vdst_11[_blk].y), "=r"(_vdst_11[_blk].z), "=r"(_vdst_11[_blk].w) : "l"((const void*)(_vptr_11 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                        }
                    }
                }
                int parf = it & 1;
                float dq = 0.0f;
                #pragma unroll
                for (int v_4 = 0; v_4 < 2; v_4++) {
                    unsigned int wvd[4];
                    #pragma unroll
                    for (int e = 0; e < 4; e++) {
                        unsigned int gwd = gwP0[v_4 * 4 + e];
                        unsigned int xwd = xwP0[v_4 * 4 + e];
                        float g0 = __uint_as_float(gwd << 16);
                        float g1 = __uint_as_float(gwd & 4294901760u);
                        float x0 = __uint_as_float(xwd << 16);
                        float x1 = __uint_as_float(xwd & 4294901760u);
                        dq = __fmaf_rn(g0 * __uint_as_float(ww[v_4 * 4 + e] << 16), x0, dq);
                        dq = __fmaf_rn(g1 * __uint_as_float(ww[v_4 * 4 + e] & 4294901760u), x1, dq);
                    }
                }
                float dots[1];
                float _warp_reduce_12 = dq;
                #pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1)
                    _warp_reduce_12 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_12, offset);
                dots[0] = _warp_reduce_12;
                float rq = rrP0[0];
                if (lane == 0) {
                    dot_slots[parf * 12 + warp] = dots[0];
                }
                __syncthreads();
                float tot = 0.0f;
                #pragma unroll
                for (int k_1 = 0; k_1 < 12; k_1++) {
                    float d = dot_slots[parf * 12 + rg * 12 + k_1];
                    tot += d;
                }
                dots[0] = tot;
                float cq = __fdiv_rn(dots[0], 6144.0f);
                float kx = rq * rq * rq * cq;
                float nkx = -kx;
                unsigned long long dx_base = (unsigned long long)rowf * 6144;
                #pragma unroll
                for (int v_5 = 0; v_5 < 2; v_5++) {
                    float out[8];
                    unsigned int wv[4];
                    #pragma unroll
                    for (int e_1 = 0; e_1 < 4; e_1++) {
                        unsigned int gwd_1 = gwP0[v_5 * 4 + e_1];
                        unsigned int xwd_1 = xwP0[v_5 * 4 + e_1];
                        float g0_1 = __uint_as_float(gwd_1 << 16);
                        float g1_1 = __uint_as_float(gwd_1 & 4294901760u);
                        float x0_1 = __uint_as_float(xwd_1 << 16);
                        float x1_1 = __uint_as_float(xwd_1 & 4294901760u);
                        float y = __fmaf_rn(g0_1 * x0_1, rq, -comp[v_5 * 8 + 2 * e_1]);
                        float t = acc[v_5 * 8 + 2 * e_1] + y;
                        comp[v_5 * 8 + 2 * e_1] = t - acc[v_5 * 8 + 2 * e_1] - y;
                        acc[v_5 * 8 + 2 * e_1] = t;
                        float y_0 = __fmaf_rn(g1_1 * x1_1, rq, -comp[v_5 * 8 + 2 * e_1 + 1]);
                        float t_0 = acc[v_5 * 8 + 2 * e_1 + 1] + y_0;
                        comp[v_5 * 8 + 2 * e_1 + 1] = t_0 - acc[v_5 * 8 + 2 * e_1 + 1] - y_0;
                        acc[v_5 * 8 + 2 * e_1 + 1] = t_0;
                        unsigned int ghwd = ghwP0[v_5 * 4 + e_1];
                        out[2 * e_1] = __fmaf_rn(nkx, x0_1, rq * (g0_1 * __uint_as_float(ww[v_5 * 4 + e_1] << 16))) + __uint_as_float(ghwd << 16);
                        out[2 * e_1 + 1] = __fmaf_rn(nkx, x1_1, rq * (g1_1 * __uint_as_float(ww[v_5 * 4 + e_1] & 4294901760u))) + __uint_as_float(ghwd & 4294901760u);
                    }
                    int col = (v_5 * 384 + tir) * 8;
                    unsigned int packed[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out[_lp*2 + 0], out[_lp*2+1 + 0]));
                        packed[_lp] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out[_lp*2 + 2], out[_lp*2+1 + 2]));
                        packed[_lp + 1] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out[_lp*2 + 4], out[_lp*2+1 + 4]));
                        packed[_lp + 2] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out[_lp*2 + 6], out[_lp*2+1 + 6]));
                        packed[_lp + 3] = *(uint32_t*)&_bf2;
                    }
                    reinterpret_cast<int4*>(dx + (dx_base + (unsigned long long)col))[0] = reinterpret_cast<int4*>(packed)[0];
                }
                int rowf_0 = row_begin + (it + 1) + rg;
                int rowf_ld_0 = ((rowf_0 + 2) < (last_row) ? (rowf_0 + 2) : (last_row));
                unsigned long long g_base_2 = (unsigned long long)rowf_ld_0 * (unsigned long long)g_stride;
                unsigned long long x_base_2 = (unsigned long long)rowf_ld_0 * (unsigned long long)x_stride;
                rrP0[0] = r[rowf_ld_0];
                #pragma unroll
                for (int v_6 = 0; v_6 < 2; v_6++) {
                    int coln_3 = (v_6 * 384 + tir) * 8;
                    {
                        const uint4* _vptr_13 = reinterpret_cast<const uint4*>(g + g_base_2 + (unsigned long long)coln_3);
                        uint4* _vdst_13 = reinterpret_cast<uint4*>(&gwP0[v_6 * 4]);
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                : "=r"(_vdst_13[_blk].x), "=r"(_vdst_13[_blk].y), "=r"(_vdst_13[_blk].z), "=r"(_vdst_13[_blk].w) : "l"((const void*)(_vptr_13 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                        }
                    }
                    {
                        const uint4* _vptr_14 = reinterpret_cast<const uint4*>(x + x_base_2 + (unsigned long long)coln_3);
                        uint4* _vdst_14 = reinterpret_cast<uint4*>(&xwP0[v_6 * 4]);
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                : "=r"(_vdst_14[_blk].x), "=r"(_vdst_14[_blk].y), "=r"(_vdst_14[_blk].z), "=r"(_vdst_14[_blk].w) : "l"((const void*)(_vptr_14 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                        }
                    }
                }
                unsigned long long gh_base2_1 = (unsigned long long)rowf_ld_0 * (unsigned long long)gh_stride;
                #pragma unroll
                for (int v2_3 = 0; v2_3 < 2; v2_3++) {
                    int coln2_3 = (v2_3 * 384 + tir) * 8;
                    {
                        const uint4* _vptr_15 = reinterpret_cast<const uint4*>(g_h + gh_base2_1 + (unsigned long long)coln2_3);
                        uint4* _vdst_15 = reinterpret_cast<uint4*>(&ghwP0[v2_3 * 4]);
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                : "=r"(_vdst_15[_blk].x), "=r"(_vdst_15[_blk].y), "=r"(_vdst_15[_blk].z), "=r"(_vdst_15[_blk].w) : "l"((const void*)(_vptr_15 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                        }
                    }
                }
                int parf_0 = it + 1 & 1;
                float dq_1 = 0.0f;
                #pragma unroll
                for (int v_7 = 0; v_7 < 2; v_7++) {
                    unsigned int wvd_1[4];
                    #pragma unroll
                    for (int e_2 = 0; e_2 < 4; e_2++) {
                        unsigned int gwd_2 = gwP1[v_7 * 4 + e_2];
                        unsigned int xwd_2 = xwP1[v_7 * 4 + e_2];
                        float g0_2 = __uint_as_float(gwd_2 << 16);
                        float g1_2 = __uint_as_float(gwd_2 & 4294901760u);
                        float x0_2 = __uint_as_float(xwd_2 << 16);
                        float x1_2 = __uint_as_float(xwd_2 & 4294901760u);
                        dq_1 = __fmaf_rn(g0_2 * __uint_as_float(ww[v_7 * 4 + e_2] << 16), x0_2, dq_1);
                        dq_1 = __fmaf_rn(g1_2 * __uint_as_float(ww[v_7 * 4 + e_2] & 4294901760u), x1_2, dq_1);
                    }
                }
                float dots_0[1];
                float _warp_reduce_16 = dq_1;
                #pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1)
                    _warp_reduce_16 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_16, offset);
                dots_0[0] = _warp_reduce_16;
                float rq_0 = rrP1[0];
                if (lane == 0) {
                    dot_slots[parf_0 * 12 + warp] = dots_0[0];
                }
                __syncthreads();
                float tot_0 = 0.0f;
                #pragma unroll
                for (int k_2 = 0; k_2 < 12; k_2++) {
                    float d_1 = dot_slots[parf_0 * 12 + rg * 12 + k_2];
                    tot_0 += d_1;
                }
                dots_0[0] = tot_0;
                float cq_0 = __fdiv_rn(dots_0[0], 6144.0f);
                float kx_0 = rq_0 * rq_0 * rq_0 * cq_0;
                float nkx_0 = -kx_0;
                unsigned long long dx_base_0 = (unsigned long long)rowf_0 * 6144;
                #pragma unroll
                for (int v_8 = 0; v_8 < 2; v_8++) {
                    float out_1[8];
                    unsigned int wv_1[4];
                    #pragma unroll
                    for (int e_3 = 0; e_3 < 4; e_3++) {
                        unsigned int gwd_3 = gwP1[v_8 * 4 + e_3];
                        unsigned int xwd_3 = xwP1[v_8 * 4 + e_3];
                        float g0_3 = __uint_as_float(gwd_3 << 16);
                        float g1_3 = __uint_as_float(gwd_3 & 4294901760u);
                        float x0_3 = __uint_as_float(xwd_3 << 16);
                        float x1_3 = __uint_as_float(xwd_3 & 4294901760u);
                        float y_1 = __fmaf_rn(g0_3 * x0_3, rq_0, -comp[v_8 * 8 + 2 * e_3]);
                        float t_1 = acc[v_8 * 8 + 2 * e_3] + y_1;
                        comp[v_8 * 8 + 2 * e_3] = t_1 - acc[v_8 * 8 + 2 * e_3] - y_1;
                        acc[v_8 * 8 + 2 * e_3] = t_1;
                        float y_1_1 = __fmaf_rn(g1_3 * x1_3, rq_0, -comp[v_8 * 8 + 2 * e_3 + 1]);
                        float t_1_1 = acc[v_8 * 8 + 2 * e_3 + 1] + y_1_1;
                        comp[v_8 * 8 + 2 * e_3 + 1] = t_1_1 - acc[v_8 * 8 + 2 * e_3 + 1] - y_1_1;
                        acc[v_8 * 8 + 2 * e_3 + 1] = t_1_1;
                        unsigned int ghwd_1 = ghwP1[v_8 * 4 + e_3];
                        out_1[2 * e_3] = __fmaf_rn(nkx_0, x0_3, rq_0 * (g0_3 * __uint_as_float(ww[v_8 * 4 + e_3] << 16))) + __uint_as_float(ghwd_1 << 16);
                        out_1[2 * e_3 + 1] = __fmaf_rn(nkx_0, x1_3, rq_0 * (g1_3 * __uint_as_float(ww[v_8 * 4 + e_3] & 4294901760u))) + __uint_as_float(ghwd_1 & 4294901760u);
                    }
                    int col_1 = (v_8 * 384 + tir) * 8;
                    unsigned int packed_1[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_1[_lp*2 + 0], out_1[_lp*2+1 + 0]));
                        packed_1[_lp] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_1[_lp*2 + 2], out_1[_lp*2+1 + 2]));
                        packed_1[_lp + 1] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_1[_lp*2 + 4], out_1[_lp*2+1 + 4]));
                        packed_1[_lp + 2] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_1[_lp*2 + 6], out_1[_lp*2+1 + 6]));
                        packed_1[_lp + 3] = *(uint32_t*)&_bf2;
                    }
                    reinterpret_cast<int4*>(dx + (dx_base_0 + (unsigned long long)col_1))[0] = reinterpret_cast<int4*>(packed_1)[0];
                }
                int rowf_1 = row_begin + (it + 2) + rg;
                int rowf_ld_1 = ((rowf_1 + 2) < (last_row) ? (rowf_1 + 2) : (last_row));
                unsigned long long g_base_3 = (unsigned long long)rowf_ld_1 * (unsigned long long)g_stride;
                unsigned long long x_base_3 = (unsigned long long)rowf_ld_1 * (unsigned long long)x_stride;
                rrP1[0] = r[rowf_ld_1];
                #pragma unroll
                for (int v_9 = 0; v_9 < 2; v_9++) {
                    int coln_4 = (v_9 * 384 + tir) * 8;
                    {
                        const uint4* _vptr_17 = reinterpret_cast<const uint4*>(g + g_base_3 + (unsigned long long)coln_4);
                        uint4* _vdst_17 = reinterpret_cast<uint4*>(&gwP1[v_9 * 4]);
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                : "=r"(_vdst_17[_blk].x), "=r"(_vdst_17[_blk].y), "=r"(_vdst_17[_blk].z), "=r"(_vdst_17[_blk].w) : "l"((const void*)(_vptr_17 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                        }
                    }
                    {
                        const uint4* _vptr_18 = reinterpret_cast<const uint4*>(x + x_base_3 + (unsigned long long)coln_4);
                        uint4* _vdst_18 = reinterpret_cast<uint4*>(&xwP1[v_9 * 4]);
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                : "=r"(_vdst_18[_blk].x), "=r"(_vdst_18[_blk].y), "=r"(_vdst_18[_blk].z), "=r"(_vdst_18[_blk].w) : "l"((const void*)(_vptr_18 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                        }
                    }
                }
                unsigned long long gh_base2_3 = (unsigned long long)rowf_ld_1 * (unsigned long long)gh_stride;
                #pragma unroll
                for (int v2_4 = 0; v2_4 < 2; v2_4++) {
                    int coln2_4 = (v2_4 * 384 + tir) * 8;
                    {
                        const uint4* _vptr_19 = reinterpret_cast<const uint4*>(g_h + gh_base2_3 + (unsigned long long)coln2_4);
                        uint4* _vdst_19 = reinterpret_cast<uint4*>(&ghwP1[v2_4 * 4]);
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                : "=r"(_vdst_19[_blk].x), "=r"(_vdst_19[_blk].y), "=r"(_vdst_19[_blk].z), "=r"(_vdst_19[_blk].w) : "l"((const void*)(_vptr_19 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                        }
                    }
                }
                int parf_1 = it + 2 & 1;
                float dq_2 = 0.0f;
                #pragma unroll
                for (int v_10 = 0; v_10 < 2; v_10++) {
                    unsigned int wvd_2[4];
                    #pragma unroll
                    for (int e_4 = 0; e_4 < 4; e_4++) {
                        unsigned int gwd_4 = gwP2[v_10 * 4 + e_4];
                        unsigned int xwd_4 = xwP2[v_10 * 4 + e_4];
                        float g0_4 = __uint_as_float(gwd_4 << 16);
                        float g1_4 = __uint_as_float(gwd_4 & 4294901760u);
                        float x0_4 = __uint_as_float(xwd_4 << 16);
                        float x1_4 = __uint_as_float(xwd_4 & 4294901760u);
                        dq_2 = __fmaf_rn(g0_4 * __uint_as_float(ww[v_10 * 4 + e_4] << 16), x0_4, dq_2);
                        dq_2 = __fmaf_rn(g1_4 * __uint_as_float(ww[v_10 * 4 + e_4] & 4294901760u), x1_4, dq_2);
                    }
                }
                float dots_2[1];
                float _warp_reduce_20 = dq_2;
                #pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1)
                    _warp_reduce_20 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_20, offset);
                dots_2[0] = _warp_reduce_20;
                float rq_1 = rrP2[0];
                if (lane == 0) {
                    dot_slots[parf_1 * 12 + warp] = dots_2[0];
                }
                __syncthreads();
                float tot_1 = 0.0f;
                #pragma unroll
                for (int k_3 = 0; k_3 < 12; k_3++) {
                    float d_2 = dot_slots[parf_1 * 12 + rg * 12 + k_3];
                    tot_1 += d_2;
                }
                dots_2[0] = tot_1;
                float cq_1 = __fdiv_rn(dots_2[0], 6144.0f);
                float kx_1 = rq_1 * rq_1 * rq_1 * cq_1;
                float nkx_1 = -kx_1;
                unsigned long long dx_base_1 = (unsigned long long)rowf_1 * 6144;
                #pragma unroll
                for (int v_11 = 0; v_11 < 2; v_11++) {
                    float out_2[8];
                    unsigned int wv_2[4];
                    #pragma unroll
                    for (int e_5 = 0; e_5 < 4; e_5++) {
                        unsigned int gwd_5 = gwP2[v_11 * 4 + e_5];
                        unsigned int xwd_5 = xwP2[v_11 * 4 + e_5];
                        float g0_5 = __uint_as_float(gwd_5 << 16);
                        float g1_5 = __uint_as_float(gwd_5 & 4294901760u);
                        float x0_5 = __uint_as_float(xwd_5 << 16);
                        float x1_5 = __uint_as_float(xwd_5 & 4294901760u);
                        float y_2 = __fmaf_rn(g0_5 * x0_5, rq_1, -comp[v_11 * 8 + 2 * e_5]);
                        float t_2 = acc[v_11 * 8 + 2 * e_5] + y_2;
                        comp[v_11 * 8 + 2 * e_5] = t_2 - acc[v_11 * 8 + 2 * e_5] - y_2;
                        acc[v_11 * 8 + 2 * e_5] = t_2;
                        float y_2_1 = __fmaf_rn(g1_5 * x1_5, rq_1, -comp[v_11 * 8 + 2 * e_5 + 1]);
                        float t_2_1 = acc[v_11 * 8 + 2 * e_5 + 1] + y_2_1;
                        comp[v_11 * 8 + 2 * e_5 + 1] = t_2_1 - acc[v_11 * 8 + 2 * e_5 + 1] - y_2_1;
                        acc[v_11 * 8 + 2 * e_5 + 1] = t_2_1;
                        unsigned int ghwd_2 = ghwP2[v_11 * 4 + e_5];
                        out_2[2 * e_5] = __fmaf_rn(nkx_1, x0_5, rq_1 * (g0_5 * __uint_as_float(ww[v_11 * 4 + e_5] << 16))) + __uint_as_float(ghwd_2 << 16);
                        out_2[2 * e_5 + 1] = __fmaf_rn(nkx_1, x1_5, rq_1 * (g1_5 * __uint_as_float(ww[v_11 * 4 + e_5] & 4294901760u))) + __uint_as_float(ghwd_2 & 4294901760u);
                    }
                    int col_2 = (v_11 * 384 + tir) * 8;
                    unsigned int packed_2[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_2[_lp*2 + 0], out_2[_lp*2+1 + 0]));
                        packed_2[_lp] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_2[_lp*2 + 2], out_2[_lp*2+1 + 2]));
                        packed_2[_lp + 1] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_2[_lp*2 + 4], out_2[_lp*2+1 + 4]));
                        packed_2[_lp + 2] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_2[_lp*2 + 6], out_2[_lp*2+1 + 6]));
                        packed_2[_lp + 3] = *(uint32_t*)&_bf2;
                    }
                    reinterpret_cast<int4*>(dx + (dx_base_1 + (unsigned long long)col_2))[0] = reinterpret_cast<int4*>(packed_2)[0];
                }
            }
            #pragma unroll 1
            for (int it_1 = full_end; it_1 < n_iters; it_1 += 3) {
                if (n_iters > it_1) {
                    int rowk = row_begin + it_1 + rg;
                    if (n_iters > it_1 + 2) {
                        int rowk_ld = ((rowk + 2) < (last_row) ? (rowk + 2) : (last_row));
                        unsigned long long g_base_4 = (unsigned long long)rowk_ld * (unsigned long long)g_stride;
                        unsigned long long x_base_4 = (unsigned long long)rowk_ld * (unsigned long long)x_stride;
                        rrP2[0] = r[rowk_ld];
                        #pragma unroll
                        for (int v_12 = 0; v_12 < 2; v_12++) {
                            int coln_5 = (v_12 * 384 + tir) * 8;
                            {
                                const uint4* _vptr_21 = reinterpret_cast<const uint4*>(g + g_base_4 + (unsigned long long)coln_5);
                                uint4* _vdst_21 = reinterpret_cast<uint4*>(&gwP2[v_12 * 4]);
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
                                    asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                        : "=r"(_vdst_21[_blk].x), "=r"(_vdst_21[_blk].y), "=r"(_vdst_21[_blk].z), "=r"(_vdst_21[_blk].w) : "l"((const void*)(_vptr_21 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                                }
                            }
                            {
                                const uint4* _vptr_22 = reinterpret_cast<const uint4*>(x + x_base_4 + (unsigned long long)coln_5);
                                uint4* _vdst_22 = reinterpret_cast<uint4*>(&xwP2[v_12 * 4]);
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
                                    asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                        : "=r"(_vdst_22[_blk].x), "=r"(_vdst_22[_blk].y), "=r"(_vdst_22[_blk].z), "=r"(_vdst_22[_blk].w) : "l"((const void*)(_vptr_22 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                                }
                            }
                        }
                        unsigned long long gh_base2_4 = (unsigned long long)rowk_ld * (unsigned long long)gh_stride;
                        #pragma unroll
                        for (int v2_5 = 0; v2_5 < 2; v2_5++) {
                            int coln2_5 = (v2_5 * 384 + tir) * 8;
                            {
                                const uint4* _vptr_23 = reinterpret_cast<const uint4*>(g_h + gh_base2_4 + (unsigned long long)coln2_5);
                                uint4* _vdst_23 = reinterpret_cast<uint4*>(&ghwP2[v2_5 * 4]);
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
                                    asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                        : "=r"(_vdst_23[_blk].x), "=r"(_vdst_23[_blk].y), "=r"(_vdst_23[_blk].z), "=r"(_vdst_23[_blk].w) : "l"((const void*)(_vptr_23 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                                }
                            }
                        }
                    }
                    int park = it_1 & 1;
                    float dq_3 = 0.0f;
                    #pragma unroll
                    for (int v_13 = 0; v_13 < 2; v_13++) {
                        unsigned int wvd_3[4];
                        #pragma unroll
                        for (int e_6 = 0; e_6 < 4; e_6++) {
                            unsigned int gwd_6 = gwP0[v_13 * 4 + e_6];
                            unsigned int xwd_6 = xwP0[v_13 * 4 + e_6];
                            float g0_6 = __uint_as_float(gwd_6 << 16);
                            float g1_6 = __uint_as_float(gwd_6 & 4294901760u);
                            float x0_6 = __uint_as_float(xwd_6 << 16);
                            float x1_6 = __uint_as_float(xwd_6 & 4294901760u);
                            dq_3 = __fmaf_rn(g0_6 * __uint_as_float(ww[v_13 * 4 + e_6] << 16), x0_6, dq_3);
                            dq_3 = __fmaf_rn(g1_6 * __uint_as_float(ww[v_13 * 4 + e_6] & 4294901760u), x1_6, dq_3);
                        }
                    }
                    float dots_1[1];
                    float _warp_reduce_24 = dq_3;
                    #pragma unroll
                    for (int offset = 16; offset > 0; offset >>= 1)
                        _warp_reduce_24 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_24, offset);
                    dots_1[0] = _warp_reduce_24;
                    float rq_2 = rrP0[0];
                    if (lane == 0) {
                        dot_slots[park * 12 + warp] = dots_1[0];
                    }
                    __syncthreads();
                    float tot_2 = 0.0f;
                    #pragma unroll
                    for (int k_4 = 0; k_4 < 12; k_4++) {
                        float d_3 = dot_slots[park * 12 + rg * 12 + k_4];
                        tot_2 += d_3;
                    }
                    dots_1[0] = tot_2;
                    float cq_2 = __fdiv_rn(dots_1[0], 6144.0f);
                    float kx_2 = rq_2 * rq_2 * rq_2 * cq_2;
                    float nkx_2 = -kx_2;
                    unsigned long long dx_base_2 = (unsigned long long)rowk * 6144;
                    #pragma unroll
                    for (int v_14 = 0; v_14 < 2; v_14++) {
                        float out_3[8];
                        unsigned int wv_3[4];
                        #pragma unroll
                        for (int e_7 = 0; e_7 < 4; e_7++) {
                            unsigned int gwd_7 = gwP0[v_14 * 4 + e_7];
                            unsigned int xwd_7 = xwP0[v_14 * 4 + e_7];
                            float g0_7 = __uint_as_float(gwd_7 << 16);
                            float g1_7 = __uint_as_float(gwd_7 & 4294901760u);
                            float x0_7 = __uint_as_float(xwd_7 << 16);
                            float x1_7 = __uint_as_float(xwd_7 & 4294901760u);
                            float y_3 = __fmaf_rn(g0_7 * x0_7, rq_2, -comp[v_14 * 8 + 2 * e_7]);
                            float t_3 = acc[v_14 * 8 + 2 * e_7] + y_3;
                            comp[v_14 * 8 + 2 * e_7] = t_3 - acc[v_14 * 8 + 2 * e_7] - y_3;
                            acc[v_14 * 8 + 2 * e_7] = t_3;
                            float y_3_1 = __fmaf_rn(g1_7 * x1_7, rq_2, -comp[v_14 * 8 + 2 * e_7 + 1]);
                            float t_3_1 = acc[v_14 * 8 + 2 * e_7 + 1] + y_3_1;
                            comp[v_14 * 8 + 2 * e_7 + 1] = t_3_1 - acc[v_14 * 8 + 2 * e_7 + 1] - y_3_1;
                            acc[v_14 * 8 + 2 * e_7 + 1] = t_3_1;
                            unsigned int ghwd_3 = ghwP0[v_14 * 4 + e_7];
                            out_3[2 * e_7] = __fmaf_rn(nkx_2, x0_7, rq_2 * (g0_7 * __uint_as_float(ww[v_14 * 4 + e_7] << 16))) + __uint_as_float(ghwd_3 << 16);
                            out_3[2 * e_7 + 1] = __fmaf_rn(nkx_2, x1_7, rq_2 * (g1_7 * __uint_as_float(ww[v_14 * 4 + e_7] & 4294901760u))) + __uint_as_float(ghwd_3 & 4294901760u);
                        }
                        int col_3 = (v_14 * 384 + tir) * 8;
                        unsigned int packed_3[4];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_3[_lp*2 + 0], out_3[_lp*2+1 + 0]));
                            packed_3[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_3[_lp*2 + 2], out_3[_lp*2+1 + 2]));
                            packed_3[_lp + 1] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_3[_lp*2 + 4], out_3[_lp*2+1 + 4]));
                            packed_3[_lp + 2] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_3[_lp*2 + 6], out_3[_lp*2+1 + 6]));
                            packed_3[_lp + 3] = *(uint32_t*)&_bf2;
                        }
                        reinterpret_cast<int4*>(dx + (dx_base_2 + (unsigned long long)col_3))[0] = reinterpret_cast<int4*>(packed_3)[0];
                    }
                }
                if (n_iters > it_1 + 1) {
                    int rowk_1 = row_begin + (it_1 + 1) + rg;
                    if (n_iters > it_1 + 1 + 2) {
                        int rowk_ld_1 = ((rowk_1 + 2) < (last_row) ? (rowk_1 + 2) : (last_row));
                        unsigned long long g_base_5 = (unsigned long long)rowk_ld_1 * (unsigned long long)g_stride;
                        unsigned long long x_base_5 = (unsigned long long)rowk_ld_1 * (unsigned long long)x_stride;
                        rrP0[0] = r[rowk_ld_1];
                        #pragma unroll
                        for (int v_15 = 0; v_15 < 2; v_15++) {
                            int coln_6 = (v_15 * 384 + tir) * 8;
                            {
                                const uint4* _vptr_25 = reinterpret_cast<const uint4*>(g + g_base_5 + (unsigned long long)coln_6);
                                uint4* _vdst_25 = reinterpret_cast<uint4*>(&gwP0[v_15 * 4]);
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
                                    asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                        : "=r"(_vdst_25[_blk].x), "=r"(_vdst_25[_blk].y), "=r"(_vdst_25[_blk].z), "=r"(_vdst_25[_blk].w) : "l"((const void*)(_vptr_25 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                                }
                            }
                            {
                                const uint4* _vptr_26 = reinterpret_cast<const uint4*>(x + x_base_5 + (unsigned long long)coln_6);
                                uint4* _vdst_26 = reinterpret_cast<uint4*>(&xwP0[v_15 * 4]);
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
                                    asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                        : "=r"(_vdst_26[_blk].x), "=r"(_vdst_26[_blk].y), "=r"(_vdst_26[_blk].z), "=r"(_vdst_26[_blk].w) : "l"((const void*)(_vptr_26 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                                }
                            }
                        }
                        unsigned long long gh_base2_5 = (unsigned long long)rowk_ld_1 * (unsigned long long)gh_stride;
                        #pragma unroll
                        for (int v2_6 = 0; v2_6 < 2; v2_6++) {
                            int coln2_6 = (v2_6 * 384 + tir) * 8;
                            {
                                const uint4* _vptr_27 = reinterpret_cast<const uint4*>(g_h + gh_base2_5 + (unsigned long long)coln2_6);
                                uint4* _vdst_27 = reinterpret_cast<uint4*>(&ghwP0[v2_6 * 4]);
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
                                    asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                        : "=r"(_vdst_27[_blk].x), "=r"(_vdst_27[_blk].y), "=r"(_vdst_27[_blk].z), "=r"(_vdst_27[_blk].w) : "l"((const void*)(_vptr_27 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                                }
                            }
                        }
                    }
                    int park_1 = it_1 + 1 & 1;
                    float dq_4 = 0.0f;
                    #pragma unroll
                    for (int v_16 = 0; v_16 < 2; v_16++) {
                        unsigned int wvd_4[4];
                        #pragma unroll
                        for (int e_8 = 0; e_8 < 4; e_8++) {
                            unsigned int gwd_8 = gwP1[v_16 * 4 + e_8];
                            unsigned int xwd_8 = xwP1[v_16 * 4 + e_8];
                            float g0_8 = __uint_as_float(gwd_8 << 16);
                            float g1_8 = __uint_as_float(gwd_8 & 4294901760u);
                            float x0_8 = __uint_as_float(xwd_8 << 16);
                            float x1_8 = __uint_as_float(xwd_8 & 4294901760u);
                            dq_4 = __fmaf_rn(g0_8 * __uint_as_float(ww[v_16 * 4 + e_8] << 16), x0_8, dq_4);
                            dq_4 = __fmaf_rn(g1_8 * __uint_as_float(ww[v_16 * 4 + e_8] & 4294901760u), x1_8, dq_4);
                        }
                    }
                    float dots_3[1];
                    float _warp_reduce_28 = dq_4;
                    #pragma unroll
                    for (int offset = 16; offset > 0; offset >>= 1)
                        _warp_reduce_28 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_28, offset);
                    dots_3[0] = _warp_reduce_28;
                    float rq_3 = rrP1[0];
                    if (lane == 0) {
                        dot_slots[park_1 * 12 + warp] = dots_3[0];
                    }
                    __syncthreads();
                    float tot_3 = 0.0f;
                    #pragma unroll
                    for (int k_5 = 0; k_5 < 12; k_5++) {
                        float d_4 = dot_slots[park_1 * 12 + rg * 12 + k_5];
                        tot_3 += d_4;
                    }
                    dots_3[0] = tot_3;
                    float cq_3 = __fdiv_rn(dots_3[0], 6144.0f);
                    float kx_3 = rq_3 * rq_3 * rq_3 * cq_3;
                    float nkx_3 = -kx_3;
                    unsigned long long dx_base_3 = (unsigned long long)rowk_1 * 6144;
                    #pragma unroll
                    for (int v_17 = 0; v_17 < 2; v_17++) {
                        float out_4[8];
                        unsigned int wv_4[4];
                        #pragma unroll
                        for (int e_9 = 0; e_9 < 4; e_9++) {
                            unsigned int gwd_9 = gwP1[v_17 * 4 + e_9];
                            unsigned int xwd_9 = xwP1[v_17 * 4 + e_9];
                            float g0_9 = __uint_as_float(gwd_9 << 16);
                            float g1_9 = __uint_as_float(gwd_9 & 4294901760u);
                            float x0_9 = __uint_as_float(xwd_9 << 16);
                            float x1_9 = __uint_as_float(xwd_9 & 4294901760u);
                            float y_4 = __fmaf_rn(g0_9 * x0_9, rq_3, -comp[v_17 * 8 + 2 * e_9]);
                            float t_4 = acc[v_17 * 8 + 2 * e_9] + y_4;
                            comp[v_17 * 8 + 2 * e_9] = t_4 - acc[v_17 * 8 + 2 * e_9] - y_4;
                            acc[v_17 * 8 + 2 * e_9] = t_4;
                            float y_4_1 = __fmaf_rn(g1_9 * x1_9, rq_3, -comp[v_17 * 8 + 2 * e_9 + 1]);
                            float t_4_1 = acc[v_17 * 8 + 2 * e_9 + 1] + y_4_1;
                            comp[v_17 * 8 + 2 * e_9 + 1] = t_4_1 - acc[v_17 * 8 + 2 * e_9 + 1] - y_4_1;
                            acc[v_17 * 8 + 2 * e_9 + 1] = t_4_1;
                            unsigned int ghwd_4 = ghwP1[v_17 * 4 + e_9];
                            out_4[2 * e_9] = __fmaf_rn(nkx_3, x0_9, rq_3 * (g0_9 * __uint_as_float(ww[v_17 * 4 + e_9] << 16))) + __uint_as_float(ghwd_4 << 16);
                            out_4[2 * e_9 + 1] = __fmaf_rn(nkx_3, x1_9, rq_3 * (g1_9 * __uint_as_float(ww[v_17 * 4 + e_9] & 4294901760u))) + __uint_as_float(ghwd_4 & 4294901760u);
                        }
                        int col_4 = (v_17 * 384 + tir) * 8;
                        unsigned int packed_4[4];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_4[_lp*2 + 0], out_4[_lp*2+1 + 0]));
                            packed_4[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_4[_lp*2 + 2], out_4[_lp*2+1 + 2]));
                            packed_4[_lp + 1] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_4[_lp*2 + 4], out_4[_lp*2+1 + 4]));
                            packed_4[_lp + 2] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_4[_lp*2 + 6], out_4[_lp*2+1 + 6]));
                            packed_4[_lp + 3] = *(uint32_t*)&_bf2;
                        }
                        reinterpret_cast<int4*>(dx + (dx_base_3 + (unsigned long long)col_4))[0] = reinterpret_cast<int4*>(packed_4)[0];
                    }
                }
                if (n_iters > it_1 + 2) {
                    int rowk_2 = row_begin + (it_1 + 2) + rg;
                    if (n_iters > it_1 + 2 + 2) {
                        int rowk_ld_2 = ((rowk_2 + 2) < (last_row) ? (rowk_2 + 2) : (last_row));
                        unsigned long long g_base_6 = (unsigned long long)rowk_ld_2 * (unsigned long long)g_stride;
                        unsigned long long x_base_6 = (unsigned long long)rowk_ld_2 * (unsigned long long)x_stride;
                        rrP1[0] = r[rowk_ld_2];
                        #pragma unroll
                        for (int v_18 = 0; v_18 < 2; v_18++) {
                            int coln_7 = (v_18 * 384 + tir) * 8;
                            {
                                const uint4* _vptr_29 = reinterpret_cast<const uint4*>(g + g_base_6 + (unsigned long long)coln_7);
                                uint4* _vdst_29 = reinterpret_cast<uint4*>(&gwP1[v_18 * 4]);
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
                                    asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                        : "=r"(_vdst_29[_blk].x), "=r"(_vdst_29[_blk].y), "=r"(_vdst_29[_blk].z), "=r"(_vdst_29[_blk].w) : "l"((const void*)(_vptr_29 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                                }
                            }
                            {
                                const uint4* _vptr_30 = reinterpret_cast<const uint4*>(x + x_base_6 + (unsigned long long)coln_7);
                                uint4* _vdst_30 = reinterpret_cast<uint4*>(&xwP1[v_18 * 4]);
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
                                    asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                        : "=r"(_vdst_30[_blk].x), "=r"(_vdst_30[_blk].y), "=r"(_vdst_30[_blk].z), "=r"(_vdst_30[_blk].w) : "l"((const void*)(_vptr_30 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                                }
                            }
                        }
                        unsigned long long gh_base2_6 = (unsigned long long)rowk_ld_2 * (unsigned long long)gh_stride;
                        #pragma unroll
                        for (int v2_7 = 0; v2_7 < 2; v2_7++) {
                            int coln2_7 = (v2_7 * 384 + tir) * 8;
                            {
                                const uint4* _vptr_31 = reinterpret_cast<const uint4*>(g_h + gh_base2_6 + (unsigned long long)coln2_7);
                                uint4* _vdst_31 = reinterpret_cast<uint4*>(&ghwP1[v2_7 * 4]);
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
                                    asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                        : "=r"(_vdst_31[_blk].x), "=r"(_vdst_31[_blk].y), "=r"(_vdst_31[_blk].z), "=r"(_vdst_31[_blk].w) : "l"((const void*)(_vptr_31 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                                }
                            }
                        }
                    }
                    int park_2 = it_1 + 2 & 1;
                    float dq_5 = 0.0f;
                    #pragma unroll
                    for (int v_19 = 0; v_19 < 2; v_19++) {
                        unsigned int wvd_5[4];
                        #pragma unroll
                        for (int e_10 = 0; e_10 < 4; e_10++) {
                            unsigned int gwd_10 = gwP2[v_19 * 4 + e_10];
                            unsigned int xwd_10 = xwP2[v_19 * 4 + e_10];
                            float g0_10 = __uint_as_float(gwd_10 << 16);
                            float g1_10 = __uint_as_float(gwd_10 & 4294901760u);
                            float x0_10 = __uint_as_float(xwd_10 << 16);
                            float x1_10 = __uint_as_float(xwd_10 & 4294901760u);
                            dq_5 = __fmaf_rn(g0_10 * __uint_as_float(ww[v_19 * 4 + e_10] << 16), x0_10, dq_5);
                            dq_5 = __fmaf_rn(g1_10 * __uint_as_float(ww[v_19 * 4 + e_10] & 4294901760u), x1_10, dq_5);
                        }
                    }
                    float dots_4[1];
                    float _warp_reduce_32 = dq_5;
                    #pragma unroll
                    for (int offset = 16; offset > 0; offset >>= 1)
                        _warp_reduce_32 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_32, offset);
                    dots_4[0] = _warp_reduce_32;
                    float rq_4 = rrP2[0];
                    if (lane == 0) {
                        dot_slots[park_2 * 12 + warp] = dots_4[0];
                    }
                    __syncthreads();
                    float tot_4 = 0.0f;
                    #pragma unroll
                    for (int k_6 = 0; k_6 < 12; k_6++) {
                        float d_5 = dot_slots[park_2 * 12 + rg * 12 + k_6];
                        tot_4 += d_5;
                    }
                    dots_4[0] = tot_4;
                    float cq_4 = __fdiv_rn(dots_4[0], 6144.0f);
                    float kx_4 = rq_4 * rq_4 * rq_4 * cq_4;
                    float nkx_4 = -kx_4;
                    unsigned long long dx_base_4 = (unsigned long long)rowk_2 * 6144;
                    #pragma unroll
                    for (int v_20 = 0; v_20 < 2; v_20++) {
                        float out_5[8];
                        unsigned int wv_5[4];
                        #pragma unroll
                        for (int e_11 = 0; e_11 < 4; e_11++) {
                            unsigned int gwd_11 = gwP2[v_20 * 4 + e_11];
                            unsigned int xwd_11 = xwP2[v_20 * 4 + e_11];
                            float g0_11 = __uint_as_float(gwd_11 << 16);
                            float g1_11 = __uint_as_float(gwd_11 & 4294901760u);
                            float x0_11 = __uint_as_float(xwd_11 << 16);
                            float x1_11 = __uint_as_float(xwd_11 & 4294901760u);
                            float y_5 = __fmaf_rn(g0_11 * x0_11, rq_4, -comp[v_20 * 8 + 2 * e_11]);
                            float t_5 = acc[v_20 * 8 + 2 * e_11] + y_5;
                            comp[v_20 * 8 + 2 * e_11] = t_5 - acc[v_20 * 8 + 2 * e_11] - y_5;
                            acc[v_20 * 8 + 2 * e_11] = t_5;
                            float y_5_1 = __fmaf_rn(g1_11 * x1_11, rq_4, -comp[v_20 * 8 + 2 * e_11 + 1]);
                            float t_5_1 = acc[v_20 * 8 + 2 * e_11 + 1] + y_5_1;
                            comp[v_20 * 8 + 2 * e_11 + 1] = t_5_1 - acc[v_20 * 8 + 2 * e_11 + 1] - y_5_1;
                            acc[v_20 * 8 + 2 * e_11 + 1] = t_5_1;
                            unsigned int ghwd_5 = ghwP2[v_20 * 4 + e_11];
                            out_5[2 * e_11] = __fmaf_rn(nkx_4, x0_11, rq_4 * (g0_11 * __uint_as_float(ww[v_20 * 4 + e_11] << 16))) + __uint_as_float(ghwd_5 << 16);
                            out_5[2 * e_11 + 1] = __fmaf_rn(nkx_4, x1_11, rq_4 * (g1_11 * __uint_as_float(ww[v_20 * 4 + e_11] & 4294901760u))) + __uint_as_float(ghwd_5 & 4294901760u);
                        }
                        int col_5 = (v_20 * 384 + tir) * 8;
                        unsigned int packed_5[4];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_5[_lp*2 + 0], out_5[_lp*2+1 + 0]));
                            packed_5[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_5[_lp*2 + 2], out_5[_lp*2+1 + 2]));
                            packed_5[_lp + 1] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_5[_lp*2 + 4], out_5[_lp*2+1 + 4]));
                            packed_5[_lp + 2] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_5[_lp*2 + 6], out_5[_lp*2+1 + 6]));
                            packed_5[_lp + 3] = *(uint32_t*)&_bf2;
                        }
                        reinterpret_cast<int4*>(dx + (dx_base_4 + (unsigned long long)col_5))[0] = reinterpret_cast<int4*>(packed_5)[0];
                    }
                }
            }
        }
        #pragma unroll
        for (int j2 = 0; j2 < 16; j2++) {
            acc[j2] = acc[j2] - comp[j2];
        }
        unsigned long long pbase = (unsigned long long)c * 6144;
        #pragma unroll
        for (int v_21 = 0; v_21 < 2; v_21++) {
            int pcolv = (v_21 * 384 + tir) * 8;
            {
                unsigned _stv8_33_0 = __float_as_uint(acc[v_21 * 8 + 0]);
                unsigned _stv8_33_1 = __float_as_uint(acc[v_21 * 8 + 1]);
                unsigned _stv8_33_2 = __float_as_uint(acc[v_21 * 8 + 2]);
                unsigned _stv8_33_3 = __float_as_uint(acc[v_21 * 8 + 3]);
                unsigned _stv8_33_4 = __float_as_uint(acc[v_21 * 8 + 4]);
                unsigned _stv8_33_5 = __float_as_uint(acc[v_21 * 8 + 5]);
                unsigned _stv8_33_6 = __float_as_uint(acc[v_21 * 8 + 6]);
                unsigned _stv8_33_7 = __float_as_uint(acc[v_21 * 8 + 7]);
                asm volatile(
                    "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                    :: "l"((void*)(partial + (pbase + (unsigned long long)pcolv))), "r"(_stv8_33_0), "r"(_stv8_33_1), "r"(_stv8_33_2), "r"(_stv8_33_3), "r"(_stv8_33_4), "r"(_stv8_33_5), "r"(_stv8_33_6), "r"(_stv8_33_7) : "memory");
            }
        }
        if (tid == 0) {
            flags[claim_it & 1] = (int)nxt;
        }
        __syncthreads();
        c = flags[claim_it & 1];
    }
#if __CUDA_ARCH__ != 1070
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
#endif
    if (tid == 0) {
        if (c == n_chunks + (int)gridDim.x - 1) {
            *(reinterpret_cast<unsigned int*>(counters) + (0)) = 0;
        }
    }
}

} // extern "C"
