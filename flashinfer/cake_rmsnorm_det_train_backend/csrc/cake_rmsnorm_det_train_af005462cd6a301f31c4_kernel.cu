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
#define SMEM_DOT_SLOTS_STAGE_BYTES 32
#define SMEM_DOT_SLOTS_STRIDE 32
#define SMEM_FLAGS_OFF 128
#define SMEM_FLAGS_STAGE_BYTES 32
#define SMEM_FLAGS_STRIDE 32
#define SMEM_RED_FOLD_OFF 256
#define SMEM_RED_FOLD_STAGE_BYTES 2048
#define SMEM_RED_FOLD_STRIDE 2048
#define SMEM_ROW_FOLD_OFF 2304
#define SMEM_ROW_FOLD_STAGE_BYTES 4096
#define SMEM_ROW_FOLD_STRIDE 4096
#define SMEM_COMP_SMEM_OFF 6400
#define SMEM_COMP_SMEM_STAGE_BYTES 128
#define SMEM_COMP_SMEM_STRIDE 128
#define SMEM_W_SM_OFF 6528
#define SMEM_W_SM_STAGE_BYTES 16
#define SMEM_W_SM_STRIDE 16
#define SMEM_PART_SM_OFF 6656
#define SMEM_PART_SM_STAGE_BYTES 16
#define SMEM_PART_SM_STRIDE 16
#define SMEM_TOTAL 6784
#define THREADS 128
#define LAUNCH_MIN_BLOCKS 4

#include <math_constants.h>

extern "C" {

__global__ __launch_bounds__(128, LAUNCH_MIN_BLOCKS) void
kernel_cake_rmsnorm_det_train_af005462cd6a301f31c4(__nv_bfloat16* __restrict__ g, __nv_bfloat16* __restrict__ x, __nv_bfloat16* __restrict__ w, float* __restrict__ r, __nv_bfloat16* __restrict__ dx, float* __restrict__ dw, float* __restrict__ partial, unsigned int* __restrict__ counters, __nv_bfloat16* __restrict__ g_h, int T, int n_chunks, int rows_per_chunk, long long g_stride_p0, long long x_stride_p0, long long g_h_stride_p0)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
#if __CUDA_ARCH__ == 1000
    asm volatile("{ .reg .u64 smem_ptr; cvta.to.shared.u64 smem_ptr, %1; cvt.u32.u64 %0, smem_ptr; }" : "=r"(smem) : "l"(smem_raw));
    smem = make_warp_uniform(smem);
#else
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);
#endif

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    float* dot_slots = reinterpret_cast<float*>(smem_raw + 0);
    const int dot_slots_addr = smem + 0;
    int* flags = reinterpret_cast<int*>(smem_raw + 128);
    const int flags_addr = smem + 128;
    float* red_fold = reinterpret_cast<float*>(smem_raw + 256);
    const int red_fold_addr = smem + 256;
    float* row_fold = reinterpret_cast<float*>(smem_raw + 2304);
    const int row_fold_addr = smem + 2304;
    float* comp_smem = reinterpret_cast<float*>(smem_raw + 6400);
    const int comp_smem_addr = smem + 6400;
    unsigned int* w_sm = reinterpret_cast<unsigned int*>(smem_raw + 6528);
    const int w_sm_addr = smem + 6528;
    float* part_sm = reinterpret_cast<float*>(smem_raw + 6656);
    const int part_sm_addr = smem + 6656;

    // === Task calls (declaration order) ===
    asm volatile("griddepcontrol.wait;" ::: "memory");
    unsigned int nxt = 0;
    if (tid == 0) {
        unsigned int _atomic_fetch_add_0 = atomicAdd(&counters[0], 1);
        nxt = _atomic_fetch_add_0;
    }
    long long g_stride = g_stride_p0;
    long long x_stride = x_stride_p0;
    long long gh_stride = g_h_stride_p0;
    int c = 0;
    int rg = warp;
    int tir = tid - rg * 32;
    int row_begin = 0;
    int row_end = 0;
    int n_rows = 0;
    int n_iters = 0;
    int last_row = 0;
    float wc[8];
    unsigned int ww[4];
    #pragma unroll
    for (int v = 0; v < 1; v++) {
        int wcolp = (v * 32 + tir) * 8;
        {
            const uint4* _vptr_1 = reinterpret_cast<const uint4*>(w + (unsigned long long)wcolp);
            uint4* _vdst_1 = reinterpret_cast<uint4*>(&ww[v * 4]);
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vdst_1[_blk] = _vptr_1[_blk];
            }
        }
    }
    float acc[8];
    float comp[8];
    unsigned int compw[4];
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
        int stages = (T + 3) / 4;
        int sl = rows_per_chunk;
        if (sl > stages) {
            sl = stages;
        }
        if (sl < 1) {
            sl = 1;
        }
        int share = (stages + sl - 1) / sl;
        int tail = (share + 3) / 4;
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
        row_begin = begin_it * 4;
        row_end = row_begin + rows * 4;
        if (row_begin > T) {
            row_begin = T;
        }
        if (row_end > T) {
            row_end = T;
        }
        n_rows = row_end - row_begin;
        n_iters = (n_rows + 3) / 4;
        last_row = row_end - 1;
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            acc[j] = 0.0f;
            comp[j] = 0.0f;
        }
        unsigned int gwA[4];
        unsigned int xwA[4];
        unsigned int ghwA[4];
        float rrA[1];
        unsigned int gwB[4];
        unsigned int xwB[4];
        unsigned int ghwB[4];
        float rrB[1];
        unsigned int gwC[4];
        unsigned int xwC[4];
        float rrC[1];
        unsigned int gwD[4];
        unsigned int xwD[4];
        float rrD[1];
        if (n_iters > 0) {
            int row0 = ((row_begin + rg) < (last_row) ? (row_begin + rg) : (last_row));
            unsigned long long g_base = (unsigned long long)row0 * (unsigned long long)g_stride;
            unsigned long long x_base = (unsigned long long)row0 * (unsigned long long)x_stride;
            rrA[0] = r[row0];
            #pragma unroll
            for (int v_1 = 0; v_1 < 1; v_1++) {
                int coln = (v_1 * 32 + tir) * 8;
                {
                    const uint4* _vptr_3 = reinterpret_cast<const uint4*>(g + g_base + (unsigned long long)coln);
                    uint4* _vdst_3 = reinterpret_cast<uint4*>(&gwA[v_1 * 4]);
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                            : "=r"(_vdst_3[_blk].x), "=r"(_vdst_3[_blk].y), "=r"(_vdst_3[_blk].z), "=r"(_vdst_3[_blk].w) : "l"((const void*)(_vptr_3 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                    }
                }
                {
                    const uint4* _vptr_4 = reinterpret_cast<const uint4*>(x + x_base + (unsigned long long)coln);
                    uint4* _vdst_4 = reinterpret_cast<uint4*>(&xwA[v_1 * 4]);
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                            : "=r"(_vdst_4[_blk].x), "=r"(_vdst_4[_blk].y), "=r"(_vdst_4[_blk].z), "=r"(_vdst_4[_blk].w) : "l"((const void*)(_vptr_4 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                    }
                }
            }
            int rowp1_ld = ((row_begin + 4 + rg) < (last_row) ? (row_begin + 4 + rg) : (last_row));
            unsigned long long g_base_1 = (unsigned long long)rowp1_ld * (unsigned long long)g_stride;
            unsigned long long x_base_0 = (unsigned long long)rowp1_ld * (unsigned long long)x_stride;
            rrB[0] = r[rowp1_ld];
            #pragma unroll
            for (int v_2 = 0; v_2 < 1; v_2++) {
                int coln_1 = (v_2 * 32 + tir) * 8;
                {
                    const uint4* _vptr_5 = reinterpret_cast<const uint4*>(g + g_base_1 + (unsigned long long)coln_1);
                    uint4* _vdst_5 = reinterpret_cast<uint4*>(&gwB[v_2 * 4]);
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                            : "=r"(_vdst_5[_blk].x), "=r"(_vdst_5[_blk].y), "=r"(_vdst_5[_blk].z), "=r"(_vdst_5[_blk].w) : "l"((const void*)(_vptr_5 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                    }
                }
                {
                    const uint4* _vptr_6 = reinterpret_cast<const uint4*>(x + x_base_0 + (unsigned long long)coln_1);
                    uint4* _vdst_6 = reinterpret_cast<uint4*>(&xwB[v_2 * 4]);
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                            : "=r"(_vdst_6[_blk].x), "=r"(_vdst_6[_blk].y), "=r"(_vdst_6[_blk].z), "=r"(_vdst_6[_blk].w) : "l"((const void*)(_vptr_6 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                    }
                }
            }
            int rowp2_ld = ((row_begin + 8 + rg) < (last_row) ? (row_begin + 8 + rg) : (last_row));
            unsigned long long g_base_2 = (unsigned long long)rowp2_ld * (unsigned long long)g_stride;
            unsigned long long x_base_2 = (unsigned long long)rowp2_ld * (unsigned long long)x_stride;
            rrC[0] = r[rowp2_ld];
            #pragma unroll
            for (int v_3 = 0; v_3 < 1; v_3++) {
                int coln_2 = (v_3 * 32 + tir) * 8;
                {
                    const uint4* _vptr_7 = reinterpret_cast<const uint4*>(g + g_base_2 + (unsigned long long)coln_2);
                    uint4* _vdst_7 = reinterpret_cast<uint4*>(&gwC[v_3 * 4]);
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                            : "=r"(_vdst_7[_blk].x), "=r"(_vdst_7[_blk].y), "=r"(_vdst_7[_blk].z), "=r"(_vdst_7[_blk].w) : "l"((const void*)(_vptr_7 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                    }
                }
                {
                    const uint4* _vptr_8 = reinterpret_cast<const uint4*>(x + x_base_2 + (unsigned long long)coln_2);
                    uint4* _vdst_8 = reinterpret_cast<uint4*>(&xwC[v_3 * 4]);
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                            : "=r"(_vdst_8[_blk].x), "=r"(_vdst_8[_blk].y), "=r"(_vdst_8[_blk].z), "=r"(_vdst_8[_blk].w) : "l"((const void*)(_vptr_8 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                    }
                }
            }
            int _max_9 = ((n_iters - 3) > (0) ? (n_iters - 3) : (0));
            int n_full4 = _max_9 / 4 * 4;
            #pragma unroll 1
            for (int it = 0; it < n_full4; it += 4) {
                int row4a = row_begin + it * 4 + rg;
                int row4d_ld = ((row4a + 12) < (last_row) ? (row4a + 12) : (last_row));
                unsigned long long g_base_3 = (unsigned long long)row4d_ld * (unsigned long long)g_stride;
                unsigned long long x_base_3 = (unsigned long long)row4d_ld * (unsigned long long)x_stride;
                rrD[0] = r[row4d_ld];
                #pragma unroll
                for (int v_4 = 0; v_4 < 1; v_4++) {
                    int coln_3 = (v_4 * 32 + tir) * 8;
                    {
                        const uint4* _vptr_10 = reinterpret_cast<const uint4*>(g + g_base_3 + (unsigned long long)coln_3);
                        uint4* _vdst_10 = reinterpret_cast<uint4*>(&gwD[v_4 * 4]);
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                : "=r"(_vdst_10[_blk].x), "=r"(_vdst_10[_blk].y), "=r"(_vdst_10[_blk].z), "=r"(_vdst_10[_blk].w) : "l"((const void*)(_vptr_10 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                        }
                    }
                    {
                        const uint4* _vptr_11 = reinterpret_cast<const uint4*>(x + x_base_3 + (unsigned long long)coln_3);
                        uint4* _vdst_11 = reinterpret_cast<uint4*>(&xwD[v_4 * 4]);
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                : "=r"(_vdst_11[_blk].x), "=r"(_vdst_11[_blk].y), "=r"(_vdst_11[_blk].z), "=r"(_vdst_11[_blk].w) : "l"((const void*)(_vptr_11 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                        }
                    }
                }
                float dq = 0.0f;
                #pragma unroll
                for (int v_5 = 0; v_5 < 1; v_5++) {
                    unsigned int wvd[4];
                    #pragma unroll
                    for (int e = 0; e < 4; e++) {
                        unsigned int gwd = gwA[v_5 * 4 + e];
                        unsigned int xwd = xwA[v_5 * 4 + e];
                        float g0 = __uint_as_float(gwd << 16);
                        float g1 = __uint_as_float(gwd & 4294901760u);
                        float x0 = __uint_as_float(xwd << 16);
                        float x1 = __uint_as_float(xwd & 4294901760u);
                        dq = __fmaf_rn(g0 * __uint_as_float(ww[v_5 * 4 + e] << 16), x0, dq);
                        dq = __fmaf_rn(g1 * __uint_as_float(ww[v_5 * 4 + e] & 4294901760u), x1, dq);
                    }
                }
                float dots[1];
                float _warp_reduce_12 = dq;
                #pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1)
                    _warp_reduce_12 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_12, offset);
                dots[0] = _warp_reduce_12;
                float rq = rrA[0];
                float cq = __fdiv_rn(dots[0], 256.0f);
                float kx = rq * rq * rq * cq;
                float nkx = -kx;
                if (row4a < row_end) {
                    unsigned long long dx_base = (unsigned long long)row4a * 256;
                    #pragma unroll
                    for (int v_6 = 0; v_6 < 1; v_6++) {
                        float out[8];
                        unsigned int wv[4];
                        #pragma unroll
                        for (int e_1 = 0; e_1 < 4; e_1++) {
                            unsigned int gwd_1 = gwA[v_6 * 4 + e_1];
                            unsigned int xwd_1 = xwA[v_6 * 4 + e_1];
                            float g0_1 = __uint_as_float(gwd_1 << 16);
                            float g1_1 = __uint_as_float(gwd_1 & 4294901760u);
                            float x0_1 = __uint_as_float(xwd_1 << 16);
                            float x1_1 = __uint_as_float(xwd_1 & 4294901760u);
                            float y = __fmaf_rn(g0_1 * x0_1, rq, -comp[v_6 * 8 + 2 * e_1]);
                            float t = acc[v_6 * 8 + 2 * e_1] + y;
                            comp[v_6 * 8 + 2 * e_1] = t - acc[v_6 * 8 + 2 * e_1] - y;
                            acc[v_6 * 8 + 2 * e_1] = t;
                            float y_0 = __fmaf_rn(g1_1 * x1_1, rq, -comp[v_6 * 8 + 2 * e_1 + 1]);
                            float t_0 = acc[v_6 * 8 + 2 * e_1 + 1] + y_0;
                            comp[v_6 * 8 + 2 * e_1 + 1] = t_0 - acc[v_6 * 8 + 2 * e_1 + 1] - y_0;
                            acc[v_6 * 8 + 2 * e_1 + 1] = t_0;
                            out[2 * e_1] = __fmaf_rn(nkx, x0_1, rq * (g0_1 * __uint_as_float(ww[v_6 * 4 + e_1] << 16)));
                            out[2 * e_1 + 1] = __fmaf_rn(nkx, x1_1, rq * (g1_1 * __uint_as_float(ww[v_6 * 4 + e_1] & 4294901760u)));
                        }
                        int col = (v_6 * 32 + tir) * 8;
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
                }
                int row4a_ld = ((row4a + 16) < (last_row) ? (row4a + 16) : (last_row));
                unsigned long long g_base_4 = (unsigned long long)row4a_ld * (unsigned long long)g_stride;
                unsigned long long x_base_4 = (unsigned long long)row4a_ld * (unsigned long long)x_stride;
                rrA[0] = r[row4a_ld];
                #pragma unroll
                for (int v_7 = 0; v_7 < 1; v_7++) {
                    int coln_4 = (v_7 * 32 + tir) * 8;
                    {
                        const uint4* _vptr_13 = reinterpret_cast<const uint4*>(g + g_base_4 + (unsigned long long)coln_4);
                        uint4* _vdst_13 = reinterpret_cast<uint4*>(&gwA[v_7 * 4]);
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                : "=r"(_vdst_13[_blk].x), "=r"(_vdst_13[_blk].y), "=r"(_vdst_13[_blk].z), "=r"(_vdst_13[_blk].w) : "l"((const void*)(_vptr_13 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                        }
                    }
                    {
                        const uint4* _vptr_14 = reinterpret_cast<const uint4*>(x + x_base_4 + (unsigned long long)coln_4);
                        uint4* _vdst_14 = reinterpret_cast<uint4*>(&xwA[v_7 * 4]);
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                : "=r"(_vdst_14[_blk].x), "=r"(_vdst_14[_blk].y), "=r"(_vdst_14[_blk].z), "=r"(_vdst_14[_blk].w) : "l"((const void*)(_vptr_14 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                        }
                    }
                }
                float dq_1 = 0.0f;
                #pragma unroll
                for (int v_8 = 0; v_8 < 1; v_8++) {
                    unsigned int wvd_1[4];
                    #pragma unroll
                    for (int e_2 = 0; e_2 < 4; e_2++) {
                        unsigned int gwd_2 = gwB[v_8 * 4 + e_2];
                        unsigned int xwd_2 = xwB[v_8 * 4 + e_2];
                        float g0_2 = __uint_as_float(gwd_2 << 16);
                        float g1_2 = __uint_as_float(gwd_2 & 4294901760u);
                        float x0_2 = __uint_as_float(xwd_2 << 16);
                        float x1_2 = __uint_as_float(xwd_2 & 4294901760u);
                        dq_1 = __fmaf_rn(g0_2 * __uint_as_float(ww[v_8 * 4 + e_2] << 16), x0_2, dq_1);
                        dq_1 = __fmaf_rn(g1_2 * __uint_as_float(ww[v_8 * 4 + e_2] & 4294901760u), x1_2, dq_1);
                    }
                }
                float dots_0[1];
                float _warp_reduce_15 = dq_1;
                #pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1)
                    _warp_reduce_15 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_15, offset);
                dots_0[0] = _warp_reduce_15;
                float rq_0 = rrB[0];
                float cq_0 = __fdiv_rn(dots_0[0], 256.0f);
                float kx_0 = rq_0 * rq_0 * rq_0 * cq_0;
                float nkx_0 = -kx_0;
                if (row4a + 4 < row_end) {
                    unsigned long long dx_base_1 = (unsigned long long)(row4a + 4) * 256;
                    #pragma unroll
                    for (int v_9 = 0; v_9 < 1; v_9++) {
                        float out_1[8];
                        unsigned int wv_1[4];
                        #pragma unroll
                        for (int e_3 = 0; e_3 < 4; e_3++) {
                            unsigned int gwd_3 = gwB[v_9 * 4 + e_3];
                            unsigned int xwd_3 = xwB[v_9 * 4 + e_3];
                            float g0_3 = __uint_as_float(gwd_3 << 16);
                            float g1_3 = __uint_as_float(gwd_3 & 4294901760u);
                            float x0_3 = __uint_as_float(xwd_3 << 16);
                            float x1_3 = __uint_as_float(xwd_3 & 4294901760u);
                            float y_1 = __fmaf_rn(g0_3 * x0_3, rq_0, -comp[v_9 * 8 + 2 * e_3]);
                            float t_1 = acc[v_9 * 8 + 2 * e_3] + y_1;
                            comp[v_9 * 8 + 2 * e_3] = t_1 - acc[v_9 * 8 + 2 * e_3] - y_1;
                            acc[v_9 * 8 + 2 * e_3] = t_1;
                            float y_1_1 = __fmaf_rn(g1_3 * x1_3, rq_0, -comp[v_9 * 8 + 2 * e_3 + 1]);
                            float t_1_1 = acc[v_9 * 8 + 2 * e_3 + 1] + y_1_1;
                            comp[v_9 * 8 + 2 * e_3 + 1] = t_1_1 - acc[v_9 * 8 + 2 * e_3 + 1] - y_1_1;
                            acc[v_9 * 8 + 2 * e_3 + 1] = t_1_1;
                            out_1[2 * e_3] = __fmaf_rn(nkx_0, x0_3, rq_0 * (g0_3 * __uint_as_float(ww[v_9 * 4 + e_3] << 16)));
                            out_1[2 * e_3 + 1] = __fmaf_rn(nkx_0, x1_3, rq_0 * (g1_3 * __uint_as_float(ww[v_9 * 4 + e_3] & 4294901760u)));
                        }
                        int col_1 = (v_9 * 32 + tir) * 8;
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
                        reinterpret_cast<int4*>(dx + (dx_base_1 + (unsigned long long)col_1))[0] = reinterpret_cast<int4*>(packed_1)[0];
                    }
                }
                int row4b_ld = ((row4a + 20) < (last_row) ? (row4a + 20) : (last_row));
                unsigned long long g_base_5 = (unsigned long long)row4b_ld * (unsigned long long)g_stride;
                unsigned long long x_base_5 = (unsigned long long)row4b_ld * (unsigned long long)x_stride;
                rrB[0] = r[row4b_ld];
                #pragma unroll
                for (int v_10 = 0; v_10 < 1; v_10++) {
                    int coln_5 = (v_10 * 32 + tir) * 8;
                    {
                        const uint4* _vptr_16 = reinterpret_cast<const uint4*>(g + g_base_5 + (unsigned long long)coln_5);
                        uint4* _vdst_16 = reinterpret_cast<uint4*>(&gwB[v_10 * 4]);
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                : "=r"(_vdst_16[_blk].x), "=r"(_vdst_16[_blk].y), "=r"(_vdst_16[_blk].z), "=r"(_vdst_16[_blk].w) : "l"((const void*)(_vptr_16 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                        }
                    }
                    {
                        const uint4* _vptr_17 = reinterpret_cast<const uint4*>(x + x_base_5 + (unsigned long long)coln_5);
                        uint4* _vdst_17 = reinterpret_cast<uint4*>(&xwB[v_10 * 4]);
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                : "=r"(_vdst_17[_blk].x), "=r"(_vdst_17[_blk].y), "=r"(_vdst_17[_blk].z), "=r"(_vdst_17[_blk].w) : "l"((const void*)(_vptr_17 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                        }
                    }
                }
                float dq_2 = 0.0f;
                #pragma unroll
                for (int v_11 = 0; v_11 < 1; v_11++) {
                    unsigned int wvd_2[4];
                    #pragma unroll
                    for (int e_4 = 0; e_4 < 4; e_4++) {
                        unsigned int gwd_4 = gwC[v_11 * 4 + e_4];
                        unsigned int xwd_4 = xwC[v_11 * 4 + e_4];
                        float g0_4 = __uint_as_float(gwd_4 << 16);
                        float g1_4 = __uint_as_float(gwd_4 & 4294901760u);
                        float x0_4 = __uint_as_float(xwd_4 << 16);
                        float x1_4 = __uint_as_float(xwd_4 & 4294901760u);
                        dq_2 = __fmaf_rn(g0_4 * __uint_as_float(ww[v_11 * 4 + e_4] << 16), x0_4, dq_2);
                        dq_2 = __fmaf_rn(g1_4 * __uint_as_float(ww[v_11 * 4 + e_4] & 4294901760u), x1_4, dq_2);
                    }
                }
                float dots_2[1];
                float _warp_reduce_18 = dq_2;
                #pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1)
                    _warp_reduce_18 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_18, offset);
                dots_2[0] = _warp_reduce_18;
                float rq_1 = rrC[0];
                float cq_1 = __fdiv_rn(dots_2[0], 256.0f);
                float kx_1 = rq_1 * rq_1 * rq_1 * cq_1;
                float nkx_1 = -kx_1;
                if (row4a + 8 < row_end) {
                    unsigned long long dx_base_2 = (unsigned long long)(row4a + 8) * 256;
                    #pragma unroll
                    for (int v_12 = 0; v_12 < 1; v_12++) {
                        float out_2[8];
                        unsigned int wv_2[4];
                        #pragma unroll
                        for (int e_5 = 0; e_5 < 4; e_5++) {
                            unsigned int gwd_5 = gwC[v_12 * 4 + e_5];
                            unsigned int xwd_5 = xwC[v_12 * 4 + e_5];
                            float g0_5 = __uint_as_float(gwd_5 << 16);
                            float g1_5 = __uint_as_float(gwd_5 & 4294901760u);
                            float x0_5 = __uint_as_float(xwd_5 << 16);
                            float x1_5 = __uint_as_float(xwd_5 & 4294901760u);
                            float y_2 = __fmaf_rn(g0_5 * x0_5, rq_1, -comp[v_12 * 8 + 2 * e_5]);
                            float t_2 = acc[v_12 * 8 + 2 * e_5] + y_2;
                            comp[v_12 * 8 + 2 * e_5] = t_2 - acc[v_12 * 8 + 2 * e_5] - y_2;
                            acc[v_12 * 8 + 2 * e_5] = t_2;
                            float y_2_1 = __fmaf_rn(g1_5 * x1_5, rq_1, -comp[v_12 * 8 + 2 * e_5 + 1]);
                            float t_2_1 = acc[v_12 * 8 + 2 * e_5 + 1] + y_2_1;
                            comp[v_12 * 8 + 2 * e_5 + 1] = t_2_1 - acc[v_12 * 8 + 2 * e_5 + 1] - y_2_1;
                            acc[v_12 * 8 + 2 * e_5 + 1] = t_2_1;
                            out_2[2 * e_5] = __fmaf_rn(nkx_1, x0_5, rq_1 * (g0_5 * __uint_as_float(ww[v_12 * 4 + e_5] << 16)));
                            out_2[2 * e_5 + 1] = __fmaf_rn(nkx_1, x1_5, rq_1 * (g1_5 * __uint_as_float(ww[v_12 * 4 + e_5] & 4294901760u)));
                        }
                        int col_2 = (v_12 * 32 + tir) * 8;
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
                        reinterpret_cast<int4*>(dx + (dx_base_2 + (unsigned long long)col_2))[0] = reinterpret_cast<int4*>(packed_2)[0];
                    }
                }
                int row4c_ld = ((row4a + 24) < (last_row) ? (row4a + 24) : (last_row));
                unsigned long long g_base_6 = (unsigned long long)row4c_ld * (unsigned long long)g_stride;
                unsigned long long x_base_6 = (unsigned long long)row4c_ld * (unsigned long long)x_stride;
                rrC[0] = r[row4c_ld];
                #pragma unroll
                for (int v_13 = 0; v_13 < 1; v_13++) {
                    int coln_6 = (v_13 * 32 + tir) * 8;
                    {
                        const uint4* _vptr_19 = reinterpret_cast<const uint4*>(g + g_base_6 + (unsigned long long)coln_6);
                        uint4* _vdst_19 = reinterpret_cast<uint4*>(&gwC[v_13 * 4]);
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                : "=r"(_vdst_19[_blk].x), "=r"(_vdst_19[_blk].y), "=r"(_vdst_19[_blk].z), "=r"(_vdst_19[_blk].w) : "l"((const void*)(_vptr_19 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                        }
                    }
                    {
                        const uint4* _vptr_20 = reinterpret_cast<const uint4*>(x + x_base_6 + (unsigned long long)coln_6);
                        uint4* _vdst_20 = reinterpret_cast<uint4*>(&xwC[v_13 * 4]);
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                : "=r"(_vdst_20[_blk].x), "=r"(_vdst_20[_blk].y), "=r"(_vdst_20[_blk].z), "=r"(_vdst_20[_blk].w) : "l"((const void*)(_vptr_20 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                        }
                    }
                }
                float dq_3 = 0.0f;
                #pragma unroll
                for (int v_14 = 0; v_14 < 1; v_14++) {
                    unsigned int wvd_3[4];
                    #pragma unroll
                    for (int e_6 = 0; e_6 < 4; e_6++) {
                        unsigned int gwd_6 = gwD[v_14 * 4 + e_6];
                        unsigned int xwd_6 = xwD[v_14 * 4 + e_6];
                        float g0_6 = __uint_as_float(gwd_6 << 16);
                        float g1_6 = __uint_as_float(gwd_6 & 4294901760u);
                        float x0_6 = __uint_as_float(xwd_6 << 16);
                        float x1_6 = __uint_as_float(xwd_6 & 4294901760u);
                        dq_3 = __fmaf_rn(g0_6 * __uint_as_float(ww[v_14 * 4 + e_6] << 16), x0_6, dq_3);
                        dq_3 = __fmaf_rn(g1_6 * __uint_as_float(ww[v_14 * 4 + e_6] & 4294901760u), x1_6, dq_3);
                    }
                }
                float dots_3[1];
                float _warp_reduce_21 = dq_3;
                #pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1)
                    _warp_reduce_21 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_21, offset);
                dots_3[0] = _warp_reduce_21;
                float rq_3 = rrD[0];
                float cq_2 = __fdiv_rn(dots_3[0], 256.0f);
                float kx_2 = rq_3 * rq_3 * rq_3 * cq_2;
                float nkx_2 = -kx_2;
                if (row4a + 12 < row_end) {
                    unsigned long long dx_base_3 = (unsigned long long)(row4a + 12) * 256;
                    #pragma unroll
                    for (int v_15 = 0; v_15 < 1; v_15++) {
                        float out_3[8];
                        unsigned int wv_3[4];
                        #pragma unroll
                        for (int e_7 = 0; e_7 < 4; e_7++) {
                            unsigned int gwd_7 = gwD[v_15 * 4 + e_7];
                            unsigned int xwd_7 = xwD[v_15 * 4 + e_7];
                            float g0_7 = __uint_as_float(gwd_7 << 16);
                            float g1_7 = __uint_as_float(gwd_7 & 4294901760u);
                            float x0_7 = __uint_as_float(xwd_7 << 16);
                            float x1_7 = __uint_as_float(xwd_7 & 4294901760u);
                            float y_3 = __fmaf_rn(g0_7 * x0_7, rq_3, -comp[v_15 * 8 + 2 * e_7]);
                            float t_3 = acc[v_15 * 8 + 2 * e_7] + y_3;
                            comp[v_15 * 8 + 2 * e_7] = t_3 - acc[v_15 * 8 + 2 * e_7] - y_3;
                            acc[v_15 * 8 + 2 * e_7] = t_3;
                            float y_3_1 = __fmaf_rn(g1_7 * x1_7, rq_3, -comp[v_15 * 8 + 2 * e_7 + 1]);
                            float t_3_1 = acc[v_15 * 8 + 2 * e_7 + 1] + y_3_1;
                            comp[v_15 * 8 + 2 * e_7 + 1] = t_3_1 - acc[v_15 * 8 + 2 * e_7 + 1] - y_3_1;
                            acc[v_15 * 8 + 2 * e_7 + 1] = t_3_1;
                            out_3[2 * e_7] = __fmaf_rn(nkx_2, x0_7, rq_3 * (g0_7 * __uint_as_float(ww[v_15 * 4 + e_7] << 16)));
                            out_3[2 * e_7 + 1] = __fmaf_rn(nkx_2, x1_7, rq_3 * (g1_7 * __uint_as_float(ww[v_15 * 4 + e_7] & 4294901760u)));
                        }
                        int col_3 = (v_15 * 32 + tir) * 8;
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
                        reinterpret_cast<int4*>(dx + (dx_base_3 + (unsigned long long)col_3))[0] = reinterpret_cast<int4*>(packed_3)[0];
                    }
                }
            }
            #pragma unroll 1
            for (int it_1 = n_full4; it_1 < n_iters; it_1 += 4) {
                int row5a = row_begin + it_1 * 4 + rg;
                if (n_iters > it_1 + 3) {
                    int row5d_ld = ((row5a + 12) < (last_row) ? (row5a + 12) : (last_row));
                    unsigned long long g_base_7 = (unsigned long long)row5d_ld * (unsigned long long)g_stride;
                    unsigned long long x_base_7 = (unsigned long long)row5d_ld * (unsigned long long)x_stride;
                    rrD[0] = r[row5d_ld];
                    #pragma unroll
                    for (int v_16 = 0; v_16 < 1; v_16++) {
                        int coln_7 = (v_16 * 32 + tir) * 8;
                        {
                            const uint4* _vptr_22 = reinterpret_cast<const uint4*>(g + g_base_7 + (unsigned long long)coln_7);
                            uint4* _vdst_22 = reinterpret_cast<uint4*>(&gwD[v_16 * 4]);
                            #pragma unroll
                            for (int _blk = 0; _blk < 1; _blk++) {
                                asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                    : "=r"(_vdst_22[_blk].x), "=r"(_vdst_22[_blk].y), "=r"(_vdst_22[_blk].z), "=r"(_vdst_22[_blk].w) : "l"((const void*)(_vptr_22 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                            }
                        }
                        {
                            const uint4* _vptr_23 = reinterpret_cast<const uint4*>(x + x_base_7 + (unsigned long long)coln_7);
                            uint4* _vdst_23 = reinterpret_cast<uint4*>(&xwD[v_16 * 4]);
                            #pragma unroll
                            for (int _blk = 0; _blk < 1; _blk++) {
                                asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                    : "=r"(_vdst_23[_blk].x), "=r"(_vdst_23[_blk].y), "=r"(_vdst_23[_blk].z), "=r"(_vdst_23[_blk].w) : "l"((const void*)(_vptr_23 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                            }
                        }
                    }
                }
                float dq_4 = 0.0f;
                #pragma unroll
                for (int v_17 = 0; v_17 < 1; v_17++) {
                    unsigned int wvd_4[4];
                    #pragma unroll
                    for (int e_8 = 0; e_8 < 4; e_8++) {
                        unsigned int gwd_8 = gwA[v_17 * 4 + e_8];
                        unsigned int xwd_8 = xwA[v_17 * 4 + e_8];
                        float g0_8 = __uint_as_float(gwd_8 << 16);
                        float g1_8 = __uint_as_float(gwd_8 & 4294901760u);
                        float x0_8 = __uint_as_float(xwd_8 << 16);
                        float x1_8 = __uint_as_float(xwd_8 & 4294901760u);
                        dq_4 = __fmaf_rn(g0_8 * __uint_as_float(ww[v_17 * 4 + e_8] << 16), x0_8, dq_4);
                        dq_4 = __fmaf_rn(g1_8 * __uint_as_float(ww[v_17 * 4 + e_8] & 4294901760u), x1_8, dq_4);
                    }
                }
                float dots_1[1];
                float _warp_reduce_24 = dq_4;
                #pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1)
                    _warp_reduce_24 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_24, offset);
                dots_1[0] = _warp_reduce_24;
                float rq_2 = rrA[0];
                float cq_3 = __fdiv_rn(dots_1[0], 256.0f);
                float kx_3 = rq_2 * rq_2 * rq_2 * cq_3;
                float nkx_3 = -kx_3;
                if (row5a < row_end) {
                    unsigned long long dx_base_4 = (unsigned long long)row5a * 256;
                    #pragma unroll
                    for (int v_18 = 0; v_18 < 1; v_18++) {
                        float out_4[8];
                        unsigned int wv_4[4];
                        #pragma unroll
                        for (int e_9 = 0; e_9 < 4; e_9++) {
                            unsigned int gwd_9 = gwA[v_18 * 4 + e_9];
                            unsigned int xwd_9 = xwA[v_18 * 4 + e_9];
                            float g0_9 = __uint_as_float(gwd_9 << 16);
                            float g1_9 = __uint_as_float(gwd_9 & 4294901760u);
                            float x0_9 = __uint_as_float(xwd_9 << 16);
                            float x1_9 = __uint_as_float(xwd_9 & 4294901760u);
                            float y_4 = __fmaf_rn(g0_9 * x0_9, rq_2, -comp[v_18 * 8 + 2 * e_9]);
                            float t_4 = acc[v_18 * 8 + 2 * e_9] + y_4;
                            comp[v_18 * 8 + 2 * e_9] = t_4 - acc[v_18 * 8 + 2 * e_9] - y_4;
                            acc[v_18 * 8 + 2 * e_9] = t_4;
                            float y_4_1 = __fmaf_rn(g1_9 * x1_9, rq_2, -comp[v_18 * 8 + 2 * e_9 + 1]);
                            float t_4_1 = acc[v_18 * 8 + 2 * e_9 + 1] + y_4_1;
                            comp[v_18 * 8 + 2 * e_9 + 1] = t_4_1 - acc[v_18 * 8 + 2 * e_9 + 1] - y_4_1;
                            acc[v_18 * 8 + 2 * e_9 + 1] = t_4_1;
                            out_4[2 * e_9] = __fmaf_rn(nkx_3, x0_9, rq_2 * (g0_9 * __uint_as_float(ww[v_18 * 4 + e_9] << 16)));
                            out_4[2 * e_9 + 1] = __fmaf_rn(nkx_3, x1_9, rq_2 * (g1_9 * __uint_as_float(ww[v_18 * 4 + e_9] & 4294901760u)));
                        }
                        int col_4 = (v_18 * 32 + tir) * 8;
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
                        reinterpret_cast<int4*>(dx + (dx_base_4 + (unsigned long long)col_4))[0] = reinterpret_cast<int4*>(packed_4)[0];
                    }
                }
                if (n_iters > it_1 + 1) {
                    if (n_iters > it_1 + 4) {
                        int row5a_ld = ((row5a + 16) < (last_row) ? (row5a + 16) : (last_row));
                        unsigned long long g_base_8 = (unsigned long long)row5a_ld * (unsigned long long)g_stride;
                        unsigned long long x_base_8 = (unsigned long long)row5a_ld * (unsigned long long)x_stride;
                        rrA[0] = r[row5a_ld];
                        #pragma unroll
                        for (int v_19 = 0; v_19 < 1; v_19++) {
                            int coln_8 = (v_19 * 32 + tir) * 8;
                            {
                                const uint4* _vptr_25 = reinterpret_cast<const uint4*>(g + g_base_8 + (unsigned long long)coln_8);
                                uint4* _vdst_25 = reinterpret_cast<uint4*>(&gwA[v_19 * 4]);
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
                                    asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                        : "=r"(_vdst_25[_blk].x), "=r"(_vdst_25[_blk].y), "=r"(_vdst_25[_blk].z), "=r"(_vdst_25[_blk].w) : "l"((const void*)(_vptr_25 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                                }
                            }
                            {
                                const uint4* _vptr_26 = reinterpret_cast<const uint4*>(x + x_base_8 + (unsigned long long)coln_8);
                                uint4* _vdst_26 = reinterpret_cast<uint4*>(&xwA[v_19 * 4]);
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
                                    asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                        : "=r"(_vdst_26[_blk].x), "=r"(_vdst_26[_blk].y), "=r"(_vdst_26[_blk].z), "=r"(_vdst_26[_blk].w) : "l"((const void*)(_vptr_26 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                                }
                            }
                        }
                    }
                    float dq_4_1 = 0.0f;
                    #pragma unroll
                    for (int v_20 = 0; v_20 < 1; v_20++) {
                        unsigned int wvd_5[4];
                        #pragma unroll
                        for (int e_10 = 0; e_10 < 4; e_10++) {
                            unsigned int gwd_10 = gwB[v_20 * 4 + e_10];
                            unsigned int xwd_10 = xwB[v_20 * 4 + e_10];
                            float g0_10 = __uint_as_float(gwd_10 << 16);
                            float g1_10 = __uint_as_float(gwd_10 & 4294901760u);
                            float x0_10 = __uint_as_float(xwd_10 << 16);
                            float x1_10 = __uint_as_float(xwd_10 & 4294901760u);
                            dq_4_1 = __fmaf_rn(g0_10 * __uint_as_float(ww[v_20 * 4 + e_10] << 16), x0_10, dq_4_1);
                            dq_4_1 = __fmaf_rn(g1_10 * __uint_as_float(ww[v_20 * 4 + e_10] & 4294901760u), x1_10, dq_4_1);
                        }
                    }
                    float dots_4[1];
                    float _warp_reduce_27 = dq_4_1;
                    #pragma unroll
                    for (int offset = 16; offset > 0; offset >>= 1)
                        _warp_reduce_27 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_27, offset);
                    dots_4[0] = _warp_reduce_27;
                    float rq_4 = rrB[0];
                    float cq_3_1 = __fdiv_rn(dots_4[0], 256.0f);
                    float kx_3_1 = rq_4 * rq_4 * rq_4 * cq_3_1;
                    float nkx_3_1 = -kx_3_1;
                    if (row5a + 4 < row_end) {
                        unsigned long long dx_base_5 = (unsigned long long)(row5a + 4) * 256;
                        #pragma unroll
                        for (int v_21 = 0; v_21 < 1; v_21++) {
                            float out_5[8];
                            unsigned int wv_5[4];
                            #pragma unroll
                            for (int e_11 = 0; e_11 < 4; e_11++) {
                                unsigned int gwd_11 = gwB[v_21 * 4 + e_11];
                                unsigned int xwd_11 = xwB[v_21 * 4 + e_11];
                                float g0_11 = __uint_as_float(gwd_11 << 16);
                                float g1_11 = __uint_as_float(gwd_11 & 4294901760u);
                                float x0_11 = __uint_as_float(xwd_11 << 16);
                                float x1_11 = __uint_as_float(xwd_11 & 4294901760u);
                                float y_5 = __fmaf_rn(g0_11 * x0_11, rq_4, -comp[v_21 * 8 + 2 * e_11]);
                                float t_5 = acc[v_21 * 8 + 2 * e_11] + y_5;
                                comp[v_21 * 8 + 2 * e_11] = t_5 - acc[v_21 * 8 + 2 * e_11] - y_5;
                                acc[v_21 * 8 + 2 * e_11] = t_5;
                                float y_5_1 = __fmaf_rn(g1_11 * x1_11, rq_4, -comp[v_21 * 8 + 2 * e_11 + 1]);
                                float t_5_1 = acc[v_21 * 8 + 2 * e_11 + 1] + y_5_1;
                                comp[v_21 * 8 + 2 * e_11 + 1] = t_5_1 - acc[v_21 * 8 + 2 * e_11 + 1] - y_5_1;
                                acc[v_21 * 8 + 2 * e_11 + 1] = t_5_1;
                                out_5[2 * e_11] = __fmaf_rn(nkx_3_1, x0_11, rq_4 * (g0_11 * __uint_as_float(ww[v_21 * 4 + e_11] << 16)));
                                out_5[2 * e_11 + 1] = __fmaf_rn(nkx_3_1, x1_11, rq_4 * (g1_11 * __uint_as_float(ww[v_21 * 4 + e_11] & 4294901760u)));
                            }
                            int col_5 = (v_21 * 32 + tir) * 8;
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
                            reinterpret_cast<int4*>(dx + (dx_base_5 + (unsigned long long)col_5))[0] = reinterpret_cast<int4*>(packed_5)[0];
                        }
                    }
                }
                if (n_iters > it_1 + 2) {
                    if (n_iters > it_1 + 5) {
                        int row5b_ld = ((row5a + 20) < (last_row) ? (row5a + 20) : (last_row));
                        unsigned long long g_base_9 = (unsigned long long)row5b_ld * (unsigned long long)g_stride;
                        unsigned long long x_base_9 = (unsigned long long)row5b_ld * (unsigned long long)x_stride;
                        rrB[0] = r[row5b_ld];
                        #pragma unroll
                        for (int v_22 = 0; v_22 < 1; v_22++) {
                            int coln_9 = (v_22 * 32 + tir) * 8;
                            {
                                const uint4* _vptr_28 = reinterpret_cast<const uint4*>(g + g_base_9 + (unsigned long long)coln_9);
                                uint4* _vdst_28 = reinterpret_cast<uint4*>(&gwB[v_22 * 4]);
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
                                    asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                        : "=r"(_vdst_28[_blk].x), "=r"(_vdst_28[_blk].y), "=r"(_vdst_28[_blk].z), "=r"(_vdst_28[_blk].w) : "l"((const void*)(_vptr_28 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                                }
                            }
                            {
                                const uint4* _vptr_29 = reinterpret_cast<const uint4*>(x + x_base_9 + (unsigned long long)coln_9);
                                uint4* _vdst_29 = reinterpret_cast<uint4*>(&xwB[v_22 * 4]);
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
                                    asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                        : "=r"(_vdst_29[_blk].x), "=r"(_vdst_29[_blk].y), "=r"(_vdst_29[_blk].z), "=r"(_vdst_29[_blk].w) : "l"((const void*)(_vptr_29 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                                }
                            }
                        }
                    }
                    float dq_5 = 0.0f;
                    #pragma unroll
                    for (int v_23 = 0; v_23 < 1; v_23++) {
                        unsigned int wvd_6[4];
                        #pragma unroll
                        for (int e_12 = 0; e_12 < 4; e_12++) {
                            unsigned int gwd_12 = gwC[v_23 * 4 + e_12];
                            unsigned int xwd_12 = xwC[v_23 * 4 + e_12];
                            float g0_12 = __uint_as_float(gwd_12 << 16);
                            float g1_12 = __uint_as_float(gwd_12 & 4294901760u);
                            float x0_12 = __uint_as_float(xwd_12 << 16);
                            float x1_12 = __uint_as_float(xwd_12 & 4294901760u);
                            dq_5 = __fmaf_rn(g0_12 * __uint_as_float(ww[v_23 * 4 + e_12] << 16), x0_12, dq_5);
                            dq_5 = __fmaf_rn(g1_12 * __uint_as_float(ww[v_23 * 4 + e_12] & 4294901760u), x1_12, dq_5);
                        }
                    }
                    float dots_5[1];
                    float _warp_reduce_30 = dq_5;
                    #pragma unroll
                    for (int offset = 16; offset > 0; offset >>= 1)
                        _warp_reduce_30 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_30, offset);
                    dots_5[0] = _warp_reduce_30;
                    float rq_5 = rrC[0];
                    float cq_5 = __fdiv_rn(dots_5[0], 256.0f);
                    float kx_4 = rq_5 * rq_5 * rq_5 * cq_5;
                    float nkx_4 = -kx_4;
                    if (row5a + 8 < row_end) {
                        unsigned long long dx_base_6 = (unsigned long long)(row5a + 8) * 256;
                        #pragma unroll
                        for (int v_24 = 0; v_24 < 1; v_24++) {
                            float out_6[8];
                            unsigned int wv_6[4];
                            #pragma unroll
                            for (int e_13 = 0; e_13 < 4; e_13++) {
                                unsigned int gwd_13 = gwC[v_24 * 4 + e_13];
                                unsigned int xwd_13 = xwC[v_24 * 4 + e_13];
                                float g0_13 = __uint_as_float(gwd_13 << 16);
                                float g1_13 = __uint_as_float(gwd_13 & 4294901760u);
                                float x0_13 = __uint_as_float(xwd_13 << 16);
                                float x1_13 = __uint_as_float(xwd_13 & 4294901760u);
                                float y_6 = __fmaf_rn(g0_13 * x0_13, rq_5, -comp[v_24 * 8 + 2 * e_13]);
                                float t_6 = acc[v_24 * 8 + 2 * e_13] + y_6;
                                comp[v_24 * 8 + 2 * e_13] = t_6 - acc[v_24 * 8 + 2 * e_13] - y_6;
                                acc[v_24 * 8 + 2 * e_13] = t_6;
                                float y_6_1 = __fmaf_rn(g1_13 * x1_13, rq_5, -comp[v_24 * 8 + 2 * e_13 + 1]);
                                float t_6_1 = acc[v_24 * 8 + 2 * e_13 + 1] + y_6_1;
                                comp[v_24 * 8 + 2 * e_13 + 1] = t_6_1 - acc[v_24 * 8 + 2 * e_13 + 1] - y_6_1;
                                acc[v_24 * 8 + 2 * e_13 + 1] = t_6_1;
                                out_6[2 * e_13] = __fmaf_rn(nkx_4, x0_13, rq_5 * (g0_13 * __uint_as_float(ww[v_24 * 4 + e_13] << 16)));
                                out_6[2 * e_13 + 1] = __fmaf_rn(nkx_4, x1_13, rq_5 * (g1_13 * __uint_as_float(ww[v_24 * 4 + e_13] & 4294901760u)));
                            }
                            int col_6 = (v_24 * 32 + tir) * 8;
                            unsigned int packed_6[4];
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_6[_lp*2 + 0], out_6[_lp*2+1 + 0]));
                                packed_6[_lp] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_6[_lp*2 + 2], out_6[_lp*2+1 + 2]));
                                packed_6[_lp + 1] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_6[_lp*2 + 4], out_6[_lp*2+1 + 4]));
                                packed_6[_lp + 2] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_6[_lp*2 + 6], out_6[_lp*2+1 + 6]));
                                packed_6[_lp + 3] = *(uint32_t*)&_bf2;
                            }
                            reinterpret_cast<int4*>(dx + (dx_base_6 + (unsigned long long)col_6))[0] = reinterpret_cast<int4*>(packed_6)[0];
                        }
                    }
                }
                if (n_iters > it_1 + 3) {
                    float dq_6 = 0.0f;
                    #pragma unroll
                    for (int v_25 = 0; v_25 < 1; v_25++) {
                        unsigned int wvd_7[4];
                        #pragma unroll
                        for (int e_14 = 0; e_14 < 4; e_14++) {
                            unsigned int gwd_14 = gwD[v_25 * 4 + e_14];
                            unsigned int xwd_14 = xwD[v_25 * 4 + e_14];
                            float g0_14 = __uint_as_float(gwd_14 << 16);
                            float g1_14 = __uint_as_float(gwd_14 & 4294901760u);
                            float x0_14 = __uint_as_float(xwd_14 << 16);
                            float x1_14 = __uint_as_float(xwd_14 & 4294901760u);
                            dq_6 = __fmaf_rn(g0_14 * __uint_as_float(ww[v_25 * 4 + e_14] << 16), x0_14, dq_6);
                            dq_6 = __fmaf_rn(g1_14 * __uint_as_float(ww[v_25 * 4 + e_14] & 4294901760u), x1_14, dq_6);
                        }
                    }
                    float dots_6[1];
                    float _warp_reduce_31 = dq_6;
                    #pragma unroll
                    for (int offset = 16; offset > 0; offset >>= 1)
                        _warp_reduce_31 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_31, offset);
                    dots_6[0] = _warp_reduce_31;
                    float rq_6 = rrD[0];
                    float cq_6 = __fdiv_rn(dots_6[0], 256.0f);
                    float kx_6 = rq_6 * rq_6 * rq_6 * cq_6;
                    float nkx_5 = -kx_6;
                    if (row5a + 12 < row_end) {
                        unsigned long long dx_base_7 = (unsigned long long)(row5a + 12) * 256;
                        #pragma unroll
                        for (int v_26 = 0; v_26 < 1; v_26++) {
                            float out_7[8];
                            unsigned int wv_7[4];
                            #pragma unroll
                            for (int e_15 = 0; e_15 < 4; e_15++) {
                                unsigned int gwd_15 = gwD[v_26 * 4 + e_15];
                                unsigned int xwd_15 = xwD[v_26 * 4 + e_15];
                                float g0_15 = __uint_as_float(gwd_15 << 16);
                                float g1_15 = __uint_as_float(gwd_15 & 4294901760u);
                                float x0_15 = __uint_as_float(xwd_15 << 16);
                                float x1_15 = __uint_as_float(xwd_15 & 4294901760u);
                                float y_7 = __fmaf_rn(g0_15 * x0_15, rq_6, -comp[v_26 * 8 + 2 * e_15]);
                                float t_7 = acc[v_26 * 8 + 2 * e_15] + y_7;
                                comp[v_26 * 8 + 2 * e_15] = t_7 - acc[v_26 * 8 + 2 * e_15] - y_7;
                                acc[v_26 * 8 + 2 * e_15] = t_7;
                                float y_7_1 = __fmaf_rn(g1_15 * x1_15, rq_6, -comp[v_26 * 8 + 2 * e_15 + 1]);
                                float t_7_1 = acc[v_26 * 8 + 2 * e_15 + 1] + y_7_1;
                                comp[v_26 * 8 + 2 * e_15 + 1] = t_7_1 - acc[v_26 * 8 + 2 * e_15 + 1] - y_7_1;
                                acc[v_26 * 8 + 2 * e_15 + 1] = t_7_1;
                                out_7[2 * e_15] = __fmaf_rn(nkx_5, x0_15, rq_6 * (g0_15 * __uint_as_float(ww[v_26 * 4 + e_15] << 16)));
                                out_7[2 * e_15 + 1] = __fmaf_rn(nkx_5, x1_15, rq_6 * (g1_15 * __uint_as_float(ww[v_26 * 4 + e_15] & 4294901760u)));
                            }
                            int col_7 = (v_26 * 32 + tir) * 8;
                            unsigned int packed_7[4];
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_7[_lp*2 + 0], out_7[_lp*2+1 + 0]));
                                packed_7[_lp] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_7[_lp*2 + 2], out_7[_lp*2+1 + 2]));
                                packed_7[_lp + 1] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_7[_lp*2 + 4], out_7[_lp*2+1 + 4]));
                                packed_7[_lp + 2] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_7[_lp*2 + 6], out_7[_lp*2+1 + 6]));
                                packed_7[_lp + 3] = *(uint32_t*)&_bf2;
                            }
                            reinterpret_cast<int4*>(dx + (dx_base_7 + (unsigned long long)col_7))[0] = reinterpret_cast<int4*>(packed_7)[0];
                        }
                    }
                }
            }
        }
        #pragma unroll
        for (int j2 = 0; j2 < 8; j2++) {
            acc[j2] = acc[j2] - comp[j2];
        }
        unsigned long long pbase = (unsigned long long)c * 256;
        #pragma unroll
        for (int v_27 = 0; v_27 < 1; v_27++) {
            int fcol = (v_27 * 32 + tir) * 8;
            #pragma unroll
            for (int e_16 = 0; e_16 < 8; e_16++) {
                row_fold[rg * 256 + fcol + e_16] = acc[v_27 * 8 + e_16];
            }
        }
        __syncthreads();
        float fold_acc[2];
        #pragma unroll
        for (int e_17 = 0; e_17 < 2; e_17++) {
            fold_acc[e_17] = 0.0f;
        }
        #pragma unroll
        for (int k_1 = 0; k_1 < 4; k_1++) {
            #pragma unroll
            for (int e_18 = 0; e_18 < 2; e_18++) {
                float d2 = row_fold[k_1 * 256 + tid * 2 + e_18];
                fold_acc[e_18] = fold_acc[e_18] + d2;
            }
        }
        int pcol = tid * 2;
        {
            float2 _v2 = make_float2(fold_acc[0 + 0], fold_acc[0 + 1]);
            *reinterpret_cast<float2*>(partial + pbase + (unsigned long long)pcol) = _v2;
        }
        if (tid == 0) {
            flags[claim_it & 1] = (int)nxt;
        }
        __syncthreads();
        c = flags[claim_it & 1];
    }
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    if (tid == 0) {
        if (c == n_chunks + (int)gridDim.x - 1) {
            *(reinterpret_cast<unsigned int*>(counters) + (0)) = 0;
        }
    }
}

} // extern "C"
