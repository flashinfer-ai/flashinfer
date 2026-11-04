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
#define SMEM_DOT_SLOTS_STAGE_BYTES 64
#define SMEM_DOT_SLOTS_STRIDE 64
#define SMEM_FLAGS_OFF 128
#define SMEM_FLAGS_STAGE_BYTES 256
#define SMEM_FLAGS_STRIDE 256
#define SMEM_RED_FOLD_OFF 384
#define SMEM_RED_FOLD_STAGE_BYTES 4096
#define SMEM_RED_FOLD_STRIDE 4096
#define SMEM_ROW_FOLD_OFF 4480
#define SMEM_ROW_FOLD_STAGE_BYTES 16384
#define SMEM_ROW_FOLD_STRIDE 16384
#define SMEM_COMP_SMEM_OFF 20864
#define SMEM_COMP_SMEM_STAGE_BYTES 128
#define SMEM_COMP_SMEM_STRIDE 128
#define SMEM_W_SM_OFF 20992
#define SMEM_W_SM_STAGE_BYTES 16
#define SMEM_W_SM_STRIDE 16
#define SMEM_PART_SM_OFF 21120
#define SMEM_PART_SM_STAGE_BYTES 16
#define SMEM_PART_SM_STRIDE 16
#define SMEM_TOTAL 21248
#define THREADS 256
#define LAUNCH_MIN_BLOCKS 2

#include <math_constants.h>

extern "C" {

__global__ __launch_bounds__(256, LAUNCH_MIN_BLOCKS) void
kernel_cake_rmsnorm_train_84efa2af7907877dfb56(__nv_bfloat16* __restrict__ g, __nv_bfloat16* __restrict__ x, __nv_bfloat16* __restrict__ w, float* __restrict__ r, __nv_bfloat16* __restrict__ dx, float* __restrict__ dw, float* __restrict__ partial, unsigned int* __restrict__ counters, __nv_bfloat16* __restrict__ g_h, int T, int n_chunks, int rows_per_chunk, long long g_stride_p0, long long x_stride_p0, long long g_h_stride_p0)
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
    float* red_fold = reinterpret_cast<float*>(smem_raw + 384);
    const int red_fold_addr = smem + 384;
    float* row_fold = reinterpret_cast<float*>(smem_raw + 4480);
    const int row_fold_addr = smem + 4480;
    float* comp_smem = reinterpret_cast<float*>(smem_raw + 20864);
    const int comp_smem_addr = smem + 20864;
    unsigned int* w_sm = reinterpret_cast<unsigned int*>(smem_raw + 20992);
    const int w_sm_addr = smem + 20992;
    float* part_sm = reinterpret_cast<float*>(smem_raw + 21120);
    const int part_sm_addr = smem + 21120;

    // === Task calls (declaration order) ===
    asm volatile("griddepcontrol.wait;" ::: "memory");
    unsigned int nxt = 0;
    if (tid == 0) {
        unsigned int _atomic_fetch_add_0 = atomicAdd(&counters[0], 1);
        nxt = _atomic_fetch_add_0;
    }
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    long long g_stride = g_stride_p0;
    long long x_stride = x_stride_p0;
    long long gh_stride = g_h_stride_p0;
    int c = 0;
    int rg = warp / 4;
    int tir = tid - rg * 128;
    int row_begin = 0;
    int row_end = 0;
    int n_rows = 0;
    int n_iters = 0;
    int last_row = 0;
    float wc[16];
    unsigned int ww[8];
    #pragma unroll
    for (int v = 0; v < 2; v++) {
        int wcolp = (v * 128 + tir) * 8;
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
        int stages = (T + 1) / 2;
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
        row_begin = begin_it * 2;
        row_end = row_begin + rows * 2;
        if (row_begin > T) {
            row_begin = T;
        }
        if (row_end > T) {
            row_end = T;
        }
        n_rows = row_end - row_begin;
        n_iters = (n_rows + 1) / 2;
        last_row = row_end - 1;
        #pragma unroll
        for (int j = 0; j < 16; j++) {
            acc[j] = 0.0f;
            comp[j] = 0.0f;
        }
        unsigned int gwA[8];
        unsigned int xwA[8];
        unsigned int ghwA[8];
        float rrA[1];
        unsigned int gwB[8];
        unsigned int xwB[8];
        unsigned int ghwB[8];
        float rrB[1];
        if (n_iters > 0) {
            int row0 = ((row_begin + rg) < (last_row) ? (row_begin + rg) : (last_row));
            unsigned long long g_base = (unsigned long long)row0 * (unsigned long long)g_stride;
            unsigned long long x_base = (unsigned long long)row0 * (unsigned long long)x_stride;
            rrA[0] = r[row0];
            #pragma unroll
            for (int v_1 = 0; v_1 < 2; v_1++) {
                int coln = (v_1 * 128 + tir) * 8;
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
            int n_full2 = (n_iters - 1) / 2 * 2;
            #pragma unroll 1
            for (int it = 0; it < n_full2; it += 2) {
                int rowa = row_begin + it * 2 + rg;
                int rowb = rowa + 2;
                int rowb_ld = ((rowb) < (last_row) ? (rowb) : (last_row));
                unsigned long long g_base_1 = (unsigned long long)rowb_ld * (unsigned long long)g_stride;
                unsigned long long x_base_0 = (unsigned long long)rowb_ld * (unsigned long long)x_stride;
                rrB[0] = r[rowb_ld];
                #pragma unroll
                for (int v_2 = 0; v_2 < 2; v_2++) {
                    int coln_1 = (v_2 * 128 + tir) * 8;
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
                float dq = 0.0f;
                #pragma unroll
                for (int v_3 = 0; v_3 < 2; v_3++) {
                    unsigned int wvd[4];
                    #pragma unroll
                    for (int e = 0; e < 4; e++) {
                        unsigned int gwd = gwA[v_3 * 4 + e];
                        unsigned int xwd = xwA[v_3 * 4 + e];
                        float g0 = __uint_as_float(gwd << 16);
                        float g1 = __uint_as_float(gwd & 4294901760u);
                        float x0 = __uint_as_float(xwd << 16);
                        float x1 = __uint_as_float(xwd & 4294901760u);
                        dq = __fmaf_rn(g0 * __uint_as_float(ww[v_3 * 4 + e] << 16), x0, dq);
                        dq = __fmaf_rn(g1 * __uint_as_float(ww[v_3 * 4 + e] & 4294901760u), x1, dq);
                    }
                }
                float dots[1];
                float _warp_reduce_7 = dq;
                #pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1)
                    _warp_reduce_7 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_7, offset);
                dots[0] = _warp_reduce_7;
                float rq = rrA[0];
                if (lane == 0) {
                    dot_slots[warp] = dots[0];
                }
                __syncthreads();
                float tot = 0.0f;
                #pragma unroll
                for (int k_1 = 0; k_1 < 4; k_1++) {
                    float d = dot_slots[rg * 4 + k_1];
                    tot += d;
                }
                dots[0] = tot;
                float cq = __fdiv_rn(dots[0], 2048.0f);
                float kx = rq * rq * rq * cq;
                float nkx = -kx;
                if (rowa < row_end) {
                    unsigned long long dx_base = (unsigned long long)rowa * 2048;
                    #pragma unroll
                    for (int v_4 = 0; v_4 < 2; v_4++) {
                        float out[8];
                        unsigned int wv[4];
                        #pragma unroll
                        for (int e_1 = 0; e_1 < 4; e_1++) {
                            unsigned int gwd_1 = gwA[v_4 * 4 + e_1];
                            unsigned int xwd_1 = xwA[v_4 * 4 + e_1];
                            float g0_1 = __uint_as_float(gwd_1 << 16);
                            float g1_1 = __uint_as_float(gwd_1 & 4294901760u);
                            float x0_1 = __uint_as_float(xwd_1 << 16);
                            float x1_1 = __uint_as_float(xwd_1 & 4294901760u);
                            float y = __fmaf_rn(g0_1 * x0_1, rq, -comp[v_4 * 8 + 2 * e_1]);
                            float t = acc[v_4 * 8 + 2 * e_1] + y;
                            comp[v_4 * 8 + 2 * e_1] = t - acc[v_4 * 8 + 2 * e_1] - y;
                            acc[v_4 * 8 + 2 * e_1] = t;
                            float y_0 = __fmaf_rn(g1_1 * x1_1, rq, -comp[v_4 * 8 + 2 * e_1 + 1]);
                            float t_0 = acc[v_4 * 8 + 2 * e_1 + 1] + y_0;
                            comp[v_4 * 8 + 2 * e_1 + 1] = t_0 - acc[v_4 * 8 + 2 * e_1 + 1] - y_0;
                            acc[v_4 * 8 + 2 * e_1 + 1] = t_0;
                            out[2 * e_1] = __fmaf_rn(nkx, x0_1, rq * (g0_1 * __uint_as_float(ww[v_4 * 4 + e_1] << 16)));
                            out[2 * e_1 + 1] = __fmaf_rn(nkx, x1_1, rq * (g1_1 * __uint_as_float(ww[v_4 * 4 + e_1] & 4294901760u)));
                        }
                        int col = (v_4 * 128 + tir) * 8;
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
                int rowa2_ld = ((rowa + 4) < (last_row) ? (rowa + 4) : (last_row));
                unsigned long long g_base_2 = (unsigned long long)rowa2_ld * (unsigned long long)g_stride;
                unsigned long long x_base_2 = (unsigned long long)rowa2_ld * (unsigned long long)x_stride;
                rrA[0] = r[rowa2_ld];
                #pragma unroll
                for (int v_5 = 0; v_5 < 2; v_5++) {
                    int coln_2 = (v_5 * 128 + tir) * 8;
                    {
                        const uint4* _vptr_8 = reinterpret_cast<const uint4*>(g + g_base_2 + (unsigned long long)coln_2);
                        uint4* _vdst_8 = reinterpret_cast<uint4*>(&gwA[v_5 * 4]);
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                : "=r"(_vdst_8[_blk].x), "=r"(_vdst_8[_blk].y), "=r"(_vdst_8[_blk].z), "=r"(_vdst_8[_blk].w) : "l"((const void*)(_vptr_8 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                        }
                    }
                    {
                        const uint4* _vptr_9 = reinterpret_cast<const uint4*>(x + x_base_2 + (unsigned long long)coln_2);
                        uint4* _vdst_9 = reinterpret_cast<uint4*>(&xwA[v_5 * 4]);
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                : "=r"(_vdst_9[_blk].x), "=r"(_vdst_9[_blk].y), "=r"(_vdst_9[_blk].z), "=r"(_vdst_9[_blk].w) : "l"((const void*)(_vptr_9 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                        }
                    }
                }
                float dq_1 = 0.0f;
                #pragma unroll
                for (int v_6 = 0; v_6 < 2; v_6++) {
                    unsigned int wvd_1[4];
                    #pragma unroll
                    for (int e_2 = 0; e_2 < 4; e_2++) {
                        unsigned int gwd_2 = gwB[v_6 * 4 + e_2];
                        unsigned int xwd_2 = xwB[v_6 * 4 + e_2];
                        float g0_2 = __uint_as_float(gwd_2 << 16);
                        float g1_2 = __uint_as_float(gwd_2 & 4294901760u);
                        float x0_2 = __uint_as_float(xwd_2 << 16);
                        float x1_2 = __uint_as_float(xwd_2 & 4294901760u);
                        dq_1 = __fmaf_rn(g0_2 * __uint_as_float(ww[v_6 * 4 + e_2] << 16), x0_2, dq_1);
                        dq_1 = __fmaf_rn(g1_2 * __uint_as_float(ww[v_6 * 4 + e_2] & 4294901760u), x1_2, dq_1);
                    }
                }
                float dots_0[1];
                float _warp_reduce_10 = dq_1;
                #pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1)
                    _warp_reduce_10 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_10, offset);
                dots_0[0] = _warp_reduce_10;
                float rq_0 = rrB[0];
                if (lane == 0) {
                    dot_slots[8 + warp] = dots_0[0];
                }
                __syncthreads();
                float tot_0 = 0.0f;
                #pragma unroll
                for (int k_2 = 0; k_2 < 4; k_2++) {
                    float d_1 = dot_slots[8 + rg * 4 + k_2];
                    tot_0 += d_1;
                }
                dots_0[0] = tot_0;
                float cq_0 = __fdiv_rn(dots_0[0], 2048.0f);
                float kx_0 = rq_0 * rq_0 * rq_0 * cq_0;
                float nkx_0 = -kx_0;
                if (rowb < row_end) {
                    unsigned long long dx_base_1 = (unsigned long long)rowb * 2048;
                    #pragma unroll
                    for (int v_7 = 0; v_7 < 2; v_7++) {
                        float out_1[8];
                        unsigned int wv_1[4];
                        #pragma unroll
                        for (int e_3 = 0; e_3 < 4; e_3++) {
                            unsigned int gwd_3 = gwB[v_7 * 4 + e_3];
                            unsigned int xwd_3 = xwB[v_7 * 4 + e_3];
                            float g0_3 = __uint_as_float(gwd_3 << 16);
                            float g1_3 = __uint_as_float(gwd_3 & 4294901760u);
                            float x0_3 = __uint_as_float(xwd_3 << 16);
                            float x1_3 = __uint_as_float(xwd_3 & 4294901760u);
                            float y_1 = __fmaf_rn(g0_3 * x0_3, rq_0, -comp[v_7 * 8 + 2 * e_3]);
                            float t_1 = acc[v_7 * 8 + 2 * e_3] + y_1;
                            comp[v_7 * 8 + 2 * e_3] = t_1 - acc[v_7 * 8 + 2 * e_3] - y_1;
                            acc[v_7 * 8 + 2 * e_3] = t_1;
                            float y_1_1 = __fmaf_rn(g1_3 * x1_3, rq_0, -comp[v_7 * 8 + 2 * e_3 + 1]);
                            float t_1_1 = acc[v_7 * 8 + 2 * e_3 + 1] + y_1_1;
                            comp[v_7 * 8 + 2 * e_3 + 1] = t_1_1 - acc[v_7 * 8 + 2 * e_3 + 1] - y_1_1;
                            acc[v_7 * 8 + 2 * e_3 + 1] = t_1_1;
                            out_1[2 * e_3] = __fmaf_rn(nkx_0, x0_3, rq_0 * (g0_3 * __uint_as_float(ww[v_7 * 4 + e_3] << 16)));
                            out_1[2 * e_3 + 1] = __fmaf_rn(nkx_0, x1_3, rq_0 * (g1_3 * __uint_as_float(ww[v_7 * 4 + e_3] & 4294901760u)));
                        }
                        int col_1 = (v_7 * 128 + tir) * 8;
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
            }
            #pragma unroll 1
            for (int it_1 = n_full2; it_1 < n_iters; it_1 += 2) {
                int rowa_1 = row_begin + it_1 * 2 + rg;
                int rowb_1 = rowa_1 + 2;
                if (n_iters > it_1 + 1) {
                    int rowb_ld_1 = ((rowb_1) < (last_row) ? (rowb_1) : (last_row));
                    unsigned long long g_base_3 = (unsigned long long)rowb_ld_1 * (unsigned long long)g_stride;
                    unsigned long long x_base_3 = (unsigned long long)rowb_ld_1 * (unsigned long long)x_stride;
                    rrB[0] = r[rowb_ld_1];
                    #pragma unroll
                    for (int v_8 = 0; v_8 < 2; v_8++) {
                        int coln_3 = (v_8 * 128 + tir) * 8;
                        {
                            const uint4* _vptr_11 = reinterpret_cast<const uint4*>(g + g_base_3 + (unsigned long long)coln_3);
                            uint4* _vdst_11 = reinterpret_cast<uint4*>(&gwB[v_8 * 4]);
                            #pragma unroll
                            for (int _blk = 0; _blk < 1; _blk++) {
                                asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                    : "=r"(_vdst_11[_blk].x), "=r"(_vdst_11[_blk].y), "=r"(_vdst_11[_blk].z), "=r"(_vdst_11[_blk].w) : "l"((const void*)(_vptr_11 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                            }
                        }
                        {
                            const uint4* _vptr_12 = reinterpret_cast<const uint4*>(x + x_base_3 + (unsigned long long)coln_3);
                            uint4* _vdst_12 = reinterpret_cast<uint4*>(&xwB[v_8 * 4]);
                            #pragma unroll
                            for (int _blk = 0; _blk < 1; _blk++) {
                                asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                    : "=r"(_vdst_12[_blk].x), "=r"(_vdst_12[_blk].y), "=r"(_vdst_12[_blk].z), "=r"(_vdst_12[_blk].w) : "l"((const void*)(_vptr_12 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                            }
                        }
                    }
                }
                float dq_2 = 0.0f;
                #pragma unroll
                for (int v_9 = 0; v_9 < 2; v_9++) {
                    unsigned int wvd_2[4];
                    #pragma unroll
                    for (int e_4 = 0; e_4 < 4; e_4++) {
                        unsigned int gwd_4 = gwA[v_9 * 4 + e_4];
                        unsigned int xwd_4 = xwA[v_9 * 4 + e_4];
                        float g0_4 = __uint_as_float(gwd_4 << 16);
                        float g1_4 = __uint_as_float(gwd_4 & 4294901760u);
                        float x0_4 = __uint_as_float(xwd_4 << 16);
                        float x1_4 = __uint_as_float(xwd_4 & 4294901760u);
                        dq_2 = __fmaf_rn(g0_4 * __uint_as_float(ww[v_9 * 4 + e_4] << 16), x0_4, dq_2);
                        dq_2 = __fmaf_rn(g1_4 * __uint_as_float(ww[v_9 * 4 + e_4] & 4294901760u), x1_4, dq_2);
                    }
                }
                float dots_1[1];
                float _warp_reduce_13 = dq_2;
                #pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1)
                    _warp_reduce_13 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_13, offset);
                dots_1[0] = _warp_reduce_13;
                float rq_1 = rrA[0];
                if (lane == 0) {
                    dot_slots[warp] = dots_1[0];
                }
                __syncthreads();
                float tot_1 = 0.0f;
                #pragma unroll
                for (int k_3 = 0; k_3 < 4; k_3++) {
                    float d_2 = dot_slots[rg * 4 + k_3];
                    tot_1 += d_2;
                }
                dots_1[0] = tot_1;
                float cq_1 = __fdiv_rn(dots_1[0], 2048.0f);
                float kx_1 = rq_1 * rq_1 * rq_1 * cq_1;
                float nkx_1 = -kx_1;
                if (rowa_1 < row_end) {
                    unsigned long long dx_base_2 = (unsigned long long)rowa_1 * 2048;
                    #pragma unroll
                    for (int v_10 = 0; v_10 < 2; v_10++) {
                        float out_2[8];
                        unsigned int wv_2[4];
                        #pragma unroll
                        for (int e_5 = 0; e_5 < 4; e_5++) {
                            unsigned int gwd_5 = gwA[v_10 * 4 + e_5];
                            unsigned int xwd_5 = xwA[v_10 * 4 + e_5];
                            float g0_5 = __uint_as_float(gwd_5 << 16);
                            float g1_5 = __uint_as_float(gwd_5 & 4294901760u);
                            float x0_5 = __uint_as_float(xwd_5 << 16);
                            float x1_5 = __uint_as_float(xwd_5 & 4294901760u);
                            float y_2 = __fmaf_rn(g0_5 * x0_5, rq_1, -comp[v_10 * 8 + 2 * e_5]);
                            float t_2 = acc[v_10 * 8 + 2 * e_5] + y_2;
                            comp[v_10 * 8 + 2 * e_5] = t_2 - acc[v_10 * 8 + 2 * e_5] - y_2;
                            acc[v_10 * 8 + 2 * e_5] = t_2;
                            float y_2_1 = __fmaf_rn(g1_5 * x1_5, rq_1, -comp[v_10 * 8 + 2 * e_5 + 1]);
                            float t_2_1 = acc[v_10 * 8 + 2 * e_5 + 1] + y_2_1;
                            comp[v_10 * 8 + 2 * e_5 + 1] = t_2_1 - acc[v_10 * 8 + 2 * e_5 + 1] - y_2_1;
                            acc[v_10 * 8 + 2 * e_5 + 1] = t_2_1;
                            out_2[2 * e_5] = __fmaf_rn(nkx_1, x0_5, rq_1 * (g0_5 * __uint_as_float(ww[v_10 * 4 + e_5] << 16)));
                            out_2[2 * e_5 + 1] = __fmaf_rn(nkx_1, x1_5, rq_1 * (g1_5 * __uint_as_float(ww[v_10 * 4 + e_5] & 4294901760u)));
                        }
                        int col_2 = (v_10 * 128 + tir) * 8;
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
                if (n_iters > it_1 + 1) {
                    if (n_iters > it_1 + 2) {
                        int rowa2_ld_1 = ((rowa_1 + 4) < (last_row) ? (rowa_1 + 4) : (last_row));
                        unsigned long long g_base_0 = (unsigned long long)rowa2_ld_1 * (unsigned long long)g_stride;
                        unsigned long long x_base_1 = (unsigned long long)rowa2_ld_1 * (unsigned long long)x_stride;
                        rrA[0] = r[rowa2_ld_1];
                        #pragma unroll
                        for (int v_11 = 0; v_11 < 2; v_11++) {
                            int coln_4 = (v_11 * 128 + tir) * 8;
                            {
                                const uint4* _vptr_14 = reinterpret_cast<const uint4*>(g + g_base_0 + (unsigned long long)coln_4);
                                uint4* _vdst_14 = reinterpret_cast<uint4*>(&gwA[v_11 * 4]);
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
                                    asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                        : "=r"(_vdst_14[_blk].x), "=r"(_vdst_14[_blk].y), "=r"(_vdst_14[_blk].z), "=r"(_vdst_14[_blk].w) : "l"((const void*)(_vptr_14 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                                }
                            }
                            {
                                const uint4* _vptr_15 = reinterpret_cast<const uint4*>(x + x_base_1 + (unsigned long long)coln_4);
                                uint4* _vdst_15 = reinterpret_cast<uint4*>(&xwA[v_11 * 4]);
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
                                    asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                        : "=r"(_vdst_15[_blk].x), "=r"(_vdst_15[_blk].y), "=r"(_vdst_15[_blk].z), "=r"(_vdst_15[_blk].w) : "l"((const void*)(_vptr_15 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                                }
                            }
                        }
                    }
                    float dq_0 = 0.0f;
                    #pragma unroll
                    for (int v_12 = 0; v_12 < 2; v_12++) {
                        unsigned int wvd_3[4];
                        #pragma unroll
                        for (int e_6 = 0; e_6 < 4; e_6++) {
                            unsigned int gwd_6 = gwB[v_12 * 4 + e_6];
                            unsigned int xwd_6 = xwB[v_12 * 4 + e_6];
                            float g0_6 = __uint_as_float(gwd_6 << 16);
                            float g1_6 = __uint_as_float(gwd_6 & 4294901760u);
                            float x0_6 = __uint_as_float(xwd_6 << 16);
                            float x1_6 = __uint_as_float(xwd_6 & 4294901760u);
                            dq_0 = __fmaf_rn(g0_6 * __uint_as_float(ww[v_12 * 4 + e_6] << 16), x0_6, dq_0);
                            dq_0 = __fmaf_rn(g1_6 * __uint_as_float(ww[v_12 * 4 + e_6] & 4294901760u), x1_6, dq_0);
                        }
                    }
                    float dots_1_1[1];
                    float _warp_reduce_16 = dq_0;
                    #pragma unroll
                    for (int offset = 16; offset > 0; offset >>= 1)
                        _warp_reduce_16 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_16, offset);
                    dots_1_1[0] = _warp_reduce_16;
                    float rq_2 = rrB[0];
                    if (lane == 0) {
                        dot_slots[8 + warp] = dots_1_1[0];
                    }
                    __syncthreads();
                    float tot_3 = 0.0f;
                    #pragma unroll
                    for (int k_4 = 0; k_4 < 4; k_4++) {
                        float d_3 = dot_slots[8 + rg * 4 + k_4];
                        tot_3 += d_3;
                    }
                    dots_1_1[0] = tot_3;
                    float cq_4 = __fdiv_rn(dots_1_1[0], 2048.0f);
                    float kx_5 = rq_2 * rq_2 * rq_2 * cq_4;
                    float nkx_6 = -kx_5;
                    if (rowb_1 < row_end) {
                        unsigned long long dx_base_3 = (unsigned long long)rowb_1 * 2048;
                        #pragma unroll
                        for (int v_13 = 0; v_13 < 2; v_13++) {
                            float out_3[8];
                            unsigned int wv_3[4];
                            #pragma unroll
                            for (int e_7 = 0; e_7 < 4; e_7++) {
                                unsigned int gwd_7 = gwB[v_13 * 4 + e_7];
                                unsigned int xwd_7 = xwB[v_13 * 4 + e_7];
                                float g0_7 = __uint_as_float(gwd_7 << 16);
                                float g1_7 = __uint_as_float(gwd_7 & 4294901760u);
                                float x0_7 = __uint_as_float(xwd_7 << 16);
                                float x1_7 = __uint_as_float(xwd_7 & 4294901760u);
                                float y_3 = __fmaf_rn(g0_7 * x0_7, rq_2, -comp[v_13 * 8 + 2 * e_7]);
                                float t_3 = acc[v_13 * 8 + 2 * e_7] + y_3;
                                comp[v_13 * 8 + 2 * e_7] = t_3 - acc[v_13 * 8 + 2 * e_7] - y_3;
                                acc[v_13 * 8 + 2 * e_7] = t_3;
                                float y_3_1 = __fmaf_rn(g1_7 * x1_7, rq_2, -comp[v_13 * 8 + 2 * e_7 + 1]);
                                float t_3_1 = acc[v_13 * 8 + 2 * e_7 + 1] + y_3_1;
                                comp[v_13 * 8 + 2 * e_7 + 1] = t_3_1 - acc[v_13 * 8 + 2 * e_7 + 1] - y_3_1;
                                acc[v_13 * 8 + 2 * e_7 + 1] = t_3_1;
                                out_3[2 * e_7] = __fmaf_rn(nkx_6, x0_7, rq_2 * (g0_7 * __uint_as_float(ww[v_13 * 4 + e_7] << 16)));
                                out_3[2 * e_7 + 1] = __fmaf_rn(nkx_6, x1_7, rq_2 * (g1_7 * __uint_as_float(ww[v_13 * 4 + e_7] & 4294901760u)));
                            }
                            int col_3 = (v_13 * 128 + tir) * 8;
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
            }
        }
        #pragma unroll
        for (int j2 = 0; j2 < 16; j2++) {
            acc[j2] = acc[j2] - comp[j2];
        }
        unsigned long long pbase = (unsigned long long)c * 2048;
        #pragma unroll
        for (int v_14 = 0; v_14 < 2; v_14++) {
            int fcol = (v_14 * 128 + tir) * 8;
            #pragma unroll
            for (int e_8 = 0; e_8 < 8; e_8++) {
                row_fold[rg * 2048 + fcol + e_8] = acc[v_14 * 8 + e_8];
            }
        }
        __syncthreads();
        float fold_acc[8];
        #pragma unroll
        for (int e_9 = 0; e_9 < 8; e_9++) {
            fold_acc[e_9] = 0.0f;
        }
        #pragma unroll
        for (int k_5 = 0; k_5 < 2; k_5++) {
            #pragma unroll
            for (int e_10 = 0; e_10 < 8; e_10++) {
                float d2 = row_fold[k_5 * 2048 + tid * 8 + e_10];
                fold_acc[e_10] = fold_acc[e_10] + d2;
            }
        }
        int pcol = tid * 8;
        {
            unsigned _stv8_17_0 = __float_as_uint(fold_acc[0 + 0]);
            unsigned _stv8_17_1 = __float_as_uint(fold_acc[0 + 1]);
            unsigned _stv8_17_2 = __float_as_uint(fold_acc[0 + 2]);
            unsigned _stv8_17_3 = __float_as_uint(fold_acc[0 + 3]);
            unsigned _stv8_17_4 = __float_as_uint(fold_acc[0 + 4]);
            unsigned _stv8_17_5 = __float_as_uint(fold_acc[0 + 5]);
            unsigned _stv8_17_6 = __float_as_uint(fold_acc[0 + 6]);
            unsigned _stv8_17_7 = __float_as_uint(fold_acc[0 + 7]);
            asm volatile(
                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                :: "l"((void*)(partial + (pbase + (unsigned long long)pcol))), "r"(_stv8_17_0), "r"(_stv8_17_1), "r"(_stv8_17_2), "r"(_stv8_17_3), "r"(_stv8_17_4), "r"(_stv8_17_5), "r"(_stv8_17_6), "r"(_stv8_17_7) : "memory");
        }
        if (tid == 0) {
            flags[claim_it & 1] = (int)nxt;
        }
        __syncthreads();
        c = flags[claim_it & 1];
    }
    if (tid == 0) {
        if (c == n_chunks + (int)gridDim.x - 1) {
            *(reinterpret_cast<unsigned int*>(counters) + (0)) = 0;
        }
    }
}

} // extern "C"
