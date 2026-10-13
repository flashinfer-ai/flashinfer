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
#define SMEM_DOT_SLOTS_OFF 0
#define SMEM_DOT_SLOTS_STAGE_BYTES 64
#define SMEM_DOT_SLOTS_STRIDE 64
#define SMEM_FLAGS_OFF 128
#define SMEM_FLAGS_STAGE_BYTES 64
#define SMEM_FLAGS_STRIDE 64
#define SMEM_RED_FOLD_OFF 256
#define SMEM_RED_FOLD_STAGE_BYTES 4096
#define SMEM_RED_FOLD_STRIDE 4096
#define SMEM_ROW_FOLD_OFF 4352
#define SMEM_ROW_FOLD_STAGE_BYTES 16384
#define SMEM_ROW_FOLD_STRIDE 16384
#define SMEM_COMP_SMEM_OFF 20736
#define SMEM_COMP_SMEM_STAGE_BYTES 128
#define SMEM_COMP_SMEM_STRIDE 128
#define SMEM_W_SM_OFF 20864
#define SMEM_W_SM_STAGE_BYTES 16
#define SMEM_W_SM_STRIDE 16
#define SMEM_PART_SM_OFF 20992
#define SMEM_PART_SM_STAGE_BYTES 16
#define SMEM_PART_SM_STRIDE 16
#define SMEM_TOTAL 21120
#define THREADS 256

#include <math_constants.h>
#include <cooperative_groups.h>

extern "C" {

__global__ __launch_bounds__(256, 2) void
kernel_cake_rmsnorm_train_01036297dc6e647b9e56(__nv_bfloat16* __restrict__ g, __nv_bfloat16* __restrict__ x, __nv_bfloat16* __restrict__ w, float* __restrict__ r, __nv_bfloat16* __restrict__ dx, float* __restrict__ dw, float* __restrict__ partial, unsigned int* __restrict__ counters, __nv_bfloat16* __restrict__ g_h, int T, long long g_stride, long long x_stride, long long gh_stride, int n_chunks, int rows_per_chunk)
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
    float* row_fold = reinterpret_cast<float*>(smem_raw + 4352);
    const int row_fold_addr = smem + 4352;
    float* comp_smem = reinterpret_cast<float*>(smem_raw + 20736);
    const int comp_smem_addr = smem + 20736;
    unsigned int* w_sm = reinterpret_cast<unsigned int*>(smem_raw + 20864);
    const int w_sm_addr = smem + 20864;
    float* part_sm = reinterpret_cast<float*>(smem_raw + 20992);
    const int part_sm_addr = smem + 20992;

    // === Task calls (dependency order) ===
    asm volatile("griddepcontrol.wait;" ::: "memory");
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    int c = bid;
    int rg = warp;
    int tir = tid - rg * 32;
    int row_begin = c * rows_per_chunk;
    int row_end = row_begin + rows_per_chunk;
    if (row_end > T) {
        row_end = T;
    }
    if (row_begin > T) {
        row_begin = T;
    }
    int n_rows = row_end - row_begin;
    int n_iters = (n_rows + 7) / 8;
    int last_row = row_end - 1;
    float wc[16];
    unsigned int ww[8];
    #pragma unroll
    for (int v = 0; v < 2; v++) {
        int wcolp = (v * 32 + tir) * 8;
        {
            const uint4* _vptr_0 = reinterpret_cast<const uint4*>(w + (unsigned long long)wcolp);
            uint4* _vdst_0 = reinterpret_cast<uint4*>(&ww[v * 4]);
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vdst_0[_blk] = _vptr_0[_blk];
            }
        }
    }
    float acc[16];
    float comp[16];
    unsigned int compw[8];
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
        int _min_0 = ((row_begin + rg) < (last_row) ? (row_begin + rg) : (last_row));
        int row0 = _min_0;
        unsigned long long g_base = (unsigned long long)row0 * (unsigned long long)g_stride;
        unsigned long long x_base = (unsigned long long)row0 * (unsigned long long)x_stride;
        rrA[0] = r[row0];
        #pragma unroll
        for (int v_1 = 0; v_1 < 2; v_1++) {
            int coln = (v_1 * 32 + tir) * 8;
            {
                const uint4* _vptr_1 = reinterpret_cast<const uint4*>(g + g_base + (unsigned long long)coln);
                uint4* _vdst_1 = reinterpret_cast<uint4*>(&gwA[v_1 * 4]);
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                        : "=r"(_vdst_1[_blk].x), "=r"(_vdst_1[_blk].y), "=r"(_vdst_1[_blk].z), "=r"(_vdst_1[_blk].w) : "l"((const void*)(_vptr_1 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                }
            }
            {
                const uint4* _vptr_2 = reinterpret_cast<const uint4*>(x + x_base + (unsigned long long)coln);
                uint4* _vdst_2 = reinterpret_cast<uint4*>(&xwA[v_1 * 4]);
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                        : "=r"(_vdst_2[_blk].x), "=r"(_vdst_2[_blk].y), "=r"(_vdst_2[_blk].z), "=r"(_vdst_2[_blk].w) : "l"((const void*)(_vptr_2 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                }
            }
        }
        int n_full2 = (n_iters - 1) / 2 * 2;
        #pragma unroll 1
        for (int it = 0; it < n_full2; it += 2) {
            int rowa = row_begin + it * 8 + rg;
            int rowb = rowa + 8;
            int _min_1 = ((rowb) < (last_row) ? (rowb) : (last_row));
            int rowb_ld = _min_1;
            unsigned long long g_base_0 = (unsigned long long)rowb_ld * (unsigned long long)g_stride;
            unsigned long long x_base_1 = (unsigned long long)rowb_ld * (unsigned long long)x_stride;
            rrB[0] = r[rowb_ld];
            #pragma unroll
            for (int v_2 = 0; v_2 < 2; v_2++) {
                int coln_1 = (v_2 * 32 + tir) * 8;
                {
                    const uint4* _vptr_3 = reinterpret_cast<const uint4*>(g + g_base_0 + (unsigned long long)coln_1);
                    uint4* _vdst_3 = reinterpret_cast<uint4*>(&gwB[v_2 * 4]);
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                            : "=r"(_vdst_3[_blk].x), "=r"(_vdst_3[_blk].y), "=r"(_vdst_3[_blk].z), "=r"(_vdst_3[_blk].w) : "l"((const void*)(_vptr_3 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                    }
                }
                {
                    const uint4* _vptr_4 = reinterpret_cast<const uint4*>(x + x_base_1 + (unsigned long long)coln_1);
                    uint4* _vdst_4 = reinterpret_cast<uint4*>(&xwB[v_2 * 4]);
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                            : "=r"(_vdst_4[_blk].x), "=r"(_vdst_4[_blk].y), "=r"(_vdst_4[_blk].z), "=r"(_vdst_4[_blk].w) : "l"((const void*)(_vptr_4 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
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
                    float _fma_0 = __fmaf_rn(g0 * __uint_as_float(ww[v_3 * 4 + e] << 16), x0, dq);
                    dq = _fma_0;
                    float _fma_1 = __fmaf_rn(g1 * __uint_as_float(ww[v_3 * 4 + e] & 4294901760u), x1, dq);
                    dq = _fma_1;
                }
            }
            float dots[1];
            float _warp_reduce_0 = dq;
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1)
                _warp_reduce_0 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_0, offset);
            dots[0] = _warp_reduce_0;
            float rq = rrA[0];
            float _fdiv_rn_0 = __fdiv_rn(dots[0], 512.0f);
            float cq = _fdiv_rn_0;
            float kx = rq * rq * rq * cq;
            float nkx = -kx;
            if (rowa < row_end) {
                unsigned long long dx_base = (unsigned long long)rowa * 512;
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
                        float _fma_2 = __fmaf_rn(g0_1 * x0_1, rq, -comp[v_4 * 8 + 2 * e_1]);
                        float y = _fma_2;
                        float t = acc[v_4 * 8 + 2 * e_1] + y;
                        comp[v_4 * 8 + 2 * e_1] = t - acc[v_4 * 8 + 2 * e_1] - y;
                        acc[v_4 * 8 + 2 * e_1] = t;
                        float _fma_3 = __fmaf_rn(g1_1 * x1_1, rq, -comp[v_4 * 8 + 2 * e_1 + 1]);
                        float y_0 = _fma_3;
                        float t_1 = acc[v_4 * 8 + 2 * e_1 + 1] + y_0;
                        comp[v_4 * 8 + 2 * e_1 + 1] = t_1 - acc[v_4 * 8 + 2 * e_1 + 1] - y_0;
                        acc[v_4 * 8 + 2 * e_1 + 1] = t_1;
                        float _fma_4 = __fmaf_rn(nkx, x0_1, rq * (g0_1 * __uint_as_float(ww[v_4 * 4 + e_1] << 16)));
                        out[2 * e_1] = _fma_4;
                        float _fma_5 = __fmaf_rn(nkx, x1_1, rq * (g1_1 * __uint_as_float(ww[v_4 * 4 + e_1] & 4294901760u)));
                        out[2 * e_1 + 1] = _fma_5;
                    }
                    int col = (v_4 * 32 + tir) * 8;
                    {
                        __nv_bfloat162 _pk[4];
                        _pk[0] = __floats2bfloat162_rn(out[0 + 0], out[0 + 1]);
                        _pk[1] = __floats2bfloat162_rn(out[0 + 2], out[0 + 3]);
                        _pk[2] = __floats2bfloat162_rn(out[0 + 4], out[0 + 5]);
                        _pk[3] = __floats2bfloat162_rn(out[0 + 6], out[0 + 7]);
                        *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(dx))[dx_base + (unsigned long long)col + 0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                    }
                }
            }
            int _min_2 = ((rowa + 16) < (last_row) ? (rowa + 16) : (last_row));
            int rowa2_ld = _min_2;
            unsigned long long g_base_2 = (unsigned long long)rowa2_ld * (unsigned long long)g_stride;
            unsigned long long x_base_3 = (unsigned long long)rowa2_ld * (unsigned long long)x_stride;
            rrA[0] = r[rowa2_ld];
            #pragma unroll
            for (int v_5 = 0; v_5 < 2; v_5++) {
                int coln_2 = (v_5 * 32 + tir) * 8;
                {
                    const uint4* _vptr_5 = reinterpret_cast<const uint4*>(g + g_base_2 + (unsigned long long)coln_2);
                    uint4* _vdst_5 = reinterpret_cast<uint4*>(&gwA[v_5 * 4]);
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                            : "=r"(_vdst_5[_blk].x), "=r"(_vdst_5[_blk].y), "=r"(_vdst_5[_blk].z), "=r"(_vdst_5[_blk].w) : "l"((const void*)(_vptr_5 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                    }
                }
                {
                    const uint4* _vptr_6 = reinterpret_cast<const uint4*>(x + x_base_3 + (unsigned long long)coln_2);
                    uint4* _vdst_6 = reinterpret_cast<uint4*>(&xwA[v_5 * 4]);
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                            : "=r"(_vdst_6[_blk].x), "=r"(_vdst_6[_blk].y), "=r"(_vdst_6[_blk].z), "=r"(_vdst_6[_blk].w) : "l"((const void*)(_vptr_6 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                    }
                }
            }
            float dq_4 = 0.0f;
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
                    float _fma_6 = __fmaf_rn(g0_2 * __uint_as_float(ww[v_6 * 4 + e_2] << 16), x0_2, dq_4);
                    dq_4 = _fma_6;
                    float _fma_7 = __fmaf_rn(g1_2 * __uint_as_float(ww[v_6 * 4 + e_2] & 4294901760u), x1_2, dq_4);
                    dq_4 = _fma_7;
                }
            }
            float dots_5[1];
            float _warp_reduce_1 = dq_4;
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1)
                _warp_reduce_1 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_1, offset);
            dots_5[0] = _warp_reduce_1;
            float rq_6 = rrB[0];
            float _fdiv_rn_1 = __fdiv_rn(dots_5[0], 512.0f);
            float cq_7 = _fdiv_rn_1;
            float kx_8 = rq_6 * rq_6 * rq_6 * cq_7;
            float nkx_9 = -kx_8;
            if (rowb < row_end) {
                unsigned long long dx_base_1 = (unsigned long long)rowb * 512;
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
                        float _fma_8 = __fmaf_rn(g0_3 * x0_3, rq_6, -comp[v_7 * 8 + 2 * e_3]);
                        float y_1 = _fma_8;
                        float t_2 = acc[v_7 * 8 + 2 * e_3] + y_1;
                        comp[v_7 * 8 + 2 * e_3] = t_2 - acc[v_7 * 8 + 2 * e_3] - y_1;
                        acc[v_7 * 8 + 2 * e_3] = t_2;
                        float _fma_9 = __fmaf_rn(g1_3 * x1_3, rq_6, -comp[v_7 * 8 + 2 * e_3 + 1]);
                        float y_0_1 = _fma_9;
                        float t_1_1 = acc[v_7 * 8 + 2 * e_3 + 1] + y_0_1;
                        comp[v_7 * 8 + 2 * e_3 + 1] = t_1_1 - acc[v_7 * 8 + 2 * e_3 + 1] - y_0_1;
                        acc[v_7 * 8 + 2 * e_3 + 1] = t_1_1;
                        float _fma_10 = __fmaf_rn(nkx_9, x0_3, rq_6 * (g0_3 * __uint_as_float(ww[v_7 * 4 + e_3] << 16)));
                        out_1[2 * e_3] = _fma_10;
                        float _fma_11 = __fmaf_rn(nkx_9, x1_3, rq_6 * (g1_3 * __uint_as_float(ww[v_7 * 4 + e_3] & 4294901760u)));
                        out_1[2 * e_3 + 1] = _fma_11;
                    }
                    int col_1 = (v_7 * 32 + tir) * 8;
                    {
                        __nv_bfloat162 _pk[4];
                        _pk[0] = __floats2bfloat162_rn(out_1[0 + 0], out_1[0 + 1]);
                        _pk[1] = __floats2bfloat162_rn(out_1[0 + 2], out_1[0 + 3]);
                        _pk[2] = __floats2bfloat162_rn(out_1[0 + 4], out_1[0 + 5]);
                        _pk[3] = __floats2bfloat162_rn(out_1[0 + 6], out_1[0 + 7]);
                        *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(dx))[dx_base_1 + (unsigned long long)col_1 + 0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                    }
                }
            }
        }
        #pragma unroll 1
        for (int it_1 = n_full2; it_1 < n_iters; it_1 += 2) {
            int rowa_1 = row_begin + it_1 * 8 + rg;
            int rowb_1 = rowa_1 + 8;
            if (n_iters > it_1 + 1) {
                int _min_3 = ((rowb_1) < (last_row) ? (rowb_1) : (last_row));
                int rowb_ld_1 = _min_3;
                unsigned long long g_base_0_1 = (unsigned long long)rowb_ld_1 * (unsigned long long)g_stride;
                unsigned long long x_base_1_1 = (unsigned long long)rowb_ld_1 * (unsigned long long)x_stride;
                rrB[0] = r[rowb_ld_1];
                #pragma unroll
                for (int v_8 = 0; v_8 < 2; v_8++) {
                    int coln_3 = (v_8 * 32 + tir) * 8;
                    {
                        const uint4* _vptr_7 = reinterpret_cast<const uint4*>(g + g_base_0_1 + (unsigned long long)coln_3);
                        uint4* _vdst_7 = reinterpret_cast<uint4*>(&gwB[v_8 * 4]);
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                : "=r"(_vdst_7[_blk].x), "=r"(_vdst_7[_blk].y), "=r"(_vdst_7[_blk].z), "=r"(_vdst_7[_blk].w) : "l"((const void*)(_vptr_7 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                        }
                    }
                    {
                        const uint4* _vptr_8 = reinterpret_cast<const uint4*>(x + x_base_1_1 + (unsigned long long)coln_3);
                        uint4* _vdst_8 = reinterpret_cast<uint4*>(&xwB[v_8 * 4]);
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                : "=r"(_vdst_8[_blk].x), "=r"(_vdst_8[_blk].y), "=r"(_vdst_8[_blk].z), "=r"(_vdst_8[_blk].w) : "l"((const void*)(_vptr_8 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                        }
                    }
                }
            }
            float dq_1 = 0.0f;
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
                    float _fma_12 = __fmaf_rn(g0_4 * __uint_as_float(ww[v_9 * 4 + e_4] << 16), x0_4, dq_1);
                    dq_1 = _fma_12;
                    float _fma_13 = __fmaf_rn(g1_4 * __uint_as_float(ww[v_9 * 4 + e_4] & 4294901760u), x1_4, dq_1);
                    dq_1 = _fma_13;
                }
            }
            float dots_1[1];
            float _warp_reduce_2 = dq_1;
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1)
                _warp_reduce_2 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_2, offset);
            dots_1[0] = _warp_reduce_2;
            float rq_1 = rrA[0];
            float _fdiv_rn_2 = __fdiv_rn(dots_1[0], 512.0f);
            float cq_1 = _fdiv_rn_2;
            float kx_1 = rq_1 * rq_1 * rq_1 * cq_1;
            float nkx_1 = -kx_1;
            if (rowa_1 < row_end) {
                unsigned long long dx_base_2 = (unsigned long long)rowa_1 * 512;
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
                        float _fma_14 = __fmaf_rn(g0_5 * x0_5, rq_1, -comp[v_10 * 8 + 2 * e_5]);
                        float y_2 = _fma_14;
                        float t_3 = acc[v_10 * 8 + 2 * e_5] + y_2;
                        comp[v_10 * 8 + 2 * e_5] = t_3 - acc[v_10 * 8 + 2 * e_5] - y_2;
                        acc[v_10 * 8 + 2 * e_5] = t_3;
                        float _fma_15 = __fmaf_rn(g1_5 * x1_5, rq_1, -comp[v_10 * 8 + 2 * e_5 + 1]);
                        float y_0_2 = _fma_15;
                        float t_1_2 = acc[v_10 * 8 + 2 * e_5 + 1] + y_0_2;
                        comp[v_10 * 8 + 2 * e_5 + 1] = t_1_2 - acc[v_10 * 8 + 2 * e_5 + 1] - y_0_2;
                        acc[v_10 * 8 + 2 * e_5 + 1] = t_1_2;
                        float _fma_16 = __fmaf_rn(nkx_1, x0_5, rq_1 * (g0_5 * __uint_as_float(ww[v_10 * 4 + e_5] << 16)));
                        out_2[2 * e_5] = _fma_16;
                        float _fma_17 = __fmaf_rn(nkx_1, x1_5, rq_1 * (g1_5 * __uint_as_float(ww[v_10 * 4 + e_5] & 4294901760u)));
                        out_2[2 * e_5 + 1] = _fma_17;
                    }
                    int col_2 = (v_10 * 32 + tir) * 8;
                    {
                        __nv_bfloat162 _pk[4];
                        _pk[0] = __floats2bfloat162_rn(out_2[0 + 0], out_2[0 + 1]);
                        _pk[1] = __floats2bfloat162_rn(out_2[0 + 2], out_2[0 + 3]);
                        _pk[2] = __floats2bfloat162_rn(out_2[0 + 4], out_2[0 + 5]);
                        _pk[3] = __floats2bfloat162_rn(out_2[0 + 6], out_2[0 + 7]);
                        *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(dx))[dx_base_2 + (unsigned long long)col_2 + 0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                    }
                }
            }
            if (n_iters > it_1 + 1) {
                if (n_iters > it_1 + 2) {
                    int _min_4 = ((rowa_1 + 16) < (last_row) ? (rowa_1 + 16) : (last_row));
                    int rowa2_ld_1 = _min_4;
                    unsigned long long g_base_0_2 = (unsigned long long)rowa2_ld_1 * (unsigned long long)g_stride;
                    unsigned long long x_base_1_2 = (unsigned long long)rowa2_ld_1 * (unsigned long long)x_stride;
                    rrA[0] = r[rowa2_ld_1];
                    #pragma unroll
                    for (int v_11 = 0; v_11 < 2; v_11++) {
                        int coln_4 = (v_11 * 32 + tir) * 8;
                        {
                            const uint4* _vptr_9 = reinterpret_cast<const uint4*>(g + g_base_0_2 + (unsigned long long)coln_4);
                            uint4* _vdst_9 = reinterpret_cast<uint4*>(&gwA[v_11 * 4]);
                            #pragma unroll
                            for (int _blk = 0; _blk < 1; _blk++) {
                                asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                    : "=r"(_vdst_9[_blk].x), "=r"(_vdst_9[_blk].y), "=r"(_vdst_9[_blk].z), "=r"(_vdst_9[_blk].w) : "l"((const void*)(_vptr_9 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                            }
                        }
                        {
                            const uint4* _vptr_10 = reinterpret_cast<const uint4*>(x + x_base_1_2 + (unsigned long long)coln_4);
                            uint4* _vdst_10 = reinterpret_cast<uint4*>(&xwA[v_11 * 4]);
                            #pragma unroll
                            for (int _blk = 0; _blk < 1; _blk++) {
                                asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                    : "=r"(_vdst_10[_blk].x), "=r"(_vdst_10[_blk].y), "=r"(_vdst_10[_blk].z), "=r"(_vdst_10[_blk].w) : "l"((const void*)(_vptr_10 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
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
                        float _fma_18 = __fmaf_rn(g0_6 * __uint_as_float(ww[v_12 * 4 + e_6] << 16), x0_6, dq_0);
                        dq_0 = _fma_18;
                        float _fma_19 = __fmaf_rn(g1_6 * __uint_as_float(ww[v_12 * 4 + e_6] & 4294901760u), x1_6, dq_0);
                        dq_0 = _fma_19;
                    }
                }
                float dots_1_1[1];
                float _warp_reduce_3 = dq_0;
                #pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1)
                    _warp_reduce_3 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_3, offset);
                dots_1_1[0] = _warp_reduce_3;
                float rq_2 = rrB[0];
                float _fdiv_rn_3 = __fdiv_rn(dots_1_1[0], 512.0f);
                float cq_3 = _fdiv_rn_3;
                float kx_4 = rq_2 * rq_2 * rq_2 * cq_3;
                float nkx_5 = -kx_4;
                if (rowb_1 < row_end) {
                    unsigned long long dx_base_3 = (unsigned long long)rowb_1 * 512;
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
                            float _fma_20 = __fmaf_rn(g0_7 * x0_7, rq_2, -comp[v_13 * 8 + 2 * e_7]);
                            float y_3 = _fma_20;
                            float t_4 = acc[v_13 * 8 + 2 * e_7] + y_3;
                            comp[v_13 * 8 + 2 * e_7] = t_4 - acc[v_13 * 8 + 2 * e_7] - y_3;
                            acc[v_13 * 8 + 2 * e_7] = t_4;
                            float _fma_21 = __fmaf_rn(g1_7 * x1_7, rq_2, -comp[v_13 * 8 + 2 * e_7 + 1]);
                            float y_0_3 = _fma_21;
                            float t_1_3 = acc[v_13 * 8 + 2 * e_7 + 1] + y_0_3;
                            comp[v_13 * 8 + 2 * e_7 + 1] = t_1_3 - acc[v_13 * 8 + 2 * e_7 + 1] - y_0_3;
                            acc[v_13 * 8 + 2 * e_7 + 1] = t_1_3;
                            float _fma_22 = __fmaf_rn(nkx_5, x0_7, rq_2 * (g0_7 * __uint_as_float(ww[v_13 * 4 + e_7] << 16)));
                            out_3[2 * e_7] = _fma_22;
                            float _fma_23 = __fmaf_rn(nkx_5, x1_7, rq_2 * (g1_7 * __uint_as_float(ww[v_13 * 4 + e_7] & 4294901760u)));
                            out_3[2 * e_7 + 1] = _fma_23;
                        }
                        int col_3 = (v_13 * 32 + tir) * 8;
                        {
                            __nv_bfloat162 _pk[4];
                            _pk[0] = __floats2bfloat162_rn(out_3[0 + 0], out_3[0 + 1]);
                            _pk[1] = __floats2bfloat162_rn(out_3[0 + 2], out_3[0 + 3]);
                            _pk[2] = __floats2bfloat162_rn(out_3[0 + 4], out_3[0 + 5]);
                            _pk[3] = __floats2bfloat162_rn(out_3[0 + 6], out_3[0 + 7]);
                            *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(dx))[dx_base_3 + (unsigned long long)col_3 + 0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                        }
                    }
                }
            }
        }
    }
    #pragma unroll
    for (int j2 = 0; j2 < 16; j2++) {
        acc[j2] = acc[j2] - comp[j2];
    }
    unsigned long long pbase = (unsigned long long)c * 512;
    #pragma unroll
    for (int v_14 = 0; v_14 < 2; v_14++) {
        int fcol = (v_14 * 32 + tir) * 8;
        #pragma unroll
        for (int e_8 = 0; e_8 < 8; e_8++) {
            row_fold[rg * 512 + fcol + e_8] = acc[v_14 * 8 + e_8];
        }
    }
    __syncthreads();
    float fold_acc[2];
    #pragma unroll
    for (int e_9 = 0; e_9 < 2; e_9++) {
        fold_acc[e_9] = 0.0f;
    }
    #pragma unroll
    for (int k = 0; k < 8; k++) {
        #pragma unroll
        for (int e_10 = 0; e_10 < 2; e_10++) {
            float d2 = row_fold[k * 512 + tid * 2 + e_10];
            fold_acc[e_10] = fold_acc[e_10] + d2;
        }
    }
    int pcol = tid * 2;
    {
        float2 _v2 = make_float2(fold_acc[0 + 0], fold_acc[0 + 1]);
        *reinterpret_cast<float2*>(partial + pbase + (unsigned long long)pcol) = _v2;
    }
    cooperative_groups::this_grid().sync();
    #pragma unroll 1
    for (int s2 = bid; s2 < 16; s2 += n_chunks) {
        int rsub = tid / 8;
        int cg = tid - rsub * 8;
        int col_4 = s2 * 32 + cg * 4;
        float a4[4];
        float c4[4];
        #pragma unroll
        for (int e_11 = 0; e_11 < 4; e_11++) {
            a4[e_11] = 0.0f;
            c4[e_11] = 0.0f;
        }
        int n_trips = (n_chunks + 31) / 32;
        #pragma unroll 4
        for (int k_1 = 0; k_1 < n_trips; k_1++) {
            int cc = rsub + k_1 * 32;
            if (cc < n_chunks) {
                float _vec_load_0[4];
                {
                    float4 _v4 = *reinterpret_cast<const float4*>(partial + (unsigned long long)cc * 512 + (unsigned long long)col_4);
                    _vec_load_0[0 + 0] = _v4.x;
                    _vec_load_0[0 + 1] = _v4.y;
                    _vec_load_0[0 + 2] = _v4.z;
                    _vec_load_0[0 + 3] = _v4.w;
                }
                #pragma unroll
                for (int e_12 = 0; e_12 < 4; e_12++) {
                    float _fma_24 = __fmaf_rn(_vec_load_0[e_12], 1.0f, -c4[e_12]);
                    float y_4 = _fma_24;
                    float t_5 = a4[e_12] + y_4;
                    c4[e_12] = t_5 - a4[e_12] - y_4;
                    a4[e_12] = t_5;
                }
            }
        }
        #pragma unroll
        for (int e_13 = 0; e_13 < 4; e_13++) {
            red_fold[rsub * 32 + cg * 4 + e_13] = a4[e_13] - c4[e_13];
        }
        __syncthreads();
        if (tid < 8) {
            float o4[4];
            float oc4[4];
            #pragma unroll
            for (int e_14 = 0; e_14 < 4; e_14++) {
                o4[e_14] = 0.0f;
                oc4[e_14] = 0.0f;
            }
            #pragma unroll
            for (int k_2 = 0; k_2 < 32; k_2++) {
                #pragma unroll
                for (int e_15 = 0; e_15 < 4; e_15++) {
                    float d = red_fold[k_2 * 32 + tid * 4 + e_15];
                    float _fma_25 = __fmaf_rn(d, 1.0f, -oc4[e_15]);
                    float y_5 = _fma_25;
                    float t_6 = o4[e_15] + y_5;
                    oc4[e_15] = t_6 - o4[e_15] - y_5;
                    o4[e_15] = t_6;
                }
            }
            #pragma unroll
            for (int e_16 = 0; e_16 < 4; e_16++) {
                o4[e_16] = o4[e_16] - oc4[e_16];
            }
            int ocol = s2 * 32 + tid * 4;
            {
                float4 _v4 = make_float4(o4[0 + 0], o4[0 + 1], o4[0 + 2], o4[0 + 3]);
                *reinterpret_cast<float4*>(dw + (unsigned long long)ocol) = _v4;
            }
        }
        __syncthreads();
    }
}

} // extern "C"
