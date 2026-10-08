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
#define SMEM_FLAGS_STAGE_BYTES 768
#define SMEM_FLAGS_STRIDE 768
#define SMEM_RED_FOLD_OFF 896
#define SMEM_RED_FOLD_STAGE_BYTES 4096
#define SMEM_RED_FOLD_STRIDE 4096
#define SMEM_ROW_FOLD_OFF 4992
#define SMEM_ROW_FOLD_STAGE_BYTES 128
#define SMEM_ROW_FOLD_STRIDE 128
#define SMEM_COMP_SMEM_OFF 5120
#define SMEM_COMP_SMEM_STAGE_BYTES 128
#define SMEM_COMP_SMEM_STRIDE 128
#define SMEM_W_SM_OFF 5248
#define SMEM_W_SM_STAGE_BYTES 16
#define SMEM_W_SM_STRIDE 16
#define SMEM_PART_SM_OFF 5376
#define SMEM_PART_SM_STAGE_BYTES 24576
#define SMEM_PART_SM_STRIDE 24576
#define SMEM_TOTAL 29952
#define THREADS 256

#include <math_constants.h>
#include <cooperative_groups.h>

extern "C" {

__global__ __launch_bounds__(256, 2) void
kernel_cake_rmsnorm_train_7cbb889e8e3e28efde83(__nv_bfloat16* __restrict__ g, __nv_bfloat16* __restrict__ x, __nv_bfloat16* __restrict__ w, float* __restrict__ r, __nv_bfloat16* __restrict__ dx, float* __restrict__ dw, float* __restrict__ partial, unsigned int* __restrict__ counters, __nv_bfloat16* __restrict__ g_h, int T, long long g_stride, long long x_stride, long long gh_stride, int n_chunks, int rows_per_chunk)
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
    float* red_fold = reinterpret_cast<float*>(smem_raw + 896);
    const int red_fold_addr = smem + 896;
    float* row_fold = reinterpret_cast<float*>(smem_raw + 4992);
    const int row_fold_addr = smem + 4992;
    float* comp_smem = reinterpret_cast<float*>(smem_raw + 5120);
    const int comp_smem_addr = smem + 5120;
    unsigned int* w_sm = reinterpret_cast<unsigned int*>(smem_raw + 5248);
    const int w_sm_addr = smem + 5248;
    float* part_sm = reinterpret_cast<float*>(smem_raw + 5376);
    const int part_sm_addr = smem + 5376;

    // === Task calls (dependency order) ===
    asm volatile("griddepcontrol.wait;" ::: "memory");
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    int c = bid;
    int rg = warp / 8;
    int tir = tid - rg * 256;
    int row_begin = c * rows_per_chunk;
    int row_end = row_begin + rows_per_chunk;
    if (row_end > T) {
        row_end = T;
    }
    if (row_begin > T) {
        row_begin = T;
    }
    int n_rows = row_end - row_begin;
    int n_iters = n_rows;
    int last_row = row_end - 1;
    float wc[24];
    unsigned int ww[12];
    #pragma unroll
    for (int v = 0; v < 3; v++) {
        int wcolp = (v * 256 + tir) * 8;
        {
            const uint4* _vptr_0 = reinterpret_cast<const uint4*>(w + (unsigned long long)wcolp);
            uint4* _vdst_0 = reinterpret_cast<uint4*>(&ww[v * 4]);
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vdst_0[_blk] = _vptr_0[_blk];
            }
        }
    }
    float acc[24];
    float comp[24];
    unsigned int compw[12];
    #pragma unroll
    for (int j = 0; j < 24; j++) {
        acc[j] = 0.0f;
        comp[j] = 0.0f;
    }
    {
        #pragma unroll
        for (int jp0 = 0; jp0 < 24; jp0++) {
            part_sm[jp0 * 256 + tid] = 0.0f;
        }
    }
    unsigned int gwA[12];
    unsigned int xwA[12];
    unsigned int ghwA[12];
    float rrA[1];
    unsigned int gwB[12];
    unsigned int xwB[12];
    unsigned int ghwB[12];
    float rrB[1];
    if (n_iters > 0) {
        int _min_0 = ((row_begin + rg) < (last_row) ? (row_begin + rg) : (last_row));
        int row0 = _min_0;
        unsigned long long g_base = (unsigned long long)row0 * (unsigned long long)g_stride;
        unsigned long long x_base = (unsigned long long)row0 * (unsigned long long)x_stride;
        rrA[0] = r[row0];
        #pragma unroll
        for (int v_1 = 0; v_1 < 3; v_1++) {
            int coln = (v_1 * 256 + tir) * 8;
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
        #pragma unroll 1
        for (int it = 0; it < n_iters; it += 2) {
            int rowa = row_begin + it + rg;
            int rowb = rowa + 1;
            if (n_iters > it + 1) {
                int _min_1 = ((rowb) < (last_row) ? (rowb) : (last_row));
                int rowb_ld = _min_1;
                unsigned long long g_base_0 = (unsigned long long)rowb_ld * (unsigned long long)g_stride;
                unsigned long long x_base_1 = (unsigned long long)rowb_ld * (unsigned long long)x_stride;
                rrB[0] = r[rowb_ld];
                #pragma unroll
                for (int v_2 = 0; v_2 < 3; v_2++) {
                    int coln_1 = (v_2 * 256 + tir) * 8;
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
            }
            float dq = 0.0f;
            #pragma unroll
            for (int v_3 = 0; v_3 < 3; v_3++) {
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
            if (lane == 0) {
                dot_slots[warp] = dots[0];
            }
            __syncthreads();
            float tot = 0.0f;
            #pragma unroll
            for (int k = 0; k < 8; k++) {
                float d = dot_slots[rg * 8 + k];
                tot += d;
            }
            dots[0] = tot;
            float _fdiv_rn_0 = __fdiv_rn(dots[0], 6144.0f);
            float cq = _fdiv_rn_0;
            float kx = rq * rq * rq * cq;
            float nkx = -kx;
            unsigned long long dx_base = (unsigned long long)rowa * 6144;
            #pragma unroll
            for (int v_4 = 0; v_4 < 3; v_4++) {
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
                    float _fma_2 = __fmaf_rn(g0_1 * x0_1, rq, acc[v_4 * 8 + 2 * e_1]);
                    acc[v_4 * 8 + 2 * e_1] = _fma_2;
                    float _fma_3 = __fmaf_rn(g1_1 * x1_1, rq, acc[v_4 * 8 + 2 * e_1 + 1]);
                    acc[v_4 * 8 + 2 * e_1 + 1] = _fma_3;
                    float _fma_4 = __fmaf_rn(nkx, x0_1, rq * (g0_1 * __uint_as_float(ww[v_4 * 4 + e_1] << 16)));
                    out[2 * e_1] = _fma_4;
                    float _fma_5 = __fmaf_rn(nkx, x1_1, rq * (g1_1 * __uint_as_float(ww[v_4 * 4 + e_1] & 4294901760u)));
                    out[2 * e_1 + 1] = _fma_5;
                }
                int col = (v_4 * 256 + tir) * 8;
                {
                    __nv_bfloat162 _pk[4];
                    _pk[0] = __floats2bfloat162_rn(out[0 + 0], out[0 + 1]);
                    _pk[1] = __floats2bfloat162_rn(out[0 + 2], out[0 + 3]);
                    _pk[2] = __floats2bfloat162_rn(out[0 + 4], out[0 + 5]);
                    _pk[3] = __floats2bfloat162_rn(out[0 + 6], out[0 + 7]);
                    *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(dx))[dx_base + (unsigned long long)col + 0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                }
            }
            if ((it + 1 & 7) == 0) {
                #pragma unroll
                for (int jf = 0; jf < 24; jf++) {
                    float pf = part_sm[jf * 256 + tid];
                    part_sm[jf * 256 + tid] = pf + acc[jf];
                    acc[jf] = 0.0f;
                }
            }
            if (n_iters > it + 1) {
                if (n_iters > it + 2) {
                    int _min_2 = ((rowa + 2) < (last_row) ? (rowa + 2) : (last_row));
                    int rowa2_ld = _min_2;
                    unsigned long long g_base_0_1 = (unsigned long long)rowa2_ld * (unsigned long long)g_stride;
                    unsigned long long x_base_1_1 = (unsigned long long)rowa2_ld * (unsigned long long)x_stride;
                    rrA[0] = r[rowa2_ld];
                    #pragma unroll
                    for (int v_5 = 0; v_5 < 3; v_5++) {
                        int coln_2 = (v_5 * 256 + tir) * 8;
                        {
                            const uint4* _vptr_5 = reinterpret_cast<const uint4*>(g + g_base_0_1 + (unsigned long long)coln_2);
                            uint4* _vdst_5 = reinterpret_cast<uint4*>(&gwA[v_5 * 4]);
                            #pragma unroll
                            for (int _blk = 0; _blk < 1; _blk++) {
                                asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                    : "=r"(_vdst_5[_blk].x), "=r"(_vdst_5[_blk].y), "=r"(_vdst_5[_blk].z), "=r"(_vdst_5[_blk].w) : "l"((const void*)(_vptr_5 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                            }
                        }
                        {
                            const uint4* _vptr_6 = reinterpret_cast<const uint4*>(x + x_base_1_1 + (unsigned long long)coln_2);
                            uint4* _vdst_6 = reinterpret_cast<uint4*>(&xwA[v_5 * 4]);
                            #pragma unroll
                            for (int _blk = 0; _blk < 1; _blk++) {
                                asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                    : "=r"(_vdst_6[_blk].x), "=r"(_vdst_6[_blk].y), "=r"(_vdst_6[_blk].z), "=r"(_vdst_6[_blk].w) : "l"((const void*)(_vptr_6 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                            }
                        }
                    }
                }
                float dq_0 = 0.0f;
                #pragma unroll
                for (int v_6 = 0; v_6 < 3; v_6++) {
                    unsigned int wvd_1[4];
                    #pragma unroll
                    for (int e_2 = 0; e_2 < 4; e_2++) {
                        unsigned int gwd_2 = gwB[v_6 * 4 + e_2];
                        unsigned int xwd_2 = xwB[v_6 * 4 + e_2];
                        float g0_2 = __uint_as_float(gwd_2 << 16);
                        float g1_2 = __uint_as_float(gwd_2 & 4294901760u);
                        float x0_2 = __uint_as_float(xwd_2 << 16);
                        float x1_2 = __uint_as_float(xwd_2 & 4294901760u);
                        float _fma_6 = __fmaf_rn(g0_2 * __uint_as_float(ww[v_6 * 4 + e_2] << 16), x0_2, dq_0);
                        dq_0 = _fma_6;
                        float _fma_7 = __fmaf_rn(g1_2 * __uint_as_float(ww[v_6 * 4 + e_2] & 4294901760u), x1_2, dq_0);
                        dq_0 = _fma_7;
                    }
                }
                float dots_1[1];
                float _warp_reduce_1 = dq_0;
                #pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1)
                    _warp_reduce_1 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_1, offset);
                dots_1[0] = _warp_reduce_1;
                float rq_2 = rrB[0];
                if (lane == 0) {
                    dot_slots[8 + warp] = dots_1[0];
                }
                __syncthreads();
                float tot_3 = 0.0f;
                #pragma unroll
                for (int k_1 = 0; k_1 < 8; k_1++) {
                    float d_1 = dot_slots[8 + rg * 8 + k_1];
                    tot_3 += d_1;
                }
                dots_1[0] = tot_3;
                float _fdiv_rn_1 = __fdiv_rn(dots_1[0], 6144.0f);
                float cq_4 = _fdiv_rn_1;
                float kx_5 = rq_2 * rq_2 * rq_2 * cq_4;
                float nkx_6 = -kx_5;
                unsigned long long dx_base_7 = (unsigned long long)rowb * 6144;
                #pragma unroll
                for (int v_7 = 0; v_7 < 3; v_7++) {
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
                        float _fma_8 = __fmaf_rn(g0_3 * x0_3, rq_2, acc[v_7 * 8 + 2 * e_3]);
                        acc[v_7 * 8 + 2 * e_3] = _fma_8;
                        float _fma_9 = __fmaf_rn(g1_3 * x1_3, rq_2, acc[v_7 * 8 + 2 * e_3 + 1]);
                        acc[v_7 * 8 + 2 * e_3 + 1] = _fma_9;
                        float _fma_10 = __fmaf_rn(nkx_6, x0_3, rq_2 * (g0_3 * __uint_as_float(ww[v_7 * 4 + e_3] << 16)));
                        out_1[2 * e_3] = _fma_10;
                        float _fma_11 = __fmaf_rn(nkx_6, x1_3, rq_2 * (g1_3 * __uint_as_float(ww[v_7 * 4 + e_3] & 4294901760u)));
                        out_1[2 * e_3 + 1] = _fma_11;
                    }
                    int col_1 = (v_7 * 256 + tir) * 8;
                    {
                        __nv_bfloat162 _pk[4];
                        _pk[0] = __floats2bfloat162_rn(out_1[0 + 0], out_1[0 + 1]);
                        _pk[1] = __floats2bfloat162_rn(out_1[0 + 2], out_1[0 + 3]);
                        _pk[2] = __floats2bfloat162_rn(out_1[0 + 4], out_1[0 + 5]);
                        _pk[3] = __floats2bfloat162_rn(out_1[0 + 6], out_1[0 + 7]);
                        *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(dx))[dx_base_7 + (unsigned long long)col_1 + 0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                    }
                }
                if ((it + 1 + 1 & 7) == 0) {
                    #pragma unroll
                    for (int jf_1 = 0; jf_1 < 24; jf_1++) {
                        float pf_1 = part_sm[jf_1 * 256 + tid];
                        part_sm[jf_1 * 256 + tid] = pf_1 + acc[jf_1];
                        acc[jf_1] = 0.0f;
                    }
                }
            }
        }
    }
    {
        #pragma unroll
        for (int jp = 0; jp < 24; jp++) {
            float pfin = part_sm[jp * 256 + tid];
            acc[jp] = pfin + acc[jp];
        }
    }
    unsigned long long pbase = (unsigned long long)c * 6144;
    #pragma unroll
    for (int v_8 = 0; v_8 < 3; v_8++) {
        int pcolv = (v_8 * 256 + tir) * 8;
        {
            unsigned _stv8_7_0 = __float_as_uint(acc[v_8 * 8 + 0]);
            unsigned _stv8_7_1 = __float_as_uint(acc[v_8 * 8 + 1]);
            unsigned _stv8_7_2 = __float_as_uint(acc[v_8 * 8 + 2]);
            unsigned _stv8_7_3 = __float_as_uint(acc[v_8 * 8 + 3]);
            unsigned _stv8_7_4 = __float_as_uint(acc[v_8 * 8 + 4]);
            unsigned _stv8_7_5 = __float_as_uint(acc[v_8 * 8 + 5]);
            unsigned _stv8_7_6 = __float_as_uint(acc[v_8 * 8 + 6]);
            unsigned _stv8_7_7 = __float_as_uint(acc[v_8 * 8 + 7]);
            asm volatile(
                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                :: "l"((void*)(partial + (pbase + (unsigned long long)pcolv))), "r"(_stv8_7_0), "r"(_stv8_7_1), "r"(_stv8_7_2), "r"(_stv8_7_3), "r"(_stv8_7_4), "r"(_stv8_7_5), "r"(_stv8_7_6), "r"(_stv8_7_7) : "memory");
        }
    }
    cooperative_groups::this_grid().sync();
    #pragma unroll 1
    for (int s2 = bid; s2 < 192; s2 += n_chunks) {
        int rsub = tid / 8;
        int cg = tid - rsub * 8;
        int col_2 = s2 * 32 + cg * 4;
        float a4[4];
        float c4[4];
        #pragma unroll
        for (int e_4 = 0; e_4 < 4; e_4++) {
            a4[e_4] = 0.0f;
            c4[e_4] = 0.0f;
        }
        int n_trips = (n_chunks + 31) / 32;
        #pragma unroll 4
        for (int k_2 = 0; k_2 < n_trips; k_2++) {
            int cc = rsub + k_2 * 32;
            if (cc < n_chunks) {
                float _vec_load_0[4];
                {
                    float4 _v4 = *reinterpret_cast<const float4*>(partial + (unsigned long long)cc * 6144 + (unsigned long long)col_2);
                    _vec_load_0[0 + 0] = _v4.x;
                    _vec_load_0[0 + 1] = _v4.y;
                    _vec_load_0[0 + 2] = _v4.z;
                    _vec_load_0[0 + 3] = _v4.w;
                }
                #pragma unroll
                for (int e_5 = 0; e_5 < 4; e_5++) {
                    float _fma_12 = __fmaf_rn(_vec_load_0[e_5], 1.0f, -c4[e_5]);
                    float y = _fma_12;
                    float t = a4[e_5] + y;
                    c4[e_5] = t - a4[e_5] - y;
                    a4[e_5] = t;
                }
            }
        }
        #pragma unroll
        for (int e_6 = 0; e_6 < 4; e_6++) {
            red_fold[rsub * 32 + cg * 4 + e_6] = a4[e_6] - c4[e_6];
        }
        __syncthreads();
        if (tid < 8) {
            float o4[4];
            float oc4[4];
            #pragma unroll
            for (int e_7 = 0; e_7 < 4; e_7++) {
                o4[e_7] = 0.0f;
                oc4[e_7] = 0.0f;
            }
            #pragma unroll
            for (int k_3 = 0; k_3 < 32; k_3++) {
                #pragma unroll
                for (int e_8 = 0; e_8 < 4; e_8++) {
                    float d_2 = red_fold[k_3 * 32 + tid * 4 + e_8];
                    float _fma_13 = __fmaf_rn(d_2, 1.0f, -oc4[e_8]);
                    float y_1 = _fma_13;
                    float t_1 = o4[e_8] + y_1;
                    oc4[e_8] = t_1 - o4[e_8] - y_1;
                    o4[e_8] = t_1;
                }
            }
            #pragma unroll
            for (int e_9 = 0; e_9 < 4; e_9++) {
                o4[e_9] = o4[e_9] - oc4[e_9];
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
