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
#define SMEM_FLAGS_OFF 0
#define SMEM_FLAGS_STAGE_BYTES 32
#define SMEM_FLAGS_STRIDE 32
#define SMEM_DOTS_OFF 32
#define SMEM_DOTS_STAGE_BYTES 64
#define SMEM_DOTS_STRIDE 64
#define SMEM_SUMS_OFF 96
#define SMEM_SUMS_STAGE_BYTES 32768
#define SMEM_SUMS_STRIDE 32768
#define SMEM_DWBUF_OFF 96
#define SMEM_DWBUF_STAGE_BYTES 2048
#define SMEM_DWBUF_STRIDE 2048
#define SMEM_TOTAL 32896
#define THREADS 256
#define LAUNCH_MIN_BLOCKS 2

#include <math_constants.h>

extern "C" {

__global__ __launch_bounds__(256, LAUNCH_MIN_BLOCKS) void
kernel_cake_rmsnorm_train_d0bde38aa443d42a5f04(__nv_bfloat16* __restrict__ g, __nv_bfloat16* __restrict__ x, __nv_bfloat16* __restrict__ w, float* __restrict__ r, __nv_bfloat16* __restrict__ dx, float* __restrict__ dw, double* __restrict__ partial, unsigned int* __restrict__ counters, __nv_bfloat16* __restrict__ g_h, int T, int n_chunks, int rows_per_chunk, long long g_stride_p0, long long x_stride_p0, long long g_h_stride_p0)
{
    // Grid-dependency payloads: written while the kernel runs, read with coherent loads only.
    asm volatile("" : "+l"(partial));
    __builtin_assume(__isGlobal(partial));

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
    int* flags = reinterpret_cast<int*>(smem_raw + 0);
    const int flags_addr = smem + 0;
    float* dots = reinterpret_cast<float*>(smem_raw + 32);
    const int dots_addr = smem + 32;
    double* sums = reinterpret_cast<double*>(smem_raw + 96);
    const int sums_addr = smem + 96;
    float* dwbuf = reinterpret_cast<float*>(smem_raw + 96);
    const int dwbuf_addr = smem + 96;

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
    int rg = warp / 1;
    int tir = tid - rg * 32;
    unsigned int ww[8];
    {
        const uint4* _vptr_1 = reinterpret_cast<const uint4*>(w + (unsigned long long)(tir * 8));
        uint4* _vdst_1 = reinterpret_cast<uint4*>(&ww[0]);
        #pragma unroll
        for (int _blk = 0; _blk < 1; _blk++) {
            _vdst_1[_blk] = _vptr_1[_blk];
        }
    }
    {
        const uint4* _vptr_2 = reinterpret_cast<const uint4*>(w + (unsigned long long)((32 + tir) * 8));
        uint4* _vdst_2 = reinterpret_cast<uint4*>(&ww[4]);
        #pragma unroll
        for (int _blk = 0; _blk < 1; _blk++) {
            _vdst_2[_blk] = _vptr_2[_blk];
        }
    }
    float acc[16];
    float comp[1];
    acc[0] = 0.0f;
    acc[1] = 0.0f;
    acc[2] = 0.0f;
    acc[3] = 0.0f;
    acc[4] = 0.0f;
    acc[5] = 0.0f;
    acc[6] = 0.0f;
    acc[7] = 0.0f;
    acc[8] = 0.0f;
    acc[9] = 0.0f;
    acc[10] = 0.0f;
    acc[11] = 0.0f;
    acc[12] = 0.0f;
    acc[13] = 0.0f;
    acc[14] = 0.0f;
    acc[15] = 0.0f;
    int c = 0;
    int first_c = 0;
    sums[rg * 512 + tir] = 0.0;
    sums[rg * 512 + 32 + tir] = 0.0;
    sums[rg * 512 + 64 + tir] = 0.0;
    sums[rg * 512 + 96 + tir] = 0.0;
    sums[rg * 512 + 128 + tir] = 0.0;
    sums[rg * 512 + 160 + tir] = 0.0;
    sums[rg * 512 + 192 + tir] = 0.0;
    sums[rg * 512 + 224 + tir] = 0.0;
    sums[rg * 512 + 256 + tir] = 0.0;
    sums[rg * 512 + 288 + tir] = 0.0;
    sums[rg * 512 + 320 + tir] = 0.0;
    sums[rg * 512 + 352 + tir] = 0.0;
    sums[rg * 512 + 384 + tir] = 0.0;
    sums[rg * 512 + 416 + tir] = 0.0;
    sums[rg * 512 + 448 + tir] = 0.0;
    sums[rg * 512 + 480 + tir] = 0.0;
    unsigned int gw[16];
    unsigned int xw[16];
    unsigned int hw[1];
    float rr[2];
    if (tid == 0) {
        flags[4] = (int)nxt;
    }
    __syncthreads();
    c = flags[4];
    first_c = c;
    int row_begin = 0;
    int row_end = 0;
    int n_iters = 0;
    int last_row = 0;
    int window = 0;
    #pragma unroll 1
    for (int claim_it = 0; claim_it < n_chunks; claim_it++) {
        if (c >= n_chunks) {
            break;
        }
        if (tid == 0) {
            unsigned int _atomic_fetch_add_3 = atomicAdd(&counters[0], 1);
            nxt = _atomic_fetch_add_3;
        }
        int stages = (T + 7) / 8;
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
        int span = 0;
        int halving = 1;
        int begin_it = 0;
        #pragma unroll 1
        for (int k = 0; k < 5; k++) {
            if (halving != 0) {
                if (size > tail) {
                    if (rem >= sl * size) {
                        if (c < first + sl) {
                            span = size;
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
        if (span == 0) {
            int quot = rem / sl;
            int extra = rem - quot * sl;
            if (c < first + extra) {
                span = quot + 1;
                begin_it = base + (c - first) * (quot + 1);
            } else if (c < first + sl) {
                span = quot;
                begin_it = base + extra * (quot + 1) + (c - first - extra) * quot;
            } else {
                begin_it = stages;
            }
        }
        int _min_4 = ((begin_it * 8) < (T) ? (begin_it * 8) : (T));
        row_begin = _min_4;
        int _min_5 = ((begin_it * 8 + span * 8) < (T) ? (begin_it * 8 + span * 8) : (T));
        row_end = _min_5;
        n_iters = (row_end - row_begin + 7) / 8;
        last_row = row_end - 1;
        if (n_iters > 0) {
            int _min_6 = ((row_begin + rg) < (last_row) ? (row_begin + rg) : (last_row));
            unsigned long long g_base = (unsigned long long)_min_6 * (unsigned long long)g_stride;
            int _min_7 = ((row_begin + rg) < (last_row) ? (row_begin + rg) : (last_row));
            unsigned long long x_base = (unsigned long long)_min_7 * (unsigned long long)x_stride;
            int _min_8 = ((row_begin + rg) < (last_row) ? (row_begin + rg) : (last_row));
            rr[0] = r[_min_8];
            {
                const uint4* _vptr_9 = reinterpret_cast<const uint4*>(g + g_base + (unsigned long long)(tir * 8));
                uint4* _vdst_9 = reinterpret_cast<uint4*>(&gw[0]);
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                        : "=r"(_vdst_9[_blk].x), "=r"(_vdst_9[_blk].y), "=r"(_vdst_9[_blk].z), "=r"(_vdst_9[_blk].w) : "l"((const void*)(_vptr_9 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                }
            }
            {
                const uint4* _vptr_10 = reinterpret_cast<const uint4*>(x + x_base + (unsigned long long)(tir * 8));
                uint4* _vdst_10 = reinterpret_cast<uint4*>(&xw[0]);
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                        : "=r"(_vdst_10[_blk].x), "=r"(_vdst_10[_blk].y), "=r"(_vdst_10[_blk].z), "=r"(_vdst_10[_blk].w) : "l"((const void*)(_vptr_10 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                }
            }
            {
                const uint4* _vptr_11 = reinterpret_cast<const uint4*>(g + g_base + (unsigned long long)((32 + tir) * 8));
                uint4* _vdst_11 = reinterpret_cast<uint4*>(&gw[4]);
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                        : "=r"(_vdst_11[_blk].x), "=r"(_vdst_11[_blk].y), "=r"(_vdst_11[_blk].z), "=r"(_vdst_11[_blk].w) : "l"((const void*)(_vptr_11 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                }
            }
            {
                const uint4* _vptr_12 = reinterpret_cast<const uint4*>(x + x_base + (unsigned long long)((32 + tir) * 8));
                uint4* _vdst_12 = reinterpret_cast<uint4*>(&xw[4]);
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                        : "=r"(_vdst_12[_blk].x), "=r"(_vdst_12[_blk].y), "=r"(_vdst_12[_blk].z), "=r"(_vdst_12[_blk].w) : "l"((const void*)(_vptr_12 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                }
            }
        }
        #pragma unroll 1
        for (int it0 = 0; it0 < n_iters; it0 += 2) {
            if (n_iters > it0) {
                if (n_iters > it0 + 1) {
                    int _min_13 = ((row_begin + (it0 + 2 - 1) * 8 + rg) < (last_row) ? (row_begin + (it0 + 2 - 1) * 8 + rg) : (last_row));
                    unsigned long long g_base_1 = (unsigned long long)_min_13 * (unsigned long long)g_stride;
                    int _min_14 = ((row_begin + (it0 + 2 - 1) * 8 + rg) < (last_row) ? (row_begin + (it0 + 2 - 1) * 8 + rg) : (last_row));
                    unsigned long long x_base_1 = (unsigned long long)_min_14 * (unsigned long long)x_stride;
                    int _min_15 = ((row_begin + (it0 + 2 - 1) * 8 + rg) < (last_row) ? (row_begin + (it0 + 2 - 1) * 8 + rg) : (last_row));
                    rr[1] = r[_min_15];
                    {
                        const uint4* _vptr_16 = reinterpret_cast<const uint4*>(g + g_base_1 + (unsigned long long)(tir * 8));
                        uint4* _vdst_16 = reinterpret_cast<uint4*>(&gw[8]);
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                : "=r"(_vdst_16[_blk].x), "=r"(_vdst_16[_blk].y), "=r"(_vdst_16[_blk].z), "=r"(_vdst_16[_blk].w) : "l"((const void*)(_vptr_16 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                        }
                    }
                    {
                        const uint4* _vptr_17 = reinterpret_cast<const uint4*>(x + x_base_1 + (unsigned long long)(tir * 8));
                        uint4* _vdst_17 = reinterpret_cast<uint4*>(&xw[8]);
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                : "=r"(_vdst_17[_blk].x), "=r"(_vdst_17[_blk].y), "=r"(_vdst_17[_blk].z), "=r"(_vdst_17[_blk].w) : "l"((const void*)(_vptr_17 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                        }
                    }
                    {
                        const uint4* _vptr_18 = reinterpret_cast<const uint4*>(g + g_base_1 + (unsigned long long)((32 + tir) * 8));
                        uint4* _vdst_18 = reinterpret_cast<uint4*>(&gw[12]);
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                : "=r"(_vdst_18[_blk].x), "=r"(_vdst_18[_blk].y), "=r"(_vdst_18[_blk].z), "=r"(_vdst_18[_blk].w) : "l"((const void*)(_vptr_18 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                        }
                    }
                    {
                        const uint4* _vptr_19 = reinterpret_cast<const uint4*>(x + x_base_1 + (unsigned long long)((32 + tir) * 8));
                        uint4* _vdst_19 = reinterpret_cast<uint4*>(&xw[12]);
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                : "=r"(_vdst_19[_blk].x), "=r"(_vdst_19[_blk].y), "=r"(_vdst_19[_blk].z), "=r"(_vdst_19[_blk].w) : "l"((const void*)(_vptr_19 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                        }
                    }
                }
                float dot[1];
                float dq = 0.0f;
                dq = __fmaf_rn(__uint_as_float(gw[0] << 16) * __uint_as_float(ww[0] << 16), __uint_as_float(xw[0] << 16), dq);
                dq = __fmaf_rn(__uint_as_float(gw[0] & 4294901760u) * __uint_as_float(ww[0] & 4294901760u), __uint_as_float(xw[0] & 4294901760u), dq);
                dq = __fmaf_rn(__uint_as_float(gw[1] << 16) * __uint_as_float(ww[1] << 16), __uint_as_float(xw[1] << 16), dq);
                dq = __fmaf_rn(__uint_as_float(gw[1] & 4294901760u) * __uint_as_float(ww[1] & 4294901760u), __uint_as_float(xw[1] & 4294901760u), dq);
                dq = __fmaf_rn(__uint_as_float(gw[2] << 16) * __uint_as_float(ww[2] << 16), __uint_as_float(xw[2] << 16), dq);
                dq = __fmaf_rn(__uint_as_float(gw[2] & 4294901760u) * __uint_as_float(ww[2] & 4294901760u), __uint_as_float(xw[2] & 4294901760u), dq);
                dq = __fmaf_rn(__uint_as_float(gw[3] << 16) * __uint_as_float(ww[3] << 16), __uint_as_float(xw[3] << 16), dq);
                dq = __fmaf_rn(__uint_as_float(gw[3] & 4294901760u) * __uint_as_float(ww[3] & 4294901760u), __uint_as_float(xw[3] & 4294901760u), dq);
                dq = __fmaf_rn(__uint_as_float(gw[4] << 16) * __uint_as_float(ww[4] << 16), __uint_as_float(xw[4] << 16), dq);
                dq = __fmaf_rn(__uint_as_float(gw[4] & 4294901760u) * __uint_as_float(ww[4] & 4294901760u), __uint_as_float(xw[4] & 4294901760u), dq);
                dq = __fmaf_rn(__uint_as_float(gw[5] << 16) * __uint_as_float(ww[5] << 16), __uint_as_float(xw[5] << 16), dq);
                dq = __fmaf_rn(__uint_as_float(gw[5] & 4294901760u) * __uint_as_float(ww[5] & 4294901760u), __uint_as_float(xw[5] & 4294901760u), dq);
                dq = __fmaf_rn(__uint_as_float(gw[6] << 16) * __uint_as_float(ww[6] << 16), __uint_as_float(xw[6] << 16), dq);
                dq = __fmaf_rn(__uint_as_float(gw[6] & 4294901760u) * __uint_as_float(ww[6] & 4294901760u), __uint_as_float(xw[6] & 4294901760u), dq);
                dq = __fmaf_rn(__uint_as_float(gw[7] << 16) * __uint_as_float(ww[7] << 16), __uint_as_float(xw[7] << 16), dq);
                dq = __fmaf_rn(__uint_as_float(gw[7] & 4294901760u) * __uint_as_float(ww[7] & 4294901760u), __uint_as_float(xw[7] & 4294901760u), dq);
                float _warp_reduce_20 = dq;
                #pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1)
                    _warp_reduce_20 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_20, offset);
                dot[0] = _warp_reduce_20;
                float rq = rr[0];
                float cq = __fdiv_rn(dot[0], 512.0f);
                float kx = rq * rq * rq * cq;
                float nkx = -kx;
                if (row_begin + it0 * 8 + rg < row_end) {
                    float out_0[8];
                    float g0_0 = __uint_as_float(gw[0] << 16);
                    float g1_0 = __uint_as_float(gw[0] & 4294901760u);
                    float x0_0 = __uint_as_float(xw[0] << 16);
                    float x1_0 = __uint_as_float(xw[0] & 4294901760u);
                    acc[0] = __fmaf_rn(g0_0 * x0_0, rq, acc[0]);
                    acc[1] = __fmaf_rn(g1_0 * x1_0, rq, acc[1]);
                    float y0_0 = __fmaf_rn(nkx, x0_0, rq * (g0_0 * __uint_as_float(ww[0] << 16)));
                    float y1_0 = __fmaf_rn(nkx, x1_0, rq * (g1_0 * __uint_as_float(ww[0] & 4294901760u)));
                    out_0[0] = y0_0;
                    out_0[1] = y1_0;
                    float g0_1 = __uint_as_float(gw[1] << 16);
                    float g1_1 = __uint_as_float(gw[1] & 4294901760u);
                    float x0_1 = __uint_as_float(xw[1] << 16);
                    float x1_1 = __uint_as_float(xw[1] & 4294901760u);
                    acc[2] = __fmaf_rn(g0_1 * x0_1, rq, acc[2]);
                    acc[3] = __fmaf_rn(g1_1 * x1_1, rq, acc[3]);
                    float y0_1 = __fmaf_rn(nkx, x0_1, rq * (g0_1 * __uint_as_float(ww[1] << 16)));
                    float y1_1 = __fmaf_rn(nkx, x1_1, rq * (g1_1 * __uint_as_float(ww[1] & 4294901760u)));
                    out_0[2] = y0_1;
                    out_0[3] = y1_1;
                    float g0_2 = __uint_as_float(gw[2] << 16);
                    float g1_2 = __uint_as_float(gw[2] & 4294901760u);
                    float x0_2 = __uint_as_float(xw[2] << 16);
                    float x1_2 = __uint_as_float(xw[2] & 4294901760u);
                    acc[4] = __fmaf_rn(g0_2 * x0_2, rq, acc[4]);
                    acc[5] = __fmaf_rn(g1_2 * x1_2, rq, acc[5]);
                    float y0_2 = __fmaf_rn(nkx, x0_2, rq * (g0_2 * __uint_as_float(ww[2] << 16)));
                    float y1_2 = __fmaf_rn(nkx, x1_2, rq * (g1_2 * __uint_as_float(ww[2] & 4294901760u)));
                    out_0[4] = y0_2;
                    out_0[5] = y1_2;
                    float g0_3 = __uint_as_float(gw[3] << 16);
                    float g1_3 = __uint_as_float(gw[3] & 4294901760u);
                    float x0_3 = __uint_as_float(xw[3] << 16);
                    float x1_3 = __uint_as_float(xw[3] & 4294901760u);
                    acc[6] = __fmaf_rn(g0_3 * x0_3, rq, acc[6]);
                    acc[7] = __fmaf_rn(g1_3 * x1_3, rq, acc[7]);
                    float y0_3 = __fmaf_rn(nkx, x0_3, rq * (g0_3 * __uint_as_float(ww[3] << 16)));
                    float y1_3 = __fmaf_rn(nkx, x1_3, rq * (g1_3 * __uint_as_float(ww[3] & 4294901760u)));
                    out_0[6] = y0_3;
                    out_0[7] = y1_3;
                    unsigned int packed_0[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_0[_lp*2 + 0], out_0[_lp*2+1 + 0]));
                        packed_0[_lp] = *(uint32_t*)&_bf2;
                    }
                    reinterpret_cast<int4*>(dx + ((unsigned long long)(row_begin + it0 * 8 + rg) * 512 + (unsigned long long)(tir * 8)))[0] = reinterpret_cast<int4*>(packed_0)[0];
                    float out_1[8];
                    float g0_4 = __uint_as_float(gw[4] << 16);
                    float g1_4 = __uint_as_float(gw[4] & 4294901760u);
                    float x0_4 = __uint_as_float(xw[4] << 16);
                    float x1_4 = __uint_as_float(xw[4] & 4294901760u);
                    acc[8] = __fmaf_rn(g0_4 * x0_4, rq, acc[8]);
                    acc[9] = __fmaf_rn(g1_4 * x1_4, rq, acc[9]);
                    float y0_4 = __fmaf_rn(nkx, x0_4, rq * (g0_4 * __uint_as_float(ww[4] << 16)));
                    float y1_4 = __fmaf_rn(nkx, x1_4, rq * (g1_4 * __uint_as_float(ww[4] & 4294901760u)));
                    out_1[0] = y0_4;
                    out_1[1] = y1_4;
                    float g0_5 = __uint_as_float(gw[5] << 16);
                    float g1_5 = __uint_as_float(gw[5] & 4294901760u);
                    float x0_5 = __uint_as_float(xw[5] << 16);
                    float x1_5 = __uint_as_float(xw[5] & 4294901760u);
                    acc[10] = __fmaf_rn(g0_5 * x0_5, rq, acc[10]);
                    acc[11] = __fmaf_rn(g1_5 * x1_5, rq, acc[11]);
                    float y0_5 = __fmaf_rn(nkx, x0_5, rq * (g0_5 * __uint_as_float(ww[5] << 16)));
                    float y1_5 = __fmaf_rn(nkx, x1_5, rq * (g1_5 * __uint_as_float(ww[5] & 4294901760u)));
                    out_1[2] = y0_5;
                    out_1[3] = y1_5;
                    float g0_6 = __uint_as_float(gw[6] << 16);
                    float g1_6 = __uint_as_float(gw[6] & 4294901760u);
                    float x0_6 = __uint_as_float(xw[6] << 16);
                    float x1_6 = __uint_as_float(xw[6] & 4294901760u);
                    acc[12] = __fmaf_rn(g0_6 * x0_6, rq, acc[12]);
                    acc[13] = __fmaf_rn(g1_6 * x1_6, rq, acc[13]);
                    float y0_6 = __fmaf_rn(nkx, x0_6, rq * (g0_6 * __uint_as_float(ww[6] << 16)));
                    float y1_6 = __fmaf_rn(nkx, x1_6, rq * (g1_6 * __uint_as_float(ww[6] & 4294901760u)));
                    out_1[4] = y0_6;
                    out_1[5] = y1_6;
                    float g0_7 = __uint_as_float(gw[7] << 16);
                    float g1_7 = __uint_as_float(gw[7] & 4294901760u);
                    float x0_7 = __uint_as_float(xw[7] << 16);
                    float x1_7 = __uint_as_float(xw[7] & 4294901760u);
                    acc[14] = __fmaf_rn(g0_7 * x0_7, rq, acc[14]);
                    acc[15] = __fmaf_rn(g1_7 * x1_7, rq, acc[15]);
                    float y0_7 = __fmaf_rn(nkx, x0_7, rq * (g0_7 * __uint_as_float(ww[7] << 16)));
                    float y1_7 = __fmaf_rn(nkx, x1_7, rq * (g1_7 * __uint_as_float(ww[7] & 4294901760u)));
                    out_1[6] = y0_7;
                    out_1[7] = y1_7;
                    unsigned int packed_1[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_1[_lp*2 + 0], out_1[_lp*2+1 + 0]));
                        packed_1[_lp] = *(uint32_t*)&_bf2;
                    }
                    reinterpret_cast<int4*>(dx + ((unsigned long long)(row_begin + it0 * 8 + rg) * 512 + (unsigned long long)((32 + tir) * 8)))[0] = reinterpret_cast<int4*>(packed_1)[0];
                }
            }
            if (n_iters > it0 + 1) {
                if (n_iters > it0 + 1 + 1) {
                    int _min_21 = ((row_begin + (it0 + 1 + 2 - 1) * 8 + rg) < (last_row) ? (row_begin + (it0 + 1 + 2 - 1) * 8 + rg) : (last_row));
                    unsigned long long g_base_2 = (unsigned long long)_min_21 * (unsigned long long)g_stride;
                    int _min_22 = ((row_begin + (it0 + 1 + 2 - 1) * 8 + rg) < (last_row) ? (row_begin + (it0 + 1 + 2 - 1) * 8 + rg) : (last_row));
                    unsigned long long x_base_2 = (unsigned long long)_min_22 * (unsigned long long)x_stride;
                    int _min_23 = ((row_begin + (it0 + 1 + 2 - 1) * 8 + rg) < (last_row) ? (row_begin + (it0 + 1 + 2 - 1) * 8 + rg) : (last_row));
                    rr[0] = r[_min_23];
                    {
                        const uint4* _vptr_24 = reinterpret_cast<const uint4*>(g + g_base_2 + (unsigned long long)(tir * 8));
                        uint4* _vdst_24 = reinterpret_cast<uint4*>(&gw[0]);
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                : "=r"(_vdst_24[_blk].x), "=r"(_vdst_24[_blk].y), "=r"(_vdst_24[_blk].z), "=r"(_vdst_24[_blk].w) : "l"((const void*)(_vptr_24 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                        }
                    }
                    {
                        const uint4* _vptr_25 = reinterpret_cast<const uint4*>(x + x_base_2 + (unsigned long long)(tir * 8));
                        uint4* _vdst_25 = reinterpret_cast<uint4*>(&xw[0]);
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                : "=r"(_vdst_25[_blk].x), "=r"(_vdst_25[_blk].y), "=r"(_vdst_25[_blk].z), "=r"(_vdst_25[_blk].w) : "l"((const void*)(_vptr_25 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                        }
                    }
                    {
                        const uint4* _vptr_26 = reinterpret_cast<const uint4*>(g + g_base_2 + (unsigned long long)((32 + tir) * 8));
                        uint4* _vdst_26 = reinterpret_cast<uint4*>(&gw[4]);
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                : "=r"(_vdst_26[_blk].x), "=r"(_vdst_26[_blk].y), "=r"(_vdst_26[_blk].z), "=r"(_vdst_26[_blk].w) : "l"((const void*)(_vptr_26 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                        }
                    }
                    {
                        const uint4* _vptr_27 = reinterpret_cast<const uint4*>(x + x_base_2 + (unsigned long long)((32 + tir) * 8));
                        uint4* _vdst_27 = reinterpret_cast<uint4*>(&xw[4]);
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                : "=r"(_vdst_27[_blk].x), "=r"(_vdst_27[_blk].y), "=r"(_vdst_27[_blk].z), "=r"(_vdst_27[_blk].w) : "l"((const void*)(_vptr_27 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                        }
                    }
                }
                float dot_1[1];
                float dq_1 = 0.0f;
                dq_1 = __fmaf_rn(__uint_as_float(gw[8] << 16) * __uint_as_float(ww[0] << 16), __uint_as_float(xw[8] << 16), dq_1);
                dq_1 = __fmaf_rn(__uint_as_float(gw[8] & 4294901760u) * __uint_as_float(ww[0] & 4294901760u), __uint_as_float(xw[8] & 4294901760u), dq_1);
                dq_1 = __fmaf_rn(__uint_as_float(gw[9] << 16) * __uint_as_float(ww[1] << 16), __uint_as_float(xw[9] << 16), dq_1);
                dq_1 = __fmaf_rn(__uint_as_float(gw[9] & 4294901760u) * __uint_as_float(ww[1] & 4294901760u), __uint_as_float(xw[9] & 4294901760u), dq_1);
                dq_1 = __fmaf_rn(__uint_as_float(gw[10] << 16) * __uint_as_float(ww[2] << 16), __uint_as_float(xw[10] << 16), dq_1);
                dq_1 = __fmaf_rn(__uint_as_float(gw[10] & 4294901760u) * __uint_as_float(ww[2] & 4294901760u), __uint_as_float(xw[10] & 4294901760u), dq_1);
                dq_1 = __fmaf_rn(__uint_as_float(gw[11] << 16) * __uint_as_float(ww[3] << 16), __uint_as_float(xw[11] << 16), dq_1);
                dq_1 = __fmaf_rn(__uint_as_float(gw[11] & 4294901760u) * __uint_as_float(ww[3] & 4294901760u), __uint_as_float(xw[11] & 4294901760u), dq_1);
                dq_1 = __fmaf_rn(__uint_as_float(gw[12] << 16) * __uint_as_float(ww[4] << 16), __uint_as_float(xw[12] << 16), dq_1);
                dq_1 = __fmaf_rn(__uint_as_float(gw[12] & 4294901760u) * __uint_as_float(ww[4] & 4294901760u), __uint_as_float(xw[12] & 4294901760u), dq_1);
                dq_1 = __fmaf_rn(__uint_as_float(gw[13] << 16) * __uint_as_float(ww[5] << 16), __uint_as_float(xw[13] << 16), dq_1);
                dq_1 = __fmaf_rn(__uint_as_float(gw[13] & 4294901760u) * __uint_as_float(ww[5] & 4294901760u), __uint_as_float(xw[13] & 4294901760u), dq_1);
                dq_1 = __fmaf_rn(__uint_as_float(gw[14] << 16) * __uint_as_float(ww[6] << 16), __uint_as_float(xw[14] << 16), dq_1);
                dq_1 = __fmaf_rn(__uint_as_float(gw[14] & 4294901760u) * __uint_as_float(ww[6] & 4294901760u), __uint_as_float(xw[14] & 4294901760u), dq_1);
                dq_1 = __fmaf_rn(__uint_as_float(gw[15] << 16) * __uint_as_float(ww[7] << 16), __uint_as_float(xw[15] << 16), dq_1);
                dq_1 = __fmaf_rn(__uint_as_float(gw[15] & 4294901760u) * __uint_as_float(ww[7] & 4294901760u), __uint_as_float(xw[15] & 4294901760u), dq_1);
                float _warp_reduce_28 = dq_1;
                #pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1)
                    _warp_reduce_28 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_28, offset);
                dot_1[0] = _warp_reduce_28;
                float rq_1 = rr[1];
                float cq_1 = __fdiv_rn(dot_1[0], 512.0f);
                float kx_1 = rq_1 * rq_1 * rq_1 * cq_1;
                float nkx_1 = -kx_1;
                if (row_begin + (it0 + 1) * 8 + rg < row_end) {
                    float out_2[8];
                    float g0_8 = __uint_as_float(gw[8] << 16);
                    float g1_8 = __uint_as_float(gw[8] & 4294901760u);
                    float x0_8 = __uint_as_float(xw[8] << 16);
                    float x1_8 = __uint_as_float(xw[8] & 4294901760u);
                    acc[0] = __fmaf_rn(g0_8 * x0_8, rq_1, acc[0]);
                    acc[1] = __fmaf_rn(g1_8 * x1_8, rq_1, acc[1]);
                    float y0_8 = __fmaf_rn(nkx_1, x0_8, rq_1 * (g0_8 * __uint_as_float(ww[0] << 16)));
                    float y1_8 = __fmaf_rn(nkx_1, x1_8, rq_1 * (g1_8 * __uint_as_float(ww[0] & 4294901760u)));
                    out_2[0] = y0_8;
                    out_2[1] = y1_8;
                    float g0_9 = __uint_as_float(gw[9] << 16);
                    float g1_9 = __uint_as_float(gw[9] & 4294901760u);
                    float x0_9 = __uint_as_float(xw[9] << 16);
                    float x1_9 = __uint_as_float(xw[9] & 4294901760u);
                    acc[2] = __fmaf_rn(g0_9 * x0_9, rq_1, acc[2]);
                    acc[3] = __fmaf_rn(g1_9 * x1_9, rq_1, acc[3]);
                    float y0_9 = __fmaf_rn(nkx_1, x0_9, rq_1 * (g0_9 * __uint_as_float(ww[1] << 16)));
                    float y1_9 = __fmaf_rn(nkx_1, x1_9, rq_1 * (g1_9 * __uint_as_float(ww[1] & 4294901760u)));
                    out_2[2] = y0_9;
                    out_2[3] = y1_9;
                    float g0_10 = __uint_as_float(gw[10] << 16);
                    float g1_10 = __uint_as_float(gw[10] & 4294901760u);
                    float x0_10 = __uint_as_float(xw[10] << 16);
                    float x1_10 = __uint_as_float(xw[10] & 4294901760u);
                    acc[4] = __fmaf_rn(g0_10 * x0_10, rq_1, acc[4]);
                    acc[5] = __fmaf_rn(g1_10 * x1_10, rq_1, acc[5]);
                    float y0_10 = __fmaf_rn(nkx_1, x0_10, rq_1 * (g0_10 * __uint_as_float(ww[2] << 16)));
                    float y1_10 = __fmaf_rn(nkx_1, x1_10, rq_1 * (g1_10 * __uint_as_float(ww[2] & 4294901760u)));
                    out_2[4] = y0_10;
                    out_2[5] = y1_10;
                    float g0_11 = __uint_as_float(gw[11] << 16);
                    float g1_11 = __uint_as_float(gw[11] & 4294901760u);
                    float x0_11 = __uint_as_float(xw[11] << 16);
                    float x1_11 = __uint_as_float(xw[11] & 4294901760u);
                    acc[6] = __fmaf_rn(g0_11 * x0_11, rq_1, acc[6]);
                    acc[7] = __fmaf_rn(g1_11 * x1_11, rq_1, acc[7]);
                    float y0_11 = __fmaf_rn(nkx_1, x0_11, rq_1 * (g0_11 * __uint_as_float(ww[3] << 16)));
                    float y1_11 = __fmaf_rn(nkx_1, x1_11, rq_1 * (g1_11 * __uint_as_float(ww[3] & 4294901760u)));
                    out_2[6] = y0_11;
                    out_2[7] = y1_11;
                    unsigned int packed_2[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_2[_lp*2 + 0], out_2[_lp*2+1 + 0]));
                        packed_2[_lp] = *(uint32_t*)&_bf2;
                    }
                    reinterpret_cast<int4*>(dx + ((unsigned long long)(row_begin + (it0 + 1) * 8 + rg) * 512 + (unsigned long long)(tir * 8)))[0] = reinterpret_cast<int4*>(packed_2)[0];
                    float out_3[8];
                    float g0_12 = __uint_as_float(gw[12] << 16);
                    float g1_12 = __uint_as_float(gw[12] & 4294901760u);
                    float x0_12 = __uint_as_float(xw[12] << 16);
                    float x1_12 = __uint_as_float(xw[12] & 4294901760u);
                    acc[8] = __fmaf_rn(g0_12 * x0_12, rq_1, acc[8]);
                    acc[9] = __fmaf_rn(g1_12 * x1_12, rq_1, acc[9]);
                    float y0_12 = __fmaf_rn(nkx_1, x0_12, rq_1 * (g0_12 * __uint_as_float(ww[4] << 16)));
                    float y1_12 = __fmaf_rn(nkx_1, x1_12, rq_1 * (g1_12 * __uint_as_float(ww[4] & 4294901760u)));
                    out_3[0] = y0_12;
                    out_3[1] = y1_12;
                    float g0_13 = __uint_as_float(gw[13] << 16);
                    float g1_13 = __uint_as_float(gw[13] & 4294901760u);
                    float x0_13 = __uint_as_float(xw[13] << 16);
                    float x1_13 = __uint_as_float(xw[13] & 4294901760u);
                    acc[10] = __fmaf_rn(g0_13 * x0_13, rq_1, acc[10]);
                    acc[11] = __fmaf_rn(g1_13 * x1_13, rq_1, acc[11]);
                    float y0_13 = __fmaf_rn(nkx_1, x0_13, rq_1 * (g0_13 * __uint_as_float(ww[5] << 16)));
                    float y1_13 = __fmaf_rn(nkx_1, x1_13, rq_1 * (g1_13 * __uint_as_float(ww[5] & 4294901760u)));
                    out_3[2] = y0_13;
                    out_3[3] = y1_13;
                    float g0_14 = __uint_as_float(gw[14] << 16);
                    float g1_14 = __uint_as_float(gw[14] & 4294901760u);
                    float x0_14 = __uint_as_float(xw[14] << 16);
                    float x1_14 = __uint_as_float(xw[14] & 4294901760u);
                    acc[12] = __fmaf_rn(g0_14 * x0_14, rq_1, acc[12]);
                    acc[13] = __fmaf_rn(g1_14 * x1_14, rq_1, acc[13]);
                    float y0_14 = __fmaf_rn(nkx_1, x0_14, rq_1 * (g0_14 * __uint_as_float(ww[6] << 16)));
                    float y1_14 = __fmaf_rn(nkx_1, x1_14, rq_1 * (g1_14 * __uint_as_float(ww[6] & 4294901760u)));
                    out_3[4] = y0_14;
                    out_3[5] = y1_14;
                    float g0_15 = __uint_as_float(gw[15] << 16);
                    float g1_15 = __uint_as_float(gw[15] & 4294901760u);
                    float x0_15 = __uint_as_float(xw[15] << 16);
                    float x1_15 = __uint_as_float(xw[15] & 4294901760u);
                    acc[14] = __fmaf_rn(g0_15 * x0_15, rq_1, acc[14]);
                    acc[15] = __fmaf_rn(g1_15 * x1_15, rq_1, acc[15]);
                    float y0_15 = __fmaf_rn(nkx_1, x0_15, rq_1 * (g0_15 * __uint_as_float(ww[7] << 16)));
                    float y1_15 = __fmaf_rn(nkx_1, x1_15, rq_1 * (g1_15 * __uint_as_float(ww[7] & 4294901760u)));
                    out_3[6] = y0_15;
                    out_3[7] = y1_15;
                    unsigned int packed_3[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_3[_lp*2 + 0], out_3[_lp*2+1 + 0]));
                        packed_3[_lp] = *(uint32_t*)&_bf2;
                    }
                    reinterpret_cast<int4*>(dx + ((unsigned long long)(row_begin + (it0 + 1) * 8 + rg) * 512 + (unsigned long long)((32 + tir) * 8)))[0] = reinterpret_cast<int4*>(packed_3)[0];
                }
            }
            window = window + 1;
            if (window == 8) {
                sums[rg * 512 + tir] = sums[rg * 512 + tir] + (double)acc[0];
                acc[0] = 0.0f;
                sums[rg * 512 + 32 + tir] = sums[rg * 512 + 32 + tir] + (double)acc[1];
                acc[1] = 0.0f;
                sums[rg * 512 + 64 + tir] = sums[rg * 512 + 64 + tir] + (double)acc[2];
                acc[2] = 0.0f;
                sums[rg * 512 + 96 + tir] = sums[rg * 512 + 96 + tir] + (double)acc[3];
                acc[3] = 0.0f;
                sums[rg * 512 + 128 + tir] = sums[rg * 512 + 128 + tir] + (double)acc[4];
                acc[4] = 0.0f;
                sums[rg * 512 + 160 + tir] = sums[rg * 512 + 160 + tir] + (double)acc[5];
                acc[5] = 0.0f;
                sums[rg * 512 + 192 + tir] = sums[rg * 512 + 192 + tir] + (double)acc[6];
                acc[6] = 0.0f;
                sums[rg * 512 + 224 + tir] = sums[rg * 512 + 224 + tir] + (double)acc[7];
                acc[7] = 0.0f;
                sums[rg * 512 + 256 + tir] = sums[rg * 512 + 256 + tir] + (double)acc[8];
                acc[8] = 0.0f;
                sums[rg * 512 + 288 + tir] = sums[rg * 512 + 288 + tir] + (double)acc[9];
                acc[9] = 0.0f;
                sums[rg * 512 + 320 + tir] = sums[rg * 512 + 320 + tir] + (double)acc[10];
                acc[10] = 0.0f;
                sums[rg * 512 + 352 + tir] = sums[rg * 512 + 352 + tir] + (double)acc[11];
                acc[11] = 0.0f;
                sums[rg * 512 + 384 + tir] = sums[rg * 512 + 384 + tir] + (double)acc[12];
                acc[12] = 0.0f;
                sums[rg * 512 + 416 + tir] = sums[rg * 512 + 416 + tir] + (double)acc[13];
                acc[13] = 0.0f;
                sums[rg * 512 + 448 + tir] = sums[rg * 512 + 448 + tir] + (double)acc[14];
                acc[14] = 0.0f;
                sums[rg * 512 + 480 + tir] = sums[rg * 512 + 480 + tir] + (double)acc[15];
                acc[15] = 0.0f;
                window = 0;
            }
        }
        if (tid == 0) {
            flags[claim_it & 1] = (int)nxt;
        }
        __syncthreads();
        c = flags[claim_it & 1];
    }
    sums[rg * 512 + tir] = sums[rg * 512 + tir] + (double)acc[0];
    acc[0] = 0.0f;
    sums[rg * 512 + 32 + tir] = sums[rg * 512 + 32 + tir] + (double)acc[1];
    acc[1] = 0.0f;
    sums[rg * 512 + 64 + tir] = sums[rg * 512 + 64 + tir] + (double)acc[2];
    acc[2] = 0.0f;
    sums[rg * 512 + 96 + tir] = sums[rg * 512 + 96 + tir] + (double)acc[3];
    acc[3] = 0.0f;
    sums[rg * 512 + 128 + tir] = sums[rg * 512 + 128 + tir] + (double)acc[4];
    acc[4] = 0.0f;
    sums[rg * 512 + 160 + tir] = sums[rg * 512 + 160 + tir] + (double)acc[5];
    acc[5] = 0.0f;
    sums[rg * 512 + 192 + tir] = sums[rg * 512 + 192 + tir] + (double)acc[6];
    acc[6] = 0.0f;
    sums[rg * 512 + 224 + tir] = sums[rg * 512 + 224 + tir] + (double)acc[7];
    acc[7] = 0.0f;
    sums[rg * 512 + 256 + tir] = sums[rg * 512 + 256 + tir] + (double)acc[8];
    acc[8] = 0.0f;
    sums[rg * 512 + 288 + tir] = sums[rg * 512 + 288 + tir] + (double)acc[9];
    acc[9] = 0.0f;
    sums[rg * 512 + 320 + tir] = sums[rg * 512 + 320 + tir] + (double)acc[10];
    acc[10] = 0.0f;
    sums[rg * 512 + 352 + tir] = sums[rg * 512 + 352 + tir] + (double)acc[11];
    acc[11] = 0.0f;
    sums[rg * 512 + 384 + tir] = sums[rg * 512 + 384 + tir] + (double)acc[12];
    acc[12] = 0.0f;
    sums[rg * 512 + 416 + tir] = sums[rg * 512 + 416 + tir] + (double)acc[13];
    acc[13] = 0.0f;
    sums[rg * 512 + 448 + tir] = sums[rg * 512 + 448 + tir] + (double)acc[14];
    acc[14] = 0.0f;
    sums[rg * 512 + 480 + tir] = sums[rg * 512 + 480 + tir] + (double)acc[15];
    acc[15] = 0.0f;
    __syncthreads();
    int at_0 = tid;
    sums[at_0] = sums[at_0] + sums[512 + at_0];
    sums[at_0] = sums[at_0] + sums[1024 + at_0];
    sums[at_0] = sums[at_0] + sums[1536 + at_0];
    sums[at_0] = sums[at_0] + sums[2048 + at_0];
    sums[at_0] = sums[at_0] + sums[2560 + at_0];
    sums[at_0] = sums[at_0] + sums[3072 + at_0];
    sums[at_0] = sums[at_0] + sums[3584 + at_0];
    int at_1 = 256 + tid;
    sums[at_1] = sums[at_1] + sums[512 + at_1];
    sums[at_1] = sums[at_1] + sums[1024 + at_1];
    sums[at_1] = sums[at_1] + sums[1536 + at_1];
    sums[at_1] = sums[at_1] + sums[2048 + at_1];
    sums[at_1] = sums[at_1] + sums[2560 + at_1];
    sums[at_1] = sums[at_1] + sums[3072 + at_1];
    sums[at_1] = sums[at_1] + sums[3584 + at_1];
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    __syncthreads();
    if (tid == 0) {
        if (first_c < n_chunks) {
            {
                void* _cpred_dst_29 = reinterpret_cast<void*>(partial + (long long)((int)bid % 4 * 512));
                asm volatile("cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f64"
                    " [%0], [%1], %2;"
                    :: "l"(_cpred_dst_29), "r"(sums_addr + 0), "r"((uint32_t)(1024))
                    : "memory");
            }
            {
                void* _cpred_dst_30 = reinterpret_cast<void*>(partial + (long long)((int)bid % 4 * 512 + 128));
                asm volatile("cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f64"
                    " [%0], [%1], %2;"
                    :: "l"(_cpred_dst_30), "r"(sums_addr + 1024), "r"((uint32_t)(1024))
                    : "memory");
            }
            {
                void* _cpred_dst_31 = reinterpret_cast<void*>(partial + (long long)((int)bid % 4 * 512 + 256));
                asm volatile("cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f64"
                    " [%0], [%1], %2;"
                    :: "l"(_cpred_dst_31), "r"(sums_addr + 2048), "r"((uint32_t)(1024))
                    : "memory");
            }
            {
                void* _cpred_dst_32 = reinterpret_cast<void*>(partial + (long long)((int)bid % 4 * 512 + 384));
                asm volatile("cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f64"
                    " [%0], [%1], %2;"
                    :: "l"(_cpred_dst_32), "r"(sums_addr + 3072), "r"((uint32_t)(1024))
                    : "memory");
            }
            asm volatile("cp.async.bulk.commit_group;");
            asm volatile("cp.async.bulk.wait_group 0;");
            asm volatile("fence.proxy.async.global;" ::: "memory");
        }
        unsigned int old;
        asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], %2;"
            : "=r"(old) : "l"(&counters[1]), "r"(static_cast<uint32_t>(1)) : "memory");
        int last = 0;
        if (old + 1 == (unsigned int)num_bids) {
            *(reinterpret_cast<unsigned int*>(counters) + (1)) = 0;
            last = 1;
        }
        flags[2] = last;
        if (c == n_chunks + (int)gridDim.x - 1) {
            *(reinterpret_cast<unsigned int*>(counters) + (0)) = 0;
        }
    }
    __syncthreads();
    if (flags[2] != 0) {
        double buf[16];
        {
            uint64_t _scalar_bits_33;
            asm volatile("ld.global.cg.b64 %0, [%1];"
                : "=l"(_scalar_bits_33) : "l"((const void*)(partial + (tid))) : "memory");
            buf[0] = __longlong_as_double(_scalar_bits_33);
        }
        {
            uint64_t _scalar_bits_34;
            asm volatile("ld.global.cg.b64 %0, [%1];"
                : "=l"(_scalar_bits_34) : "l"((const void*)(partial + (512 + tid))) : "memory");
            buf[1] = __longlong_as_double(_scalar_bits_34);
        }
        {
            uint64_t _scalar_bits_35;
            asm volatile("ld.global.cg.b64 %0, [%1];"
                : "=l"(_scalar_bits_35) : "l"((const void*)(partial + (1024 + tid))) : "memory");
            buf[2] = __longlong_as_double(_scalar_bits_35);
        }
        {
            uint64_t _scalar_bits_36;
            asm volatile("ld.global.cg.b64 %0, [%1];"
                : "=l"(_scalar_bits_36) : "l"((const void*)(partial + (1536 + tid))) : "memory");
            buf[3] = __longlong_as_double(_scalar_bits_36);
        }
        {
            uint64_t _scalar_bits_37;
            asm volatile("ld.global.cg.b64 %0, [%1];"
                : "=l"(_scalar_bits_37) : "l"((const void*)(partial + (256 + tid))) : "memory");
            buf[4] = __longlong_as_double(_scalar_bits_37);
        }
        {
            uint64_t _scalar_bits_38;
            asm volatile("ld.global.cg.b64 %0, [%1];"
                : "=l"(_scalar_bits_38) : "l"((const void*)(partial + (768 + tid))) : "memory");
            buf[5] = __longlong_as_double(_scalar_bits_38);
        }
        {
            uint64_t _scalar_bits_39;
            asm volatile("ld.global.cg.b64 %0, [%1];"
                : "=l"(_scalar_bits_39) : "l"((const void*)(partial + (1280 + tid))) : "memory");
            buf[6] = __longlong_as_double(_scalar_bits_39);
        }
        {
            uint64_t _scalar_bits_40;
            asm volatile("ld.global.cg.b64 %0, [%1];"
                : "=l"(_scalar_bits_40) : "l"((const void*)(partial + (1792 + tid))) : "memory");
            buf[7] = __longlong_as_double(_scalar_bits_40);
        }
        double tot_0 = buf[0];
        tot_0 = tot_0 + buf[1];
        tot_0 = tot_0 + buf[2];
        tot_0 = tot_0 + buf[3];
        int at_w_0 = tid;
        dwbuf[(at_w_0 / 32 / 8 * 32 + at_w_0 % 32) * 8 + at_w_0 / 32 % 8] = (float)tot_0;
        *reinterpret_cast<unsigned long long*>(partial + (at_w_0)) = (unsigned long long)((unsigned long long)(0.0));
        *reinterpret_cast<unsigned long long*>(partial + (512 + at_w_0)) = (unsigned long long)((unsigned long long)(0.0));
        *reinterpret_cast<unsigned long long*>(partial + (1024 + at_w_0)) = (unsigned long long)((unsigned long long)(0.0));
        *reinterpret_cast<unsigned long long*>(partial + (1536 + at_w_0)) = (unsigned long long)((unsigned long long)(0.0));
        double tot_1 = buf[4];
        tot_1 = tot_1 + buf[5];
        tot_1 = tot_1 + buf[6];
        tot_1 = tot_1 + buf[7];
        int at_w_1 = 256 + tid;
        dwbuf[(at_w_1 / 32 / 8 * 32 + at_w_1 % 32) * 8 + at_w_1 / 32 % 8] = (float)tot_1;
        *reinterpret_cast<unsigned long long*>(partial + (at_w_1)) = (unsigned long long)((unsigned long long)(0.0));
        *reinterpret_cast<unsigned long long*>(partial + (512 + at_w_1)) = (unsigned long long)((unsigned long long)(0.0));
        *reinterpret_cast<unsigned long long*>(partial + (1024 + at_w_1)) = (unsigned long long)((unsigned long long)(0.0));
        *reinterpret_cast<unsigned long long*>(partial + (1536 + at_w_1)) = (unsigned long long)((unsigned long long)(0.0));
        __syncthreads();
        int col_0 = tid * 4;
        if (col_0 < 512) {
            float o4_0[4];
            o4_0[0] = dwbuf[col_0];
            o4_0[1] = dwbuf[col_0 + 1];
            o4_0[2] = dwbuf[col_0 + 2];
            o4_0[3] = dwbuf[col_0 + 3];
            {
                float4 _v4 = make_float4(o4_0[0 + 0], o4_0[0 + 1], o4_0[0 + 2], o4_0[0 + 3]);
                *reinterpret_cast<float4*>(dw + col_0) = _v4;
            }
        }
    }
}

} // extern "C"
