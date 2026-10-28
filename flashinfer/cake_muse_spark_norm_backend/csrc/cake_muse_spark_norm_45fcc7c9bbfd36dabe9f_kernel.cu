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
#define NUM_ROWS_STAGES 3
#define SMEM_PARTIALS_OFF 1024
#define SMEM_PARTIALS_STAGE_BYTES 256
#define SMEM_PARTIALS_STRIDE 256
#define SMEM_RVIEW_OFF 1344
#define SMEM_RVIEW_STAGE_BYTES 32768
#define SMEM_RVIEW_STRIDE 32768
#define SMEM_TOTAL 99712
#define THREADS 1024
#define SPLIT 1
#define SEG 8192
#define LAUNCH_BOUNDS_THREADS 1024

#include <math_constants.h>

__device__ __forceinline__ uint32_t elect_sync() {
    uint32_t pred = 0;
    asm volatile(
        "{\n\t"
        ".reg .pred %%px;\n\t"
        "elect.sync _|%%px, %1;\n\t"
        "@%%px mov.s32 %0, 1;\n\t"
        "}\n"
        : "+r"(pred)
        : "r"(0xFFFFFFFF));
    return pred;
}


__device__ __forceinline__ void mbarrier_init(int mbar_addr, int count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;"
        :: "r"(mbar_addr), "r"(count) : "memory");
}



// CTA-local pipelines have short, resident producer/consumer edges.  Omitting
// suspendTimeHint keeps a miss on the lightweight TRYWAIT retry path; the
// explicit loop still makes this helper blocking until acquire succeeds.
__device__ __forceinline__ void mbarrier_wait(int mbar_addr, int phase) {
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "LAB_WAIT:\n\t"
        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64"
        " P1, [%0], %1;\n\t"
        "@P1 bra.uni DONE;\n\t"
        "bra.uni LAB_WAIT;\n\t"
        "DONE:\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"(phase) : "memory");
}


__device__ __forceinline__ void mbarrier_arrive(int mbar_addr) {
    asm volatile(
        "mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];"
        :: "r"(mbar_addr) : "memory");
}


__device__ __forceinline__ void mbarrier_arrive_expect_tx(int mbar_addr, uint32_t bytes) {
    asm volatile(
        "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
        :: "r"(mbar_addr), "r"(bytes) : "memory");
}




__device__ __forceinline__ __nv_bfloat162 __as_bf16x2(unsigned int v) {
    __nv_bfloat162_raw raw;
    raw.x = static_cast<unsigned short>(v);
    raw.y = static_cast<unsigned short>(v >> 16);
    return __nv_bfloat162(raw);
}

__device__ __forceinline__ float2 cake_bf16x2_to_f32x2(__nv_bfloat162 v) {
    const unsigned int w = *reinterpret_cast<const unsigned int*>(&v);
    return make_float2(__uint_as_float(w << 16), __uint_as_float(w & 0xffff0000u));
}

extern "C" {

__global__ __launch_bounds__(LAUNCH_BOUNDS_THREADS) void
kernel_cake_muse_spark_norm_45fcc7c9bbfd36dabe9f(__nv_bfloat16* __restrict__ branch, float* __restrict__ residual, __nv_bfloat16* __restrict__ weight, float* __restrict__ alpha, float* __restrict__ beta, float* __restrict__ out, int M, float eps, float gain_center, long long branch_stride_p0, long long residual_stride_p0, long long out_stride_p0)
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

    const int mbar_base = smem;
    #define rows_full_addr (mbar_base + 0)
    #define rows_empty_addr (mbar_base + 24)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    float* partials = reinterpret_cast<float*>(smem_raw + 1024);
    const int partials_addr = smem + 1024;
    float* rview = reinterpret_cast<float*>(smem_raw + 1344);
    const int rview_addr = smem + 1344;

    // Mbarrier init (2 pipeline groups, 0 ordered-sequence groups, 6 barriers)
    // Mbarriers at smem_raw[0..48)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'rows' ---
            // rows_full: 3 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            mbarrier_init(smem + 8, 1);
            mbarrier_init(smem + 16, 1);
            // rows_empty: 3 barriers, init_count=1024
            mbarrier_init(smem + 24, 1024);
            mbarrier_init(smem + 32, 1024);
            mbarrier_init(smem + 40, 1024);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncthreads();

    // === Task calls (declaration order) ===
    int t = tid;
    int warp_0 = warp;
    int seg = 0;
    int first = bid;
    int row_step = num_bids;
    int lo = seg * SEG;
    int hi = lo + SEG;
    int cbase = 0;
    int it = 0;
    unsigned int cur[4];
    unsigned int nxt[4];
    int krow = 0;
    float x[8];
    float al[8];
    float be[8];
    float res[8];
    float y[8];
    long long fbase = (long long)first * branch_stride_p0;
    #pragma unroll
    for (int c = 0; c < 1; c++) {
        {
            unsigned _ldv8_0_0;
            unsigned _ldv8_0_1;
            unsigned _ldv8_0_2;
            unsigned _ldv8_0_3;
            unsigned _ldv8_0_4;
            unsigned _ldv8_0_5;
            unsigned _ldv8_0_6;
            unsigned _ldv8_0_7;
            asm volatile("ld.global.nc.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                : "=r"(_ldv8_0_0), "=r"(_ldv8_0_1), "=r"(_ldv8_0_2), "=r"(_ldv8_0_3), "=r"(_ldv8_0_4), "=r"(_ldv8_0_5), "=r"(_ldv8_0_6), "=r"(_ldv8_0_7) : "l"((const void*)(alpha + (cbase + c * 8192 + 8 * t) + (0))) : "memory");
            al[c * 8 + 0] = __uint_as_float(_ldv8_0_0);
            al[c * 8 + 1] = __uint_as_float(_ldv8_0_1);
            al[c * 8 + 2] = __uint_as_float(_ldv8_0_2);
            al[c * 8 + 3] = __uint_as_float(_ldv8_0_3);
            al[c * 8 + 4] = __uint_as_float(_ldv8_0_4);
            al[c * 8 + 5] = __uint_as_float(_ldv8_0_5);
            al[c * 8 + 6] = __uint_as_float(_ldv8_0_6);
            al[c * 8 + 7] = __uint_as_float(_ldv8_0_7);
        }
        {
            unsigned _ldv8_1_0;
            unsigned _ldv8_1_1;
            unsigned _ldv8_1_2;
            unsigned _ldv8_1_3;
            unsigned _ldv8_1_4;
            unsigned _ldv8_1_5;
            unsigned _ldv8_1_6;
            unsigned _ldv8_1_7;
            asm volatile("ld.global.nc.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                : "=r"(_ldv8_1_0), "=r"(_ldv8_1_1), "=r"(_ldv8_1_2), "=r"(_ldv8_1_3), "=r"(_ldv8_1_4), "=r"(_ldv8_1_5), "=r"(_ldv8_1_6), "=r"(_ldv8_1_7) : "l"((const void*)(beta + (cbase + c * 8192 + 8 * t) + (0))) : "memory");
            be[c * 8 + 0] = __uint_as_float(_ldv8_1_0);
            be[c * 8 + 1] = __uint_as_float(_ldv8_1_1);
            be[c * 8 + 2] = __uint_as_float(_ldv8_1_2);
            be[c * 8 + 3] = __uint_as_float(_ldv8_1_3);
            be[c * 8 + 4] = __uint_as_float(_ldv8_1_4);
            be[c * 8 + 5] = __uint_as_float(_ldv8_1_5);
            be[c * 8 + 6] = __uint_as_float(_ldv8_1_6);
            be[c * 8 + 7] = __uint_as_float(_ldv8_1_7);
        }
    }
    {
        const uint4* _vptr_2 = reinterpret_cast<const uint4*>(branch + (fbase + (long long)(cbase + 8 * t)) + 0);
        uint4* _vdst_2 = reinterpret_cast<uint4*>(&cur[0]);
        #pragma unroll
        for (int _blk = 0; _blk < 1; _blk++) {
            asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                : "=r"(_vdst_2[_blk].x), "=r"(_vdst_2[_blk].y), "=r"(_vdst_2[_blk].z), "=r"(_vdst_2[_blk].w) : "l"((const void*)(_vptr_2 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
        }
    }
    unsigned int sr_off = (unsigned int)((cbase + 8 * t) * 4);
    if (warp == 0) {
        if (elect_sync()) {
            int prow_2_0 = first;
            if (prow_2_0 < M) {
                long long nr_1 = (long long)prow_2_0 * residual_stride_p0;
                mbarrier_arrive_expect_tx(rows_full_addr, 32768);
                asm volatile(
                    "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes "
                    "[%0], [%1], %2, [%3];"
                    :: "r"(rview_addr), "l"(residual + nr_1), "r"((uint32_t)(32768)), "r"(rows_full_addr)
                    : "memory");
            }
            int prow_2_1 = first + row_step;
            if (prow_2_1 < M) {
                long long nr_1_1 = (long long)prow_2_1 * residual_stride_p0;
                mbarrier_arrive_expect_tx(rows_full_addr + 8, 32768);
                asm volatile(
                    "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes "
                    "[%0], [%1], %2, [%3];"
                    :: "r"(rview_addr + 32768), "l"(residual + nr_1_1), "r"((uint32_t)(32768)), "r"(rows_full_addr + 8)
                    : "memory");
            }
            int prow_2_2 = first + 2 * row_step;
            if (prow_2_2 < M) {
                long long nr_1_2 = (long long)prow_2_2 * residual_stride_p0;
                mbarrier_arrive_expect_tx(rows_full_addr + 16, 32768);
                asm volatile(
                    "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes "
                    "[%0], [%1], %2, [%3];"
                    :: "r"(rview_addr + 65536), "l"(residual + nr_1_2), "r"((uint32_t)(32768)), "r"(rows_full_addr + 16)
                    : "memory");
            }
        }
    }
    #pragma unroll 1
    for (int row = first; row < M; row += row_step) {
        long long row64 = (long long)row;
        long long sbase = row64 * residual_stride_p0;
        long long obase = row64 * out_stride_p0;
        int par = (it & 1) * 32;
        int _stage_rows_full = (int)((unsigned int)krow % 3);
        int _parity_rows_full = (int)((unsigned int)krow / 3 % 2);
        mbarrier_wait(rows_full_addr + (_stage_rows_full) * 8, _parity_rows_full);
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&res[0])), "=r"(*reinterpret_cast<uint32_t*>(&res[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&res[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&res[(0) + 3]))
            : "r"(rview_addr + (unsigned int)(krow % 3 * 32768) + sr_off));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&res[4])), "=r"(*reinterpret_cast<uint32_t*>(&res[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&res[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&res[(4) + 3]))
            : "r"(rview_addr + (unsigned int)(krow % 3 * 32768) + sr_off + 16));
        mbarrier_arrive(rows_empty_addr + ((int)((unsigned int)krow % 3)) * 8);
        int nrow = row + row_step;
        if (nrow < M) {
            long long nbase = (long long)nrow * branch_stride_p0;
            #pragma unroll
            for (int c_1 = 0; c_1 < 1; c_1++) {
                {
                    const uint4* _vptr_3 = reinterpret_cast<const uint4*>(branch + (nbase + (long long)(cbase + c_1 * 8192 + 8 * t)) + 0);
                    uint4* _vdst_3 = reinterpret_cast<uint4*>(&nxt[c_1 * 4]);
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                            : "=r"(_vdst_3[_blk].x), "=r"(_vdst_3[_blk].y), "=r"(_vdst_3[_blk].z), "=r"(_vdst_3[_blk].w) : "l"((const void*)(_vptr_3 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                    }
                }
            }
        }
        float acc = 0.0f;
        float2 _cvt_f32_0 = cake_bf16x2_to_f32x2(__as_bf16x2(cur[0]));
        float x0_0 = _cvt_f32_0.x;
        float x1_0 = _cvt_f32_0.y;
        x[0] = x0_0;
        x[1] = x1_0;
        acc = __fmaf_rn(x0_0, x0_0, acc);
        acc = __fmaf_rn(x1_0, x1_0, acc);
        float2 _cvt_f32_1 = cake_bf16x2_to_f32x2(__as_bf16x2(cur[1]));
        float x0_1 = _cvt_f32_1.x;
        float x1_1 = _cvt_f32_1.y;
        x[2] = x0_1;
        x[3] = x1_1;
        acc = __fmaf_rn(x0_1, x0_1, acc);
        acc = __fmaf_rn(x1_1, x1_1, acc);
        float2 _cvt_f32_2 = cake_bf16x2_to_f32x2(__as_bf16x2(cur[2]));
        float x0_2 = _cvt_f32_2.x;
        float x1_2 = _cvt_f32_2.y;
        x[4] = x0_2;
        x[5] = x1_2;
        acc = __fmaf_rn(x0_2, x0_2, acc);
        acc = __fmaf_rn(x1_2, x1_2, acc);
        float2 _cvt_f32_3 = cake_bf16x2_to_f32x2(__as_bf16x2(cur[3]));
        float x0_3 = _cvt_f32_3.x;
        float x1_3 = _cvt_f32_3.y;
        x[6] = x0_3;
        x[7] = x1_3;
        acc = __fmaf_rn(x0_3, x0_3, acc);
        acc = __fmaf_rn(x1_3, x1_3, acc);
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            float pre = al[i] * res[i];
            res[i] = pre;
        }
        float _warp_reduce_4 = acc;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            _warp_reduce_4 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_4, offset);
        acc = _warp_reduce_4;
        if (lane == 0) {
            partials[par + warp_0] = acc;
        }
        __syncthreads();
        float total = partials[par];
        #pragma unroll
        for (int k = 1; k < 32; k++) {
            total += partials[par + k];
        }
        it += 1;
        int nrow3 = row + 3 * row_step;
        if (nrow3 < M) {
            if (warp == 0) {
                if (elect_sync()) {
                    int _stage_rows_empty = (int)((unsigned int)(krow + 3) % 3);
                    int _parity_rows_empty = (int)((unsigned int)(krow + 3) / 3 % 2) ^ 1;
                    mbarrier_wait(rows_empty_addr + (_stage_rows_empty) * 8, _parity_rows_empty);
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    long long nr_1_3 = (long long)nrow3 * residual_stride_p0;
                    mbarrier_arrive_expect_tx(rows_full_addr + ((int)((unsigned int)(krow + 3) % 3)) * 8, 32768);
                    asm volatile(
                        "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes "
                        "[%0], [%1], %2, [%3];"
                        :: "r"(rview_addr + (unsigned int)((krow + 3) % 3 * 32768)), "l"(residual + nr_1_3), "r"((uint32_t)(32768)), "r"(rows_full_addr + ((int)((unsigned int)(krow + 3) % 3)) * 8)
                        : "memory");
                }
            }
        }
        float mean = __fmaf_rn(total, 0.00012207031f, eps);
        float _sqrt_5;
        asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(_sqrt_5) : "f"(mean));
        float inv = __frcp_rn(_sqrt_5);
        #pragma unroll
        for (int c_2 = 0; c_2 < 1; c_2++) {
            float yv0_1_0 = __fmaf_rn(be[c_2 * 8], x[c_2 * 8] * inv, res[c_2 * 8]);
            y[c_2 * 8] = yv0_1_0;
            float yv0_1_1 = __fmaf_rn(be[c_2 * 8 + 1], x[c_2 * 8 + 1] * inv, res[c_2 * 8 + 1]);
            y[c_2 * 8 + 1] = yv0_1_1;
            float yv0_1_2 = __fmaf_rn(be[c_2 * 8 + 2], x[c_2 * 8 + 2] * inv, res[c_2 * 8 + 2]);
            y[c_2 * 8 + 2] = yv0_1_2;
            float yv0_1_3 = __fmaf_rn(be[c_2 * 8 + 3], x[c_2 * 8 + 3] * inv, res[c_2 * 8 + 3]);
            y[c_2 * 8 + 3] = yv0_1_3;
            float yv0_1_4 = __fmaf_rn(be[c_2 * 8 + 4], x[c_2 * 8 + 4] * inv, res[c_2 * 8 + 4]);
            y[c_2 * 8 + 4] = yv0_1_4;
            float yv0_1_5 = __fmaf_rn(be[c_2 * 8 + 5], x[c_2 * 8 + 5] * inv, res[c_2 * 8 + 5]);
            y[c_2 * 8 + 5] = yv0_1_5;
            float yv0_1_6 = __fmaf_rn(be[c_2 * 8 + 6], x[c_2 * 8 + 6] * inv, res[c_2 * 8 + 6]);
            y[c_2 * 8 + 6] = yv0_1_6;
            float yv0_1_7 = __fmaf_rn(be[c_2 * 8 + 7], x[c_2 * 8 + 7] * inv, res[c_2 * 8 + 7]);
            y[c_2 * 8 + 7] = yv0_1_7;
            {
                unsigned _stv8_6_0 = __float_as_uint(y[c_2 * 8 + 0]);
                unsigned _stv8_6_1 = __float_as_uint(y[c_2 * 8 + 1]);
                unsigned _stv8_6_2 = __float_as_uint(y[c_2 * 8 + 2]);
                unsigned _stv8_6_3 = __float_as_uint(y[c_2 * 8 + 3]);
                unsigned _stv8_6_4 = __float_as_uint(y[c_2 * 8 + 4]);
                unsigned _stv8_6_5 = __float_as_uint(y[c_2 * 8 + 5]);
                unsigned _stv8_6_6 = __float_as_uint(y[c_2 * 8 + 6]);
                unsigned _stv8_6_7 = __float_as_uint(y[c_2 * 8 + 7]);
                asm volatile(
                    "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                    :: "l"((void*)(out + (obase + (long long)(cbase + c_2 * 8192 + 8 * t)) + (0))), "r"(_stv8_6_0), "r"(_stv8_6_1), "r"(_stv8_6_2), "r"(_stv8_6_3), "r"(_stv8_6_4), "r"(_stv8_6_5), "r"(_stv8_6_6), "r"(_stv8_6_7) : "memory");
            }
        }
        krow += 1;
        #pragma unroll
        for (int i_1 = 0; i_1 < 4; i_1++) {
            cur[i_1] = nxt[i_1];
        }
    }

    // Cleanup
    __syncthreads();
}

} // extern "C"
