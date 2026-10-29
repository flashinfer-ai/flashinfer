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
#define NUM_ROWS_STAGES 3
#define SMEM_SLOTS_OFF 1024
#define SMEM_SLOTS_STAGE_BYTES 128
#define SMEM_SLOTS_STRIDE 128
#define SMEM_BVIEW_OFF 2048
#define SMEM_BVIEW_STAGE_BYTES 16384
#define SMEM_BVIEW_STRIDE 49152
#define SMEM_RVIEW_OFF 18432
#define SMEM_RVIEW_STAGE_BYTES 32768
#define SMEM_RVIEW_STRIDE 49152
#define SMEM_TOTAL 149504
#define THREADS 512
#define LAUNCH_BOUNDS_THREADS 512

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


__device__ __forceinline__ float2 add_f32x2_noftz(float2 a, float2 b) {
    float2 r;
    asm("add.rn.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(unsigned long long*)&a), "l"(*(unsigned long long*)&b));
    return r;
}

__device__ __forceinline__ float2 mul_f32x2_noftz(float2 a, float2 b) {
    float2 r;
    asm("mul.rn.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(unsigned long long*)&a), "l"(*(unsigned long long*)&b));
    return r;
}


// ex2_emulation_f32x2 defined in softmax_frag_exp2_cast helper (or standalone)

__device__ __forceinline__ float2 fma_f32x2_rn_noftz(float2 a, float2 b, float2 c) {
    float2 r;
    asm("fma.rn.f32x2 %0, %1, %2, %3;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b),
          "l"(*(const unsigned long long*)&c));
    return r;
}




__device__ __forceinline__ __nv_bfloat162 __as_bf16x2(unsigned int v) {
    __nv_bfloat162_raw raw;
    raw.x = static_cast<unsigned short>(v);
    raw.y = static_cast<unsigned short>(v >> 16);
    return __nv_bfloat162(raw);
}

__device__ __forceinline__ float2 cake_bf16x2_to_f32x2(__nv_bfloat162 v) {
    return __bfloat1622float2(v);
}

extern "C" {

__global__ __launch_bounds__(LAUNCH_BOUNDS_THREADS) void
kernel_cake_muse_spark_norm_84b31104ac86bd40d3fe(const __nv_bfloat16* branch, const float* residual, const float* __restrict__ alpha, const float* __restrict__ beta, const __nv_bfloat16* __restrict__ next_weight, const __nv_bfloat16* __restrict__ post_weight, float* residual_out, __nv_bfloat16* ffn_input, int M, float post_eps, float next_eps, float post_gain_center, float next_gain_center, float alpha_value)
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
    float* slots = reinterpret_cast<float*>(smem_raw + 1024);
    const int slots_addr = smem + 1024;
    unsigned int* bview = reinterpret_cast<unsigned int*>(smem_raw + 2048);
    const int bview_addr = smem + 2048;
    float* rview = reinterpret_cast<float*>(smem_raw + 18432);
    const int rview_addr = smem + 18432;

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
            // rows_empty: 3 barriers, init_count=512
            mbarrier_init(smem + 24, 512);
            mbarrier_init(smem + 32, 512);
            mbarrier_init(smem + 40, 512);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncthreads();

    // === Task calls (declaration order) ===
    int warp_0 = warp;
    int lane_1 = lane;
    int rank = 0;
    int first = bid;
    int stride = num_bids;
    int col0 = (rank * 512 + tid) * 16;
    long long col64 = (long long)col0;
    long long ccol_0 = (long long)tid * 8;
    long long ccol_1 = (long long)tid * 8 + 4096;
    unsigned int sb_off_0 = (unsigned int)tid * 16;
    unsigned int sb_off_1 = (unsigned int)tid * 16 + 8192;
    unsigned int sr_off_0 = (unsigned int)tid * 32;
    unsigned int sr_off_1 = (unsigned int)tid * 32 + 16384;
    float2 pgc2 = make_float2(post_gain_center, post_gain_center);
    float2 ngc2 = make_float2(next_gain_center, next_gain_center);
    float al[16];
    float be[16];
    unsigned int nw[8];
    unsigned int pw[8];
    {
        unsigned _ldv8_0_0;
        unsigned _ldv8_0_1;
        unsigned _ldv8_0_2;
        unsigned _ldv8_0_3;
        unsigned _ldv8_0_4;
        unsigned _ldv8_0_5;
        unsigned _ldv8_0_6;
        unsigned _ldv8_0_7;
        asm volatile(
            "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
            : "=r"(_ldv8_0_0), "=r"(_ldv8_0_1), "=r"(_ldv8_0_2), "=r"(_ldv8_0_3), "=r"(_ldv8_0_4), "=r"(_ldv8_0_5), "=r"(_ldv8_0_6), "=r"(_ldv8_0_7) : "l"((const void*)(alpha + ccol_0 + (0))) : "memory");
        al[0 + 0] = __uint_as_float(_ldv8_0_0);
        al[0 + 1] = __uint_as_float(_ldv8_0_1);
        al[0 + 2] = __uint_as_float(_ldv8_0_2);
        al[0 + 3] = __uint_as_float(_ldv8_0_3);
        al[0 + 4] = __uint_as_float(_ldv8_0_4);
        al[0 + 5] = __uint_as_float(_ldv8_0_5);
        al[0 + 6] = __uint_as_float(_ldv8_0_6);
        al[0 + 7] = __uint_as_float(_ldv8_0_7);
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
        asm volatile(
            "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
            : "=r"(_ldv8_1_0), "=r"(_ldv8_1_1), "=r"(_ldv8_1_2), "=r"(_ldv8_1_3), "=r"(_ldv8_1_4), "=r"(_ldv8_1_5), "=r"(_ldv8_1_6), "=r"(_ldv8_1_7) : "l"((const void*)(beta + ccol_0 + (0))) : "memory");
        be[0 + 0] = __uint_as_float(_ldv8_1_0);
        be[0 + 1] = __uint_as_float(_ldv8_1_1);
        be[0 + 2] = __uint_as_float(_ldv8_1_2);
        be[0 + 3] = __uint_as_float(_ldv8_1_3);
        be[0 + 4] = __uint_as_float(_ldv8_1_4);
        be[0 + 5] = __uint_as_float(_ldv8_1_5);
        be[0 + 6] = __uint_as_float(_ldv8_1_6);
        be[0 + 7] = __uint_as_float(_ldv8_1_7);
    }
    {
        unsigned _ldv8_2_0;
        unsigned _ldv8_2_1;
        unsigned _ldv8_2_2;
        unsigned _ldv8_2_3;
        unsigned _ldv8_2_4;
        unsigned _ldv8_2_5;
        unsigned _ldv8_2_6;
        unsigned _ldv8_2_7;
        asm volatile(
            "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
            : "=r"(_ldv8_2_0), "=r"(_ldv8_2_1), "=r"(_ldv8_2_2), "=r"(_ldv8_2_3), "=r"(_ldv8_2_4), "=r"(_ldv8_2_5), "=r"(_ldv8_2_6), "=r"(_ldv8_2_7) : "l"((const void*)(alpha + ccol_1 + (0))) : "memory");
        al[8 + 0] = __uint_as_float(_ldv8_2_0);
        al[8 + 1] = __uint_as_float(_ldv8_2_1);
        al[8 + 2] = __uint_as_float(_ldv8_2_2);
        al[8 + 3] = __uint_as_float(_ldv8_2_3);
        al[8 + 4] = __uint_as_float(_ldv8_2_4);
        al[8 + 5] = __uint_as_float(_ldv8_2_5);
        al[8 + 6] = __uint_as_float(_ldv8_2_6);
        al[8 + 7] = __uint_as_float(_ldv8_2_7);
    }
    {
        unsigned _ldv8_3_0;
        unsigned _ldv8_3_1;
        unsigned _ldv8_3_2;
        unsigned _ldv8_3_3;
        unsigned _ldv8_3_4;
        unsigned _ldv8_3_5;
        unsigned _ldv8_3_6;
        unsigned _ldv8_3_7;
        asm volatile(
            "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
            : "=r"(_ldv8_3_0), "=r"(_ldv8_3_1), "=r"(_ldv8_3_2), "=r"(_ldv8_3_3), "=r"(_ldv8_3_4), "=r"(_ldv8_3_5), "=r"(_ldv8_3_6), "=r"(_ldv8_3_7) : "l"((const void*)(beta + ccol_1 + (0))) : "memory");
        be[8 + 0] = __uint_as_float(_ldv8_3_0);
        be[8 + 1] = __uint_as_float(_ldv8_3_1);
        be[8 + 2] = __uint_as_float(_ldv8_3_2);
        be[8 + 3] = __uint_as_float(_ldv8_3_3);
        be[8 + 4] = __uint_as_float(_ldv8_3_4);
        be[8 + 5] = __uint_as_float(_ldv8_3_5);
        be[8 + 6] = __uint_as_float(_ldv8_3_6);
        be[8 + 7] = __uint_as_float(_ldv8_3_7);
    }
    {
        const uint4* _vptr_4 = reinterpret_cast<const uint4*>(next_weight + ccol_0 + 0);
        uint4* _vdst_4 = reinterpret_cast<uint4*>(&nw[0]);
        #pragma unroll
        for (int _blk = 0; _blk < 1; _blk++) {
            _vdst_4[_blk] = _vptr_4[_blk];
        }
    }
    {
        const uint4* _vptr_5 = reinterpret_cast<const uint4*>(post_weight + ccol_0 + 0);
        uint4* _vdst_5 = reinterpret_cast<uint4*>(&pw[0]);
        #pragma unroll
        for (int _blk = 0; _blk < 1; _blk++) {
            _vdst_5[_blk] = _vptr_5[_blk];
        }
    }
    {
        const uint4* _vptr_6 = reinterpret_cast<const uint4*>(next_weight + ccol_1 + 0);
        uint4* _vdst_6 = reinterpret_cast<uint4*>(&nw[4]);
        #pragma unroll
        for (int _blk = 0; _blk < 1; _blk++) {
            _vdst_6[_blk] = _vptr_6[_blk];
        }
    }
    {
        const uint4* _vptr_7 = reinterpret_cast<const uint4*>(post_weight + ccol_1 + 0);
        uint4* _vdst_7 = reinterpret_cast<uint4*>(&pw[4]);
        #pragma unroll
        for (int _blk = 0; _blk < 1; _blk++) {
            _vdst_7[_blk] = _vptr_7[_blk];
        }
    }
    unsigned int cur_b[8];
    float cur_r[16];
    int krow = 0;
    if (warp == 0) {
        if (elect_sync()) {
            int prow_2_0 = first;
            if (prow_2_0 < M) {
                long long nb = (long long)prow_2_0 * 8192;
                mbarrier_arrive_expect_tx(rows_full_addr, 49152);
                asm volatile(
                    "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes "
                    "[%0], [%1], %2, [%3];"
                    :: "r"(bview_addr), "l"(branch + nb), "r"((uint32_t)(16384)), "r"(rows_full_addr)
                    : "memory");
                asm volatile(
                    "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes "
                    "[%0], [%1], %2, [%3];"
                    :: "r"(rview_addr), "l"(residual + nb), "r"((uint32_t)(32768)), "r"(rows_full_addr)
                    : "memory");
            }
            int prow_2_1 = first + stride;
            if (prow_2_1 < M) {
                long long nb_1 = (long long)prow_2_1 * 8192;
                mbarrier_arrive_expect_tx(rows_full_addr + 8, 49152);
                asm volatile(
                    "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes "
                    "[%0], [%1], %2, [%3];"
                    :: "r"(bview_addr + 49152), "l"(branch + nb_1), "r"((uint32_t)(16384)), "r"(rows_full_addr + 8)
                    : "memory");
                asm volatile(
                    "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes "
                    "[%0], [%1], %2, [%3];"
                    :: "r"(rview_addr + 49152), "l"(residual + nb_1), "r"((uint32_t)(32768)), "r"(rows_full_addr + 8)
                    : "memory");
            }
            int prow_2_2 = first + 2 * stride;
            if (prow_2_2 < M) {
                long long nb_2 = (long long)prow_2_2 * 8192;
                mbarrier_arrive_expect_tx(rows_full_addr + 16, 49152);
                asm volatile(
                    "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes "
                    "[%0], [%1], %2, [%3];"
                    :: "r"(bview_addr + 98304), "l"(branch + nb_2), "r"((uint32_t)(16384)), "r"(rows_full_addr + 16)
                    : "memory");
                asm volatile(
                    "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes "
                    "[%0], [%1], %2, [%3];"
                    :: "r"(rview_addr + 98304), "l"(residual + nb_2), "r"((uint32_t)(32768)), "r"(rows_full_addr + 16)
                    : "memory");
            }
        }
    }
    #pragma unroll 1
    for (int row0 = first; row0 < M; row0 += stride) {
        int row = row0;
        long long rbase = (long long)row * 8192 + col64;
        long long rb_0 = (long long)row * 8192 + ccol_0;
        long long rb_1 = (long long)row * 8192 + ccol_1;
        int _stage_rows_full = (int)((unsigned int)krow % 3);
        int _parity_rows_full = (int)((unsigned int)krow / 3 % 2);
        mbarrier_wait(rows_full_addr + (_stage_rows_full) * 8, _parity_rows_full);
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&cur_b[0])), "=r"(*reinterpret_cast<uint32_t*>(&cur_b[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&cur_b[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&cur_b[(0) + 3]))
            : "r"(bview_addr + (unsigned int)(krow % 3 * 49152) + sb_off_0));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&cur_b[4])), "=r"(*reinterpret_cast<uint32_t*>(&cur_b[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&cur_b[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&cur_b[(4) + 3]))
            : "r"(bview_addr + (unsigned int)(krow % 3 * 49152) + sb_off_1));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&cur_r[0])), "=r"(*reinterpret_cast<uint32_t*>(&cur_r[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&cur_r[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&cur_r[(0) + 3]))
            : "r"(rview_addr + (unsigned int)(krow % 3 * 49152) + sr_off_0));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&cur_r[4])), "=r"(*reinterpret_cast<uint32_t*>(&cur_r[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&cur_r[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&cur_r[(4) + 3]))
            : "r"(rview_addr + (unsigned int)(krow % 3 * 49152) + sr_off_0 + 16));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&cur_r[8])), "=r"(*reinterpret_cast<uint32_t*>(&cur_r[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&cur_r[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&cur_r[(8) + 3]))
            : "r"(rview_addr + (unsigned int)(krow % 3 * 49152) + sr_off_1));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&cur_r[12])), "=r"(*reinterpret_cast<uint32_t*>(&cur_r[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&cur_r[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&cur_r[(12) + 3]))
            : "r"(rview_addr + (unsigned int)(krow % 3 * 49152) + sr_off_1 + 16));
        mbarrier_arrive(rows_empty_addr + ((int)((unsigned int)krow % 3)) * 8);
        float bb[16];
        #pragma unroll
        for (int k = 0; k < 8; k++) {
            float2 _f32x2_mul_a_8 = cake_bf16x2_to_f32x2(__as_bf16x2(cur_b[k]));
            float2 _f32x2_mul_b_9 = add_f32x2_noftz(cake_bf16x2_to_f32x2(__as_bf16x2(pw[k])), pgc2);
            float2 g;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g) : "l"(*(const unsigned long long*)&_f32x2_mul_a_8), "l"(*(const unsigned long long*)&_f32x2_mul_b_9));
            bb[2 * k] = g.x;
            bb[2 * k + 1] = g.y;
        }
        float pa = 0.0f;
        #pragma unroll
        for (int j = 0; j < 16; j++) {
            pa = __fmaf_rn(bb[j], bb[j], pa);
        }
        #pragma unroll
        for (int k_1 = 0; k_1 < 8; k_1++) {
            float2 _f32x2_mul_a_10 = make_float2(al[2 * k_1], al[2 * k_1 + 1]);
            float2 _f32x2_mul_b_11 = make_float2(cur_r[2 * k_1], cur_r[2 * k_1 + 1]);
            float2 u;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&u) : "l"(*(const unsigned long long*)&_f32x2_mul_a_10), "l"(*(const unsigned long long*)&_f32x2_mul_b_11));
            cur_r[2 * k_1] = u.x;
            cur_r[2 * k_1 + 1] = u.y;
            float2 _f32x2_mul_a_12 = make_float2(be[2 * k_1], be[2 * k_1 + 1]);
            float2 _f32x2_mul_b_13 = make_float2(bb[2 * k_1], bb[2 * k_1 + 1]);
            float2 v;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&v) : "l"(*(const unsigned long long*)&_f32x2_mul_a_12), "l"(*(const unsigned long long*)&_f32x2_mul_b_13));
            bb[2 * k_1] = v.x;
            bb[2 * k_1 + 1] = v.y;
        }
        float _warp_reduce_0 = pa;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            _warp_reduce_0 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_0, offset);
        if (lane_1 == 0) {
            slots[warp_0] = _warp_reduce_0;
        }
        __syncthreads();
        int idx = ((lane_1) < (15) ? (lane_1) : (15));
        float value = slots[idx];
        if (lane_1 >= 16) {
            value = 0.0f;
        }
        float _warp_reduce_1 = value;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            _warp_reduce_1 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_1, offset);
        int nrow = row + 3 * stride;
        if (nrow < M) {
            if (warp == 0) {
                if (elect_sync()) {
                    int _stage_rows_empty = (int)((unsigned int)(krow + 3) % 3);
                    int _parity_rows_empty = (int)((unsigned int)(krow + 3) / 3 % 2) ^ 1;
                    mbarrier_wait(rows_empty_addr + (_stage_rows_empty) * 8, _parity_rows_empty);
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    long long nb_3 = (long long)nrow * 8192;
                    mbarrier_arrive_expect_tx(rows_full_addr + ((int)((unsigned int)(krow + 3) % 3)) * 8, 49152);
                    asm volatile(
                        "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes "
                        "[%0], [%1], %2, [%3];"
                        :: "r"(bview_addr + (unsigned int)((krow + 3) % 3 * 49152)), "l"(branch + nb_3), "r"((uint32_t)(16384)), "r"(rows_full_addr + ((int)((unsigned int)(krow + 3) % 3)) * 8)
                        : "memory");
                    asm volatile(
                        "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes "
                        "[%0], [%1], %2, [%3];"
                        :: "r"(rview_addr + (unsigned int)((krow + 3) % 3 * 49152)), "l"(residual + nb_3), "r"((uint32_t)(32768)), "r"(rows_full_addr + ((int)((unsigned int)(krow + 3) % 3)) * 8)
                        : "memory");
                }
            }
        }
        float _sqrt_14;
        asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(_sqrt_14) : "f"(__fmaf_rn(_warp_reduce_1, 0.00012207031f, post_eps)));
        float r1 = __frcp_rn(_sqrt_14);
        float p2 = 0.0f;
        float2 r1_2 = make_float2(r1, r1);
        #pragma unroll
        for (int k_2 = 0; k_2 < 8; k_2++) {
            float2 w = fma_f32x2_rn_noftz(make_float2(bb[2 * k_2], bb[2 * k_2 + 1]), r1_2, make_float2(cur_r[2 * k_2], cur_r[2 * k_2 + 1]));
            cur_r[2 * k_2] = w.x;
            cur_r[2 * k_2 + 1] = w.y;
        }
        #pragma unroll
        for (int j_1 = 0; j_1 < 16; j_1++) {
            p2 = __fmaf_rn(cur_r[j_1], cur_r[j_1], p2);
        }
        {
            unsigned _stv8_15_0 = __float_as_uint(cur_r[0 + 0]);
            unsigned _stv8_15_1 = __float_as_uint(cur_r[0 + 1]);
            unsigned _stv8_15_2 = __float_as_uint(cur_r[0 + 2]);
            unsigned _stv8_15_3 = __float_as_uint(cur_r[0 + 3]);
            unsigned _stv8_15_4 = __float_as_uint(cur_r[0 + 4]);
            unsigned _stv8_15_5 = __float_as_uint(cur_r[0 + 5]);
            unsigned _stv8_15_6 = __float_as_uint(cur_r[0 + 6]);
            unsigned _stv8_15_7 = __float_as_uint(cur_r[0 + 7]);
            asm volatile(
                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                :: "l"((void*)(residual_out + rb_0 + (0))), "r"(_stv8_15_0), "r"(_stv8_15_1), "r"(_stv8_15_2), "r"(_stv8_15_3), "r"(_stv8_15_4), "r"(_stv8_15_5), "r"(_stv8_15_6), "r"(_stv8_15_7) : "memory");
        }
        {
            unsigned _stv8_16_0 = __float_as_uint(cur_r[8 + 0]);
            unsigned _stv8_16_1 = __float_as_uint(cur_r[8 + 1]);
            unsigned _stv8_16_2 = __float_as_uint(cur_r[8 + 2]);
            unsigned _stv8_16_3 = __float_as_uint(cur_r[8 + 3]);
            unsigned _stv8_16_4 = __float_as_uint(cur_r[8 + 4]);
            unsigned _stv8_16_5 = __float_as_uint(cur_r[8 + 5]);
            unsigned _stv8_16_6 = __float_as_uint(cur_r[8 + 6]);
            unsigned _stv8_16_7 = __float_as_uint(cur_r[8 + 7]);
            asm volatile(
                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                :: "l"((void*)(residual_out + rb_1 + (0))), "r"(_stv8_16_0), "r"(_stv8_16_1), "r"(_stv8_16_2), "r"(_stv8_16_3), "r"(_stv8_16_4), "r"(_stv8_16_5), "r"(_stv8_16_6), "r"(_stv8_16_7) : "memory");
        }
        float warp_reduce = p2;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            warp_reduce += __shfl_xor_sync(0xFFFFFFFF, warp_reduce, offset);
        if (lane_1 == 0) {
            slots[16 + warp_0] = warp_reduce;
        }
        __syncthreads();
        int idx_0 = ((lane_1) < (15) ? (lane_1) : (15));
        float value_1 = slots[16 + idx_0];
        if (lane_1 >= 16) {
            value_1 = 0.0f;
        }
        float _warp_reduce_18 = value_1;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            _warp_reduce_18 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_18, offset);
        float _sqrt_17;
        asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(_sqrt_17) : "f"(__fmaf_rn(_warp_reduce_18, 0.00012207031f, next_eps)));
        float r2 = __frcp_rn(_sqrt_17);
        float2 r2_2 = make_float2(r2, r2);
        #pragma unroll
        for (int k_3 = 0; k_3 < 8; k_3++) {
            float2 _f32x2_mul_a_19 = mul_f32x2_noftz(make_float2(cur_r[2 * k_3], cur_r[2 * k_3 + 1]), r2_2);
            float2 _f32x2_mul_b_20 = add_f32x2_noftz(cake_bf16x2_to_f32x2(__as_bf16x2(nw[k_3])), ngc2);
            float2 t;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t) : "l"(*(const unsigned long long*)&_f32x2_mul_a_19), "l"(*(const unsigned long long*)&_f32x2_mul_b_20));
            bb[2 * k_3] = t.x;
            bb[2 * k_3 + 1] = t.y;
        }
        unsigned int _packed_0[4];
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(bb[_lp*2 + 0], bb[_lp*2+1 + 0]));
            _packed_0[_lp] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(bb[_lp*2 + 2], bb[_lp*2+1 + 2]));
            _packed_0[_lp + 1] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(bb[_lp*2 + 4], bb[_lp*2+1 + 4]));
            _packed_0[_lp + 2] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(bb[_lp*2 + 6], bb[_lp*2+1 + 6]));
            _packed_0[_lp + 3] = *(uint32_t*)&_bf2;
        }
        reinterpret_cast<int4*>(ffn_input + rb_0)[0] = reinterpret_cast<int4*>(_packed_0)[0];
        unsigned int _packed_1[4];
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(bb[_lp*2 + 8], bb[_lp*2+1 + 8]));
            _packed_1[_lp] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(bb[_lp*2 + 10], bb[_lp*2+1 + 10]));
            _packed_1[_lp + 1] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(bb[_lp*2 + 12], bb[_lp*2+1 + 12]));
            _packed_1[_lp + 2] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(bb[_lp*2 + 14], bb[_lp*2+1 + 14]));
            _packed_1[_lp + 3] = *(uint32_t*)&_bf2;
        }
        reinterpret_cast<int4*>(ffn_input + rb_1)[0] = reinterpret_cast<int4*>(_packed_1)[0];
        krow += 1;
    }

    // Cleanup
    __syncthreads();
}

} // extern "C"
