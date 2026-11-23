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
#define SMEM_PARTIALS_OFF 1024
#define SMEM_PARTIALS_STAGE_BYTES 128
#define SMEM_PARTIALS_STRIDE 128
#define SMEM_ABUF_OFF 2048
#define SMEM_ABUF_STAGE_BYTES 32768
#define SMEM_ABUF_STRIDE 32768
#define SMEM_BBUF_OFF 34816
#define SMEM_BBUF_STAGE_BYTES 32768
#define SMEM_BBUF_STRIDE 32768
#define SMEM_TOTAL 67584
#define THREADS 512
#define LAUNCH_BOUNDS_THREADS 512
#define LAUNCH_BOUNDS_MIN_BLOCKS 2
#define LAUNCH_MIN_BLOCKS 2

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

__global__ __launch_bounds__(LAUNCH_BOUNDS_THREADS, LAUNCH_BOUNDS_MIN_BLOCKS) void
kernel_cake_muse_spark_norm_c41d8d8efce00d30a8e8(__nv_bfloat16* __restrict__ branch, float* __restrict__ residual, __nv_bfloat16* __restrict__ weight, float* __restrict__ alpha, float* __restrict__ beta, float* __restrict__ out, int M, float eps, float gain_center)
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
    #define planes_in_addr (mbar_base + 0)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    float* partials = reinterpret_cast<float*>(smem_raw + 1024);
    const int partials_addr = smem + 1024;
    float* abuf = reinterpret_cast<float*>(smem_raw + 2048);
    const int abuf_addr = smem + 2048;
    float* bbuf = reinterpret_cast<float*>(smem_raw + 34816);
    const int bbuf_addr = smem + 34816;

    // Mbarrier init (1 pipeline groups, 0 ordered-sequence groups, 2 barriers)
    // Mbarriers at smem_raw[0..16)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // planes_in: 2 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            mbarrier_init(smem + 8, 1);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncthreads();

    // === Task calls (declaration order) ===
    int t = tid;
    int warp_1 = t >> 5;
    int first = bid;
    if (warp == 0) {
        if (elect_sync()) {
            mbarrier_arrive_expect_tx(planes_in_addr, 32768);
            asm volatile(
                "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes "
                "[%0], [%1], %2, [%3];"
                :: "r"(abuf_addr), "l"(alpha), "r"((uint32_t)(32768)), "r"(planes_in_addr)
                : "memory");
        }
    }
    if (warp == 0) {
        if (elect_sync()) {
            mbarrier_arrive_expect_tx(planes_in_addr + 8, 32768);
            asm volatile(
                "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes "
                "[%0], [%1], %2, [%3];"
                :: "r"(bbuf_addr), "l"(beta), "r"((uint32_t)(32768)), "r"(planes_in_addr + 8)
                : "memory");
        }
    }
    int it = 0;
    unsigned int cur[8];
    float x[16];
    float wf[16];
    float al[16];
    float be[16];
    float res[16];
    float y[16];
    if (first + 304 < M) {
        if (warp == 0) {
            if (elect_sync()) {
                long long pf = (long long)(first + 304) * 8192;
                asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(residual + pf), "r"((uint32_t)(32768)) : "memory");
                asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(branch + pf), "r"((uint32_t)(16384)) : "memory");
            }
        }
    }
    #pragma unroll
    for (int c = 0; c < 2; c++) {
        float _vec_load_0[8];
        {
            const uint4* _vptr_0 = reinterpret_cast<const uint4*>(weight + (c * 4096 + 8 * t) + 0);
            uint4 _vld_0[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                asm volatile("ld.global.nc.v4.b32 {%0, %1, %2, %3}, [%4];"
                    : "=r"(_vld_0[_blk].x), "=r"(_vld_0[_blk].y), "=r"(_vld_0[_blk].z), "=r"(_vld_0[_blk].w) : "l"((const void*)(_vptr_0 + _blk)) : "memory");
                uint32_t* _vpairs_0 = reinterpret_cast<uint32_t*>(&_vld_0[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_vec_load_0[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_0[_pair]) << 16);
                    (&_vec_load_0[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_0[_pair]) & 0xffff0000u);
                }
            }
        }
        #pragma unroll
        for (int v = 0; v < 8; v++) {
            float wg = _vec_load_0[v] + gain_center;
            wf[c * 8 + v] = wg;
        }
    }
    {
        const uint4* _vptr_1 = reinterpret_cast<const uint4*>(branch + ((long long)first * 8192 + (long long)(8 * t)) + 0);
        uint4* _vdst_1 = reinterpret_cast<uint4*>(&cur[0]);
        #pragma unroll
        for (int _blk = 0; _blk < 1; _blk++) {
            asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                : "=r"(_vdst_1[_blk].x), "=r"(_vdst_1[_blk].y), "=r"(_vdst_1[_blk].z), "=r"(_vdst_1[_blk].w) : "l"((const void*)(_vptr_1 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
        }
    }
    {
        const uint4* _vptr_2 = reinterpret_cast<const uint4*>(branch + ((long long)first * 8192 + (long long)(4096 + 8 * t)) + 0);
        uint4* _vdst_2 = reinterpret_cast<uint4*>(&cur[4]);
        #pragma unroll
        for (int _blk = 0; _blk < 1; _blk++) {
            asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                : "=r"(_vdst_2[_blk].x), "=r"(_vdst_2[_blk].y), "=r"(_vdst_2[_blk].z), "=r"(_vdst_2[_blk].w) : "l"((const void*)(_vptr_2 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
        }
    }
    #pragma unroll 1
    for (int row = first; row < first + 1; row++) {
        long long row64 = (long long)row;
        long long fbase = row64 * 8192;
        int par = (it & 1) * 16;
        #pragma unroll
        for (int c_1 = 0; c_1 < 2; c_1++) {
            {
                unsigned _ldv8_3_0;
                unsigned _ldv8_3_1;
                unsigned _ldv8_3_2;
                unsigned _ldv8_3_3;
                unsigned _ldv8_3_4;
                unsigned _ldv8_3_5;
                unsigned _ldv8_3_6;
                unsigned _ldv8_3_7;
                asm volatile("ld.global.L2::cache_hint.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8], %9;"
                    : "=r"(_ldv8_3_0), "=r"(_ldv8_3_1), "=r"(_ldv8_3_2), "=r"(_ldv8_3_3), "=r"(_ldv8_3_4), "=r"(_ldv8_3_5), "=r"(_ldv8_3_6), "=r"(_ldv8_3_7) : "l"((const void*)(residual + (fbase + (long long)(c_1 * 4096 + 8 * t)) + (0))), "l"(0x12F0000000000000ULL) : "memory");
                res[c_1 * 8 + 0] = __uint_as_float(_ldv8_3_0);
                res[c_1 * 8 + 1] = __uint_as_float(_ldv8_3_1);
                res[c_1 * 8 + 2] = __uint_as_float(_ldv8_3_2);
                res[c_1 * 8 + 3] = __uint_as_float(_ldv8_3_3);
                res[c_1 * 8 + 4] = __uint_as_float(_ldv8_3_4);
                res[c_1 * 8 + 5] = __uint_as_float(_ldv8_3_5);
                res[c_1 * 8 + 6] = __uint_as_float(_ldv8_3_6);
                res[c_1 * 8 + 7] = __uint_as_float(_ldv8_3_7);
            }
        }
        float acc = 0.0f;
        float2 _cvt_f32_0 = cake_bf16x2_to_f32x2(__as_bf16x2(cur[0]));
        float x0_0 = _cvt_f32_0.x * wf[0];
        float x1_0 = _cvt_f32_0.y * wf[1];
        x[0] = x0_0;
        x[1] = x1_0;
        acc = __fmaf_rn(x0_0, x0_0, acc);
        acc = __fmaf_rn(x1_0, x1_0, acc);
        float2 _cvt_f32_1 = cake_bf16x2_to_f32x2(__as_bf16x2(cur[1]));
        float x0_1 = _cvt_f32_1.x * wf[2];
        float x1_1 = _cvt_f32_1.y * wf[3];
        x[2] = x0_1;
        x[3] = x1_1;
        acc = __fmaf_rn(x0_1, x0_1, acc);
        acc = __fmaf_rn(x1_1, x1_1, acc);
        float2 _cvt_f32_2 = cake_bf16x2_to_f32x2(__as_bf16x2(cur[2]));
        float x0_2 = _cvt_f32_2.x * wf[4];
        float x1_2 = _cvt_f32_2.y * wf[5];
        x[4] = x0_2;
        x[5] = x1_2;
        acc = __fmaf_rn(x0_2, x0_2, acc);
        acc = __fmaf_rn(x1_2, x1_2, acc);
        float2 _cvt_f32_3 = cake_bf16x2_to_f32x2(__as_bf16x2(cur[3]));
        float x0_3 = _cvt_f32_3.x * wf[6];
        float x1_3 = _cvt_f32_3.y * wf[7];
        x[6] = x0_3;
        x[7] = x1_3;
        acc = __fmaf_rn(x0_3, x0_3, acc);
        acc = __fmaf_rn(x1_3, x1_3, acc);
        float2 _cvt_f32_4 = cake_bf16x2_to_f32x2(__as_bf16x2(cur[4]));
        float x0_4 = _cvt_f32_4.x * wf[8];
        float x1_4 = _cvt_f32_4.y * wf[9];
        x[8] = x0_4;
        x[9] = x1_4;
        acc = __fmaf_rn(x0_4, x0_4, acc);
        acc = __fmaf_rn(x1_4, x1_4, acc);
        float2 _cvt_f32_5 = cake_bf16x2_to_f32x2(__as_bf16x2(cur[5]));
        float x0_5 = _cvt_f32_5.x * wf[10];
        float x1_5 = _cvt_f32_5.y * wf[11];
        x[10] = x0_5;
        x[11] = x1_5;
        acc = __fmaf_rn(x0_5, x0_5, acc);
        acc = __fmaf_rn(x1_5, x1_5, acc);
        float2 _cvt_f32_6 = cake_bf16x2_to_f32x2(__as_bf16x2(cur[6]));
        float x0_6 = _cvt_f32_6.x * wf[12];
        float x1_6 = _cvt_f32_6.y * wf[13];
        x[12] = x0_6;
        x[13] = x1_6;
        acc = __fmaf_rn(x0_6, x0_6, acc);
        acc = __fmaf_rn(x1_6, x1_6, acc);
        float2 _cvt_f32_7 = cake_bf16x2_to_f32x2(__as_bf16x2(cur[7]));
        float x0_7 = _cvt_f32_7.x * wf[14];
        float x1_7 = _cvt_f32_7.y * wf[15];
        x[14] = x0_7;
        x[15] = x1_7;
        acc = __fmaf_rn(x0_7, x0_7, acc);
        acc = __fmaf_rn(x1_7, x1_7, acc);
        mbarrier_wait(planes_in_addr, 0);
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&al[0])), "=r"(*reinterpret_cast<uint32_t*>(&al[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&al[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&al[(0) + 3]))
            : "r"(abuf_addr + (unsigned int)(8 * t * 4)));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&al[4])), "=r"(*reinterpret_cast<uint32_t*>(&al[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&al[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&al[(4) + 3]))
            : "r"(abuf_addr + (unsigned int)(8 * t * 4 + 16)));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&al[8])), "=r"(*reinterpret_cast<uint32_t*>(&al[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&al[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&al[(8) + 3]))
            : "r"(abuf_addr + (unsigned int)((4096 + 8 * t) * 4)));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&al[12])), "=r"(*reinterpret_cast<uint32_t*>(&al[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&al[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&al[(12) + 3]))
            : "r"(abuf_addr + (unsigned int)((4096 + 8 * t) * 4 + 16)));
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            float pre = al[i] * res[i];
            res[i] = pre;
        }
        float _warp_reduce_4 = acc;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            _warp_reduce_4 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_4, offset);
        acc = _warp_reduce_4;
        if (lane == 0) {
            partials[par + warp_1] = acc;
        }
        __syncthreads();
        mbarrier_wait(planes_in_addr + 8, 0);
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&be[0])), "=r"(*reinterpret_cast<uint32_t*>(&be[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&be[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&be[(0) + 3]))
            : "r"(bbuf_addr + (unsigned int)(8 * t * 4)));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&be[4])), "=r"(*reinterpret_cast<uint32_t*>(&be[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&be[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&be[(4) + 3]))
            : "r"(bbuf_addr + (unsigned int)(8 * t * 4 + 16)));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&be[8])), "=r"(*reinterpret_cast<uint32_t*>(&be[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&be[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&be[(8) + 3]))
            : "r"(bbuf_addr + (unsigned int)((4096 + 8 * t) * 4)));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&be[12])), "=r"(*reinterpret_cast<uint32_t*>(&be[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&be[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&be[(12) + 3]))
            : "r"(bbuf_addr + (unsigned int)((4096 + 8 * t) * 4 + 16)));
        float total = partials[par];
        #pragma unroll
        for (int k = 1; k < 16; k++) {
            total += partials[par + k];
        }
        it += 1;
        float mean = __fmaf_rn(total, 0.00012207031f, eps);
        float _sqrt_5;
        asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(_sqrt_5) : "f"(mean));
        float inv = __frcp_rn(_sqrt_5);
        #pragma unroll
        for (int c_2 = 0; c_2 < 2; c_2++) {
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
                    :: "l"((void*)(out + (fbase + (long long)(c_2 * 4096 + 8 * t)) + (0))), "r"(_stv8_6_0), "r"(_stv8_6_1), "r"(_stv8_6_2), "r"(_stv8_6_3), "r"(_stv8_6_4), "r"(_stv8_6_5), "r"(_stv8_6_6), "r"(_stv8_6_7) : "memory");
            }
        }
    }

    // Cleanup
    __syncthreads();
}

} // extern "C"
