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
#define SMEM_PARTIALS_OFF 0
#define SMEM_PARTIALS_STAGE_BYTES 128
#define SMEM_PARTIALS_STRIDE 128
#define SMEM_TOTAL 384
#define THREADS 512
#define SPLIT 1
#define SEG 8192
#define LAUNCH_BOUNDS_THREADS 512

#include <math_constants.h>

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
    const unsigned int w = *reinterpret_cast<const unsigned int*>(&v);
    return make_float2(__uint_as_float(w << 16), __uint_as_float(w & 0xffff0000u));
}

extern "C" {

__global__ __launch_bounds__(LAUNCH_BOUNDS_THREADS) void
kernel_cake_muse_spark_norm_37adff50d4fef57403a1(__nv_bfloat16* __restrict__ branch, float* __restrict__ residual, __nv_bfloat16* __restrict__ weight, float* __restrict__ alpha, float* __restrict__ beta, float* __restrict__ out, int M, float eps, float gain_center)
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
    float* partials = reinterpret_cast<float*>(smem_raw + 0);
    const int partials_addr = smem + 0;

    // === Task calls (declaration order) ===
    int t = tid;
    int warp_1 = t >> 5;
    int seg = 0;
    int first = bid;
    int lo = seg * SEG;
    int hi = lo + SEG;
    int cbase = 0;
    int it = 0;
    unsigned int cur[8];
    float x[16];
    float al[16];
    float be[16];
    float res[16];
    float y[16];
    float pv[16];
    long long fbase = (long long)first * 8192;
    {
        const uint4* _vptr_0 = reinterpret_cast<const uint4*>(branch + (fbase + (long long)(cbase + 8 * t)) + 0);
        uint4* _vdst_0 = reinterpret_cast<uint4*>(&cur[0]);
        #pragma unroll
        for (int _blk = 0; _blk < 1; _blk++) {
            asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                : "=r"(_vdst_0[_blk].x), "=r"(_vdst_0[_blk].y), "=r"(_vdst_0[_blk].z), "=r"(_vdst_0[_blk].w) : "l"((const void*)(_vptr_0 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
        }
    }
    {
        const uint4* _vptr_1 = reinterpret_cast<const uint4*>(branch + (fbase + (long long)(cbase + 4096 + 8 * t)) + 0);
        uint4* _vdst_1 = reinterpret_cast<uint4*>(&cur[4]);
        #pragma unroll
        for (int _blk = 0; _blk < 1; _blk++) {
            asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                : "=r"(_vdst_1[_blk].x), "=r"(_vdst_1[_blk].y), "=r"(_vdst_1[_blk].z), "=r"(_vdst_1[_blk].w) : "l"((const void*)(_vptr_1 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
        }
    }
    #pragma unroll
    for (int c = 0; c < 2; c++) {
        {
            unsigned _ldv8_2_0;
            unsigned _ldv8_2_1;
            unsigned _ldv8_2_2;
            unsigned _ldv8_2_3;
            unsigned _ldv8_2_4;
            unsigned _ldv8_2_5;
            unsigned _ldv8_2_6;
            unsigned _ldv8_2_7;
            asm volatile("ld.global.nc.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                : "=r"(_ldv8_2_0), "=r"(_ldv8_2_1), "=r"(_ldv8_2_2), "=r"(_ldv8_2_3), "=r"(_ldv8_2_4), "=r"(_ldv8_2_5), "=r"(_ldv8_2_6), "=r"(_ldv8_2_7) : "l"((const void*)(alpha + (cbase + c * 4096 + 8 * t) + (0))) : "memory");
            al[c * 8 + 0] = __uint_as_float(_ldv8_2_0);
            al[c * 8 + 1] = __uint_as_float(_ldv8_2_1);
            al[c * 8 + 2] = __uint_as_float(_ldv8_2_2);
            al[c * 8 + 3] = __uint_as_float(_ldv8_2_3);
            al[c * 8 + 4] = __uint_as_float(_ldv8_2_4);
            al[c * 8 + 5] = __uint_as_float(_ldv8_2_5);
            al[c * 8 + 6] = __uint_as_float(_ldv8_2_6);
            al[c * 8 + 7] = __uint_as_float(_ldv8_2_7);
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
            asm volatile("ld.global.nc.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                : "=r"(_ldv8_3_0), "=r"(_ldv8_3_1), "=r"(_ldv8_3_2), "=r"(_ldv8_3_3), "=r"(_ldv8_3_4), "=r"(_ldv8_3_5), "=r"(_ldv8_3_6), "=r"(_ldv8_3_7) : "l"((const void*)(beta + (cbase + c * 4096 + 8 * t) + (0))) : "memory");
            be[c * 8 + 0] = __uint_as_float(_ldv8_3_0);
            be[c * 8 + 1] = __uint_as_float(_ldv8_3_1);
            be[c * 8 + 2] = __uint_as_float(_ldv8_3_2);
            be[c * 8 + 3] = __uint_as_float(_ldv8_3_3);
            be[c * 8 + 4] = __uint_as_float(_ldv8_3_4);
            be[c * 8 + 5] = __uint_as_float(_ldv8_3_5);
            be[c * 8 + 6] = __uint_as_float(_ldv8_3_6);
            be[c * 8 + 7] = __uint_as_float(_ldv8_3_7);
        }
    }
    #pragma unroll 1
    for (int row = first; row < first + 1; row++) {
        long long row64 = (long long)row;
        long long sbase = row64 * 8192;
        long long obase = row64 * 8192;
        int par = (it & 1) * 16;
        #pragma unroll
        for (int c_1 = 0; c_1 < 2; c_1++) {
            {
                unsigned _ldv8_4_0;
                unsigned _ldv8_4_1;
                unsigned _ldv8_4_2;
                unsigned _ldv8_4_3;
                unsigned _ldv8_4_4;
                unsigned _ldv8_4_5;
                unsigned _ldv8_4_6;
                unsigned _ldv8_4_7;
                asm volatile("ld.global.L2::cache_hint.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8], %9;"
                    : "=r"(_ldv8_4_0), "=r"(_ldv8_4_1), "=r"(_ldv8_4_2), "=r"(_ldv8_4_3), "=r"(_ldv8_4_4), "=r"(_ldv8_4_5), "=r"(_ldv8_4_6), "=r"(_ldv8_4_7) : "l"((const void*)(residual + (sbase + (long long)(cbase + c_1 * 4096 + 8 * t)) + (0))), "l"(0x12F0000000000000ULL) : "memory");
                res[c_1 * 8 + 0] = __uint_as_float(_ldv8_4_0);
                res[c_1 * 8 + 1] = __uint_as_float(_ldv8_4_1);
                res[c_1 * 8 + 2] = __uint_as_float(_ldv8_4_2);
                res[c_1 * 8 + 3] = __uint_as_float(_ldv8_4_3);
                res[c_1 * 8 + 4] = __uint_as_float(_ldv8_4_4);
                res[c_1 * 8 + 5] = __uint_as_float(_ldv8_4_5);
                res[c_1 * 8 + 6] = __uint_as_float(_ldv8_4_6);
                res[c_1 * 8 + 7] = __uint_as_float(_ldv8_4_7);
            }
        }
        float acc = 0.0f;
        if (M > 0) {
            float2 acc2_0 = make_float2(0.0f, 0.0f);
            float2 x2_0 = cake_bf16x2_to_f32x2(__as_bf16x2(cur[0]));
            x[0] = x2_0.x;
            x[1] = x2_0.y;
            float2 acc2_1 = fma_f32x2_rn_noftz(x2_0, x2_0, acc2_0);
            float2 x2_2 = cake_bf16x2_to_f32x2(__as_bf16x2(cur[1]));
            x[2] = x2_2.x;
            x[3] = x2_2.y;
            float2 acc2_2 = fma_f32x2_rn_noftz(x2_2, x2_2, acc2_1);
            float2 x2_4 = cake_bf16x2_to_f32x2(__as_bf16x2(cur[2]));
            x[4] = x2_4.x;
            x[5] = x2_4.y;
            float2 acc2_3 = fma_f32x2_rn_noftz(x2_4, x2_4, acc2_2);
            float2 x2_6 = cake_bf16x2_to_f32x2(__as_bf16x2(cur[3]));
            x[6] = x2_6.x;
            x[7] = x2_6.y;
            float2 acc2_4 = fma_f32x2_rn_noftz(x2_6, x2_6, acc2_3);
            float2 x2_8 = cake_bf16x2_to_f32x2(__as_bf16x2(cur[4]));
            x[8] = x2_8.x;
            x[9] = x2_8.y;
            float2 acc2_5 = fma_f32x2_rn_noftz(x2_8, x2_8, acc2_4);
            float2 x2_10 = cake_bf16x2_to_f32x2(__as_bf16x2(cur[5]));
            x[10] = x2_10.x;
            x[11] = x2_10.y;
            float2 acc2_6 = fma_f32x2_rn_noftz(x2_10, x2_10, acc2_5);
            float2 x2_12 = cake_bf16x2_to_f32x2(__as_bf16x2(cur[6]));
            x[12] = x2_12.x;
            x[13] = x2_12.y;
            float2 acc2_7 = fma_f32x2_rn_noftz(x2_12, x2_12, acc2_6);
            float2 x2_14 = cake_bf16x2_to_f32x2(__as_bf16x2(cur[7]));
            x[14] = x2_14.x;
            x[15] = x2_14.y;
            float2 acc2_8 = fma_f32x2_rn_noftz(x2_14, x2_14, acc2_7);
            acc = acc + acc2_8.x;
            acc = acc + acc2_8.y;
        }
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            float pre = al[i] * res[i];
            res[i] = pre;
        }
        float _warp_reduce_5 = acc;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            _warp_reduce_5 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_5, offset);
        acc = _warp_reduce_5;
        if (lane == 0) {
            partials[par + warp_1] = acc;
        }
        __syncthreads();
        unsigned int pbyte = (unsigned int)((it & 1) * 64);
        __int128_t _smem_raw_6;
        asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_6) : "r"(partials_addr + pbyte));
        *reinterpret_cast<uint32_t*>(&pv[0]) = reinterpret_cast<const uint32_t*>(&_smem_raw_6)[0];
        *reinterpret_cast<uint32_t*>(&pv[(0) + 1]) = reinterpret_cast<const uint32_t*>(&_smem_raw_6)[1];
        *reinterpret_cast<uint32_t*>(&pv[(0) + 2]) = reinterpret_cast<const uint32_t*>(&_smem_raw_6)[2];
        *reinterpret_cast<uint32_t*>(&pv[(0) + 3]) = reinterpret_cast<const uint32_t*>(&_smem_raw_6)[3];
        __int128_t _smem_raw_7;
        asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_7) : "r"(partials_addr + pbyte + 16));
        *reinterpret_cast<uint32_t*>(&pv[4]) = reinterpret_cast<const uint32_t*>(&_smem_raw_7)[0];
        *reinterpret_cast<uint32_t*>(&pv[(4) + 1]) = reinterpret_cast<const uint32_t*>(&_smem_raw_7)[1];
        *reinterpret_cast<uint32_t*>(&pv[(4) + 2]) = reinterpret_cast<const uint32_t*>(&_smem_raw_7)[2];
        *reinterpret_cast<uint32_t*>(&pv[(4) + 3]) = reinterpret_cast<const uint32_t*>(&_smem_raw_7)[3];
        __int128_t _smem_raw_8;
        asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_8) : "r"(partials_addr + pbyte + 32));
        *reinterpret_cast<uint32_t*>(&pv[8]) = reinterpret_cast<const uint32_t*>(&_smem_raw_8)[0];
        *reinterpret_cast<uint32_t*>(&pv[(8) + 1]) = reinterpret_cast<const uint32_t*>(&_smem_raw_8)[1];
        *reinterpret_cast<uint32_t*>(&pv[(8) + 2]) = reinterpret_cast<const uint32_t*>(&_smem_raw_8)[2];
        *reinterpret_cast<uint32_t*>(&pv[(8) + 3]) = reinterpret_cast<const uint32_t*>(&_smem_raw_8)[3];
        __int128_t _smem_raw_9;
        asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_9) : "r"(partials_addr + pbyte + 48));
        *reinterpret_cast<uint32_t*>(&pv[12]) = reinterpret_cast<const uint32_t*>(&_smem_raw_9)[0];
        *reinterpret_cast<uint32_t*>(&pv[(12) + 1]) = reinterpret_cast<const uint32_t*>(&_smem_raw_9)[1];
        *reinterpret_cast<uint32_t*>(&pv[(12) + 2]) = reinterpret_cast<const uint32_t*>(&_smem_raw_9)[2];
        *reinterpret_cast<uint32_t*>(&pv[(12) + 3]) = reinterpret_cast<const uint32_t*>(&_smem_raw_9)[3];
        pv[0] = pv[0] + pv[8];
        pv[1] = pv[1] + pv[9];
        pv[2] = pv[2] + pv[10];
        pv[3] = pv[3] + pv[11];
        pv[4] = pv[4] + pv[12];
        pv[5] = pv[5] + pv[13];
        pv[6] = pv[6] + pv[14];
        pv[7] = pv[7] + pv[15];
        pv[0] = pv[0] + pv[4];
        pv[1] = pv[1] + pv[5];
        pv[2] = pv[2] + pv[6];
        pv[3] = pv[3] + pv[7];
        pv[0] = pv[0] + pv[2];
        pv[1] = pv[1] + pv[3];
        pv[0] = pv[0] + pv[1];
        float total = pv[0];
        it += 1;
        float mean = __fmaf_rn(total, 0.00012207031f, eps);
        float inv;
        asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(inv) : "f"(mean));
        float2 inv2 = make_float2(inv, inv);
        #pragma unroll
        for (int c_2 = 0; c_2 < 2; c_2++) {
            float2 _fma_0 = fma_f32x2_rn_noftz(make_float2(be[c_2 * 8], be[c_2 * 8 + 1]), mul_f32x2_noftz(make_float2(x[c_2 * 8], x[c_2 * 8 + 1]), inv2), make_float2(res[c_2 * 8], res[c_2 * 8 + 1]));
            y[c_2 * 8] = _fma_0.x;
            y[c_2 * 8 + 1] = _fma_0.y;
            float2 _fma_1 = fma_f32x2_rn_noftz(make_float2(be[c_2 * 8 + 2], be[c_2 * 8 + 2 + 1]), mul_f32x2_noftz(make_float2(x[c_2 * 8 + 2], x[c_2 * 8 + 2 + 1]), inv2), make_float2(res[c_2 * 8 + 2], res[c_2 * 8 + 2 + 1]));
            y[c_2 * 8 + 2] = _fma_1.x;
            y[c_2 * 8 + 2 + 1] = _fma_1.y;
            float2 _fma_2 = fma_f32x2_rn_noftz(make_float2(be[c_2 * 8 + 4], be[c_2 * 8 + 4 + 1]), mul_f32x2_noftz(make_float2(x[c_2 * 8 + 4], x[c_2 * 8 + 4 + 1]), inv2), make_float2(res[c_2 * 8 + 4], res[c_2 * 8 + 4 + 1]));
            y[c_2 * 8 + 4] = _fma_2.x;
            y[c_2 * 8 + 4 + 1] = _fma_2.y;
            float2 _fma_3 = fma_f32x2_rn_noftz(make_float2(be[c_2 * 8 + 6], be[c_2 * 8 + 6 + 1]), mul_f32x2_noftz(make_float2(x[c_2 * 8 + 6], x[c_2 * 8 + 6 + 1]), inv2), make_float2(res[c_2 * 8 + 6], res[c_2 * 8 + 6 + 1]));
            y[c_2 * 8 + 6] = _fma_3.x;
            y[c_2 * 8 + 6 + 1] = _fma_3.y;
            {
                unsigned _stv8_10_0 = __float_as_uint(y[c_2 * 8 + 0]);
                unsigned _stv8_10_1 = __float_as_uint(y[c_2 * 8 + 1]);
                unsigned _stv8_10_2 = __float_as_uint(y[c_2 * 8 + 2]);
                unsigned _stv8_10_3 = __float_as_uint(y[c_2 * 8 + 3]);
                unsigned _stv8_10_4 = __float_as_uint(y[c_2 * 8 + 4]);
                unsigned _stv8_10_5 = __float_as_uint(y[c_2 * 8 + 5]);
                unsigned _stv8_10_6 = __float_as_uint(y[c_2 * 8 + 6]);
                unsigned _stv8_10_7 = __float_as_uint(y[c_2 * 8 + 7]);
                asm volatile(
                    "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                    :: "l"((void*)(out + (obase + (long long)(cbase + c_2 * 4096 + 8 * t)) + (0))), "r"(_stv8_10_0), "r"(_stv8_10_1), "r"(_stv8_10_2), "r"(_stv8_10_3), "r"(_stv8_10_4), "r"(_stv8_10_5), "r"(_stv8_10_6), "r"(_stv8_10_7) : "memory");
            }
        }
    }
}

} // extern "C"
