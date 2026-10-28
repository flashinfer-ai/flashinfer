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
#define SMEM_PLANES_OFF 320
#define SMEM_PLANES_STAGE_BYTES 32768
#define SMEM_PLANES_STRIDE 32768
#define SMEM_TOTAL 33152
#define THREADS 512
#define SPLIT 1
#define SEG 8192
#define LAUNCH_BOUNDS_THREADS 512

#include <math_constants.h>


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
kernel_cake_muse_spark_norm_1a810e28e0c14fe4e541(__nv_bfloat16* __restrict__ branch, float* __restrict__ residual, __nv_bfloat16* __restrict__ weight, float* __restrict__ alpha, float* __restrict__ beta, float* __restrict__ out, int M, float eps, float gain_center)
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
    float* planes = reinterpret_cast<float*>(smem_raw + 320);
    const int planes_addr = smem + 320;

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
    float wf[16];
    float al[16];
    float be[16];
    float res[16];
    float y[16];
    long long fbase = (long long)first * 8192;
    #pragma unroll
    for (int c = 0; c < 2; c++) {
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
                : "=r"(_ldv8_0_0), "=r"(_ldv8_0_1), "=r"(_ldv8_0_2), "=r"(_ldv8_0_3), "=r"(_ldv8_0_4), "=r"(_ldv8_0_5), "=r"(_ldv8_0_6), "=r"(_ldv8_0_7) : "l"((const void*)(alpha + (cbase + c * 4096 + 8 * t) + (0))) : "memory");
            al[c * 8 + 0] = __uint_as_float(_ldv8_0_0);
            al[c * 8 + 1] = __uint_as_float(_ldv8_0_1);
            al[c * 8 + 2] = __uint_as_float(_ldv8_0_2);
            al[c * 8 + 3] = __uint_as_float(_ldv8_0_3);
            al[c * 8 + 4] = __uint_as_float(_ldv8_0_4);
            al[c * 8 + 5] = __uint_as_float(_ldv8_0_5);
            al[c * 8 + 6] = __uint_as_float(_ldv8_0_6);
            al[c * 8 + 7] = __uint_as_float(_ldv8_0_7);
        }
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
            : "=r"(_ldv8_1_0), "=r"(_ldv8_1_1), "=r"(_ldv8_1_2), "=r"(_ldv8_1_3), "=r"(_ldv8_1_4), "=r"(_ldv8_1_5), "=r"(_ldv8_1_6), "=r"(_ldv8_1_7) : "l"((const void*)(beta + (cbase + 8 * t) + (0))) : "memory");
        be[0 + 0] = __uint_as_float(_ldv8_1_0);
        be[0 + 1] = __uint_as_float(_ldv8_1_1);
        be[0 + 2] = __uint_as_float(_ldv8_1_2);
        be[0 + 3] = __uint_as_float(_ldv8_1_3);
        be[0 + 4] = __uint_as_float(_ldv8_1_4);
        be[0 + 5] = __uint_as_float(_ldv8_1_5);
        be[0 + 6] = __uint_as_float(_ldv8_1_6);
        be[0 + 7] = __uint_as_float(_ldv8_1_7);
    }
    asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(planes_addr + (unsigned int)(t * 16)), "f"(be[0]), "f"(be[1]), "f"(be[2]), "f"(be[3]) : "memory");
    asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(planes_addr + (unsigned int)(t * 16 + 8192)), "f"(be[4]), "f"(be[5]), "f"(be[6]), "f"(be[7]) : "memory");
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
            : "=r"(_ldv8_2_0), "=r"(_ldv8_2_1), "=r"(_ldv8_2_2), "=r"(_ldv8_2_3), "=r"(_ldv8_2_4), "=r"(_ldv8_2_5), "=r"(_ldv8_2_6), "=r"(_ldv8_2_7) : "l"((const void*)(beta + (cbase + 4096 + 8 * t) + (0))) : "memory");
        be[8 + 0] = __uint_as_float(_ldv8_2_0);
        be[8 + 1] = __uint_as_float(_ldv8_2_1);
        be[8 + 2] = __uint_as_float(_ldv8_2_2);
        be[8 + 3] = __uint_as_float(_ldv8_2_3);
        be[8 + 4] = __uint_as_float(_ldv8_2_4);
        be[8 + 5] = __uint_as_float(_ldv8_2_5);
        be[8 + 6] = __uint_as_float(_ldv8_2_6);
        be[8 + 7] = __uint_as_float(_ldv8_2_7);
    }
    asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(planes_addr + (unsigned int)(t * 16 + 16384)), "f"(be[8]), "f"(be[9]), "f"(be[10]), "f"(be[11]) : "memory");
    asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(planes_addr + (unsigned int)(t * 16 + 24576)), "f"(be[12]), "f"(be[13]), "f"(be[14]), "f"(be[15]) : "memory");
    #pragma unroll
    for (int c_1 = 0; c_1 < 2; c_1++) {
        float _vec_load_0[8];
        {
            const uint4* _vptr_3 = reinterpret_cast<const uint4*>(weight + (cbase + c_1 * 4096 + 8 * t) + 0);
            uint4 _vld_3[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                asm volatile("ld.global.nc.v4.b32 {%0, %1, %2, %3}, [%4];"
                    : "=r"(_vld_3[_blk].x), "=r"(_vld_3[_blk].y), "=r"(_vld_3[_blk].z), "=r"(_vld_3[_blk].w) : "l"((const void*)(_vptr_3 + _blk)) : "memory");
                uint32_t* _vpairs_3 = reinterpret_cast<uint32_t*>(&_vld_3[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_vec_load_0[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_3[_pair]) << 16);
                    (&_vec_load_0[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_3[_pair]) & 0xffff0000u);
                }
            }
        }
        #pragma unroll
        for (int v = 0; v < 8; v++) {
            float wg = _vec_load_0[v] + gain_center;
            wf[c_1 * 8 + v] = wg;
        }
    }
    {
        const uint4* _vptr_4 = reinterpret_cast<const uint4*>(branch + (fbase + (long long)(cbase + 8 * t)) + 0);
        uint4* _vdst_4 = reinterpret_cast<uint4*>(&cur[0]);
        #pragma unroll
        for (int _blk = 0; _blk < 1; _blk++) {
            asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                : "=r"(_vdst_4[_blk].x), "=r"(_vdst_4[_blk].y), "=r"(_vdst_4[_blk].z), "=r"(_vdst_4[_blk].w) : "l"((const void*)(_vptr_4 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
        }
    }
    {
        const uint4* _vptr_5 = reinterpret_cast<const uint4*>(branch + (fbase + (long long)(cbase + 4096 + 8 * t)) + 0);
        uint4* _vdst_5 = reinterpret_cast<uint4*>(&cur[4]);
        #pragma unroll
        for (int _blk = 0; _blk < 1; _blk++) {
            asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                : "=r"(_vdst_5[_blk].x), "=r"(_vdst_5[_blk].y), "=r"(_vdst_5[_blk].z), "=r"(_vdst_5[_blk].w) : "l"((const void*)(_vptr_5 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
        }
    }
    #pragma unroll 1
    for (int row = first; row < first + 1; row++) {
        long long row64 = (long long)row;
        long long sbase = row64 * 8192;
        long long obase = row64 * 8192;
        int par = (it & 1) * 16;
        #pragma unroll
        for (int c_2 = 0; c_2 < 2; c_2++) {
            {
                unsigned _ldv8_6_0;
                unsigned _ldv8_6_1;
                unsigned _ldv8_6_2;
                unsigned _ldv8_6_3;
                unsigned _ldv8_6_4;
                unsigned _ldv8_6_5;
                unsigned _ldv8_6_6;
                unsigned _ldv8_6_7;
                asm volatile("ld.global.L2::cache_hint.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8], %9;"
                    : "=r"(_ldv8_6_0), "=r"(_ldv8_6_1), "=r"(_ldv8_6_2), "=r"(_ldv8_6_3), "=r"(_ldv8_6_4), "=r"(_ldv8_6_5), "=r"(_ldv8_6_6), "=r"(_ldv8_6_7) : "l"((const void*)(residual + (sbase + (long long)(cbase + c_2 * 4096 + 8 * t)) + (0))), "l"(0x12F0000000000000ULL) : "memory");
                res[c_2 * 8 + 0] = __uint_as_float(_ldv8_6_0);
                res[c_2 * 8 + 1] = __uint_as_float(_ldv8_6_1);
                res[c_2 * 8 + 2] = __uint_as_float(_ldv8_6_2);
                res[c_2 * 8 + 3] = __uint_as_float(_ldv8_6_3);
                res[c_2 * 8 + 4] = __uint_as_float(_ldv8_6_4);
                res[c_2 * 8 + 5] = __uint_as_float(_ldv8_6_5);
                res[c_2 * 8 + 6] = __uint_as_float(_ldv8_6_6);
                res[c_2 * 8 + 7] = __uint_as_float(_ldv8_6_7);
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
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            float pre = al[i] * res[i];
            res[i] = pre;
        }
        float _warp_reduce_7 = acc;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            _warp_reduce_7 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_7, offset);
        acc = _warp_reduce_7;
        if (lane == 0) {
            partials[par + warp_1] = acc;
        }
        __syncthreads();
        float total = partials[par];
        #pragma unroll
        for (int k = 1; k < 16; k++) {
            total += partials[par + k];
        }
        it += 1;
        float mean = __fmaf_rn(total, 0.00012207031f, eps);
        float _sqrt_8;
        asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(_sqrt_8) : "f"(mean));
        float inv = __frcp_rn(_sqrt_8);
        __int128_t _smem_raw_9;
        asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_9) : "r"(planes_addr + (unsigned int)(t * 16)));
        *reinterpret_cast<uint32_t*>(&be[0]) = reinterpret_cast<const uint32_t*>(&_smem_raw_9)[0];
        *reinterpret_cast<uint32_t*>(&be[(0) + 1]) = reinterpret_cast<const uint32_t*>(&_smem_raw_9)[1];
        *reinterpret_cast<uint32_t*>(&be[(0) + 2]) = reinterpret_cast<const uint32_t*>(&_smem_raw_9)[2];
        *reinterpret_cast<uint32_t*>(&be[(0) + 3]) = reinterpret_cast<const uint32_t*>(&_smem_raw_9)[3];
        __int128_t _smem_raw_10;
        asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_10) : "r"(planes_addr + (unsigned int)(t * 16 + 8192)));
        *reinterpret_cast<uint32_t*>(&be[4]) = reinterpret_cast<const uint32_t*>(&_smem_raw_10)[0];
        *reinterpret_cast<uint32_t*>(&be[(4) + 1]) = reinterpret_cast<const uint32_t*>(&_smem_raw_10)[1];
        *reinterpret_cast<uint32_t*>(&be[(4) + 2]) = reinterpret_cast<const uint32_t*>(&_smem_raw_10)[2];
        *reinterpret_cast<uint32_t*>(&be[(4) + 3]) = reinterpret_cast<const uint32_t*>(&_smem_raw_10)[3];
        __int128_t _smem_raw_11;
        asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_11) : "r"(planes_addr + (unsigned int)(t * 16 + 16384)));
        *reinterpret_cast<uint32_t*>(&be[8]) = reinterpret_cast<const uint32_t*>(&_smem_raw_11)[0];
        *reinterpret_cast<uint32_t*>(&be[(8) + 1]) = reinterpret_cast<const uint32_t*>(&_smem_raw_11)[1];
        *reinterpret_cast<uint32_t*>(&be[(8) + 2]) = reinterpret_cast<const uint32_t*>(&_smem_raw_11)[2];
        *reinterpret_cast<uint32_t*>(&be[(8) + 3]) = reinterpret_cast<const uint32_t*>(&_smem_raw_11)[3];
        __int128_t _smem_raw_12;
        asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_12) : "r"(planes_addr + (unsigned int)(t * 16 + 24576)));
        *reinterpret_cast<uint32_t*>(&be[12]) = reinterpret_cast<const uint32_t*>(&_smem_raw_12)[0];
        *reinterpret_cast<uint32_t*>(&be[(12) + 1]) = reinterpret_cast<const uint32_t*>(&_smem_raw_12)[1];
        *reinterpret_cast<uint32_t*>(&be[(12) + 2]) = reinterpret_cast<const uint32_t*>(&_smem_raw_12)[2];
        *reinterpret_cast<uint32_t*>(&be[(12) + 3]) = reinterpret_cast<const uint32_t*>(&_smem_raw_12)[3];
        #pragma unroll
        for (int c_3 = 0; c_3 < 2; c_3++) {
            float yv0_1_0 = __fmaf_rn(be[c_3 * 8], x[c_3 * 8] * inv, res[c_3 * 8]);
            y[c_3 * 8] = yv0_1_0;
            float yv0_1_1 = __fmaf_rn(be[c_3 * 8 + 1], x[c_3 * 8 + 1] * inv, res[c_3 * 8 + 1]);
            y[c_3 * 8 + 1] = yv0_1_1;
            float yv0_1_2 = __fmaf_rn(be[c_3 * 8 + 2], x[c_3 * 8 + 2] * inv, res[c_3 * 8 + 2]);
            y[c_3 * 8 + 2] = yv0_1_2;
            float yv0_1_3 = __fmaf_rn(be[c_3 * 8 + 3], x[c_3 * 8 + 3] * inv, res[c_3 * 8 + 3]);
            y[c_3 * 8 + 3] = yv0_1_3;
            float yv0_1_4 = __fmaf_rn(be[c_3 * 8 + 4], x[c_3 * 8 + 4] * inv, res[c_3 * 8 + 4]);
            y[c_3 * 8 + 4] = yv0_1_4;
            float yv0_1_5 = __fmaf_rn(be[c_3 * 8 + 5], x[c_3 * 8 + 5] * inv, res[c_3 * 8 + 5]);
            y[c_3 * 8 + 5] = yv0_1_5;
            float yv0_1_6 = __fmaf_rn(be[c_3 * 8 + 6], x[c_3 * 8 + 6] * inv, res[c_3 * 8 + 6]);
            y[c_3 * 8 + 6] = yv0_1_6;
            float yv0_1_7 = __fmaf_rn(be[c_3 * 8 + 7], x[c_3 * 8 + 7] * inv, res[c_3 * 8 + 7]);
            y[c_3 * 8 + 7] = yv0_1_7;
            {
                unsigned _stv8_13_0 = __float_as_uint(y[c_3 * 8 + 0]);
                unsigned _stv8_13_1 = __float_as_uint(y[c_3 * 8 + 1]);
                unsigned _stv8_13_2 = __float_as_uint(y[c_3 * 8 + 2]);
                unsigned _stv8_13_3 = __float_as_uint(y[c_3 * 8 + 3]);
                unsigned _stv8_13_4 = __float_as_uint(y[c_3 * 8 + 4]);
                unsigned _stv8_13_5 = __float_as_uint(y[c_3 * 8 + 5]);
                unsigned _stv8_13_6 = __float_as_uint(y[c_3 * 8 + 6]);
                unsigned _stv8_13_7 = __float_as_uint(y[c_3 * 8 + 7]);
                asm volatile(
                    "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                    :: "l"((void*)(out + (obase + (long long)(cbase + c_3 * 4096 + 8 * t)) + (0))), "r"(_stv8_13_0), "r"(_stv8_13_1), "r"(_stv8_13_2), "r"(_stv8_13_3), "r"(_stv8_13_4), "r"(_stv8_13_5), "r"(_stv8_13_6), "r"(_stv8_13_7) : "memory");
            }
        }
    }
}

} // extern "C"
