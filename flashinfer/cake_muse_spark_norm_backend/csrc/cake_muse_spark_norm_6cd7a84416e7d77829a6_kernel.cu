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
#define SMEM_STAGE_OFF 320
#define SMEM_STAGE_STAGE_BYTES 32768
#define SMEM_STAGE_STRIDE 32768
#define SMEM_TOTAL 33152
#define THREADS 512
#define SPLIT 2
#define SEG 4096
#define LAUNCH_BOUNDS_THREADS 512

#include <math_constants.h>


__device__ __forceinline__ __nv_bfloat162 __as_bf16x2(unsigned int v) {
    __nv_bfloat162_raw raw;
    raw.x = static_cast<unsigned short>(v);
    raw.y = static_cast<unsigned short>(v >> 16);
    return __nv_bfloat162(raw);
}

extern "C" {

__global__ __launch_bounds__(LAUNCH_BOUNDS_THREADS) __cluster_dims__(2,1,1) void
kernel_cake_muse_spark_norm_6cd7a84416e7d77829a6(__nv_bfloat16* __restrict__ branch, float* __restrict__ residual, __nv_bfloat16* __restrict__ weight, float* __restrict__ alpha, float* __restrict__ beta, float* __restrict__ out, int M, float eps, float gain_center, long long branch_stride_p0, long long residual_stride_p0, long long out_stride_p0)
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
    const unsigned int clusters_x = gridDim.x / 2;
    const unsigned int cluster_id = ((blockIdx.z * gridDim.y + blockIdx.y) * clusters_x) + blockIdx.x / 2;
    const unsigned int num_clusters = clusters_x * gridDim.y * gridDim.z;

    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));

    // Kernel setup ops
    float* partials = reinterpret_cast<float*>(smem_raw + 0);
    const int partials_addr = smem + 0;
    float* stage = reinterpret_cast<float*>(smem_raw + 320);
    const int stage_addr = smem + 320;

    // === Task calls (declaration order) ===
    int t = tid;
    int warp_0 = warp;
    int seg = bid & SPLIT - 1;
    int first = bid >> 1;
    int lo = seg * SEG;
    int hi = lo + SEG;
    int cbase = 0;
    int it = 0;
    unsigned int cur[8];
    float x[16];
    float al[8];
    float be[8];
    float res[8];
    float xs[8];
    float y[8];
    long long fbase = (long long)first * branch_stride_p0;
    #pragma unroll
    for (int j = 0; j < 1; j++) {
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
                : "=r"(_ldv8_0_0), "=r"(_ldv8_0_1), "=r"(_ldv8_0_2), "=r"(_ldv8_0_3), "=r"(_ldv8_0_4), "=r"(_ldv8_0_5), "=r"(_ldv8_0_6), "=r"(_ldv8_0_7) : "l"((const void*)(alpha + (lo + j * 4096 + 8 * t) + (0))) : "memory");
            al[j * 8 + 0] = __uint_as_float(_ldv8_0_0);
            al[j * 8 + 1] = __uint_as_float(_ldv8_0_1);
            al[j * 8 + 2] = __uint_as_float(_ldv8_0_2);
            al[j * 8 + 3] = __uint_as_float(_ldv8_0_3);
            al[j * 8 + 4] = __uint_as_float(_ldv8_0_4);
            al[j * 8 + 5] = __uint_as_float(_ldv8_0_5);
            al[j * 8 + 6] = __uint_as_float(_ldv8_0_6);
            al[j * 8 + 7] = __uint_as_float(_ldv8_0_7);
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
                : "=r"(_ldv8_1_0), "=r"(_ldv8_1_1), "=r"(_ldv8_1_2), "=r"(_ldv8_1_3), "=r"(_ldv8_1_4), "=r"(_ldv8_1_5), "=r"(_ldv8_1_6), "=r"(_ldv8_1_7) : "l"((const void*)(beta + (lo + j * 4096 + 8 * t) + (0))) : "memory");
            be[j * 8 + 0] = __uint_as_float(_ldv8_1_0);
            be[j * 8 + 1] = __uint_as_float(_ldv8_1_1);
            be[j * 8 + 2] = __uint_as_float(_ldv8_1_2);
            be[j * 8 + 3] = __uint_as_float(_ldv8_1_3);
            be[j * 8 + 4] = __uint_as_float(_ldv8_1_4);
            be[j * 8 + 5] = __uint_as_float(_ldv8_1_5);
            be[j * 8 + 6] = __uint_as_float(_ldv8_1_6);
            be[j * 8 + 7] = __uint_as_float(_ldv8_1_7);
        }
    }
    asm volatile("griddepcontrol.wait;" ::: "memory");
    {
        const uint4* _vptr_2 = reinterpret_cast<const uint4*>(branch + (fbase + (long long)(cbase + 8 * t)) + 0);
        uint4* _vdst_2 = reinterpret_cast<uint4*>(&cur[0]);
        #pragma unroll
        for (int _blk = 0; _blk < 1; _blk++) {
            asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                : "=r"(_vdst_2[_blk].x), "=r"(_vdst_2[_blk].y), "=r"(_vdst_2[_blk].z), "=r"(_vdst_2[_blk].w) : "l"((const void*)(_vptr_2 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
        }
    }
    {
        const uint4* _vptr_3 = reinterpret_cast<const uint4*>(branch + (fbase + (long long)(cbase + 4096 + 8 * t)) + 0);
        uint4* _vdst_3 = reinterpret_cast<uint4*>(&cur[4]);
        #pragma unroll
        for (int _blk = 0; _blk < 1; _blk++) {
            asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                : "=r"(_vdst_3[_blk].x), "=r"(_vdst_3[_blk].y), "=r"(_vdst_3[_blk].z), "=r"(_vdst_3[_blk].w) : "l"((const void*)(_vptr_3 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
        }
    }
    #pragma unroll 1
    for (int row = first; row < first + 1; row++) {
        long long row64 = (long long)row;
        long long sbase = row64 * residual_stride_p0;
        long long obase = row64 * out_stride_p0;
        int par = (it & 1) * 16;
        #pragma unroll
        for (int j_1 = 0; j_1 < 1; j_1++) {
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
                    : "=r"(_ldv8_4_0), "=r"(_ldv8_4_1), "=r"(_ldv8_4_2), "=r"(_ldv8_4_3), "=r"(_ldv8_4_4), "=r"(_ldv8_4_5), "=r"(_ldv8_4_6), "=r"(_ldv8_4_7) : "l"((const void*)(residual + (sbase + (long long)(lo + j_1 * 4096 + 8 * t)) + (0))), "l"(0x12F0000000000000ULL) : "memory");
                res[j_1 * 8 + 0] = __uint_as_float(_ldv8_4_0);
                res[j_1 * 8 + 1] = __uint_as_float(_ldv8_4_1);
                res[j_1 * 8 + 2] = __uint_as_float(_ldv8_4_2);
                res[j_1 * 8 + 3] = __uint_as_float(_ldv8_4_3);
                res[j_1 * 8 + 4] = __uint_as_float(_ldv8_4_4);
                res[j_1 * 8 + 5] = __uint_as_float(_ldv8_4_5);
                res[j_1 * 8 + 6] = __uint_as_float(_ldv8_4_6);
                res[j_1 * 8 + 7] = __uint_as_float(_ldv8_4_7);
            }
        }
        float acc = 0.0f;
        float2 _cvt_f32_0 = __bfloat1622float2(__as_bf16x2(cur[0]));
        float x0_0 = _cvt_f32_0.x;
        float x1_0 = _cvt_f32_0.y;
        x[0] = x0_0;
        x[1] = x1_0;
        acc = __fmaf_rn(x0_0, x0_0, acc);
        acc = __fmaf_rn(x1_0, x1_0, acc);
        float2 _cvt_f32_1 = __bfloat1622float2(__as_bf16x2(cur[1]));
        float x0_2 = _cvt_f32_1.x;
        float x1_2 = _cvt_f32_1.y;
        x[2] = x0_2;
        x[3] = x1_2;
        acc = __fmaf_rn(x0_2, x0_2, acc);
        acc = __fmaf_rn(x1_2, x1_2, acc);
        float2 _cvt_f32_2 = __bfloat1622float2(__as_bf16x2(cur[2]));
        float x0_4 = _cvt_f32_2.x;
        float x1_4 = _cvt_f32_2.y;
        x[4] = x0_4;
        x[5] = x1_4;
        acc = __fmaf_rn(x0_4, x0_4, acc);
        acc = __fmaf_rn(x1_4, x1_4, acc);
        float2 _cvt_f32_3 = __bfloat1622float2(__as_bf16x2(cur[3]));
        float x0_6 = _cvt_f32_3.x;
        float x1_6 = _cvt_f32_3.y;
        x[6] = x0_6;
        x[7] = x1_6;
        acc = __fmaf_rn(x0_6, x0_6, acc);
        acc = __fmaf_rn(x1_6, x1_6, acc);
        float2 _cvt_f32_4 = __bfloat1622float2(__as_bf16x2(cur[4]));
        float x0_8 = _cvt_f32_4.x;
        float x1_8 = _cvt_f32_4.y;
        x[8] = x0_8;
        x[9] = x1_8;
        acc = __fmaf_rn(x0_8, x0_8, acc);
        acc = __fmaf_rn(x1_8, x1_8, acc);
        float2 _cvt_f32_5 = __bfloat1622float2(__as_bf16x2(cur[5]));
        float x0_10 = _cvt_f32_5.x;
        float x1_10 = _cvt_f32_5.y;
        x[10] = x0_10;
        x[11] = x1_10;
        acc = __fmaf_rn(x0_10, x0_10, acc);
        acc = __fmaf_rn(x1_10, x1_10, acc);
        float2 _cvt_f32_6 = __bfloat1622float2(__as_bf16x2(cur[6]));
        float x0_12 = _cvt_f32_6.x;
        float x1_12 = _cvt_f32_6.y;
        x[12] = x0_12;
        x[13] = x1_12;
        acc = __fmaf_rn(x0_12, x0_12, acc);
        acc = __fmaf_rn(x1_12, x1_12, acc);
        float2 _cvt_f32_7 = __bfloat1622float2(__as_bf16x2(cur[7]));
        float x0_14 = _cvt_f32_7.x;
        float x1_14 = _cvt_f32_7.y;
        x[14] = x0_14;
        x[15] = x1_14;
        acc = __fmaf_rn(x0_14, x0_14, acc);
        acc = __fmaf_rn(x1_14, x1_14, acc);
        unsigned int spar = (unsigned int)((it & 1) * 16384);
        int scol0 = cbase + 8 * t;
        if (scol0 >= lo && scol0 < hi) {
            unsigned int sbyte0 = spar + (unsigned int)((scol0 - lo) * 4);
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(stage_addr + sbyte0), "f"(x[0]), "f"(x[1]), "f"(x[2]), "f"(x[3]) : "memory");
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(stage_addr + sbyte0 + 16), "f"(x[4]), "f"(x[5]), "f"(x[6]), "f"(x[7]) : "memory");
        }
        int scol1 = cbase + 4096 + 8 * t;
        if (scol1 >= lo && scol1 < hi) {
            unsigned int sbyte1 = spar + (unsigned int)((scol1 - lo) * 4);
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(stage_addr + sbyte1), "f"(x[8]), "f"(x[9]), "f"(x[10]), "f"(x[11]) : "memory");
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(stage_addr + sbyte1 + 16), "f"(x[12]), "f"(x[13]), "f"(x[14]), "f"(x[15]) : "memory");
        }
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            float pre = al[i] * res[i];
            res[i] = pre;
        }
        float _warp_reduce_5 = acc;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            _warp_reduce_5 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_5, offset);
        acc = _warp_reduce_5;
        if (lane == 0) {
            partials[par + warp_0] = acc;
        }
        __syncthreads();
        float total = partials[par];
        #pragma unroll
        for (int k = 1; k < 16; k++) {
            total += partials[par + k];
        }
        it += 1;
        float mean = __fmaf_rn(total, 0.00012207031f, eps);
        float _sqrt_6;
        asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(_sqrt_6) : "f"(mean));
        float inv = __frcp_rn(_sqrt_6);
        if (it == 1) {
            asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
        }
        #pragma unroll
        for (int j_2 = 0; j_2 < 1; j_2++) {
            __int128_t _smem_raw_7;
            asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_7) : "r"(stage_addr + (spar + (unsigned int)((j_2 * 4096 + 8 * t) * 4))));
            *reinterpret_cast<uint32_t*>(&xs[j_2 * 8]) = reinterpret_cast<const uint32_t*>(&_smem_raw_7)[0];
            *reinterpret_cast<uint32_t*>(&xs[(j_2 * 8) + 1]) = reinterpret_cast<const uint32_t*>(&_smem_raw_7)[1];
            *reinterpret_cast<uint32_t*>(&xs[(j_2 * 8) + 2]) = reinterpret_cast<const uint32_t*>(&_smem_raw_7)[2];
            *reinterpret_cast<uint32_t*>(&xs[(j_2 * 8) + 3]) = reinterpret_cast<const uint32_t*>(&_smem_raw_7)[3];
            __int128_t _smem_raw_8;
            asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_8) : "r"(stage_addr + (spar + (unsigned int)((j_2 * 4096 + 8 * t) * 4)) + 16));
            *reinterpret_cast<uint32_t*>(&xs[j_2 * 8 + 4]) = reinterpret_cast<const uint32_t*>(&_smem_raw_8)[0];
            *reinterpret_cast<uint32_t*>(&xs[(j_2 * 8 + 4) + 1]) = reinterpret_cast<const uint32_t*>(&_smem_raw_8)[1];
            *reinterpret_cast<uint32_t*>(&xs[(j_2 * 8 + 4) + 2]) = reinterpret_cast<const uint32_t*>(&_smem_raw_8)[2];
            *reinterpret_cast<uint32_t*>(&xs[(j_2 * 8 + 4) + 3]) = reinterpret_cast<const uint32_t*>(&_smem_raw_8)[3];
            #pragma unroll
            for (int v = 0; v < 8; v++) {
                float yv = __fmaf_rn(be[j_2 * 8 + v], xs[j_2 * 8 + v] * inv, res[j_2 * 8 + v]);
                y[j_2 * 8 + v] = yv;
            }
            {
                unsigned _stv8_9_0 = __float_as_uint(y[j_2 * 8 + 0]);
                unsigned _stv8_9_1 = __float_as_uint(y[j_2 * 8 + 1]);
                unsigned _stv8_9_2 = __float_as_uint(y[j_2 * 8 + 2]);
                unsigned _stv8_9_3 = __float_as_uint(y[j_2 * 8 + 3]);
                unsigned _stv8_9_4 = __float_as_uint(y[j_2 * 8 + 4]);
                unsigned _stv8_9_5 = __float_as_uint(y[j_2 * 8 + 5]);
                unsigned _stv8_9_6 = __float_as_uint(y[j_2 * 8 + 6]);
                unsigned _stv8_9_7 = __float_as_uint(y[j_2 * 8 + 7]);
                asm volatile(
                    "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                    :: "l"((void*)(out + (obase + (long long)(lo + j_2 * 4096 + 8 * t)) + (0))), "r"(_stv8_9_0), "r"(_stv8_9_1), "r"(_stv8_9_2), "r"(_stv8_9_3), "r"(_stv8_9_4), "r"(_stv8_9_5), "r"(_stv8_9_6), "r"(_stv8_9_7) : "memory");
            }
        }
    }
}

} // extern "C"
