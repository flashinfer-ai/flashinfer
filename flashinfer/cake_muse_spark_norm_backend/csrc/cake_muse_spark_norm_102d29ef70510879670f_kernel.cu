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
#define SMEM_PARTIALS_STAGE_BYTES 64
#define SMEM_PARTIALS_STRIDE 64
#define SMEM_STAGE_OFF 320
#define SMEM_STAGE_STAGE_BYTES 8192
#define SMEM_STAGE_STRIDE 8192
#define SMEM_TOTAL 8576
#define THREADS 256
#define SPLIT 8
#define SEG 1024
#define LAUNCH_BOUNDS_THREADS 256

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

__global__ __launch_bounds__(LAUNCH_BOUNDS_THREADS) __cluster_dims__(2,1,1) void
kernel_cake_muse_spark_norm_102d29ef70510879670f(__nv_bfloat16* __restrict__ branch, float* __restrict__ residual, __nv_bfloat16* __restrict__ weight, float* __restrict__ alpha, float* __restrict__ beta, float* __restrict__ out, int M, float eps, float gain_center)
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
    int warp_1 = t >> 5;
    int seg = bid & SPLIT - 1;
    int first = bid >> 3;
    int lo = seg * SEG;
    int hi = lo + SEG;
    int cbase = 0;
    int it = 0;
    unsigned int cur[16];
    float x[32];
    float al[4];
    float be[4];
    float res[4];
    float xs[4];
    float y[4];
    float pv[8];
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
        const uint4* _vptr_1 = reinterpret_cast<const uint4*>(branch + (fbase + (long long)(cbase + 2048 + 8 * t)) + 0);
        uint4* _vdst_1 = reinterpret_cast<uint4*>(&cur[4]);
        #pragma unroll
        for (int _blk = 0; _blk < 1; _blk++) {
            asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                : "=r"(_vdst_1[_blk].x), "=r"(_vdst_1[_blk].y), "=r"(_vdst_1[_blk].z), "=r"(_vdst_1[_blk].w) : "l"((const void*)(_vptr_1 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
        }
    }
    {
        const uint4* _vptr_2 = reinterpret_cast<const uint4*>(branch + (fbase + (long long)(cbase + 4096 + 8 * t)) + 0);
        uint4* _vdst_2 = reinterpret_cast<uint4*>(&cur[8]);
        #pragma unroll
        for (int _blk = 0; _blk < 1; _blk++) {
            asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                : "=r"(_vdst_2[_blk].x), "=r"(_vdst_2[_blk].y), "=r"(_vdst_2[_blk].z), "=r"(_vdst_2[_blk].w) : "l"((const void*)(_vptr_2 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
        }
    }
    {
        const uint4* _vptr_3 = reinterpret_cast<const uint4*>(branch + (fbase + (long long)(cbase + 6144 + 8 * t)) + 0);
        uint4* _vdst_3 = reinterpret_cast<uint4*>(&cur[12]);
        #pragma unroll
        for (int _blk = 0; _blk < 1; _blk++) {
            asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                : "=r"(_vdst_3[_blk].x), "=r"(_vdst_3[_blk].y), "=r"(_vdst_3[_blk].z), "=r"(_vdst_3[_blk].w) : "l"((const void*)(_vptr_3 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
        }
    }
    #pragma unroll
    for (int j = 0; j < 1; j++) {
        {
            unsigned _v4_4_0;
            unsigned _v4_4_1;
            unsigned _v4_4_2;
            unsigned _v4_4_3;
            asm volatile("ld.global.nc.v4.b32 {%0, %1, %2, %3}, [%4];"
                : "=r"(_v4_4_0), "=r"(_v4_4_1), "=r"(_v4_4_2), "=r"(_v4_4_3) : "l"((const void*)(alpha + (lo + j * 1024 + 4 * t) + (0))) : "memory");
            al[j * 4 + 0] = __uint_as_float(_v4_4_0);
            al[j * 4 + 1] = __uint_as_float(_v4_4_1);
            al[j * 4 + 2] = __uint_as_float(_v4_4_2);
            al[j * 4 + 3] = __uint_as_float(_v4_4_3);
        }
        {
            unsigned _v4_5_0;
            unsigned _v4_5_1;
            unsigned _v4_5_2;
            unsigned _v4_5_3;
            asm volatile("ld.global.nc.v4.b32 {%0, %1, %2, %3}, [%4];"
                : "=r"(_v4_5_0), "=r"(_v4_5_1), "=r"(_v4_5_2), "=r"(_v4_5_3) : "l"((const void*)(beta + (lo + j * 1024 + 4 * t) + (0))) : "memory");
            be[j * 4 + 0] = __uint_as_float(_v4_5_0);
            be[j * 4 + 1] = __uint_as_float(_v4_5_1);
            be[j * 4 + 2] = __uint_as_float(_v4_5_2);
            be[j * 4 + 3] = __uint_as_float(_v4_5_3);
        }
    }
    #pragma unroll 1
    for (int row = first; row < first + 1; row++) {
        long long row64 = (long long)row;
        long long sbase = row64 * 8192;
        long long obase = row64 * 8192;
        int par = (it & 1) * 8;
        #pragma unroll
        for (int j_1 = 0; j_1 < 1; j_1++) {
            {
                unsigned _v4_6_0;
                unsigned _v4_6_1;
                unsigned _v4_6_2;
                unsigned _v4_6_3;
                asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                    : "=r"(_v4_6_0), "=r"(_v4_6_1), "=r"(_v4_6_2), "=r"(_v4_6_3) : "l"((const void*)(residual + (sbase + (long long)(lo + j_1 * 1024 + 4 * t)) + (0))), "l"(0x12F0000000000000ULL) : "memory");
                res[j_1 * 4 + 0] = __uint_as_float(_v4_6_0);
                res[j_1 * 4 + 1] = __uint_as_float(_v4_6_1);
                res[j_1 * 4 + 2] = __uint_as_float(_v4_6_2);
                res[j_1 * 4 + 3] = __uint_as_float(_v4_6_3);
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
            float2 x2_16 = cake_bf16x2_to_f32x2(__as_bf16x2(cur[8]));
            x[16] = x2_16.x;
            x[17] = x2_16.y;
            float2 acc2_9 = fma_f32x2_rn_noftz(x2_16, x2_16, acc2_8);
            float2 x2_18 = cake_bf16x2_to_f32x2(__as_bf16x2(cur[9]));
            x[18] = x2_18.x;
            x[19] = x2_18.y;
            float2 acc2_10 = fma_f32x2_rn_noftz(x2_18, x2_18, acc2_9);
            float2 x2_20 = cake_bf16x2_to_f32x2(__as_bf16x2(cur[10]));
            x[20] = x2_20.x;
            x[21] = x2_20.y;
            float2 acc2_11 = fma_f32x2_rn_noftz(x2_20, x2_20, acc2_10);
            float2 x2_22 = cake_bf16x2_to_f32x2(__as_bf16x2(cur[11]));
            x[22] = x2_22.x;
            x[23] = x2_22.y;
            float2 acc2_12 = fma_f32x2_rn_noftz(x2_22, x2_22, acc2_11);
            float2 x2_24 = cake_bf16x2_to_f32x2(__as_bf16x2(cur[12]));
            x[24] = x2_24.x;
            x[25] = x2_24.y;
            float2 acc2_13 = fma_f32x2_rn_noftz(x2_24, x2_24, acc2_12);
            float2 x2_26 = cake_bf16x2_to_f32x2(__as_bf16x2(cur[13]));
            x[26] = x2_26.x;
            x[27] = x2_26.y;
            float2 acc2_14 = fma_f32x2_rn_noftz(x2_26, x2_26, acc2_13);
            float2 x2_28 = cake_bf16x2_to_f32x2(__as_bf16x2(cur[14]));
            x[28] = x2_28.x;
            x[29] = x2_28.y;
            float2 acc2_15 = fma_f32x2_rn_noftz(x2_28, x2_28, acc2_14);
            float2 x2_30 = cake_bf16x2_to_f32x2(__as_bf16x2(cur[15]));
            x[30] = x2_30.x;
            x[31] = x2_30.y;
            float2 acc2_16 = fma_f32x2_rn_noftz(x2_30, x2_30, acc2_15);
            acc = acc + acc2_16.x;
            acc = acc + acc2_16.y;
        }
        unsigned int spar = (unsigned int)((it & 1) * 4096);
        int scol0_1_0 = cbase + 8 * t;
        if (scol0_1_0 >= lo && scol0_1_0 < hi) {
            unsigned int sbyte0_1_0 = spar + (unsigned int)((scol0_1_0 - lo) * 4);
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(stage_addr + sbyte0_1_0), "f"(x[0]), "f"(x[1]), "f"(x[2]), "f"(x[3]) : "memory");
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(stage_addr + sbyte0_1_0 + 16), "f"(x[4]), "f"(x[5]), "f"(x[6]), "f"(x[7]) : "memory");
        }
        int scol0_1_1 = cbase + 2048 + 8 * t;
        if (scol0_1_1 >= lo && scol0_1_1 < hi) {
            unsigned int sbyte0_1_1 = spar + (unsigned int)((scol0_1_1 - lo) * 4);
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(stage_addr + sbyte0_1_1), "f"(x[8]), "f"(x[9]), "f"(x[10]), "f"(x[11]) : "memory");
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(stage_addr + sbyte0_1_1 + 16), "f"(x[12]), "f"(x[13]), "f"(x[14]), "f"(x[15]) : "memory");
        }
        int scol0_1_2 = cbase + 4096 + 8 * t;
        if (scol0_1_2 >= lo && scol0_1_2 < hi) {
            unsigned int sbyte0_1_2 = spar + (unsigned int)((scol0_1_2 - lo) * 4);
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(stage_addr + sbyte0_1_2), "f"(x[16]), "f"(x[17]), "f"(x[18]), "f"(x[19]) : "memory");
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(stage_addr + sbyte0_1_2 + 16), "f"(x[20]), "f"(x[21]), "f"(x[22]), "f"(x[23]) : "memory");
        }
        int scol0_1_3 = cbase + 6144 + 8 * t;
        if (scol0_1_3 >= lo && scol0_1_3 < hi) {
            unsigned int sbyte0_1_3 = spar + (unsigned int)((scol0_1_3 - lo) * 4);
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(stage_addr + sbyte0_1_3), "f"(x[24]), "f"(x[25]), "f"(x[26]), "f"(x[27]) : "memory");
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(stage_addr + sbyte0_1_3 + 16), "f"(x[28]), "f"(x[29]), "f"(x[30]), "f"(x[31]) : "memory");
        }
        #pragma unroll
        for (int i = 0; i < 4; i++) {
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
        unsigned int pbyte = (unsigned int)((it & 1) * 32);
        __int128_t _smem_raw_8;
        asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_8) : "r"(partials_addr + pbyte));
        *reinterpret_cast<uint32_t*>(&pv[0]) = reinterpret_cast<const uint32_t*>(&_smem_raw_8)[0];
        *reinterpret_cast<uint32_t*>(&pv[(0) + 1]) = reinterpret_cast<const uint32_t*>(&_smem_raw_8)[1];
        *reinterpret_cast<uint32_t*>(&pv[(0) + 2]) = reinterpret_cast<const uint32_t*>(&_smem_raw_8)[2];
        *reinterpret_cast<uint32_t*>(&pv[(0) + 3]) = reinterpret_cast<const uint32_t*>(&_smem_raw_8)[3];
        __int128_t _smem_raw_9;
        asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_9) : "r"(partials_addr + pbyte + 16));
        *reinterpret_cast<uint32_t*>(&pv[4]) = reinterpret_cast<const uint32_t*>(&_smem_raw_9)[0];
        *reinterpret_cast<uint32_t*>(&pv[(4) + 1]) = reinterpret_cast<const uint32_t*>(&_smem_raw_9)[1];
        *reinterpret_cast<uint32_t*>(&pv[(4) + 2]) = reinterpret_cast<const uint32_t*>(&_smem_raw_9)[2];
        *reinterpret_cast<uint32_t*>(&pv[(4) + 3]) = reinterpret_cast<const uint32_t*>(&_smem_raw_9)[3];
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
        for (int j_2 = 0; j_2 < 1; j_2++) {
            __int128_t _smem_raw_10;
            asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_10) : "r"(stage_addr + (spar + (unsigned int)((j_2 * 1024 + 4 * t) * 4))));
            *reinterpret_cast<uint32_t*>(&xs[j_2 * 4]) = reinterpret_cast<const uint32_t*>(&_smem_raw_10)[0];
            *reinterpret_cast<uint32_t*>(&xs[(j_2 * 4) + 1]) = reinterpret_cast<const uint32_t*>(&_smem_raw_10)[1];
            *reinterpret_cast<uint32_t*>(&xs[(j_2 * 4) + 2]) = reinterpret_cast<const uint32_t*>(&_smem_raw_10)[2];
            *reinterpret_cast<uint32_t*>(&xs[(j_2 * 4) + 3]) = reinterpret_cast<const uint32_t*>(&_smem_raw_10)[3];
            float2 _fma_0 = fma_f32x2_rn_noftz(make_float2(be[j_2 * 4], be[j_2 * 4 + 1]), mul_f32x2_noftz(make_float2(xs[j_2 * 4], xs[j_2 * 4 + 1]), inv2), make_float2(res[j_2 * 4], res[j_2 * 4 + 1]));
            y[j_2 * 4] = _fma_0.x;
            y[j_2 * 4 + 1] = _fma_0.y;
            float2 _fma_1 = fma_f32x2_rn_noftz(make_float2(be[j_2 * 4 + 2], be[j_2 * 4 + 2 + 1]), mul_f32x2_noftz(make_float2(xs[j_2 * 4 + 2], xs[j_2 * 4 + 2 + 1]), inv2), make_float2(res[j_2 * 4 + 2], res[j_2 * 4 + 2 + 1]));
            y[j_2 * 4 + 2] = _fma_1.x;
            y[j_2 * 4 + 2 + 1] = _fma_1.y;
            {
                float4 _v4 = make_float4(y[j_2 * 4 + 0], y[j_2 * 4 + 1], y[j_2 * 4 + 2], y[j_2 * 4 + 3]);
                *reinterpret_cast<float4*>((out + (obase + (long long)(lo + j_2 * 1024 + 4 * t))) + 0) = _v4;
            }
        }
    }
}

} // extern "C"
