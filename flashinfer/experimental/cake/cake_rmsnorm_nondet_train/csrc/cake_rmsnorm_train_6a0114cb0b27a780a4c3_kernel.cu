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

#define LAUNCH_MIN_BLOCKS 1

#include <math_constants.h>

__device__ __forceinline__ float2 add_f32x2_noftz(float2 a, float2 b) {
    float2 r;
    asm("add.rn.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(unsigned long long*)&a), "l"(*(unsigned long long*)&b));
    return r;
}

__device__ __forceinline__ float2 sub_f32x2_noftz(float2 a, float2 b) {
    float2 r;
    asm("sub.f32x2 %0, %1, %2;"
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

extern "C" {

__global__ __launch_bounds__(256, LAUNCH_MIN_BLOCKS) void
kernel_cake_rmsnorm_train_6a0114cb0b27a780a4c3(__nv_bfloat16* __restrict__ g, __nv_bfloat16* __restrict__ x, float* __restrict__ w, float* __restrict__ r, __nv_bfloat16* __restrict__ dx, float* __restrict__ dw, double* __restrict__ partial, unsigned int* __restrict__ counters, __nv_bfloat16* __restrict__ g_h, int T, int n_chunks, int rows_per_chunk, long long g_stride_p0, long long x_stride_p0, long long g_h_stride_p0)
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

    // Prologue
    float* pairs = reinterpret_cast<float*>(smem_raw + 0);
    const int pairs_addr = smem + 0;

    // === Task calls (declaration order) ===
    asm volatile("griddepcontrol.wait;" ::: "memory");
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    int rg = warp;
    int ct = tid - rg * 32;
    int col = ct * 8;
    float ww[8];
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
            : "=r"(_ldv8_0_0), "=r"(_ldv8_0_1), "=r"(_ldv8_0_2), "=r"(_ldv8_0_3), "=r"(_ldv8_0_4), "=r"(_ldv8_0_5), "=r"(_ldv8_0_6), "=r"(_ldv8_0_7) : "l"((const void*)(w + ((unsigned long long)col))) : "memory");
        ww[0 + 0] = __uint_as_float(_ldv8_0_0);
        ww[0 + 1] = __uint_as_float(_ldv8_0_1);
        ww[0 + 2] = __uint_as_float(_ldv8_0_2);
        ww[0 + 3] = __uint_as_float(_ldv8_0_3);
        ww[0 + 4] = __uint_as_float(_ldv8_0_4);
        ww[0 + 5] = __uint_as_float(_ldv8_0_5);
        ww[0 + 6] = __uint_as_float(_ldv8_0_6);
        ww[0 + 7] = __uint_as_float(_ldv8_0_7);
    }
    unsigned int gw[32];
    unsigned int xw[32];
    unsigned int hw[1];
    float rr[8];
    int lrow_0 = rg;
    if (lrow_0 < T) {
        rr[0] = r[lrow_0];
        {
            const uint4* _vptr_1 = reinterpret_cast<const uint4*>(g + (unsigned long long)lrow_0 * (unsigned long long)g_stride_p0 + (unsigned long long)col);
            uint4* _vdst_1 = reinterpret_cast<uint4*>(&gw[0]);
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vdst_1[_blk] = _vptr_1[_blk];
            }
        }
        {
            const uint4* _vptr_2 = reinterpret_cast<const uint4*>(x + (unsigned long long)lrow_0 * (unsigned long long)x_stride_p0 + (unsigned long long)col);
            uint4* _vdst_2 = reinterpret_cast<uint4*>(&xw[0]);
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vdst_2[_blk] = _vptr_2[_blk];
            }
        }
    }
    int lrow_1 = 8 + rg;
    if (lrow_1 < T) {
        rr[1] = r[lrow_1];
        {
            const uint4* _vptr_3 = reinterpret_cast<const uint4*>(g + (unsigned long long)lrow_1 * (unsigned long long)g_stride_p0 + (unsigned long long)col);
            uint4* _vdst_3 = reinterpret_cast<uint4*>(&gw[4]);
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vdst_3[_blk] = _vptr_3[_blk];
            }
        }
        {
            const uint4* _vptr_4 = reinterpret_cast<const uint4*>(x + (unsigned long long)lrow_1 * (unsigned long long)x_stride_p0 + (unsigned long long)col);
            uint4* _vdst_4 = reinterpret_cast<uint4*>(&xw[4]);
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vdst_4[_blk] = _vptr_4[_blk];
            }
        }
    }
    int lrow_2 = 16 + rg;
    if (lrow_2 < T) {
        rr[2] = r[lrow_2];
        {
            const uint4* _vptr_5 = reinterpret_cast<const uint4*>(g + (unsigned long long)lrow_2 * (unsigned long long)g_stride_p0 + (unsigned long long)col);
            uint4* _vdst_5 = reinterpret_cast<uint4*>(&gw[8]);
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vdst_5[_blk] = _vptr_5[_blk];
            }
        }
        {
            const uint4* _vptr_6 = reinterpret_cast<const uint4*>(x + (unsigned long long)lrow_2 * (unsigned long long)x_stride_p0 + (unsigned long long)col);
            uint4* _vdst_6 = reinterpret_cast<uint4*>(&xw[8]);
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vdst_6[_blk] = _vptr_6[_blk];
            }
        }
    }
    int lrow_3 = 24 + rg;
    if (lrow_3 < T) {
        rr[3] = r[lrow_3];
        {
            const uint4* _vptr_7 = reinterpret_cast<const uint4*>(g + (unsigned long long)lrow_3 * (unsigned long long)g_stride_p0 + (unsigned long long)col);
            uint4* _vdst_7 = reinterpret_cast<uint4*>(&gw[12]);
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vdst_7[_blk] = _vptr_7[_blk];
            }
        }
        {
            const uint4* _vptr_8 = reinterpret_cast<const uint4*>(x + (unsigned long long)lrow_3 * (unsigned long long)x_stride_p0 + (unsigned long long)col);
            uint4* _vdst_8 = reinterpret_cast<uint4*>(&xw[12]);
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vdst_8[_blk] = _vptr_8[_blk];
            }
        }
    }
    int lrow_4 = 32 + rg;
    if (lrow_4 < T) {
        rr[4] = r[lrow_4];
        {
            const uint4* _vptr_9 = reinterpret_cast<const uint4*>(g + (unsigned long long)lrow_4 * (unsigned long long)g_stride_p0 + (unsigned long long)col);
            uint4* _vdst_9 = reinterpret_cast<uint4*>(&gw[16]);
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vdst_9[_blk] = _vptr_9[_blk];
            }
        }
        {
            const uint4* _vptr_10 = reinterpret_cast<const uint4*>(x + (unsigned long long)lrow_4 * (unsigned long long)x_stride_p0 + (unsigned long long)col);
            uint4* _vdst_10 = reinterpret_cast<uint4*>(&xw[16]);
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vdst_10[_blk] = _vptr_10[_blk];
            }
        }
    }
    int lrow_5 = 40 + rg;
    if (lrow_5 < T) {
        rr[5] = r[lrow_5];
        {
            const uint4* _vptr_11 = reinterpret_cast<const uint4*>(g + (unsigned long long)lrow_5 * (unsigned long long)g_stride_p0 + (unsigned long long)col);
            uint4* _vdst_11 = reinterpret_cast<uint4*>(&gw[20]);
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vdst_11[_blk] = _vptr_11[_blk];
            }
        }
        {
            const uint4* _vptr_12 = reinterpret_cast<const uint4*>(x + (unsigned long long)lrow_5 * (unsigned long long)x_stride_p0 + (unsigned long long)col);
            uint4* _vdst_12 = reinterpret_cast<uint4*>(&xw[20]);
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vdst_12[_blk] = _vptr_12[_blk];
            }
        }
    }
    int lrow_6 = 48 + rg;
    if (lrow_6 < T) {
        rr[6] = r[lrow_6];
        {
            const uint4* _vptr_13 = reinterpret_cast<const uint4*>(g + (unsigned long long)lrow_6 * (unsigned long long)g_stride_p0 + (unsigned long long)col);
            uint4* _vdst_13 = reinterpret_cast<uint4*>(&gw[24]);
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vdst_13[_blk] = _vptr_13[_blk];
            }
        }
        {
            const uint4* _vptr_14 = reinterpret_cast<const uint4*>(x + (unsigned long long)lrow_6 * (unsigned long long)x_stride_p0 + (unsigned long long)col);
            uint4* _vdst_14 = reinterpret_cast<uint4*>(&xw[24]);
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vdst_14[_blk] = _vptr_14[_blk];
            }
        }
    }
    int lrow_7 = 56 + rg;
    if (lrow_7 < T) {
        rr[7] = r[lrow_7];
        {
            const uint4* _vptr_15 = reinterpret_cast<const uint4*>(g + (unsigned long long)lrow_7 * (unsigned long long)g_stride_p0 + (unsigned long long)col);
            uint4* _vdst_15 = reinterpret_cast<uint4*>(&gw[28]);
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vdst_15[_blk] = _vptr_15[_blk];
            }
        }
        {
            const uint4* _vptr_16 = reinterpret_cast<const uint4*>(x + (unsigned long long)lrow_7 * (unsigned long long)x_stride_p0 + (unsigned long long)col);
            uint4* _vdst_16 = reinterpret_cast<uint4*>(&xw[28]);
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vdst_16[_blk] = _vptr_16[_blk];
            }
        }
    }
    float acc[8];
    float comp[8];
    float dq[8];
    acc[0] = 0.0f;
    comp[0] = 0.0f;
    acc[1] = 0.0f;
    comp[1] = 0.0f;
    acc[2] = 0.0f;
    comp[2] = 0.0f;
    acc[3] = 0.0f;
    comp[3] = 0.0f;
    acc[4] = 0.0f;
    comp[4] = 0.0f;
    acc[5] = 0.0f;
    comp[5] = 0.0f;
    acc[6] = 0.0f;
    comp[6] = 0.0f;
    acc[7] = 0.0f;
    comp[7] = 0.0f;
    dq[0] = 0.0f;
    if (rg < T) {
        float2 rr2 = make_float2(rr[0], rr[0]);
        float dot = 0.0f;
        float g0_0 = __uint_as_float(gw[0] << 16);
        float g1_0 = __uint_as_float(gw[0] & 4294901760u);
        float x0_0 = __uint_as_float(xw[0] << 16);
        float x1_0 = __uint_as_float(xw[0] & 4294901760u);
        dot = __fmaf_rn(g0_0 * ww[0], x0_0, dot);
        dot = __fmaf_rn(g1_0 * ww[1], x1_0, dot);
        float2 _f32x2_mul_a_17 = make_float2(g0_0, g1_0);
        float2 _f32x2_mul_b_18 = make_float2(x0_0, x1_0);
        float2 gx;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx) : "l"(*(const unsigned long long*)&_f32x2_mul_a_17), "l"(*(const unsigned long long*)&_f32x2_mul_b_18));
        float2 p;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&p) : "l"(*(const unsigned long long*)&gx), "l"(*(const unsigned long long*)&rr2));
        float2 a = make_float2(acc[0], acc[1]);
        float2 s = add_f32x2_noftz(a, p);
        float2 bn = sub_f32x2_noftz(a, s);
        float2 av = add_f32x2_noftz(s, bn);
        float2 db = fma_f32x2_rn_noftz(gx, rr2, bn);
        float2 c = add_f32x2_noftz(add_f32x2_noftz(make_float2(comp[0], comp[1]), sub_f32x2_noftz(a, av)), db);
        acc[0] = s.x;
        acc[1] = s.y;
        comp[0] = c.x;
        comp[1] = c.y;
        float g0_1 = __uint_as_float(gw[1] << 16);
        float g1_1 = __uint_as_float(gw[1] & 4294901760u);
        float x0_1 = __uint_as_float(xw[1] << 16);
        float x1_1 = __uint_as_float(xw[1] & 4294901760u);
        dot = __fmaf_rn(g0_1 * ww[2], x0_1, dot);
        dot = __fmaf_rn(g1_1 * ww[3], x1_1, dot);
        float2 _f32x2_mul_a_19 = make_float2(g0_1, g1_1);
        float2 _f32x2_mul_b_20 = make_float2(x0_1, x1_1);
        float2 gx_0;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx_0) : "l"(*(const unsigned long long*)&_f32x2_mul_a_19), "l"(*(const unsigned long long*)&_f32x2_mul_b_20));
        float2 p_0;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&p_0) : "l"(*(const unsigned long long*)&gx_0), "l"(*(const unsigned long long*)&rr2));
        float2 a_0 = make_float2(acc[2], acc[3]);
        float2 s_0 = add_f32x2_noftz(a_0, p_0);
        float2 bn_0 = sub_f32x2_noftz(a_0, s_0);
        float2 av_0 = add_f32x2_noftz(s_0, bn_0);
        float2 db_0 = fma_f32x2_rn_noftz(gx_0, rr2, bn_0);
        float2 c_0 = add_f32x2_noftz(add_f32x2_noftz(make_float2(comp[2], comp[3]), sub_f32x2_noftz(a_0, av_0)), db_0);
        acc[2] = s_0.x;
        acc[3] = s_0.y;
        comp[2] = c_0.x;
        comp[3] = c_0.y;
        float g0_2 = __uint_as_float(gw[2] << 16);
        float g1_2 = __uint_as_float(gw[2] & 4294901760u);
        float x0_2 = __uint_as_float(xw[2] << 16);
        float x1_2 = __uint_as_float(xw[2] & 4294901760u);
        dot = __fmaf_rn(g0_2 * ww[4], x0_2, dot);
        dot = __fmaf_rn(g1_2 * ww[5], x1_2, dot);
        float2 _f32x2_mul_a_21 = make_float2(g0_2, g1_2);
        float2 _f32x2_mul_b_22 = make_float2(x0_2, x1_2);
        float2 gx_1;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_21), "l"(*(const unsigned long long*)&_f32x2_mul_b_22));
        float2 p_1;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&p_1) : "l"(*(const unsigned long long*)&gx_1), "l"(*(const unsigned long long*)&rr2));
        float2 a_1 = make_float2(acc[4], acc[5]);
        float2 s_1 = add_f32x2_noftz(a_1, p_1);
        float2 bn_1 = sub_f32x2_noftz(a_1, s_1);
        float2 av_1 = add_f32x2_noftz(s_1, bn_1);
        float2 db_1 = fma_f32x2_rn_noftz(gx_1, rr2, bn_1);
        float2 c_1 = add_f32x2_noftz(add_f32x2_noftz(make_float2(comp[4], comp[5]), sub_f32x2_noftz(a_1, av_1)), db_1);
        acc[4] = s_1.x;
        acc[5] = s_1.y;
        comp[4] = c_1.x;
        comp[5] = c_1.y;
        float g0_3 = __uint_as_float(gw[3] << 16);
        float g1_3 = __uint_as_float(gw[3] & 4294901760u);
        float x0_3 = __uint_as_float(xw[3] << 16);
        float x1_3 = __uint_as_float(xw[3] & 4294901760u);
        dot = __fmaf_rn(g0_3 * ww[6], x0_3, dot);
        dot = __fmaf_rn(g1_3 * ww[7], x1_3, dot);
        float2 _f32x2_mul_a_23 = make_float2(g0_3, g1_3);
        float2 _f32x2_mul_b_24 = make_float2(x0_3, x1_3);
        float2 gx_2;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx_2) : "l"(*(const unsigned long long*)&_f32x2_mul_a_23), "l"(*(const unsigned long long*)&_f32x2_mul_b_24));
        float2 p_2;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&p_2) : "l"(*(const unsigned long long*)&gx_2), "l"(*(const unsigned long long*)&rr2));
        float2 a_2 = make_float2(acc[6], acc[7]);
        float2 s_2 = add_f32x2_noftz(a_2, p_2);
        float2 bn_2 = sub_f32x2_noftz(a_2, s_2);
        float2 av_2 = add_f32x2_noftz(s_2, bn_2);
        float2 db_2 = fma_f32x2_rn_noftz(gx_2, rr2, bn_2);
        float2 c_2 = add_f32x2_noftz(add_f32x2_noftz(make_float2(comp[6], comp[7]), sub_f32x2_noftz(a_2, av_2)), db_2);
        acc[6] = s_2.x;
        acc[7] = s_2.y;
        comp[6] = c_2.x;
        comp[7] = c_2.y;
        dq[0] = dot;
    }
    dq[1] = 0.0f;
    if (8 + rg < T) {
        float2 rr2_1 = make_float2(rr[1], rr[1]);
        float dot_1 = 0.0f;
        float g0_4 = __uint_as_float(gw[4] << 16);
        float g1_4 = __uint_as_float(gw[4] & 4294901760u);
        float x0_4 = __uint_as_float(xw[4] << 16);
        float x1_4 = __uint_as_float(xw[4] & 4294901760u);
        dot_1 = __fmaf_rn(g0_4 * ww[0], x0_4, dot_1);
        dot_1 = __fmaf_rn(g1_4 * ww[1], x1_4, dot_1);
        float2 _f32x2_mul_a_25 = make_float2(g0_4, g1_4);
        float2 _f32x2_mul_b_26 = make_float2(x0_4, x1_4);
        float2 gx_3;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx_3) : "l"(*(const unsigned long long*)&_f32x2_mul_a_25), "l"(*(const unsigned long long*)&_f32x2_mul_b_26));
        float2 p_3;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&p_3) : "l"(*(const unsigned long long*)&gx_3), "l"(*(const unsigned long long*)&rr2_1));
        float2 a_3 = make_float2(acc[0], acc[1]);
        float2 s_3 = add_f32x2_noftz(a_3, p_3);
        float2 bn_3 = sub_f32x2_noftz(a_3, s_3);
        float2 av_3 = add_f32x2_noftz(s_3, bn_3);
        float2 db_3 = fma_f32x2_rn_noftz(gx_3, rr2_1, bn_3);
        float2 c_3 = add_f32x2_noftz(add_f32x2_noftz(make_float2(comp[0], comp[1]), sub_f32x2_noftz(a_3, av_3)), db_3);
        acc[0] = s_3.x;
        acc[1] = s_3.y;
        comp[0] = c_3.x;
        comp[1] = c_3.y;
        float g0_5 = __uint_as_float(gw[5] << 16);
        float g1_5 = __uint_as_float(gw[5] & 4294901760u);
        float x0_5 = __uint_as_float(xw[5] << 16);
        float x1_5 = __uint_as_float(xw[5] & 4294901760u);
        dot_1 = __fmaf_rn(g0_5 * ww[2], x0_5, dot_1);
        dot_1 = __fmaf_rn(g1_5 * ww[3], x1_5, dot_1);
        float2 _f32x2_mul_a_27 = make_float2(g0_5, g1_5);
        float2 _f32x2_mul_b_28 = make_float2(x0_5, x1_5);
        float2 gx_3_1;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx_3_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_27), "l"(*(const unsigned long long*)&_f32x2_mul_b_28));
        float2 p_3_1;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&p_3_1) : "l"(*(const unsigned long long*)&gx_3_1), "l"(*(const unsigned long long*)&rr2_1));
        float2 a_3_1 = make_float2(acc[2], acc[3]);
        float2 s_3_1 = add_f32x2_noftz(a_3_1, p_3_1);
        float2 bn_3_1 = sub_f32x2_noftz(a_3_1, s_3_1);
        float2 av_3_1 = add_f32x2_noftz(s_3_1, bn_3_1);
        float2 db_3_1 = fma_f32x2_rn_noftz(gx_3_1, rr2_1, bn_3_1);
        float2 c_3_1 = add_f32x2_noftz(add_f32x2_noftz(make_float2(comp[2], comp[3]), sub_f32x2_noftz(a_3_1, av_3_1)), db_3_1);
        acc[2] = s_3_1.x;
        acc[3] = s_3_1.y;
        comp[2] = c_3_1.x;
        comp[3] = c_3_1.y;
        float g0_6 = __uint_as_float(gw[6] << 16);
        float g1_6 = __uint_as_float(gw[6] & 4294901760u);
        float x0_6 = __uint_as_float(xw[6] << 16);
        float x1_6 = __uint_as_float(xw[6] & 4294901760u);
        dot_1 = __fmaf_rn(g0_6 * ww[4], x0_6, dot_1);
        dot_1 = __fmaf_rn(g1_6 * ww[5], x1_6, dot_1);
        float2 _f32x2_mul_a_29 = make_float2(g0_6, g1_6);
        float2 _f32x2_mul_b_30 = make_float2(x0_6, x1_6);
        float2 gx_4;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx_4) : "l"(*(const unsigned long long*)&_f32x2_mul_a_29), "l"(*(const unsigned long long*)&_f32x2_mul_b_30));
        float2 p_4;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&p_4) : "l"(*(const unsigned long long*)&gx_4), "l"(*(const unsigned long long*)&rr2_1));
        float2 a_4 = make_float2(acc[4], acc[5]);
        float2 s_4 = add_f32x2_noftz(a_4, p_4);
        float2 bn_4 = sub_f32x2_noftz(a_4, s_4);
        float2 av_4 = add_f32x2_noftz(s_4, bn_4);
        float2 db_4 = fma_f32x2_rn_noftz(gx_4, rr2_1, bn_4);
        float2 c_4 = add_f32x2_noftz(add_f32x2_noftz(make_float2(comp[4], comp[5]), sub_f32x2_noftz(a_4, av_4)), db_4);
        acc[4] = s_4.x;
        acc[5] = s_4.y;
        comp[4] = c_4.x;
        comp[5] = c_4.y;
        float g0_7 = __uint_as_float(gw[7] << 16);
        float g1_7 = __uint_as_float(gw[7] & 4294901760u);
        float x0_7 = __uint_as_float(xw[7] << 16);
        float x1_7 = __uint_as_float(xw[7] & 4294901760u);
        dot_1 = __fmaf_rn(g0_7 * ww[6], x0_7, dot_1);
        dot_1 = __fmaf_rn(g1_7 * ww[7], x1_7, dot_1);
        float2 _f32x2_mul_a_31 = make_float2(g0_7, g1_7);
        float2 _f32x2_mul_b_32 = make_float2(x0_7, x1_7);
        float2 gx_5;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx_5) : "l"(*(const unsigned long long*)&_f32x2_mul_a_31), "l"(*(const unsigned long long*)&_f32x2_mul_b_32));
        float2 p_5;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&p_5) : "l"(*(const unsigned long long*)&gx_5), "l"(*(const unsigned long long*)&rr2_1));
        float2 a_5 = make_float2(acc[6], acc[7]);
        float2 s_5 = add_f32x2_noftz(a_5, p_5);
        float2 bn_5 = sub_f32x2_noftz(a_5, s_5);
        float2 av_5 = add_f32x2_noftz(s_5, bn_5);
        float2 db_5 = fma_f32x2_rn_noftz(gx_5, rr2_1, bn_5);
        float2 c_5 = add_f32x2_noftz(add_f32x2_noftz(make_float2(comp[6], comp[7]), sub_f32x2_noftz(a_5, av_5)), db_5);
        acc[6] = s_5.x;
        acc[7] = s_5.y;
        comp[6] = c_5.x;
        comp[7] = c_5.y;
        dq[1] = dot_1;
    }
    dq[2] = 0.0f;
    if (16 + rg < T) {
        float2 rr2_2 = make_float2(rr[2], rr[2]);
        float dot_2 = 0.0f;
        float g0_8 = __uint_as_float(gw[8] << 16);
        float g1_8 = __uint_as_float(gw[8] & 4294901760u);
        float x0_8 = __uint_as_float(xw[8] << 16);
        float x1_8 = __uint_as_float(xw[8] & 4294901760u);
        dot_2 = __fmaf_rn(g0_8 * ww[0], x0_8, dot_2);
        dot_2 = __fmaf_rn(g1_8 * ww[1], x1_8, dot_2);
        float2 _f32x2_mul_a_33 = make_float2(g0_8, g1_8);
        float2 _f32x2_mul_b_34 = make_float2(x0_8, x1_8);
        float2 gx_6;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx_6) : "l"(*(const unsigned long long*)&_f32x2_mul_a_33), "l"(*(const unsigned long long*)&_f32x2_mul_b_34));
        float2 p_6;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&p_6) : "l"(*(const unsigned long long*)&gx_6), "l"(*(const unsigned long long*)&rr2_2));
        float2 a_6 = make_float2(acc[0], acc[1]);
        float2 s_6 = add_f32x2_noftz(a_6, p_6);
        float2 bn_6 = sub_f32x2_noftz(a_6, s_6);
        float2 av_6 = add_f32x2_noftz(s_6, bn_6);
        float2 db_6 = fma_f32x2_rn_noftz(gx_6, rr2_2, bn_6);
        float2 c_6 = add_f32x2_noftz(add_f32x2_noftz(make_float2(comp[0], comp[1]), sub_f32x2_noftz(a_6, av_6)), db_6);
        acc[0] = s_6.x;
        acc[1] = s_6.y;
        comp[0] = c_6.x;
        comp[1] = c_6.y;
        float g0_9 = __uint_as_float(gw[9] << 16);
        float g1_9 = __uint_as_float(gw[9] & 4294901760u);
        float x0_9 = __uint_as_float(xw[9] << 16);
        float x1_9 = __uint_as_float(xw[9] & 4294901760u);
        dot_2 = __fmaf_rn(g0_9 * ww[2], x0_9, dot_2);
        dot_2 = __fmaf_rn(g1_9 * ww[3], x1_9, dot_2);
        float2 _f32x2_mul_a_35 = make_float2(g0_9, g1_9);
        float2 _f32x2_mul_b_36 = make_float2(x0_9, x1_9);
        float2 gx_6_1;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx_6_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_35), "l"(*(const unsigned long long*)&_f32x2_mul_b_36));
        float2 p_6_1;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&p_6_1) : "l"(*(const unsigned long long*)&gx_6_1), "l"(*(const unsigned long long*)&rr2_2));
        float2 a_6_1 = make_float2(acc[2], acc[3]);
        float2 s_6_1 = add_f32x2_noftz(a_6_1, p_6_1);
        float2 bn_6_1 = sub_f32x2_noftz(a_6_1, s_6_1);
        float2 av_6_1 = add_f32x2_noftz(s_6_1, bn_6_1);
        float2 db_6_1 = fma_f32x2_rn_noftz(gx_6_1, rr2_2, bn_6_1);
        float2 c_6_1 = add_f32x2_noftz(add_f32x2_noftz(make_float2(comp[2], comp[3]), sub_f32x2_noftz(a_6_1, av_6_1)), db_6_1);
        acc[2] = s_6_1.x;
        acc[3] = s_6_1.y;
        comp[2] = c_6_1.x;
        comp[3] = c_6_1.y;
        float g0_10 = __uint_as_float(gw[10] << 16);
        float g1_10 = __uint_as_float(gw[10] & 4294901760u);
        float x0_10 = __uint_as_float(xw[10] << 16);
        float x1_10 = __uint_as_float(xw[10] & 4294901760u);
        dot_2 = __fmaf_rn(g0_10 * ww[4], x0_10, dot_2);
        dot_2 = __fmaf_rn(g1_10 * ww[5], x1_10, dot_2);
        float2 _f32x2_mul_a_37 = make_float2(g0_10, g1_10);
        float2 _f32x2_mul_b_38 = make_float2(x0_10, x1_10);
        float2 gx_7;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx_7) : "l"(*(const unsigned long long*)&_f32x2_mul_a_37), "l"(*(const unsigned long long*)&_f32x2_mul_b_38));
        float2 p_7;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&p_7) : "l"(*(const unsigned long long*)&gx_7), "l"(*(const unsigned long long*)&rr2_2));
        float2 a_7 = make_float2(acc[4], acc[5]);
        float2 s_7 = add_f32x2_noftz(a_7, p_7);
        float2 bn_7 = sub_f32x2_noftz(a_7, s_7);
        float2 av_7 = add_f32x2_noftz(s_7, bn_7);
        float2 db_7 = fma_f32x2_rn_noftz(gx_7, rr2_2, bn_7);
        float2 c_7 = add_f32x2_noftz(add_f32x2_noftz(make_float2(comp[4], comp[5]), sub_f32x2_noftz(a_7, av_7)), db_7);
        acc[4] = s_7.x;
        acc[5] = s_7.y;
        comp[4] = c_7.x;
        comp[5] = c_7.y;
        float g0_11 = __uint_as_float(gw[11] << 16);
        float g1_11 = __uint_as_float(gw[11] & 4294901760u);
        float x0_11 = __uint_as_float(xw[11] << 16);
        float x1_11 = __uint_as_float(xw[11] & 4294901760u);
        dot_2 = __fmaf_rn(g0_11 * ww[6], x0_11, dot_2);
        dot_2 = __fmaf_rn(g1_11 * ww[7], x1_11, dot_2);
        float2 _f32x2_mul_a_39 = make_float2(g0_11, g1_11);
        float2 _f32x2_mul_b_40 = make_float2(x0_11, x1_11);
        float2 gx_8;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx_8) : "l"(*(const unsigned long long*)&_f32x2_mul_a_39), "l"(*(const unsigned long long*)&_f32x2_mul_b_40));
        float2 p_8;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&p_8) : "l"(*(const unsigned long long*)&gx_8), "l"(*(const unsigned long long*)&rr2_2));
        float2 a_8 = make_float2(acc[6], acc[7]);
        float2 s_8 = add_f32x2_noftz(a_8, p_8);
        float2 bn_8 = sub_f32x2_noftz(a_8, s_8);
        float2 av_8 = add_f32x2_noftz(s_8, bn_8);
        float2 db_8 = fma_f32x2_rn_noftz(gx_8, rr2_2, bn_8);
        float2 c_8 = add_f32x2_noftz(add_f32x2_noftz(make_float2(comp[6], comp[7]), sub_f32x2_noftz(a_8, av_8)), db_8);
        acc[6] = s_8.x;
        acc[7] = s_8.y;
        comp[6] = c_8.x;
        comp[7] = c_8.y;
        dq[2] = dot_2;
    }
    dq[3] = 0.0f;
    if (24 + rg < T) {
        float2 rr2_3 = make_float2(rr[3], rr[3]);
        float dot_3 = 0.0f;
        float g0_12 = __uint_as_float(gw[12] << 16);
        float g1_12 = __uint_as_float(gw[12] & 4294901760u);
        float x0_12 = __uint_as_float(xw[12] << 16);
        float x1_12 = __uint_as_float(xw[12] & 4294901760u);
        dot_3 = __fmaf_rn(g0_12 * ww[0], x0_12, dot_3);
        dot_3 = __fmaf_rn(g1_12 * ww[1], x1_12, dot_3);
        float2 _f32x2_mul_a_41 = make_float2(g0_12, g1_12);
        float2 _f32x2_mul_b_42 = make_float2(x0_12, x1_12);
        float2 gx_9;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx_9) : "l"(*(const unsigned long long*)&_f32x2_mul_a_41), "l"(*(const unsigned long long*)&_f32x2_mul_b_42));
        float2 p_9;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&p_9) : "l"(*(const unsigned long long*)&gx_9), "l"(*(const unsigned long long*)&rr2_3));
        float2 a_9 = make_float2(acc[0], acc[1]);
        float2 s_9 = add_f32x2_noftz(a_9, p_9);
        float2 bn_9 = sub_f32x2_noftz(a_9, s_9);
        float2 av_9 = add_f32x2_noftz(s_9, bn_9);
        float2 db_9 = fma_f32x2_rn_noftz(gx_9, rr2_3, bn_9);
        float2 c_9 = add_f32x2_noftz(add_f32x2_noftz(make_float2(comp[0], comp[1]), sub_f32x2_noftz(a_9, av_9)), db_9);
        acc[0] = s_9.x;
        acc[1] = s_9.y;
        comp[0] = c_9.x;
        comp[1] = c_9.y;
        float g0_13 = __uint_as_float(gw[13] << 16);
        float g1_13 = __uint_as_float(gw[13] & 4294901760u);
        float x0_13 = __uint_as_float(xw[13] << 16);
        float x1_13 = __uint_as_float(xw[13] & 4294901760u);
        dot_3 = __fmaf_rn(g0_13 * ww[2], x0_13, dot_3);
        dot_3 = __fmaf_rn(g1_13 * ww[3], x1_13, dot_3);
        float2 _f32x2_mul_a_43 = make_float2(g0_13, g1_13);
        float2 _f32x2_mul_b_44 = make_float2(x0_13, x1_13);
        float2 gx_9_1;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx_9_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_43), "l"(*(const unsigned long long*)&_f32x2_mul_b_44));
        float2 p_9_1;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&p_9_1) : "l"(*(const unsigned long long*)&gx_9_1), "l"(*(const unsigned long long*)&rr2_3));
        float2 a_9_1 = make_float2(acc[2], acc[3]);
        float2 s_9_1 = add_f32x2_noftz(a_9_1, p_9_1);
        float2 bn_9_1 = sub_f32x2_noftz(a_9_1, s_9_1);
        float2 av_9_1 = add_f32x2_noftz(s_9_1, bn_9_1);
        float2 db_9_1 = fma_f32x2_rn_noftz(gx_9_1, rr2_3, bn_9_1);
        float2 c_9_1 = add_f32x2_noftz(add_f32x2_noftz(make_float2(comp[2], comp[3]), sub_f32x2_noftz(a_9_1, av_9_1)), db_9_1);
        acc[2] = s_9_1.x;
        acc[3] = s_9_1.y;
        comp[2] = c_9_1.x;
        comp[3] = c_9_1.y;
        float g0_14 = __uint_as_float(gw[14] << 16);
        float g1_14 = __uint_as_float(gw[14] & 4294901760u);
        float x0_14 = __uint_as_float(xw[14] << 16);
        float x1_14 = __uint_as_float(xw[14] & 4294901760u);
        dot_3 = __fmaf_rn(g0_14 * ww[4], x0_14, dot_3);
        dot_3 = __fmaf_rn(g1_14 * ww[5], x1_14, dot_3);
        float2 _f32x2_mul_a_45 = make_float2(g0_14, g1_14);
        float2 _f32x2_mul_b_46 = make_float2(x0_14, x1_14);
        float2 gx_10;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx_10) : "l"(*(const unsigned long long*)&_f32x2_mul_a_45), "l"(*(const unsigned long long*)&_f32x2_mul_b_46));
        float2 p_10;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&p_10) : "l"(*(const unsigned long long*)&gx_10), "l"(*(const unsigned long long*)&rr2_3));
        float2 a_10 = make_float2(acc[4], acc[5]);
        float2 s_10 = add_f32x2_noftz(a_10, p_10);
        float2 bn_10 = sub_f32x2_noftz(a_10, s_10);
        float2 av_10 = add_f32x2_noftz(s_10, bn_10);
        float2 db_10 = fma_f32x2_rn_noftz(gx_10, rr2_3, bn_10);
        float2 c_10 = add_f32x2_noftz(add_f32x2_noftz(make_float2(comp[4], comp[5]), sub_f32x2_noftz(a_10, av_10)), db_10);
        acc[4] = s_10.x;
        acc[5] = s_10.y;
        comp[4] = c_10.x;
        comp[5] = c_10.y;
        float g0_15 = __uint_as_float(gw[15] << 16);
        float g1_15 = __uint_as_float(gw[15] & 4294901760u);
        float x0_15 = __uint_as_float(xw[15] << 16);
        float x1_15 = __uint_as_float(xw[15] & 4294901760u);
        dot_3 = __fmaf_rn(g0_15 * ww[6], x0_15, dot_3);
        dot_3 = __fmaf_rn(g1_15 * ww[7], x1_15, dot_3);
        float2 _f32x2_mul_a_47 = make_float2(g0_15, g1_15);
        float2 _f32x2_mul_b_48 = make_float2(x0_15, x1_15);
        float2 gx_11;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx_11) : "l"(*(const unsigned long long*)&_f32x2_mul_a_47), "l"(*(const unsigned long long*)&_f32x2_mul_b_48));
        float2 p_11;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&p_11) : "l"(*(const unsigned long long*)&gx_11), "l"(*(const unsigned long long*)&rr2_3));
        float2 a_11 = make_float2(acc[6], acc[7]);
        float2 s_11 = add_f32x2_noftz(a_11, p_11);
        float2 bn_11 = sub_f32x2_noftz(a_11, s_11);
        float2 av_11 = add_f32x2_noftz(s_11, bn_11);
        float2 db_11 = fma_f32x2_rn_noftz(gx_11, rr2_3, bn_11);
        float2 c_11 = add_f32x2_noftz(add_f32x2_noftz(make_float2(comp[6], comp[7]), sub_f32x2_noftz(a_11, av_11)), db_11);
        acc[6] = s_11.x;
        acc[7] = s_11.y;
        comp[6] = c_11.x;
        comp[7] = c_11.y;
        dq[3] = dot_3;
    }
    dq[4] = 0.0f;
    if (32 + rg < T) {
        float2 rr2_4 = make_float2(rr[4], rr[4]);
        float dot_4 = 0.0f;
        float g0_16 = __uint_as_float(gw[16] << 16);
        float g1_16 = __uint_as_float(gw[16] & 4294901760u);
        float x0_16 = __uint_as_float(xw[16] << 16);
        float x1_16 = __uint_as_float(xw[16] & 4294901760u);
        dot_4 = __fmaf_rn(g0_16 * ww[0], x0_16, dot_4);
        dot_4 = __fmaf_rn(g1_16 * ww[1], x1_16, dot_4);
        float2 _f32x2_mul_a_49 = make_float2(g0_16, g1_16);
        float2 _f32x2_mul_b_50 = make_float2(x0_16, x1_16);
        float2 gx_12;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx_12) : "l"(*(const unsigned long long*)&_f32x2_mul_a_49), "l"(*(const unsigned long long*)&_f32x2_mul_b_50));
        float2 p_12;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&p_12) : "l"(*(const unsigned long long*)&gx_12), "l"(*(const unsigned long long*)&rr2_4));
        float2 a_12 = make_float2(acc[0], acc[1]);
        float2 s_12 = add_f32x2_noftz(a_12, p_12);
        float2 bn_12 = sub_f32x2_noftz(a_12, s_12);
        float2 av_12 = add_f32x2_noftz(s_12, bn_12);
        float2 db_12 = fma_f32x2_rn_noftz(gx_12, rr2_4, bn_12);
        float2 c_12 = add_f32x2_noftz(add_f32x2_noftz(make_float2(comp[0], comp[1]), sub_f32x2_noftz(a_12, av_12)), db_12);
        acc[0] = s_12.x;
        acc[1] = s_12.y;
        comp[0] = c_12.x;
        comp[1] = c_12.y;
        float g0_17 = __uint_as_float(gw[17] << 16);
        float g1_17 = __uint_as_float(gw[17] & 4294901760u);
        float x0_17 = __uint_as_float(xw[17] << 16);
        float x1_17 = __uint_as_float(xw[17] & 4294901760u);
        dot_4 = __fmaf_rn(g0_17 * ww[2], x0_17, dot_4);
        dot_4 = __fmaf_rn(g1_17 * ww[3], x1_17, dot_4);
        float2 _f32x2_mul_a_51 = make_float2(g0_17, g1_17);
        float2 _f32x2_mul_b_52 = make_float2(x0_17, x1_17);
        float2 gx_12_1;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx_12_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_51), "l"(*(const unsigned long long*)&_f32x2_mul_b_52));
        float2 p_12_1;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&p_12_1) : "l"(*(const unsigned long long*)&gx_12_1), "l"(*(const unsigned long long*)&rr2_4));
        float2 a_12_1 = make_float2(acc[2], acc[3]);
        float2 s_12_1 = add_f32x2_noftz(a_12_1, p_12_1);
        float2 bn_12_1 = sub_f32x2_noftz(a_12_1, s_12_1);
        float2 av_12_1 = add_f32x2_noftz(s_12_1, bn_12_1);
        float2 db_12_1 = fma_f32x2_rn_noftz(gx_12_1, rr2_4, bn_12_1);
        float2 c_12_1 = add_f32x2_noftz(add_f32x2_noftz(make_float2(comp[2], comp[3]), sub_f32x2_noftz(a_12_1, av_12_1)), db_12_1);
        acc[2] = s_12_1.x;
        acc[3] = s_12_1.y;
        comp[2] = c_12_1.x;
        comp[3] = c_12_1.y;
        float g0_18 = __uint_as_float(gw[18] << 16);
        float g1_18 = __uint_as_float(gw[18] & 4294901760u);
        float x0_18 = __uint_as_float(xw[18] << 16);
        float x1_18 = __uint_as_float(xw[18] & 4294901760u);
        dot_4 = __fmaf_rn(g0_18 * ww[4], x0_18, dot_4);
        dot_4 = __fmaf_rn(g1_18 * ww[5], x1_18, dot_4);
        float2 _f32x2_mul_a_53 = make_float2(g0_18, g1_18);
        float2 _f32x2_mul_b_54 = make_float2(x0_18, x1_18);
        float2 gx_13;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx_13) : "l"(*(const unsigned long long*)&_f32x2_mul_a_53), "l"(*(const unsigned long long*)&_f32x2_mul_b_54));
        float2 p_13;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&p_13) : "l"(*(const unsigned long long*)&gx_13), "l"(*(const unsigned long long*)&rr2_4));
        float2 a_13 = make_float2(acc[4], acc[5]);
        float2 s_13 = add_f32x2_noftz(a_13, p_13);
        float2 bn_13 = sub_f32x2_noftz(a_13, s_13);
        float2 av_13 = add_f32x2_noftz(s_13, bn_13);
        float2 db_13 = fma_f32x2_rn_noftz(gx_13, rr2_4, bn_13);
        float2 c_13 = add_f32x2_noftz(add_f32x2_noftz(make_float2(comp[4], comp[5]), sub_f32x2_noftz(a_13, av_13)), db_13);
        acc[4] = s_13.x;
        acc[5] = s_13.y;
        comp[4] = c_13.x;
        comp[5] = c_13.y;
        float g0_19 = __uint_as_float(gw[19] << 16);
        float g1_19 = __uint_as_float(gw[19] & 4294901760u);
        float x0_19 = __uint_as_float(xw[19] << 16);
        float x1_19 = __uint_as_float(xw[19] & 4294901760u);
        dot_4 = __fmaf_rn(g0_19 * ww[6], x0_19, dot_4);
        dot_4 = __fmaf_rn(g1_19 * ww[7], x1_19, dot_4);
        float2 _f32x2_mul_a_55 = make_float2(g0_19, g1_19);
        float2 _f32x2_mul_b_56 = make_float2(x0_19, x1_19);
        float2 gx_14;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx_14) : "l"(*(const unsigned long long*)&_f32x2_mul_a_55), "l"(*(const unsigned long long*)&_f32x2_mul_b_56));
        float2 p_14;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&p_14) : "l"(*(const unsigned long long*)&gx_14), "l"(*(const unsigned long long*)&rr2_4));
        float2 a_14 = make_float2(acc[6], acc[7]);
        float2 s_14 = add_f32x2_noftz(a_14, p_14);
        float2 bn_14 = sub_f32x2_noftz(a_14, s_14);
        float2 av_14 = add_f32x2_noftz(s_14, bn_14);
        float2 db_14 = fma_f32x2_rn_noftz(gx_14, rr2_4, bn_14);
        float2 c_14 = add_f32x2_noftz(add_f32x2_noftz(make_float2(comp[6], comp[7]), sub_f32x2_noftz(a_14, av_14)), db_14);
        acc[6] = s_14.x;
        acc[7] = s_14.y;
        comp[6] = c_14.x;
        comp[7] = c_14.y;
        dq[4] = dot_4;
    }
    dq[5] = 0.0f;
    if (40 + rg < T) {
        float2 rr2_5 = make_float2(rr[5], rr[5]);
        float dot_5 = 0.0f;
        float g0_20 = __uint_as_float(gw[20] << 16);
        float g1_20 = __uint_as_float(gw[20] & 4294901760u);
        float x0_20 = __uint_as_float(xw[20] << 16);
        float x1_20 = __uint_as_float(xw[20] & 4294901760u);
        dot_5 = __fmaf_rn(g0_20 * ww[0], x0_20, dot_5);
        dot_5 = __fmaf_rn(g1_20 * ww[1], x1_20, dot_5);
        float2 _f32x2_mul_a_57 = make_float2(g0_20, g1_20);
        float2 _f32x2_mul_b_58 = make_float2(x0_20, x1_20);
        float2 gx_15;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx_15) : "l"(*(const unsigned long long*)&_f32x2_mul_a_57), "l"(*(const unsigned long long*)&_f32x2_mul_b_58));
        float2 p_15;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&p_15) : "l"(*(const unsigned long long*)&gx_15), "l"(*(const unsigned long long*)&rr2_5));
        float2 a_15 = make_float2(acc[0], acc[1]);
        float2 s_15 = add_f32x2_noftz(a_15, p_15);
        float2 bn_15 = sub_f32x2_noftz(a_15, s_15);
        float2 av_15 = add_f32x2_noftz(s_15, bn_15);
        float2 db_15 = fma_f32x2_rn_noftz(gx_15, rr2_5, bn_15);
        float2 c_15 = add_f32x2_noftz(add_f32x2_noftz(make_float2(comp[0], comp[1]), sub_f32x2_noftz(a_15, av_15)), db_15);
        acc[0] = s_15.x;
        acc[1] = s_15.y;
        comp[0] = c_15.x;
        comp[1] = c_15.y;
        float g0_21 = __uint_as_float(gw[21] << 16);
        float g1_21 = __uint_as_float(gw[21] & 4294901760u);
        float x0_21 = __uint_as_float(xw[21] << 16);
        float x1_21 = __uint_as_float(xw[21] & 4294901760u);
        dot_5 = __fmaf_rn(g0_21 * ww[2], x0_21, dot_5);
        dot_5 = __fmaf_rn(g1_21 * ww[3], x1_21, dot_5);
        float2 _f32x2_mul_a_59 = make_float2(g0_21, g1_21);
        float2 _f32x2_mul_b_60 = make_float2(x0_21, x1_21);
        float2 gx_15_1;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx_15_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_59), "l"(*(const unsigned long long*)&_f32x2_mul_b_60));
        float2 p_15_1;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&p_15_1) : "l"(*(const unsigned long long*)&gx_15_1), "l"(*(const unsigned long long*)&rr2_5));
        float2 a_15_1 = make_float2(acc[2], acc[3]);
        float2 s_15_1 = add_f32x2_noftz(a_15_1, p_15_1);
        float2 bn_15_1 = sub_f32x2_noftz(a_15_1, s_15_1);
        float2 av_15_1 = add_f32x2_noftz(s_15_1, bn_15_1);
        float2 db_15_1 = fma_f32x2_rn_noftz(gx_15_1, rr2_5, bn_15_1);
        float2 c_15_1 = add_f32x2_noftz(add_f32x2_noftz(make_float2(comp[2], comp[3]), sub_f32x2_noftz(a_15_1, av_15_1)), db_15_1);
        acc[2] = s_15_1.x;
        acc[3] = s_15_1.y;
        comp[2] = c_15_1.x;
        comp[3] = c_15_1.y;
        float g0_22 = __uint_as_float(gw[22] << 16);
        float g1_22 = __uint_as_float(gw[22] & 4294901760u);
        float x0_22 = __uint_as_float(xw[22] << 16);
        float x1_22 = __uint_as_float(xw[22] & 4294901760u);
        dot_5 = __fmaf_rn(g0_22 * ww[4], x0_22, dot_5);
        dot_5 = __fmaf_rn(g1_22 * ww[5], x1_22, dot_5);
        float2 _f32x2_mul_a_61 = make_float2(g0_22, g1_22);
        float2 _f32x2_mul_b_62 = make_float2(x0_22, x1_22);
        float2 gx_16;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx_16) : "l"(*(const unsigned long long*)&_f32x2_mul_a_61), "l"(*(const unsigned long long*)&_f32x2_mul_b_62));
        float2 p_16;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&p_16) : "l"(*(const unsigned long long*)&gx_16), "l"(*(const unsigned long long*)&rr2_5));
        float2 a_16 = make_float2(acc[4], acc[5]);
        float2 s_16 = add_f32x2_noftz(a_16, p_16);
        float2 bn_16 = sub_f32x2_noftz(a_16, s_16);
        float2 av_16 = add_f32x2_noftz(s_16, bn_16);
        float2 db_16 = fma_f32x2_rn_noftz(gx_16, rr2_5, bn_16);
        float2 c_16 = add_f32x2_noftz(add_f32x2_noftz(make_float2(comp[4], comp[5]), sub_f32x2_noftz(a_16, av_16)), db_16);
        acc[4] = s_16.x;
        acc[5] = s_16.y;
        comp[4] = c_16.x;
        comp[5] = c_16.y;
        float g0_23 = __uint_as_float(gw[23] << 16);
        float g1_23 = __uint_as_float(gw[23] & 4294901760u);
        float x0_23 = __uint_as_float(xw[23] << 16);
        float x1_23 = __uint_as_float(xw[23] & 4294901760u);
        dot_5 = __fmaf_rn(g0_23 * ww[6], x0_23, dot_5);
        dot_5 = __fmaf_rn(g1_23 * ww[7], x1_23, dot_5);
        float2 _f32x2_mul_a_63 = make_float2(g0_23, g1_23);
        float2 _f32x2_mul_b_64 = make_float2(x0_23, x1_23);
        float2 gx_17;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx_17) : "l"(*(const unsigned long long*)&_f32x2_mul_a_63), "l"(*(const unsigned long long*)&_f32x2_mul_b_64));
        float2 p_17;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&p_17) : "l"(*(const unsigned long long*)&gx_17), "l"(*(const unsigned long long*)&rr2_5));
        float2 a_17 = make_float2(acc[6], acc[7]);
        float2 s_17 = add_f32x2_noftz(a_17, p_17);
        float2 bn_17 = sub_f32x2_noftz(a_17, s_17);
        float2 av_17 = add_f32x2_noftz(s_17, bn_17);
        float2 db_17 = fma_f32x2_rn_noftz(gx_17, rr2_5, bn_17);
        float2 c_17 = add_f32x2_noftz(add_f32x2_noftz(make_float2(comp[6], comp[7]), sub_f32x2_noftz(a_17, av_17)), db_17);
        acc[6] = s_17.x;
        acc[7] = s_17.y;
        comp[6] = c_17.x;
        comp[7] = c_17.y;
        dq[5] = dot_5;
    }
    dq[6] = 0.0f;
    if (48 + rg < T) {
        float2 rr2_6 = make_float2(rr[6], rr[6]);
        float dot_6 = 0.0f;
        float g0_24 = __uint_as_float(gw[24] << 16);
        float g1_24 = __uint_as_float(gw[24] & 4294901760u);
        float x0_24 = __uint_as_float(xw[24] << 16);
        float x1_24 = __uint_as_float(xw[24] & 4294901760u);
        dot_6 = __fmaf_rn(g0_24 * ww[0], x0_24, dot_6);
        dot_6 = __fmaf_rn(g1_24 * ww[1], x1_24, dot_6);
        float2 _f32x2_mul_a_65 = make_float2(g0_24, g1_24);
        float2 _f32x2_mul_b_66 = make_float2(x0_24, x1_24);
        float2 gx_18;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx_18) : "l"(*(const unsigned long long*)&_f32x2_mul_a_65), "l"(*(const unsigned long long*)&_f32x2_mul_b_66));
        float2 p_18;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&p_18) : "l"(*(const unsigned long long*)&gx_18), "l"(*(const unsigned long long*)&rr2_6));
        float2 a_18 = make_float2(acc[0], acc[1]);
        float2 s_18 = add_f32x2_noftz(a_18, p_18);
        float2 bn_18 = sub_f32x2_noftz(a_18, s_18);
        float2 av_18 = add_f32x2_noftz(s_18, bn_18);
        float2 db_18 = fma_f32x2_rn_noftz(gx_18, rr2_6, bn_18);
        float2 c_18 = add_f32x2_noftz(add_f32x2_noftz(make_float2(comp[0], comp[1]), sub_f32x2_noftz(a_18, av_18)), db_18);
        acc[0] = s_18.x;
        acc[1] = s_18.y;
        comp[0] = c_18.x;
        comp[1] = c_18.y;
        float g0_25 = __uint_as_float(gw[25] << 16);
        float g1_25 = __uint_as_float(gw[25] & 4294901760u);
        float x0_25 = __uint_as_float(xw[25] << 16);
        float x1_25 = __uint_as_float(xw[25] & 4294901760u);
        dot_6 = __fmaf_rn(g0_25 * ww[2], x0_25, dot_6);
        dot_6 = __fmaf_rn(g1_25 * ww[3], x1_25, dot_6);
        float2 _f32x2_mul_a_67 = make_float2(g0_25, g1_25);
        float2 _f32x2_mul_b_68 = make_float2(x0_25, x1_25);
        float2 gx_18_1;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx_18_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_67), "l"(*(const unsigned long long*)&_f32x2_mul_b_68));
        float2 p_18_1;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&p_18_1) : "l"(*(const unsigned long long*)&gx_18_1), "l"(*(const unsigned long long*)&rr2_6));
        float2 a_18_1 = make_float2(acc[2], acc[3]);
        float2 s_18_1 = add_f32x2_noftz(a_18_1, p_18_1);
        float2 bn_18_1 = sub_f32x2_noftz(a_18_1, s_18_1);
        float2 av_18_1 = add_f32x2_noftz(s_18_1, bn_18_1);
        float2 db_18_1 = fma_f32x2_rn_noftz(gx_18_1, rr2_6, bn_18_1);
        float2 c_18_1 = add_f32x2_noftz(add_f32x2_noftz(make_float2(comp[2], comp[3]), sub_f32x2_noftz(a_18_1, av_18_1)), db_18_1);
        acc[2] = s_18_1.x;
        acc[3] = s_18_1.y;
        comp[2] = c_18_1.x;
        comp[3] = c_18_1.y;
        float g0_26 = __uint_as_float(gw[26] << 16);
        float g1_26 = __uint_as_float(gw[26] & 4294901760u);
        float x0_26 = __uint_as_float(xw[26] << 16);
        float x1_26 = __uint_as_float(xw[26] & 4294901760u);
        dot_6 = __fmaf_rn(g0_26 * ww[4], x0_26, dot_6);
        dot_6 = __fmaf_rn(g1_26 * ww[5], x1_26, dot_6);
        float2 _f32x2_mul_a_69 = make_float2(g0_26, g1_26);
        float2 _f32x2_mul_b_70 = make_float2(x0_26, x1_26);
        float2 gx_19;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx_19) : "l"(*(const unsigned long long*)&_f32x2_mul_a_69), "l"(*(const unsigned long long*)&_f32x2_mul_b_70));
        float2 p_19;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&p_19) : "l"(*(const unsigned long long*)&gx_19), "l"(*(const unsigned long long*)&rr2_6));
        float2 a_19 = make_float2(acc[4], acc[5]);
        float2 s_19 = add_f32x2_noftz(a_19, p_19);
        float2 bn_19 = sub_f32x2_noftz(a_19, s_19);
        float2 av_19 = add_f32x2_noftz(s_19, bn_19);
        float2 db_19 = fma_f32x2_rn_noftz(gx_19, rr2_6, bn_19);
        float2 c_19 = add_f32x2_noftz(add_f32x2_noftz(make_float2(comp[4], comp[5]), sub_f32x2_noftz(a_19, av_19)), db_19);
        acc[4] = s_19.x;
        acc[5] = s_19.y;
        comp[4] = c_19.x;
        comp[5] = c_19.y;
        float g0_27 = __uint_as_float(gw[27] << 16);
        float g1_27 = __uint_as_float(gw[27] & 4294901760u);
        float x0_27 = __uint_as_float(xw[27] << 16);
        float x1_27 = __uint_as_float(xw[27] & 4294901760u);
        dot_6 = __fmaf_rn(g0_27 * ww[6], x0_27, dot_6);
        dot_6 = __fmaf_rn(g1_27 * ww[7], x1_27, dot_6);
        float2 _f32x2_mul_a_71 = make_float2(g0_27, g1_27);
        float2 _f32x2_mul_b_72 = make_float2(x0_27, x1_27);
        float2 gx_20;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx_20) : "l"(*(const unsigned long long*)&_f32x2_mul_a_71), "l"(*(const unsigned long long*)&_f32x2_mul_b_72));
        float2 p_20;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&p_20) : "l"(*(const unsigned long long*)&gx_20), "l"(*(const unsigned long long*)&rr2_6));
        float2 a_20 = make_float2(acc[6], acc[7]);
        float2 s_20 = add_f32x2_noftz(a_20, p_20);
        float2 bn_20 = sub_f32x2_noftz(a_20, s_20);
        float2 av_20 = add_f32x2_noftz(s_20, bn_20);
        float2 db_20 = fma_f32x2_rn_noftz(gx_20, rr2_6, bn_20);
        float2 c_20 = add_f32x2_noftz(add_f32x2_noftz(make_float2(comp[6], comp[7]), sub_f32x2_noftz(a_20, av_20)), db_20);
        acc[6] = s_20.x;
        acc[7] = s_20.y;
        comp[6] = c_20.x;
        comp[7] = c_20.y;
        dq[6] = dot_6;
    }
    dq[7] = 0.0f;
    if (56 + rg < T) {
        float2 rr2_7 = make_float2(rr[7], rr[7]);
        float dot_7 = 0.0f;
        float g0_28 = __uint_as_float(gw[28] << 16);
        float g1_28 = __uint_as_float(gw[28] & 4294901760u);
        float x0_28 = __uint_as_float(xw[28] << 16);
        float x1_28 = __uint_as_float(xw[28] & 4294901760u);
        dot_7 = __fmaf_rn(g0_28 * ww[0], x0_28, dot_7);
        dot_7 = __fmaf_rn(g1_28 * ww[1], x1_28, dot_7);
        float2 _f32x2_mul_a_73 = make_float2(g0_28, g1_28);
        float2 _f32x2_mul_b_74 = make_float2(x0_28, x1_28);
        float2 gx_21;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx_21) : "l"(*(const unsigned long long*)&_f32x2_mul_a_73), "l"(*(const unsigned long long*)&_f32x2_mul_b_74));
        float2 p_21;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&p_21) : "l"(*(const unsigned long long*)&gx_21), "l"(*(const unsigned long long*)&rr2_7));
        float2 a_21 = make_float2(acc[0], acc[1]);
        float2 s_21 = add_f32x2_noftz(a_21, p_21);
        float2 bn_21 = sub_f32x2_noftz(a_21, s_21);
        float2 av_21 = add_f32x2_noftz(s_21, bn_21);
        float2 db_21 = fma_f32x2_rn_noftz(gx_21, rr2_7, bn_21);
        float2 c_21 = add_f32x2_noftz(add_f32x2_noftz(make_float2(comp[0], comp[1]), sub_f32x2_noftz(a_21, av_21)), db_21);
        acc[0] = s_21.x;
        acc[1] = s_21.y;
        comp[0] = c_21.x;
        comp[1] = c_21.y;
        float g0_29 = __uint_as_float(gw[29] << 16);
        float g1_29 = __uint_as_float(gw[29] & 4294901760u);
        float x0_29 = __uint_as_float(xw[29] << 16);
        float x1_29 = __uint_as_float(xw[29] & 4294901760u);
        dot_7 = __fmaf_rn(g0_29 * ww[2], x0_29, dot_7);
        dot_7 = __fmaf_rn(g1_29 * ww[3], x1_29, dot_7);
        float2 _f32x2_mul_a_75 = make_float2(g0_29, g1_29);
        float2 _f32x2_mul_b_76 = make_float2(x0_29, x1_29);
        float2 gx_21_1;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx_21_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_75), "l"(*(const unsigned long long*)&_f32x2_mul_b_76));
        float2 p_21_1;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&p_21_1) : "l"(*(const unsigned long long*)&gx_21_1), "l"(*(const unsigned long long*)&rr2_7));
        float2 a_21_1 = make_float2(acc[2], acc[3]);
        float2 s_21_1 = add_f32x2_noftz(a_21_1, p_21_1);
        float2 bn_21_1 = sub_f32x2_noftz(a_21_1, s_21_1);
        float2 av_21_1 = add_f32x2_noftz(s_21_1, bn_21_1);
        float2 db_21_1 = fma_f32x2_rn_noftz(gx_21_1, rr2_7, bn_21_1);
        float2 c_21_1 = add_f32x2_noftz(add_f32x2_noftz(make_float2(comp[2], comp[3]), sub_f32x2_noftz(a_21_1, av_21_1)), db_21_1);
        acc[2] = s_21_1.x;
        acc[3] = s_21_1.y;
        comp[2] = c_21_1.x;
        comp[3] = c_21_1.y;
        float g0_30 = __uint_as_float(gw[30] << 16);
        float g1_30 = __uint_as_float(gw[30] & 4294901760u);
        float x0_30 = __uint_as_float(xw[30] << 16);
        float x1_30 = __uint_as_float(xw[30] & 4294901760u);
        dot_7 = __fmaf_rn(g0_30 * ww[4], x0_30, dot_7);
        dot_7 = __fmaf_rn(g1_30 * ww[5], x1_30, dot_7);
        float2 _f32x2_mul_a_77 = make_float2(g0_30, g1_30);
        float2 _f32x2_mul_b_78 = make_float2(x0_30, x1_30);
        float2 gx_22;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx_22) : "l"(*(const unsigned long long*)&_f32x2_mul_a_77), "l"(*(const unsigned long long*)&_f32x2_mul_b_78));
        float2 p_22;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&p_22) : "l"(*(const unsigned long long*)&gx_22), "l"(*(const unsigned long long*)&rr2_7));
        float2 a_22 = make_float2(acc[4], acc[5]);
        float2 s_22 = add_f32x2_noftz(a_22, p_22);
        float2 bn_22 = sub_f32x2_noftz(a_22, s_22);
        float2 av_22 = add_f32x2_noftz(s_22, bn_22);
        float2 db_22 = fma_f32x2_rn_noftz(gx_22, rr2_7, bn_22);
        float2 c_22 = add_f32x2_noftz(add_f32x2_noftz(make_float2(comp[4], comp[5]), sub_f32x2_noftz(a_22, av_22)), db_22);
        acc[4] = s_22.x;
        acc[5] = s_22.y;
        comp[4] = c_22.x;
        comp[5] = c_22.y;
        float g0_31 = __uint_as_float(gw[31] << 16);
        float g1_31 = __uint_as_float(gw[31] & 4294901760u);
        float x0_31 = __uint_as_float(xw[31] << 16);
        float x1_31 = __uint_as_float(xw[31] & 4294901760u);
        dot_7 = __fmaf_rn(g0_31 * ww[6], x0_31, dot_7);
        dot_7 = __fmaf_rn(g1_31 * ww[7], x1_31, dot_7);
        float2 _f32x2_mul_a_79 = make_float2(g0_31, g1_31);
        float2 _f32x2_mul_b_80 = make_float2(x0_31, x1_31);
        float2 gx_23;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx_23) : "l"(*(const unsigned long long*)&_f32x2_mul_a_79), "l"(*(const unsigned long long*)&_f32x2_mul_b_80));
        float2 p_23;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&p_23) : "l"(*(const unsigned long long*)&gx_23), "l"(*(const unsigned long long*)&rr2_7));
        float2 a_23 = make_float2(acc[6], acc[7]);
        float2 s_23 = add_f32x2_noftz(a_23, p_23);
        float2 bn_23 = sub_f32x2_noftz(a_23, s_23);
        float2 av_23 = add_f32x2_noftz(s_23, bn_23);
        float2 db_23 = fma_f32x2_rn_noftz(gx_23, rr2_7, bn_23);
        float2 c_23 = add_f32x2_noftz(add_f32x2_noftz(make_float2(comp[6], comp[7]), sub_f32x2_noftz(a_23, av_23)), db_23);
        acc[6] = s_23.x;
        acc[7] = s_23.y;
        comp[6] = c_23.x;
        comp[7] = c_23.y;
        dq[7] = dot_7;
    }
    float _warp_reduce_81 = dq[0];
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        _warp_reduce_81 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_81, offset);
    dq[0] = _warp_reduce_81;
    float _warp_reduce_82 = dq[1];
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        _warp_reduce_82 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_82, offset);
    dq[1] = _warp_reduce_82;
    float _warp_reduce_83 = dq[2];
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        _warp_reduce_83 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_83, offset);
    dq[2] = _warp_reduce_83;
    float _warp_reduce_84 = dq[3];
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        _warp_reduce_84 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_84, offset);
    dq[3] = _warp_reduce_84;
    float _warp_reduce_85 = dq[4];
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        _warp_reduce_85 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_85, offset);
    dq[4] = _warp_reduce_85;
    float _warp_reduce_86 = dq[5];
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        _warp_reduce_86 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_86, offset);
    dq[5] = _warp_reduce_86;
    float _warp_reduce_87 = dq[6];
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        _warp_reduce_87 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_87, offset);
    dq[6] = _warp_reduce_87;
    float _warp_reduce_88 = dq[7];
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        _warp_reduce_88 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_88, offset);
    dq[7] = _warp_reduce_88;
    pairs[rg * 2 * 8 * 32 + ct] = acc[0];
    pairs[(rg * 2 * 8 + 1) * 32 + ct] = comp[0];
    pairs[(rg * 2 * 8 + 2) * 32 + ct] = acc[1];
    pairs[(rg * 2 * 8 + 2 + 1) * 32 + ct] = comp[1];
    pairs[(rg * 2 * 8 + 4) * 32 + ct] = acc[2];
    pairs[(rg * 2 * 8 + 4 + 1) * 32 + ct] = comp[2];
    pairs[(rg * 2 * 8 + 6) * 32 + ct] = acc[3];
    pairs[(rg * 2 * 8 + 6 + 1) * 32 + ct] = comp[3];
    pairs[(rg * 2 * 8 + 8) * 32 + ct] = acc[4];
    pairs[(rg * 2 * 8 + 8 + 1) * 32 + ct] = comp[4];
    pairs[(rg * 2 * 8 + 10) * 32 + ct] = acc[5];
    pairs[(rg * 2 * 8 + 10 + 1) * 32 + ct] = comp[5];
    pairs[(rg * 2 * 8 + 12) * 32 + ct] = acc[6];
    pairs[(rg * 2 * 8 + 12 + 1) * 32 + ct] = comp[6];
    pairs[(rg * 2 * 8 + 14) * 32 + ct] = acc[7];
    pairs[(rg * 2 * 8 + 14 + 1) * 32 + ct] = comp[7];
    __syncthreads();
    float dwo[1];
    float sc[2];
    int jc_0 = rg;
    sc[0] = pairs[2 * jc_0 * 32 + ct];
    sc[1] = pairs[(2 * jc_0 + 1) * 32 + ct];
    float s_24 = sc[0] + pairs[(16 + 2 * jc_0) * 32 + ct];
    float bp = s_24 - sc[0];
    float ap = s_24 - bp;
    float e = sc[0] - ap + (pairs[(16 + 2 * jc_0) * 32 + ct] - bp);
    sc[0] = s_24;
    sc[1] = sc[1] + pairs[(16 + 2 * jc_0 + 1) * 32 + ct] + e;
    float s_24_1 = sc[0] + pairs[(32 + 2 * jc_0) * 32 + ct];
    float bp_0 = s_24_1 - sc[0];
    float ap_0 = s_24_1 - bp_0;
    float e_0 = sc[0] - ap_0 + (pairs[(32 + 2 * jc_0) * 32 + ct] - bp_0);
    sc[0] = s_24_1;
    sc[1] = sc[1] + pairs[(32 + 2 * jc_0 + 1) * 32 + ct] + e_0;
    float s_25 = sc[0] + pairs[(48 + 2 * jc_0) * 32 + ct];
    float bp_1 = s_25 - sc[0];
    float ap_1 = s_25 - bp_1;
    float e_1 = sc[0] - ap_1 + (pairs[(48 + 2 * jc_0) * 32 + ct] - bp_1);
    sc[0] = s_25;
    sc[1] = sc[1] + pairs[(48 + 2 * jc_0 + 1) * 32 + ct] + e_1;
    float s_26 = sc[0] + pairs[(64 + 2 * jc_0) * 32 + ct];
    float bp_2 = s_26 - sc[0];
    float ap_2 = s_26 - bp_2;
    float e_2 = sc[0] - ap_2 + (pairs[(64 + 2 * jc_0) * 32 + ct] - bp_2);
    sc[0] = s_26;
    sc[1] = sc[1] + pairs[(64 + 2 * jc_0 + 1) * 32 + ct] + e_2;
    float s_27 = sc[0] + pairs[(80 + 2 * jc_0) * 32 + ct];
    float bp_3 = s_27 - sc[0];
    float ap_3 = s_27 - bp_3;
    float e_3 = sc[0] - ap_3 + (pairs[(80 + 2 * jc_0) * 32 + ct] - bp_3);
    sc[0] = s_27;
    sc[1] = sc[1] + pairs[(80 + 2 * jc_0 + 1) * 32 + ct] + e_3;
    float s_28 = sc[0] + pairs[(96 + 2 * jc_0) * 32 + ct];
    float bp_4 = s_28 - sc[0];
    float ap_4 = s_28 - bp_4;
    float e_4 = sc[0] - ap_4 + (pairs[(96 + 2 * jc_0) * 32 + ct] - bp_4);
    sc[0] = s_28;
    sc[1] = sc[1] + pairs[(96 + 2 * jc_0 + 1) * 32 + ct] + e_4;
    float s_29 = sc[0] + pairs[(112 + 2 * jc_0) * 32 + ct];
    float bp_5 = s_29 - sc[0];
    float ap_5 = s_29 - bp_5;
    float e_5 = sc[0] - ap_5 + (pairs[(112 + 2 * jc_0) * 32 + ct] - bp_5);
    sc[0] = s_29;
    sc[1] = sc[1] + pairs[(112 + 2 * jc_0 + 1) * 32 + ct] + e_5;
    dwo[0] = sc[0] + sc[1];
    *(reinterpret_cast<float*>(dw) + (col + rg)) = dwo[0];
    int row_0 = rg;
    if (row_0 < T) {
        float tot_0 = dq[0];
        float rq = rr[0];
        float cq = tot_0 * 0.00390625f;
        float kx = rq * rq * rq * cq;
        float nkx = -kx;
        float out[8];
        float g0_32 = __uint_as_float(gw[0] << 16);
        float g1_32 = __uint_as_float(gw[0] & 4294901760u);
        float x0_32 = __uint_as_float(xw[0] << 16);
        float x1_32 = __uint_as_float(xw[0] & 4294901760u);
        float y0_0 = __fmaf_rn(nkx, x0_32, rq * (g0_32 * ww[0]));
        float y1_0 = __fmaf_rn(nkx, x1_32, rq * (g1_32 * ww[1]));
        out[0] = y0_0;
        out[1] = y1_0;
        float g0_33 = __uint_as_float(gw[1] << 16);
        float g1_33 = __uint_as_float(gw[1] & 4294901760u);
        float x0_33 = __uint_as_float(xw[1] << 16);
        float x1_33 = __uint_as_float(xw[1] & 4294901760u);
        float y0_1 = __fmaf_rn(nkx, x0_33, rq * (g0_33 * ww[2]));
        float y1_1 = __fmaf_rn(nkx, x1_33, rq * (g1_33 * ww[3]));
        out[2] = y0_1;
        out[3] = y1_1;
        float g0_34 = __uint_as_float(gw[2] << 16);
        float g1_34 = __uint_as_float(gw[2] & 4294901760u);
        float x0_34 = __uint_as_float(xw[2] << 16);
        float x1_34 = __uint_as_float(xw[2] & 4294901760u);
        float y0_2 = __fmaf_rn(nkx, x0_34, rq * (g0_34 * ww[4]));
        float y1_2 = __fmaf_rn(nkx, x1_34, rq * (g1_34 * ww[5]));
        out[4] = y0_2;
        out[5] = y1_2;
        float g0_35 = __uint_as_float(gw[3] << 16);
        float g1_35 = __uint_as_float(gw[3] & 4294901760u);
        float x0_35 = __uint_as_float(xw[3] << 16);
        float x1_35 = __uint_as_float(xw[3] & 4294901760u);
        float y0_3 = __fmaf_rn(nkx, x0_35, rq * (g0_35 * ww[6]));
        float y1_3 = __fmaf_rn(nkx, x1_35, rq * (g1_35 * ww[7]));
        out[6] = y0_3;
        out[7] = y1_3;
        unsigned int packed[4];
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out[_lp*2 + 0], out[_lp*2+1 + 0]));
            packed[_lp] = *(uint32_t*)&_bf2;
        }
        reinterpret_cast<int4*>(dx + ((unsigned long long)row_0 * 256ull + (unsigned long long)col))[0] = reinterpret_cast<int4*>(packed)[0];
    }
    int row_1 = 8 + rg;
    if (row_1 < T) {
        float tot_1 = dq[1];
        float rq_1 = rr[1];
        float cq_1 = tot_1 * 0.00390625f;
        float kx_1 = rq_1 * rq_1 * rq_1 * cq_1;
        float nkx_1 = -kx_1;
        float out_1[8];
        float g0_36 = __uint_as_float(gw[4] << 16);
        float g1_36 = __uint_as_float(gw[4] & 4294901760u);
        float x0_36 = __uint_as_float(xw[4] << 16);
        float x1_36 = __uint_as_float(xw[4] & 4294901760u);
        float y0_4 = __fmaf_rn(nkx_1, x0_36, rq_1 * (g0_36 * ww[0]));
        float y1_4 = __fmaf_rn(nkx_1, x1_36, rq_1 * (g1_36 * ww[1]));
        out_1[0] = y0_4;
        out_1[1] = y1_4;
        float g0_37 = __uint_as_float(gw[5] << 16);
        float g1_37 = __uint_as_float(gw[5] & 4294901760u);
        float x0_37 = __uint_as_float(xw[5] << 16);
        float x1_37 = __uint_as_float(xw[5] & 4294901760u);
        float y0_5 = __fmaf_rn(nkx_1, x0_37, rq_1 * (g0_37 * ww[2]));
        float y1_5 = __fmaf_rn(nkx_1, x1_37, rq_1 * (g1_37 * ww[3]));
        out_1[2] = y0_5;
        out_1[3] = y1_5;
        float g0_38 = __uint_as_float(gw[6] << 16);
        float g1_38 = __uint_as_float(gw[6] & 4294901760u);
        float x0_38 = __uint_as_float(xw[6] << 16);
        float x1_38 = __uint_as_float(xw[6] & 4294901760u);
        float y0_6 = __fmaf_rn(nkx_1, x0_38, rq_1 * (g0_38 * ww[4]));
        float y1_6 = __fmaf_rn(nkx_1, x1_38, rq_1 * (g1_38 * ww[5]));
        out_1[4] = y0_6;
        out_1[5] = y1_6;
        float g0_39 = __uint_as_float(gw[7] << 16);
        float g1_39 = __uint_as_float(gw[7] & 4294901760u);
        float x0_39 = __uint_as_float(xw[7] << 16);
        float x1_39 = __uint_as_float(xw[7] & 4294901760u);
        float y0_7 = __fmaf_rn(nkx_1, x0_39, rq_1 * (g0_39 * ww[6]));
        float y1_7 = __fmaf_rn(nkx_1, x1_39, rq_1 * (g1_39 * ww[7]));
        out_1[6] = y0_7;
        out_1[7] = y1_7;
        unsigned int packed_1[4];
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_1[_lp*2 + 0], out_1[_lp*2+1 + 0]));
            packed_1[_lp] = *(uint32_t*)&_bf2;
        }
        reinterpret_cast<int4*>(dx + ((unsigned long long)row_1 * 256ull + (unsigned long long)col))[0] = reinterpret_cast<int4*>(packed_1)[0];
    }
    int row_2 = 16 + rg;
    if (row_2 < T) {
        float tot_2 = dq[2];
        float rq_2 = rr[2];
        float cq_2 = tot_2 * 0.00390625f;
        float kx_2 = rq_2 * rq_2 * rq_2 * cq_2;
        float nkx_2 = -kx_2;
        float out_2[8];
        float g0_40 = __uint_as_float(gw[8] << 16);
        float g1_40 = __uint_as_float(gw[8] & 4294901760u);
        float x0_40 = __uint_as_float(xw[8] << 16);
        float x1_40 = __uint_as_float(xw[8] & 4294901760u);
        float y0_8 = __fmaf_rn(nkx_2, x0_40, rq_2 * (g0_40 * ww[0]));
        float y1_8 = __fmaf_rn(nkx_2, x1_40, rq_2 * (g1_40 * ww[1]));
        out_2[0] = y0_8;
        out_2[1] = y1_8;
        float g0_41 = __uint_as_float(gw[9] << 16);
        float g1_41 = __uint_as_float(gw[9] & 4294901760u);
        float x0_41 = __uint_as_float(xw[9] << 16);
        float x1_41 = __uint_as_float(xw[9] & 4294901760u);
        float y0_9 = __fmaf_rn(nkx_2, x0_41, rq_2 * (g0_41 * ww[2]));
        float y1_9 = __fmaf_rn(nkx_2, x1_41, rq_2 * (g1_41 * ww[3]));
        out_2[2] = y0_9;
        out_2[3] = y1_9;
        float g0_42 = __uint_as_float(gw[10] << 16);
        float g1_42 = __uint_as_float(gw[10] & 4294901760u);
        float x0_42 = __uint_as_float(xw[10] << 16);
        float x1_42 = __uint_as_float(xw[10] & 4294901760u);
        float y0_10 = __fmaf_rn(nkx_2, x0_42, rq_2 * (g0_42 * ww[4]));
        float y1_10 = __fmaf_rn(nkx_2, x1_42, rq_2 * (g1_42 * ww[5]));
        out_2[4] = y0_10;
        out_2[5] = y1_10;
        float g0_43 = __uint_as_float(gw[11] << 16);
        float g1_43 = __uint_as_float(gw[11] & 4294901760u);
        float x0_43 = __uint_as_float(xw[11] << 16);
        float x1_43 = __uint_as_float(xw[11] & 4294901760u);
        float y0_11 = __fmaf_rn(nkx_2, x0_43, rq_2 * (g0_43 * ww[6]));
        float y1_11 = __fmaf_rn(nkx_2, x1_43, rq_2 * (g1_43 * ww[7]));
        out_2[6] = y0_11;
        out_2[7] = y1_11;
        unsigned int packed_2[4];
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_2[_lp*2 + 0], out_2[_lp*2+1 + 0]));
            packed_2[_lp] = *(uint32_t*)&_bf2;
        }
        reinterpret_cast<int4*>(dx + ((unsigned long long)row_2 * 256ull + (unsigned long long)col))[0] = reinterpret_cast<int4*>(packed_2)[0];
    }
    int row_3 = 24 + rg;
    if (row_3 < T) {
        float tot_3 = dq[3];
        float rq_3 = rr[3];
        float cq_3 = tot_3 * 0.00390625f;
        float kx_3 = rq_3 * rq_3 * rq_3 * cq_3;
        float nkx_3 = -kx_3;
        float out_3[8];
        float g0_44 = __uint_as_float(gw[12] << 16);
        float g1_44 = __uint_as_float(gw[12] & 4294901760u);
        float x0_44 = __uint_as_float(xw[12] << 16);
        float x1_44 = __uint_as_float(xw[12] & 4294901760u);
        float y0_12 = __fmaf_rn(nkx_3, x0_44, rq_3 * (g0_44 * ww[0]));
        float y1_12 = __fmaf_rn(nkx_3, x1_44, rq_3 * (g1_44 * ww[1]));
        out_3[0] = y0_12;
        out_3[1] = y1_12;
        float g0_45 = __uint_as_float(gw[13] << 16);
        float g1_45 = __uint_as_float(gw[13] & 4294901760u);
        float x0_45 = __uint_as_float(xw[13] << 16);
        float x1_45 = __uint_as_float(xw[13] & 4294901760u);
        float y0_13 = __fmaf_rn(nkx_3, x0_45, rq_3 * (g0_45 * ww[2]));
        float y1_13 = __fmaf_rn(nkx_3, x1_45, rq_3 * (g1_45 * ww[3]));
        out_3[2] = y0_13;
        out_3[3] = y1_13;
        float g0_46 = __uint_as_float(gw[14] << 16);
        float g1_46 = __uint_as_float(gw[14] & 4294901760u);
        float x0_46 = __uint_as_float(xw[14] << 16);
        float x1_46 = __uint_as_float(xw[14] & 4294901760u);
        float y0_14 = __fmaf_rn(nkx_3, x0_46, rq_3 * (g0_46 * ww[4]));
        float y1_14 = __fmaf_rn(nkx_3, x1_46, rq_3 * (g1_46 * ww[5]));
        out_3[4] = y0_14;
        out_3[5] = y1_14;
        float g0_47 = __uint_as_float(gw[15] << 16);
        float g1_47 = __uint_as_float(gw[15] & 4294901760u);
        float x0_47 = __uint_as_float(xw[15] << 16);
        float x1_47 = __uint_as_float(xw[15] & 4294901760u);
        float y0_15 = __fmaf_rn(nkx_3, x0_47, rq_3 * (g0_47 * ww[6]));
        float y1_15 = __fmaf_rn(nkx_3, x1_47, rq_3 * (g1_47 * ww[7]));
        out_3[6] = y0_15;
        out_3[7] = y1_15;
        unsigned int packed_3[4];
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_3[_lp*2 + 0], out_3[_lp*2+1 + 0]));
            packed_3[_lp] = *(uint32_t*)&_bf2;
        }
        reinterpret_cast<int4*>(dx + ((unsigned long long)row_3 * 256ull + (unsigned long long)col))[0] = reinterpret_cast<int4*>(packed_3)[0];
    }
    int row_4 = 32 + rg;
    if (row_4 < T) {
        float tot_4 = dq[4];
        float rq_4 = rr[4];
        float cq_4 = tot_4 * 0.00390625f;
        float kx_4 = rq_4 * rq_4 * rq_4 * cq_4;
        float nkx_4 = -kx_4;
        float out_4[8];
        float g0_48 = __uint_as_float(gw[16] << 16);
        float g1_48 = __uint_as_float(gw[16] & 4294901760u);
        float x0_48 = __uint_as_float(xw[16] << 16);
        float x1_48 = __uint_as_float(xw[16] & 4294901760u);
        float y0_16 = __fmaf_rn(nkx_4, x0_48, rq_4 * (g0_48 * ww[0]));
        float y1_16 = __fmaf_rn(nkx_4, x1_48, rq_4 * (g1_48 * ww[1]));
        out_4[0] = y0_16;
        out_4[1] = y1_16;
        float g0_49 = __uint_as_float(gw[17] << 16);
        float g1_49 = __uint_as_float(gw[17] & 4294901760u);
        float x0_49 = __uint_as_float(xw[17] << 16);
        float x1_49 = __uint_as_float(xw[17] & 4294901760u);
        float y0_17 = __fmaf_rn(nkx_4, x0_49, rq_4 * (g0_49 * ww[2]));
        float y1_17 = __fmaf_rn(nkx_4, x1_49, rq_4 * (g1_49 * ww[3]));
        out_4[2] = y0_17;
        out_4[3] = y1_17;
        float g0_50 = __uint_as_float(gw[18] << 16);
        float g1_50 = __uint_as_float(gw[18] & 4294901760u);
        float x0_50 = __uint_as_float(xw[18] << 16);
        float x1_50 = __uint_as_float(xw[18] & 4294901760u);
        float y0_18 = __fmaf_rn(nkx_4, x0_50, rq_4 * (g0_50 * ww[4]));
        float y1_18 = __fmaf_rn(nkx_4, x1_50, rq_4 * (g1_50 * ww[5]));
        out_4[4] = y0_18;
        out_4[5] = y1_18;
        float g0_51 = __uint_as_float(gw[19] << 16);
        float g1_51 = __uint_as_float(gw[19] & 4294901760u);
        float x0_51 = __uint_as_float(xw[19] << 16);
        float x1_51 = __uint_as_float(xw[19] & 4294901760u);
        float y0_19 = __fmaf_rn(nkx_4, x0_51, rq_4 * (g0_51 * ww[6]));
        float y1_19 = __fmaf_rn(nkx_4, x1_51, rq_4 * (g1_51 * ww[7]));
        out_4[6] = y0_19;
        out_4[7] = y1_19;
        unsigned int packed_4[4];
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_4[_lp*2 + 0], out_4[_lp*2+1 + 0]));
            packed_4[_lp] = *(uint32_t*)&_bf2;
        }
        reinterpret_cast<int4*>(dx + ((unsigned long long)row_4 * 256ull + (unsigned long long)col))[0] = reinterpret_cast<int4*>(packed_4)[0];
    }
    int row_5 = 40 + rg;
    if (row_5 < T) {
        float tot_5 = dq[5];
        float rq_5 = rr[5];
        float cq_5 = tot_5 * 0.00390625f;
        float kx_5 = rq_5 * rq_5 * rq_5 * cq_5;
        float nkx_5 = -kx_5;
        float out_5[8];
        float g0_52 = __uint_as_float(gw[20] << 16);
        float g1_52 = __uint_as_float(gw[20] & 4294901760u);
        float x0_52 = __uint_as_float(xw[20] << 16);
        float x1_52 = __uint_as_float(xw[20] & 4294901760u);
        float y0_20 = __fmaf_rn(nkx_5, x0_52, rq_5 * (g0_52 * ww[0]));
        float y1_20 = __fmaf_rn(nkx_5, x1_52, rq_5 * (g1_52 * ww[1]));
        out_5[0] = y0_20;
        out_5[1] = y1_20;
        float g0_53 = __uint_as_float(gw[21] << 16);
        float g1_53 = __uint_as_float(gw[21] & 4294901760u);
        float x0_53 = __uint_as_float(xw[21] << 16);
        float x1_53 = __uint_as_float(xw[21] & 4294901760u);
        float y0_21 = __fmaf_rn(nkx_5, x0_53, rq_5 * (g0_53 * ww[2]));
        float y1_21 = __fmaf_rn(nkx_5, x1_53, rq_5 * (g1_53 * ww[3]));
        out_5[2] = y0_21;
        out_5[3] = y1_21;
        float g0_54 = __uint_as_float(gw[22] << 16);
        float g1_54 = __uint_as_float(gw[22] & 4294901760u);
        float x0_54 = __uint_as_float(xw[22] << 16);
        float x1_54 = __uint_as_float(xw[22] & 4294901760u);
        float y0_22 = __fmaf_rn(nkx_5, x0_54, rq_5 * (g0_54 * ww[4]));
        float y1_22 = __fmaf_rn(nkx_5, x1_54, rq_5 * (g1_54 * ww[5]));
        out_5[4] = y0_22;
        out_5[5] = y1_22;
        float g0_55 = __uint_as_float(gw[23] << 16);
        float g1_55 = __uint_as_float(gw[23] & 4294901760u);
        float x0_55 = __uint_as_float(xw[23] << 16);
        float x1_55 = __uint_as_float(xw[23] & 4294901760u);
        float y0_23 = __fmaf_rn(nkx_5, x0_55, rq_5 * (g0_55 * ww[6]));
        float y1_23 = __fmaf_rn(nkx_5, x1_55, rq_5 * (g1_55 * ww[7]));
        out_5[6] = y0_23;
        out_5[7] = y1_23;
        unsigned int packed_5[4];
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_5[_lp*2 + 0], out_5[_lp*2+1 + 0]));
            packed_5[_lp] = *(uint32_t*)&_bf2;
        }
        reinterpret_cast<int4*>(dx + ((unsigned long long)row_5 * 256ull + (unsigned long long)col))[0] = reinterpret_cast<int4*>(packed_5)[0];
    }
    int row_6 = 48 + rg;
    if (row_6 < T) {
        float tot_6 = dq[6];
        float rq_6 = rr[6];
        float cq_6 = tot_6 * 0.00390625f;
        float kx_6 = rq_6 * rq_6 * rq_6 * cq_6;
        float nkx_6 = -kx_6;
        float out_6[8];
        float g0_56 = __uint_as_float(gw[24] << 16);
        float g1_56 = __uint_as_float(gw[24] & 4294901760u);
        float x0_56 = __uint_as_float(xw[24] << 16);
        float x1_56 = __uint_as_float(xw[24] & 4294901760u);
        float y0_24 = __fmaf_rn(nkx_6, x0_56, rq_6 * (g0_56 * ww[0]));
        float y1_24 = __fmaf_rn(nkx_6, x1_56, rq_6 * (g1_56 * ww[1]));
        out_6[0] = y0_24;
        out_6[1] = y1_24;
        float g0_57 = __uint_as_float(gw[25] << 16);
        float g1_57 = __uint_as_float(gw[25] & 4294901760u);
        float x0_57 = __uint_as_float(xw[25] << 16);
        float x1_57 = __uint_as_float(xw[25] & 4294901760u);
        float y0_25 = __fmaf_rn(nkx_6, x0_57, rq_6 * (g0_57 * ww[2]));
        float y1_25 = __fmaf_rn(nkx_6, x1_57, rq_6 * (g1_57 * ww[3]));
        out_6[2] = y0_25;
        out_6[3] = y1_25;
        float g0_58 = __uint_as_float(gw[26] << 16);
        float g1_58 = __uint_as_float(gw[26] & 4294901760u);
        float x0_58 = __uint_as_float(xw[26] << 16);
        float x1_58 = __uint_as_float(xw[26] & 4294901760u);
        float y0_26 = __fmaf_rn(nkx_6, x0_58, rq_6 * (g0_58 * ww[4]));
        float y1_26 = __fmaf_rn(nkx_6, x1_58, rq_6 * (g1_58 * ww[5]));
        out_6[4] = y0_26;
        out_6[5] = y1_26;
        float g0_59 = __uint_as_float(gw[27] << 16);
        float g1_59 = __uint_as_float(gw[27] & 4294901760u);
        float x0_59 = __uint_as_float(xw[27] << 16);
        float x1_59 = __uint_as_float(xw[27] & 4294901760u);
        float y0_27 = __fmaf_rn(nkx_6, x0_59, rq_6 * (g0_59 * ww[6]));
        float y1_27 = __fmaf_rn(nkx_6, x1_59, rq_6 * (g1_59 * ww[7]));
        out_6[6] = y0_27;
        out_6[7] = y1_27;
        unsigned int packed_6[4];
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_6[_lp*2 + 0], out_6[_lp*2+1 + 0]));
            packed_6[_lp] = *(uint32_t*)&_bf2;
        }
        reinterpret_cast<int4*>(dx + ((unsigned long long)row_6 * 256ull + (unsigned long long)col))[0] = reinterpret_cast<int4*>(packed_6)[0];
    }
    int row_7 = 56 + rg;
    if (row_7 < T) {
        float tot_7 = dq[7];
        float rq_7 = rr[7];
        float cq_7 = tot_7 * 0.00390625f;
        float kx_7 = rq_7 * rq_7 * rq_7 * cq_7;
        float nkx_7 = -kx_7;
        float out_7[8];
        float g0_60 = __uint_as_float(gw[28] << 16);
        float g1_60 = __uint_as_float(gw[28] & 4294901760u);
        float x0_60 = __uint_as_float(xw[28] << 16);
        float x1_60 = __uint_as_float(xw[28] & 4294901760u);
        float y0_28 = __fmaf_rn(nkx_7, x0_60, rq_7 * (g0_60 * ww[0]));
        float y1_28 = __fmaf_rn(nkx_7, x1_60, rq_7 * (g1_60 * ww[1]));
        out_7[0] = y0_28;
        out_7[1] = y1_28;
        float g0_61 = __uint_as_float(gw[29] << 16);
        float g1_61 = __uint_as_float(gw[29] & 4294901760u);
        float x0_61 = __uint_as_float(xw[29] << 16);
        float x1_61 = __uint_as_float(xw[29] & 4294901760u);
        float y0_29 = __fmaf_rn(nkx_7, x0_61, rq_7 * (g0_61 * ww[2]));
        float y1_29 = __fmaf_rn(nkx_7, x1_61, rq_7 * (g1_61 * ww[3]));
        out_7[2] = y0_29;
        out_7[3] = y1_29;
        float g0_62 = __uint_as_float(gw[30] << 16);
        float g1_62 = __uint_as_float(gw[30] & 4294901760u);
        float x0_62 = __uint_as_float(xw[30] << 16);
        float x1_62 = __uint_as_float(xw[30] & 4294901760u);
        float y0_30 = __fmaf_rn(nkx_7, x0_62, rq_7 * (g0_62 * ww[4]));
        float y1_30 = __fmaf_rn(nkx_7, x1_62, rq_7 * (g1_62 * ww[5]));
        out_7[4] = y0_30;
        out_7[5] = y1_30;
        float g0_63 = __uint_as_float(gw[31] << 16);
        float g1_63 = __uint_as_float(gw[31] & 4294901760u);
        float x0_63 = __uint_as_float(xw[31] << 16);
        float x1_63 = __uint_as_float(xw[31] & 4294901760u);
        float y0_31 = __fmaf_rn(nkx_7, x0_63, rq_7 * (g0_63 * ww[6]));
        float y1_31 = __fmaf_rn(nkx_7, x1_63, rq_7 * (g1_63 * ww[7]));
        out_7[6] = y0_31;
        out_7[7] = y1_31;
        unsigned int packed_7[4];
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_7[_lp*2 + 0], out_7[_lp*2+1 + 0]));
            packed_7[_lp] = *(uint32_t*)&_bf2;
        }
        reinterpret_cast<int4*>(dx + ((unsigned long long)row_7 * 256ull + (unsigned long long)col))[0] = reinterpret_cast<int4*>(packed_7)[0];
    }
}

} // extern "C"
