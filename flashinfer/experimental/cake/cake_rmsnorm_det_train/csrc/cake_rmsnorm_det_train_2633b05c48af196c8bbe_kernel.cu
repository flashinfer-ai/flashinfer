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

#define LAUNCH_MIN_BLOCKS 2

#include <math_constants.h>

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

extern "C" {

__global__ __launch_bounds__(128, LAUNCH_MIN_BLOCKS) void
kernel_cake_rmsnorm_det_train_2633b05c48af196c8bbe(__nv_bfloat16* __restrict__ g, __nv_bfloat16* __restrict__ x, float* __restrict__ w, float* __restrict__ r, __nv_bfloat16* __restrict__ dx, float* __restrict__ dw, double* __restrict__ partial, unsigned int* __restrict__ counters, __nv_bfloat16* __restrict__ g_h, int T, int n_chunks, int rows_per_chunk, long long g_stride_p0, long long x_stride_p0, long long g_h_stride_p0)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Prologue
    int* flags = reinterpret_cast<int*>(smem_raw + 0);
    const int flags_addr = smem + 0;
    float* dots = reinterpret_cast<float*>(smem_raw + 32);
    const int dots_addr = smem + 32;

    // === Task calls (declaration order) ===
    asm volatile("griddepcontrol.wait;" ::: "memory");
    unsigned int nxt = 0;
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    long long g_stride = g_stride_p0;
    long long x_stride = x_stride_p0;
    long long gh_stride = g_h_stride_p0;
    int rg = warp;
    int tir = tid - rg * 32;
    float ww[16];
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
            : "=r"(_ldv8_0_0), "=r"(_ldv8_0_1), "=r"(_ldv8_0_2), "=r"(_ldv8_0_3), "=r"(_ldv8_0_4), "=r"(_ldv8_0_5), "=r"(_ldv8_0_6), "=r"(_ldv8_0_7) : "l"((const void*)(w + ((unsigned long long)(tir * 8)))) : "memory");
        ww[0 + 0] = __uint_as_float(_ldv8_0_0);
        ww[0 + 1] = __uint_as_float(_ldv8_0_1);
        ww[0 + 2] = __uint_as_float(_ldv8_0_2);
        ww[0 + 3] = __uint_as_float(_ldv8_0_3);
        ww[0 + 4] = __uint_as_float(_ldv8_0_4);
        ww[0 + 5] = __uint_as_float(_ldv8_0_5);
        ww[0 + 6] = __uint_as_float(_ldv8_0_6);
        ww[0 + 7] = __uint_as_float(_ldv8_0_7);
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
            : "=r"(_ldv8_1_0), "=r"(_ldv8_1_1), "=r"(_ldv8_1_2), "=r"(_ldv8_1_3), "=r"(_ldv8_1_4), "=r"(_ldv8_1_5), "=r"(_ldv8_1_6), "=r"(_ldv8_1_7) : "l"((const void*)(w + ((unsigned long long)((32 + tir) * 8)))) : "memory");
        ww[8 + 0] = __uint_as_float(_ldv8_1_0);
        ww[8 + 1] = __uint_as_float(_ldv8_1_1);
        ww[8 + 2] = __uint_as_float(_ldv8_1_2);
        ww[8 + 3] = __uint_as_float(_ldv8_1_3);
        ww[8 + 4] = __uint_as_float(_ldv8_1_4);
        ww[8 + 5] = __uint_as_float(_ldv8_1_5);
        ww[8 + 6] = __uint_as_float(_ldv8_1_6);
        ww[8 + 7] = __uint_as_float(_ldv8_1_7);
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
    unsigned int gw[8];
    unsigned int xw[8];
    unsigned int hw[1];
    float rr[1];
    float dot[1];
    int n_rows = (T + 3) / 4;
    int n_items = n_rows + 32;
    if (T == 1) {
        n_items = n_rows;
        n_items = 0;
        if ((int)bid == 0) {
            unsigned int gq[2];
            unsigned int xq[2];
            unsigned int hq[2];
            float wq[4];
            float rq = r[0];
            {
                uint2 _vld_2;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_2.x), "=r"(_vld_2.y) : "l"((const void*)(g + (unsigned long long)(4 * (int)tid))), "l"(0x14F0000000000000ULL) : "memory");
                uint2* _vdst_2 = reinterpret_cast<uint2*>(&gq[0]);
                *_vdst_2 = _vld_2;
            }
            {
                uint2 _vld_3;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_3.x), "=r"(_vld_3.y) : "l"((const void*)(x + (unsigned long long)(4 * (int)tid))), "l"(0x14F0000000000000ULL) : "memory");
                uint2* _vdst_3 = reinterpret_cast<uint2*>(&xq[0]);
                *_vdst_3 = _vld_3;
            }
            {
                float4 _v4 = *reinterpret_cast<const float4*>(w + (unsigned long long)(4 * (int)tid));
                wq[0 + 0] = _v4.x;
                wq[0 + 1] = _v4.y;
                wq[0 + 2] = _v4.z;
                wq[0 + 3] = _v4.w;
            }
            float2 rr2 = make_float2(rq, rq);
            float2 zero2 = make_float2(0.0f, 0.0f);
            float o4_0[4];
            float2 _f32x2_mul_a_5 = make_float2(__uint_as_float(gq[0] << 16), __uint_as_float(gq[0] & 4294901760u));
            float2 _f32x2_mul_b_6 = make_float2(__uint_as_float(xq[0] << 16), __uint_as_float(xq[0] & 4294901760u));
            float2 gx_0;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx_0) : "l"(*(const unsigned long long*)&_f32x2_mul_a_5), "l"(*(const unsigned long long*)&_f32x2_mul_b_6));
            float2 p_0 = add_f32x2_noftz(mul_f32x2_noftz(gx_0, rr2), zero2);
            o4_0[0] = p_0.x;
            o4_0[1] = p_0.y;
            float2 _f32x2_mul_a_7 = make_float2(__uint_as_float(gq[1] << 16), __uint_as_float(gq[1] & 4294901760u));
            float2 _f32x2_mul_b_8 = make_float2(__uint_as_float(xq[1] << 16), __uint_as_float(xq[1] & 4294901760u));
            float2 gx_1;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_7), "l"(*(const unsigned long long*)&_f32x2_mul_b_8));
            float2 p_1 = add_f32x2_noftz(mul_f32x2_noftz(gx_1, rr2), zero2);
            o4_0[2] = p_1.x;
            o4_0[3] = p_1.y;
            {
                float4 _v4 = make_float4(o4_0[0 + 0], o4_0[0 + 1], o4_0[0 + 2], o4_0[0 + 3]);
                *reinterpret_cast<float4*>(dw + (unsigned long long)(4 * (int)tid)) = _v4;
            }
            float dq = 0.0f;
            dq = __fmaf_rn(__uint_as_float(gq[0] << 16) * wq[0], __uint_as_float(xq[0] << 16), dq);
            dq = __fmaf_rn(__uint_as_float(gq[0] & 4294901760u) * wq[1], __uint_as_float(xq[0] & 4294901760u), dq);
            dq = __fmaf_rn(__uint_as_float(gq[1] << 16) * wq[2], __uint_as_float(xq[1] << 16), dq);
            dq = __fmaf_rn(__uint_as_float(gq[1] & 4294901760u) * wq[3], __uint_as_float(xq[1] & 4294901760u), dq);
            float ws = dq;
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1)
                ws += __shfl_xor_sync(0xFFFFFFFF, ws, offset);
            if (lane == 0) {
                dots[warp] = ws;
            }
            __syncthreads();
            float tree[4];
            tree[0] = dots[0];
            tree[1] = dots[1];
            tree[2] = dots[2];
            tree[3] = dots[3];
            tree[0] = tree[0] + tree[1];
            tree[2] = tree[2] + tree[3];
            tree[0] = tree[0] + tree[2];
            float cq = 0.0f;
            cq = tree[0] * 0.001953125f;
            float kx = rq * rq * rq * cq;
            float nkx = -kx;
            float out[4];
            float g0_0 = __uint_as_float(gq[0] << 16);
            float g1_0 = __uint_as_float(gq[0] & 4294901760u);
            float x0_0 = __uint_as_float(xq[0] << 16);
            float x1_0 = __uint_as_float(xq[0] & 4294901760u);
            float y0_0 = __fmaf_rn(nkx, x0_0, rq * (g0_0 * wq[0]));
            float y1_0 = __fmaf_rn(nkx, x1_0, rq * (g1_0 * wq[1]));
            out[0] = y0_0;
            out[1] = y1_0;
            float g0_1 = __uint_as_float(gq[1] << 16);
            float g1_1 = __uint_as_float(gq[1] & 4294901760u);
            float x0_1 = __uint_as_float(xq[1] << 16);
            float x1_1 = __uint_as_float(xq[1] & 4294901760u);
            float y0_1 = __fmaf_rn(nkx, x0_1, rq * (g0_1 * wq[2]));
            float y1_1 = __fmaf_rn(nkx, x1_1, rq * (g1_1 * wq[3]));
            out[2] = y0_1;
            out[3] = y1_1;
            unsigned int packed[2];
            #pragma unroll
            for (int _lp = 0; _lp < 2; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out[_lp*2 + 0], out[_lp*2+1 + 0]));
                packed[_lp] = *(uint32_t*)&_bf2;
            }
            reinterpret_cast<int2*>(dx + ((unsigned long long)(4 * (int)tid)))[0] = reinterpret_cast<int2*>(packed)[0];
        }
    }
}

} // extern "C"
