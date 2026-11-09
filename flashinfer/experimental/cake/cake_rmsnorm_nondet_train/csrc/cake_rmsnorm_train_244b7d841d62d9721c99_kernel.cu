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

__global__ __launch_bounds__(320, LAUNCH_MIN_BLOCKS) void
kernel_cake_rmsnorm_train_244b7d841d62d9721c99(__nv_bfloat16* __restrict__ g, __nv_bfloat16* __restrict__ x, float* __restrict__ w, float* __restrict__ r, __nv_bfloat16* __restrict__ dx, float* __restrict__ dw, double* __restrict__ partial, unsigned int* __restrict__ counters, __nv_bfloat16* __restrict__ g_h, int T, int n_chunks, int rows_per_chunk, long long g_stride_p0, long long x_stride_p0, long long g_h_stride_p0)
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
    int* flags = reinterpret_cast<int*>(smem_raw + 0);
    const int flags_addr = smem + 0;
    float* dots = reinterpret_cast<float*>(smem_raw + 32);
    const int dots_addr = smem + 32;
    double* sums = reinterpret_cast<double*>(smem_raw + 112);
    const int sums_addr = smem + 112;
    float* dwbuf = reinterpret_cast<float*>(smem_raw + 112);
    const int dwbuf_addr = smem + 112;

    // === Task calls (declaration order) ===
    asm volatile("griddepcontrol.wait;" ::: "memory");
    unsigned int nxt = 0;
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    long long g_stride = g_stride_p0;
    long long x_stride = x_stride_p0;
    long long gh_stride = g_h_stride_p0;
    int rg = warp / 10;
    int tir = tid - rg * 320;
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
            : "=r"(_ldv8_1_0), "=r"(_ldv8_1_1), "=r"(_ldv8_1_2), "=r"(_ldv8_1_3), "=r"(_ldv8_1_4), "=r"(_ldv8_1_5), "=r"(_ldv8_1_6), "=r"(_ldv8_1_7) : "l"((const void*)(w + ((unsigned long long)((320 + tir) * 8)))) : "memory");
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
    int n_rows = T;
    #pragma unroll 1
    for (int k = 0; k < n_rows + 80; k++) {
        int it = (int)bid + k * (int)gridDim.x;
        if (it >= n_rows + 80) {
            break;
        }
        if (it < n_rows) {
            int row = it + rg;
            int _min_2 = ((row) < (T - 1) ? (row) : (T - 1));
            unsigned long long g_base = (unsigned long long)_min_2 * (unsigned long long)g_stride;
            int _min_3 = ((row) < (T - 1) ? (row) : (T - 1));
            unsigned long long x_base = (unsigned long long)_min_3 * (unsigned long long)x_stride;
            int _min_4 = ((row) < (T - 1) ? (row) : (T - 1));
            rr[0] = r[_min_4];
            {
                const uint4* _vptr_5 = reinterpret_cast<const uint4*>(g + g_base + (unsigned long long)(tir * 8));
                uint4* _vdst_5 = reinterpret_cast<uint4*>(&gw[0]);
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                        : "=r"(_vdst_5[_blk].x), "=r"(_vdst_5[_blk].y), "=r"(_vdst_5[_blk].z), "=r"(_vdst_5[_blk].w) : "l"((const void*)(_vptr_5 + _blk)), "l"(0x14F0000000000000ULL) : "memory");
                }
            }
            {
                const uint4* _vptr_6 = reinterpret_cast<const uint4*>(x + x_base + (unsigned long long)(tir * 8));
                uint4* _vdst_6 = reinterpret_cast<uint4*>(&xw[0]);
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                        : "=r"(_vdst_6[_blk].x), "=r"(_vdst_6[_blk].y), "=r"(_vdst_6[_blk].z), "=r"(_vdst_6[_blk].w) : "l"((const void*)(_vptr_6 + _blk)), "l"(0x14F0000000000000ULL) : "memory");
                }
            }
            {
                const uint4* _vptr_7 = reinterpret_cast<const uint4*>(g + g_base + (unsigned long long)((320 + tir) * 8));
                uint4* _vdst_7 = reinterpret_cast<uint4*>(&gw[4]);
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                        : "=r"(_vdst_7[_blk].x), "=r"(_vdst_7[_blk].y), "=r"(_vdst_7[_blk].z), "=r"(_vdst_7[_blk].w) : "l"((const void*)(_vptr_7 + _blk)), "l"(0x14F0000000000000ULL) : "memory");
                }
            }
            {
                const uint4* _vptr_8 = reinterpret_cast<const uint4*>(x + x_base + (unsigned long long)((320 + tir) * 8));
                uint4* _vdst_8 = reinterpret_cast<uint4*>(&xw[4]);
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                        : "=r"(_vdst_8[_blk].x), "=r"(_vdst_8[_blk].y), "=r"(_vdst_8[_blk].z), "=r"(_vdst_8[_blk].w) : "l"((const void*)(_vptr_8 + _blk)), "l"(0x14F0000000000000ULL) : "memory");
                }
            }
            float dq = 0.0f;
            dq = __fmaf_rn(__uint_as_float(gw[0] << 16) * ww[0], __uint_as_float(xw[0] << 16), dq);
            dq = __fmaf_rn(__uint_as_float(gw[0] & 4294901760u) * ww[1], __uint_as_float(xw[0] & 4294901760u), dq);
            dq = __fmaf_rn(__uint_as_float(gw[1] << 16) * ww[2], __uint_as_float(xw[1] << 16), dq);
            dq = __fmaf_rn(__uint_as_float(gw[1] & 4294901760u) * ww[3], __uint_as_float(xw[1] & 4294901760u), dq);
            dq = __fmaf_rn(__uint_as_float(gw[2] << 16) * ww[4], __uint_as_float(xw[2] << 16), dq);
            dq = __fmaf_rn(__uint_as_float(gw[2] & 4294901760u) * ww[5], __uint_as_float(xw[2] & 4294901760u), dq);
            dq = __fmaf_rn(__uint_as_float(gw[3] << 16) * ww[6], __uint_as_float(xw[3] << 16), dq);
            dq = __fmaf_rn(__uint_as_float(gw[3] & 4294901760u) * ww[7], __uint_as_float(xw[3] & 4294901760u), dq);
            dq = __fmaf_rn(__uint_as_float(gw[4] << 16) * ww[8], __uint_as_float(xw[4] << 16), dq);
            dq = __fmaf_rn(__uint_as_float(gw[4] & 4294901760u) * ww[9], __uint_as_float(xw[4] & 4294901760u), dq);
            dq = __fmaf_rn(__uint_as_float(gw[5] << 16) * ww[10], __uint_as_float(xw[5] << 16), dq);
            dq = __fmaf_rn(__uint_as_float(gw[5] & 4294901760u) * ww[11], __uint_as_float(xw[5] & 4294901760u), dq);
            dq = __fmaf_rn(__uint_as_float(gw[6] << 16) * ww[12], __uint_as_float(xw[6] << 16), dq);
            dq = __fmaf_rn(__uint_as_float(gw[6] & 4294901760u) * ww[13], __uint_as_float(xw[6] & 4294901760u), dq);
            dq = __fmaf_rn(__uint_as_float(gw[7] << 16) * ww[14], __uint_as_float(xw[7] << 16), dq);
            dq = __fmaf_rn(__uint_as_float(gw[7] & 4294901760u) * ww[15], __uint_as_float(xw[7] & 4294901760u), dq);
            float _warp_reduce_9 = dq;
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1)
                _warp_reduce_9 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_9, offset);
            dot[0] = _warp_reduce_9;
            if (lane == 0) {
                dots[(k & 1) * 10 + warp] = dot[0];
            }
            __syncthreads();
            float tot = 0.0f;
            tot += dots[(k & 1) * 10 + rg * 10];
            tot += dots[(k & 1) * 10 + rg * 10 + 1];
            tot += dots[(k & 1) * 10 + rg * 10 + 2];
            tot += dots[(k & 1) * 10 + rg * 10 + 3];
            tot += dots[(k & 1) * 10 + rg * 10 + 4];
            tot += dots[(k & 1) * 10 + rg * 10 + 5];
            tot += dots[(k & 1) * 10 + rg * 10 + 6];
            tot += dots[(k & 1) * 10 + rg * 10 + 7];
            tot += dots[(k & 1) * 10 + rg * 10 + 8];
            tot += dots[(k & 1) * 10 + rg * 10 + 9];
            dot[0] = tot;
            float rq = rr[0];
            float cq = __fdiv_rn(dot[0], 5120.0f);
            float kx = rq * rq * rq * cq;
            float nkx = -kx;
            if (row < T) {
                float out_0[8];
                float g0_0 = __uint_as_float(gw[0] << 16);
                float g1_0 = __uint_as_float(gw[0] & 4294901760u);
                float x0_0 = __uint_as_float(xw[0] << 16);
                float x1_0 = __uint_as_float(xw[0] & 4294901760u);
                float y0_0 = __fmaf_rn(nkx, x0_0, rq * (g0_0 * ww[0]));
                float y1_0 = __fmaf_rn(nkx, x1_0, rq * (g1_0 * ww[1]));
                out_0[0] = y0_0;
                out_0[1] = y1_0;
                float g0_1 = __uint_as_float(gw[1] << 16);
                float g1_1 = __uint_as_float(gw[1] & 4294901760u);
                float x0_1 = __uint_as_float(xw[1] << 16);
                float x1_1 = __uint_as_float(xw[1] & 4294901760u);
                float y0_1 = __fmaf_rn(nkx, x0_1, rq * (g0_1 * ww[2]));
                float y1_1 = __fmaf_rn(nkx, x1_1, rq * (g1_1 * ww[3]));
                out_0[2] = y0_1;
                out_0[3] = y1_1;
                float g0_2 = __uint_as_float(gw[2] << 16);
                float g1_2 = __uint_as_float(gw[2] & 4294901760u);
                float x0_2 = __uint_as_float(xw[2] << 16);
                float x1_2 = __uint_as_float(xw[2] & 4294901760u);
                float y0_2 = __fmaf_rn(nkx, x0_2, rq * (g0_2 * ww[4]));
                float y1_2 = __fmaf_rn(nkx, x1_2, rq * (g1_2 * ww[5]));
                out_0[4] = y0_2;
                out_0[5] = y1_2;
                float g0_3 = __uint_as_float(gw[3] << 16);
                float g1_3 = __uint_as_float(gw[3] & 4294901760u);
                float x0_3 = __uint_as_float(xw[3] << 16);
                float x1_3 = __uint_as_float(xw[3] & 4294901760u);
                float y0_3 = __fmaf_rn(nkx, x0_3, rq * (g0_3 * ww[6]));
                float y1_3 = __fmaf_rn(nkx, x1_3, rq * (g1_3 * ww[7]));
                out_0[6] = y0_3;
                out_0[7] = y1_3;
                unsigned int packed_0[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_0[_lp*2 + 0], out_0[_lp*2+1 + 0]));
                    packed_0[_lp] = *(uint32_t*)&_bf2;
                }
                reinterpret_cast<int4*>(dx + ((unsigned long long)row * 5120ull + (unsigned long long)(tir * 8)))[0] = reinterpret_cast<int4*>(packed_0)[0];
                float out_1[8];
                float g0_4 = __uint_as_float(gw[4] << 16);
                float g1_4 = __uint_as_float(gw[4] & 4294901760u);
                float x0_4 = __uint_as_float(xw[4] << 16);
                float x1_4 = __uint_as_float(xw[4] & 4294901760u);
                float y0_4 = __fmaf_rn(nkx, x0_4, rq * (g0_4 * ww[8]));
                float y1_4 = __fmaf_rn(nkx, x1_4, rq * (g1_4 * ww[9]));
                out_1[0] = y0_4;
                out_1[1] = y1_4;
                float g0_5 = __uint_as_float(gw[5] << 16);
                float g1_5 = __uint_as_float(gw[5] & 4294901760u);
                float x0_5 = __uint_as_float(xw[5] << 16);
                float x1_5 = __uint_as_float(xw[5] & 4294901760u);
                float y0_5 = __fmaf_rn(nkx, x0_5, rq * (g0_5 * ww[10]));
                float y1_5 = __fmaf_rn(nkx, x1_5, rq * (g1_5 * ww[11]));
                out_1[2] = y0_5;
                out_1[3] = y1_5;
                float g0_6 = __uint_as_float(gw[6] << 16);
                float g1_6 = __uint_as_float(gw[6] & 4294901760u);
                float x0_6 = __uint_as_float(xw[6] << 16);
                float x1_6 = __uint_as_float(xw[6] & 4294901760u);
                float y0_6 = __fmaf_rn(nkx, x0_6, rq * (g0_6 * ww[12]));
                float y1_6 = __fmaf_rn(nkx, x1_6, rq * (g1_6 * ww[13]));
                out_1[4] = y0_6;
                out_1[5] = y1_6;
                float g0_7 = __uint_as_float(gw[7] << 16);
                float g1_7 = __uint_as_float(gw[7] & 4294901760u);
                float x0_7 = __uint_as_float(xw[7] << 16);
                float x1_7 = __uint_as_float(xw[7] & 4294901760u);
                float y0_7 = __fmaf_rn(nkx, x0_7, rq * (g0_7 * ww[14]));
                float y1_7 = __fmaf_rn(nkx, x1_7, rq * (g1_7 * ww[15]));
                out_1[6] = y0_7;
                out_1[7] = y1_7;
                unsigned int packed_1[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_1[_lp*2 + 0], out_1[_lp*2+1 + 0]));
                    packed_1[_lp] = *(uint32_t*)&_bf2;
                }
                reinterpret_cast<int4*>(dx + ((unsigned long long)row * 5120ull + (unsigned long long)((320 + tir) * 8)))[0] = reinterpret_cast<int4*>(packed_1)[0];
            }
        } else {
            int vl = tid % 8;
            int rl = tid / 8;
            unsigned int gw_0[32];
            unsigned int xw_0[32];
            unsigned int hw_0[1];
            float rr_0[8];
            float acc_0[8];
            float comp_0[8];
            acc_0[0] = 0.0f;
            comp_0[0] = 0.0f;
            acc_0[1] = 0.0f;
            comp_0[1] = 0.0f;
            acc_0[2] = 0.0f;
            comp_0[2] = 0.0f;
            acc_0[3] = 0.0f;
            comp_0[3] = 0.0f;
            acc_0[4] = 0.0f;
            comp_0[4] = 0.0f;
            acc_0[5] = 0.0f;
            comp_0[5] = 0.0f;
            acc_0[6] = 0.0f;
            comp_0[6] = 0.0f;
            acc_0[7] = 0.0f;
            comp_0[7] = 0.0f;
            #pragma unroll 1
            for (int t0 = rl; t0 < T; t0 += 320) {
                if (t0 < T) {
                    unsigned long long g_base_1 = (unsigned long long)t0 * (unsigned long long)g_stride;
                    unsigned long long x_base_1 = (unsigned long long)t0 * (unsigned long long)x_stride;
                    rr_0[0] = r[t0];
                    {
                        const uint4* _vptr_10 = reinterpret_cast<const uint4*>(g + g_base_1 + (unsigned long long)(((it - n_rows) * 8 + vl) * 8));
                        uint4* _vdst_10 = reinterpret_cast<uint4*>(&gw_0[0]);
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                : "=r"(_vdst_10[_blk].x), "=r"(_vdst_10[_blk].y), "=r"(_vdst_10[_blk].z), "=r"(_vdst_10[_blk].w) : "l"((const void*)(_vptr_10 + _blk)), "l"(0x14F0000000000000ULL) : "memory");
                        }
                    }
                    {
                        const uint4* _vptr_11 = reinterpret_cast<const uint4*>(x + x_base_1 + (unsigned long long)(((it - n_rows) * 8 + vl) * 8));
                        uint4* _vdst_11 = reinterpret_cast<uint4*>(&xw_0[0]);
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                : "=r"(_vdst_11[_blk].x), "=r"(_vdst_11[_blk].y), "=r"(_vdst_11[_blk].z), "=r"(_vdst_11[_blk].w) : "l"((const void*)(_vptr_11 + _blk)), "l"(0x14F0000000000000ULL) : "memory");
                        }
                    }
                }
                if (t0 + 40 < T) {
                    unsigned long long g_base_2 = (unsigned long long)(t0 + 40) * (unsigned long long)g_stride;
                    unsigned long long x_base_2 = (unsigned long long)(t0 + 40) * (unsigned long long)x_stride;
                    rr_0[1] = r[t0 + 40];
                    {
                        const uint4* _vptr_12 = reinterpret_cast<const uint4*>(g + g_base_2 + (unsigned long long)(((it - n_rows) * 8 + vl) * 8));
                        uint4* _vdst_12 = reinterpret_cast<uint4*>(&gw_0[4]);
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                : "=r"(_vdst_12[_blk].x), "=r"(_vdst_12[_blk].y), "=r"(_vdst_12[_blk].z), "=r"(_vdst_12[_blk].w) : "l"((const void*)(_vptr_12 + _blk)), "l"(0x14F0000000000000ULL) : "memory");
                        }
                    }
                    {
                        const uint4* _vptr_13 = reinterpret_cast<const uint4*>(x + x_base_2 + (unsigned long long)(((it - n_rows) * 8 + vl) * 8));
                        uint4* _vdst_13 = reinterpret_cast<uint4*>(&xw_0[4]);
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                : "=r"(_vdst_13[_blk].x), "=r"(_vdst_13[_blk].y), "=r"(_vdst_13[_blk].z), "=r"(_vdst_13[_blk].w) : "l"((const void*)(_vptr_13 + _blk)), "l"(0x14F0000000000000ULL) : "memory");
                        }
                    }
                }
                if (t0 + 80 < T) {
                    unsigned long long g_base_3 = (unsigned long long)(t0 + 80) * (unsigned long long)g_stride;
                    unsigned long long x_base_3 = (unsigned long long)(t0 + 80) * (unsigned long long)x_stride;
                    rr_0[2] = r[t0 + 80];
                    {
                        const uint4* _vptr_14 = reinterpret_cast<const uint4*>(g + g_base_3 + (unsigned long long)(((it - n_rows) * 8 + vl) * 8));
                        uint4* _vdst_14 = reinterpret_cast<uint4*>(&gw_0[8]);
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                : "=r"(_vdst_14[_blk].x), "=r"(_vdst_14[_blk].y), "=r"(_vdst_14[_blk].z), "=r"(_vdst_14[_blk].w) : "l"((const void*)(_vptr_14 + _blk)), "l"(0x14F0000000000000ULL) : "memory");
                        }
                    }
                    {
                        const uint4* _vptr_15 = reinterpret_cast<const uint4*>(x + x_base_3 + (unsigned long long)(((it - n_rows) * 8 + vl) * 8));
                        uint4* _vdst_15 = reinterpret_cast<uint4*>(&xw_0[8]);
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                : "=r"(_vdst_15[_blk].x), "=r"(_vdst_15[_blk].y), "=r"(_vdst_15[_blk].z), "=r"(_vdst_15[_blk].w) : "l"((const void*)(_vptr_15 + _blk)), "l"(0x14F0000000000000ULL) : "memory");
                        }
                    }
                }
                if (t0 + 120 < T) {
                    unsigned long long g_base_4 = (unsigned long long)(t0 + 120) * (unsigned long long)g_stride;
                    unsigned long long x_base_4 = (unsigned long long)(t0 + 120) * (unsigned long long)x_stride;
                    rr_0[3] = r[t0 + 120];
                    {
                        const uint4* _vptr_16 = reinterpret_cast<const uint4*>(g + g_base_4 + (unsigned long long)(((it - n_rows) * 8 + vl) * 8));
                        uint4* _vdst_16 = reinterpret_cast<uint4*>(&gw_0[12]);
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                : "=r"(_vdst_16[_blk].x), "=r"(_vdst_16[_blk].y), "=r"(_vdst_16[_blk].z), "=r"(_vdst_16[_blk].w) : "l"((const void*)(_vptr_16 + _blk)), "l"(0x14F0000000000000ULL) : "memory");
                        }
                    }
                    {
                        const uint4* _vptr_17 = reinterpret_cast<const uint4*>(x + x_base_4 + (unsigned long long)(((it - n_rows) * 8 + vl) * 8));
                        uint4* _vdst_17 = reinterpret_cast<uint4*>(&xw_0[12]);
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                : "=r"(_vdst_17[_blk].x), "=r"(_vdst_17[_blk].y), "=r"(_vdst_17[_blk].z), "=r"(_vdst_17[_blk].w) : "l"((const void*)(_vptr_17 + _blk)), "l"(0x14F0000000000000ULL) : "memory");
                        }
                    }
                }
                if (t0 + 160 < T) {
                    unsigned long long g_base_5 = (unsigned long long)(t0 + 160) * (unsigned long long)g_stride;
                    unsigned long long x_base_5 = (unsigned long long)(t0 + 160) * (unsigned long long)x_stride;
                    rr_0[4] = r[t0 + 160];
                    {
                        const uint4* _vptr_18 = reinterpret_cast<const uint4*>(g + g_base_5 + (unsigned long long)(((it - n_rows) * 8 + vl) * 8));
                        uint4* _vdst_18 = reinterpret_cast<uint4*>(&gw_0[16]);
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                : "=r"(_vdst_18[_blk].x), "=r"(_vdst_18[_blk].y), "=r"(_vdst_18[_blk].z), "=r"(_vdst_18[_blk].w) : "l"((const void*)(_vptr_18 + _blk)), "l"(0x14F0000000000000ULL) : "memory");
                        }
                    }
                    {
                        const uint4* _vptr_19 = reinterpret_cast<const uint4*>(x + x_base_5 + (unsigned long long)(((it - n_rows) * 8 + vl) * 8));
                        uint4* _vdst_19 = reinterpret_cast<uint4*>(&xw_0[16]);
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                : "=r"(_vdst_19[_blk].x), "=r"(_vdst_19[_blk].y), "=r"(_vdst_19[_blk].z), "=r"(_vdst_19[_blk].w) : "l"((const void*)(_vptr_19 + _blk)), "l"(0x14F0000000000000ULL) : "memory");
                        }
                    }
                }
                if (t0 + 200 < T) {
                    unsigned long long g_base_6 = (unsigned long long)(t0 + 200) * (unsigned long long)g_stride;
                    unsigned long long x_base_6 = (unsigned long long)(t0 + 200) * (unsigned long long)x_stride;
                    rr_0[5] = r[t0 + 200];
                    {
                        const uint4* _vptr_20 = reinterpret_cast<const uint4*>(g + g_base_6 + (unsigned long long)(((it - n_rows) * 8 + vl) * 8));
                        uint4* _vdst_20 = reinterpret_cast<uint4*>(&gw_0[20]);
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                : "=r"(_vdst_20[_blk].x), "=r"(_vdst_20[_blk].y), "=r"(_vdst_20[_blk].z), "=r"(_vdst_20[_blk].w) : "l"((const void*)(_vptr_20 + _blk)), "l"(0x14F0000000000000ULL) : "memory");
                        }
                    }
                    {
                        const uint4* _vptr_21 = reinterpret_cast<const uint4*>(x + x_base_6 + (unsigned long long)(((it - n_rows) * 8 + vl) * 8));
                        uint4* _vdst_21 = reinterpret_cast<uint4*>(&xw_0[20]);
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                : "=r"(_vdst_21[_blk].x), "=r"(_vdst_21[_blk].y), "=r"(_vdst_21[_blk].z), "=r"(_vdst_21[_blk].w) : "l"((const void*)(_vptr_21 + _blk)), "l"(0x14F0000000000000ULL) : "memory");
                        }
                    }
                }
                if (t0 + 240 < T) {
                    unsigned long long g_base_7 = (unsigned long long)(t0 + 240) * (unsigned long long)g_stride;
                    unsigned long long x_base_7 = (unsigned long long)(t0 + 240) * (unsigned long long)x_stride;
                    rr_0[6] = r[t0 + 240];
                    {
                        const uint4* _vptr_22 = reinterpret_cast<const uint4*>(g + g_base_7 + (unsigned long long)(((it - n_rows) * 8 + vl) * 8));
                        uint4* _vdst_22 = reinterpret_cast<uint4*>(&gw_0[24]);
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                : "=r"(_vdst_22[_blk].x), "=r"(_vdst_22[_blk].y), "=r"(_vdst_22[_blk].z), "=r"(_vdst_22[_blk].w) : "l"((const void*)(_vptr_22 + _blk)), "l"(0x14F0000000000000ULL) : "memory");
                        }
                    }
                    {
                        const uint4* _vptr_23 = reinterpret_cast<const uint4*>(x + x_base_7 + (unsigned long long)(((it - n_rows) * 8 + vl) * 8));
                        uint4* _vdst_23 = reinterpret_cast<uint4*>(&xw_0[24]);
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                : "=r"(_vdst_23[_blk].x), "=r"(_vdst_23[_blk].y), "=r"(_vdst_23[_blk].z), "=r"(_vdst_23[_blk].w) : "l"((const void*)(_vptr_23 + _blk)), "l"(0x14F0000000000000ULL) : "memory");
                        }
                    }
                }
                if (t0 + 280 < T) {
                    unsigned long long g_base_8 = (unsigned long long)(t0 + 280) * (unsigned long long)g_stride;
                    unsigned long long x_base_8 = (unsigned long long)(t0 + 280) * (unsigned long long)x_stride;
                    rr_0[7] = r[t0 + 280];
                    {
                        const uint4* _vptr_24 = reinterpret_cast<const uint4*>(g + g_base_8 + (unsigned long long)(((it - n_rows) * 8 + vl) * 8));
                        uint4* _vdst_24 = reinterpret_cast<uint4*>(&gw_0[28]);
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                : "=r"(_vdst_24[_blk].x), "=r"(_vdst_24[_blk].y), "=r"(_vdst_24[_blk].z), "=r"(_vdst_24[_blk].w) : "l"((const void*)(_vptr_24 + _blk)), "l"(0x14F0000000000000ULL) : "memory");
                        }
                    }
                    {
                        const uint4* _vptr_25 = reinterpret_cast<const uint4*>(x + x_base_8 + (unsigned long long)(((it - n_rows) * 8 + vl) * 8));
                        uint4* _vdst_25 = reinterpret_cast<uint4*>(&xw_0[28]);
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                : "=r"(_vdst_25[_blk].x), "=r"(_vdst_25[_blk].y), "=r"(_vdst_25[_blk].z), "=r"(_vdst_25[_blk].w) : "l"((const void*)(_vptr_25 + _blk)), "l"(0x14F0000000000000ULL) : "memory");
                        }
                    }
                }
                if (t0 < T) {
                    float2 rr2_0 = make_float2(rr_0[0], rr_0[0]);
                    float2 _f32x2_mul_a_26 = make_float2(__uint_as_float(gw_0[0] << 16), __uint_as_float(gw_0[0] & 4294901760u));
                    float2 _f32x2_mul_b_27 = make_float2(__uint_as_float(xw_0[0] << 16), __uint_as_float(xw_0[0] & 4294901760u));
                    float2 gx;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx) : "l"(*(const unsigned long long*)&_f32x2_mul_a_26), "l"(*(const unsigned long long*)&_f32x2_mul_b_27));
                    float2 p;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&p) : "l"(*(const unsigned long long*)&gx), "l"(*(const unsigned long long*)&rr2_0));
                    float2 a = make_float2(acc_0[0], acc_0[1]);
                    float2 s = add_f32x2_noftz(a, p);
                    float2 bn = sub_f32x2_noftz(a, s);
                    float2 av = add_f32x2_noftz(s, bn);
                    float2 db = fma_f32x2_rn_noftz(gx, rr2_0, bn);
                    float2 c_0 = add_f32x2_noftz(add_f32x2_noftz(make_float2(comp_0[0], comp_0[1]), sub_f32x2_noftz(a, av)), db);
                    acc_0[0] = s.x;
                    acc_0[1] = s.y;
                    comp_0[0] = c_0.x;
                    comp_0[1] = c_0.y;
                    float2 _f32x2_mul_a_28 = make_float2(__uint_as_float(gw_0[1] << 16), __uint_as_float(gw_0[1] & 4294901760u));
                    float2 _f32x2_mul_b_29 = make_float2(__uint_as_float(xw_0[1] << 16), __uint_as_float(xw_0[1] & 4294901760u));
                    float2 gx_0;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx_0) : "l"(*(const unsigned long long*)&_f32x2_mul_a_28), "l"(*(const unsigned long long*)&_f32x2_mul_b_29));
                    float2 p_0;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&p_0) : "l"(*(const unsigned long long*)&gx_0), "l"(*(const unsigned long long*)&rr2_0));
                    float2 a_0 = make_float2(acc_0[2], acc_0[3]);
                    float2 s_0 = add_f32x2_noftz(a_0, p_0);
                    float2 bn_0 = sub_f32x2_noftz(a_0, s_0);
                    float2 av_0 = add_f32x2_noftz(s_0, bn_0);
                    float2 db_0 = fma_f32x2_rn_noftz(gx_0, rr2_0, bn_0);
                    float2 c_1 = add_f32x2_noftz(add_f32x2_noftz(make_float2(comp_0[2], comp_0[3]), sub_f32x2_noftz(a_0, av_0)), db_0);
                    acc_0[2] = s_0.x;
                    acc_0[3] = s_0.y;
                    comp_0[2] = c_1.x;
                    comp_0[3] = c_1.y;
                    float2 _f32x2_mul_a_30 = make_float2(__uint_as_float(gw_0[2] << 16), __uint_as_float(gw_0[2] & 4294901760u));
                    float2 _f32x2_mul_b_31 = make_float2(__uint_as_float(xw_0[2] << 16), __uint_as_float(xw_0[2] & 4294901760u));
                    float2 gx_1;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_30), "l"(*(const unsigned long long*)&_f32x2_mul_b_31));
                    float2 p_1;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&p_1) : "l"(*(const unsigned long long*)&gx_1), "l"(*(const unsigned long long*)&rr2_0));
                    float2 a_1 = make_float2(acc_0[4], acc_0[5]);
                    float2 s_1 = add_f32x2_noftz(a_1, p_1);
                    float2 bn_1 = sub_f32x2_noftz(a_1, s_1);
                    float2 av_1 = add_f32x2_noftz(s_1, bn_1);
                    float2 db_1 = fma_f32x2_rn_noftz(gx_1, rr2_0, bn_1);
                    float2 c_2 = add_f32x2_noftz(add_f32x2_noftz(make_float2(comp_0[4], comp_0[5]), sub_f32x2_noftz(a_1, av_1)), db_1);
                    acc_0[4] = s_1.x;
                    acc_0[5] = s_1.y;
                    comp_0[4] = c_2.x;
                    comp_0[5] = c_2.y;
                    float2 _f32x2_mul_a_32 = make_float2(__uint_as_float(gw_0[3] << 16), __uint_as_float(gw_0[3] & 4294901760u));
                    float2 _f32x2_mul_b_33 = make_float2(__uint_as_float(xw_0[3] << 16), __uint_as_float(xw_0[3] & 4294901760u));
                    float2 gx_2;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx_2) : "l"(*(const unsigned long long*)&_f32x2_mul_a_32), "l"(*(const unsigned long long*)&_f32x2_mul_b_33));
                    float2 p_2;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&p_2) : "l"(*(const unsigned long long*)&gx_2), "l"(*(const unsigned long long*)&rr2_0));
                    float2 a_2 = make_float2(acc_0[6], acc_0[7]);
                    float2 s_2 = add_f32x2_noftz(a_2, p_2);
                    float2 bn_2 = sub_f32x2_noftz(a_2, s_2);
                    float2 av_2 = add_f32x2_noftz(s_2, bn_2);
                    float2 db_2 = fma_f32x2_rn_noftz(gx_2, rr2_0, bn_2);
                    float2 c_3 = add_f32x2_noftz(add_f32x2_noftz(make_float2(comp_0[6], comp_0[7]), sub_f32x2_noftz(a_2, av_2)), db_2);
                    acc_0[6] = s_2.x;
                    acc_0[7] = s_2.y;
                    comp_0[6] = c_3.x;
                    comp_0[7] = c_3.y;
                }
                if (t0 + 40 < T) {
                    float2 rr2_1 = make_float2(rr_0[1], rr_0[1]);
                    float2 _f32x2_mul_a_34 = make_float2(__uint_as_float(gw_0[4] << 16), __uint_as_float(gw_0[4] & 4294901760u));
                    float2 _f32x2_mul_b_35 = make_float2(__uint_as_float(xw_0[4] << 16), __uint_as_float(xw_0[4] & 4294901760u));
                    float2 gx_3;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx_3) : "l"(*(const unsigned long long*)&_f32x2_mul_a_34), "l"(*(const unsigned long long*)&_f32x2_mul_b_35));
                    float2 p_3;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&p_3) : "l"(*(const unsigned long long*)&gx_3), "l"(*(const unsigned long long*)&rr2_1));
                    float2 a_3 = make_float2(acc_0[0], acc_0[1]);
                    float2 s_3 = add_f32x2_noftz(a_3, p_3);
                    float2 bn_3 = sub_f32x2_noftz(a_3, s_3);
                    float2 av_3 = add_f32x2_noftz(s_3, bn_3);
                    float2 db_3 = fma_f32x2_rn_noftz(gx_3, rr2_1, bn_3);
                    float2 c_4 = add_f32x2_noftz(add_f32x2_noftz(make_float2(comp_0[0], comp_0[1]), sub_f32x2_noftz(a_3, av_3)), db_3);
                    acc_0[0] = s_3.x;
                    acc_0[1] = s_3.y;
                    comp_0[0] = c_4.x;
                    comp_0[1] = c_4.y;
                    float2 _f32x2_mul_a_36 = make_float2(__uint_as_float(gw_0[5] << 16), __uint_as_float(gw_0[5] & 4294901760u));
                    float2 _f32x2_mul_b_37 = make_float2(__uint_as_float(xw_0[5] << 16), __uint_as_float(xw_0[5] & 4294901760u));
                    float2 gx_3_1;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx_3_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_36), "l"(*(const unsigned long long*)&_f32x2_mul_b_37));
                    float2 p_3_1;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&p_3_1) : "l"(*(const unsigned long long*)&gx_3_1), "l"(*(const unsigned long long*)&rr2_1));
                    float2 a_3_1 = make_float2(acc_0[2], acc_0[3]);
                    float2 s_3_1 = add_f32x2_noftz(a_3_1, p_3_1);
                    float2 bn_3_1 = sub_f32x2_noftz(a_3_1, s_3_1);
                    float2 av_3_1 = add_f32x2_noftz(s_3_1, bn_3_1);
                    float2 db_3_1 = fma_f32x2_rn_noftz(gx_3_1, rr2_1, bn_3_1);
                    float2 c_5 = add_f32x2_noftz(add_f32x2_noftz(make_float2(comp_0[2], comp_0[3]), sub_f32x2_noftz(a_3_1, av_3_1)), db_3_1);
                    acc_0[2] = s_3_1.x;
                    acc_0[3] = s_3_1.y;
                    comp_0[2] = c_5.x;
                    comp_0[3] = c_5.y;
                    float2 _f32x2_mul_a_38 = make_float2(__uint_as_float(gw_0[6] << 16), __uint_as_float(gw_0[6] & 4294901760u));
                    float2 _f32x2_mul_b_39 = make_float2(__uint_as_float(xw_0[6] << 16), __uint_as_float(xw_0[6] & 4294901760u));
                    float2 gx_4;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx_4) : "l"(*(const unsigned long long*)&_f32x2_mul_a_38), "l"(*(const unsigned long long*)&_f32x2_mul_b_39));
                    float2 p_4;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&p_4) : "l"(*(const unsigned long long*)&gx_4), "l"(*(const unsigned long long*)&rr2_1));
                    float2 a_4 = make_float2(acc_0[4], acc_0[5]);
                    float2 s_4 = add_f32x2_noftz(a_4, p_4);
                    float2 bn_4 = sub_f32x2_noftz(a_4, s_4);
                    float2 av_4 = add_f32x2_noftz(s_4, bn_4);
                    float2 db_4 = fma_f32x2_rn_noftz(gx_4, rr2_1, bn_4);
                    float2 c_6 = add_f32x2_noftz(add_f32x2_noftz(make_float2(comp_0[4], comp_0[5]), sub_f32x2_noftz(a_4, av_4)), db_4);
                    acc_0[4] = s_4.x;
                    acc_0[5] = s_4.y;
                    comp_0[4] = c_6.x;
                    comp_0[5] = c_6.y;
                    float2 _f32x2_mul_a_40 = make_float2(__uint_as_float(gw_0[7] << 16), __uint_as_float(gw_0[7] & 4294901760u));
                    float2 _f32x2_mul_b_41 = make_float2(__uint_as_float(xw_0[7] << 16), __uint_as_float(xw_0[7] & 4294901760u));
                    float2 gx_5;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx_5) : "l"(*(const unsigned long long*)&_f32x2_mul_a_40), "l"(*(const unsigned long long*)&_f32x2_mul_b_41));
                    float2 p_5;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&p_5) : "l"(*(const unsigned long long*)&gx_5), "l"(*(const unsigned long long*)&rr2_1));
                    float2 a_5 = make_float2(acc_0[6], acc_0[7]);
                    float2 s_5 = add_f32x2_noftz(a_5, p_5);
                    float2 bn_5 = sub_f32x2_noftz(a_5, s_5);
                    float2 av_5 = add_f32x2_noftz(s_5, bn_5);
                    float2 db_5 = fma_f32x2_rn_noftz(gx_5, rr2_1, bn_5);
                    float2 c_7 = add_f32x2_noftz(add_f32x2_noftz(make_float2(comp_0[6], comp_0[7]), sub_f32x2_noftz(a_5, av_5)), db_5);
                    acc_0[6] = s_5.x;
                    acc_0[7] = s_5.y;
                    comp_0[6] = c_7.x;
                    comp_0[7] = c_7.y;
                }
                if (t0 + 80 < T) {
                    float2 rr2_2 = make_float2(rr_0[2], rr_0[2]);
                    float2 _f32x2_mul_a_42 = make_float2(__uint_as_float(gw_0[8] << 16), __uint_as_float(gw_0[8] & 4294901760u));
                    float2 _f32x2_mul_b_43 = make_float2(__uint_as_float(xw_0[8] << 16), __uint_as_float(xw_0[8] & 4294901760u));
                    float2 gx_6;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx_6) : "l"(*(const unsigned long long*)&_f32x2_mul_a_42), "l"(*(const unsigned long long*)&_f32x2_mul_b_43));
                    float2 p_6;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&p_6) : "l"(*(const unsigned long long*)&gx_6), "l"(*(const unsigned long long*)&rr2_2));
                    float2 a_6 = make_float2(acc_0[0], acc_0[1]);
                    float2 s_6 = add_f32x2_noftz(a_6, p_6);
                    float2 bn_6 = sub_f32x2_noftz(a_6, s_6);
                    float2 av_6 = add_f32x2_noftz(s_6, bn_6);
                    float2 db_6 = fma_f32x2_rn_noftz(gx_6, rr2_2, bn_6);
                    float2 c_8 = add_f32x2_noftz(add_f32x2_noftz(make_float2(comp_0[0], comp_0[1]), sub_f32x2_noftz(a_6, av_6)), db_6);
                    acc_0[0] = s_6.x;
                    acc_0[1] = s_6.y;
                    comp_0[0] = c_8.x;
                    comp_0[1] = c_8.y;
                    float2 _f32x2_mul_a_44 = make_float2(__uint_as_float(gw_0[9] << 16), __uint_as_float(gw_0[9] & 4294901760u));
                    float2 _f32x2_mul_b_45 = make_float2(__uint_as_float(xw_0[9] << 16), __uint_as_float(xw_0[9] & 4294901760u));
                    float2 gx_6_1;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx_6_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_44), "l"(*(const unsigned long long*)&_f32x2_mul_b_45));
                    float2 p_6_1;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&p_6_1) : "l"(*(const unsigned long long*)&gx_6_1), "l"(*(const unsigned long long*)&rr2_2));
                    float2 a_6_1 = make_float2(acc_0[2], acc_0[3]);
                    float2 s_6_1 = add_f32x2_noftz(a_6_1, p_6_1);
                    float2 bn_6_1 = sub_f32x2_noftz(a_6_1, s_6_1);
                    float2 av_6_1 = add_f32x2_noftz(s_6_1, bn_6_1);
                    float2 db_6_1 = fma_f32x2_rn_noftz(gx_6_1, rr2_2, bn_6_1);
                    float2 c_9 = add_f32x2_noftz(add_f32x2_noftz(make_float2(comp_0[2], comp_0[3]), sub_f32x2_noftz(a_6_1, av_6_1)), db_6_1);
                    acc_0[2] = s_6_1.x;
                    acc_0[3] = s_6_1.y;
                    comp_0[2] = c_9.x;
                    comp_0[3] = c_9.y;
                    float2 _f32x2_mul_a_46 = make_float2(__uint_as_float(gw_0[10] << 16), __uint_as_float(gw_0[10] & 4294901760u));
                    float2 _f32x2_mul_b_47 = make_float2(__uint_as_float(xw_0[10] << 16), __uint_as_float(xw_0[10] & 4294901760u));
                    float2 gx_7;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx_7) : "l"(*(const unsigned long long*)&_f32x2_mul_a_46), "l"(*(const unsigned long long*)&_f32x2_mul_b_47));
                    float2 p_7;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&p_7) : "l"(*(const unsigned long long*)&gx_7), "l"(*(const unsigned long long*)&rr2_2));
                    float2 a_7 = make_float2(acc_0[4], acc_0[5]);
                    float2 s_7 = add_f32x2_noftz(a_7, p_7);
                    float2 bn_7 = sub_f32x2_noftz(a_7, s_7);
                    float2 av_7 = add_f32x2_noftz(s_7, bn_7);
                    float2 db_7 = fma_f32x2_rn_noftz(gx_7, rr2_2, bn_7);
                    float2 c_10 = add_f32x2_noftz(add_f32x2_noftz(make_float2(comp_0[4], comp_0[5]), sub_f32x2_noftz(a_7, av_7)), db_7);
                    acc_0[4] = s_7.x;
                    acc_0[5] = s_7.y;
                    comp_0[4] = c_10.x;
                    comp_0[5] = c_10.y;
                    float2 _f32x2_mul_a_48 = make_float2(__uint_as_float(gw_0[11] << 16), __uint_as_float(gw_0[11] & 4294901760u));
                    float2 _f32x2_mul_b_49 = make_float2(__uint_as_float(xw_0[11] << 16), __uint_as_float(xw_0[11] & 4294901760u));
                    float2 gx_8;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx_8) : "l"(*(const unsigned long long*)&_f32x2_mul_a_48), "l"(*(const unsigned long long*)&_f32x2_mul_b_49));
                    float2 p_8;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&p_8) : "l"(*(const unsigned long long*)&gx_8), "l"(*(const unsigned long long*)&rr2_2));
                    float2 a_8 = make_float2(acc_0[6], acc_0[7]);
                    float2 s_8 = add_f32x2_noftz(a_8, p_8);
                    float2 bn_8 = sub_f32x2_noftz(a_8, s_8);
                    float2 av_8 = add_f32x2_noftz(s_8, bn_8);
                    float2 db_8 = fma_f32x2_rn_noftz(gx_8, rr2_2, bn_8);
                    float2 c_11 = add_f32x2_noftz(add_f32x2_noftz(make_float2(comp_0[6], comp_0[7]), sub_f32x2_noftz(a_8, av_8)), db_8);
                    acc_0[6] = s_8.x;
                    acc_0[7] = s_8.y;
                    comp_0[6] = c_11.x;
                    comp_0[7] = c_11.y;
                }
                if (t0 + 120 < T) {
                    float2 rr2_3 = make_float2(rr_0[3], rr_0[3]);
                    float2 _f32x2_mul_a_50 = make_float2(__uint_as_float(gw_0[12] << 16), __uint_as_float(gw_0[12] & 4294901760u));
                    float2 _f32x2_mul_b_51 = make_float2(__uint_as_float(xw_0[12] << 16), __uint_as_float(xw_0[12] & 4294901760u));
                    float2 gx_9;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx_9) : "l"(*(const unsigned long long*)&_f32x2_mul_a_50), "l"(*(const unsigned long long*)&_f32x2_mul_b_51));
                    float2 p_9;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&p_9) : "l"(*(const unsigned long long*)&gx_9), "l"(*(const unsigned long long*)&rr2_3));
                    float2 a_9 = make_float2(acc_0[0], acc_0[1]);
                    float2 s_9 = add_f32x2_noftz(a_9, p_9);
                    float2 bn_9 = sub_f32x2_noftz(a_9, s_9);
                    float2 av_9 = add_f32x2_noftz(s_9, bn_9);
                    float2 db_9 = fma_f32x2_rn_noftz(gx_9, rr2_3, bn_9);
                    float2 c_12 = add_f32x2_noftz(add_f32x2_noftz(make_float2(comp_0[0], comp_0[1]), sub_f32x2_noftz(a_9, av_9)), db_9);
                    acc_0[0] = s_9.x;
                    acc_0[1] = s_9.y;
                    comp_0[0] = c_12.x;
                    comp_0[1] = c_12.y;
                    float2 _f32x2_mul_a_52 = make_float2(__uint_as_float(gw_0[13] << 16), __uint_as_float(gw_0[13] & 4294901760u));
                    float2 _f32x2_mul_b_53 = make_float2(__uint_as_float(xw_0[13] << 16), __uint_as_float(xw_0[13] & 4294901760u));
                    float2 gx_9_1;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx_9_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_52), "l"(*(const unsigned long long*)&_f32x2_mul_b_53));
                    float2 p_9_1;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&p_9_1) : "l"(*(const unsigned long long*)&gx_9_1), "l"(*(const unsigned long long*)&rr2_3));
                    float2 a_9_1 = make_float2(acc_0[2], acc_0[3]);
                    float2 s_9_1 = add_f32x2_noftz(a_9_1, p_9_1);
                    float2 bn_9_1 = sub_f32x2_noftz(a_9_1, s_9_1);
                    float2 av_9_1 = add_f32x2_noftz(s_9_1, bn_9_1);
                    float2 db_9_1 = fma_f32x2_rn_noftz(gx_9_1, rr2_3, bn_9_1);
                    float2 c_13 = add_f32x2_noftz(add_f32x2_noftz(make_float2(comp_0[2], comp_0[3]), sub_f32x2_noftz(a_9_1, av_9_1)), db_9_1);
                    acc_0[2] = s_9_1.x;
                    acc_0[3] = s_9_1.y;
                    comp_0[2] = c_13.x;
                    comp_0[3] = c_13.y;
                    float2 _f32x2_mul_a_54 = make_float2(__uint_as_float(gw_0[14] << 16), __uint_as_float(gw_0[14] & 4294901760u));
                    float2 _f32x2_mul_b_55 = make_float2(__uint_as_float(xw_0[14] << 16), __uint_as_float(xw_0[14] & 4294901760u));
                    float2 gx_10;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx_10) : "l"(*(const unsigned long long*)&_f32x2_mul_a_54), "l"(*(const unsigned long long*)&_f32x2_mul_b_55));
                    float2 p_10;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&p_10) : "l"(*(const unsigned long long*)&gx_10), "l"(*(const unsigned long long*)&rr2_3));
                    float2 a_10 = make_float2(acc_0[4], acc_0[5]);
                    float2 s_10 = add_f32x2_noftz(a_10, p_10);
                    float2 bn_10 = sub_f32x2_noftz(a_10, s_10);
                    float2 av_10 = add_f32x2_noftz(s_10, bn_10);
                    float2 db_10 = fma_f32x2_rn_noftz(gx_10, rr2_3, bn_10);
                    float2 c_14 = add_f32x2_noftz(add_f32x2_noftz(make_float2(comp_0[4], comp_0[5]), sub_f32x2_noftz(a_10, av_10)), db_10);
                    acc_0[4] = s_10.x;
                    acc_0[5] = s_10.y;
                    comp_0[4] = c_14.x;
                    comp_0[5] = c_14.y;
                    float2 _f32x2_mul_a_56 = make_float2(__uint_as_float(gw_0[15] << 16), __uint_as_float(gw_0[15] & 4294901760u));
                    float2 _f32x2_mul_b_57 = make_float2(__uint_as_float(xw_0[15] << 16), __uint_as_float(xw_0[15] & 4294901760u));
                    float2 gx_11;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx_11) : "l"(*(const unsigned long long*)&_f32x2_mul_a_56), "l"(*(const unsigned long long*)&_f32x2_mul_b_57));
                    float2 p_11;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&p_11) : "l"(*(const unsigned long long*)&gx_11), "l"(*(const unsigned long long*)&rr2_3));
                    float2 a_11 = make_float2(acc_0[6], acc_0[7]);
                    float2 s_11 = add_f32x2_noftz(a_11, p_11);
                    float2 bn_11 = sub_f32x2_noftz(a_11, s_11);
                    float2 av_11 = add_f32x2_noftz(s_11, bn_11);
                    float2 db_11 = fma_f32x2_rn_noftz(gx_11, rr2_3, bn_11);
                    float2 c_15 = add_f32x2_noftz(add_f32x2_noftz(make_float2(comp_0[6], comp_0[7]), sub_f32x2_noftz(a_11, av_11)), db_11);
                    acc_0[6] = s_11.x;
                    acc_0[7] = s_11.y;
                    comp_0[6] = c_15.x;
                    comp_0[7] = c_15.y;
                }
                if (t0 + 160 < T) {
                    float2 rr2_4 = make_float2(rr_0[4], rr_0[4]);
                    float2 _f32x2_mul_a_58 = make_float2(__uint_as_float(gw_0[16] << 16), __uint_as_float(gw_0[16] & 4294901760u));
                    float2 _f32x2_mul_b_59 = make_float2(__uint_as_float(xw_0[16] << 16), __uint_as_float(xw_0[16] & 4294901760u));
                    float2 gx_12;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx_12) : "l"(*(const unsigned long long*)&_f32x2_mul_a_58), "l"(*(const unsigned long long*)&_f32x2_mul_b_59));
                    float2 p_12;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&p_12) : "l"(*(const unsigned long long*)&gx_12), "l"(*(const unsigned long long*)&rr2_4));
                    float2 a_12 = make_float2(acc_0[0], acc_0[1]);
                    float2 s_12 = add_f32x2_noftz(a_12, p_12);
                    float2 bn_12 = sub_f32x2_noftz(a_12, s_12);
                    float2 av_12 = add_f32x2_noftz(s_12, bn_12);
                    float2 db_12 = fma_f32x2_rn_noftz(gx_12, rr2_4, bn_12);
                    float2 c_16 = add_f32x2_noftz(add_f32x2_noftz(make_float2(comp_0[0], comp_0[1]), sub_f32x2_noftz(a_12, av_12)), db_12);
                    acc_0[0] = s_12.x;
                    acc_0[1] = s_12.y;
                    comp_0[0] = c_16.x;
                    comp_0[1] = c_16.y;
                    float2 _f32x2_mul_a_60 = make_float2(__uint_as_float(gw_0[17] << 16), __uint_as_float(gw_0[17] & 4294901760u));
                    float2 _f32x2_mul_b_61 = make_float2(__uint_as_float(xw_0[17] << 16), __uint_as_float(xw_0[17] & 4294901760u));
                    float2 gx_12_1;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx_12_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_60), "l"(*(const unsigned long long*)&_f32x2_mul_b_61));
                    float2 p_12_1;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&p_12_1) : "l"(*(const unsigned long long*)&gx_12_1), "l"(*(const unsigned long long*)&rr2_4));
                    float2 a_12_1 = make_float2(acc_0[2], acc_0[3]);
                    float2 s_12_1 = add_f32x2_noftz(a_12_1, p_12_1);
                    float2 bn_12_1 = sub_f32x2_noftz(a_12_1, s_12_1);
                    float2 av_12_1 = add_f32x2_noftz(s_12_1, bn_12_1);
                    float2 db_12_1 = fma_f32x2_rn_noftz(gx_12_1, rr2_4, bn_12_1);
                    float2 c_17 = add_f32x2_noftz(add_f32x2_noftz(make_float2(comp_0[2], comp_0[3]), sub_f32x2_noftz(a_12_1, av_12_1)), db_12_1);
                    acc_0[2] = s_12_1.x;
                    acc_0[3] = s_12_1.y;
                    comp_0[2] = c_17.x;
                    comp_0[3] = c_17.y;
                    float2 _f32x2_mul_a_62 = make_float2(__uint_as_float(gw_0[18] << 16), __uint_as_float(gw_0[18] & 4294901760u));
                    float2 _f32x2_mul_b_63 = make_float2(__uint_as_float(xw_0[18] << 16), __uint_as_float(xw_0[18] & 4294901760u));
                    float2 gx_13;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx_13) : "l"(*(const unsigned long long*)&_f32x2_mul_a_62), "l"(*(const unsigned long long*)&_f32x2_mul_b_63));
                    float2 p_13;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&p_13) : "l"(*(const unsigned long long*)&gx_13), "l"(*(const unsigned long long*)&rr2_4));
                    float2 a_13 = make_float2(acc_0[4], acc_0[5]);
                    float2 s_13 = add_f32x2_noftz(a_13, p_13);
                    float2 bn_13 = sub_f32x2_noftz(a_13, s_13);
                    float2 av_13 = add_f32x2_noftz(s_13, bn_13);
                    float2 db_13 = fma_f32x2_rn_noftz(gx_13, rr2_4, bn_13);
                    float2 c_18 = add_f32x2_noftz(add_f32x2_noftz(make_float2(comp_0[4], comp_0[5]), sub_f32x2_noftz(a_13, av_13)), db_13);
                    acc_0[4] = s_13.x;
                    acc_0[5] = s_13.y;
                    comp_0[4] = c_18.x;
                    comp_0[5] = c_18.y;
                    float2 _f32x2_mul_a_64 = make_float2(__uint_as_float(gw_0[19] << 16), __uint_as_float(gw_0[19] & 4294901760u));
                    float2 _f32x2_mul_b_65 = make_float2(__uint_as_float(xw_0[19] << 16), __uint_as_float(xw_0[19] & 4294901760u));
                    float2 gx_14;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx_14) : "l"(*(const unsigned long long*)&_f32x2_mul_a_64), "l"(*(const unsigned long long*)&_f32x2_mul_b_65));
                    float2 p_14;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&p_14) : "l"(*(const unsigned long long*)&gx_14), "l"(*(const unsigned long long*)&rr2_4));
                    float2 a_14 = make_float2(acc_0[6], acc_0[7]);
                    float2 s_14 = add_f32x2_noftz(a_14, p_14);
                    float2 bn_14 = sub_f32x2_noftz(a_14, s_14);
                    float2 av_14 = add_f32x2_noftz(s_14, bn_14);
                    float2 db_14 = fma_f32x2_rn_noftz(gx_14, rr2_4, bn_14);
                    float2 c_19 = add_f32x2_noftz(add_f32x2_noftz(make_float2(comp_0[6], comp_0[7]), sub_f32x2_noftz(a_14, av_14)), db_14);
                    acc_0[6] = s_14.x;
                    acc_0[7] = s_14.y;
                    comp_0[6] = c_19.x;
                    comp_0[7] = c_19.y;
                }
                if (t0 + 200 < T) {
                    float2 rr2_5 = make_float2(rr_0[5], rr_0[5]);
                    float2 _f32x2_mul_a_66 = make_float2(__uint_as_float(gw_0[20] << 16), __uint_as_float(gw_0[20] & 4294901760u));
                    float2 _f32x2_mul_b_67 = make_float2(__uint_as_float(xw_0[20] << 16), __uint_as_float(xw_0[20] & 4294901760u));
                    float2 gx_15;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx_15) : "l"(*(const unsigned long long*)&_f32x2_mul_a_66), "l"(*(const unsigned long long*)&_f32x2_mul_b_67));
                    float2 p_15;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&p_15) : "l"(*(const unsigned long long*)&gx_15), "l"(*(const unsigned long long*)&rr2_5));
                    float2 a_15 = make_float2(acc_0[0], acc_0[1]);
                    float2 s_15 = add_f32x2_noftz(a_15, p_15);
                    float2 bn_15 = sub_f32x2_noftz(a_15, s_15);
                    float2 av_15 = add_f32x2_noftz(s_15, bn_15);
                    float2 db_15 = fma_f32x2_rn_noftz(gx_15, rr2_5, bn_15);
                    float2 c_20 = add_f32x2_noftz(add_f32x2_noftz(make_float2(comp_0[0], comp_0[1]), sub_f32x2_noftz(a_15, av_15)), db_15);
                    acc_0[0] = s_15.x;
                    acc_0[1] = s_15.y;
                    comp_0[0] = c_20.x;
                    comp_0[1] = c_20.y;
                    float2 _f32x2_mul_a_68 = make_float2(__uint_as_float(gw_0[21] << 16), __uint_as_float(gw_0[21] & 4294901760u));
                    float2 _f32x2_mul_b_69 = make_float2(__uint_as_float(xw_0[21] << 16), __uint_as_float(xw_0[21] & 4294901760u));
                    float2 gx_15_1;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx_15_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_68), "l"(*(const unsigned long long*)&_f32x2_mul_b_69));
                    float2 p_15_1;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&p_15_1) : "l"(*(const unsigned long long*)&gx_15_1), "l"(*(const unsigned long long*)&rr2_5));
                    float2 a_15_1 = make_float2(acc_0[2], acc_0[3]);
                    float2 s_15_1 = add_f32x2_noftz(a_15_1, p_15_1);
                    float2 bn_15_1 = sub_f32x2_noftz(a_15_1, s_15_1);
                    float2 av_15_1 = add_f32x2_noftz(s_15_1, bn_15_1);
                    float2 db_15_1 = fma_f32x2_rn_noftz(gx_15_1, rr2_5, bn_15_1);
                    float2 c_21 = add_f32x2_noftz(add_f32x2_noftz(make_float2(comp_0[2], comp_0[3]), sub_f32x2_noftz(a_15_1, av_15_1)), db_15_1);
                    acc_0[2] = s_15_1.x;
                    acc_0[3] = s_15_1.y;
                    comp_0[2] = c_21.x;
                    comp_0[3] = c_21.y;
                    float2 _f32x2_mul_a_70 = make_float2(__uint_as_float(gw_0[22] << 16), __uint_as_float(gw_0[22] & 4294901760u));
                    float2 _f32x2_mul_b_71 = make_float2(__uint_as_float(xw_0[22] << 16), __uint_as_float(xw_0[22] & 4294901760u));
                    float2 gx_16;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx_16) : "l"(*(const unsigned long long*)&_f32x2_mul_a_70), "l"(*(const unsigned long long*)&_f32x2_mul_b_71));
                    float2 p_16;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&p_16) : "l"(*(const unsigned long long*)&gx_16), "l"(*(const unsigned long long*)&rr2_5));
                    float2 a_16 = make_float2(acc_0[4], acc_0[5]);
                    float2 s_16 = add_f32x2_noftz(a_16, p_16);
                    float2 bn_16 = sub_f32x2_noftz(a_16, s_16);
                    float2 av_16 = add_f32x2_noftz(s_16, bn_16);
                    float2 db_16 = fma_f32x2_rn_noftz(gx_16, rr2_5, bn_16);
                    float2 c_22 = add_f32x2_noftz(add_f32x2_noftz(make_float2(comp_0[4], comp_0[5]), sub_f32x2_noftz(a_16, av_16)), db_16);
                    acc_0[4] = s_16.x;
                    acc_0[5] = s_16.y;
                    comp_0[4] = c_22.x;
                    comp_0[5] = c_22.y;
                    float2 _f32x2_mul_a_72 = make_float2(__uint_as_float(gw_0[23] << 16), __uint_as_float(gw_0[23] & 4294901760u));
                    float2 _f32x2_mul_b_73 = make_float2(__uint_as_float(xw_0[23] << 16), __uint_as_float(xw_0[23] & 4294901760u));
                    float2 gx_17;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx_17) : "l"(*(const unsigned long long*)&_f32x2_mul_a_72), "l"(*(const unsigned long long*)&_f32x2_mul_b_73));
                    float2 p_17;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&p_17) : "l"(*(const unsigned long long*)&gx_17), "l"(*(const unsigned long long*)&rr2_5));
                    float2 a_17 = make_float2(acc_0[6], acc_0[7]);
                    float2 s_17 = add_f32x2_noftz(a_17, p_17);
                    float2 bn_17 = sub_f32x2_noftz(a_17, s_17);
                    float2 av_17 = add_f32x2_noftz(s_17, bn_17);
                    float2 db_17 = fma_f32x2_rn_noftz(gx_17, rr2_5, bn_17);
                    float2 c_23 = add_f32x2_noftz(add_f32x2_noftz(make_float2(comp_0[6], comp_0[7]), sub_f32x2_noftz(a_17, av_17)), db_17);
                    acc_0[6] = s_17.x;
                    acc_0[7] = s_17.y;
                    comp_0[6] = c_23.x;
                    comp_0[7] = c_23.y;
                }
                if (t0 + 240 < T) {
                    float2 rr2_6 = make_float2(rr_0[6], rr_0[6]);
                    float2 _f32x2_mul_a_74 = make_float2(__uint_as_float(gw_0[24] << 16), __uint_as_float(gw_0[24] & 4294901760u));
                    float2 _f32x2_mul_b_75 = make_float2(__uint_as_float(xw_0[24] << 16), __uint_as_float(xw_0[24] & 4294901760u));
                    float2 gx_18;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx_18) : "l"(*(const unsigned long long*)&_f32x2_mul_a_74), "l"(*(const unsigned long long*)&_f32x2_mul_b_75));
                    float2 p_18;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&p_18) : "l"(*(const unsigned long long*)&gx_18), "l"(*(const unsigned long long*)&rr2_6));
                    float2 a_18 = make_float2(acc_0[0], acc_0[1]);
                    float2 s_18 = add_f32x2_noftz(a_18, p_18);
                    float2 bn_18 = sub_f32x2_noftz(a_18, s_18);
                    float2 av_18 = add_f32x2_noftz(s_18, bn_18);
                    float2 db_18 = fma_f32x2_rn_noftz(gx_18, rr2_6, bn_18);
                    float2 c_24 = add_f32x2_noftz(add_f32x2_noftz(make_float2(comp_0[0], comp_0[1]), sub_f32x2_noftz(a_18, av_18)), db_18);
                    acc_0[0] = s_18.x;
                    acc_0[1] = s_18.y;
                    comp_0[0] = c_24.x;
                    comp_0[1] = c_24.y;
                    float2 _f32x2_mul_a_76 = make_float2(__uint_as_float(gw_0[25] << 16), __uint_as_float(gw_0[25] & 4294901760u));
                    float2 _f32x2_mul_b_77 = make_float2(__uint_as_float(xw_0[25] << 16), __uint_as_float(xw_0[25] & 4294901760u));
                    float2 gx_18_1;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx_18_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_76), "l"(*(const unsigned long long*)&_f32x2_mul_b_77));
                    float2 p_18_1;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&p_18_1) : "l"(*(const unsigned long long*)&gx_18_1), "l"(*(const unsigned long long*)&rr2_6));
                    float2 a_18_1 = make_float2(acc_0[2], acc_0[3]);
                    float2 s_18_1 = add_f32x2_noftz(a_18_1, p_18_1);
                    float2 bn_18_1 = sub_f32x2_noftz(a_18_1, s_18_1);
                    float2 av_18_1 = add_f32x2_noftz(s_18_1, bn_18_1);
                    float2 db_18_1 = fma_f32x2_rn_noftz(gx_18_1, rr2_6, bn_18_1);
                    float2 c_25 = add_f32x2_noftz(add_f32x2_noftz(make_float2(comp_0[2], comp_0[3]), sub_f32x2_noftz(a_18_1, av_18_1)), db_18_1);
                    acc_0[2] = s_18_1.x;
                    acc_0[3] = s_18_1.y;
                    comp_0[2] = c_25.x;
                    comp_0[3] = c_25.y;
                    float2 _f32x2_mul_a_78 = make_float2(__uint_as_float(gw_0[26] << 16), __uint_as_float(gw_0[26] & 4294901760u));
                    float2 _f32x2_mul_b_79 = make_float2(__uint_as_float(xw_0[26] << 16), __uint_as_float(xw_0[26] & 4294901760u));
                    float2 gx_19;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx_19) : "l"(*(const unsigned long long*)&_f32x2_mul_a_78), "l"(*(const unsigned long long*)&_f32x2_mul_b_79));
                    float2 p_19;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&p_19) : "l"(*(const unsigned long long*)&gx_19), "l"(*(const unsigned long long*)&rr2_6));
                    float2 a_19 = make_float2(acc_0[4], acc_0[5]);
                    float2 s_19 = add_f32x2_noftz(a_19, p_19);
                    float2 bn_19 = sub_f32x2_noftz(a_19, s_19);
                    float2 av_19 = add_f32x2_noftz(s_19, bn_19);
                    float2 db_19 = fma_f32x2_rn_noftz(gx_19, rr2_6, bn_19);
                    float2 c_26 = add_f32x2_noftz(add_f32x2_noftz(make_float2(comp_0[4], comp_0[5]), sub_f32x2_noftz(a_19, av_19)), db_19);
                    acc_0[4] = s_19.x;
                    acc_0[5] = s_19.y;
                    comp_0[4] = c_26.x;
                    comp_0[5] = c_26.y;
                    float2 _f32x2_mul_a_80 = make_float2(__uint_as_float(gw_0[27] << 16), __uint_as_float(gw_0[27] & 4294901760u));
                    float2 _f32x2_mul_b_81 = make_float2(__uint_as_float(xw_0[27] << 16), __uint_as_float(xw_0[27] & 4294901760u));
                    float2 gx_20;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx_20) : "l"(*(const unsigned long long*)&_f32x2_mul_a_80), "l"(*(const unsigned long long*)&_f32x2_mul_b_81));
                    float2 p_20;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&p_20) : "l"(*(const unsigned long long*)&gx_20), "l"(*(const unsigned long long*)&rr2_6));
                    float2 a_20 = make_float2(acc_0[6], acc_0[7]);
                    float2 s_20 = add_f32x2_noftz(a_20, p_20);
                    float2 bn_20 = sub_f32x2_noftz(a_20, s_20);
                    float2 av_20 = add_f32x2_noftz(s_20, bn_20);
                    float2 db_20 = fma_f32x2_rn_noftz(gx_20, rr2_6, bn_20);
                    float2 c_27 = add_f32x2_noftz(add_f32x2_noftz(make_float2(comp_0[6], comp_0[7]), sub_f32x2_noftz(a_20, av_20)), db_20);
                    acc_0[6] = s_20.x;
                    acc_0[7] = s_20.y;
                    comp_0[6] = c_27.x;
                    comp_0[7] = c_27.y;
                }
                if (t0 + 280 < T) {
                    float2 rr2_7 = make_float2(rr_0[7], rr_0[7]);
                    float2 _f32x2_mul_a_82 = make_float2(__uint_as_float(gw_0[28] << 16), __uint_as_float(gw_0[28] & 4294901760u));
                    float2 _f32x2_mul_b_83 = make_float2(__uint_as_float(xw_0[28] << 16), __uint_as_float(xw_0[28] & 4294901760u));
                    float2 gx_21;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx_21) : "l"(*(const unsigned long long*)&_f32x2_mul_a_82), "l"(*(const unsigned long long*)&_f32x2_mul_b_83));
                    float2 p_21;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&p_21) : "l"(*(const unsigned long long*)&gx_21), "l"(*(const unsigned long long*)&rr2_7));
                    float2 a_21 = make_float2(acc_0[0], acc_0[1]);
                    float2 s_21 = add_f32x2_noftz(a_21, p_21);
                    float2 bn_21 = sub_f32x2_noftz(a_21, s_21);
                    float2 av_21 = add_f32x2_noftz(s_21, bn_21);
                    float2 db_21 = fma_f32x2_rn_noftz(gx_21, rr2_7, bn_21);
                    float2 c_28 = add_f32x2_noftz(add_f32x2_noftz(make_float2(comp_0[0], comp_0[1]), sub_f32x2_noftz(a_21, av_21)), db_21);
                    acc_0[0] = s_21.x;
                    acc_0[1] = s_21.y;
                    comp_0[0] = c_28.x;
                    comp_0[1] = c_28.y;
                    float2 _f32x2_mul_a_84 = make_float2(__uint_as_float(gw_0[29] << 16), __uint_as_float(gw_0[29] & 4294901760u));
                    float2 _f32x2_mul_b_85 = make_float2(__uint_as_float(xw_0[29] << 16), __uint_as_float(xw_0[29] & 4294901760u));
                    float2 gx_21_1;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx_21_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_84), "l"(*(const unsigned long long*)&_f32x2_mul_b_85));
                    float2 p_21_1;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&p_21_1) : "l"(*(const unsigned long long*)&gx_21_1), "l"(*(const unsigned long long*)&rr2_7));
                    float2 a_21_1 = make_float2(acc_0[2], acc_0[3]);
                    float2 s_21_1 = add_f32x2_noftz(a_21_1, p_21_1);
                    float2 bn_21_1 = sub_f32x2_noftz(a_21_1, s_21_1);
                    float2 av_21_1 = add_f32x2_noftz(s_21_1, bn_21_1);
                    float2 db_21_1 = fma_f32x2_rn_noftz(gx_21_1, rr2_7, bn_21_1);
                    float2 c_29 = add_f32x2_noftz(add_f32x2_noftz(make_float2(comp_0[2], comp_0[3]), sub_f32x2_noftz(a_21_1, av_21_1)), db_21_1);
                    acc_0[2] = s_21_1.x;
                    acc_0[3] = s_21_1.y;
                    comp_0[2] = c_29.x;
                    comp_0[3] = c_29.y;
                    float2 _f32x2_mul_a_86 = make_float2(__uint_as_float(gw_0[30] << 16), __uint_as_float(gw_0[30] & 4294901760u));
                    float2 _f32x2_mul_b_87 = make_float2(__uint_as_float(xw_0[30] << 16), __uint_as_float(xw_0[30] & 4294901760u));
                    float2 gx_22;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx_22) : "l"(*(const unsigned long long*)&_f32x2_mul_a_86), "l"(*(const unsigned long long*)&_f32x2_mul_b_87));
                    float2 p_22;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&p_22) : "l"(*(const unsigned long long*)&gx_22), "l"(*(const unsigned long long*)&rr2_7));
                    float2 a_22 = make_float2(acc_0[4], acc_0[5]);
                    float2 s_22 = add_f32x2_noftz(a_22, p_22);
                    float2 bn_22 = sub_f32x2_noftz(a_22, s_22);
                    float2 av_22 = add_f32x2_noftz(s_22, bn_22);
                    float2 db_22 = fma_f32x2_rn_noftz(gx_22, rr2_7, bn_22);
                    float2 c_30 = add_f32x2_noftz(add_f32x2_noftz(make_float2(comp_0[4], comp_0[5]), sub_f32x2_noftz(a_22, av_22)), db_22);
                    acc_0[4] = s_22.x;
                    acc_0[5] = s_22.y;
                    comp_0[4] = c_30.x;
                    comp_0[5] = c_30.y;
                    float2 _f32x2_mul_a_88 = make_float2(__uint_as_float(gw_0[31] << 16), __uint_as_float(gw_0[31] & 4294901760u));
                    float2 _f32x2_mul_b_89 = make_float2(__uint_as_float(xw_0[31] << 16), __uint_as_float(xw_0[31] & 4294901760u));
                    float2 gx_23;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx_23) : "l"(*(const unsigned long long*)&_f32x2_mul_a_88), "l"(*(const unsigned long long*)&_f32x2_mul_b_89));
                    float2 p_23;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&p_23) : "l"(*(const unsigned long long*)&gx_23), "l"(*(const unsigned long long*)&rr2_7));
                    float2 a_23 = make_float2(acc_0[6], acc_0[7]);
                    float2 s_23 = add_f32x2_noftz(a_23, p_23);
                    float2 bn_23 = sub_f32x2_noftz(a_23, s_23);
                    float2 av_23 = add_f32x2_noftz(s_23, bn_23);
                    float2 db_23 = fma_f32x2_rn_noftz(gx_23, rr2_7, bn_23);
                    float2 c_31 = add_f32x2_noftz(add_f32x2_noftz(make_float2(comp_0[6], comp_0[7]), sub_f32x2_noftz(a_23, av_23)), db_23);
                    acc_0[6] = s_23.x;
                    acc_0[7] = s_23.y;
                    comp_0[6] = c_31.x;
                    comp_0[7] = c_31.y;
                }
            }
            if (rl < T) {
                sums[rl * 64 + vl * 8] = (double)acc_0[0] + (double)comp_0[0];
                sums[rl * 64 + vl * 8 + 1] = (double)acc_0[1] + (double)comp_0[1];
                sums[rl * 64 + vl * 8 + 2] = (double)acc_0[2] + (double)comp_0[2];
                sums[rl * 64 + vl * 8 + 3] = (double)acc_0[3] + (double)comp_0[3];
                sums[rl * 64 + vl * 8 + 4] = (double)acc_0[4] + (double)comp_0[4];
                sums[rl * 64 + vl * 8 + 5] = (double)acc_0[5] + (double)comp_0[5];
                sums[rl * 64 + vl * 8 + 6] = (double)acc_0[6] + (double)comp_0[6];
                sums[rl * 64 + vl * 8 + 7] = (double)acc_0[7] + (double)comp_0[7];
            }
            __syncthreads();
            int cl = tid % 64;
            int s8 = tid / 64 * 8;
            if (s8 < T) {
                double part = sums[s8 * 64 + cl];
                if (s8 + 1 < T) {
                    part = part + sums[(s8 + 1) * 64 + cl];
                }
                if (s8 + 1 + 1 < T) {
                    part = part + sums[(s8 + 1 + 1) * 64 + cl];
                }
                if (s8 + 2 + 1 < T) {
                    part = part + sums[(s8 + 2 + 1) * 64 + cl];
                }
                if (s8 + 3 + 1 < T) {
                    part = part + sums[(s8 + 3 + 1) * 64 + cl];
                }
                if (s8 + 4 + 1 < T) {
                    part = part + sums[(s8 + 4 + 1) * 64 + cl];
                }
                if (s8 + 5 + 1 < T) {
                    part = part + sums[(s8 + 5 + 1) * 64 + cl];
                }
                if (s8 + 6 + 1 < T) {
                    part = part + sums[(s8 + 6 + 1) * 64 + cl];
                }
                sums[s8 * 64 + cl] = part;
            }
            __syncthreads();
            if (tid < 64) {
                double tot_1 = sums[cl];
                if (8 < T) {
                    tot_1 = tot_1 + sums[512 + cl];
                }
                if (16 < T) {
                    tot_1 = tot_1 + sums[1024 + cl];
                }
                if (24 < T) {
                    tot_1 = tot_1 + sums[1536 + cl];
                }
                if (32 < T) {
                    tot_1 = tot_1 + sums[2048 + cl];
                }
                *(reinterpret_cast<float*>(dw) + ((it - n_rows) * 64 + cl)) = (float)tot_1;
            }
        }
        __syncthreads();
    }
}

} // extern "C"
