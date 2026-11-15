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


__device__ __forceinline__ unsigned int __as_u32(float v) {
    unsigned int u;
    asm("mov.b32 %0, %1;" : "=r"(u) : "f"(v));
    return u;
}
__device__ __forceinline__ unsigned int __as_u32(__nv_bfloat162 v) {
    return *reinterpret_cast<const unsigned int*>(&v);
}
__device__ __forceinline__ unsigned int __as_u32(unsigned int v) { return v; }
__device__ __forceinline__ unsigned int __as_u32(int v) {
    unsigned int u;
    asm("mov.b32 %0, %1;" : "=r"(u) : "r"(v));
    return u;
}

extern "C" {

__global__ __launch_bounds__(64, LAUNCH_MIN_BLOCKS) void
kernel_cake_rmsnorm_det_train_3c06f3e2f267255a1989(__nv_bfloat16* __restrict__ g, __nv_bfloat16* __restrict__ x, __nv_bfloat16* __restrict__ w, float* __restrict__ r, __nv_bfloat16* __restrict__ dx, float* __restrict__ dw, double* __restrict__ partial, unsigned int* __restrict__ counters, __nv_bfloat16* __restrict__ g_h, int T, int n_chunks, int rows_per_chunk, long long g_stride_p0, long long x_stride_p0, long long g_h_stride_p0)
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
    double* sums = reinterpret_cast<double*>(smem_raw + 48);
    const int sums_addr = smem + 48;
    float* sumsf = reinterpret_cast<float*>(smem_raw + 48);
    const int sumsf_addr = smem + 48;
    float* dwbuf = reinterpret_cast<float*>(smem_raw + 48);
    const int dwbuf_addr = smem + 48;

    // === Task calls (declaration order) ===
    asm volatile("griddepcontrol.wait;" ::: "memory");
    unsigned int nxt = 0;
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    long long g_stride = g_stride_p0;
    long long x_stride = x_stride_p0;
    long long gh_stride = g_h_stride_p0;
    int rg = warp;
    int tir = tid - rg * 32;
    unsigned int ww[8];
    {
        const uint4* _vptr_0 = reinterpret_cast<const uint4*>(w + (unsigned long long)(tir * 8));
        uint4* _vdst_0 = reinterpret_cast<uint4*>(&ww[0]);
        #pragma unroll
        for (int _blk = 0; _blk < 1; _blk++) {
            _vdst_0[_blk] = _vptr_0[_blk];
        }
    }
    {
        const uint4* _vptr_1 = reinterpret_cast<const uint4*>(w + (unsigned long long)((32 + tir) * 8));
        uint4* _vdst_1 = reinterpret_cast<uint4*>(&ww[4]);
        #pragma unroll
        for (int _blk = 0; _blk < 1; _blk++) {
            _vdst_1[_blk] = _vptr_1[_blk];
        }
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
    unsigned int wl[8];
    int n_rows = (T + 1) / 2;
    #pragma unroll 1
    for (int k = 0; k < n_rows + 32; k++) {
        int it = (int)bid + k * (int)gridDim.x;
        if (it >= n_rows + 32) {
            break;
        }
        if (it < n_rows) {
            int row = it * 2 + rg;
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
                const uint4* _vptr_7 = reinterpret_cast<const uint4*>(g + g_base + (unsigned long long)((32 + tir) * 8));
                uint4* _vdst_7 = reinterpret_cast<uint4*>(&gw[4]);
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                        : "=r"(_vdst_7[_blk].x), "=r"(_vdst_7[_blk].y), "=r"(_vdst_7[_blk].z), "=r"(_vdst_7[_blk].w) : "l"((const void*)(_vptr_7 + _blk)), "l"(0x14F0000000000000ULL) : "memory");
                }
            }
            {
                const uint4* _vptr_8 = reinterpret_cast<const uint4*>(x + x_base + (unsigned long long)((32 + tir) * 8));
                uint4* _vdst_8 = reinterpret_cast<uint4*>(&xw[4]);
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                        : "=r"(_vdst_8[_blk].x), "=r"(_vdst_8[_blk].y), "=r"(_vdst_8[_blk].z), "=r"(_vdst_8[_blk].w) : "l"((const void*)(_vptr_8 + _blk)), "l"(0x14F0000000000000ULL) : "memory");
                }
            }
            {
                const uint4* _vptr_9 = reinterpret_cast<const uint4*>(w + (unsigned long long)(tir * 8));
                uint4* _vdst_9 = reinterpret_cast<uint4*>(&wl[0]);
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    _vdst_9[_blk] = _vptr_9[_blk];
                }
            }
            {
                const uint4* _vptr_10 = reinterpret_cast<const uint4*>(w + (unsigned long long)((32 + tir) * 8));
                uint4* _vdst_10 = reinterpret_cast<uint4*>(&wl[4]);
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    _vdst_10[_blk] = _vptr_10[_blk];
                }
            }
            float dq = 0.0f;
            dq = __fmaf_rn(__uint_as_float(gw[0] << 16) * __uint_as_float(wl[0] << 16), __uint_as_float(xw[0] << 16), dq);
            dq = __fmaf_rn(__uint_as_float(gw[0] & 4294901760u) * __uint_as_float(wl[0] & 4294901760u), __uint_as_float(xw[0] & 4294901760u), dq);
            dq = __fmaf_rn(__uint_as_float(gw[1] << 16) * __uint_as_float(wl[1] << 16), __uint_as_float(xw[1] << 16), dq);
            dq = __fmaf_rn(__uint_as_float(gw[1] & 4294901760u) * __uint_as_float(wl[1] & 4294901760u), __uint_as_float(xw[1] & 4294901760u), dq);
            dq = __fmaf_rn(__uint_as_float(gw[2] << 16) * __uint_as_float(wl[2] << 16), __uint_as_float(xw[2] << 16), dq);
            dq = __fmaf_rn(__uint_as_float(gw[2] & 4294901760u) * __uint_as_float(wl[2] & 4294901760u), __uint_as_float(xw[2] & 4294901760u), dq);
            dq = __fmaf_rn(__uint_as_float(gw[3] << 16) * __uint_as_float(wl[3] << 16), __uint_as_float(xw[3] << 16), dq);
            dq = __fmaf_rn(__uint_as_float(gw[3] & 4294901760u) * __uint_as_float(wl[3] & 4294901760u), __uint_as_float(xw[3] & 4294901760u), dq);
            dq = __fmaf_rn(__uint_as_float(gw[4] << 16) * __uint_as_float(wl[4] << 16), __uint_as_float(xw[4] << 16), dq);
            dq = __fmaf_rn(__uint_as_float(gw[4] & 4294901760u) * __uint_as_float(wl[4] & 4294901760u), __uint_as_float(xw[4] & 4294901760u), dq);
            dq = __fmaf_rn(__uint_as_float(gw[5] << 16) * __uint_as_float(wl[5] << 16), __uint_as_float(xw[5] << 16), dq);
            dq = __fmaf_rn(__uint_as_float(gw[5] & 4294901760u) * __uint_as_float(wl[5] & 4294901760u), __uint_as_float(xw[5] & 4294901760u), dq);
            dq = __fmaf_rn(__uint_as_float(gw[6] << 16) * __uint_as_float(wl[6] << 16), __uint_as_float(xw[6] << 16), dq);
            dq = __fmaf_rn(__uint_as_float(gw[6] & 4294901760u) * __uint_as_float(wl[6] & 4294901760u), __uint_as_float(xw[6] & 4294901760u), dq);
            dq = __fmaf_rn(__uint_as_float(gw[7] << 16) * __uint_as_float(wl[7] << 16), __uint_as_float(xw[7] << 16), dq);
            dq = __fmaf_rn(__uint_as_float(gw[7] & 4294901760u) * __uint_as_float(wl[7] & 4294901760u), __uint_as_float(xw[7] & 4294901760u), dq);
            float _warp_reduce_11 = dq;
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1)
                _warp_reduce_11 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_11, offset);
            dot[0] = _warp_reduce_11;
            float rq = rr[0];
            float cq = __fdiv_rn(dot[0], 512.0f);
            float kx = rq * rq * rq * cq;
            float nkx = -kx;
            if (row < T) {
                float out_0[8];
                float g0_0 = __uint_as_float(gw[0] << 16);
                float g1_0 = __uint_as_float(gw[0] & 4294901760u);
                float x0_0 = __uint_as_float(xw[0] << 16);
                float x1_0 = __uint_as_float(xw[0] & 4294901760u);
                float y0_0 = __fmaf_rn(nkx, x0_0, rq * (g0_0 * __uint_as_float(wl[0] << 16)));
                float y1_0 = __fmaf_rn(nkx, x1_0, rq * (g1_0 * __uint_as_float(wl[0] & 4294901760u)));
                out_0[0] = y0_0;
                out_0[1] = y1_0;
                float g0_1 = __uint_as_float(gw[1] << 16);
                float g1_1 = __uint_as_float(gw[1] & 4294901760u);
                float x0_1 = __uint_as_float(xw[1] << 16);
                float x1_1 = __uint_as_float(xw[1] & 4294901760u);
                float y0_1 = __fmaf_rn(nkx, x0_1, rq * (g0_1 * __uint_as_float(wl[1] << 16)));
                float y1_1 = __fmaf_rn(nkx, x1_1, rq * (g1_1 * __uint_as_float(wl[1] & 4294901760u)));
                out_0[2] = y0_1;
                out_0[3] = y1_1;
                float g0_2 = __uint_as_float(gw[2] << 16);
                float g1_2 = __uint_as_float(gw[2] & 4294901760u);
                float x0_2 = __uint_as_float(xw[2] << 16);
                float x1_2 = __uint_as_float(xw[2] & 4294901760u);
                float y0_2 = __fmaf_rn(nkx, x0_2, rq * (g0_2 * __uint_as_float(wl[2] << 16)));
                float y1_2 = __fmaf_rn(nkx, x1_2, rq * (g1_2 * __uint_as_float(wl[2] & 4294901760u)));
                out_0[4] = y0_2;
                out_0[5] = y1_2;
                float g0_3 = __uint_as_float(gw[3] << 16);
                float g1_3 = __uint_as_float(gw[3] & 4294901760u);
                float x0_3 = __uint_as_float(xw[3] << 16);
                float x1_3 = __uint_as_float(xw[3] & 4294901760u);
                float y0_3 = __fmaf_rn(nkx, x0_3, rq * (g0_3 * __uint_as_float(wl[3] << 16)));
                float y1_3 = __fmaf_rn(nkx, x1_3, rq * (g1_3 * __uint_as_float(wl[3] & 4294901760u)));
                out_0[6] = y0_3;
                out_0[7] = y1_3;
                unsigned int packed_0[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_0[_lp*2 + 0], out_0[_lp*2+1 + 0]));
                    packed_0[_lp] = *(uint32_t*)&_bf2;
                }
                reinterpret_cast<int4*>(dx + ((unsigned long long)row * 512ull + (unsigned long long)(tir * 8)))[0] = reinterpret_cast<int4*>(packed_0)[0];
                float out_1[8];
                float g0_4 = __uint_as_float(gw[4] << 16);
                float g1_4 = __uint_as_float(gw[4] & 4294901760u);
                float x0_4 = __uint_as_float(xw[4] << 16);
                float x1_4 = __uint_as_float(xw[4] & 4294901760u);
                float y0_4 = __fmaf_rn(nkx, x0_4, rq * (g0_4 * __uint_as_float(wl[4] << 16)));
                float y1_4 = __fmaf_rn(nkx, x1_4, rq * (g1_4 * __uint_as_float(wl[4] & 4294901760u)));
                out_1[0] = y0_4;
                out_1[1] = y1_4;
                float g0_5 = __uint_as_float(gw[5] << 16);
                float g1_5 = __uint_as_float(gw[5] & 4294901760u);
                float x0_5 = __uint_as_float(xw[5] << 16);
                float x1_5 = __uint_as_float(xw[5] & 4294901760u);
                float y0_5 = __fmaf_rn(nkx, x0_5, rq * (g0_5 * __uint_as_float(wl[5] << 16)));
                float y1_5 = __fmaf_rn(nkx, x1_5, rq * (g1_5 * __uint_as_float(wl[5] & 4294901760u)));
                out_1[2] = y0_5;
                out_1[3] = y1_5;
                float g0_6 = __uint_as_float(gw[6] << 16);
                float g1_6 = __uint_as_float(gw[6] & 4294901760u);
                float x0_6 = __uint_as_float(xw[6] << 16);
                float x1_6 = __uint_as_float(xw[6] & 4294901760u);
                float y0_6 = __fmaf_rn(nkx, x0_6, rq * (g0_6 * __uint_as_float(wl[6] << 16)));
                float y1_6 = __fmaf_rn(nkx, x1_6, rq * (g1_6 * __uint_as_float(wl[6] & 4294901760u)));
                out_1[4] = y0_6;
                out_1[5] = y1_6;
                float g0_7 = __uint_as_float(gw[7] << 16);
                float g1_7 = __uint_as_float(gw[7] & 4294901760u);
                float x0_7 = __uint_as_float(xw[7] << 16);
                float x1_7 = __uint_as_float(xw[7] & 4294901760u);
                float y0_7 = __fmaf_rn(nkx, x0_7, rq * (g0_7 * __uint_as_float(wl[7] << 16)));
                float y1_7 = __fmaf_rn(nkx, x1_7, rq * (g1_7 * __uint_as_float(wl[7] & 4294901760u)));
                out_1[6] = y0_7;
                out_1[7] = y1_7;
                unsigned int packed_1[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_1[_lp*2 + 0], out_1[_lp*2+1 + 0]));
                    packed_1[_lp] = *(uint32_t*)&_bf2;
                }
                reinterpret_cast<int4*>(dx + ((unsigned long long)row * 512ull + (unsigned long long)((32 + tir) * 8)))[0] = reinterpret_cast<int4*>(packed_1)[0];
            }
        } else {
            int vl = tid % 2;
            int rl = tid / 2;
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
            for (int t0 = rl; t0 < T; t0 += 256) {
                if (t0 < T) {
                    unsigned long long g_base_1 = (unsigned long long)t0 * (unsigned long long)g_stride;
                    unsigned long long x_base_1 = (unsigned long long)t0 * (unsigned long long)x_stride;
                    rr_0[0] = r[t0];
                    {
                        const uint4* _vptr_12 = reinterpret_cast<const uint4*>(g + g_base_1 + (unsigned long long)(((it - n_rows) * 2 + vl) * 8));
                        uint4* _vdst_12 = reinterpret_cast<uint4*>(&gw_0[0]);
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                : "=r"(_vdst_12[_blk].x), "=r"(_vdst_12[_blk].y), "=r"(_vdst_12[_blk].z), "=r"(_vdst_12[_blk].w) : "l"((const void*)(_vptr_12 + _blk)), "l"(0x14F0000000000000ULL) : "memory");
                        }
                    }
                    {
                        const uint4* _vptr_13 = reinterpret_cast<const uint4*>(x + x_base_1 + (unsigned long long)(((it - n_rows) * 2 + vl) * 8));
                        uint4* _vdst_13 = reinterpret_cast<uint4*>(&xw_0[0]);
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                : "=r"(_vdst_13[_blk].x), "=r"(_vdst_13[_blk].y), "=r"(_vdst_13[_blk].z), "=r"(_vdst_13[_blk].w) : "l"((const void*)(_vptr_13 + _blk)), "l"(0x14F0000000000000ULL) : "memory");
                        }
                    }
                }
                if (t0 + 32 < T) {
                    unsigned long long g_base_2 = (unsigned long long)(t0 + 32) * (unsigned long long)g_stride;
                    unsigned long long x_base_2 = (unsigned long long)(t0 + 32) * (unsigned long long)x_stride;
                    rr_0[1] = r[t0 + 32];
                    {
                        const uint4* _vptr_14 = reinterpret_cast<const uint4*>(g + g_base_2 + (unsigned long long)(((it - n_rows) * 2 + vl) * 8));
                        uint4* _vdst_14 = reinterpret_cast<uint4*>(&gw_0[4]);
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                : "=r"(_vdst_14[_blk].x), "=r"(_vdst_14[_blk].y), "=r"(_vdst_14[_blk].z), "=r"(_vdst_14[_blk].w) : "l"((const void*)(_vptr_14 + _blk)), "l"(0x14F0000000000000ULL) : "memory");
                        }
                    }
                    {
                        const uint4* _vptr_15 = reinterpret_cast<const uint4*>(x + x_base_2 + (unsigned long long)(((it - n_rows) * 2 + vl) * 8));
                        uint4* _vdst_15 = reinterpret_cast<uint4*>(&xw_0[4]);
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                : "=r"(_vdst_15[_blk].x), "=r"(_vdst_15[_blk].y), "=r"(_vdst_15[_blk].z), "=r"(_vdst_15[_blk].w) : "l"((const void*)(_vptr_15 + _blk)), "l"(0x14F0000000000000ULL) : "memory");
                        }
                    }
                }
                if (t0 + 64 < T) {
                    unsigned long long g_base_3 = (unsigned long long)(t0 + 64) * (unsigned long long)g_stride;
                    unsigned long long x_base_3 = (unsigned long long)(t0 + 64) * (unsigned long long)x_stride;
                    rr_0[2] = r[t0 + 64];
                    {
                        const uint4* _vptr_16 = reinterpret_cast<const uint4*>(g + g_base_3 + (unsigned long long)(((it - n_rows) * 2 + vl) * 8));
                        uint4* _vdst_16 = reinterpret_cast<uint4*>(&gw_0[8]);
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                : "=r"(_vdst_16[_blk].x), "=r"(_vdst_16[_blk].y), "=r"(_vdst_16[_blk].z), "=r"(_vdst_16[_blk].w) : "l"((const void*)(_vptr_16 + _blk)), "l"(0x14F0000000000000ULL) : "memory");
                        }
                    }
                    {
                        const uint4* _vptr_17 = reinterpret_cast<const uint4*>(x + x_base_3 + (unsigned long long)(((it - n_rows) * 2 + vl) * 8));
                        uint4* _vdst_17 = reinterpret_cast<uint4*>(&xw_0[8]);
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                : "=r"(_vdst_17[_blk].x), "=r"(_vdst_17[_blk].y), "=r"(_vdst_17[_blk].z), "=r"(_vdst_17[_blk].w) : "l"((const void*)(_vptr_17 + _blk)), "l"(0x14F0000000000000ULL) : "memory");
                        }
                    }
                }
                if (t0 + 96 < T) {
                    unsigned long long g_base_4 = (unsigned long long)(t0 + 96) * (unsigned long long)g_stride;
                    unsigned long long x_base_4 = (unsigned long long)(t0 + 96) * (unsigned long long)x_stride;
                    rr_0[3] = r[t0 + 96];
                    {
                        const uint4* _vptr_18 = reinterpret_cast<const uint4*>(g + g_base_4 + (unsigned long long)(((it - n_rows) * 2 + vl) * 8));
                        uint4* _vdst_18 = reinterpret_cast<uint4*>(&gw_0[12]);
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                : "=r"(_vdst_18[_blk].x), "=r"(_vdst_18[_blk].y), "=r"(_vdst_18[_blk].z), "=r"(_vdst_18[_blk].w) : "l"((const void*)(_vptr_18 + _blk)), "l"(0x14F0000000000000ULL) : "memory");
                        }
                    }
                    {
                        const uint4* _vptr_19 = reinterpret_cast<const uint4*>(x + x_base_4 + (unsigned long long)(((it - n_rows) * 2 + vl) * 8));
                        uint4* _vdst_19 = reinterpret_cast<uint4*>(&xw_0[12]);
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                : "=r"(_vdst_19[_blk].x), "=r"(_vdst_19[_blk].y), "=r"(_vdst_19[_blk].z), "=r"(_vdst_19[_blk].w) : "l"((const void*)(_vptr_19 + _blk)), "l"(0x14F0000000000000ULL) : "memory");
                        }
                    }
                }
                if (t0 + 128 < T) {
                    unsigned long long g_base_5 = (unsigned long long)(t0 + 128) * (unsigned long long)g_stride;
                    unsigned long long x_base_5 = (unsigned long long)(t0 + 128) * (unsigned long long)x_stride;
                    rr_0[4] = r[t0 + 128];
                    {
                        const uint4* _vptr_20 = reinterpret_cast<const uint4*>(g + g_base_5 + (unsigned long long)(((it - n_rows) * 2 + vl) * 8));
                        uint4* _vdst_20 = reinterpret_cast<uint4*>(&gw_0[16]);
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                : "=r"(_vdst_20[_blk].x), "=r"(_vdst_20[_blk].y), "=r"(_vdst_20[_blk].z), "=r"(_vdst_20[_blk].w) : "l"((const void*)(_vptr_20 + _blk)), "l"(0x14F0000000000000ULL) : "memory");
                        }
                    }
                    {
                        const uint4* _vptr_21 = reinterpret_cast<const uint4*>(x + x_base_5 + (unsigned long long)(((it - n_rows) * 2 + vl) * 8));
                        uint4* _vdst_21 = reinterpret_cast<uint4*>(&xw_0[16]);
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                : "=r"(_vdst_21[_blk].x), "=r"(_vdst_21[_blk].y), "=r"(_vdst_21[_blk].z), "=r"(_vdst_21[_blk].w) : "l"((const void*)(_vptr_21 + _blk)), "l"(0x14F0000000000000ULL) : "memory");
                        }
                    }
                }
                if (t0 + 160 < T) {
                    unsigned long long g_base_6 = (unsigned long long)(t0 + 160) * (unsigned long long)g_stride;
                    unsigned long long x_base_6 = (unsigned long long)(t0 + 160) * (unsigned long long)x_stride;
                    rr_0[5] = r[t0 + 160];
                    {
                        const uint4* _vptr_22 = reinterpret_cast<const uint4*>(g + g_base_6 + (unsigned long long)(((it - n_rows) * 2 + vl) * 8));
                        uint4* _vdst_22 = reinterpret_cast<uint4*>(&gw_0[20]);
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                : "=r"(_vdst_22[_blk].x), "=r"(_vdst_22[_blk].y), "=r"(_vdst_22[_blk].z), "=r"(_vdst_22[_blk].w) : "l"((const void*)(_vptr_22 + _blk)), "l"(0x14F0000000000000ULL) : "memory");
                        }
                    }
                    {
                        const uint4* _vptr_23 = reinterpret_cast<const uint4*>(x + x_base_6 + (unsigned long long)(((it - n_rows) * 2 + vl) * 8));
                        uint4* _vdst_23 = reinterpret_cast<uint4*>(&xw_0[20]);
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                : "=r"(_vdst_23[_blk].x), "=r"(_vdst_23[_blk].y), "=r"(_vdst_23[_blk].z), "=r"(_vdst_23[_blk].w) : "l"((const void*)(_vptr_23 + _blk)), "l"(0x14F0000000000000ULL) : "memory");
                        }
                    }
                }
                if (t0 + 192 < T) {
                    unsigned long long g_base_7 = (unsigned long long)(t0 + 192) * (unsigned long long)g_stride;
                    unsigned long long x_base_7 = (unsigned long long)(t0 + 192) * (unsigned long long)x_stride;
                    rr_0[6] = r[t0 + 192];
                    {
                        const uint4* _vptr_24 = reinterpret_cast<const uint4*>(g + g_base_7 + (unsigned long long)(((it - n_rows) * 2 + vl) * 8));
                        uint4* _vdst_24 = reinterpret_cast<uint4*>(&gw_0[24]);
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                : "=r"(_vdst_24[_blk].x), "=r"(_vdst_24[_blk].y), "=r"(_vdst_24[_blk].z), "=r"(_vdst_24[_blk].w) : "l"((const void*)(_vptr_24 + _blk)), "l"(0x14F0000000000000ULL) : "memory");
                        }
                    }
                    {
                        const uint4* _vptr_25 = reinterpret_cast<const uint4*>(x + x_base_7 + (unsigned long long)(((it - n_rows) * 2 + vl) * 8));
                        uint4* _vdst_25 = reinterpret_cast<uint4*>(&xw_0[24]);
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                : "=r"(_vdst_25[_blk].x), "=r"(_vdst_25[_blk].y), "=r"(_vdst_25[_blk].z), "=r"(_vdst_25[_blk].w) : "l"((const void*)(_vptr_25 + _blk)), "l"(0x14F0000000000000ULL) : "memory");
                        }
                    }
                }
                if (t0 + 224 < T) {
                    unsigned long long g_base_8 = (unsigned long long)(t0 + 224) * (unsigned long long)g_stride;
                    unsigned long long x_base_8 = (unsigned long long)(t0 + 224) * (unsigned long long)x_stride;
                    rr_0[7] = r[t0 + 224];
                    {
                        const uint4* _vptr_26 = reinterpret_cast<const uint4*>(g + g_base_8 + (unsigned long long)(((it - n_rows) * 2 + vl) * 8));
                        uint4* _vdst_26 = reinterpret_cast<uint4*>(&gw_0[28]);
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                : "=r"(_vdst_26[_blk].x), "=r"(_vdst_26[_blk].y), "=r"(_vdst_26[_blk].z), "=r"(_vdst_26[_blk].w) : "l"((const void*)(_vptr_26 + _blk)), "l"(0x14F0000000000000ULL) : "memory");
                        }
                    }
                    {
                        const uint4* _vptr_27 = reinterpret_cast<const uint4*>(x + x_base_8 + (unsigned long long)(((it - n_rows) * 2 + vl) * 8));
                        uint4* _vdst_27 = reinterpret_cast<uint4*>(&xw_0[28]);
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                : "=r"(_vdst_27[_blk].x), "=r"(_vdst_27[_blk].y), "=r"(_vdst_27[_blk].z), "=r"(_vdst_27[_blk].w) : "l"((const void*)(_vptr_27 + _blk)), "l"(0x14F0000000000000ULL) : "memory");
                        }
                    }
                }
                if (t0 < T) {
                    float2 rr2_0 = make_float2(rr_0[0], rr_0[0]);
                    float2 _f32x2_mul_a_28 = make_float2(__uint_as_float(gw_0[0] << 16), __uint_as_float(gw_0[0] & 4294901760u));
                    float2 _f32x2_mul_b_29 = make_float2(__uint_as_float(xw_0[0] << 16), __uint_as_float(xw_0[0] & 4294901760u));
                    float2 gx;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx) : "l"(*(const unsigned long long*)&_f32x2_mul_a_28), "l"(*(const unsigned long long*)&_f32x2_mul_b_29));
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
                    float2 _f32x2_mul_a_30 = make_float2(__uint_as_float(gw_0[1] << 16), __uint_as_float(gw_0[1] & 4294901760u));
                    float2 _f32x2_mul_b_31 = make_float2(__uint_as_float(xw_0[1] << 16), __uint_as_float(xw_0[1] & 4294901760u));
                    float2 gx_0;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx_0) : "l"(*(const unsigned long long*)&_f32x2_mul_a_30), "l"(*(const unsigned long long*)&_f32x2_mul_b_31));
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
                    float2 _f32x2_mul_a_32 = make_float2(__uint_as_float(gw_0[2] << 16), __uint_as_float(gw_0[2] & 4294901760u));
                    float2 _f32x2_mul_b_33 = make_float2(__uint_as_float(xw_0[2] << 16), __uint_as_float(xw_0[2] & 4294901760u));
                    float2 gx_1;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_32), "l"(*(const unsigned long long*)&_f32x2_mul_b_33));
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
                    float2 _f32x2_mul_a_34 = make_float2(__uint_as_float(gw_0[3] << 16), __uint_as_float(gw_0[3] & 4294901760u));
                    float2 _f32x2_mul_b_35 = make_float2(__uint_as_float(xw_0[3] << 16), __uint_as_float(xw_0[3] & 4294901760u));
                    float2 gx_2;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx_2) : "l"(*(const unsigned long long*)&_f32x2_mul_a_34), "l"(*(const unsigned long long*)&_f32x2_mul_b_35));
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
                if (t0 + 32 < T) {
                    float2 rr2_1 = make_float2(rr_0[1], rr_0[1]);
                    float2 _f32x2_mul_a_36 = make_float2(__uint_as_float(gw_0[4] << 16), __uint_as_float(gw_0[4] & 4294901760u));
                    float2 _f32x2_mul_b_37 = make_float2(__uint_as_float(xw_0[4] << 16), __uint_as_float(xw_0[4] & 4294901760u));
                    float2 gx_3;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx_3) : "l"(*(const unsigned long long*)&_f32x2_mul_a_36), "l"(*(const unsigned long long*)&_f32x2_mul_b_37));
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
                    float2 _f32x2_mul_a_38 = make_float2(__uint_as_float(gw_0[5] << 16), __uint_as_float(gw_0[5] & 4294901760u));
                    float2 _f32x2_mul_b_39 = make_float2(__uint_as_float(xw_0[5] << 16), __uint_as_float(xw_0[5] & 4294901760u));
                    float2 gx_3_1;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx_3_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_38), "l"(*(const unsigned long long*)&_f32x2_mul_b_39));
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
                    float2 _f32x2_mul_a_40 = make_float2(__uint_as_float(gw_0[6] << 16), __uint_as_float(gw_0[6] & 4294901760u));
                    float2 _f32x2_mul_b_41 = make_float2(__uint_as_float(xw_0[6] << 16), __uint_as_float(xw_0[6] & 4294901760u));
                    float2 gx_4;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx_4) : "l"(*(const unsigned long long*)&_f32x2_mul_a_40), "l"(*(const unsigned long long*)&_f32x2_mul_b_41));
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
                    float2 _f32x2_mul_a_42 = make_float2(__uint_as_float(gw_0[7] << 16), __uint_as_float(gw_0[7] & 4294901760u));
                    float2 _f32x2_mul_b_43 = make_float2(__uint_as_float(xw_0[7] << 16), __uint_as_float(xw_0[7] & 4294901760u));
                    float2 gx_5;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx_5) : "l"(*(const unsigned long long*)&_f32x2_mul_a_42), "l"(*(const unsigned long long*)&_f32x2_mul_b_43));
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
                if (t0 + 64 < T) {
                    float2 rr2_2 = make_float2(rr_0[2], rr_0[2]);
                    float2 _f32x2_mul_a_44 = make_float2(__uint_as_float(gw_0[8] << 16), __uint_as_float(gw_0[8] & 4294901760u));
                    float2 _f32x2_mul_b_45 = make_float2(__uint_as_float(xw_0[8] << 16), __uint_as_float(xw_0[8] & 4294901760u));
                    float2 gx_6;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx_6) : "l"(*(const unsigned long long*)&_f32x2_mul_a_44), "l"(*(const unsigned long long*)&_f32x2_mul_b_45));
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
                    float2 _f32x2_mul_a_46 = make_float2(__uint_as_float(gw_0[9] << 16), __uint_as_float(gw_0[9] & 4294901760u));
                    float2 _f32x2_mul_b_47 = make_float2(__uint_as_float(xw_0[9] << 16), __uint_as_float(xw_0[9] & 4294901760u));
                    float2 gx_6_1;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx_6_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_46), "l"(*(const unsigned long long*)&_f32x2_mul_b_47));
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
                    float2 _f32x2_mul_a_48 = make_float2(__uint_as_float(gw_0[10] << 16), __uint_as_float(gw_0[10] & 4294901760u));
                    float2 _f32x2_mul_b_49 = make_float2(__uint_as_float(xw_0[10] << 16), __uint_as_float(xw_0[10] & 4294901760u));
                    float2 gx_7;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx_7) : "l"(*(const unsigned long long*)&_f32x2_mul_a_48), "l"(*(const unsigned long long*)&_f32x2_mul_b_49));
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
                    float2 _f32x2_mul_a_50 = make_float2(__uint_as_float(gw_0[11] << 16), __uint_as_float(gw_0[11] & 4294901760u));
                    float2 _f32x2_mul_b_51 = make_float2(__uint_as_float(xw_0[11] << 16), __uint_as_float(xw_0[11] & 4294901760u));
                    float2 gx_8;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx_8) : "l"(*(const unsigned long long*)&_f32x2_mul_a_50), "l"(*(const unsigned long long*)&_f32x2_mul_b_51));
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
                if (t0 + 96 < T) {
                    float2 rr2_3 = make_float2(rr_0[3], rr_0[3]);
                    float2 _f32x2_mul_a_52 = make_float2(__uint_as_float(gw_0[12] << 16), __uint_as_float(gw_0[12] & 4294901760u));
                    float2 _f32x2_mul_b_53 = make_float2(__uint_as_float(xw_0[12] << 16), __uint_as_float(xw_0[12] & 4294901760u));
                    float2 gx_9;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx_9) : "l"(*(const unsigned long long*)&_f32x2_mul_a_52), "l"(*(const unsigned long long*)&_f32x2_mul_b_53));
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
                    float2 _f32x2_mul_a_54 = make_float2(__uint_as_float(gw_0[13] << 16), __uint_as_float(gw_0[13] & 4294901760u));
                    float2 _f32x2_mul_b_55 = make_float2(__uint_as_float(xw_0[13] << 16), __uint_as_float(xw_0[13] & 4294901760u));
                    float2 gx_9_1;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx_9_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_54), "l"(*(const unsigned long long*)&_f32x2_mul_b_55));
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
                    float2 _f32x2_mul_a_56 = make_float2(__uint_as_float(gw_0[14] << 16), __uint_as_float(gw_0[14] & 4294901760u));
                    float2 _f32x2_mul_b_57 = make_float2(__uint_as_float(xw_0[14] << 16), __uint_as_float(xw_0[14] & 4294901760u));
                    float2 gx_10;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx_10) : "l"(*(const unsigned long long*)&_f32x2_mul_a_56), "l"(*(const unsigned long long*)&_f32x2_mul_b_57));
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
                    float2 _f32x2_mul_a_58 = make_float2(__uint_as_float(gw_0[15] << 16), __uint_as_float(gw_0[15] & 4294901760u));
                    float2 _f32x2_mul_b_59 = make_float2(__uint_as_float(xw_0[15] << 16), __uint_as_float(xw_0[15] & 4294901760u));
                    float2 gx_11;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx_11) : "l"(*(const unsigned long long*)&_f32x2_mul_a_58), "l"(*(const unsigned long long*)&_f32x2_mul_b_59));
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
                if (t0 + 128 < T) {
                    float2 rr2_4 = make_float2(rr_0[4], rr_0[4]);
                    float2 _f32x2_mul_a_60 = make_float2(__uint_as_float(gw_0[16] << 16), __uint_as_float(gw_0[16] & 4294901760u));
                    float2 _f32x2_mul_b_61 = make_float2(__uint_as_float(xw_0[16] << 16), __uint_as_float(xw_0[16] & 4294901760u));
                    float2 gx_12;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx_12) : "l"(*(const unsigned long long*)&_f32x2_mul_a_60), "l"(*(const unsigned long long*)&_f32x2_mul_b_61));
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
                    float2 _f32x2_mul_a_62 = make_float2(__uint_as_float(gw_0[17] << 16), __uint_as_float(gw_0[17] & 4294901760u));
                    float2 _f32x2_mul_b_63 = make_float2(__uint_as_float(xw_0[17] << 16), __uint_as_float(xw_0[17] & 4294901760u));
                    float2 gx_12_1;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx_12_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_62), "l"(*(const unsigned long long*)&_f32x2_mul_b_63));
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
                    float2 _f32x2_mul_a_64 = make_float2(__uint_as_float(gw_0[18] << 16), __uint_as_float(gw_0[18] & 4294901760u));
                    float2 _f32x2_mul_b_65 = make_float2(__uint_as_float(xw_0[18] << 16), __uint_as_float(xw_0[18] & 4294901760u));
                    float2 gx_13;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx_13) : "l"(*(const unsigned long long*)&_f32x2_mul_a_64), "l"(*(const unsigned long long*)&_f32x2_mul_b_65));
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
                    float2 _f32x2_mul_a_66 = make_float2(__uint_as_float(gw_0[19] << 16), __uint_as_float(gw_0[19] & 4294901760u));
                    float2 _f32x2_mul_b_67 = make_float2(__uint_as_float(xw_0[19] << 16), __uint_as_float(xw_0[19] & 4294901760u));
                    float2 gx_14;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx_14) : "l"(*(const unsigned long long*)&_f32x2_mul_a_66), "l"(*(const unsigned long long*)&_f32x2_mul_b_67));
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
                if (t0 + 160 < T) {
                    float2 rr2_5 = make_float2(rr_0[5], rr_0[5]);
                    float2 _f32x2_mul_a_68 = make_float2(__uint_as_float(gw_0[20] << 16), __uint_as_float(gw_0[20] & 4294901760u));
                    float2 _f32x2_mul_b_69 = make_float2(__uint_as_float(xw_0[20] << 16), __uint_as_float(xw_0[20] & 4294901760u));
                    float2 gx_15;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx_15) : "l"(*(const unsigned long long*)&_f32x2_mul_a_68), "l"(*(const unsigned long long*)&_f32x2_mul_b_69));
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
                    float2 _f32x2_mul_a_70 = make_float2(__uint_as_float(gw_0[21] << 16), __uint_as_float(gw_0[21] & 4294901760u));
                    float2 _f32x2_mul_b_71 = make_float2(__uint_as_float(xw_0[21] << 16), __uint_as_float(xw_0[21] & 4294901760u));
                    float2 gx_15_1;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx_15_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_70), "l"(*(const unsigned long long*)&_f32x2_mul_b_71));
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
                    float2 _f32x2_mul_a_72 = make_float2(__uint_as_float(gw_0[22] << 16), __uint_as_float(gw_0[22] & 4294901760u));
                    float2 _f32x2_mul_b_73 = make_float2(__uint_as_float(xw_0[22] << 16), __uint_as_float(xw_0[22] & 4294901760u));
                    float2 gx_16;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx_16) : "l"(*(const unsigned long long*)&_f32x2_mul_a_72), "l"(*(const unsigned long long*)&_f32x2_mul_b_73));
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
                    float2 _f32x2_mul_a_74 = make_float2(__uint_as_float(gw_0[23] << 16), __uint_as_float(gw_0[23] & 4294901760u));
                    float2 _f32x2_mul_b_75 = make_float2(__uint_as_float(xw_0[23] << 16), __uint_as_float(xw_0[23] & 4294901760u));
                    float2 gx_17;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx_17) : "l"(*(const unsigned long long*)&_f32x2_mul_a_74), "l"(*(const unsigned long long*)&_f32x2_mul_b_75));
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
                if (t0 + 192 < T) {
                    float2 rr2_6 = make_float2(rr_0[6], rr_0[6]);
                    float2 _f32x2_mul_a_76 = make_float2(__uint_as_float(gw_0[24] << 16), __uint_as_float(gw_0[24] & 4294901760u));
                    float2 _f32x2_mul_b_77 = make_float2(__uint_as_float(xw_0[24] << 16), __uint_as_float(xw_0[24] & 4294901760u));
                    float2 gx_18;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx_18) : "l"(*(const unsigned long long*)&_f32x2_mul_a_76), "l"(*(const unsigned long long*)&_f32x2_mul_b_77));
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
                    float2 _f32x2_mul_a_78 = make_float2(__uint_as_float(gw_0[25] << 16), __uint_as_float(gw_0[25] & 4294901760u));
                    float2 _f32x2_mul_b_79 = make_float2(__uint_as_float(xw_0[25] << 16), __uint_as_float(xw_0[25] & 4294901760u));
                    float2 gx_18_1;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx_18_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_78), "l"(*(const unsigned long long*)&_f32x2_mul_b_79));
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
                    float2 _f32x2_mul_a_80 = make_float2(__uint_as_float(gw_0[26] << 16), __uint_as_float(gw_0[26] & 4294901760u));
                    float2 _f32x2_mul_b_81 = make_float2(__uint_as_float(xw_0[26] << 16), __uint_as_float(xw_0[26] & 4294901760u));
                    float2 gx_19;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx_19) : "l"(*(const unsigned long long*)&_f32x2_mul_a_80), "l"(*(const unsigned long long*)&_f32x2_mul_b_81));
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
                    float2 _f32x2_mul_a_82 = make_float2(__uint_as_float(gw_0[27] << 16), __uint_as_float(gw_0[27] & 4294901760u));
                    float2 _f32x2_mul_b_83 = make_float2(__uint_as_float(xw_0[27] << 16), __uint_as_float(xw_0[27] & 4294901760u));
                    float2 gx_20;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx_20) : "l"(*(const unsigned long long*)&_f32x2_mul_a_82), "l"(*(const unsigned long long*)&_f32x2_mul_b_83));
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
                if (t0 + 224 < T) {
                    float2 rr2_7 = make_float2(rr_0[7], rr_0[7]);
                    float2 _f32x2_mul_a_84 = make_float2(__uint_as_float(gw_0[28] << 16), __uint_as_float(gw_0[28] & 4294901760u));
                    float2 _f32x2_mul_b_85 = make_float2(__uint_as_float(xw_0[28] << 16), __uint_as_float(xw_0[28] & 4294901760u));
                    float2 gx_21;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx_21) : "l"(*(const unsigned long long*)&_f32x2_mul_a_84), "l"(*(const unsigned long long*)&_f32x2_mul_b_85));
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
                    float2 _f32x2_mul_a_86 = make_float2(__uint_as_float(gw_0[29] << 16), __uint_as_float(gw_0[29] & 4294901760u));
                    float2 _f32x2_mul_b_87 = make_float2(__uint_as_float(xw_0[29] << 16), __uint_as_float(xw_0[29] & 4294901760u));
                    float2 gx_21_1;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx_21_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_86), "l"(*(const unsigned long long*)&_f32x2_mul_b_87));
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
                    float2 _f32x2_mul_a_88 = make_float2(__uint_as_float(gw_0[30] << 16), __uint_as_float(gw_0[30] & 4294901760u));
                    float2 _f32x2_mul_b_89 = make_float2(__uint_as_float(xw_0[30] << 16), __uint_as_float(xw_0[30] & 4294901760u));
                    float2 gx_22;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx_22) : "l"(*(const unsigned long long*)&_f32x2_mul_a_88), "l"(*(const unsigned long long*)&_f32x2_mul_b_89));
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
                    float2 _f32x2_mul_a_90 = make_float2(__uint_as_float(gw_0[31] << 16), __uint_as_float(gw_0[31] & 4294901760u));
                    float2 _f32x2_mul_b_91 = make_float2(__uint_as_float(xw_0[31] << 16), __uint_as_float(xw_0[31] & 4294901760u));
                    float2 gx_23;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx_23) : "l"(*(const unsigned long long*)&_f32x2_mul_a_90), "l"(*(const unsigned long long*)&_f32x2_mul_b_91));
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
                sumsf[2 * (rl * 16 + vl * 8)] = acc_0[0];
                sumsf[2 * (rl * 16 + vl * 8) + 1] = comp_0[0];
                sumsf[2 * (rl * 16 + vl * 8 + 1)] = acc_0[1];
                sumsf[2 * (rl * 16 + vl * 8 + 1) + 1] = comp_0[1];
                sumsf[2 * (rl * 16 + vl * 8 + 2)] = acc_0[2];
                sumsf[2 * (rl * 16 + vl * 8 + 2) + 1] = comp_0[2];
                sumsf[2 * (rl * 16 + vl * 8 + 3)] = acc_0[3];
                sumsf[2 * (rl * 16 + vl * 8 + 3) + 1] = comp_0[3];
                sumsf[2 * (rl * 16 + vl * 8 + 4)] = acc_0[4];
                sumsf[2 * (rl * 16 + vl * 8 + 4) + 1] = comp_0[4];
                sumsf[2 * (rl * 16 + vl * 8 + 5)] = acc_0[5];
                sumsf[2 * (rl * 16 + vl * 8 + 5) + 1] = comp_0[5];
                sumsf[2 * (rl * 16 + vl * 8 + 6)] = acc_0[6];
                sumsf[2 * (rl * 16 + vl * 8 + 6) + 1] = comp_0[6];
                sumsf[2 * (rl * 16 + vl * 8 + 7)] = acc_0[7];
                sumsf[2 * (rl * 16 + vl * 8 + 7) + 1] = comp_0[7];
            }
            __syncthreads();
            int cl = tid % 16;
            int s8 = tid / 16 * 8;
            if (s8 < T) {
                float p_24[3];
                p_24[0] = 0.0f;
                p_24[1] = 0.0f;
                p_24[2] = 0.0f;
                float a_24 = p_24[0] + sumsf[2 * (s8 * 16 + cl)];
                float bp = a_24 - p_24[0];
                float ap = a_24 - bp;
                float e0 = p_24[0] - ap + (sumsf[2 * (s8 * 16 + cl)] - bp);
                p_24[0] = a_24;
                float b = p_24[1] + e0;
                float bq = b - p_24[1];
                float aq = b - bq;
                float e1 = p_24[1] - aq + (e0 - bq);
                p_24[1] = b;
                p_24[2] = p_24[2] + e1;
                float a_24_1 = p_24[0] + sumsf[2 * (s8 * 16 + cl) + 1];
                float bp_0 = a_24_1 - p_24[0];
                float ap_0 = a_24_1 - bp_0;
                float e0_0 = p_24[0] - ap_0 + (sumsf[2 * (s8 * 16 + cl) + 1] - bp_0);
                p_24[0] = a_24_1;
                float b_0 = p_24[1] + e0_0;
                float bq_0 = b_0 - p_24[1];
                float aq_0 = b_0 - bq_0;
                float e1_0 = p_24[1] - aq_0 + (e0_0 - bq_0);
                p_24[1] = b_0;
                p_24[2] = p_24[2] + e1_0;
                if (s8 + 1 < T) {
                    float a_25 = p_24[0] + sumsf[2 * ((s8 + 1) * 16 + cl)];
                    float bp_1 = a_25 - p_24[0];
                    float ap_1 = a_25 - bp_1;
                    float e0_1 = p_24[0] - ap_1 + (sumsf[2 * ((s8 + 1) * 16 + cl)] - bp_1);
                    p_24[0] = a_25;
                    float b_1 = p_24[1] + e0_1;
                    float bq_1 = b_1 - p_24[1];
                    float aq_1 = b_1 - bq_1;
                    float e1_1 = p_24[1] - aq_1 + (e0_1 - bq_1);
                    p_24[1] = b_1;
                    p_24[2] = p_24[2] + e1_1;
                    float a_26 = p_24[0] + sumsf[2 * ((s8 + 1) * 16 + cl) + 1];
                    float bp_2 = a_26 - p_24[0];
                    float ap_2 = a_26 - bp_2;
                    float e0_2 = p_24[0] - ap_2 + (sumsf[2 * ((s8 + 1) * 16 + cl) + 1] - bp_2);
                    p_24[0] = a_26;
                    float b_2 = p_24[1] + e0_2;
                    float bq_2 = b_2 - p_24[1];
                    float aq_2 = b_2 - bq_2;
                    float e1_2 = p_24[1] - aq_2 + (e0_2 - bq_2);
                    p_24[1] = b_2;
                    p_24[2] = p_24[2] + e1_2;
                }
                if (s8 + 1 + 1 < T) {
                    float a_27 = p_24[0] + sumsf[2 * ((s8 + 1 + 1) * 16 + cl)];
                    float bp_3 = a_27 - p_24[0];
                    float ap_3 = a_27 - bp_3;
                    float e0_3 = p_24[0] - ap_3 + (sumsf[2 * ((s8 + 1 + 1) * 16 + cl)] - bp_3);
                    p_24[0] = a_27;
                    float b_3 = p_24[1] + e0_3;
                    float bq_3 = b_3 - p_24[1];
                    float aq_3 = b_3 - bq_3;
                    float e1_3 = p_24[1] - aq_3 + (e0_3 - bq_3);
                    p_24[1] = b_3;
                    p_24[2] = p_24[2] + e1_3;
                    float a_28 = p_24[0] + sumsf[2 * ((s8 + 1 + 1) * 16 + cl) + 1];
                    float bp_4 = a_28 - p_24[0];
                    float ap_4 = a_28 - bp_4;
                    float e0_4 = p_24[0] - ap_4 + (sumsf[2 * ((s8 + 1 + 1) * 16 + cl) + 1] - bp_4);
                    p_24[0] = a_28;
                    float b_4 = p_24[1] + e0_4;
                    float bq_4 = b_4 - p_24[1];
                    float aq_4 = b_4 - bq_4;
                    float e1_4 = p_24[1] - aq_4 + (e0_4 - bq_4);
                    p_24[1] = b_4;
                    p_24[2] = p_24[2] + e1_4;
                }
                if (s8 + 2 + 1 < T) {
                    float a_29 = p_24[0] + sumsf[2 * ((s8 + 2 + 1) * 16 + cl)];
                    float bp_5 = a_29 - p_24[0];
                    float ap_5 = a_29 - bp_5;
                    float e0_5 = p_24[0] - ap_5 + (sumsf[2 * ((s8 + 2 + 1) * 16 + cl)] - bp_5);
                    p_24[0] = a_29;
                    float b_5 = p_24[1] + e0_5;
                    float bq_5 = b_5 - p_24[1];
                    float aq_5 = b_5 - bq_5;
                    float e1_5 = p_24[1] - aq_5 + (e0_5 - bq_5);
                    p_24[1] = b_5;
                    p_24[2] = p_24[2] + e1_5;
                    float a_30 = p_24[0] + sumsf[2 * ((s8 + 2 + 1) * 16 + cl) + 1];
                    float bp_6 = a_30 - p_24[0];
                    float ap_6 = a_30 - bp_6;
                    float e0_6 = p_24[0] - ap_6 + (sumsf[2 * ((s8 + 2 + 1) * 16 + cl) + 1] - bp_6);
                    p_24[0] = a_30;
                    float b_6 = p_24[1] + e0_6;
                    float bq_6 = b_6 - p_24[1];
                    float aq_6 = b_6 - bq_6;
                    float e1_6 = p_24[1] - aq_6 + (e0_6 - bq_6);
                    p_24[1] = b_6;
                    p_24[2] = p_24[2] + e1_6;
                }
                if (s8 + 3 + 1 < T) {
                    float a_31 = p_24[0] + sumsf[2 * ((s8 + 3 + 1) * 16 + cl)];
                    float bp_7 = a_31 - p_24[0];
                    float ap_7 = a_31 - bp_7;
                    float e0_7 = p_24[0] - ap_7 + (sumsf[2 * ((s8 + 3 + 1) * 16 + cl)] - bp_7);
                    p_24[0] = a_31;
                    float b_7 = p_24[1] + e0_7;
                    float bq_7 = b_7 - p_24[1];
                    float aq_7 = b_7 - bq_7;
                    float e1_7 = p_24[1] - aq_7 + (e0_7 - bq_7);
                    p_24[1] = b_7;
                    p_24[2] = p_24[2] + e1_7;
                    float a_32 = p_24[0] + sumsf[2 * ((s8 + 3 + 1) * 16 + cl) + 1];
                    float bp_8 = a_32 - p_24[0];
                    float ap_8 = a_32 - bp_8;
                    float e0_8 = p_24[0] - ap_8 + (sumsf[2 * ((s8 + 3 + 1) * 16 + cl) + 1] - bp_8);
                    p_24[0] = a_32;
                    float b_8 = p_24[1] + e0_8;
                    float bq_8 = b_8 - p_24[1];
                    float aq_8 = b_8 - bq_8;
                    float e1_8 = p_24[1] - aq_8 + (e0_8 - bq_8);
                    p_24[1] = b_8;
                    p_24[2] = p_24[2] + e1_8;
                }
                if (s8 + 4 + 1 < T) {
                    float a_33 = p_24[0] + sumsf[2 * ((s8 + 4 + 1) * 16 + cl)];
                    float bp_9 = a_33 - p_24[0];
                    float ap_9 = a_33 - bp_9;
                    float e0_9 = p_24[0] - ap_9 + (sumsf[2 * ((s8 + 4 + 1) * 16 + cl)] - bp_9);
                    p_24[0] = a_33;
                    float b_9 = p_24[1] + e0_9;
                    float bq_9 = b_9 - p_24[1];
                    float aq_9 = b_9 - bq_9;
                    float e1_9 = p_24[1] - aq_9 + (e0_9 - bq_9);
                    p_24[1] = b_9;
                    p_24[2] = p_24[2] + e1_9;
                    float a_34 = p_24[0] + sumsf[2 * ((s8 + 4 + 1) * 16 + cl) + 1];
                    float bp_10 = a_34 - p_24[0];
                    float ap_10 = a_34 - bp_10;
                    float e0_10 = p_24[0] - ap_10 + (sumsf[2 * ((s8 + 4 + 1) * 16 + cl) + 1] - bp_10);
                    p_24[0] = a_34;
                    float b_10 = p_24[1] + e0_10;
                    float bq_10 = b_10 - p_24[1];
                    float aq_10 = b_10 - bq_10;
                    float e1_10 = p_24[1] - aq_10 + (e0_10 - bq_10);
                    p_24[1] = b_10;
                    p_24[2] = p_24[2] + e1_10;
                }
                if (s8 + 5 + 1 < T) {
                    float a_35 = p_24[0] + sumsf[2 * ((s8 + 5 + 1) * 16 + cl)];
                    float bp_11 = a_35 - p_24[0];
                    float ap_11 = a_35 - bp_11;
                    float e0_11 = p_24[0] - ap_11 + (sumsf[2 * ((s8 + 5 + 1) * 16 + cl)] - bp_11);
                    p_24[0] = a_35;
                    float b_11 = p_24[1] + e0_11;
                    float bq_11 = b_11 - p_24[1];
                    float aq_11 = b_11 - bq_11;
                    float e1_11 = p_24[1] - aq_11 + (e0_11 - bq_11);
                    p_24[1] = b_11;
                    p_24[2] = p_24[2] + e1_11;
                    float a_36 = p_24[0] + sumsf[2 * ((s8 + 5 + 1) * 16 + cl) + 1];
                    float bp_12 = a_36 - p_24[0];
                    float ap_12 = a_36 - bp_12;
                    float e0_12 = p_24[0] - ap_12 + (sumsf[2 * ((s8 + 5 + 1) * 16 + cl) + 1] - bp_12);
                    p_24[0] = a_36;
                    float b_12 = p_24[1] + e0_12;
                    float bq_12 = b_12 - p_24[1];
                    float aq_12 = b_12 - bq_12;
                    float e1_12 = p_24[1] - aq_12 + (e0_12 - bq_12);
                    p_24[1] = b_12;
                    p_24[2] = p_24[2] + e1_12;
                }
                if (s8 + 6 + 1 < T) {
                    float a_37 = p_24[0] + sumsf[2 * ((s8 + 6 + 1) * 16 + cl)];
                    float bp_13 = a_37 - p_24[0];
                    float ap_13 = a_37 - bp_13;
                    float e0_13 = p_24[0] - ap_13 + (sumsf[2 * ((s8 + 6 + 1) * 16 + cl)] - bp_13);
                    p_24[0] = a_37;
                    float b_13 = p_24[1] + e0_13;
                    float bq_13 = b_13 - p_24[1];
                    float aq_13 = b_13 - bq_13;
                    float e1_13 = p_24[1] - aq_13 + (e0_13 - bq_13);
                    p_24[1] = b_13;
                    p_24[2] = p_24[2] + e1_13;
                    float a_38 = p_24[0] + sumsf[2 * ((s8 + 6 + 1) * 16 + cl) + 1];
                    float bp_14 = a_38 - p_24[0];
                    float ap_14 = a_38 - bp_14;
                    float e0_14 = p_24[0] - ap_14 + (sumsf[2 * ((s8 + 6 + 1) * 16 + cl) + 1] - bp_14);
                    p_24[0] = a_38;
                    float b_14 = p_24[1] + e0_14;
                    float bq_14 = b_14 - p_24[1];
                    float aq_14 = b_14 - bq_14;
                    float e1_14 = p_24[1] - aq_14 + (e0_14 - bq_14);
                    p_24[1] = b_14;
                    p_24[2] = p_24[2] + e1_14;
                }
                sumsf[2 * (s8 * 16 + cl)] = p_24[0];
                sumsf[2 * (s8 * 16 + cl) + 1] = p_24[1];
                sumsf[2 * ((s8 + 1) * 16 + cl)] = p_24[2];
            }
            __syncthreads();
            if (tid < 16) {
                float q[3];
                q[0] = 0.0f;
                q[1] = 0.0f;
                q[2] = 0.0f;
                float a_39 = q[0] + sumsf[2 * cl];
                float bp_15 = a_39 - q[0];
                float ap_15 = a_39 - bp_15;
                float e0_15 = q[0] - ap_15 + (sumsf[2 * cl] - bp_15);
                q[0] = a_39;
                float b_15 = q[1] + e0_15;
                float bq_15 = b_15 - q[1];
                float aq_15 = b_15 - bq_15;
                float e1_15 = q[1] - aq_15 + (e0_15 - bq_15);
                q[1] = b_15;
                q[2] = q[2] + e1_15;
                float a_39_1 = q[0] + sumsf[2 * cl + 1];
                float bp_15_1 = a_39_1 - q[0];
                float ap_15_1 = a_39_1 - bp_15_1;
                float e0_15_1 = q[0] - ap_15_1 + (sumsf[2 * cl + 1] - bp_15_1);
                q[0] = a_39_1;
                float b_15_1 = q[1] + e0_15_1;
                float bq_15_1 = b_15_1 - q[1];
                float aq_15_1 = b_15_1 - bq_15_1;
                float e1_15_1 = q[1] - aq_15_1 + (e0_15_1 - bq_15_1);
                q[1] = b_15_1;
                q[2] = q[2] + e1_15_1;
                float a_40 = q[0] + sumsf[2 * (16 + cl)];
                float bp_16 = a_40 - q[0];
                float ap_16 = a_40 - bp_16;
                float e0_16 = q[0] - ap_16 + (sumsf[2 * (16 + cl)] - bp_16);
                q[0] = a_40;
                float b_16 = q[1] + e0_16;
                float bq_16 = b_16 - q[1];
                float aq_16 = b_16 - bq_16;
                float e1_16 = q[1] - aq_16 + (e0_16 - bq_16);
                q[1] = b_16;
                q[2] = q[2] + e1_16;
                if (8 < T) {
                    float a_41 = q[0] + sumsf[2 * (128 + cl)];
                    float bp_17 = a_41 - q[0];
                    float ap_17 = a_41 - bp_17;
                    float e0_17 = q[0] - ap_17 + (sumsf[2 * (128 + cl)] - bp_17);
                    q[0] = a_41;
                    float b_17 = q[1] + e0_17;
                    float bq_17 = b_17 - q[1];
                    float aq_17 = b_17 - bq_17;
                    float e1_17 = q[1] - aq_17 + (e0_17 - bq_17);
                    q[1] = b_17;
                    q[2] = q[2] + e1_17;
                    float a_42 = q[0] + sumsf[2 * (128 + cl) + 1];
                    float bp_18 = a_42 - q[0];
                    float ap_18 = a_42 - bp_18;
                    float e0_18 = q[0] - ap_18 + (sumsf[2 * (128 + cl) + 1] - bp_18);
                    q[0] = a_42;
                    float b_18 = q[1] + e0_18;
                    float bq_18 = b_18 - q[1];
                    float aq_18 = b_18 - bq_18;
                    float e1_18 = q[1] - aq_18 + (e0_18 - bq_18);
                    q[1] = b_18;
                    q[2] = q[2] + e1_18;
                    float a_43 = q[0] + sumsf[2 * (144 + cl)];
                    float bp_19 = a_43 - q[0];
                    float ap_19 = a_43 - bp_19;
                    float e0_19 = q[0] - ap_19 + (sumsf[2 * (144 + cl)] - bp_19);
                    q[0] = a_43;
                    float b_19 = q[1] + e0_19;
                    float bq_19 = b_19 - q[1];
                    float aq_19 = b_19 - bq_19;
                    float e1_19 = q[1] - aq_19 + (e0_19 - bq_19);
                    q[1] = b_19;
                    q[2] = q[2] + e1_19;
                }
                if (16 < T) {
                    float a_44 = q[0] + sumsf[2 * (256 + cl)];
                    float bp_20 = a_44 - q[0];
                    float ap_20 = a_44 - bp_20;
                    float e0_20 = q[0] - ap_20 + (sumsf[2 * (256 + cl)] - bp_20);
                    q[0] = a_44;
                    float b_20 = q[1] + e0_20;
                    float bq_20 = b_20 - q[1];
                    float aq_20 = b_20 - bq_20;
                    float e1_20 = q[1] - aq_20 + (e0_20 - bq_20);
                    q[1] = b_20;
                    q[2] = q[2] + e1_20;
                    float a_45 = q[0] + sumsf[2 * (256 + cl) + 1];
                    float bp_21 = a_45 - q[0];
                    float ap_21 = a_45 - bp_21;
                    float e0_21 = q[0] - ap_21 + (sumsf[2 * (256 + cl) + 1] - bp_21);
                    q[0] = a_45;
                    float b_21 = q[1] + e0_21;
                    float bq_21 = b_21 - q[1];
                    float aq_21 = b_21 - bq_21;
                    float e1_21 = q[1] - aq_21 + (e0_21 - bq_21);
                    q[1] = b_21;
                    q[2] = q[2] + e1_21;
                    float a_46 = q[0] + sumsf[2 * (272 + cl)];
                    float bp_22 = a_46 - q[0];
                    float ap_22 = a_46 - bp_22;
                    float e0_22 = q[0] - ap_22 + (sumsf[2 * (272 + cl)] - bp_22);
                    q[0] = a_46;
                    float b_22 = q[1] + e0_22;
                    float bq_22 = b_22 - q[1];
                    float aq_22 = b_22 - bq_22;
                    float e1_22 = q[1] - aq_22 + (e0_22 - bq_22);
                    q[1] = b_22;
                    q[2] = q[2] + e1_22;
                }
                if (24 < T) {
                    float a_47 = q[0] + sumsf[2 * (384 + cl)];
                    float bp_23 = a_47 - q[0];
                    float ap_23 = a_47 - bp_23;
                    float e0_23 = q[0] - ap_23 + (sumsf[2 * (384 + cl)] - bp_23);
                    q[0] = a_47;
                    float b_23 = q[1] + e0_23;
                    float bq_23 = b_23 - q[1];
                    float aq_23 = b_23 - bq_23;
                    float e1_23 = q[1] - aq_23 + (e0_23 - bq_23);
                    q[1] = b_23;
                    q[2] = q[2] + e1_23;
                    float a_48 = q[0] + sumsf[2 * (384 + cl) + 1];
                    float bp_24 = a_48 - q[0];
                    float ap_24 = a_48 - bp_24;
                    float e0_24 = q[0] - ap_24 + (sumsf[2 * (384 + cl) + 1] - bp_24);
                    q[0] = a_48;
                    float b_24 = q[1] + e0_24;
                    float bq_24 = b_24 - q[1];
                    float aq_24 = b_24 - bq_24;
                    float e1_24 = q[1] - aq_24 + (e0_24 - bq_24);
                    q[1] = b_24;
                    q[2] = q[2] + e1_24;
                    float a_49 = q[0] + sumsf[2 * (400 + cl)];
                    float bp_25 = a_49 - q[0];
                    float ap_25 = a_49 - bp_25;
                    float e0_25 = q[0] - ap_25 + (sumsf[2 * (400 + cl)] - bp_25);
                    q[0] = a_49;
                    float b_25 = q[1] + e0_25;
                    float bq_25 = b_25 - q[1];
                    float aq_25 = b_25 - bq_25;
                    float e1_25 = q[1] - aq_25 + (e0_25 - bq_25);
                    q[1] = b_25;
                    q[2] = q[2] + e1_25;
                }
                float o[1];
                float a1 = q[1] + q[2];
                float b1p = a1 - q[1];
                float a1p = a1 - b1p;
                float b1 = q[1] - a1p + (q[2] - b1p);
                float hi = q[0] + a1;
                float b2p = hi - q[0];
                float a2p = hi - b2p;
                float lo = q[0] - a2p + (a1 - b2p);
                float m = lo + b1;
                float b3p = m - lo;
                float a3p = m - b3p;
                float e = lo - a3p + (b1 - b3p);
                unsigned int bits = __as_u32(m);
                if (e != 0.0f) {
                    if ((bits & 1) == 0) {
                        bits = bits + 1 - 2 * ((__as_u32(e) ^ bits) >> 31);
                    }
                }
                o[0] = hi + __uint_as_float(bits);
                *(reinterpret_cast<float*>(dw) + ((it - n_rows) * 16 + cl)) = o[0];
            }
        }
        __syncthreads();
    }
}

} // extern "C"
