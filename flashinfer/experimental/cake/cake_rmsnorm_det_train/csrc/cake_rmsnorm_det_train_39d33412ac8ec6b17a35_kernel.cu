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

extern "C" {

__global__ __launch_bounds__(256, LAUNCH_MIN_BLOCKS) void
kernel_cake_rmsnorm_det_train_39d33412ac8ec6b17a35(__nv_bfloat16* __restrict__ g, __nv_bfloat16* __restrict__ x, __nv_bfloat16* __restrict__ w, float* __restrict__ r, __nv_bfloat16* __restrict__ dx, float* __restrict__ dw, double* __restrict__ partial, unsigned int* __restrict__ counters, __nv_bfloat16* __restrict__ g_h, int T, int n_chunks, int rows_per_chunk, long long g_stride_p0, long long x_stride_p0, long long g_h_stride_p0)
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
    double* sums = reinterpret_cast<double*>(smem_raw + 96);
    const int sums_addr = smem + 96;
    float* dwbuf = reinterpret_cast<float*>(smem_raw + 96);
    const int dwbuf_addr = smem + 96;

    // === Task calls (declaration order) ===
    asm volatile("griddepcontrol.wait;" ::: "memory");
    unsigned int nxt = 0;
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    long long g_stride = g_stride_p0;
    long long x_stride = x_stride_p0;
    long long gh_stride = g_h_stride_p0;
    int rg = warp / 4;
    int tir = tid - rg * 128;
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
        const uint4* _vptr_1 = reinterpret_cast<const uint4*>(w + (unsigned long long)((128 + tir) * 8));
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
    for (int k = 0; k < n_rows + 64; k++) {
        int it = (int)bid + k * (int)gridDim.x;
        if (it >= n_rows + 64) {
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
                const uint4* _vptr_7 = reinterpret_cast<const uint4*>(g + g_base + (unsigned long long)((128 + tir) * 8));
                uint4* _vdst_7 = reinterpret_cast<uint4*>(&gw[4]);
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                        : "=r"(_vdst_7[_blk].x), "=r"(_vdst_7[_blk].y), "=r"(_vdst_7[_blk].z), "=r"(_vdst_7[_blk].w) : "l"((const void*)(_vptr_7 + _blk)), "l"(0x14F0000000000000ULL) : "memory");
                }
            }
            {
                const uint4* _vptr_8 = reinterpret_cast<const uint4*>(x + x_base + (unsigned long long)((128 + tir) * 8));
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
                const uint4* _vptr_10 = reinterpret_cast<const uint4*>(w + (unsigned long long)((128 + tir) * 8));
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
            if (lane == 0) {
                dots[(k & 1) * 8 + warp] = dot[0];
            }
            __syncthreads();
            float tot = 0.0f;
            tot += dots[(k & 1) * 8 + rg * 4];
            tot += dots[(k & 1) * 8 + rg * 4 + 1];
            tot += dots[(k & 1) * 8 + rg * 4 + 2];
            tot += dots[(k & 1) * 8 + rg * 4 + 3];
            dot[0] = tot;
            float rq = rr[0];
            float cq = __fdiv_rn(dot[0], 2048.0f);
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
                reinterpret_cast<int4*>(dx + ((unsigned long long)row * 2048ull + (unsigned long long)(tir * 8)))[0] = reinterpret_cast<int4*>(packed_0)[0];
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
                reinterpret_cast<int4*>(dx + ((unsigned long long)row * 2048ull + (unsigned long long)((128 + tir) * 8)))[0] = reinterpret_cast<int4*>(packed_1)[0];
            }
        } else {
            int vl = tid % 4;
            int rl = tid / 4;
            unsigned int gw_0[8];
            unsigned int xw_0[8];
            unsigned int hw_0[1];
            float rr_0[2];
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
            for (int t0 = rl; t0 < T; t0 += 128) {
                if (t0 < T) {
                    unsigned long long g_base_1 = (unsigned long long)t0 * (unsigned long long)g_stride;
                    unsigned long long x_base_1 = (unsigned long long)t0 * (unsigned long long)x_stride;
                    rr_0[0] = r[t0];
                    {
                        const uint4* _vptr_12 = reinterpret_cast<const uint4*>(g + g_base_1 + (unsigned long long)(((it - n_rows) * 4 + vl) * 8));
                        uint4* _vdst_12 = reinterpret_cast<uint4*>(&gw_0[0]);
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                : "=r"(_vdst_12[_blk].x), "=r"(_vdst_12[_blk].y), "=r"(_vdst_12[_blk].z), "=r"(_vdst_12[_blk].w) : "l"((const void*)(_vptr_12 + _blk)), "l"(0x14F0000000000000ULL) : "memory");
                        }
                    }
                    {
                        const uint4* _vptr_13 = reinterpret_cast<const uint4*>(x + x_base_1 + (unsigned long long)(((it - n_rows) * 4 + vl) * 8));
                        uint4* _vdst_13 = reinterpret_cast<uint4*>(&xw_0[0]);
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                : "=r"(_vdst_13[_blk].x), "=r"(_vdst_13[_blk].y), "=r"(_vdst_13[_blk].z), "=r"(_vdst_13[_blk].w) : "l"((const void*)(_vptr_13 + _blk)), "l"(0x14F0000000000000ULL) : "memory");
                        }
                    }
                }
                if (t0 + 64 < T) {
                    unsigned long long g_base_2 = (unsigned long long)(t0 + 64) * (unsigned long long)g_stride;
                    unsigned long long x_base_2 = (unsigned long long)(t0 + 64) * (unsigned long long)x_stride;
                    rr_0[1] = r[t0 + 64];
                    {
                        const uint4* _vptr_14 = reinterpret_cast<const uint4*>(g + g_base_2 + (unsigned long long)(((it - n_rows) * 4 + vl) * 8));
                        uint4* _vdst_14 = reinterpret_cast<uint4*>(&gw_0[4]);
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                : "=r"(_vdst_14[_blk].x), "=r"(_vdst_14[_blk].y), "=r"(_vdst_14[_blk].z), "=r"(_vdst_14[_blk].w) : "l"((const void*)(_vptr_14 + _blk)), "l"(0x14F0000000000000ULL) : "memory");
                        }
                    }
                    {
                        const uint4* _vptr_15 = reinterpret_cast<const uint4*>(x + x_base_2 + (unsigned long long)(((it - n_rows) * 4 + vl) * 8));
                        uint4* _vdst_15 = reinterpret_cast<uint4*>(&xw_0[4]);
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                : "=r"(_vdst_15[_blk].x), "=r"(_vdst_15[_blk].y), "=r"(_vdst_15[_blk].z), "=r"(_vdst_15[_blk].w) : "l"((const void*)(_vptr_15 + _blk)), "l"(0x14F0000000000000ULL) : "memory");
                        }
                    }
                }
                if (t0 < T) {
                    float2 rr2_0 = make_float2(rr_0[0], rr_0[0]);
                    float2 _f32x2_mul_a_16 = make_float2(__uint_as_float(gw_0[0] << 16), __uint_as_float(gw_0[0] & 4294901760u));
                    float2 _f32x2_mul_b_17 = make_float2(__uint_as_float(xw_0[0] << 16), __uint_as_float(xw_0[0] & 4294901760u));
                    float2 gx;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx) : "l"(*(const unsigned long long*)&_f32x2_mul_a_16), "l"(*(const unsigned long long*)&_f32x2_mul_b_17));
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
                    float2 _f32x2_mul_a_18 = make_float2(__uint_as_float(gw_0[1] << 16), __uint_as_float(gw_0[1] & 4294901760u));
                    float2 _f32x2_mul_b_19 = make_float2(__uint_as_float(xw_0[1] << 16), __uint_as_float(xw_0[1] & 4294901760u));
                    float2 gx_0;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx_0) : "l"(*(const unsigned long long*)&_f32x2_mul_a_18), "l"(*(const unsigned long long*)&_f32x2_mul_b_19));
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
                    float2 _f32x2_mul_a_20 = make_float2(__uint_as_float(gw_0[2] << 16), __uint_as_float(gw_0[2] & 4294901760u));
                    float2 _f32x2_mul_b_21 = make_float2(__uint_as_float(xw_0[2] << 16), __uint_as_float(xw_0[2] & 4294901760u));
                    float2 gx_1;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_20), "l"(*(const unsigned long long*)&_f32x2_mul_b_21));
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
                    float2 _f32x2_mul_a_22 = make_float2(__uint_as_float(gw_0[3] << 16), __uint_as_float(gw_0[3] & 4294901760u));
                    float2 _f32x2_mul_b_23 = make_float2(__uint_as_float(xw_0[3] << 16), __uint_as_float(xw_0[3] & 4294901760u));
                    float2 gx_2;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx_2) : "l"(*(const unsigned long long*)&_f32x2_mul_a_22), "l"(*(const unsigned long long*)&_f32x2_mul_b_23));
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
                if (t0 + 64 < T) {
                    float2 rr2_1 = make_float2(rr_0[1], rr_0[1]);
                    float2 _f32x2_mul_a_24 = make_float2(__uint_as_float(gw_0[4] << 16), __uint_as_float(gw_0[4] & 4294901760u));
                    float2 _f32x2_mul_b_25 = make_float2(__uint_as_float(xw_0[4] << 16), __uint_as_float(xw_0[4] & 4294901760u));
                    float2 gx_3;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx_3) : "l"(*(const unsigned long long*)&_f32x2_mul_a_24), "l"(*(const unsigned long long*)&_f32x2_mul_b_25));
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
                    float2 _f32x2_mul_a_26 = make_float2(__uint_as_float(gw_0[5] << 16), __uint_as_float(gw_0[5] & 4294901760u));
                    float2 _f32x2_mul_b_27 = make_float2(__uint_as_float(xw_0[5] << 16), __uint_as_float(xw_0[5] & 4294901760u));
                    float2 gx_3_1;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx_3_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_26), "l"(*(const unsigned long long*)&_f32x2_mul_b_27));
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
                    float2 _f32x2_mul_a_28 = make_float2(__uint_as_float(gw_0[6] << 16), __uint_as_float(gw_0[6] & 4294901760u));
                    float2 _f32x2_mul_b_29 = make_float2(__uint_as_float(xw_0[6] << 16), __uint_as_float(xw_0[6] & 4294901760u));
                    float2 gx_4;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx_4) : "l"(*(const unsigned long long*)&_f32x2_mul_a_28), "l"(*(const unsigned long long*)&_f32x2_mul_b_29));
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
                    float2 _f32x2_mul_a_30 = make_float2(__uint_as_float(gw_0[7] << 16), __uint_as_float(gw_0[7] & 4294901760u));
                    float2 _f32x2_mul_b_31 = make_float2(__uint_as_float(xw_0[7] << 16), __uint_as_float(xw_0[7] & 4294901760u));
                    float2 gx_5;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx_5) : "l"(*(const unsigned long long*)&_f32x2_mul_a_30), "l"(*(const unsigned long long*)&_f32x2_mul_b_31));
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
            }
            if (rl < T) {
                sums[rl * 32 + vl * 8] = (double)acc_0[0] + (double)comp_0[0];
                sums[rl * 32 + vl * 8 + 1] = (double)acc_0[1] + (double)comp_0[1];
                sums[rl * 32 + vl * 8 + 2] = (double)acc_0[2] + (double)comp_0[2];
                sums[rl * 32 + vl * 8 + 3] = (double)acc_0[3] + (double)comp_0[3];
                sums[rl * 32 + vl * 8 + 4] = (double)acc_0[4] + (double)comp_0[4];
                sums[rl * 32 + vl * 8 + 5] = (double)acc_0[5] + (double)comp_0[5];
                sums[rl * 32 + vl * 8 + 6] = (double)acc_0[6] + (double)comp_0[6];
                sums[rl * 32 + vl * 8 + 7] = (double)acc_0[7] + (double)comp_0[7];
            }
            __syncthreads();
            int cl = tid % 32;
            int s8 = tid / 32 * 8;
            if (s8 < T) {
                double part = sums[s8 * 32 + cl];
                if (s8 + 1 < T) {
                    part = part + sums[(s8 + 1) * 32 + cl];
                }
                if (s8 + 1 + 1 < T) {
                    part = part + sums[(s8 + 1 + 1) * 32 + cl];
                }
                if (s8 + 2 + 1 < T) {
                    part = part + sums[(s8 + 2 + 1) * 32 + cl];
                }
                if (s8 + 3 + 1 < T) {
                    part = part + sums[(s8 + 3 + 1) * 32 + cl];
                }
                if (s8 + 4 + 1 < T) {
                    part = part + sums[(s8 + 4 + 1) * 32 + cl];
                }
                if (s8 + 5 + 1 < T) {
                    part = part + sums[(s8 + 5 + 1) * 32 + cl];
                }
                if (s8 + 6 + 1 < T) {
                    part = part + sums[(s8 + 6 + 1) * 32 + cl];
                }
                sums[s8 * 32 + cl] = part;
            }
            __syncthreads();
            if (tid < 32) {
                double tot_1 = sums[cl];
                if (8 < T) {
                    tot_1 = tot_1 + sums[256 + cl];
                }
                if (16 < T) {
                    tot_1 = tot_1 + sums[512 + cl];
                }
                if (24 < T) {
                    tot_1 = tot_1 + sums[768 + cl];
                }
                if (32 < T) {
                    tot_1 = tot_1 + sums[1024 + cl];
                }
                if (40 < T) {
                    tot_1 = tot_1 + sums[1280 + cl];
                }
                if (48 < T) {
                    tot_1 = tot_1 + sums[1536 + cl];
                }
                if (56 < T) {
                    tot_1 = tot_1 + sums[1792 + cl];
                }
                *(reinterpret_cast<float*>(dw) + ((it - n_rows) * 32 + cl)) = (float)tot_1;
            }
        }
        __syncthreads();
    }
}

} // extern "C"
