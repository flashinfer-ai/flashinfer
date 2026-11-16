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

__device__ __forceinline__ float2 mul_f32x2_noftz(float2 a, float2 b) {
    float2 r;
    asm("mul.rn.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(unsigned long long*)&a), "l"(*(unsigned long long*)&b));
    return r;
}


// ex2_emulation_f32x2 defined in softmax_frag_exp2_cast helper (or standalone)

extern "C" {

__global__ __launch_bounds__(512, LAUNCH_MIN_BLOCKS) void
kernel_cake_rmsnorm_det_train_bc9d86b3b63d513a1aac(__nv_bfloat16* __restrict__ g, __nv_bfloat16* __restrict__ x, __nv_bfloat16* __restrict__ w, float* __restrict__ r, __nv_bfloat16* __restrict__ dx, float* __restrict__ dw, double* __restrict__ partial, unsigned int* __restrict__ counters, __nv_bfloat16* __restrict__ g_h, int T, int n_chunks, int rows_per_chunk, long long g_stride_p0, long long x_stride_p0, long long g_h_stride_p0)
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
    int n_rows = (T + 3) / 4;
    int n_items = n_rows + 32;
    if (T == 1) {
        n_items = n_rows;
        n_items = 0;
        if ((int)bid == 0) {
            unsigned int gq[2];
            unsigned int xq[2];
            unsigned int hq[2];
            unsigned int wq[2];
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
                uint2 _vld_4;
                _vld_4 = *reinterpret_cast<const uint2*>(w + (unsigned long long)(4 * (int)tid));
                uint2* _vdst_4 = reinterpret_cast<uint2*>(&wq[0]);
                *_vdst_4 = _vld_4;
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
            dq = __fmaf_rn(__uint_as_float(gq[0] << 16) * __uint_as_float(wq[0] << 16), __uint_as_float(xq[0] << 16), dq);
            dq = __fmaf_rn(__uint_as_float(gq[0] & 4294901760u) * __uint_as_float(wq[0] & 4294901760u), __uint_as_float(xq[0] & 4294901760u), dq);
            dq = __fmaf_rn(__uint_as_float(gq[1] << 16) * __uint_as_float(wq[1] << 16), __uint_as_float(xq[1] << 16), dq);
            dq = __fmaf_rn(__uint_as_float(gq[1] & 4294901760u) * __uint_as_float(wq[1] & 4294901760u), __uint_as_float(xq[1] & 4294901760u), dq);
            float ws = dq;
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1)
                ws += __shfl_xor_sync(0xFFFFFFFF, ws, offset);
            if (lane == 0) {
                dots[warp] = ws;
            }
            __syncthreads();
            float tree[16];
            tree[0] = dots[0];
            tree[1] = dots[1];
            tree[2] = dots[2];
            tree[3] = dots[3];
            tree[4] = dots[4];
            tree[5] = dots[5];
            tree[6] = dots[6];
            tree[7] = dots[7];
            tree[8] = dots[8];
            tree[9] = dots[9];
            tree[10] = dots[10];
            tree[11] = dots[11];
            tree[12] = dots[12];
            tree[13] = dots[13];
            tree[14] = dots[14];
            tree[15] = dots[15];
            tree[0] = tree[0] + tree[1];
            tree[2] = tree[2] + tree[3];
            tree[4] = tree[4] + tree[5];
            tree[6] = tree[6] + tree[7];
            tree[8] = tree[8] + tree[9];
            tree[10] = tree[10] + tree[11];
            tree[12] = tree[12] + tree[13];
            tree[14] = tree[14] + tree[15];
            tree[0] = tree[0] + tree[2];
            tree[4] = tree[4] + tree[6];
            tree[8] = tree[8] + tree[10];
            tree[12] = tree[12] + tree[14];
            tree[0] = tree[0] + tree[4];
            tree[8] = tree[8] + tree[12];
            tree[0] = tree[0] + tree[8];
            float cq = 0.0f;
            cq = tree[0] * 0.00048828125f;
            float kx = rq * rq * rq * cq;
            float nkx = -kx;
            float out[4];
            float g0_0 = __uint_as_float(gq[0] << 16);
            float g1_0 = __uint_as_float(gq[0] & 4294901760u);
            float x0_0 = __uint_as_float(xq[0] << 16);
            float x1_0 = __uint_as_float(xq[0] & 4294901760u);
            float y0_0 = __fmaf_rn(nkx, x0_0, rq * (g0_0 * __uint_as_float(wq[0] << 16)));
            float y1_0 = __fmaf_rn(nkx, x1_0, rq * (g1_0 * __uint_as_float(wq[0] & 4294901760u)));
            out[0] = y0_0;
            out[1] = y1_0;
            float g0_1 = __uint_as_float(gq[1] << 16);
            float g1_1 = __uint_as_float(gq[1] & 4294901760u);
            float x0_1 = __uint_as_float(xq[1] << 16);
            float x1_1 = __uint_as_float(xq[1] & 4294901760u);
            float y0_1 = __fmaf_rn(nkx, x0_1, rq * (g0_1 * __uint_as_float(wq[1] << 16)));
            float y1_1 = __fmaf_rn(nkx, x1_1, rq * (g1_1 * __uint_as_float(wq[1] & 4294901760u)));
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
