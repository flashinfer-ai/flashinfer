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
#define THREADS 256
#define LAUNCH_BOUNDS_THREADS 256

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
kernel_cake_rmsnorm_det_train_18e3c0a5204006b3a56f(__nv_bfloat16* __restrict__ x, __nv_bfloat16* __restrict__ u, float* __restrict__ w, __nv_bfloat16* __restrict__ y, __nv_bfloat16* __restrict__ h_new, float* __restrict__ r, int T, float eps, long long x_stride_p0, long long u_stride_p0)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // === Task calls (declaration order) ===
    int col = lane * 8;
    float wreg[8];
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
            : "=r"(_ldv8_0_0), "=r"(_ldv8_0_1), "=r"(_ldv8_0_2), "=r"(_ldv8_0_3), "=r"(_ldv8_0_4), "=r"(_ldv8_0_5), "=r"(_ldv8_0_6), "=r"(_ldv8_0_7) : "l"((const void*)(w + col + (0))) : "memory");
        wreg[0 + 0] = __uint_as_float(_ldv8_0_0);
        wreg[0 + 1] = __uint_as_float(_ldv8_0_1);
        wreg[0 + 2] = __uint_as_float(_ldv8_0_2);
        wreg[0 + 3] = __uint_as_float(_ldv8_0_3);
        wreg[0 + 4] = __uint_as_float(_ldv8_0_4);
        wreg[0 + 5] = __uint_as_float(_ldv8_0_5);
        wreg[0 + 6] = __uint_as_float(_ldv8_0_6);
        wreg[0 + 7] = __uint_as_float(_ldv8_0_7);
    }
    int nblocks = (T + 31) / 32;
    #pragma unroll 1
    for (int blk = bid; blk < nblocks; blk += num_bids) {
        int row0 = blk * 32 + warp * 4;
        unsigned int xs[16];
        int _min_1 = ((row0) < (T - 1) ? (row0) : (T - 1));
        {
            const uint4* _vptr_2 = reinterpret_cast<const uint4*>(x + ((long long)_min_1 * x_stride_p0 + (long long)col) + 0);
            uint4* _vdst_2 = reinterpret_cast<uint4*>(&xs[0]);
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vdst_2[_blk] = _vptr_2[_blk];
            }
        }
        int _min_3 = ((row0 + 1) < (T - 1) ? (row0 + 1) : (T - 1));
        {
            const uint4* _vptr_4 = reinterpret_cast<const uint4*>(x + ((long long)_min_3 * x_stride_p0 + (long long)col) + 0);
            uint4* _vdst_4 = reinterpret_cast<uint4*>(&xs[4]);
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vdst_4[_blk] = _vptr_4[_blk];
            }
        }
        int _min_5 = ((row0 + 2) < (T - 1) ? (row0 + 2) : (T - 1));
        {
            const uint4* _vptr_6 = reinterpret_cast<const uint4*>(x + ((long long)_min_5 * x_stride_p0 + (long long)col) + 0);
            uint4* _vdst_6 = reinterpret_cast<uint4*>(&xs[8]);
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vdst_6[_blk] = _vptr_6[_blk];
            }
        }
        int _min_7 = ((row0 + 3) < (T - 1) ? (row0 + 3) : (T - 1));
        {
            const uint4* _vptr_8 = reinterpret_cast<const uint4*>(x + ((long long)_min_7 * x_stride_p0 + (long long)col) + 0);
            uint4* _vdst_8 = reinterpret_cast<uint4*>(&xs[12]);
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vdst_8[_blk] = _vptr_8[_blk];
            }
        }
        float tot[4];
        float sum_sq_0 = 0.0f;
        float _dot_9;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_9) : "r"(xs[0]), "r"(xs[0]), "f"(sum_sq_0));
        sum_sq_0 = _dot_9;
        float _dot_10;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_10) : "r"(xs[1]), "r"(xs[1]), "f"(sum_sq_0));
        sum_sq_0 = _dot_10;
        float _dot_11;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_11) : "r"(xs[2]), "r"(xs[2]), "f"(sum_sq_0));
        sum_sq_0 = _dot_11;
        float _dot_12;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_12) : "r"(xs[3]), "r"(xs[3]), "f"(sum_sq_0));
        sum_sq_0 = _dot_12;
        float _warp_reduce_13 = sum_sq_0;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            _warp_reduce_13 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_13, offset);
        tot[0] = _warp_reduce_13;
        float sum_sq_1 = 0.0f;
        float _dot_14;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_14) : "r"(xs[4]), "r"(xs[4]), "f"(sum_sq_1));
        sum_sq_1 = _dot_14;
        float _dot_15;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_15) : "r"(xs[5]), "r"(xs[5]), "f"(sum_sq_1));
        sum_sq_1 = _dot_15;
        float _dot_16;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_16) : "r"(xs[6]), "r"(xs[6]), "f"(sum_sq_1));
        sum_sq_1 = _dot_16;
        float _dot_17;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_17) : "r"(xs[7]), "r"(xs[7]), "f"(sum_sq_1));
        sum_sq_1 = _dot_17;
        float _warp_reduce_18 = sum_sq_1;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            _warp_reduce_18 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_18, offset);
        tot[1] = _warp_reduce_18;
        float sum_sq_2 = 0.0f;
        float _dot_19;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_19) : "r"(xs[8]), "r"(xs[8]), "f"(sum_sq_2));
        sum_sq_2 = _dot_19;
        float _dot_20;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_20) : "r"(xs[9]), "r"(xs[9]), "f"(sum_sq_2));
        sum_sq_2 = _dot_20;
        float _dot_21;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_21) : "r"(xs[10]), "r"(xs[10]), "f"(sum_sq_2));
        sum_sq_2 = _dot_21;
        float _dot_22;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_22) : "r"(xs[11]), "r"(xs[11]), "f"(sum_sq_2));
        sum_sq_2 = _dot_22;
        float _warp_reduce_23 = sum_sq_2;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            _warp_reduce_23 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_23, offset);
        tot[2] = _warp_reduce_23;
        float sum_sq_3 = 0.0f;
        float _dot_24;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_24) : "r"(xs[12]), "r"(xs[12]), "f"(sum_sq_3));
        sum_sq_3 = _dot_24;
        float _dot_25;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_25) : "r"(xs[13]), "r"(xs[13]), "f"(sum_sq_3));
        sum_sq_3 = _dot_25;
        float _dot_26;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_26) : "r"(xs[14]), "r"(xs[14]), "f"(sum_sq_3));
        sum_sq_3 = _dot_26;
        float _dot_27;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_27) : "r"(xs[15]), "r"(xs[15]), "f"(sum_sq_3));
        sum_sq_3 = _dot_27;
        float _warp_reduce_28 = sum_sq_3;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            _warp_reduce_28 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_28, offset);
        tot[3] = _warp_reduce_28;
        float own = tot[0];
        if (lane == 1) {
            own = tot[1];
        }
        if (lane == 2) {
            own = tot[2];
        }
        if (lane == 3) {
            own = tot[3];
        }
        float mean = own * 0.00390625f;
        float _sqrt_29;
        asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(_sqrt_29) : "f"(mean + eps));
        float rstd = __frcp_rn(_sqrt_29);
        float rk_0 = __shfl_sync(0xFFFFFFFF, rstd, 0);
        float out_0[8];
        float2 xv_0 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[0]));
        out_0[0] = xv_0.x * rk_0 * wreg[0];
        out_0[1] = xv_0.y * rk_0 * wreg[1];
        float2 xv_1 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[1]));
        out_0[2] = xv_1.x * rk_0 * wreg[2];
        out_0[3] = xv_1.y * rk_0 * wreg[3];
        float2 xv_2 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[2]));
        out_0[4] = xv_2.x * rk_0 * wreg[4];
        out_0[5] = xv_2.y * rk_0 * wreg[5];
        float2 xv_3 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[3]));
        out_0[6] = xv_3.x * rk_0 * wreg[6];
        out_0[7] = xv_3.y * rk_0 * wreg[7];
        unsigned int packed_0[4];
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_0[_lp*2 + 0], out_0[_lp*2+1 + 0]));
            packed_0[_lp] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_0[_lp*2 + 2], out_0[_lp*2+1 + 2]));
            packed_0[_lp + 1] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_0[_lp*2 + 4], out_0[_lp*2+1 + 4]));
            packed_0[_lp + 2] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_0[_lp*2 + 6], out_0[_lp*2+1 + 6]));
            packed_0[_lp + 3] = *(uint32_t*)&_bf2;
        }
        if (row0 < T) {
            reinterpret_cast<int4*>(y + ((long long)row0 * 256 + (long long)col))[0] = reinterpret_cast<int4*>(packed_0)[0];
        }
        float rk_1 = __shfl_sync(0xFFFFFFFF, rstd, 1);
        float out_1[8];
        float2 xv_4 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[4]));
        out_1[0] = xv_4.x * rk_1 * wreg[0];
        out_1[1] = xv_4.y * rk_1 * wreg[1];
        float2 xv_5 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[5]));
        out_1[2] = xv_5.x * rk_1 * wreg[2];
        out_1[3] = xv_5.y * rk_1 * wreg[3];
        float2 xv_6 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[6]));
        out_1[4] = xv_6.x * rk_1 * wreg[4];
        out_1[5] = xv_6.y * rk_1 * wreg[5];
        float2 xv_7 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[7]));
        out_1[6] = xv_7.x * rk_1 * wreg[6];
        out_1[7] = xv_7.y * rk_1 * wreg[7];
        unsigned int packed_1[4];
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_1[_lp*2 + 0], out_1[_lp*2+1 + 0]));
            packed_1[_lp] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_1[_lp*2 + 2], out_1[_lp*2+1 + 2]));
            packed_1[_lp + 1] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_1[_lp*2 + 4], out_1[_lp*2+1 + 4]));
            packed_1[_lp + 2] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_1[_lp*2 + 6], out_1[_lp*2+1 + 6]));
            packed_1[_lp + 3] = *(uint32_t*)&_bf2;
        }
        if (row0 + 1 < T) {
            reinterpret_cast<int4*>(y + ((long long)(row0 + 1) * 256 + (long long)col))[0] = reinterpret_cast<int4*>(packed_1)[0];
        }
        float rk_2 = __shfl_sync(0xFFFFFFFF, rstd, 2);
        float out_2[8];
        float2 xv_8 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[8]));
        out_2[0] = xv_8.x * rk_2 * wreg[0];
        out_2[1] = xv_8.y * rk_2 * wreg[1];
        float2 xv_9 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[9]));
        out_2[2] = xv_9.x * rk_2 * wreg[2];
        out_2[3] = xv_9.y * rk_2 * wreg[3];
        float2 xv_10 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[10]));
        out_2[4] = xv_10.x * rk_2 * wreg[4];
        out_2[5] = xv_10.y * rk_2 * wreg[5];
        float2 xv_11 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[11]));
        out_2[6] = xv_11.x * rk_2 * wreg[6];
        out_2[7] = xv_11.y * rk_2 * wreg[7];
        unsigned int packed_2[4];
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_2[_lp*2 + 0], out_2[_lp*2+1 + 0]));
            packed_2[_lp] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_2[_lp*2 + 2], out_2[_lp*2+1 + 2]));
            packed_2[_lp + 1] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_2[_lp*2 + 4], out_2[_lp*2+1 + 4]));
            packed_2[_lp + 2] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_2[_lp*2 + 6], out_2[_lp*2+1 + 6]));
            packed_2[_lp + 3] = *(uint32_t*)&_bf2;
        }
        if (row0 + 2 < T) {
            reinterpret_cast<int4*>(y + ((long long)(row0 + 2) * 256 + (long long)col))[0] = reinterpret_cast<int4*>(packed_2)[0];
        }
        float rk_3 = __shfl_sync(0xFFFFFFFF, rstd, 3);
        float out_3[8];
        float2 xv_12 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[12]));
        out_3[0] = xv_12.x * rk_3 * wreg[0];
        out_3[1] = xv_12.y * rk_3 * wreg[1];
        float2 xv_13 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[13]));
        out_3[2] = xv_13.x * rk_3 * wreg[2];
        out_3[3] = xv_13.y * rk_3 * wreg[3];
        float2 xv_14 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[14]));
        out_3[4] = xv_14.x * rk_3 * wreg[4];
        out_3[5] = xv_14.y * rk_3 * wreg[5];
        float2 xv_15 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[15]));
        out_3[6] = xv_15.x * rk_3 * wreg[6];
        out_3[7] = xv_15.y * rk_3 * wreg[7];
        unsigned int packed_3[4];
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_3[_lp*2 + 0], out_3[_lp*2+1 + 0]));
            packed_3[_lp] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_3[_lp*2 + 2], out_3[_lp*2+1 + 2]));
            packed_3[_lp + 1] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_3[_lp*2 + 4], out_3[_lp*2+1 + 4]));
            packed_3[_lp + 2] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_3[_lp*2 + 6], out_3[_lp*2+1 + 6]));
            packed_3[_lp + 3] = *(uint32_t*)&_bf2;
        }
        if (row0 + 3 < T) {
            reinterpret_cast<int4*>(y + ((long long)(row0 + 3) * 256 + (long long)col))[0] = reinterpret_cast<int4*>(packed_3)[0];
        }
        if ((lane & 0) == 0) {
            if (lane < 4) {
                if (row0 + lane < T) {
                    r[row0 + lane] = rstd;
                }
            }
        }
    }
}

} // extern "C"
