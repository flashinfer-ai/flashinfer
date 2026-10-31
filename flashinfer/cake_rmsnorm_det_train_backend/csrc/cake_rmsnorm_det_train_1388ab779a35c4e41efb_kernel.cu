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
#define THREADS 128
#define LAUNCH_BOUNDS_THREADS 128

#include <math_constants.h>

__device__ __forceinline__ float2 mul_f32x2_noftz(float2 a, float2 b) {
    float2 r;
    asm("mul.rn.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(unsigned long long*)&a), "l"(*(unsigned long long*)&b));
    return r;
}


// ex2_emulation_f32x2 defined in softmax_frag_exp2_cast helper (or standalone)


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
kernel_cake_rmsnorm_det_train_1388ab779a35c4e41efb(__nv_bfloat16* __restrict__ x, __nv_bfloat16* __restrict__ u, __nv_bfloat16* __restrict__ w, __nv_bfloat16* __restrict__ y, __nv_bfloat16* __restrict__ h_new, float* __restrict__ r, int T, float eps, long long x_stride_p0, long long u_stride_p0)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // === Task calls (declaration order) ===
    int col = lane * 8;
    unsigned int wreg[4];
    {
        const uint4* _vptr_0 = reinterpret_cast<const uint4*>(w + col + 0);
        uint4* _vdst_0 = reinterpret_cast<uint4*>(&wreg[0]);
        #pragma unroll
        for (int _blk = 0; _blk < 1; _blk++) {
            asm volatile("ld.global.nc.v4.b32 {%0, %1, %2, %3}, [%4];"
                : "=r"(_vdst_0[_blk].x), "=r"(_vdst_0[_blk].y), "=r"(_vdst_0[_blk].z), "=r"(_vdst_0[_blk].w) : "l"((const void*)(_vptr_0 + _blk)) : "memory");
        }
    }
    int nblocks = (T + 31) / 32;
    #pragma unroll 1
    for (int blk = bid; blk < nblocks; blk += num_bids) {
        int row0 = blk * 32 + warp * 8;
        unsigned int xs[32];
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
        int _min_9 = ((row0 + 4) < (T - 1) ? (row0 + 4) : (T - 1));
        {
            const uint4* _vptr_10 = reinterpret_cast<const uint4*>(x + ((long long)_min_9 * x_stride_p0 + (long long)col) + 0);
            uint4* _vdst_10 = reinterpret_cast<uint4*>(&xs[16]);
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vdst_10[_blk] = _vptr_10[_blk];
            }
        }
        int _min_11 = ((row0 + 5) < (T - 1) ? (row0 + 5) : (T - 1));
        {
            const uint4* _vptr_12 = reinterpret_cast<const uint4*>(x + ((long long)_min_11 * x_stride_p0 + (long long)col) + 0);
            uint4* _vdst_12 = reinterpret_cast<uint4*>(&xs[20]);
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vdst_12[_blk] = _vptr_12[_blk];
            }
        }
        int _min_13 = ((row0 + 6) < (T - 1) ? (row0 + 6) : (T - 1));
        {
            const uint4* _vptr_14 = reinterpret_cast<const uint4*>(x + ((long long)_min_13 * x_stride_p0 + (long long)col) + 0);
            uint4* _vdst_14 = reinterpret_cast<uint4*>(&xs[24]);
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vdst_14[_blk] = _vptr_14[_blk];
            }
        }
        int _min_15 = ((row0 + 7) < (T - 1) ? (row0 + 7) : (T - 1));
        {
            const uint4* _vptr_16 = reinterpret_cast<const uint4*>(x + ((long long)_min_15 * x_stride_p0 + (long long)col) + 0);
            uint4* _vdst_16 = reinterpret_cast<uint4*>(&xs[28]);
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vdst_16[_blk] = _vptr_16[_blk];
            }
        }
        float tot[8];
        float sum_sq_0 = 0.0f;
        float _dot_17;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_17) : "r"(xs[0]), "r"(xs[0]), "f"(sum_sq_0));
        sum_sq_0 = _dot_17;
        float _dot_18;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_18) : "r"(xs[1]), "r"(xs[1]), "f"(sum_sq_0));
        sum_sq_0 = _dot_18;
        float _dot_19;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_19) : "r"(xs[2]), "r"(xs[2]), "f"(sum_sq_0));
        sum_sq_0 = _dot_19;
        float _dot_20;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_20) : "r"(xs[3]), "r"(xs[3]), "f"(sum_sq_0));
        sum_sq_0 = _dot_20;
        tot[0] = sum_sq_0;
        float sum_sq_1 = 0.0f;
        float _dot_21;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_21) : "r"(xs[4]), "r"(xs[4]), "f"(sum_sq_1));
        sum_sq_1 = _dot_21;
        float _dot_22;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_22) : "r"(xs[5]), "r"(xs[5]), "f"(sum_sq_1));
        sum_sq_1 = _dot_22;
        float _dot_23;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_23) : "r"(xs[6]), "r"(xs[6]), "f"(sum_sq_1));
        sum_sq_1 = _dot_23;
        float _dot_24;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_24) : "r"(xs[7]), "r"(xs[7]), "f"(sum_sq_1));
        sum_sq_1 = _dot_24;
        tot[1] = sum_sq_1;
        float sum_sq_2 = 0.0f;
        float _dot_25;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_25) : "r"(xs[8]), "r"(xs[8]), "f"(sum_sq_2));
        sum_sq_2 = _dot_25;
        float _dot_26;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_26) : "r"(xs[9]), "r"(xs[9]), "f"(sum_sq_2));
        sum_sq_2 = _dot_26;
        float _dot_27;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_27) : "r"(xs[10]), "r"(xs[10]), "f"(sum_sq_2));
        sum_sq_2 = _dot_27;
        float _dot_28;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_28) : "r"(xs[11]), "r"(xs[11]), "f"(sum_sq_2));
        sum_sq_2 = _dot_28;
        tot[2] = sum_sq_2;
        float sum_sq_3 = 0.0f;
        float _dot_29;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_29) : "r"(xs[12]), "r"(xs[12]), "f"(sum_sq_3));
        sum_sq_3 = _dot_29;
        float _dot_30;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_30) : "r"(xs[13]), "r"(xs[13]), "f"(sum_sq_3));
        sum_sq_3 = _dot_30;
        float _dot_31;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_31) : "r"(xs[14]), "r"(xs[14]), "f"(sum_sq_3));
        sum_sq_3 = _dot_31;
        float _dot_32;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_32) : "r"(xs[15]), "r"(xs[15]), "f"(sum_sq_3));
        sum_sq_3 = _dot_32;
        tot[3] = sum_sq_3;
        float sum_sq_4 = 0.0f;
        float _dot_33;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_33) : "r"(xs[16]), "r"(xs[16]), "f"(sum_sq_4));
        sum_sq_4 = _dot_33;
        float _dot_34;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_34) : "r"(xs[17]), "r"(xs[17]), "f"(sum_sq_4));
        sum_sq_4 = _dot_34;
        float _dot_35;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_35) : "r"(xs[18]), "r"(xs[18]), "f"(sum_sq_4));
        sum_sq_4 = _dot_35;
        float _dot_36;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_36) : "r"(xs[19]), "r"(xs[19]), "f"(sum_sq_4));
        sum_sq_4 = _dot_36;
        tot[4] = sum_sq_4;
        float sum_sq_5 = 0.0f;
        float _dot_37;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_37) : "r"(xs[20]), "r"(xs[20]), "f"(sum_sq_5));
        sum_sq_5 = _dot_37;
        float _dot_38;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_38) : "r"(xs[21]), "r"(xs[21]), "f"(sum_sq_5));
        sum_sq_5 = _dot_38;
        float _dot_39;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_39) : "r"(xs[22]), "r"(xs[22]), "f"(sum_sq_5));
        sum_sq_5 = _dot_39;
        float _dot_40;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_40) : "r"(xs[23]), "r"(xs[23]), "f"(sum_sq_5));
        sum_sq_5 = _dot_40;
        tot[5] = sum_sq_5;
        float sum_sq_6 = 0.0f;
        float _dot_41;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_41) : "r"(xs[24]), "r"(xs[24]), "f"(sum_sq_6));
        sum_sq_6 = _dot_41;
        float _dot_42;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_42) : "r"(xs[25]), "r"(xs[25]), "f"(sum_sq_6));
        sum_sq_6 = _dot_42;
        float _dot_43;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_43) : "r"(xs[26]), "r"(xs[26]), "f"(sum_sq_6));
        sum_sq_6 = _dot_43;
        float _dot_44;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_44) : "r"(xs[27]), "r"(xs[27]), "f"(sum_sq_6));
        sum_sq_6 = _dot_44;
        tot[6] = sum_sq_6;
        float sum_sq_7 = 0.0f;
        float _dot_45;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_45) : "r"(xs[28]), "r"(xs[28]), "f"(sum_sq_7));
        sum_sq_7 = _dot_45;
        float _dot_46;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_46) : "r"(xs[29]), "r"(xs[29]), "f"(sum_sq_7));
        sum_sq_7 = _dot_46;
        float _dot_47;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_47) : "r"(xs[30]), "r"(xs[30]), "f"(sum_sq_7));
        sum_sq_7 = _dot_47;
        float _dot_48;
        asm volatile(
            "{\n\t"
            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
            "mov.b32 {a_lo, a_hi}, %1;\n\t"
            "mov.b32 {b_lo, b_hi}, %2;\n\t"
            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
            "}\n"
            : "=f"(_dot_48) : "r"(xs[31]), "r"(xs[31]), "f"(sum_sq_7));
        sum_sq_7 = _dot_48;
        tot[7] = sum_sq_7;
        int h16 = lane >> 4 & 1;
        float l16[4];
        float keep16_0 = ((h16 != 0) ? tot[4] : tot[0]);
        float send16_0 = ((h16 != 0) ? tot[0] : tot[4]);
        float _shfl_xor_sync_49 = __shfl_xor_sync(0xFFFFFFFF, send16_0, 16);
        l16[0] = keep16_0 + _shfl_xor_sync_49;
        float keep16_1 = ((h16 != 0) ? tot[5] : tot[1]);
        float send16_1 = ((h16 != 0) ? tot[1] : tot[5]);
        float _shfl_xor_sync_50 = __shfl_xor_sync(0xFFFFFFFF, send16_1, 16);
        l16[1] = keep16_1 + _shfl_xor_sync_50;
        float keep16_2 = ((h16 != 0) ? tot[6] : tot[2]);
        float send16_2 = ((h16 != 0) ? tot[2] : tot[6]);
        float _shfl_xor_sync_51 = __shfl_xor_sync(0xFFFFFFFF, send16_2, 16);
        l16[2] = keep16_2 + _shfl_xor_sync_51;
        float keep16_3 = ((h16 != 0) ? tot[7] : tot[3]);
        float send16_3 = ((h16 != 0) ? tot[3] : tot[7]);
        float _shfl_xor_sync_52 = __shfl_xor_sync(0xFFFFFFFF, send16_3, 16);
        l16[3] = keep16_3 + _shfl_xor_sync_52;
        int h8 = lane >> 3 & 1;
        float l8[2];
        float keep8_0 = ((h8 != 0) ? l16[2] : l16[0]);
        float send8_0 = ((h8 != 0) ? l16[0] : l16[2]);
        float _shfl_xor_sync_53 = __shfl_xor_sync(0xFFFFFFFF, send8_0, 8);
        l8[0] = keep8_0 + _shfl_xor_sync_53;
        float keep8_1 = ((h8 != 0) ? l16[3] : l16[1]);
        float send8_1 = ((h8 != 0) ? l16[1] : l16[3]);
        float _shfl_xor_sync_54 = __shfl_xor_sync(0xFFFFFFFF, send8_1, 8);
        l8[1] = keep8_1 + _shfl_xor_sync_54;
        int h4 = lane >> 2 & 1;
        float keep4 = ((h4 != 0) ? l8[1] : l8[0]);
        float send4 = ((h4 != 0) ? l8[0] : l8[1]);
        float _shfl_xor_sync_55 = __shfl_xor_sync(0xFFFFFFFF, send4, 4);
        float own = keep4 + _shfl_xor_sync_55;
        float _shfl_xor_sync_56 = __shfl_xor_sync(0xFFFFFFFF, own, 2);
        own = own + _shfl_xor_sync_56;
        float _shfl_xor_sync_57 = __shfl_xor_sync(0xFFFFFFFF, own, 1);
        own = own + _shfl_xor_sync_57;
        float mean = own * 0.00390625f;
        float _sqrt_58;
        asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(_sqrt_58) : "f"(mean + eps));
        float rstd = __frcp_rn(_sqrt_58);
        float rk_0 = __shfl_sync(0xFFFFFFFF, rstd, 0);
        float out_0[8];
        float2 rk2_0 = make_float2(rk_0, rk_0);
        float2 xv_0 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[0]));
        float2 _f32x2_mul_a_59 = mul_f32x2_noftz(xv_0, rk2_0);
        float2 _f32x2_mul_b_60 = cake_bf16x2_to_f32x2(__as_bf16x2(wreg[0]));
        float2 o2_0;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&o2_0) : "l"(*(const unsigned long long*)&_f32x2_mul_a_59), "l"(*(const unsigned long long*)&_f32x2_mul_b_60));
        out_0[0] = o2_0.x;
        out_0[1] = o2_0.y;
        float2 xv_1 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[1]));
        float2 _f32x2_mul_a_61 = mul_f32x2_noftz(xv_1, rk2_0);
        float2 _f32x2_mul_b_62 = cake_bf16x2_to_f32x2(__as_bf16x2(wreg[1]));
        float2 o2_1;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&o2_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_61), "l"(*(const unsigned long long*)&_f32x2_mul_b_62));
        out_0[2] = o2_1.x;
        out_0[3] = o2_1.y;
        float2 xv_2 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[2]));
        float2 _f32x2_mul_a_63 = mul_f32x2_noftz(xv_2, rk2_0);
        float2 _f32x2_mul_b_64 = cake_bf16x2_to_f32x2(__as_bf16x2(wreg[2]));
        float2 o2_2;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&o2_2) : "l"(*(const unsigned long long*)&_f32x2_mul_a_63), "l"(*(const unsigned long long*)&_f32x2_mul_b_64));
        out_0[4] = o2_2.x;
        out_0[5] = o2_2.y;
        float2 xv_3 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[3]));
        float2 _f32x2_mul_a_65 = mul_f32x2_noftz(xv_3, rk2_0);
        float2 _f32x2_mul_b_66 = cake_bf16x2_to_f32x2(__as_bf16x2(wreg[3]));
        float2 o2_3;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&o2_3) : "l"(*(const unsigned long long*)&_f32x2_mul_a_65), "l"(*(const unsigned long long*)&_f32x2_mul_b_66));
        out_0[6] = o2_3.x;
        out_0[7] = o2_3.y;
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
        float rk_1 = __shfl_sync(0xFFFFFFFF, rstd, 4);
        float out_1[8];
        float2 rk2_1 = make_float2(rk_1, rk_1);
        float2 xv_4 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[4]));
        float2 _f32x2_mul_a_67 = mul_f32x2_noftz(xv_4, rk2_1);
        float2 _f32x2_mul_b_68 = cake_bf16x2_to_f32x2(__as_bf16x2(wreg[0]));
        float2 o2_4;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&o2_4) : "l"(*(const unsigned long long*)&_f32x2_mul_a_67), "l"(*(const unsigned long long*)&_f32x2_mul_b_68));
        out_1[0] = o2_4.x;
        out_1[1] = o2_4.y;
        float2 xv_5 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[5]));
        float2 _f32x2_mul_a_69 = mul_f32x2_noftz(xv_5, rk2_1);
        float2 _f32x2_mul_b_70 = cake_bf16x2_to_f32x2(__as_bf16x2(wreg[1]));
        float2 o2_5;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&o2_5) : "l"(*(const unsigned long long*)&_f32x2_mul_a_69), "l"(*(const unsigned long long*)&_f32x2_mul_b_70));
        out_1[2] = o2_5.x;
        out_1[3] = o2_5.y;
        float2 xv_6 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[6]));
        float2 _f32x2_mul_a_71 = mul_f32x2_noftz(xv_6, rk2_1);
        float2 _f32x2_mul_b_72 = cake_bf16x2_to_f32x2(__as_bf16x2(wreg[2]));
        float2 o2_6;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&o2_6) : "l"(*(const unsigned long long*)&_f32x2_mul_a_71), "l"(*(const unsigned long long*)&_f32x2_mul_b_72));
        out_1[4] = o2_6.x;
        out_1[5] = o2_6.y;
        float2 xv_7 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[7]));
        float2 _f32x2_mul_a_73 = mul_f32x2_noftz(xv_7, rk2_1);
        float2 _f32x2_mul_b_74 = cake_bf16x2_to_f32x2(__as_bf16x2(wreg[3]));
        float2 o2_7;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&o2_7) : "l"(*(const unsigned long long*)&_f32x2_mul_a_73), "l"(*(const unsigned long long*)&_f32x2_mul_b_74));
        out_1[6] = o2_7.x;
        out_1[7] = o2_7.y;
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
        float rk_2 = __shfl_sync(0xFFFFFFFF, rstd, 8);
        float out_2[8];
        float2 rk2_2 = make_float2(rk_2, rk_2);
        float2 xv_8 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[8]));
        float2 _f32x2_mul_a_75 = mul_f32x2_noftz(xv_8, rk2_2);
        float2 _f32x2_mul_b_76 = cake_bf16x2_to_f32x2(__as_bf16x2(wreg[0]));
        float2 o2_8;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&o2_8) : "l"(*(const unsigned long long*)&_f32x2_mul_a_75), "l"(*(const unsigned long long*)&_f32x2_mul_b_76));
        out_2[0] = o2_8.x;
        out_2[1] = o2_8.y;
        float2 xv_9 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[9]));
        float2 _f32x2_mul_a_77 = mul_f32x2_noftz(xv_9, rk2_2);
        float2 _f32x2_mul_b_78 = cake_bf16x2_to_f32x2(__as_bf16x2(wreg[1]));
        float2 o2_9;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&o2_9) : "l"(*(const unsigned long long*)&_f32x2_mul_a_77), "l"(*(const unsigned long long*)&_f32x2_mul_b_78));
        out_2[2] = o2_9.x;
        out_2[3] = o2_9.y;
        float2 xv_10 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[10]));
        float2 _f32x2_mul_a_79 = mul_f32x2_noftz(xv_10, rk2_2);
        float2 _f32x2_mul_b_80 = cake_bf16x2_to_f32x2(__as_bf16x2(wreg[2]));
        float2 o2_10;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&o2_10) : "l"(*(const unsigned long long*)&_f32x2_mul_a_79), "l"(*(const unsigned long long*)&_f32x2_mul_b_80));
        out_2[4] = o2_10.x;
        out_2[5] = o2_10.y;
        float2 xv_11 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[11]));
        float2 _f32x2_mul_a_81 = mul_f32x2_noftz(xv_11, rk2_2);
        float2 _f32x2_mul_b_82 = cake_bf16x2_to_f32x2(__as_bf16x2(wreg[3]));
        float2 o2_11;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&o2_11) : "l"(*(const unsigned long long*)&_f32x2_mul_a_81), "l"(*(const unsigned long long*)&_f32x2_mul_b_82));
        out_2[6] = o2_11.x;
        out_2[7] = o2_11.y;
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
        float rk_3 = __shfl_sync(0xFFFFFFFF, rstd, 12);
        float out_3[8];
        float2 rk2_3 = make_float2(rk_3, rk_3);
        float2 xv_12 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[12]));
        float2 _f32x2_mul_a_83 = mul_f32x2_noftz(xv_12, rk2_3);
        float2 _f32x2_mul_b_84 = cake_bf16x2_to_f32x2(__as_bf16x2(wreg[0]));
        float2 o2_12;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&o2_12) : "l"(*(const unsigned long long*)&_f32x2_mul_a_83), "l"(*(const unsigned long long*)&_f32x2_mul_b_84));
        out_3[0] = o2_12.x;
        out_3[1] = o2_12.y;
        float2 xv_13 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[13]));
        float2 _f32x2_mul_a_85 = mul_f32x2_noftz(xv_13, rk2_3);
        float2 _f32x2_mul_b_86 = cake_bf16x2_to_f32x2(__as_bf16x2(wreg[1]));
        float2 o2_13;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&o2_13) : "l"(*(const unsigned long long*)&_f32x2_mul_a_85), "l"(*(const unsigned long long*)&_f32x2_mul_b_86));
        out_3[2] = o2_13.x;
        out_3[3] = o2_13.y;
        float2 xv_14 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[14]));
        float2 _f32x2_mul_a_87 = mul_f32x2_noftz(xv_14, rk2_3);
        float2 _f32x2_mul_b_88 = cake_bf16x2_to_f32x2(__as_bf16x2(wreg[2]));
        float2 o2_14;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&o2_14) : "l"(*(const unsigned long long*)&_f32x2_mul_a_87), "l"(*(const unsigned long long*)&_f32x2_mul_b_88));
        out_3[4] = o2_14.x;
        out_3[5] = o2_14.y;
        float2 xv_15 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[15]));
        float2 _f32x2_mul_a_89 = mul_f32x2_noftz(xv_15, rk2_3);
        float2 _f32x2_mul_b_90 = cake_bf16x2_to_f32x2(__as_bf16x2(wreg[3]));
        float2 o2_15;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&o2_15) : "l"(*(const unsigned long long*)&_f32x2_mul_a_89), "l"(*(const unsigned long long*)&_f32x2_mul_b_90));
        out_3[6] = o2_15.x;
        out_3[7] = o2_15.y;
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
        float rk_4 = __shfl_sync(0xFFFFFFFF, rstd, 16);
        float out_4[8];
        float2 rk2_4 = make_float2(rk_4, rk_4);
        float2 xv_16 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[16]));
        float2 _f32x2_mul_a_91 = mul_f32x2_noftz(xv_16, rk2_4);
        float2 _f32x2_mul_b_92 = cake_bf16x2_to_f32x2(__as_bf16x2(wreg[0]));
        float2 o2_16;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&o2_16) : "l"(*(const unsigned long long*)&_f32x2_mul_a_91), "l"(*(const unsigned long long*)&_f32x2_mul_b_92));
        out_4[0] = o2_16.x;
        out_4[1] = o2_16.y;
        float2 xv_17 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[17]));
        float2 _f32x2_mul_a_93 = mul_f32x2_noftz(xv_17, rk2_4);
        float2 _f32x2_mul_b_94 = cake_bf16x2_to_f32x2(__as_bf16x2(wreg[1]));
        float2 o2_17;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&o2_17) : "l"(*(const unsigned long long*)&_f32x2_mul_a_93), "l"(*(const unsigned long long*)&_f32x2_mul_b_94));
        out_4[2] = o2_17.x;
        out_4[3] = o2_17.y;
        float2 xv_18 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[18]));
        float2 _f32x2_mul_a_95 = mul_f32x2_noftz(xv_18, rk2_4);
        float2 _f32x2_mul_b_96 = cake_bf16x2_to_f32x2(__as_bf16x2(wreg[2]));
        float2 o2_18;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&o2_18) : "l"(*(const unsigned long long*)&_f32x2_mul_a_95), "l"(*(const unsigned long long*)&_f32x2_mul_b_96));
        out_4[4] = o2_18.x;
        out_4[5] = o2_18.y;
        float2 xv_19 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[19]));
        float2 _f32x2_mul_a_97 = mul_f32x2_noftz(xv_19, rk2_4);
        float2 _f32x2_mul_b_98 = cake_bf16x2_to_f32x2(__as_bf16x2(wreg[3]));
        float2 o2_19;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&o2_19) : "l"(*(const unsigned long long*)&_f32x2_mul_a_97), "l"(*(const unsigned long long*)&_f32x2_mul_b_98));
        out_4[6] = o2_19.x;
        out_4[7] = o2_19.y;
        unsigned int packed_4[4];
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_4[_lp*2 + 0], out_4[_lp*2+1 + 0]));
            packed_4[_lp] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_4[_lp*2 + 2], out_4[_lp*2+1 + 2]));
            packed_4[_lp + 1] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_4[_lp*2 + 4], out_4[_lp*2+1 + 4]));
            packed_4[_lp + 2] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_4[_lp*2 + 6], out_4[_lp*2+1 + 6]));
            packed_4[_lp + 3] = *(uint32_t*)&_bf2;
        }
        if (row0 + 4 < T) {
            reinterpret_cast<int4*>(y + ((long long)(row0 + 4) * 256 + (long long)col))[0] = reinterpret_cast<int4*>(packed_4)[0];
        }
        float rk_5 = __shfl_sync(0xFFFFFFFF, rstd, 20);
        float out_5[8];
        float2 rk2_5 = make_float2(rk_5, rk_5);
        float2 xv_20 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[20]));
        float2 _f32x2_mul_a_99 = mul_f32x2_noftz(xv_20, rk2_5);
        float2 _f32x2_mul_b_100 = cake_bf16x2_to_f32x2(__as_bf16x2(wreg[0]));
        float2 o2_20;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&o2_20) : "l"(*(const unsigned long long*)&_f32x2_mul_a_99), "l"(*(const unsigned long long*)&_f32x2_mul_b_100));
        out_5[0] = o2_20.x;
        out_5[1] = o2_20.y;
        float2 xv_21 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[21]));
        float2 _f32x2_mul_a_101 = mul_f32x2_noftz(xv_21, rk2_5);
        float2 _f32x2_mul_b_102 = cake_bf16x2_to_f32x2(__as_bf16x2(wreg[1]));
        float2 o2_21;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&o2_21) : "l"(*(const unsigned long long*)&_f32x2_mul_a_101), "l"(*(const unsigned long long*)&_f32x2_mul_b_102));
        out_5[2] = o2_21.x;
        out_5[3] = o2_21.y;
        float2 xv_22 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[22]));
        float2 _f32x2_mul_a_103 = mul_f32x2_noftz(xv_22, rk2_5);
        float2 _f32x2_mul_b_104 = cake_bf16x2_to_f32x2(__as_bf16x2(wreg[2]));
        float2 o2_22;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&o2_22) : "l"(*(const unsigned long long*)&_f32x2_mul_a_103), "l"(*(const unsigned long long*)&_f32x2_mul_b_104));
        out_5[4] = o2_22.x;
        out_5[5] = o2_22.y;
        float2 xv_23 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[23]));
        float2 _f32x2_mul_a_105 = mul_f32x2_noftz(xv_23, rk2_5);
        float2 _f32x2_mul_b_106 = cake_bf16x2_to_f32x2(__as_bf16x2(wreg[3]));
        float2 o2_23;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&o2_23) : "l"(*(const unsigned long long*)&_f32x2_mul_a_105), "l"(*(const unsigned long long*)&_f32x2_mul_b_106));
        out_5[6] = o2_23.x;
        out_5[7] = o2_23.y;
        unsigned int packed_5[4];
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_5[_lp*2 + 0], out_5[_lp*2+1 + 0]));
            packed_5[_lp] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_5[_lp*2 + 2], out_5[_lp*2+1 + 2]));
            packed_5[_lp + 1] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_5[_lp*2 + 4], out_5[_lp*2+1 + 4]));
            packed_5[_lp + 2] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_5[_lp*2 + 6], out_5[_lp*2+1 + 6]));
            packed_5[_lp + 3] = *(uint32_t*)&_bf2;
        }
        if (row0 + 5 < T) {
            reinterpret_cast<int4*>(y + ((long long)(row0 + 5) * 256 + (long long)col))[0] = reinterpret_cast<int4*>(packed_5)[0];
        }
        float rk_6 = __shfl_sync(0xFFFFFFFF, rstd, 24);
        float out_6[8];
        float2 rk2_6 = make_float2(rk_6, rk_6);
        float2 xv_24 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[24]));
        float2 _f32x2_mul_a_107 = mul_f32x2_noftz(xv_24, rk2_6);
        float2 _f32x2_mul_b_108 = cake_bf16x2_to_f32x2(__as_bf16x2(wreg[0]));
        float2 o2_24;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&o2_24) : "l"(*(const unsigned long long*)&_f32x2_mul_a_107), "l"(*(const unsigned long long*)&_f32x2_mul_b_108));
        out_6[0] = o2_24.x;
        out_6[1] = o2_24.y;
        float2 xv_25 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[25]));
        float2 _f32x2_mul_a_109 = mul_f32x2_noftz(xv_25, rk2_6);
        float2 _f32x2_mul_b_110 = cake_bf16x2_to_f32x2(__as_bf16x2(wreg[1]));
        float2 o2_25;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&o2_25) : "l"(*(const unsigned long long*)&_f32x2_mul_a_109), "l"(*(const unsigned long long*)&_f32x2_mul_b_110));
        out_6[2] = o2_25.x;
        out_6[3] = o2_25.y;
        float2 xv_26 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[26]));
        float2 _f32x2_mul_a_111 = mul_f32x2_noftz(xv_26, rk2_6);
        float2 _f32x2_mul_b_112 = cake_bf16x2_to_f32x2(__as_bf16x2(wreg[2]));
        float2 o2_26;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&o2_26) : "l"(*(const unsigned long long*)&_f32x2_mul_a_111), "l"(*(const unsigned long long*)&_f32x2_mul_b_112));
        out_6[4] = o2_26.x;
        out_6[5] = o2_26.y;
        float2 xv_27 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[27]));
        float2 _f32x2_mul_a_113 = mul_f32x2_noftz(xv_27, rk2_6);
        float2 _f32x2_mul_b_114 = cake_bf16x2_to_f32x2(__as_bf16x2(wreg[3]));
        float2 o2_27;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&o2_27) : "l"(*(const unsigned long long*)&_f32x2_mul_a_113), "l"(*(const unsigned long long*)&_f32x2_mul_b_114));
        out_6[6] = o2_27.x;
        out_6[7] = o2_27.y;
        unsigned int packed_6[4];
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_6[_lp*2 + 0], out_6[_lp*2+1 + 0]));
            packed_6[_lp] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_6[_lp*2 + 2], out_6[_lp*2+1 + 2]));
            packed_6[_lp + 1] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_6[_lp*2 + 4], out_6[_lp*2+1 + 4]));
            packed_6[_lp + 2] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_6[_lp*2 + 6], out_6[_lp*2+1 + 6]));
            packed_6[_lp + 3] = *(uint32_t*)&_bf2;
        }
        if (row0 + 6 < T) {
            reinterpret_cast<int4*>(y + ((long long)(row0 + 6) * 256 + (long long)col))[0] = reinterpret_cast<int4*>(packed_6)[0];
        }
        float rk_7 = __shfl_sync(0xFFFFFFFF, rstd, 28);
        float out_7[8];
        float2 rk2_7 = make_float2(rk_7, rk_7);
        float2 xv_28 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[28]));
        float2 _f32x2_mul_a_115 = mul_f32x2_noftz(xv_28, rk2_7);
        float2 _f32x2_mul_b_116 = cake_bf16x2_to_f32x2(__as_bf16x2(wreg[0]));
        float2 o2_28;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&o2_28) : "l"(*(const unsigned long long*)&_f32x2_mul_a_115), "l"(*(const unsigned long long*)&_f32x2_mul_b_116));
        out_7[0] = o2_28.x;
        out_7[1] = o2_28.y;
        float2 xv_29 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[29]));
        float2 _f32x2_mul_a_117 = mul_f32x2_noftz(xv_29, rk2_7);
        float2 _f32x2_mul_b_118 = cake_bf16x2_to_f32x2(__as_bf16x2(wreg[1]));
        float2 o2_29;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&o2_29) : "l"(*(const unsigned long long*)&_f32x2_mul_a_117), "l"(*(const unsigned long long*)&_f32x2_mul_b_118));
        out_7[2] = o2_29.x;
        out_7[3] = o2_29.y;
        float2 xv_30 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[30]));
        float2 _f32x2_mul_a_119 = mul_f32x2_noftz(xv_30, rk2_7);
        float2 _f32x2_mul_b_120 = cake_bf16x2_to_f32x2(__as_bf16x2(wreg[2]));
        float2 o2_30;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&o2_30) : "l"(*(const unsigned long long*)&_f32x2_mul_a_119), "l"(*(const unsigned long long*)&_f32x2_mul_b_120));
        out_7[4] = o2_30.x;
        out_7[5] = o2_30.y;
        float2 xv_31 = cake_bf16x2_to_f32x2(__as_bf16x2(xs[31]));
        float2 _f32x2_mul_a_121 = mul_f32x2_noftz(xv_31, rk2_7);
        float2 _f32x2_mul_b_122 = cake_bf16x2_to_f32x2(__as_bf16x2(wreg[3]));
        float2 o2_31;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&o2_31) : "l"(*(const unsigned long long*)&_f32x2_mul_a_121), "l"(*(const unsigned long long*)&_f32x2_mul_b_122));
        out_7[6] = o2_31.x;
        out_7[7] = o2_31.y;
        unsigned int packed_7[4];
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_7[_lp*2 + 0], out_7[_lp*2+1 + 0]));
            packed_7[_lp] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_7[_lp*2 + 2], out_7[_lp*2+1 + 2]));
            packed_7[_lp + 1] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_7[_lp*2 + 4], out_7[_lp*2+1 + 4]));
            packed_7[_lp + 2] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_7[_lp*2 + 6], out_7[_lp*2+1 + 6]));
            packed_7[_lp + 3] = *(uint32_t*)&_bf2;
        }
        if (row0 + 7 < T) {
            reinterpret_cast<int4*>(y + ((long long)(row0 + 7) * 256 + (long long)col))[0] = reinterpret_cast<int4*>(packed_7)[0];
        }
        if ((lane & 3) == 0) {
            if (lane >> 2 < 8) {
                if (row0 + (lane >> 2) < T) {
                    r[row0 + (lane >> 2)] = rstd;
                }
            }
        }
    }
}

} // extern "C"
