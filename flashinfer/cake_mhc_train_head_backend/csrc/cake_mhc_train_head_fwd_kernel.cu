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
struct CakeFastDivmod { int32_t divisor; uint32_t multiplier; uint32_t shift_right; };
static_assert(sizeof(CakeFastDivmod) == 12, "CakeFastDivmod CUDA ABI must be 12 bytes");
static_assert(alignof(CakeFastDivmod) == 4, "CakeFastDivmod CUDA ABI must be 4-byte aligned");

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
#define THREADS 256

#include <math_constants.h>


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

__device__ __forceinline__ float mul_f32_rn_noftz(float a, float b) {
    float r;
    asm("mul.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}

extern "C" {

__global__ __launch_bounds__(256) void
kernel_cake_mhc_train_head_fwd(__nv_bfloat16* __restrict__ x, __nv_bfloat16* __restrict__ y, int T, CakeFastDivmod hv)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // === Task calls (declaration order) ===
    asm volatile("griddepcontrol.wait;" ::: "memory");
    unsigned int vectors = (unsigned int)T * ((uint32_t)(hv.divisor));
    unsigned int v = (unsigned int)bid * 256 + (unsigned int)tid;
    if (v < vectors) {
        uint32_t token = (hv.divisor != 1) ? (__umulhi((uint32_t)(v), hv.multiplier) >> hv.shift_right) : (uint32_t)(v);
        uint32_t col = (uint32_t)(v) - token * (uint32_t)(hv.divisor);
        long long row = (long long)((uint32_t)(hv.divisor)) * 8;
        long long xbase = (long long)token * (row * 4) + (long long)col * 8;
        unsigned int words[16];
        {
            uint32_t _uv4_0_0;
            uint32_t _uv4_0_1;
            uint32_t _uv4_0_2;
            uint32_t _uv4_0_3;
            asm volatile("ld.global.L1::no_allocate.v4.b32 {%0, %1, %2, %3}, [%4];"
                : "=r"(_uv4_0_0), "=r"(_uv4_0_1), "=r"(_uv4_0_2), "=r"(_uv4_0_3) : "l"((const void*)(reinterpret_cast<unsigned int*>(x + (xbase + row * 0)) + (0))) : "memory");
            words[0 + 0] = _uv4_0_0;
            words[0 + 1] = _uv4_0_1;
            words[0 + 2] = _uv4_0_2;
            words[0 + 3] = _uv4_0_3;
        }
        {
            uint32_t _uv4_1_0;
            uint32_t _uv4_1_1;
            uint32_t _uv4_1_2;
            uint32_t _uv4_1_3;
            asm volatile("ld.global.L1::no_allocate.v4.b32 {%0, %1, %2, %3}, [%4];"
                : "=r"(_uv4_1_0), "=r"(_uv4_1_1), "=r"(_uv4_1_2), "=r"(_uv4_1_3) : "l"((const void*)(reinterpret_cast<unsigned int*>(x + (xbase + row * 1)) + (0))) : "memory");
            words[4 + 0] = _uv4_1_0;
            words[4 + 1] = _uv4_1_1;
            words[4 + 2] = _uv4_1_2;
            words[4 + 3] = _uv4_1_3;
        }
        {
            uint32_t _uv4_2_0;
            uint32_t _uv4_2_1;
            uint32_t _uv4_2_2;
            uint32_t _uv4_2_3;
            asm volatile("ld.global.L1::no_allocate.v4.b32 {%0, %1, %2, %3}, [%4];"
                : "=r"(_uv4_2_0), "=r"(_uv4_2_1), "=r"(_uv4_2_2), "=r"(_uv4_2_3) : "l"((const void*)(reinterpret_cast<unsigned int*>(x + (xbase + row * 2)) + (0))) : "memory");
            words[8 + 0] = _uv4_2_0;
            words[8 + 1] = _uv4_2_1;
            words[8 + 2] = _uv4_2_2;
            words[8 + 3] = _uv4_2_3;
        }
        {
            uint32_t _uv4_3_0;
            uint32_t _uv4_3_1;
            uint32_t _uv4_3_2;
            uint32_t _uv4_3_3;
            asm volatile("ld.global.L1::no_allocate.v4.b32 {%0, %1, %2, %3}, [%4];"
                : "=r"(_uv4_3_0), "=r"(_uv4_3_1), "=r"(_uv4_3_2), "=r"(_uv4_3_3) : "l"((const void*)(reinterpret_cast<unsigned int*>(x + (xbase + row * 3)) + (0))) : "memory");
            words[12 + 0] = _uv4_3_0;
            words[12 + 1] = _uv4_3_1;
            words[12 + 2] = _uv4_3_2;
            words[12 + 3] = _uv4_3_3;
        }
        unsigned int out[4];
        float acc0_0[2];
        asm(
            "{\n\t"
            ".reg .b16 lo, hi;\n\t"
            "mov.b32 {lo, hi}, %2;\n\t"
            "add.rn.f32.bf16 %0, lo, %3;\n\t"
            "add.rn.f32.bf16 %1, hi, %4;\n\t"
            "}\n"
            : "=&f"(acc0_0[0]), "=&f"(acc0_0[1]) : "r"(words[0]), "f"(0.0f), "f"(0.0f));
        float acc1_0[2];
        asm(
            "{\n\t"
            ".reg .b16 lo, hi;\n\t"
            "mov.b32 {lo, hi}, %2;\n\t"
            "add.rn.f32.bf16 %0, lo, %3;\n\t"
            "add.rn.f32.bf16 %1, hi, %4;\n\t"
            "}\n"
            : "=&f"(acc1_0[0]), "=&f"(acc1_0[1]) : "r"(words[4]), "f"(acc0_0[0]), "f"(acc0_0[1]));
        float acc2_0[2];
        asm(
            "{\n\t"
            ".reg .b16 lo, hi;\n\t"
            "mov.b32 {lo, hi}, %2;\n\t"
            "add.rn.f32.bf16 %0, lo, %3;\n\t"
            "add.rn.f32.bf16 %1, hi, %4;\n\t"
            "}\n"
            : "=&f"(acc2_0[0]), "=&f"(acc2_0[1]) : "r"(words[8]), "f"(acc1_0[0]), "f"(acc1_0[1]));
        float acc3_0[2];
        asm(
            "{\n\t"
            ".reg .b16 lo, hi;\n\t"
            "mov.b32 {lo, hi}, %2;\n\t"
            "add.rn.f32.bf16 %0, lo, %3;\n\t"
            "add.rn.f32.bf16 %1, hi, %4;\n\t"
            "}\n"
            : "=&f"(acc3_0[0]), "=&f"(acc3_0[1]) : "r"(words[12]), "f"(acc2_0[0]), "f"(acc2_0[1]));
        unsigned int mean_0 = __as_u32(__float22bfloat162_rn(make_float2(mul_f32_rn_noftz(acc3_0[0], 0.25f), mul_f32_rn_noftz(acc3_0[1], 0.25f))));
        out[0] = mean_0;
        float acc0_1[2];
        asm(
            "{\n\t"
            ".reg .b16 lo, hi;\n\t"
            "mov.b32 {lo, hi}, %2;\n\t"
            "add.rn.f32.bf16 %0, lo, %3;\n\t"
            "add.rn.f32.bf16 %1, hi, %4;\n\t"
            "}\n"
            : "=&f"(acc0_1[0]), "=&f"(acc0_1[1]) : "r"(words[1]), "f"(0.0f), "f"(0.0f));
        float acc1_1[2];
        asm(
            "{\n\t"
            ".reg .b16 lo, hi;\n\t"
            "mov.b32 {lo, hi}, %2;\n\t"
            "add.rn.f32.bf16 %0, lo, %3;\n\t"
            "add.rn.f32.bf16 %1, hi, %4;\n\t"
            "}\n"
            : "=&f"(acc1_1[0]), "=&f"(acc1_1[1]) : "r"(words[5]), "f"(acc0_1[0]), "f"(acc0_1[1]));
        float acc2_1[2];
        asm(
            "{\n\t"
            ".reg .b16 lo, hi;\n\t"
            "mov.b32 {lo, hi}, %2;\n\t"
            "add.rn.f32.bf16 %0, lo, %3;\n\t"
            "add.rn.f32.bf16 %1, hi, %4;\n\t"
            "}\n"
            : "=&f"(acc2_1[0]), "=&f"(acc2_1[1]) : "r"(words[9]), "f"(acc1_1[0]), "f"(acc1_1[1]));
        float acc3_1[2];
        asm(
            "{\n\t"
            ".reg .b16 lo, hi;\n\t"
            "mov.b32 {lo, hi}, %2;\n\t"
            "add.rn.f32.bf16 %0, lo, %3;\n\t"
            "add.rn.f32.bf16 %1, hi, %4;\n\t"
            "}\n"
            : "=&f"(acc3_1[0]), "=&f"(acc3_1[1]) : "r"(words[13]), "f"(acc2_1[0]), "f"(acc2_1[1]));
        unsigned int mean_1 = __as_u32(__float22bfloat162_rn(make_float2(mul_f32_rn_noftz(acc3_1[0], 0.25f), mul_f32_rn_noftz(acc3_1[1], 0.25f))));
        out[1] = mean_1;
        float acc0_2[2];
        asm(
            "{\n\t"
            ".reg .b16 lo, hi;\n\t"
            "mov.b32 {lo, hi}, %2;\n\t"
            "add.rn.f32.bf16 %0, lo, %3;\n\t"
            "add.rn.f32.bf16 %1, hi, %4;\n\t"
            "}\n"
            : "=&f"(acc0_2[0]), "=&f"(acc0_2[1]) : "r"(words[2]), "f"(0.0f), "f"(0.0f));
        float acc1_2[2];
        asm(
            "{\n\t"
            ".reg .b16 lo, hi;\n\t"
            "mov.b32 {lo, hi}, %2;\n\t"
            "add.rn.f32.bf16 %0, lo, %3;\n\t"
            "add.rn.f32.bf16 %1, hi, %4;\n\t"
            "}\n"
            : "=&f"(acc1_2[0]), "=&f"(acc1_2[1]) : "r"(words[6]), "f"(acc0_2[0]), "f"(acc0_2[1]));
        float acc2_2[2];
        asm(
            "{\n\t"
            ".reg .b16 lo, hi;\n\t"
            "mov.b32 {lo, hi}, %2;\n\t"
            "add.rn.f32.bf16 %0, lo, %3;\n\t"
            "add.rn.f32.bf16 %1, hi, %4;\n\t"
            "}\n"
            : "=&f"(acc2_2[0]), "=&f"(acc2_2[1]) : "r"(words[10]), "f"(acc1_2[0]), "f"(acc1_2[1]));
        float acc3_2[2];
        asm(
            "{\n\t"
            ".reg .b16 lo, hi;\n\t"
            "mov.b32 {lo, hi}, %2;\n\t"
            "add.rn.f32.bf16 %0, lo, %3;\n\t"
            "add.rn.f32.bf16 %1, hi, %4;\n\t"
            "}\n"
            : "=&f"(acc3_2[0]), "=&f"(acc3_2[1]) : "r"(words[14]), "f"(acc2_2[0]), "f"(acc2_2[1]));
        unsigned int mean_2 = __as_u32(__float22bfloat162_rn(make_float2(mul_f32_rn_noftz(acc3_2[0], 0.25f), mul_f32_rn_noftz(acc3_2[1], 0.25f))));
        out[2] = mean_2;
        float acc0_3[2];
        asm(
            "{\n\t"
            ".reg .b16 lo, hi;\n\t"
            "mov.b32 {lo, hi}, %2;\n\t"
            "add.rn.f32.bf16 %0, lo, %3;\n\t"
            "add.rn.f32.bf16 %1, hi, %4;\n\t"
            "}\n"
            : "=&f"(acc0_3[0]), "=&f"(acc0_3[1]) : "r"(words[3]), "f"(0.0f), "f"(0.0f));
        float acc1_3[2];
        asm(
            "{\n\t"
            ".reg .b16 lo, hi;\n\t"
            "mov.b32 {lo, hi}, %2;\n\t"
            "add.rn.f32.bf16 %0, lo, %3;\n\t"
            "add.rn.f32.bf16 %1, hi, %4;\n\t"
            "}\n"
            : "=&f"(acc1_3[0]), "=&f"(acc1_3[1]) : "r"(words[7]), "f"(acc0_3[0]), "f"(acc0_3[1]));
        float acc2_3[2];
        asm(
            "{\n\t"
            ".reg .b16 lo, hi;\n\t"
            "mov.b32 {lo, hi}, %2;\n\t"
            "add.rn.f32.bf16 %0, lo, %3;\n\t"
            "add.rn.f32.bf16 %1, hi, %4;\n\t"
            "}\n"
            : "=&f"(acc2_3[0]), "=&f"(acc2_3[1]) : "r"(words[11]), "f"(acc1_3[0]), "f"(acc1_3[1]));
        float acc3_3[2];
        asm(
            "{\n\t"
            ".reg .b16 lo, hi;\n\t"
            "mov.b32 {lo, hi}, %2;\n\t"
            "add.rn.f32.bf16 %0, lo, %3;\n\t"
            "add.rn.f32.bf16 %1, hi, %4;\n\t"
            "}\n"
            : "=&f"(acc3_3[0]), "=&f"(acc3_3[1]) : "r"(words[15]), "f"(acc2_3[0]), "f"(acc2_3[1]));
        unsigned int mean_3 = __as_u32(__float22bfloat162_rn(make_float2(mul_f32_rn_noftz(acc3_3[0], 0.25f), mul_f32_rn_noftz(acc3_3[1], 0.25f))));
        out[3] = mean_3;
        reinterpret_cast<int4*>(reinterpret_cast<unsigned int*>(y + ((long long)v * 8)))[0] = reinterpret_cast<int4*>(out)[0];
    }
}

} // extern "C"
