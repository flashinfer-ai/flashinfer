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
kernel_cake_mhc_det_train_head_fwd__lw64_t256_h4096(__nv_bfloat16* __restrict__ x, __nv_bfloat16* __restrict__ y, int T, CakeFastDivmod hv)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // === Task calls (declaration order) ===
    asm volatile("griddepcontrol.wait;" ::: "memory");
    unsigned int token = (unsigned int)bid / 4;
    unsigned int col = ((unsigned int)bid - token * 4) * 256 + (unsigned int)tid;
    if (token < (unsigned int)T) {
        long long row = 4096ll;
        long long xbase = (long long)token * (row * 4ll) + (long long)col * 4ll;
        unsigned int words[8];
        {
            const uint2* _ivptr_0 = reinterpret_cast<const uint2*>(reinterpret_cast<unsigned int*>(x + xbase) + 0);
            uint2 _ivld_0;
            asm volatile("ld.global.L1::no_allocate.v2.b32 {%0, %1}, [%2];"
                : "=r"(_ivld_0.x), "=r"(_ivld_0.y) : "l"((const void*)(_ivptr_0)) : "memory");
            words[0 + 0] = _ivld_0.x;
            words[0 + 1] = _ivld_0.y;
        }
        {
            const uint2* _ivptr_1 = reinterpret_cast<const uint2*>(reinterpret_cast<unsigned int*>(x + (xbase + row)) + 0);
            uint2 _ivld_1;
            asm volatile("ld.global.L1::no_allocate.v2.b32 {%0, %1}, [%2];"
                : "=r"(_ivld_1.x), "=r"(_ivld_1.y) : "l"((const void*)(_ivptr_1)) : "memory");
            words[2 + 0] = _ivld_1.x;
            words[2 + 1] = _ivld_1.y;
        }
        {
            const uint2* _ivptr_2 = reinterpret_cast<const uint2*>(reinterpret_cast<unsigned int*>(x + (xbase + row * 2ll)) + 0);
            uint2 _ivld_2;
            asm volatile("ld.global.L1::no_allocate.v2.b32 {%0, %1}, [%2];"
                : "=r"(_ivld_2.x), "=r"(_ivld_2.y) : "l"((const void*)(_ivptr_2)) : "memory");
            words[4 + 0] = _ivld_2.x;
            words[4 + 1] = _ivld_2.y;
        }
        {
            const uint2* _ivptr_3 = reinterpret_cast<const uint2*>(reinterpret_cast<unsigned int*>(x + (xbase + row * 3ll)) + 0);
            uint2 _ivld_3;
            asm volatile("ld.global.L1::no_allocate.v2.b32 {%0, %1}, [%2];"
                : "=r"(_ivld_3.x), "=r"(_ivld_3.y) : "l"((const void*)(_ivptr_3)) : "memory");
            words[6 + 0] = _ivld_3.x;
            words[6 + 1] = _ivld_3.y;
        }
        unsigned int out[2];
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
            : "=&f"(acc1_0[0]), "=&f"(acc1_0[1]) : "r"(words[2]), "f"(acc0_0[0]), "f"(acc0_0[1]));
        float acc2_0[2];
        asm(
            "{\n\t"
            ".reg .b16 lo, hi;\n\t"
            "mov.b32 {lo, hi}, %2;\n\t"
            "add.rn.f32.bf16 %0, lo, %3;\n\t"
            "add.rn.f32.bf16 %1, hi, %4;\n\t"
            "}\n"
            : "=&f"(acc2_0[0]), "=&f"(acc2_0[1]) : "r"(words[4]), "f"(acc1_0[0]), "f"(acc1_0[1]));
        float acc3_0[2];
        asm(
            "{\n\t"
            ".reg .b16 lo, hi;\n\t"
            "mov.b32 {lo, hi}, %2;\n\t"
            "add.rn.f32.bf16 %0, lo, %3;\n\t"
            "add.rn.f32.bf16 %1, hi, %4;\n\t"
            "}\n"
            : "=&f"(acc3_0[0]), "=&f"(acc3_0[1]) : "r"(words[6]), "f"(acc2_0[0]), "f"(acc2_0[1]));
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
            : "=&f"(acc1_1[0]), "=&f"(acc1_1[1]) : "r"(words[3]), "f"(acc0_1[0]), "f"(acc0_1[1]));
        float acc2_1[2];
        asm(
            "{\n\t"
            ".reg .b16 lo, hi;\n\t"
            "mov.b32 {lo, hi}, %2;\n\t"
            "add.rn.f32.bf16 %0, lo, %3;\n\t"
            "add.rn.f32.bf16 %1, hi, %4;\n\t"
            "}\n"
            : "=&f"(acc2_1[0]), "=&f"(acc2_1[1]) : "r"(words[5]), "f"(acc1_1[0]), "f"(acc1_1[1]));
        float acc3_1[2];
        asm(
            "{\n\t"
            ".reg .b16 lo, hi;\n\t"
            "mov.b32 {lo, hi}, %2;\n\t"
            "add.rn.f32.bf16 %0, lo, %3;\n\t"
            "add.rn.f32.bf16 %1, hi, %4;\n\t"
            "}\n"
            : "=&f"(acc3_1[0]), "=&f"(acc3_1[1]) : "r"(words[7]), "f"(acc2_1[0]), "f"(acc2_1[1]));
        unsigned int mean_1 = __as_u32(__float22bfloat162_rn(make_float2(mul_f32_rn_noftz(acc3_1[0], 0.25f), mul_f32_rn_noftz(acc3_1[1], 0.25f))));
        out[1] = mean_1;
        reinterpret_cast<int2*>(reinterpret_cast<unsigned int*>(y + ((long long)token * row + (long long)col * 4ll)))[0] = reinterpret_cast<int2*>(out)[0];
    }
}

} // extern "C"
