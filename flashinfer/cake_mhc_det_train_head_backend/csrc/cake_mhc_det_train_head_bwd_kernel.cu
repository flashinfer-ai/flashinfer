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

extern "C" {

__global__ __launch_bounds__(256) void
kernel_cake_mhc_det_train_head_bwd(__nv_bfloat16* __restrict__ dy, __nv_bfloat16* __restrict__ dx, int T, CakeFastDivmod hv)
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
        long long row = (long long)((uint32_t)(hv.divisor)) * 16;
        long long xbase = (long long)token * (row * 4) + (long long)col * 16;
        unsigned int grad[8];
        {
            uint32_t _iv_0_0;
            uint32_t _iv_0_1;
            uint32_t _iv_0_2;
            uint32_t _iv_0_3;
            uint32_t _iv_0_4;
            uint32_t _iv_0_5;
            uint32_t _iv_0_6;
            uint32_t _iv_0_7;
            asm volatile("ld.global.L2::cache_hint.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8], %9;"
                : "=r"(_iv_0_0), "=r"(_iv_0_1), "=r"(_iv_0_2), "=r"(_iv_0_3), "=r"(_iv_0_4), "=r"(_iv_0_5), "=r"(_iv_0_6), "=r"(_iv_0_7) : "l"((const void*)(reinterpret_cast<unsigned int*>(dy + ((long long)v * 16)) + (0))), "l"(0x12F0000000000000ULL) : "memory");
            grad[0 + 0] = (uint32_t)_iv_0_0;
            grad[0 + 1] = (uint32_t)_iv_0_1;
            grad[0 + 2] = (uint32_t)_iv_0_2;
            grad[0 + 3] = (uint32_t)_iv_0_3;
            grad[0 + 4] = (uint32_t)_iv_0_4;
            grad[0 + 5] = (uint32_t)_iv_0_5;
            grad[0 + 6] = (uint32_t)_iv_0_6;
            grad[0 + 7] = (uint32_t)_iv_0_7;
        }
        unsigned int q[8];
        uint32_t quarter_0;
        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(quarter_0) : "r"(grad[0]), "r"(1048592000));
        q[0] = quarter_0;
        uint32_t quarter_1;
        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(quarter_1) : "r"(grad[1]), "r"(1048592000));
        q[1] = quarter_1;
        uint32_t quarter_2;
        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(quarter_2) : "r"(grad[2]), "r"(1048592000));
        q[2] = quarter_2;
        uint32_t quarter_3;
        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(quarter_3) : "r"(grad[3]), "r"(1048592000));
        q[3] = quarter_3;
        uint32_t quarter_4;
        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(quarter_4) : "r"(grad[4]), "r"(1048592000));
        q[4] = quarter_4;
        uint32_t quarter_5;
        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(quarter_5) : "r"(grad[5]), "r"(1048592000));
        q[5] = quarter_5;
        uint32_t quarter_6;
        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(quarter_6) : "r"(grad[6]), "r"(1048592000));
        q[6] = quarter_6;
        uint32_t quarter_7;
        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(quarter_7) : "r"(grad[7]), "r"(1048592000));
        q[7] = quarter_7;
        {
            const unsigned* _raw_stv8_1 = reinterpret_cast<const unsigned*>(q);
            asm volatile(
                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                :: "l"((void*)(reinterpret_cast<unsigned int*>(dx + xbase))), "r"(_raw_stv8_1[0]), "r"(_raw_stv8_1[1]), "r"(_raw_stv8_1[2]), "r"(_raw_stv8_1[3]), "r"(_raw_stv8_1[4]), "r"(_raw_stv8_1[5]), "r"(_raw_stv8_1[6]), "r"(_raw_stv8_1[7]) : "memory");
        }
        {
            const unsigned* _raw_stv8_2 = reinterpret_cast<const unsigned*>(q);
            asm volatile(
                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                :: "l"((void*)(reinterpret_cast<unsigned int*>(dx + (xbase + row)))), "r"(_raw_stv8_2[0]), "r"(_raw_stv8_2[1]), "r"(_raw_stv8_2[2]), "r"(_raw_stv8_2[3]), "r"(_raw_stv8_2[4]), "r"(_raw_stv8_2[5]), "r"(_raw_stv8_2[6]), "r"(_raw_stv8_2[7]) : "memory");
        }
        {
            const unsigned* _raw_stv8_3 = reinterpret_cast<const unsigned*>(q);
            asm volatile(
                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                :: "l"((void*)(reinterpret_cast<unsigned int*>(dx + (xbase + row * 2)))), "r"(_raw_stv8_3[0]), "r"(_raw_stv8_3[1]), "r"(_raw_stv8_3[2]), "r"(_raw_stv8_3[3]), "r"(_raw_stv8_3[4]), "r"(_raw_stv8_3[5]), "r"(_raw_stv8_3[6]), "r"(_raw_stv8_3[7]) : "memory");
        }
        {
            const unsigned* _raw_stv8_4 = reinterpret_cast<const unsigned*>(q);
            asm volatile(
                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                :: "l"((void*)(reinterpret_cast<unsigned int*>(dx + (xbase + row * 3)))), "r"(_raw_stv8_4[0]), "r"(_raw_stv8_4[1]), "r"(_raw_stv8_4[2]), "r"(_raw_stv8_4[3]), "r"(_raw_stv8_4[4]), "r"(_raw_stv8_4[5]), "r"(_raw_stv8_4[6]), "r"(_raw_stv8_4[7]) : "memory");
        }
    }
}

} // extern "C"
