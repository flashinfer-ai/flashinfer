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
kernel_cake_mhc_train_head_bwd__acc(__nv_bfloat16* __restrict__ dy, __nv_bfloat16* __restrict__ dx, int T, CakeFastDivmod hv)
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
        unsigned int old[32];
        {
            uint32_t _iv_1_0;
            uint32_t _iv_1_1;
            uint32_t _iv_1_2;
            uint32_t _iv_1_3;
            uint32_t _iv_1_4;
            uint32_t _iv_1_5;
            uint32_t _iv_1_6;
            uint32_t _iv_1_7;
            asm volatile("ld.global.L1::no_allocate.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                : "=r"(_iv_1_0), "=r"(_iv_1_1), "=r"(_iv_1_2), "=r"(_iv_1_3), "=r"(_iv_1_4), "=r"(_iv_1_5), "=r"(_iv_1_6), "=r"(_iv_1_7) : "l"((const void*)(reinterpret_cast<unsigned int*>(dx + (xbase + row * 0)) + (0))) : "memory");
            old[0 + 0] = (uint32_t)_iv_1_0;
            old[0 + 1] = (uint32_t)_iv_1_1;
            old[0 + 2] = (uint32_t)_iv_1_2;
            old[0 + 3] = (uint32_t)_iv_1_3;
            old[0 + 4] = (uint32_t)_iv_1_4;
            old[0 + 5] = (uint32_t)_iv_1_5;
            old[0 + 6] = (uint32_t)_iv_1_6;
            old[0 + 7] = (uint32_t)_iv_1_7;
        }
        {
            uint32_t _iv_2_0;
            uint32_t _iv_2_1;
            uint32_t _iv_2_2;
            uint32_t _iv_2_3;
            uint32_t _iv_2_4;
            uint32_t _iv_2_5;
            uint32_t _iv_2_6;
            uint32_t _iv_2_7;
            asm volatile("ld.global.L1::no_allocate.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                : "=r"(_iv_2_0), "=r"(_iv_2_1), "=r"(_iv_2_2), "=r"(_iv_2_3), "=r"(_iv_2_4), "=r"(_iv_2_5), "=r"(_iv_2_6), "=r"(_iv_2_7) : "l"((const void*)(reinterpret_cast<unsigned int*>(dx + (xbase + row * 1)) + (0))) : "memory");
            old[8 + 0] = (uint32_t)_iv_2_0;
            old[8 + 1] = (uint32_t)_iv_2_1;
            old[8 + 2] = (uint32_t)_iv_2_2;
            old[8 + 3] = (uint32_t)_iv_2_3;
            old[8 + 4] = (uint32_t)_iv_2_4;
            old[8 + 5] = (uint32_t)_iv_2_5;
            old[8 + 6] = (uint32_t)_iv_2_6;
            old[8 + 7] = (uint32_t)_iv_2_7;
        }
        {
            uint32_t _iv_3_0;
            uint32_t _iv_3_1;
            uint32_t _iv_3_2;
            uint32_t _iv_3_3;
            uint32_t _iv_3_4;
            uint32_t _iv_3_5;
            uint32_t _iv_3_6;
            uint32_t _iv_3_7;
            asm volatile("ld.global.L1::no_allocate.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                : "=r"(_iv_3_0), "=r"(_iv_3_1), "=r"(_iv_3_2), "=r"(_iv_3_3), "=r"(_iv_3_4), "=r"(_iv_3_5), "=r"(_iv_3_6), "=r"(_iv_3_7) : "l"((const void*)(reinterpret_cast<unsigned int*>(dx + (xbase + row * 2)) + (0))) : "memory");
            old[16 + 0] = (uint32_t)_iv_3_0;
            old[16 + 1] = (uint32_t)_iv_3_1;
            old[16 + 2] = (uint32_t)_iv_3_2;
            old[16 + 3] = (uint32_t)_iv_3_3;
            old[16 + 4] = (uint32_t)_iv_3_4;
            old[16 + 5] = (uint32_t)_iv_3_5;
            old[16 + 6] = (uint32_t)_iv_3_6;
            old[16 + 7] = (uint32_t)_iv_3_7;
        }
        {
            uint32_t _iv_4_0;
            uint32_t _iv_4_1;
            uint32_t _iv_4_2;
            uint32_t _iv_4_3;
            uint32_t _iv_4_4;
            uint32_t _iv_4_5;
            uint32_t _iv_4_6;
            uint32_t _iv_4_7;
            asm volatile("ld.global.L1::no_allocate.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                : "=r"(_iv_4_0), "=r"(_iv_4_1), "=r"(_iv_4_2), "=r"(_iv_4_3), "=r"(_iv_4_4), "=r"(_iv_4_5), "=r"(_iv_4_6), "=r"(_iv_4_7) : "l"((const void*)(reinterpret_cast<unsigned int*>(dx + (xbase + row * 3)) + (0))) : "memory");
            old[24 + 0] = (uint32_t)_iv_4_0;
            old[24 + 1] = (uint32_t)_iv_4_1;
            old[24 + 2] = (uint32_t)_iv_4_2;
            old[24 + 3] = (uint32_t)_iv_4_3;
            old[24 + 4] = (uint32_t)_iv_4_4;
            old[24 + 5] = (uint32_t)_iv_4_5;
            old[24 + 6] = (uint32_t)_iv_4_6;
            old[24 + 7] = (uint32_t)_iv_4_7;
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
        unsigned int total_0[8];
        uint32_t summed_0;
        asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(summed_0) : "r"(old[0]), "r"(q[0]));
        total_0[0] = summed_0;
        uint32_t summed_1;
        asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(summed_1) : "r"(old[1]), "r"(q[1]));
        total_0[1] = summed_1;
        uint32_t summed_2;
        asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(summed_2) : "r"(old[2]), "r"(q[2]));
        total_0[2] = summed_2;
        uint32_t summed_3;
        asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(summed_3) : "r"(old[3]), "r"(q[3]));
        total_0[3] = summed_3;
        uint32_t summed_4;
        asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(summed_4) : "r"(old[4]), "r"(q[4]));
        total_0[4] = summed_4;
        uint32_t summed_5;
        asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(summed_5) : "r"(old[5]), "r"(q[5]));
        total_0[5] = summed_5;
        uint32_t summed_6;
        asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(summed_6) : "r"(old[6]), "r"(q[6]));
        total_0[6] = summed_6;
        uint32_t summed_7;
        asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(summed_7) : "r"(old[7]), "r"(q[7]));
        total_0[7] = summed_7;
        {
            const unsigned* _raw_stv8_5 = reinterpret_cast<const unsigned*>(total_0);
            asm volatile(
                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                :: "l"((void*)(reinterpret_cast<unsigned int*>(dx + (xbase + row * 0)))), "r"(_raw_stv8_5[0]), "r"(_raw_stv8_5[1]), "r"(_raw_stv8_5[2]), "r"(_raw_stv8_5[3]), "r"(_raw_stv8_5[4]), "r"(_raw_stv8_5[5]), "r"(_raw_stv8_5[6]), "r"(_raw_stv8_5[7]) : "memory");
        }
        unsigned int total_1[8];
        uint32_t summed_8;
        asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(summed_8) : "r"(old[8]), "r"(q[0]));
        total_1[0] = summed_8;
        uint32_t summed_9;
        asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(summed_9) : "r"(old[9]), "r"(q[1]));
        total_1[1] = summed_9;
        uint32_t summed_10;
        asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(summed_10) : "r"(old[10]), "r"(q[2]));
        total_1[2] = summed_10;
        uint32_t summed_11;
        asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(summed_11) : "r"(old[11]), "r"(q[3]));
        total_1[3] = summed_11;
        uint32_t summed_12;
        asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(summed_12) : "r"(old[12]), "r"(q[4]));
        total_1[4] = summed_12;
        uint32_t summed_13;
        asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(summed_13) : "r"(old[13]), "r"(q[5]));
        total_1[5] = summed_13;
        uint32_t summed_14;
        asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(summed_14) : "r"(old[14]), "r"(q[6]));
        total_1[6] = summed_14;
        uint32_t summed_15;
        asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(summed_15) : "r"(old[15]), "r"(q[7]));
        total_1[7] = summed_15;
        {
            const unsigned* _raw_stv8_6 = reinterpret_cast<const unsigned*>(total_1);
            asm volatile(
                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                :: "l"((void*)(reinterpret_cast<unsigned int*>(dx + (xbase + row * 1)))), "r"(_raw_stv8_6[0]), "r"(_raw_stv8_6[1]), "r"(_raw_stv8_6[2]), "r"(_raw_stv8_6[3]), "r"(_raw_stv8_6[4]), "r"(_raw_stv8_6[5]), "r"(_raw_stv8_6[6]), "r"(_raw_stv8_6[7]) : "memory");
        }
        unsigned int total_2[8];
        uint32_t summed_16;
        asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(summed_16) : "r"(old[16]), "r"(q[0]));
        total_2[0] = summed_16;
        uint32_t summed_17;
        asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(summed_17) : "r"(old[17]), "r"(q[1]));
        total_2[1] = summed_17;
        uint32_t summed_18;
        asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(summed_18) : "r"(old[18]), "r"(q[2]));
        total_2[2] = summed_18;
        uint32_t summed_19;
        asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(summed_19) : "r"(old[19]), "r"(q[3]));
        total_2[3] = summed_19;
        uint32_t summed_20;
        asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(summed_20) : "r"(old[20]), "r"(q[4]));
        total_2[4] = summed_20;
        uint32_t summed_21;
        asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(summed_21) : "r"(old[21]), "r"(q[5]));
        total_2[5] = summed_21;
        uint32_t summed_22;
        asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(summed_22) : "r"(old[22]), "r"(q[6]));
        total_2[6] = summed_22;
        uint32_t summed_23;
        asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(summed_23) : "r"(old[23]), "r"(q[7]));
        total_2[7] = summed_23;
        {
            const unsigned* _raw_stv8_7 = reinterpret_cast<const unsigned*>(total_2);
            asm volatile(
                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                :: "l"((void*)(reinterpret_cast<unsigned int*>(dx + (xbase + row * 2)))), "r"(_raw_stv8_7[0]), "r"(_raw_stv8_7[1]), "r"(_raw_stv8_7[2]), "r"(_raw_stv8_7[3]), "r"(_raw_stv8_7[4]), "r"(_raw_stv8_7[5]), "r"(_raw_stv8_7[6]), "r"(_raw_stv8_7[7]) : "memory");
        }
        unsigned int total_3[8];
        uint32_t summed_24;
        asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(summed_24) : "r"(old[24]), "r"(q[0]));
        total_3[0] = summed_24;
        uint32_t summed_25;
        asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(summed_25) : "r"(old[25]), "r"(q[1]));
        total_3[1] = summed_25;
        uint32_t summed_26;
        asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(summed_26) : "r"(old[26]), "r"(q[2]));
        total_3[2] = summed_26;
        uint32_t summed_27;
        asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(summed_27) : "r"(old[27]), "r"(q[3]));
        total_3[3] = summed_27;
        uint32_t summed_28;
        asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(summed_28) : "r"(old[28]), "r"(q[4]));
        total_3[4] = summed_28;
        uint32_t summed_29;
        asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(summed_29) : "r"(old[29]), "r"(q[5]));
        total_3[5] = summed_29;
        uint32_t summed_30;
        asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(summed_30) : "r"(old[30]), "r"(q[6]));
        total_3[6] = summed_30;
        uint32_t summed_31;
        asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(summed_31) : "r"(old[31]), "r"(q[7]));
        total_3[7] = summed_31;
        {
            const unsigned* _raw_stv8_8 = reinterpret_cast<const unsigned*>(total_3);
            asm volatile(
                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                :: "l"((void*)(reinterpret_cast<unsigned int*>(dx + (xbase + row * 3)))), "r"(_raw_stv8_8[0]), "r"(_raw_stv8_8[1]), "r"(_raw_stv8_8[2]), "r"(_raw_stv8_8[3]), "r"(_raw_stv8_8[4]), "r"(_raw_stv8_8[5]), "r"(_raw_stv8_8[6]), "r"(_raw_stv8_8[7]) : "memory");
        }
    }
}

} // extern "C"
