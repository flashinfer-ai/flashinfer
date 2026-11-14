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

extern "C" {

__global__ __launch_bounds__(512) void
kernel_cake_mhc_train_head_bwd__lw64_t512_h7168(__nv_bfloat16* __restrict__ dy, __nv_bfloat16* __restrict__ dx, int T, CakeFastDivmod hv)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // === Task calls (declaration order) ===
    asm volatile("griddepcontrol.wait;" ::: "memory");
    unsigned int cta_q = (unsigned int)bid / 1792;
    unsigned int cta_r = (unsigned int)bid - cta_q * 1792;
    unsigned int within = cta_r * 512 + (unsigned int)tid;
    unsigned int sub = within / 1792;
    unsigned int token = cta_q * 512 + sub;
    unsigned int col = within - sub * 1792;
    if (token < (unsigned int)T) {
        long long row = 7168ll;
        long long xbase = (long long)token * (row * 4ll) + (long long)col * 4ll;
        unsigned int grad[2];
        long long gbase = (long long)token * row + (long long)col * 4ll;
        {
            const uint2* _ivptr_0 = reinterpret_cast<const uint2*>(reinterpret_cast<unsigned int*>(dy + gbase) + 0);
            uint2 _ivld_0;
            asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                : "=r"(_ivld_0.x), "=r"(_ivld_0.y) : "l"((const void*)(_ivptr_0)), "l"(0x12F0000000000000ULL) : "memory");
            grad[0 + 0] = _ivld_0.x;
            grad[0 + 1] = _ivld_0.y;
        }
        unsigned int q[2];
        uint32_t quarter_0;
        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(quarter_0) : "r"(grad[0]), "r"(1048592000));
        q[0] = quarter_0;
        uint32_t quarter_1;
        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(quarter_1) : "r"(grad[1]), "r"(1048592000));
        q[1] = quarter_1;
        reinterpret_cast<int2*>(reinterpret_cast<unsigned int*>(dx + xbase))[0] = reinterpret_cast<int2*>(q)[0];
        reinterpret_cast<int2*>(reinterpret_cast<unsigned int*>(dx + (xbase + row)))[0] = reinterpret_cast<int2*>(q)[0];
        reinterpret_cast<int2*>(reinterpret_cast<unsigned int*>(dx + (xbase + row * 2ll)))[0] = reinterpret_cast<int2*>(q)[0];
        reinterpret_cast<int2*>(reinterpret_cast<unsigned int*>(dx + (xbase + row * 3ll)))[0] = reinterpret_cast<int2*>(q)[0];
    }
}

} // extern "C"
