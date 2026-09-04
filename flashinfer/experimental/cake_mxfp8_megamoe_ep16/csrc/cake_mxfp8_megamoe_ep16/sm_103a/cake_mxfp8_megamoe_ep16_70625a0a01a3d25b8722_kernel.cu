/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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
struct __align__(128) CakeTensorMap { uint64_t opaque[16]; };
template <int N>
struct __align__(128) CakeTensorMapPack { CakeTensorMap maps[N]; };

#if defined(__CUDACC_RTC__)
typedef struct __align__(128) { uint64_t opaque[16]; } CUtensorMap;
#else
#include <cuda.h>
#endif

static_assert(sizeof(CUtensorMap) == 128, "CUtensorMap CUDA ABI must be 128 bytes");
static_assert(alignof(CUtensorMap) == 128, "CUtensorMap CUDA ABI must be 128-byte aligned");
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

__device__ __forceinline__ uint32_t elect_sync() {
    uint32_t pred = 0;
    asm volatile(
        "{\n\t"
        ".reg .pred %%px;\n\t"
        "elect.sync _|%%px, %1;\n\t"
        "@%%px mov.s32 %0, 1;\n\t"
        "}\n"
        : "+r"(pred)
        : "r"(0xFFFFFFFF));
    return pred;
}

extern "C" {

__global__ __launch_bounds__(256) void
kernel_cake_mxfp8_megamoe_ep16_70625a0a01a3d25b8722(unsigned int* __restrict__ remote_ready, __nv_bfloat16* __restrict__ output)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // === Task calls (dependency order) ===
    int owner_rank = tid;
    if (owner_rank < 16) {
        {
            unsigned int* _sre_ptr_0 = (reinterpret_cast<unsigned int*>(remote_ready) + (owner_rank));
            const unsigned int _sre_expected_0 = static_cast<unsigned int>(1);
            const unsigned long long _sre_start_0 = clock64();
            bool _sre_matched_0 = false;
            do {
                unsigned int _sre_value_0;
                asm volatile("ld.relaxed.sys.u32 %0, [%1];" : "=r"(_sre_value_0) : "l"(_sre_ptr_0) : "memory");
                _sre_matched_0 = (_sre_value_0 == _sre_expected_0);
            } while (!_sre_matched_0 && ((clock64() - _sre_start_0) <= static_cast<unsigned long long>(4000000000)));
            if (__builtin_expect(!_sre_matched_0, 0)) {
                asm volatile("trap;");
                return;
            }
        }
    }
    asm volatile("barrier.sync 13, 256;" ::: "memory");
    if (elect_sync()) {
        asm volatile("fence.acquire.sys;" ::: "memory");
    }
    asm volatile("barrier.sync 13, 256;" ::: "memory");
    if (owner_rank < 16) {
        asm volatile("st.relaxed.sys.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(remote_ready) + (owner_rank))), "r"(static_cast<unsigned int>(0)) : "memory");
    }
}

} // extern "C"
