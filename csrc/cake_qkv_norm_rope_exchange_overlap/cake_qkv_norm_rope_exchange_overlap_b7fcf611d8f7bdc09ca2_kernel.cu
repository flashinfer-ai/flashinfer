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
template <typename T, int Capacity = 8>
struct __align__(16) CakePeerPointerTable { T* ptrs[Capacity]; };
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
#define NUM_MAIN_STAGES 1
#define THREADS 32

#include <math_constants.h>
extern "C" {

__global__ __launch_bounds__(THREADS) void
kernel_cake_qkv_norm_rope_exchange_overlap_b7fcf611d8f7bdc09ca2(int32_t pg_world, int32_t pg_rank, CakePeerPointerTable<unsigned int> pg_flags)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // === Task calls (dependency order) ===
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    // nvlink_barrier(pg_flags) phase=1 owner_warp=0
    {
        const int __ws = pg_world;
        const int __me = pg_rank;
        const int __warp = warp;
        const int __lane = lane;
        if (__warp == 0) {
            unsigned* __local_epoch = pg_flags.ptrs[__me] + 1;
            unsigned __previous_epoch;
            asm volatile("ld.relaxed.sys.global.u32 %0, [%1];"
                : "=r"(__previous_epoch) : "l"(__local_epoch) : "memory");
            const unsigned __epoch = __previous_epoch + 1u;
            const int __bank = (int)(__epoch & 1u);
            const int __mailbox_base = 2 + (1 * 2 + __bank) * __ws;
            asm volatile("fence.proxy.async.global;" ::: "memory");
            if (__lane < __ws) {
                unsigned* __peer_mailbox = pg_flags.ptrs[__lane] + __mailbox_base + __me;
                asm volatile("st.release.sys.global.u32 [%0], %1;"
                    :: "l"(__peer_mailbox), "r"(__epoch) : "memory");
                unsigned* __local_mailbox = pg_flags.ptrs[__me] + __mailbox_base + __lane;
                while (true) {
                    unsigned __v;
                    asm volatile("ld.relaxed.sys.global.u32 %0, [%1];" : "=r"(__v) : "l"(__local_mailbox) : "memory");
                    if (__v != __epoch) continue;
                    asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(__v) : "l"(__local_mailbox) : "memory");
                    if (__v == __epoch) break;
                }
                asm volatile("fence.proxy.alias;" ::: "memory");
            }
            __syncwarp();
            asm volatile("fence.proxy.async.global;" ::: "memory");
            if (__lane == 0) {
                asm volatile("st.release.sys.global.u32 [%0], %1;"
                    :: "l"(__local_epoch), "r"(__epoch) : "memory");
            }
        }
    }
}

} // extern "C"
