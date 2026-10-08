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

// Common preamble (typedefs, tensor-map ABI, compiler helpers) shared by this export's kernels.
#include "cake_all_gather_matmul_device_common.cuh"

#define CAKE_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define THREADS 32

extern "C" {

__global__ __launch_bounds__(32) void
kernel_cake_all_gather_matmul_ffe749098469b859325c(int32_t pg_world, int32_t pg_rank, CakePeerPointerTable<unsigned int> pg_flags)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // === Task calls (dependency order) ===
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    // nvlink_barrier(pg_flags) phase=0 owner_warp=0
    {
        const int __ws = pg_world;
        const int __me = pg_rank;
        const int __warp = warp;
        const int __lane = lane;
        if (__warp == 0) {
            unsigned* __local_epoch = pg_flags.ptrs[__me] + 0;
            unsigned __previous_epoch;
            asm volatile("ld.relaxed.sys.global.u32 %0, [%1];"
                : "=r"(__previous_epoch) : "l"(__local_epoch) : "memory");
            const unsigned __epoch = __previous_epoch + 1u;
            const int __bank = (int)(__epoch & 1u);
            const int __mailbox_base = 2 + (0 * 2 + __bank) * __ws;
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
