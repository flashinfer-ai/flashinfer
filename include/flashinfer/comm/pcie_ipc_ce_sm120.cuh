/*
 * Copyright (c) 2026 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef FLASHINFER_COMM_PCIE_IPC_CE_SM120_CUH_
#define FLASHINFER_COMM_PCIE_IPC_CE_SM120_CUH_

#include <stdint.h>

// Generated from the Weave binary wait-and-clear kernel.
// The ring reuses the existing packed-add and monotonic handshake kernels.
// Keep this generic load/store leaf defined for every build target so CUDA
// emits its host launch stub. Protocol admission remains SM120-only at runtime.
extern "C" {

__global__ void
kernel_pcie_ipc_ce_sm120_binary_wait_clear(unsigned int* __restrict__ self_flag)
{
    const int tid = threadIdx.x;
    const uint32_t warp = 0;
    uint32_t lane;
    asm("mov.u32 %0, %%laneid;" : "=r"(lane));


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // === Task calls (dependency order) ===
    {
        unsigned int* _gca_p = reinterpret_cast<unsigned int*>(self_flag) + (0);
        while (true) {
            unsigned int _gca_v;
            asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(_gca_v) : "l"(_gca_p));
            if (_gca_v >= (unsigned int)(1)) break;
        }
    }
    unsigned int cleared = 0;
    asm volatile("st.volatile.global.u32 [%0], %1;" :: "l"(self_flag), "r"(cleared) : "memory");
}

} // extern "C"
#endif
