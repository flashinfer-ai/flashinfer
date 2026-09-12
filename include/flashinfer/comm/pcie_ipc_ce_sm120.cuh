
#ifndef FLASHINFER_COMM_PCIE_IPC_CE_SM120_CUH_
#define FLASHINFER_COMM_PCIE_IPC_CE_SM120_CUH_
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ == 1200
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

typedef signed char        int8_t;
typedef unsigned char      uint8_t;
typedef unsigned short     uint16_t;
typedef unsigned int       uint32_t;
#if defined(__CUDACC_RTC__)
typedef unsigned long long uint64_t;
#else
typedef unsigned long      uint64_t;
#endif
static_assert(sizeof(uint64_t) == 8, "Pcie requires an LP64 CUDA host ABI");
typedef signed int         int32_t;
typedef short int          int16_t;
struct __align__(128) PcieTensorMap { uint64_t opaque[16]; };
struct __align__(64) PcieTensorMap64 { uint64_t opaque[16]; };
static_assert(sizeof(PcieTensorMap64) == 128, "64-aligned tensor-map ABI size");
static_assert(alignof(PcieTensorMap64) == 64, "64-aligned tensor-map ABI alignment");
template <int N>
struct __align__(128) PcieTensorMapPack { PcieTensorMap maps[N]; };

#if defined(__CUDACC_RTC__)
typedef struct __align__(128) { uint64_t opaque[16]; } CUtensorMap;
#else
#include <cuda.h>
#endif

static_assert(sizeof(CUtensorMap) == 128, "CUtensorMap CUDA ABI must be 128 bytes");
static_assert(alignof(PcieTensorMap) >= alignof(CUtensorMap), "PcieTensorMap alignment must cover the CUtensorMap CUDA ABI");
#include <cuda_bf16.h>
#include <cuda_fp8.h>

__device__ __forceinline__ int make_warp_uniform(int x) {
    int result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1F, 0xFFFFFFFF;"
                 : "=r"(result) : "r"(x));
    return result;
}

#include <math_constants.h>

#define PCIE_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define THREADS 256

extern "C" {

__global__ void
kernel_pcie_ipc_ce_sm120_add_bf16(unsigned int* out, const unsigned int* a, const unsigned int* b, long long num_packs)
{
    const int tid = threadIdx.x;
    const uint32_t warp = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
    uint32_t lane;
    asm("mov.u32 %0, %%laneid;" : "=r"(lane));


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // === Task calls (dependency order) ===
    long long start = (long long)bid * 256 + (long long)tid;
    long long stride = (long long)num_bids * 256;
    for (long long pack = start; pack < num_packs; pack += stride) {
        long long offset = pack * 4;
        unsigned int _vec_load_0[4];
        {
            int4 _iv4 = *reinterpret_cast<const int4*>(a + offset);
            _vec_load_0[0 + 0] = _iv4.x;
            _vec_load_0[0 + 1] = _iv4.y;
            _vec_load_0[0 + 2] = _iv4.z;
            _vec_load_0[0 + 3] = _iv4.w;
        }
        unsigned int _vec_load_1[4];
        {
            int4 _iv4 = *reinterpret_cast<const int4*>(b + offset);
            _vec_load_1[0 + 0] = _iv4.x;
            _vec_load_1[0 + 1] = _iv4.y;
            _vec_load_1[0 + 2] = _iv4.z;
            _vec_load_1[0 + 3] = _iv4.w;
        }
        unsigned int summed[4];
        uint32_t _bf16x2_add_0;
        asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_0) : "r"(_vec_load_0[0]), "r"(_vec_load_1[0]));
        summed[0] = _bf16x2_add_0;
        uint32_t _bf16x2_add_1;
        asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_1) : "r"(_vec_load_0[1]), "r"(_vec_load_1[1]));
        summed[1] = _bf16x2_add_1;
        uint32_t _bf16x2_add_2;
        asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_2) : "r"(_vec_load_0[2]), "r"(_vec_load_1[2]));
        summed[2] = _bf16x2_add_2;
        uint32_t _bf16x2_add_3;
        asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_3) : "r"(_vec_load_0[3]), "r"(_vec_load_1[3]));
        summed[3] = _bf16x2_add_3;
        reinterpret_cast<int4*>(out + offset)[0] = reinterpret_cast<int4*>(summed)[0];
    }
}

} // extern "C"

#undef PCIE_INF
#undef NUM_MAIN_STAGES
#undef THREADS

#define PCIE_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define THREADS 256

extern "C" {

__global__ void
kernel_pcie_ipc_ce_sm120_add_f16(unsigned int* out, const unsigned int* a, const unsigned int* b, long long num_packs)
{
    const int tid = threadIdx.x;
    const uint32_t warp = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
    uint32_t lane;
    asm("mov.u32 %0, %%laneid;" : "=r"(lane));


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // === Task calls (dependency order) ===
    long long start = (long long)bid * 256 + (long long)tid;
    long long stride = (long long)num_bids * 256;
    for (long long pack = start; pack < num_packs; pack += stride) {
        long long offset = pack * 4;
        unsigned int _vec_load_0[4];
        {
            int4 _iv4 = *reinterpret_cast<const int4*>(a + offset);
            _vec_load_0[0 + 0] = _iv4.x;
            _vec_load_0[0 + 1] = _iv4.y;
            _vec_load_0[0 + 2] = _iv4.z;
            _vec_load_0[0 + 3] = _iv4.w;
        }
        unsigned int _vec_load_1[4];
        {
            int4 _iv4 = *reinterpret_cast<const int4*>(b + offset);
            _vec_load_1[0 + 0] = _iv4.x;
            _vec_load_1[0 + 1] = _iv4.y;
            _vec_load_1[0 + 2] = _iv4.z;
            _vec_load_1[0 + 3] = _iv4.w;
        }
        unsigned int summed[4];
        uint32_t _f16x2_add_0;
        asm("add.f16x2 %0, %1, %2;" : "=r"(_f16x2_add_0) : "r"(_vec_load_0[0]), "r"(_vec_load_1[0]));
        summed[0] = _f16x2_add_0;
        uint32_t _f16x2_add_1;
        asm("add.f16x2 %0, %1, %2;" : "=r"(_f16x2_add_1) : "r"(_vec_load_0[1]), "r"(_vec_load_1[1]));
        summed[1] = _f16x2_add_1;
        uint32_t _f16x2_add_2;
        asm("add.f16x2 %0, %1, %2;" : "=r"(_f16x2_add_2) : "r"(_vec_load_0[2]), "r"(_vec_load_1[2]));
        summed[2] = _f16x2_add_2;
        uint32_t _f16x2_add_3;
        asm("add.f16x2 %0, %1, %2;" : "=r"(_f16x2_add_3) : "r"(_vec_load_0[3]), "r"(_vec_load_1[3]));
        summed[3] = _f16x2_add_3;
        reinterpret_cast<int4*>(out + offset)[0] = reinterpret_cast<int4*>(summed)[0];
    }
}

} // extern "C"

#undef PCIE_INF
#undef NUM_MAIN_STAGES
#undef THREADS

#define PCIE_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define THREADS 1

extern "C" {

__global__ void
kernel_pcie_ipc_ce_sm120_publish(unsigned int* __restrict__ peer_flag, unsigned int* __restrict__ send_counter)
{
    const int tid = threadIdx.x;
    const uint32_t warp = 0;
    uint32_t lane;
    asm("mov.u32 %0, %%laneid;" : "=r"(lane));


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // === Task calls (dependency order) ===
    uint32_t _sysv_u32_0;
    asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(_sysv_u32_0) : "l"(send_counter) : "memory");
    unsigned int next_count = _sysv_u32_0 + 1;
    asm volatile("st.volatile.global.u32 [%0], %1;" :: "l"(send_counter), "r"(next_count) : "memory");
    asm volatile("st.release.sys.global.u32 [%0], %1;" :: "l"(peer_flag), "r"(next_count) : "memory");
}

} // extern "C"

#undef PCIE_INF
#undef NUM_MAIN_STAGES
#undef THREADS

#define PCIE_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define THREADS 1

extern "C" {

__global__ void
kernel_pcie_ipc_ce_sm120_wait(unsigned int* __restrict__ self_flag, unsigned int* __restrict__ wait_counter)
{
    const int tid = threadIdx.x;
    const uint32_t warp = 0;
    uint32_t lane;
    asm("mov.u32 %0, %%laneid;" : "=r"(lane));


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // === Task calls (dependency order) ===
    uint32_t _sysv_u32_0;
    asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(_sysv_u32_0) : "l"(wait_counter) : "memory");
    unsigned int expected = _sysv_u32_0 + 1;
    asm volatile("st.volatile.global.u32 [%0], %1;" :: "l"(wait_counter), "r"(expected) : "memory");
    {
        unsigned int* _gca_p = reinterpret_cast<unsigned int*>(self_flag) + (0);
        while (true) {
            unsigned int _gca_v;
            asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(_gca_v) : "l"(_gca_p));
            if (static_cast<int32_t>(_gca_v - static_cast<uint32_t>(expected)) >= 0) break;
        }
    }
}

} // extern "C"

#undef PCIE_INF
#undef NUM_MAIN_STAGES
#undef THREADS

#define PCIE_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define THREADS 1

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

#undef PCIE_INF
#undef NUM_MAIN_STAGES
#undef THREADS

#endif
#endif
