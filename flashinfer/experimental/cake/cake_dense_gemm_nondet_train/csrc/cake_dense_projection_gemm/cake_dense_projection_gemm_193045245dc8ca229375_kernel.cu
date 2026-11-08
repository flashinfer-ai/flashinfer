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


__device__ __forceinline__ void mbarrier_init(int mbar_addr, int count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;"
        :: "r"(mbar_addr), "r"(count) : "memory");
}



// CTA-local pipelines have short, resident producer/consumer edges.  Omitting
// suspendTimeHint keeps a miss on the lightweight TRYWAIT retry path; the
// explicit loop still makes this helper blocking until acquire succeeds.
__device__ __forceinline__ void mbarrier_wait(int mbar_addr, int phase) {
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "LAB_WAIT:\n\t"
        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64"
        " P1, [%0], %1;\n\t"
        "@P1 bra.uni DONE;\n\t"
        "bra.uni LAB_WAIT;\n\t"
        "DONE:\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"(phase) : "memory");
}

__device__ __forceinline__ void mbarrier_wait_cluster_hint(
        int mbar_addr, int phase, uint32_t suspend_time_hint) {
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "LAB_WAIT_CLUSTER_HINT:\n\t"
        "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64"
        " P1, [%0], %1, %2;\n\t"
        "@P1 bra.uni DONE_CLUSTER_HINT;\n\t"
        "bra.uni LAB_WAIT_CLUSTER_HINT;\n\t"
        "DONE_CLUSTER_HINT:\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"(phase), "r"(suspend_time_hint) : "memory");
}




union MmaSmemDesc {
    uint64_t u64;
    uint32_t u32[2];
};


__device__ __forceinline__ void elect_commit_cg2_multicast(int mbar_addr, uint16_t cta_mask) {
    asm volatile(
        "{\n\t"
        ".reg .pred leader;\n\t"
        "elect.sync _|leader, 0xFFFFFFFF;\n\t"
        "@leader tcgen05.commit.cta_group::2.mbarrier::arrive::one"
        ".shared::cluster.multicast::cluster.b64 [%0], %1;\n\t"
        "}\n"
        :: "r"(mbar_addr), "h"(cta_mask) : "memory");
}







__device__ __forceinline__ void tma_3d_gmem2smem_cta2(
    int dst, const void *tmap_ptr, int x, int y, int z, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cluster.global"
        ".mbarrier::complete_tx::bytes.cta_group::2"
        " [%0], [%1, {%2, %3, %4}], [%5];"
        :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y), "r"(z),
           "r"(mbar_addr) : "memory");
}


__device__ __forceinline__ void tma_store_3d(
    const void *tmap, int x, int y, int z, unsigned smem_addr) {
    asm volatile(
        "cp.async.bulk.tensor.3d.global.shared::cta.tile.bulk_group"
        " [%0, {%1, %2, %3}], [%4];"
        :: "l"(tmap), "r"(x), "r"(y), "r"(z), "r"(smem_addr) : "memory");
}




extern "C" {

__global__ __launch_bounds__(256) __cluster_dims__(2,1,1) void
kernel_cake_dense_projection_gemm_193045245dc8ca229375(const __grid_constant__ CUtensorMap A, const __grid_constant__ CUtensorMap B, const __grid_constant__ CUtensorMap C, int M, int N, int K, int L)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
#if __CUDA_ARCH__ == 1000
    asm volatile("{ .reg .u64 smem_ptr; cvta.to.shared.u64 smem_ptr, %1; cvt.u32.u64 %0, smem_ptr; }" : "=r"(smem) : "l"(smem_raw));
    smem = make_warp_uniform(smem);
#else
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);
#endif

    const int mbar_base = smem;
    #define tma_full_addr (mbar_base + 0)
    #define mma_done_addr (mbar_base + 32)
    #define acc_full_addr (mbar_base + 64)
    #define acc_empty_addr (mbar_base + 72)
    #define tmem_free_addr (mbar_base + 80)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    const unsigned int clusters_x = gridDim.x / 2;
    const unsigned int cluster_id = ((blockIdx.z * gridDim.y + blockIdx.y) * clusters_x) + blockIdx.x / 2;
    const unsigned int num_clusters = clusters_x * gridDim.y * gridDim.z;

    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));

    // Prologue
    __nv_bfloat16* smem_a = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int smem_a_addr = smem + 1024;
    __nv_bfloat16* smem_b = reinterpret_cast<__nv_bfloat16*>(smem_raw + 33792);
    const int smem_b_addr = smem + 33792;
    __nv_bfloat16* smem_out = reinterpret_cast<__nv_bfloat16*>(smem_raw + 189440);
    const int smem_out_addr = smem + 189440;

    // Kernel setup ops
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");

    // Mbarrier init (11 barriers at smem_raw[0..88))

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'tma_pipe' ---
            // tma_full: 4 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            mbarrier_init(smem + 8, 1);
            mbarrier_init(smem + 16, 1);
            mbarrier_init(smem + 24, 1);
            // mma_done: 4 barriers, init_count=1
            mbarrier_init(smem + 32, 1);
            mbarrier_init(smem + 40, 1);
            mbarrier_init(smem + 48, 1);
            mbarrier_init(smem + 56, 1);
            // --- pipeline 'acc_pipe' ---
            // acc_full: 1 barriers, init_count=1
            mbarrier_init(smem + 64, 1);
            // acc_empty: 1 barriers, init_count=8
            mbarrier_init(smem + 72, 8);
            // tmem_free: 1 barriers, init_count=32
            mbarrier_init(smem + 80, 32);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();
    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");

    // TMEM alloc (512 columns, 512 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 88);
    if (warp == 6) {
        int _tmem_hold = smem + 88;
        asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(512) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
    asm volatile("griddepcontrol.wait;" ::: "memory");

    const unsigned int taddr = tmem_addr_storage[0];
    const unsigned int tmem_accum = taddr;

    // ---- Role: epilogue ----
    if (warp <= 3) {
        { // epilogue_main
            const int epi_warp = warp;
            unsigned int m_tiles_e = (unsigned int)((M + 511) / 512);
            unsigned int mn_tiles_e = m_tiles_e * (unsigned int)((N + 224 - 1) / 224);
            unsigned int total_e = mn_tiles_e * (unsigned int)L;
            unsigned int lane_addr = taddr + (unsigned int)(epi_warp * 32 << 16);
            unsigned int warp_slots = smem_out_addr + (unsigned int)(epi_warp * 2 * 2048);
            float acc[32];
            unsigned int pk[16];
            unsigned int st_row = (unsigned int)((lane & 7) * 64 + ((lane >> 3 ^ (lane & 7) >> 1) << 4));
            if (epi_warp == 0) {
                if (lane == 0) {
                    asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&C))) : "memory");
                }
            }
            unsigned int _phase_acc_full = 0;
            #pragma unroll 1
            for (unsigned int t_e = (unsigned int)cluster_id; t_e < total_e; t_e += (unsigned int)num_clusters) {
                int f_row = (int)(t_e % m_tiles_e) * 512 + cta_rank * 256 + epi_warp * 32;
                int t_col = (int)(t_e % mn_tiles_e / m_tiles_e) * 224;
                int l_e = (int)(t_e / mn_tiles_e);
                mbarrier_wait(acc_full_addr, _phase_acc_full);
                asm volatile("tcgen05.fence::after_thread_sync;");
                unsigned int col_0 = lane_addr + 0;
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&acc[0])), "=r"(*reinterpret_cast<uint32_t*>(&acc[1])), "=r"(*reinterpret_cast<uint32_t*>(&acc[2])), "=r"(*reinterpret_cast<uint32_t*>(&acc[3])), "=r"(*reinterpret_cast<uint32_t*>(&acc[4])), "=r"(*reinterpret_cast<uint32_t*>(&acc[5])), "=r"(*reinterpret_cast<uint32_t*>(&acc[6])), "=r"(*reinterpret_cast<uint32_t*>(&acc[7])), "=r"(*reinterpret_cast<uint32_t*>(&acc[8])), "=r"(*reinterpret_cast<uint32_t*>(&acc[9])), "=r"(*reinterpret_cast<uint32_t*>(&acc[10])), "=r"(*reinterpret_cast<uint32_t*>(&acc[11])), "=r"(*reinterpret_cast<uint32_t*>(&acc[12])), "=r"(*reinterpret_cast<uint32_t*>(&acc[13])), "=r"(*reinterpret_cast<uint32_t*>(&acc[14])), "=r"(*reinterpret_cast<uint32_t*>(&acc[15]))
                    : "r"(col_0 + 0));
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&acc[16])), "=r"(*reinterpret_cast<uint32_t*>(&acc[17])), "=r"(*reinterpret_cast<uint32_t*>(&acc[18])), "=r"(*reinterpret_cast<uint32_t*>(&acc[19])), "=r"(*reinterpret_cast<uint32_t*>(&acc[20])), "=r"(*reinterpret_cast<uint32_t*>(&acc[21])), "=r"(*reinterpret_cast<uint32_t*>(&acc[22])), "=r"(*reinterpret_cast<uint32_t*>(&acc[23])), "=r"(*reinterpret_cast<uint32_t*>(&acc[24])), "=r"(*reinterpret_cast<uint32_t*>(&acc[25])), "=r"(*reinterpret_cast<uint32_t*>(&acc[26])), "=r"(*reinterpret_cast<uint32_t*>(&acc[27])), "=r"(*reinterpret_cast<uint32_t*>(&acc[28])), "=r"(*reinterpret_cast<uint32_t*>(&acc[29])), "=r"(*reinterpret_cast<uint32_t*>(&acc[30])), "=r"(*reinterpret_cast<uint32_t*>(&acc[31]))
                    : "r"(col_0 + 1048576));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                #pragma unroll
                for (int _lp = 0; _lp < 16; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc[_lp*2 + 0], acc[_lp*2+1 + 0]));
                    pk[_lp] = *(uint32_t*)&_bf2;
                }
                unsigned int slot_0 = warp_slots + 0;
                if (lane == 0) {
                    asm volatile("cp.async.bulk.wait_group.read 1;");
                }
                __syncwarp();
                uint32_t _stmatrix_addr_0 = static_cast<uint32_t>(slot_0 + st_row + 0);
                asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_0), "r"(*reinterpret_cast<const uint32_t*>(&pk[0])), "r"(*reinterpret_cast<const uint32_t*>(&pk[1])), "r"(*reinterpret_cast<const uint32_t*>(&pk[8])), "r"(*reinterpret_cast<const uint32_t*>(&pk[9]))
                    : "memory");
                uint32_t _stmatrix_addr_1 = static_cast<uint32_t>(slot_0 + st_row + 512);
                asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_1), "r"(*reinterpret_cast<const uint32_t*>(&pk[2])), "r"(*reinterpret_cast<const uint32_t*>(&pk[3])), "r"(*reinterpret_cast<const uint32_t*>(&pk[10])), "r"(*reinterpret_cast<const uint32_t*>(&pk[11]))
                    : "memory");
                uint32_t _stmatrix_addr_2 = static_cast<uint32_t>(slot_0 + st_row + 1024);
                asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_2), "r"(*reinterpret_cast<const uint32_t*>(&pk[4])), "r"(*reinterpret_cast<const uint32_t*>(&pk[5])), "r"(*reinterpret_cast<const uint32_t*>(&pk[12])), "r"(*reinterpret_cast<const uint32_t*>(&pk[13]))
                    : "memory");
                uint32_t _stmatrix_addr_3 = static_cast<uint32_t>(slot_0 + st_row + 1536);
                asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_3), "r"(*reinterpret_cast<const uint32_t*>(&pk[6])), "r"(*reinterpret_cast<const uint32_t*>(&pk[7])), "r"(*reinterpret_cast<const uint32_t*>(&pk[14])), "r"(*reinterpret_cast<const uint32_t*>(&pk[15]))
                    : "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                __syncwarp();
                if (lane == 0) {
                    tma_store_3d((&C), f_row, t_col, l_e, slot_0);
                    asm volatile("cp.async.bulk.commit_group;");
                }
                unsigned int col_1 = lane_addr + 32;
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&acc[0])), "=r"(*reinterpret_cast<uint32_t*>(&acc[1])), "=r"(*reinterpret_cast<uint32_t*>(&acc[2])), "=r"(*reinterpret_cast<uint32_t*>(&acc[3])), "=r"(*reinterpret_cast<uint32_t*>(&acc[4])), "=r"(*reinterpret_cast<uint32_t*>(&acc[5])), "=r"(*reinterpret_cast<uint32_t*>(&acc[6])), "=r"(*reinterpret_cast<uint32_t*>(&acc[7])), "=r"(*reinterpret_cast<uint32_t*>(&acc[8])), "=r"(*reinterpret_cast<uint32_t*>(&acc[9])), "=r"(*reinterpret_cast<uint32_t*>(&acc[10])), "=r"(*reinterpret_cast<uint32_t*>(&acc[11])), "=r"(*reinterpret_cast<uint32_t*>(&acc[12])), "=r"(*reinterpret_cast<uint32_t*>(&acc[13])), "=r"(*reinterpret_cast<uint32_t*>(&acc[14])), "=r"(*reinterpret_cast<uint32_t*>(&acc[15]))
                    : "r"(col_1 + 0));
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&acc[16])), "=r"(*reinterpret_cast<uint32_t*>(&acc[17])), "=r"(*reinterpret_cast<uint32_t*>(&acc[18])), "=r"(*reinterpret_cast<uint32_t*>(&acc[19])), "=r"(*reinterpret_cast<uint32_t*>(&acc[20])), "=r"(*reinterpret_cast<uint32_t*>(&acc[21])), "=r"(*reinterpret_cast<uint32_t*>(&acc[22])), "=r"(*reinterpret_cast<uint32_t*>(&acc[23])), "=r"(*reinterpret_cast<uint32_t*>(&acc[24])), "=r"(*reinterpret_cast<uint32_t*>(&acc[25])), "=r"(*reinterpret_cast<uint32_t*>(&acc[26])), "=r"(*reinterpret_cast<uint32_t*>(&acc[27])), "=r"(*reinterpret_cast<uint32_t*>(&acc[28])), "=r"(*reinterpret_cast<uint32_t*>(&acc[29])), "=r"(*reinterpret_cast<uint32_t*>(&acc[30])), "=r"(*reinterpret_cast<uint32_t*>(&acc[31]))
                    : "r"(col_1 + 1048576));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                #pragma unroll
                for (int _lp = 0; _lp < 16; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc[_lp*2 + 0], acc[_lp*2+1 + 0]));
                    pk[_lp] = *(uint32_t*)&_bf2;
                }
                unsigned int slot_1 = warp_slots + 2048;
                if (lane == 0) {
                    asm volatile("cp.async.bulk.wait_group.read 1;");
                }
                __syncwarp();
                uint32_t _stmatrix_addr_4 = static_cast<uint32_t>(slot_1 + st_row + 0);
                asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_4), "r"(*reinterpret_cast<const uint32_t*>(&pk[0])), "r"(*reinterpret_cast<const uint32_t*>(&pk[1])), "r"(*reinterpret_cast<const uint32_t*>(&pk[8])), "r"(*reinterpret_cast<const uint32_t*>(&pk[9]))
                    : "memory");
                uint32_t _stmatrix_addr_5 = static_cast<uint32_t>(slot_1 + st_row + 512);
                asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_5), "r"(*reinterpret_cast<const uint32_t*>(&pk[2])), "r"(*reinterpret_cast<const uint32_t*>(&pk[3])), "r"(*reinterpret_cast<const uint32_t*>(&pk[10])), "r"(*reinterpret_cast<const uint32_t*>(&pk[11]))
                    : "memory");
                uint32_t _stmatrix_addr_6 = static_cast<uint32_t>(slot_1 + st_row + 1024);
                asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_6), "r"(*reinterpret_cast<const uint32_t*>(&pk[4])), "r"(*reinterpret_cast<const uint32_t*>(&pk[5])), "r"(*reinterpret_cast<const uint32_t*>(&pk[12])), "r"(*reinterpret_cast<const uint32_t*>(&pk[13]))
                    : "memory");
                uint32_t _stmatrix_addr_7 = static_cast<uint32_t>(slot_1 + st_row + 1536);
                asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_7), "r"(*reinterpret_cast<const uint32_t*>(&pk[6])), "r"(*reinterpret_cast<const uint32_t*>(&pk[7])), "r"(*reinterpret_cast<const uint32_t*>(&pk[14])), "r"(*reinterpret_cast<const uint32_t*>(&pk[15]))
                    : "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                __syncwarp();
                if (lane == 0) {
                    tma_store_3d((&C), f_row, t_col + 32, l_e, slot_1);
                    asm volatile("cp.async.bulk.commit_group;");
                }
                unsigned int col_2 = lane_addr + 64;
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&acc[0])), "=r"(*reinterpret_cast<uint32_t*>(&acc[1])), "=r"(*reinterpret_cast<uint32_t*>(&acc[2])), "=r"(*reinterpret_cast<uint32_t*>(&acc[3])), "=r"(*reinterpret_cast<uint32_t*>(&acc[4])), "=r"(*reinterpret_cast<uint32_t*>(&acc[5])), "=r"(*reinterpret_cast<uint32_t*>(&acc[6])), "=r"(*reinterpret_cast<uint32_t*>(&acc[7])), "=r"(*reinterpret_cast<uint32_t*>(&acc[8])), "=r"(*reinterpret_cast<uint32_t*>(&acc[9])), "=r"(*reinterpret_cast<uint32_t*>(&acc[10])), "=r"(*reinterpret_cast<uint32_t*>(&acc[11])), "=r"(*reinterpret_cast<uint32_t*>(&acc[12])), "=r"(*reinterpret_cast<uint32_t*>(&acc[13])), "=r"(*reinterpret_cast<uint32_t*>(&acc[14])), "=r"(*reinterpret_cast<uint32_t*>(&acc[15]))
                    : "r"(col_2 + 0));
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&acc[16])), "=r"(*reinterpret_cast<uint32_t*>(&acc[17])), "=r"(*reinterpret_cast<uint32_t*>(&acc[18])), "=r"(*reinterpret_cast<uint32_t*>(&acc[19])), "=r"(*reinterpret_cast<uint32_t*>(&acc[20])), "=r"(*reinterpret_cast<uint32_t*>(&acc[21])), "=r"(*reinterpret_cast<uint32_t*>(&acc[22])), "=r"(*reinterpret_cast<uint32_t*>(&acc[23])), "=r"(*reinterpret_cast<uint32_t*>(&acc[24])), "=r"(*reinterpret_cast<uint32_t*>(&acc[25])), "=r"(*reinterpret_cast<uint32_t*>(&acc[26])), "=r"(*reinterpret_cast<uint32_t*>(&acc[27])), "=r"(*reinterpret_cast<uint32_t*>(&acc[28])), "=r"(*reinterpret_cast<uint32_t*>(&acc[29])), "=r"(*reinterpret_cast<uint32_t*>(&acc[30])), "=r"(*reinterpret_cast<uint32_t*>(&acc[31]))
                    : "r"(col_2 + 1048576));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                #pragma unroll
                for (int _lp = 0; _lp < 16; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc[_lp*2 + 0], acc[_lp*2+1 + 0]));
                    pk[_lp] = *(uint32_t*)&_bf2;
                }
                unsigned int slot_2 = warp_slots + 0;
                if (lane == 0) {
                    asm volatile("cp.async.bulk.wait_group.read 1;");
                }
                __syncwarp();
                uint32_t _stmatrix_addr_8 = static_cast<uint32_t>(slot_2 + st_row + 0);
                asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_8), "r"(*reinterpret_cast<const uint32_t*>(&pk[0])), "r"(*reinterpret_cast<const uint32_t*>(&pk[1])), "r"(*reinterpret_cast<const uint32_t*>(&pk[8])), "r"(*reinterpret_cast<const uint32_t*>(&pk[9]))
                    : "memory");
                uint32_t _stmatrix_addr_9 = static_cast<uint32_t>(slot_2 + st_row + 512);
                asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_9), "r"(*reinterpret_cast<const uint32_t*>(&pk[2])), "r"(*reinterpret_cast<const uint32_t*>(&pk[3])), "r"(*reinterpret_cast<const uint32_t*>(&pk[10])), "r"(*reinterpret_cast<const uint32_t*>(&pk[11]))
                    : "memory");
                uint32_t _stmatrix_addr_10 = static_cast<uint32_t>(slot_2 + st_row + 1024);
                asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_10), "r"(*reinterpret_cast<const uint32_t*>(&pk[4])), "r"(*reinterpret_cast<const uint32_t*>(&pk[5])), "r"(*reinterpret_cast<const uint32_t*>(&pk[12])), "r"(*reinterpret_cast<const uint32_t*>(&pk[13]))
                    : "memory");
                uint32_t _stmatrix_addr_11 = static_cast<uint32_t>(slot_2 + st_row + 1536);
                asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_11), "r"(*reinterpret_cast<const uint32_t*>(&pk[6])), "r"(*reinterpret_cast<const uint32_t*>(&pk[7])), "r"(*reinterpret_cast<const uint32_t*>(&pk[14])), "r"(*reinterpret_cast<const uint32_t*>(&pk[15]))
                    : "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                __syncwarp();
                if (lane == 0) {
                    tma_store_3d((&C), f_row, t_col + 64, l_e, slot_2);
                    asm volatile("cp.async.bulk.commit_group;");
                }
                unsigned int col_3 = lane_addr + 96;
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&acc[0])), "=r"(*reinterpret_cast<uint32_t*>(&acc[1])), "=r"(*reinterpret_cast<uint32_t*>(&acc[2])), "=r"(*reinterpret_cast<uint32_t*>(&acc[3])), "=r"(*reinterpret_cast<uint32_t*>(&acc[4])), "=r"(*reinterpret_cast<uint32_t*>(&acc[5])), "=r"(*reinterpret_cast<uint32_t*>(&acc[6])), "=r"(*reinterpret_cast<uint32_t*>(&acc[7])), "=r"(*reinterpret_cast<uint32_t*>(&acc[8])), "=r"(*reinterpret_cast<uint32_t*>(&acc[9])), "=r"(*reinterpret_cast<uint32_t*>(&acc[10])), "=r"(*reinterpret_cast<uint32_t*>(&acc[11])), "=r"(*reinterpret_cast<uint32_t*>(&acc[12])), "=r"(*reinterpret_cast<uint32_t*>(&acc[13])), "=r"(*reinterpret_cast<uint32_t*>(&acc[14])), "=r"(*reinterpret_cast<uint32_t*>(&acc[15]))
                    : "r"(col_3 + 0));
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&acc[16])), "=r"(*reinterpret_cast<uint32_t*>(&acc[17])), "=r"(*reinterpret_cast<uint32_t*>(&acc[18])), "=r"(*reinterpret_cast<uint32_t*>(&acc[19])), "=r"(*reinterpret_cast<uint32_t*>(&acc[20])), "=r"(*reinterpret_cast<uint32_t*>(&acc[21])), "=r"(*reinterpret_cast<uint32_t*>(&acc[22])), "=r"(*reinterpret_cast<uint32_t*>(&acc[23])), "=r"(*reinterpret_cast<uint32_t*>(&acc[24])), "=r"(*reinterpret_cast<uint32_t*>(&acc[25])), "=r"(*reinterpret_cast<uint32_t*>(&acc[26])), "=r"(*reinterpret_cast<uint32_t*>(&acc[27])), "=r"(*reinterpret_cast<uint32_t*>(&acc[28])), "=r"(*reinterpret_cast<uint32_t*>(&acc[29])), "=r"(*reinterpret_cast<uint32_t*>(&acc[30])), "=r"(*reinterpret_cast<uint32_t*>(&acc[31]))
                    : "r"(col_3 + 1048576));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                #pragma unroll
                for (int _lp = 0; _lp < 16; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc[_lp*2 + 0], acc[_lp*2+1 + 0]));
                    pk[_lp] = *(uint32_t*)&_bf2;
                }
                unsigned int slot_3 = warp_slots + 2048;
                if (lane == 0) {
                    asm volatile("cp.async.bulk.wait_group.read 1;");
                }
                __syncwarp();
                uint32_t _stmatrix_addr_12 = static_cast<uint32_t>(slot_3 + st_row + 0);
                asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_12), "r"(*reinterpret_cast<const uint32_t*>(&pk[0])), "r"(*reinterpret_cast<const uint32_t*>(&pk[1])), "r"(*reinterpret_cast<const uint32_t*>(&pk[8])), "r"(*reinterpret_cast<const uint32_t*>(&pk[9]))
                    : "memory");
                uint32_t _stmatrix_addr_13 = static_cast<uint32_t>(slot_3 + st_row + 512);
                asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_13), "r"(*reinterpret_cast<const uint32_t*>(&pk[2])), "r"(*reinterpret_cast<const uint32_t*>(&pk[3])), "r"(*reinterpret_cast<const uint32_t*>(&pk[10])), "r"(*reinterpret_cast<const uint32_t*>(&pk[11]))
                    : "memory");
                uint32_t _stmatrix_addr_14 = static_cast<uint32_t>(slot_3 + st_row + 1024);
                asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_14), "r"(*reinterpret_cast<const uint32_t*>(&pk[4])), "r"(*reinterpret_cast<const uint32_t*>(&pk[5])), "r"(*reinterpret_cast<const uint32_t*>(&pk[12])), "r"(*reinterpret_cast<const uint32_t*>(&pk[13]))
                    : "memory");
                uint32_t _stmatrix_addr_15 = static_cast<uint32_t>(slot_3 + st_row + 1536);
                asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_15), "r"(*reinterpret_cast<const uint32_t*>(&pk[6])), "r"(*reinterpret_cast<const uint32_t*>(&pk[7])), "r"(*reinterpret_cast<const uint32_t*>(&pk[14])), "r"(*reinterpret_cast<const uint32_t*>(&pk[15]))
                    : "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                __syncwarp();
                if (lane == 0) {
                    tma_store_3d((&C), f_row, t_col + 96, l_e, slot_3);
                    asm volatile("cp.async.bulk.commit_group;");
                }
                unsigned int col_4 = lane_addr + 128;
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&acc[0])), "=r"(*reinterpret_cast<uint32_t*>(&acc[1])), "=r"(*reinterpret_cast<uint32_t*>(&acc[2])), "=r"(*reinterpret_cast<uint32_t*>(&acc[3])), "=r"(*reinterpret_cast<uint32_t*>(&acc[4])), "=r"(*reinterpret_cast<uint32_t*>(&acc[5])), "=r"(*reinterpret_cast<uint32_t*>(&acc[6])), "=r"(*reinterpret_cast<uint32_t*>(&acc[7])), "=r"(*reinterpret_cast<uint32_t*>(&acc[8])), "=r"(*reinterpret_cast<uint32_t*>(&acc[9])), "=r"(*reinterpret_cast<uint32_t*>(&acc[10])), "=r"(*reinterpret_cast<uint32_t*>(&acc[11])), "=r"(*reinterpret_cast<uint32_t*>(&acc[12])), "=r"(*reinterpret_cast<uint32_t*>(&acc[13])), "=r"(*reinterpret_cast<uint32_t*>(&acc[14])), "=r"(*reinterpret_cast<uint32_t*>(&acc[15]))
                    : "r"(col_4 + 0));
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&acc[16])), "=r"(*reinterpret_cast<uint32_t*>(&acc[17])), "=r"(*reinterpret_cast<uint32_t*>(&acc[18])), "=r"(*reinterpret_cast<uint32_t*>(&acc[19])), "=r"(*reinterpret_cast<uint32_t*>(&acc[20])), "=r"(*reinterpret_cast<uint32_t*>(&acc[21])), "=r"(*reinterpret_cast<uint32_t*>(&acc[22])), "=r"(*reinterpret_cast<uint32_t*>(&acc[23])), "=r"(*reinterpret_cast<uint32_t*>(&acc[24])), "=r"(*reinterpret_cast<uint32_t*>(&acc[25])), "=r"(*reinterpret_cast<uint32_t*>(&acc[26])), "=r"(*reinterpret_cast<uint32_t*>(&acc[27])), "=r"(*reinterpret_cast<uint32_t*>(&acc[28])), "=r"(*reinterpret_cast<uint32_t*>(&acc[29])), "=r"(*reinterpret_cast<uint32_t*>(&acc[30])), "=r"(*reinterpret_cast<uint32_t*>(&acc[31]))
                    : "r"(col_4 + 1048576));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                #pragma unroll
                for (int _lp = 0; _lp < 16; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc[_lp*2 + 0], acc[_lp*2+1 + 0]));
                    pk[_lp] = *(uint32_t*)&_bf2;
                }
                unsigned int slot_4 = warp_slots + 0;
                if (lane == 0) {
                    asm volatile("cp.async.bulk.wait_group.read 1;");
                }
                __syncwarp();
                uint32_t _stmatrix_addr_16 = static_cast<uint32_t>(slot_4 + st_row + 0);
                asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_16), "r"(*reinterpret_cast<const uint32_t*>(&pk[0])), "r"(*reinterpret_cast<const uint32_t*>(&pk[1])), "r"(*reinterpret_cast<const uint32_t*>(&pk[8])), "r"(*reinterpret_cast<const uint32_t*>(&pk[9]))
                    : "memory");
                uint32_t _stmatrix_addr_17 = static_cast<uint32_t>(slot_4 + st_row + 512);
                asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_17), "r"(*reinterpret_cast<const uint32_t*>(&pk[2])), "r"(*reinterpret_cast<const uint32_t*>(&pk[3])), "r"(*reinterpret_cast<const uint32_t*>(&pk[10])), "r"(*reinterpret_cast<const uint32_t*>(&pk[11]))
                    : "memory");
                uint32_t _stmatrix_addr_18 = static_cast<uint32_t>(slot_4 + st_row + 1024);
                asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_18), "r"(*reinterpret_cast<const uint32_t*>(&pk[4])), "r"(*reinterpret_cast<const uint32_t*>(&pk[5])), "r"(*reinterpret_cast<const uint32_t*>(&pk[12])), "r"(*reinterpret_cast<const uint32_t*>(&pk[13]))
                    : "memory");
                uint32_t _stmatrix_addr_19 = static_cast<uint32_t>(slot_4 + st_row + 1536);
                asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_19), "r"(*reinterpret_cast<const uint32_t*>(&pk[6])), "r"(*reinterpret_cast<const uint32_t*>(&pk[7])), "r"(*reinterpret_cast<const uint32_t*>(&pk[14])), "r"(*reinterpret_cast<const uint32_t*>(&pk[15]))
                    : "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                __syncwarp();
                if (lane == 0) {
                    tma_store_3d((&C), f_row, t_col + 128, l_e, slot_4);
                    asm volatile("cp.async.bulk.commit_group;");
                }
                unsigned int col_5 = lane_addr + 160;
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&acc[0])), "=r"(*reinterpret_cast<uint32_t*>(&acc[1])), "=r"(*reinterpret_cast<uint32_t*>(&acc[2])), "=r"(*reinterpret_cast<uint32_t*>(&acc[3])), "=r"(*reinterpret_cast<uint32_t*>(&acc[4])), "=r"(*reinterpret_cast<uint32_t*>(&acc[5])), "=r"(*reinterpret_cast<uint32_t*>(&acc[6])), "=r"(*reinterpret_cast<uint32_t*>(&acc[7])), "=r"(*reinterpret_cast<uint32_t*>(&acc[8])), "=r"(*reinterpret_cast<uint32_t*>(&acc[9])), "=r"(*reinterpret_cast<uint32_t*>(&acc[10])), "=r"(*reinterpret_cast<uint32_t*>(&acc[11])), "=r"(*reinterpret_cast<uint32_t*>(&acc[12])), "=r"(*reinterpret_cast<uint32_t*>(&acc[13])), "=r"(*reinterpret_cast<uint32_t*>(&acc[14])), "=r"(*reinterpret_cast<uint32_t*>(&acc[15]))
                    : "r"(col_5 + 0));
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&acc[16])), "=r"(*reinterpret_cast<uint32_t*>(&acc[17])), "=r"(*reinterpret_cast<uint32_t*>(&acc[18])), "=r"(*reinterpret_cast<uint32_t*>(&acc[19])), "=r"(*reinterpret_cast<uint32_t*>(&acc[20])), "=r"(*reinterpret_cast<uint32_t*>(&acc[21])), "=r"(*reinterpret_cast<uint32_t*>(&acc[22])), "=r"(*reinterpret_cast<uint32_t*>(&acc[23])), "=r"(*reinterpret_cast<uint32_t*>(&acc[24])), "=r"(*reinterpret_cast<uint32_t*>(&acc[25])), "=r"(*reinterpret_cast<uint32_t*>(&acc[26])), "=r"(*reinterpret_cast<uint32_t*>(&acc[27])), "=r"(*reinterpret_cast<uint32_t*>(&acc[28])), "=r"(*reinterpret_cast<uint32_t*>(&acc[29])), "=r"(*reinterpret_cast<uint32_t*>(&acc[30])), "=r"(*reinterpret_cast<uint32_t*>(&acc[31]))
                    : "r"(col_5 + 1048576));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                #pragma unroll
                for (int _lp = 0; _lp < 16; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc[_lp*2 + 0], acc[_lp*2+1 + 0]));
                    pk[_lp] = *(uint32_t*)&_bf2;
                }
                unsigned int slot_5 = warp_slots + 2048;
                if (lane == 0) {
                    asm volatile("cp.async.bulk.wait_group.read 1;");
                }
                __syncwarp();
                uint32_t _stmatrix_addr_20 = static_cast<uint32_t>(slot_5 + st_row + 0);
                asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_20), "r"(*reinterpret_cast<const uint32_t*>(&pk[0])), "r"(*reinterpret_cast<const uint32_t*>(&pk[1])), "r"(*reinterpret_cast<const uint32_t*>(&pk[8])), "r"(*reinterpret_cast<const uint32_t*>(&pk[9]))
                    : "memory");
                uint32_t _stmatrix_addr_21 = static_cast<uint32_t>(slot_5 + st_row + 512);
                asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_21), "r"(*reinterpret_cast<const uint32_t*>(&pk[2])), "r"(*reinterpret_cast<const uint32_t*>(&pk[3])), "r"(*reinterpret_cast<const uint32_t*>(&pk[10])), "r"(*reinterpret_cast<const uint32_t*>(&pk[11]))
                    : "memory");
                uint32_t _stmatrix_addr_22 = static_cast<uint32_t>(slot_5 + st_row + 1024);
                asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_22), "r"(*reinterpret_cast<const uint32_t*>(&pk[4])), "r"(*reinterpret_cast<const uint32_t*>(&pk[5])), "r"(*reinterpret_cast<const uint32_t*>(&pk[12])), "r"(*reinterpret_cast<const uint32_t*>(&pk[13]))
                    : "memory");
                uint32_t _stmatrix_addr_23 = static_cast<uint32_t>(slot_5 + st_row + 1536);
                asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_23), "r"(*reinterpret_cast<const uint32_t*>(&pk[6])), "r"(*reinterpret_cast<const uint32_t*>(&pk[7])), "r"(*reinterpret_cast<const uint32_t*>(&pk[14])), "r"(*reinterpret_cast<const uint32_t*>(&pk[15]))
                    : "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                __syncwarp();
                if (lane == 0) {
                    tma_store_3d((&C), f_row, t_col + 160, l_e, slot_5);
                    asm volatile("cp.async.bulk.commit_group;");
                }
                unsigned int col_6 = lane_addr + 192;
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&acc[0])), "=r"(*reinterpret_cast<uint32_t*>(&acc[1])), "=r"(*reinterpret_cast<uint32_t*>(&acc[2])), "=r"(*reinterpret_cast<uint32_t*>(&acc[3])), "=r"(*reinterpret_cast<uint32_t*>(&acc[4])), "=r"(*reinterpret_cast<uint32_t*>(&acc[5])), "=r"(*reinterpret_cast<uint32_t*>(&acc[6])), "=r"(*reinterpret_cast<uint32_t*>(&acc[7])), "=r"(*reinterpret_cast<uint32_t*>(&acc[8])), "=r"(*reinterpret_cast<uint32_t*>(&acc[9])), "=r"(*reinterpret_cast<uint32_t*>(&acc[10])), "=r"(*reinterpret_cast<uint32_t*>(&acc[11])), "=r"(*reinterpret_cast<uint32_t*>(&acc[12])), "=r"(*reinterpret_cast<uint32_t*>(&acc[13])), "=r"(*reinterpret_cast<uint32_t*>(&acc[14])), "=r"(*reinterpret_cast<uint32_t*>(&acc[15]))
                    : "r"(col_6 + 0));
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&acc[16])), "=r"(*reinterpret_cast<uint32_t*>(&acc[17])), "=r"(*reinterpret_cast<uint32_t*>(&acc[18])), "=r"(*reinterpret_cast<uint32_t*>(&acc[19])), "=r"(*reinterpret_cast<uint32_t*>(&acc[20])), "=r"(*reinterpret_cast<uint32_t*>(&acc[21])), "=r"(*reinterpret_cast<uint32_t*>(&acc[22])), "=r"(*reinterpret_cast<uint32_t*>(&acc[23])), "=r"(*reinterpret_cast<uint32_t*>(&acc[24])), "=r"(*reinterpret_cast<uint32_t*>(&acc[25])), "=r"(*reinterpret_cast<uint32_t*>(&acc[26])), "=r"(*reinterpret_cast<uint32_t*>(&acc[27])), "=r"(*reinterpret_cast<uint32_t*>(&acc[28])), "=r"(*reinterpret_cast<uint32_t*>(&acc[29])), "=r"(*reinterpret_cast<uint32_t*>(&acc[30])), "=r"(*reinterpret_cast<uint32_t*>(&acc[31]))
                    : "r"(col_6 + 1048576));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                #pragma unroll
                for (int _lp = 0; _lp < 16; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc[_lp*2 + 0], acc[_lp*2+1 + 0]));
                    pk[_lp] = *(uint32_t*)&_bf2;
                }
                unsigned int slot_6 = warp_slots + 0;
                if (lane == 0) {
                    asm volatile("cp.async.bulk.wait_group.read 1;");
                }
                __syncwarp();
                uint32_t _stmatrix_addr_24 = static_cast<uint32_t>(slot_6 + st_row + 0);
                asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_24), "r"(*reinterpret_cast<const uint32_t*>(&pk[0])), "r"(*reinterpret_cast<const uint32_t*>(&pk[1])), "r"(*reinterpret_cast<const uint32_t*>(&pk[8])), "r"(*reinterpret_cast<const uint32_t*>(&pk[9]))
                    : "memory");
                uint32_t _stmatrix_addr_25 = static_cast<uint32_t>(slot_6 + st_row + 512);
                asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_25), "r"(*reinterpret_cast<const uint32_t*>(&pk[2])), "r"(*reinterpret_cast<const uint32_t*>(&pk[3])), "r"(*reinterpret_cast<const uint32_t*>(&pk[10])), "r"(*reinterpret_cast<const uint32_t*>(&pk[11]))
                    : "memory");
                uint32_t _stmatrix_addr_26 = static_cast<uint32_t>(slot_6 + st_row + 1024);
                asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_26), "r"(*reinterpret_cast<const uint32_t*>(&pk[4])), "r"(*reinterpret_cast<const uint32_t*>(&pk[5])), "r"(*reinterpret_cast<const uint32_t*>(&pk[12])), "r"(*reinterpret_cast<const uint32_t*>(&pk[13]))
                    : "memory");
                uint32_t _stmatrix_addr_27 = static_cast<uint32_t>(slot_6 + st_row + 1536);
                asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_27), "r"(*reinterpret_cast<const uint32_t*>(&pk[6])), "r"(*reinterpret_cast<const uint32_t*>(&pk[7])), "r"(*reinterpret_cast<const uint32_t*>(&pk[14])), "r"(*reinterpret_cast<const uint32_t*>(&pk[15]))
                    : "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                __syncwarp();
                if (lane == 0) {
                    tma_store_3d((&C), f_row, t_col + 192, l_e, slot_6);
                    asm volatile("cp.async.bulk.commit_group;");
                }
                unsigned int col_7 = lane_addr + 256;
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&acc[0])), "=r"(*reinterpret_cast<uint32_t*>(&acc[1])), "=r"(*reinterpret_cast<uint32_t*>(&acc[2])), "=r"(*reinterpret_cast<uint32_t*>(&acc[3])), "=r"(*reinterpret_cast<uint32_t*>(&acc[4])), "=r"(*reinterpret_cast<uint32_t*>(&acc[5])), "=r"(*reinterpret_cast<uint32_t*>(&acc[6])), "=r"(*reinterpret_cast<uint32_t*>(&acc[7])), "=r"(*reinterpret_cast<uint32_t*>(&acc[8])), "=r"(*reinterpret_cast<uint32_t*>(&acc[9])), "=r"(*reinterpret_cast<uint32_t*>(&acc[10])), "=r"(*reinterpret_cast<uint32_t*>(&acc[11])), "=r"(*reinterpret_cast<uint32_t*>(&acc[12])), "=r"(*reinterpret_cast<uint32_t*>(&acc[13])), "=r"(*reinterpret_cast<uint32_t*>(&acc[14])), "=r"(*reinterpret_cast<uint32_t*>(&acc[15]))
                    : "r"(col_7 + 0));
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&acc[16])), "=r"(*reinterpret_cast<uint32_t*>(&acc[17])), "=r"(*reinterpret_cast<uint32_t*>(&acc[18])), "=r"(*reinterpret_cast<uint32_t*>(&acc[19])), "=r"(*reinterpret_cast<uint32_t*>(&acc[20])), "=r"(*reinterpret_cast<uint32_t*>(&acc[21])), "=r"(*reinterpret_cast<uint32_t*>(&acc[22])), "=r"(*reinterpret_cast<uint32_t*>(&acc[23])), "=r"(*reinterpret_cast<uint32_t*>(&acc[24])), "=r"(*reinterpret_cast<uint32_t*>(&acc[25])), "=r"(*reinterpret_cast<uint32_t*>(&acc[26])), "=r"(*reinterpret_cast<uint32_t*>(&acc[27])), "=r"(*reinterpret_cast<uint32_t*>(&acc[28])), "=r"(*reinterpret_cast<uint32_t*>(&acc[29])), "=r"(*reinterpret_cast<uint32_t*>(&acc[30])), "=r"(*reinterpret_cast<uint32_t*>(&acc[31]))
                    : "r"(col_7 + 1048576));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                #pragma unroll
                for (int _lp = 0; _lp < 16; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc[_lp*2 + 0], acc[_lp*2+1 + 0]));
                    pk[_lp] = *(uint32_t*)&_bf2;
                }
                unsigned int slot_7 = warp_slots + 2048;
                if (lane == 0) {
                    asm volatile("cp.async.bulk.wait_group.read 1;");
                }
                __syncwarp();
                uint32_t _stmatrix_addr_28 = static_cast<uint32_t>(slot_7 + st_row + 0);
                asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_28), "r"(*reinterpret_cast<const uint32_t*>(&pk[0])), "r"(*reinterpret_cast<const uint32_t*>(&pk[1])), "r"(*reinterpret_cast<const uint32_t*>(&pk[8])), "r"(*reinterpret_cast<const uint32_t*>(&pk[9]))
                    : "memory");
                uint32_t _stmatrix_addr_29 = static_cast<uint32_t>(slot_7 + st_row + 512);
                asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_29), "r"(*reinterpret_cast<const uint32_t*>(&pk[2])), "r"(*reinterpret_cast<const uint32_t*>(&pk[3])), "r"(*reinterpret_cast<const uint32_t*>(&pk[10])), "r"(*reinterpret_cast<const uint32_t*>(&pk[11]))
                    : "memory");
                uint32_t _stmatrix_addr_30 = static_cast<uint32_t>(slot_7 + st_row + 1024);
                asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_30), "r"(*reinterpret_cast<const uint32_t*>(&pk[4])), "r"(*reinterpret_cast<const uint32_t*>(&pk[5])), "r"(*reinterpret_cast<const uint32_t*>(&pk[12])), "r"(*reinterpret_cast<const uint32_t*>(&pk[13]))
                    : "memory");
                uint32_t _stmatrix_addr_31 = static_cast<uint32_t>(slot_7 + st_row + 1536);
                asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_31), "r"(*reinterpret_cast<const uint32_t*>(&pk[6])), "r"(*reinterpret_cast<const uint32_t*>(&pk[7])), "r"(*reinterpret_cast<const uint32_t*>(&pk[14])), "r"(*reinterpret_cast<const uint32_t*>(&pk[15]))
                    : "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                __syncwarp();
                if (lane == 0) {
                    tma_store_3d((&C), f_row + 128, t_col, l_e, slot_7);
                    asm volatile("cp.async.bulk.commit_group;");
                }
                unsigned int col_8 = lane_addr + 288;
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&acc[0])), "=r"(*reinterpret_cast<uint32_t*>(&acc[1])), "=r"(*reinterpret_cast<uint32_t*>(&acc[2])), "=r"(*reinterpret_cast<uint32_t*>(&acc[3])), "=r"(*reinterpret_cast<uint32_t*>(&acc[4])), "=r"(*reinterpret_cast<uint32_t*>(&acc[5])), "=r"(*reinterpret_cast<uint32_t*>(&acc[6])), "=r"(*reinterpret_cast<uint32_t*>(&acc[7])), "=r"(*reinterpret_cast<uint32_t*>(&acc[8])), "=r"(*reinterpret_cast<uint32_t*>(&acc[9])), "=r"(*reinterpret_cast<uint32_t*>(&acc[10])), "=r"(*reinterpret_cast<uint32_t*>(&acc[11])), "=r"(*reinterpret_cast<uint32_t*>(&acc[12])), "=r"(*reinterpret_cast<uint32_t*>(&acc[13])), "=r"(*reinterpret_cast<uint32_t*>(&acc[14])), "=r"(*reinterpret_cast<uint32_t*>(&acc[15]))
                    : "r"(col_8 + 0));
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&acc[16])), "=r"(*reinterpret_cast<uint32_t*>(&acc[17])), "=r"(*reinterpret_cast<uint32_t*>(&acc[18])), "=r"(*reinterpret_cast<uint32_t*>(&acc[19])), "=r"(*reinterpret_cast<uint32_t*>(&acc[20])), "=r"(*reinterpret_cast<uint32_t*>(&acc[21])), "=r"(*reinterpret_cast<uint32_t*>(&acc[22])), "=r"(*reinterpret_cast<uint32_t*>(&acc[23])), "=r"(*reinterpret_cast<uint32_t*>(&acc[24])), "=r"(*reinterpret_cast<uint32_t*>(&acc[25])), "=r"(*reinterpret_cast<uint32_t*>(&acc[26])), "=r"(*reinterpret_cast<uint32_t*>(&acc[27])), "=r"(*reinterpret_cast<uint32_t*>(&acc[28])), "=r"(*reinterpret_cast<uint32_t*>(&acc[29])), "=r"(*reinterpret_cast<uint32_t*>(&acc[30])), "=r"(*reinterpret_cast<uint32_t*>(&acc[31]))
                    : "r"(col_8 + 1048576));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                #pragma unroll
                for (int _lp = 0; _lp < 16; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc[_lp*2 + 0], acc[_lp*2+1 + 0]));
                    pk[_lp] = *(uint32_t*)&_bf2;
                }
                unsigned int slot_8 = warp_slots + 0;
                if (lane == 0) {
                    asm volatile("cp.async.bulk.wait_group.read 1;");
                }
                __syncwarp();
                uint32_t _stmatrix_addr_32 = static_cast<uint32_t>(slot_8 + st_row + 0);
                asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_32), "r"(*reinterpret_cast<const uint32_t*>(&pk[0])), "r"(*reinterpret_cast<const uint32_t*>(&pk[1])), "r"(*reinterpret_cast<const uint32_t*>(&pk[8])), "r"(*reinterpret_cast<const uint32_t*>(&pk[9]))
                    : "memory");
                uint32_t _stmatrix_addr_33 = static_cast<uint32_t>(slot_8 + st_row + 512);
                asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_33), "r"(*reinterpret_cast<const uint32_t*>(&pk[2])), "r"(*reinterpret_cast<const uint32_t*>(&pk[3])), "r"(*reinterpret_cast<const uint32_t*>(&pk[10])), "r"(*reinterpret_cast<const uint32_t*>(&pk[11]))
                    : "memory");
                uint32_t _stmatrix_addr_34 = static_cast<uint32_t>(slot_8 + st_row + 1024);
                asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_34), "r"(*reinterpret_cast<const uint32_t*>(&pk[4])), "r"(*reinterpret_cast<const uint32_t*>(&pk[5])), "r"(*reinterpret_cast<const uint32_t*>(&pk[12])), "r"(*reinterpret_cast<const uint32_t*>(&pk[13]))
                    : "memory");
                uint32_t _stmatrix_addr_35 = static_cast<uint32_t>(slot_8 + st_row + 1536);
                asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_35), "r"(*reinterpret_cast<const uint32_t*>(&pk[6])), "r"(*reinterpret_cast<const uint32_t*>(&pk[7])), "r"(*reinterpret_cast<const uint32_t*>(&pk[14])), "r"(*reinterpret_cast<const uint32_t*>(&pk[15]))
                    : "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                __syncwarp();
                if (lane == 0) {
                    tma_store_3d((&C), f_row + 128, t_col + 32, l_e, slot_8);
                    asm volatile("cp.async.bulk.commit_group;");
                }
                unsigned int col_9 = lane_addr + 320;
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&acc[0])), "=r"(*reinterpret_cast<uint32_t*>(&acc[1])), "=r"(*reinterpret_cast<uint32_t*>(&acc[2])), "=r"(*reinterpret_cast<uint32_t*>(&acc[3])), "=r"(*reinterpret_cast<uint32_t*>(&acc[4])), "=r"(*reinterpret_cast<uint32_t*>(&acc[5])), "=r"(*reinterpret_cast<uint32_t*>(&acc[6])), "=r"(*reinterpret_cast<uint32_t*>(&acc[7])), "=r"(*reinterpret_cast<uint32_t*>(&acc[8])), "=r"(*reinterpret_cast<uint32_t*>(&acc[9])), "=r"(*reinterpret_cast<uint32_t*>(&acc[10])), "=r"(*reinterpret_cast<uint32_t*>(&acc[11])), "=r"(*reinterpret_cast<uint32_t*>(&acc[12])), "=r"(*reinterpret_cast<uint32_t*>(&acc[13])), "=r"(*reinterpret_cast<uint32_t*>(&acc[14])), "=r"(*reinterpret_cast<uint32_t*>(&acc[15]))
                    : "r"(col_9 + 0));
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&acc[16])), "=r"(*reinterpret_cast<uint32_t*>(&acc[17])), "=r"(*reinterpret_cast<uint32_t*>(&acc[18])), "=r"(*reinterpret_cast<uint32_t*>(&acc[19])), "=r"(*reinterpret_cast<uint32_t*>(&acc[20])), "=r"(*reinterpret_cast<uint32_t*>(&acc[21])), "=r"(*reinterpret_cast<uint32_t*>(&acc[22])), "=r"(*reinterpret_cast<uint32_t*>(&acc[23])), "=r"(*reinterpret_cast<uint32_t*>(&acc[24])), "=r"(*reinterpret_cast<uint32_t*>(&acc[25])), "=r"(*reinterpret_cast<uint32_t*>(&acc[26])), "=r"(*reinterpret_cast<uint32_t*>(&acc[27])), "=r"(*reinterpret_cast<uint32_t*>(&acc[28])), "=r"(*reinterpret_cast<uint32_t*>(&acc[29])), "=r"(*reinterpret_cast<uint32_t*>(&acc[30])), "=r"(*reinterpret_cast<uint32_t*>(&acc[31]))
                    : "r"(col_9 + 1048576));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                #pragma unroll
                for (int _lp = 0; _lp < 16; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc[_lp*2 + 0], acc[_lp*2+1 + 0]));
                    pk[_lp] = *(uint32_t*)&_bf2;
                }
                unsigned int slot_9 = warp_slots + 2048;
                if (lane == 0) {
                    asm volatile("cp.async.bulk.wait_group.read 1;");
                }
                __syncwarp();
                uint32_t _stmatrix_addr_36 = static_cast<uint32_t>(slot_9 + st_row + 0);
                asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_36), "r"(*reinterpret_cast<const uint32_t*>(&pk[0])), "r"(*reinterpret_cast<const uint32_t*>(&pk[1])), "r"(*reinterpret_cast<const uint32_t*>(&pk[8])), "r"(*reinterpret_cast<const uint32_t*>(&pk[9]))
                    : "memory");
                uint32_t _stmatrix_addr_37 = static_cast<uint32_t>(slot_9 + st_row + 512);
                asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_37), "r"(*reinterpret_cast<const uint32_t*>(&pk[2])), "r"(*reinterpret_cast<const uint32_t*>(&pk[3])), "r"(*reinterpret_cast<const uint32_t*>(&pk[10])), "r"(*reinterpret_cast<const uint32_t*>(&pk[11]))
                    : "memory");
                uint32_t _stmatrix_addr_38 = static_cast<uint32_t>(slot_9 + st_row + 1024);
                asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_38), "r"(*reinterpret_cast<const uint32_t*>(&pk[4])), "r"(*reinterpret_cast<const uint32_t*>(&pk[5])), "r"(*reinterpret_cast<const uint32_t*>(&pk[12])), "r"(*reinterpret_cast<const uint32_t*>(&pk[13]))
                    : "memory");
                uint32_t _stmatrix_addr_39 = static_cast<uint32_t>(slot_9 + st_row + 1536);
                asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_39), "r"(*reinterpret_cast<const uint32_t*>(&pk[6])), "r"(*reinterpret_cast<const uint32_t*>(&pk[7])), "r"(*reinterpret_cast<const uint32_t*>(&pk[14])), "r"(*reinterpret_cast<const uint32_t*>(&pk[15]))
                    : "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                __syncwarp();
                if (lane == 0) {
                    tma_store_3d((&C), f_row + 128, t_col + 64, l_e, slot_9);
                    asm volatile("cp.async.bulk.commit_group;");
                }
                unsigned int col_10 = lane_addr + 352;
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&acc[0])), "=r"(*reinterpret_cast<uint32_t*>(&acc[1])), "=r"(*reinterpret_cast<uint32_t*>(&acc[2])), "=r"(*reinterpret_cast<uint32_t*>(&acc[3])), "=r"(*reinterpret_cast<uint32_t*>(&acc[4])), "=r"(*reinterpret_cast<uint32_t*>(&acc[5])), "=r"(*reinterpret_cast<uint32_t*>(&acc[6])), "=r"(*reinterpret_cast<uint32_t*>(&acc[7])), "=r"(*reinterpret_cast<uint32_t*>(&acc[8])), "=r"(*reinterpret_cast<uint32_t*>(&acc[9])), "=r"(*reinterpret_cast<uint32_t*>(&acc[10])), "=r"(*reinterpret_cast<uint32_t*>(&acc[11])), "=r"(*reinterpret_cast<uint32_t*>(&acc[12])), "=r"(*reinterpret_cast<uint32_t*>(&acc[13])), "=r"(*reinterpret_cast<uint32_t*>(&acc[14])), "=r"(*reinterpret_cast<uint32_t*>(&acc[15]))
                    : "r"(col_10 + 0));
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&acc[16])), "=r"(*reinterpret_cast<uint32_t*>(&acc[17])), "=r"(*reinterpret_cast<uint32_t*>(&acc[18])), "=r"(*reinterpret_cast<uint32_t*>(&acc[19])), "=r"(*reinterpret_cast<uint32_t*>(&acc[20])), "=r"(*reinterpret_cast<uint32_t*>(&acc[21])), "=r"(*reinterpret_cast<uint32_t*>(&acc[22])), "=r"(*reinterpret_cast<uint32_t*>(&acc[23])), "=r"(*reinterpret_cast<uint32_t*>(&acc[24])), "=r"(*reinterpret_cast<uint32_t*>(&acc[25])), "=r"(*reinterpret_cast<uint32_t*>(&acc[26])), "=r"(*reinterpret_cast<uint32_t*>(&acc[27])), "=r"(*reinterpret_cast<uint32_t*>(&acc[28])), "=r"(*reinterpret_cast<uint32_t*>(&acc[29])), "=r"(*reinterpret_cast<uint32_t*>(&acc[30])), "=r"(*reinterpret_cast<uint32_t*>(&acc[31]))
                    : "r"(col_10 + 1048576));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                #pragma unroll
                for (int _lp = 0; _lp < 16; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc[_lp*2 + 0], acc[_lp*2+1 + 0]));
                    pk[_lp] = *(uint32_t*)&_bf2;
                }
                unsigned int slot_10 = warp_slots + 0;
                if (lane == 0) {
                    asm volatile("cp.async.bulk.wait_group.read 1;");
                }
                __syncwarp();
                uint32_t _stmatrix_addr_40 = static_cast<uint32_t>(slot_10 + st_row + 0);
                asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_40), "r"(*reinterpret_cast<const uint32_t*>(&pk[0])), "r"(*reinterpret_cast<const uint32_t*>(&pk[1])), "r"(*reinterpret_cast<const uint32_t*>(&pk[8])), "r"(*reinterpret_cast<const uint32_t*>(&pk[9]))
                    : "memory");
                uint32_t _stmatrix_addr_41 = static_cast<uint32_t>(slot_10 + st_row + 512);
                asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_41), "r"(*reinterpret_cast<const uint32_t*>(&pk[2])), "r"(*reinterpret_cast<const uint32_t*>(&pk[3])), "r"(*reinterpret_cast<const uint32_t*>(&pk[10])), "r"(*reinterpret_cast<const uint32_t*>(&pk[11]))
                    : "memory");
                uint32_t _stmatrix_addr_42 = static_cast<uint32_t>(slot_10 + st_row + 1024);
                asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_42), "r"(*reinterpret_cast<const uint32_t*>(&pk[4])), "r"(*reinterpret_cast<const uint32_t*>(&pk[5])), "r"(*reinterpret_cast<const uint32_t*>(&pk[12])), "r"(*reinterpret_cast<const uint32_t*>(&pk[13]))
                    : "memory");
                uint32_t _stmatrix_addr_43 = static_cast<uint32_t>(slot_10 + st_row + 1536);
                asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_43), "r"(*reinterpret_cast<const uint32_t*>(&pk[6])), "r"(*reinterpret_cast<const uint32_t*>(&pk[7])), "r"(*reinterpret_cast<const uint32_t*>(&pk[14])), "r"(*reinterpret_cast<const uint32_t*>(&pk[15]))
                    : "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                __syncwarp();
                if (lane == 0) {
                    tma_store_3d((&C), f_row + 128, t_col + 96, l_e, slot_10);
                    asm volatile("cp.async.bulk.commit_group;");
                }
                unsigned int col_11 = lane_addr + 384;
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&acc[0])), "=r"(*reinterpret_cast<uint32_t*>(&acc[1])), "=r"(*reinterpret_cast<uint32_t*>(&acc[2])), "=r"(*reinterpret_cast<uint32_t*>(&acc[3])), "=r"(*reinterpret_cast<uint32_t*>(&acc[4])), "=r"(*reinterpret_cast<uint32_t*>(&acc[5])), "=r"(*reinterpret_cast<uint32_t*>(&acc[6])), "=r"(*reinterpret_cast<uint32_t*>(&acc[7])), "=r"(*reinterpret_cast<uint32_t*>(&acc[8])), "=r"(*reinterpret_cast<uint32_t*>(&acc[9])), "=r"(*reinterpret_cast<uint32_t*>(&acc[10])), "=r"(*reinterpret_cast<uint32_t*>(&acc[11])), "=r"(*reinterpret_cast<uint32_t*>(&acc[12])), "=r"(*reinterpret_cast<uint32_t*>(&acc[13])), "=r"(*reinterpret_cast<uint32_t*>(&acc[14])), "=r"(*reinterpret_cast<uint32_t*>(&acc[15]))
                    : "r"(col_11 + 0));
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&acc[16])), "=r"(*reinterpret_cast<uint32_t*>(&acc[17])), "=r"(*reinterpret_cast<uint32_t*>(&acc[18])), "=r"(*reinterpret_cast<uint32_t*>(&acc[19])), "=r"(*reinterpret_cast<uint32_t*>(&acc[20])), "=r"(*reinterpret_cast<uint32_t*>(&acc[21])), "=r"(*reinterpret_cast<uint32_t*>(&acc[22])), "=r"(*reinterpret_cast<uint32_t*>(&acc[23])), "=r"(*reinterpret_cast<uint32_t*>(&acc[24])), "=r"(*reinterpret_cast<uint32_t*>(&acc[25])), "=r"(*reinterpret_cast<uint32_t*>(&acc[26])), "=r"(*reinterpret_cast<uint32_t*>(&acc[27])), "=r"(*reinterpret_cast<uint32_t*>(&acc[28])), "=r"(*reinterpret_cast<uint32_t*>(&acc[29])), "=r"(*reinterpret_cast<uint32_t*>(&acc[30])), "=r"(*reinterpret_cast<uint32_t*>(&acc[31]))
                    : "r"(col_11 + 1048576));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                #pragma unroll
                for (int _lp = 0; _lp < 16; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc[_lp*2 + 0], acc[_lp*2+1 + 0]));
                    pk[_lp] = *(uint32_t*)&_bf2;
                }
                unsigned int slot_11 = warp_slots + 2048;
                if (lane == 0) {
                    asm volatile("cp.async.bulk.wait_group.read 1;");
                }
                __syncwarp();
                uint32_t _stmatrix_addr_44 = static_cast<uint32_t>(slot_11 + st_row + 0);
                asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_44), "r"(*reinterpret_cast<const uint32_t*>(&pk[0])), "r"(*reinterpret_cast<const uint32_t*>(&pk[1])), "r"(*reinterpret_cast<const uint32_t*>(&pk[8])), "r"(*reinterpret_cast<const uint32_t*>(&pk[9]))
                    : "memory");
                uint32_t _stmatrix_addr_45 = static_cast<uint32_t>(slot_11 + st_row + 512);
                asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_45), "r"(*reinterpret_cast<const uint32_t*>(&pk[2])), "r"(*reinterpret_cast<const uint32_t*>(&pk[3])), "r"(*reinterpret_cast<const uint32_t*>(&pk[10])), "r"(*reinterpret_cast<const uint32_t*>(&pk[11]))
                    : "memory");
                uint32_t _stmatrix_addr_46 = static_cast<uint32_t>(slot_11 + st_row + 1024);
                asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_46), "r"(*reinterpret_cast<const uint32_t*>(&pk[4])), "r"(*reinterpret_cast<const uint32_t*>(&pk[5])), "r"(*reinterpret_cast<const uint32_t*>(&pk[12])), "r"(*reinterpret_cast<const uint32_t*>(&pk[13]))
                    : "memory");
                uint32_t _stmatrix_addr_47 = static_cast<uint32_t>(slot_11 + st_row + 1536);
                asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_47), "r"(*reinterpret_cast<const uint32_t*>(&pk[6])), "r"(*reinterpret_cast<const uint32_t*>(&pk[7])), "r"(*reinterpret_cast<const uint32_t*>(&pk[14])), "r"(*reinterpret_cast<const uint32_t*>(&pk[15]))
                    : "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                __syncwarp();
                if (lane == 0) {
                    tma_store_3d((&C), f_row + 128, t_col + 128, l_e, slot_11);
                    asm volatile("cp.async.bulk.commit_group;");
                }
                unsigned int col_12 = lane_addr + 416;
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&acc[0])), "=r"(*reinterpret_cast<uint32_t*>(&acc[1])), "=r"(*reinterpret_cast<uint32_t*>(&acc[2])), "=r"(*reinterpret_cast<uint32_t*>(&acc[3])), "=r"(*reinterpret_cast<uint32_t*>(&acc[4])), "=r"(*reinterpret_cast<uint32_t*>(&acc[5])), "=r"(*reinterpret_cast<uint32_t*>(&acc[6])), "=r"(*reinterpret_cast<uint32_t*>(&acc[7])), "=r"(*reinterpret_cast<uint32_t*>(&acc[8])), "=r"(*reinterpret_cast<uint32_t*>(&acc[9])), "=r"(*reinterpret_cast<uint32_t*>(&acc[10])), "=r"(*reinterpret_cast<uint32_t*>(&acc[11])), "=r"(*reinterpret_cast<uint32_t*>(&acc[12])), "=r"(*reinterpret_cast<uint32_t*>(&acc[13])), "=r"(*reinterpret_cast<uint32_t*>(&acc[14])), "=r"(*reinterpret_cast<uint32_t*>(&acc[15]))
                    : "r"(col_12 + 0));
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&acc[16])), "=r"(*reinterpret_cast<uint32_t*>(&acc[17])), "=r"(*reinterpret_cast<uint32_t*>(&acc[18])), "=r"(*reinterpret_cast<uint32_t*>(&acc[19])), "=r"(*reinterpret_cast<uint32_t*>(&acc[20])), "=r"(*reinterpret_cast<uint32_t*>(&acc[21])), "=r"(*reinterpret_cast<uint32_t*>(&acc[22])), "=r"(*reinterpret_cast<uint32_t*>(&acc[23])), "=r"(*reinterpret_cast<uint32_t*>(&acc[24])), "=r"(*reinterpret_cast<uint32_t*>(&acc[25])), "=r"(*reinterpret_cast<uint32_t*>(&acc[26])), "=r"(*reinterpret_cast<uint32_t*>(&acc[27])), "=r"(*reinterpret_cast<uint32_t*>(&acc[28])), "=r"(*reinterpret_cast<uint32_t*>(&acc[29])), "=r"(*reinterpret_cast<uint32_t*>(&acc[30])), "=r"(*reinterpret_cast<uint32_t*>(&acc[31]))
                    : "r"(col_12 + 1048576));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                #pragma unroll
                for (int _lp = 0; _lp < 16; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc[_lp*2 + 0], acc[_lp*2+1 + 0]));
                    pk[_lp] = *(uint32_t*)&_bf2;
                }
                unsigned int slot_12 = warp_slots + 0;
                if (lane == 0) {
                    asm volatile("cp.async.bulk.wait_group.read 1;");
                }
                __syncwarp();
                uint32_t _stmatrix_addr_48 = static_cast<uint32_t>(slot_12 + st_row + 0);
                asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_48), "r"(*reinterpret_cast<const uint32_t*>(&pk[0])), "r"(*reinterpret_cast<const uint32_t*>(&pk[1])), "r"(*reinterpret_cast<const uint32_t*>(&pk[8])), "r"(*reinterpret_cast<const uint32_t*>(&pk[9]))
                    : "memory");
                uint32_t _stmatrix_addr_49 = static_cast<uint32_t>(slot_12 + st_row + 512);
                asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_49), "r"(*reinterpret_cast<const uint32_t*>(&pk[2])), "r"(*reinterpret_cast<const uint32_t*>(&pk[3])), "r"(*reinterpret_cast<const uint32_t*>(&pk[10])), "r"(*reinterpret_cast<const uint32_t*>(&pk[11]))
                    : "memory");
                uint32_t _stmatrix_addr_50 = static_cast<uint32_t>(slot_12 + st_row + 1024);
                asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_50), "r"(*reinterpret_cast<const uint32_t*>(&pk[4])), "r"(*reinterpret_cast<const uint32_t*>(&pk[5])), "r"(*reinterpret_cast<const uint32_t*>(&pk[12])), "r"(*reinterpret_cast<const uint32_t*>(&pk[13]))
                    : "memory");
                uint32_t _stmatrix_addr_51 = static_cast<uint32_t>(slot_12 + st_row + 1536);
                asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_51), "r"(*reinterpret_cast<const uint32_t*>(&pk[6])), "r"(*reinterpret_cast<const uint32_t*>(&pk[7])), "r"(*reinterpret_cast<const uint32_t*>(&pk[14])), "r"(*reinterpret_cast<const uint32_t*>(&pk[15]))
                    : "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                __syncwarp();
                if (lane == 0) {
                    tma_store_3d((&C), f_row + 128, t_col + 160, l_e, slot_12);
                    asm volatile("cp.async.bulk.commit_group;");
                }
                unsigned int col_13 = lane_addr + 448;
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&acc[0])), "=r"(*reinterpret_cast<uint32_t*>(&acc[1])), "=r"(*reinterpret_cast<uint32_t*>(&acc[2])), "=r"(*reinterpret_cast<uint32_t*>(&acc[3])), "=r"(*reinterpret_cast<uint32_t*>(&acc[4])), "=r"(*reinterpret_cast<uint32_t*>(&acc[5])), "=r"(*reinterpret_cast<uint32_t*>(&acc[6])), "=r"(*reinterpret_cast<uint32_t*>(&acc[7])), "=r"(*reinterpret_cast<uint32_t*>(&acc[8])), "=r"(*reinterpret_cast<uint32_t*>(&acc[9])), "=r"(*reinterpret_cast<uint32_t*>(&acc[10])), "=r"(*reinterpret_cast<uint32_t*>(&acc[11])), "=r"(*reinterpret_cast<uint32_t*>(&acc[12])), "=r"(*reinterpret_cast<uint32_t*>(&acc[13])), "=r"(*reinterpret_cast<uint32_t*>(&acc[14])), "=r"(*reinterpret_cast<uint32_t*>(&acc[15]))
                    : "r"(col_13 + 0));
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&acc[16])), "=r"(*reinterpret_cast<uint32_t*>(&acc[17])), "=r"(*reinterpret_cast<uint32_t*>(&acc[18])), "=r"(*reinterpret_cast<uint32_t*>(&acc[19])), "=r"(*reinterpret_cast<uint32_t*>(&acc[20])), "=r"(*reinterpret_cast<uint32_t*>(&acc[21])), "=r"(*reinterpret_cast<uint32_t*>(&acc[22])), "=r"(*reinterpret_cast<uint32_t*>(&acc[23])), "=r"(*reinterpret_cast<uint32_t*>(&acc[24])), "=r"(*reinterpret_cast<uint32_t*>(&acc[25])), "=r"(*reinterpret_cast<uint32_t*>(&acc[26])), "=r"(*reinterpret_cast<uint32_t*>(&acc[27])), "=r"(*reinterpret_cast<uint32_t*>(&acc[28])), "=r"(*reinterpret_cast<uint32_t*>(&acc[29])), "=r"(*reinterpret_cast<uint32_t*>(&acc[30])), "=r"(*reinterpret_cast<uint32_t*>(&acc[31]))
                    : "r"(col_13 + 1048576));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                if (elect_sync()) {
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    asm volatile(
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                        :: "r"((acc_empty_addr) & 0xFEFFFFFF) : "memory");
                }
                _phase_acc_full ^= 1;
                #pragma unroll
                for (int _lp = 0; _lp < 16; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc[_lp*2 + 0], acc[_lp*2+1 + 0]));
                    pk[_lp] = *(uint32_t*)&_bf2;
                }
                unsigned int slot_13 = warp_slots + 2048;
                if (lane == 0) {
                    asm volatile("cp.async.bulk.wait_group.read 1;");
                }
                __syncwarp();
                uint32_t _stmatrix_addr_52 = static_cast<uint32_t>(slot_13 + st_row + 0);
                asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_52), "r"(*reinterpret_cast<const uint32_t*>(&pk[0])), "r"(*reinterpret_cast<const uint32_t*>(&pk[1])), "r"(*reinterpret_cast<const uint32_t*>(&pk[8])), "r"(*reinterpret_cast<const uint32_t*>(&pk[9]))
                    : "memory");
                uint32_t _stmatrix_addr_53 = static_cast<uint32_t>(slot_13 + st_row + 512);
                asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_53), "r"(*reinterpret_cast<const uint32_t*>(&pk[2])), "r"(*reinterpret_cast<const uint32_t*>(&pk[3])), "r"(*reinterpret_cast<const uint32_t*>(&pk[10])), "r"(*reinterpret_cast<const uint32_t*>(&pk[11]))
                    : "memory");
                uint32_t _stmatrix_addr_54 = static_cast<uint32_t>(slot_13 + st_row + 1024);
                asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_54), "r"(*reinterpret_cast<const uint32_t*>(&pk[4])), "r"(*reinterpret_cast<const uint32_t*>(&pk[5])), "r"(*reinterpret_cast<const uint32_t*>(&pk[12])), "r"(*reinterpret_cast<const uint32_t*>(&pk[13]))
                    : "memory");
                uint32_t _stmatrix_addr_55 = static_cast<uint32_t>(slot_13 + st_row + 1536);
                asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_55), "r"(*reinterpret_cast<const uint32_t*>(&pk[6])), "r"(*reinterpret_cast<const uint32_t*>(&pk[7])), "r"(*reinterpret_cast<const uint32_t*>(&pk[14])), "r"(*reinterpret_cast<const uint32_t*>(&pk[15]))
                    : "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                __syncwarp();
                if (lane == 0) {
                    tma_store_3d((&C), f_row + 128, t_col + 192, l_e, slot_13);
                    asm volatile("cp.async.bulk.commit_group;");
                }
            }
            if (lane == 0) {
                asm volatile("cp.async.bulk.wait_group 0;");
            }
        }
    }
    // ---- Role: load ----
    if (warp == 4) {
        { // load_main
            unsigned int load_stage = 0;
            int k_iters = (K + 63) / 64;
            unsigned int m_tiles = (unsigned int)((M + 511) / 512);
            unsigned int mn_tiles = m_tiles * (unsigned int)((N + 224 - 1) / 224);
            unsigned int total = mn_tiles * (unsigned int)L;
            unsigned int _phase_mma_done = 1;
            if (elect_sync()) {
                asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&A))) : "memory");
                asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&B))) : "memory");
                #pragma unroll 1
                for (unsigned int t = (unsigned int)cluster_id; t < total; t += (unsigned int)num_clusters) {
                    int a_row = (int)(t % m_tiles) * 512 + cta_rank * 256;
                    int b_row = (int)(t % mn_tiles / m_tiles) * 224 + cta_rank * 112;
                    int l_idx = (int)(t / mn_tiles);
                    #pragma unroll 1
                    for (int k = 0; k < k_iters; k++) {
                        mbarrier_wait(mma_done_addr + (load_stage) * 8, _phase_mma_done);
                        int k0 = k * 64;
                        tma_3d_gmem2smem_cta2(smem_a_addr + load_stage * 47104, (&A), k0, a_row, l_idx, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                        tma_3d_gmem2smem_cta2(smem_b_addr + load_stage * 47104, (&B), k0, b_row, l_idx, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                        if (cta_rank == 0) {
                            asm volatile(
                                "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                                :: "r"((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF), "r"((uint32_t)(94208)) : "memory");
                        }
                        load_stage += 1;
                        if (load_stage == 4) { load_stage = 0; _phase_mma_done ^= 1; }
                    }
                }
                #pragma unroll
                for (int _xs = 0; _xs < 4; _xs++) {
                    mbarrier_wait(mma_done_addr + (load_stage) * 8, _phase_mma_done);
                    load_stage += 1;
                    if (load_stage == 4) { load_stage = 0; _phase_mma_done ^= 1; }
                }
            }
        }
    }
    // ---- Role: mma ----
    if (warp == 6) {
        { // mma_main
            unsigned int mma_stage = 0;
            int k_iters_m = (K + 63) / 64;
            unsigned int total_m = (unsigned int)((M + 511) / 512) * (unsigned int)((N + 224 - 1) / 224) * (unsigned int)L;
            unsigned int _phase_acc_empty = 1;
            unsigned int _phase_tma_full = 0;
            if (cta_rank == 0) {
                #pragma unroll 1
                for (unsigned int _t = (unsigned int)cluster_id; _t < total_m; _t += (unsigned int)num_clusters) {
                    mbarrier_wait(acc_empty_addr, _phase_acc_empty);
                    #pragma unroll 1
                    for (int k_m = 0; k_m < k_iters_m; k_m++) {
                        mbarrier_wait(tma_full_addr + (mma_stage) * 8, _phase_tma_full);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int init_flag = ((k_m == 0) ? 1 : 0);
                        int _a_lo = (((smem_a_addr) >> 4) & 0x3FFF) + (mma_stage) * 2944;
                        int _a_hi = (((smem_a_addr + 16384) >> 4) & 0x3FFF) + (mma_stage) * 2944;
                        int _b = (((smem_b_addr) >> 4) & 0x3FFF) + (mma_stage) * 2944;
                        uint32_t _elected_0 = elect_sync();
                        asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id, m0, m1, m2, m3, m4, m5, m6, m7;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "setp.ne.b32 leader, %4, 0;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    "mov.b32 m0, 0; mov.b32 m1, 0; mov.b32 m2, 0; mov.b32 m3, 0;\n\tmov.b32 m4, 0; mov.b32 m5, 0; mov.b32 m6, 0; mov.b32 m7, 0;\n\t"
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 272106640;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"(_a_lo), "r"(_b), "r"(tmem_accum), "r"(((init_flag) ? 0 : 1)), "r"(_elected_0) : "memory");
                        asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id, m0, m1, m2, m3, m4, m5, m6, m7;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "setp.ne.b32 leader, %4, 0;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    "mov.b32 m0, 0; mov.b32 m1, 0; mov.b32 m2, 0; mov.b32 m3, 0;\n\tmov.b32 m4, 0; mov.b32 m5, 0; mov.b32 m6, 0; mov.b32 m7, 0;\n\t"
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 272106640;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"(_a_hi), "r"(_b), "r"((tmem_accum + (256))), "r"(((init_flag) ? 0 : 1)), "r"(_elected_0) : "memory");
                        elect_commit_cg2_multicast(mma_done_addr + (mma_stage) * 8, (uint16_t)(3));
                        mma_stage += 1;
                        if (mma_stage == 4) { mma_stage = 0; _phase_tma_full ^= 1; }
                    }
                    elect_commit_cg2_multicast(acc_full_addr, (uint16_t)(3));
                    _phase_acc_empty ^= 1;
                }
                mbarrier_wait(acc_empty_addr, _phase_acc_empty);
                _phase_acc_empty ^= 1;
                asm volatile(
                    "{\n\t"
                    ".reg .b32 remAddr32;\n\t"
                    "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                    "mbarrier.arrive.release.cluster.shared::cluster.b64 _, [remAddr32];\n\t"
                    "}"
                    :: "r"(tmem_free_addr), "r"(1) : "memory");
            }
            if (cta_rank != 0) {
                mbarrier_wait_cluster_hint(tmem_free_addr, 0, 10000000);
            }
            asm volatile("tcgen05.fence::after_thread_sync;");
            asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(512));
        }
    }

    // Cleanup
}

} // extern "C"
