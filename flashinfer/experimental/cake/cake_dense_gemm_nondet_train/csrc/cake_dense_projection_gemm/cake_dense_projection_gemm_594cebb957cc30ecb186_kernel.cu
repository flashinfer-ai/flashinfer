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

#define BLOCK_M 128
#define BLOCK_N 256
#define BLOCK_K 64
#define MMA_K 16
#define CTA_GROUP 2
#define NUM_STAGES 6
#define TMEM_NCOLS 512
#define NUM_EPILOGUE_WARPS 4
#define CLUSTER_M 256

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

__global__ __launch_bounds__(192) __cluster_dims__(2,1,1) void
kernel_cake_dense_projection_gemm_594cebb957cc30ecb186(const __grid_constant__ CUtensorMap A, const __grid_constant__ CUtensorMap B, __nv_bfloat16* __restrict__ out, float* __restrict__ out32, const __grid_constant__ CUtensorMap C16, const __grid_constant__ CUtensorMap C32, int M, int N, int K, int ldo)
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
    #define mma_done_addr (mbar_base + 48)
    #define mainloop_done_addr (mbar_base + 96)
    #define epilogue_done_addr (mbar_base + 112)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    const unsigned int clusters_x = gridDim.x / 2;
    const unsigned int cluster_id = ((blockIdx.z * gridDim.y + blockIdx.y) * clusters_x) + blockIdx.x / 2;
    const unsigned int num_clusters = clusters_x * gridDim.y * gridDim.z;

    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));

    // Prologue
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 128);
    __nv_bfloat16* smem_a = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int smem_a_addr = smem + 1024;
    __nv_bfloat16* smem_b = reinterpret_cast<__nv_bfloat16*>(smem_raw + 17408);
    const int smem_b_addr = smem + 17408;
    float* smem_out = reinterpret_cast<float*>(smem_raw + 197632);
    const int smem_out_addr = smem + 197632;

    // Kernel setup ops
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");

    // Mbarrier init (16 barriers at smem_raw[0..128))

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'tma_pipe' ---
            // tma_full: 6 barriers, init_count=2
            mbarrier_init(smem + 0, 2);
            mbarrier_init(smem + 8, 2);
            mbarrier_init(smem + 16, 2);
            mbarrier_init(smem + 24, 2);
            mbarrier_init(smem + 32, 2);
            mbarrier_init(smem + 40, 2);
            // mma_done: 6 barriers, init_count=1
            mbarrier_init(smem + 48, 1);
            mbarrier_init(smem + 56, 1);
            mbarrier_init(smem + 64, 1);
            mbarrier_init(smem + 72, 1);
            mbarrier_init(smem + 80, 1);
            mbarrier_init(smem + 88, 1);
            // --- pipeline 'mainloop_pipe' ---
            // mainloop_done: 2 barriers, init_count=1
            mbarrier_init(smem + 96, 1);
            mbarrier_init(smem + 104, 1);
            // epilogue_done: 2 barriers, init_count=8
            mbarrier_init(smem + 112, 8);
            mbarrier_init(smem + 120, 8);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();
    asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");

    const unsigned int taddr = 0;
    const unsigned int tmem_accum = taddr;

    // ---- Role: load ----
    if (warp == 0) {
        { // load_main
            unsigned int load_stage = 0;
            int num_iters = K / BLOCK_K;
            unsigned int grid_c_u = (unsigned int)(N / BLOCK_N);
            unsigned int total_tiles = (unsigned int)((M + CLUSTER_M - 1) / CLUSTER_M) * grid_c_u;
            unsigned int num_clusters_u = (unsigned int)num_clusters;
            unsigned int cluster_u = (unsigned int)cluster_id;
            if (elect_sync()) {
                asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&A))) : "memory");
                asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&B))) : "memory");
            }
            asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
            asm volatile("griddepcontrol.wait;" ::: "memory");
            unsigned int _phase_mma_done = 1;
            if (elect_sync()) {
                #pragma unroll 1
                for (unsigned int t = cluster_u; t < total_tiles; t += num_clusters_u) {
                    unsigned int m_block = t / grid_c_u;
                    unsigned int n_block = t % grid_c_u;
                    int off_m = (int)m_block * CLUSTER_M + cta_rank * BLOCK_M;
                    int off_n = (int)n_block * BLOCK_N + cta_rank * (BLOCK_N / CTA_GROUP);
                    #pragma unroll 1
                    for (int iter_k = 0; iter_k < num_iters; iter_k++) {
                        mbarrier_wait(mma_done_addr + (load_stage) * 8, _phase_mma_done);
                        tma_3d_gmem2smem_cta2(smem_a_addr + load_stage * 32768, (&A), 0, off_m, iter_k, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                        tma_3d_gmem2smem_cta2(smem_b_addr + load_stage * 32768, (&B), off_n, iter_k * BLOCK_K, 0, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                        tma_3d_gmem2smem_cta2(smem_b_addr + load_stage * 32768 + 8192, (&B), off_n + 64, iter_k * BLOCK_K, 0, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                        asm volatile(
                            "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                            :: "r"((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF), "r"((uint32_t)(32768)) : "memory");
                        load_stage += 1;
                        if (load_stage == 6) { load_stage = 0; _phase_mma_done ^= 1; }
                    }
                }
                #pragma unroll
                for (int _xs = 0; _xs < 6; _xs++) {
                    mbarrier_wait(mma_done_addr + (load_stage) * 8, _phase_mma_done);
                    load_stage += 1;
                    if (load_stage == 6) { load_stage = 0; _phase_mma_done ^= 1; }
                }
            }
        }
    }
    // ---- Role: mma ----
    if (warp == 1) {
        { // mma_main
            unsigned int mma_tma_stage = 0;
            unsigned int mma_epi_stage = 0;
            int num_iters_m = K / BLOCK_K;
            unsigned int grid_c_m = (unsigned int)(N / BLOCK_N);
            unsigned int total_tiles_m = (unsigned int)((M + CLUSTER_M - 1) / CLUSTER_M) * grid_c_m;
            unsigned int num_clusters_m = (unsigned int)num_clusters;
            unsigned int cluster_m = (unsigned int)cluster_id;
            asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
            int _tmem_alloc_hold = smem + 128;
            asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_alloc_hold), "r"(512) : "memory");
            asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;");
            asm volatile("griddepcontrol.wait;" ::: "memory");
            unsigned int _phase_epilogue_done = 1;
            unsigned int _phase_tma_full = 0;
            if (cta_rank == 0) {
                #pragma unroll 1
                for (unsigned int t_m = cluster_m; t_m < total_tiles_m; t_m += num_clusters_m) {
                    mbarrier_wait(epilogue_done_addr + (mma_epi_stage) * 8, _phase_epilogue_done);
                    #pragma unroll 1
                    for (int iter_k_m = 0; iter_k_m < num_iters_m; iter_k_m++) {
                        mbarrier_wait(tma_full_addr + (mma_tma_stage) * 8, _phase_tma_full);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int init_flag = ((iter_k_m == 0) ? 1 : 0);
                        int _mma_a_lo_0 = (((smem_a_addr) >> 4) & 0x3FFF) + (mma_tma_stage) * 2048;
                        int _mma_b_lo_0 = ((((smem_b_addr) >> 4) & 0x3FFF) | 0x2000000) + (mma_tma_stage) * 2048;
                        asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id, m0, m1, m2, m3, m4, m5, m6, m7;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    "mov.b32 m0, 0; mov.b32 m1, 0; mov.b32 m2, 0; mov.b32 m3, 0;\n\tmov.b32 m4, 0; mov.b32 m5, 0; mov.b32 m6, 0; mov.b32 m7, 0;\n\t"
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 272696464;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_0), "r"(_mma_b_lo_0), "r"((tmem_accum + (mma_epi_stage * 256))), "r"(((init_flag) ? 0 : 1)) : "memory");
                        elect_commit_cg2_multicast(mma_done_addr + (mma_tma_stage) * 8, (uint16_t)(3));
                        mma_tma_stage += 1;
                        if (mma_tma_stage == 6) { mma_tma_stage = 0; _phase_tma_full ^= 1; }
                    }
                    elect_commit_cg2_multicast(mainloop_done_addr + (mma_epi_stage) * 8, (uint16_t)(3));
                    mma_epi_stage += 1;
                    if (mma_epi_stage == 2) { mma_epi_stage = 0; _phase_epilogue_done ^= 1; }
                }
                #pragma unroll
                for (int _xe = 0; _xe < 2; _xe++) {
                    mbarrier_wait(epilogue_done_addr + (mma_epi_stage) * 8, _phase_epilogue_done);
                    mma_epi_stage += 1;
                    if (mma_epi_stage == 2) { mma_epi_stage = 0; _phase_epilogue_done ^= 1; }
                }
            }
        }
    }
    // ---- Role: epilogue ----
    if (warp >= 2 && warp <= 5) {
        { // epilogue_main
            unsigned int epi_stage = 0;
            const int epi_warp = warp % 4;
            const int epi_tid = epi_warp * 32 + lane;
            unsigned int grid_c_e = (unsigned int)(N / BLOCK_N);
            unsigned int total_tiles_e = (unsigned int)((M + CLUSTER_M - 1) / CLUSTER_M) * grid_c_e;
            unsigned int num_clusters_e = (unsigned int)num_clusters;
            unsigned int cluster_e = (unsigned int)cluster_id;
            if (epi_warp % 2 == 0) {
                if (lane == 0) {
                    asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&C32))) : "memory");
                }
            }
            asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
            asm volatile("griddepcontrol.wait;" ::: "memory");
            unsigned int _phase_mainloop_done = 0;
            #pragma unroll 1
            for (unsigned int t_e = cluster_e; t_e < total_tiles_e; t_e += num_clusters_e) {
                unsigned int m_block_e = t_e / grid_c_e;
                unsigned int n_block_e = t_e % grid_c_e;
                int row_g = (int)m_block_e * CLUSTER_M + cta_rank * BLOCK_M + epi_tid;
                mbarrier_wait(mainloop_done_addr + (epi_stage) * 8, _phase_mainloop_done);
                asm volatile("tcgen05.fence::after_thread_sync;");
                int tmem_row_n = taddr + (unsigned int)(cta_rank * 128 + epi_warp * 32 << 16) + epi_stage * (unsigned int)BLOCK_N;
                float _acc_n[64];
                unsigned int row_off = (unsigned int)(lane * 128);
                int swz = lane & 7;
                int row0_t = row_g - lane;
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(_acc_n[0]), "=f"(_acc_n[1]), "=f"(_acc_n[2]), "=f"(_acc_n[3]), "=f"(_acc_n[4]), "=f"(_acc_n[5]), "=f"(_acc_n[6]), "=f"(_acc_n[7]), "=f"(_acc_n[8]), "=f"(_acc_n[9]), "=f"(_acc_n[10]), "=f"(_acc_n[11]), "=f"(_acc_n[12]), "=f"(_acc_n[13]), "=f"(_acc_n[14]), "=f"(_acc_n[15]), "=f"(_acc_n[16]), "=f"(_acc_n[17]), "=f"(_acc_n[18]), "=f"(_acc_n[19]), "=f"(_acc_n[20]), "=f"(_acc_n[21]), "=f"(_acc_n[22]), "=f"(_acc_n[23]), "=f"(_acc_n[24]), "=f"(_acc_n[25]), "=f"(_acc_n[26]), "=f"(_acc_n[27]), "=f"(_acc_n[28]), "=f"(_acc_n[29]), "=f"(_acc_n[30]), "=f"(_acc_n[31])
                    : "r"(tmem_row_n));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                unsigned int slot_0 = smem_out_addr + 0 + (unsigned int)(epi_warp * 4096);
                if (epi_warp % 2 == 0) {
                    if (lane == 0) {
                        asm volatile("cp.async.bulk.wait_group.read 1;");
                    }
                }
                asm volatile("barrier.sync 1, 128;" ::: "memory");
                #pragma unroll
                for (int q_0 = 0; q_0 < 8; q_0++) {
                    int s0_0 = q_0 * 4;
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(slot_0 + row_off + (unsigned int)((q_0 ^ swz) * 16)), "r"(__float_as_uint(_acc_n[s0_0])), "r"(__float_as_uint(_acc_n[s0_0 + 1])), "r"(__float_as_uint(_acc_n[s0_0 + 2])), "r"(__float_as_uint(_acc_n[s0_0 + 3])));
                }
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("tcgen05.fence::before_thread_sync;");
                asm volatile("barrier.sync 1, 128;" ::: "memory");
                if (epi_warp % 2 == 0) {
                    if (lane == 0) {
                        tma_store_3d((&C32), (int)n_block_e * BLOCK_N, row0_t, 0, slot_0);
                        tma_store_3d((&C32), (int)n_block_e * BLOCK_N, row0_t + 32, 0, slot_0 + 4096);
                    }
                }
                asm volatile("cp.async.bulk.commit_group;");
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(_acc_n[0]), "=f"(_acc_n[1]), "=f"(_acc_n[2]), "=f"(_acc_n[3]), "=f"(_acc_n[4]), "=f"(_acc_n[5]), "=f"(_acc_n[6]), "=f"(_acc_n[7]), "=f"(_acc_n[8]), "=f"(_acc_n[9]), "=f"(_acc_n[10]), "=f"(_acc_n[11]), "=f"(_acc_n[12]), "=f"(_acc_n[13]), "=f"(_acc_n[14]), "=f"(_acc_n[15]), "=f"(_acc_n[16]), "=f"(_acc_n[17]), "=f"(_acc_n[18]), "=f"(_acc_n[19]), "=f"(_acc_n[20]), "=f"(_acc_n[21]), "=f"(_acc_n[22]), "=f"(_acc_n[23]), "=f"(_acc_n[24]), "=f"(_acc_n[25]), "=f"(_acc_n[26]), "=f"(_acc_n[27]), "=f"(_acc_n[28]), "=f"(_acc_n[29]), "=f"(_acc_n[30]), "=f"(_acc_n[31])
                    : "r"(tmem_row_n + 32));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                unsigned int slot_1 = smem_out_addr + 16384 + (unsigned int)(epi_warp * 4096);
                if (epi_warp % 2 == 0) {
                    if (lane == 0) {
                        asm volatile("cp.async.bulk.wait_group.read 1;");
                    }
                }
                asm volatile("barrier.sync 1, 128;" ::: "memory");
                #pragma unroll
                for (int q_1 = 0; q_1 < 8; q_1++) {
                    int s0_1 = q_1 * 4;
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(slot_1 + row_off + (unsigned int)((q_1 ^ swz) * 16)), "r"(__float_as_uint(_acc_n[s0_1])), "r"(__float_as_uint(_acc_n[s0_1 + 1])), "r"(__float_as_uint(_acc_n[s0_1 + 2])), "r"(__float_as_uint(_acc_n[s0_1 + 3])));
                }
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("tcgen05.fence::before_thread_sync;");
                asm volatile("barrier.sync 1, 128;" ::: "memory");
                if (epi_warp % 2 == 0) {
                    if (lane == 0) {
                        tma_store_3d((&C32), (int)n_block_e * BLOCK_N + 32, row0_t, 0, slot_1);
                        tma_store_3d((&C32), (int)n_block_e * BLOCK_N + 32, row0_t + 32, 0, slot_1 + 4096);
                    }
                }
                asm volatile("cp.async.bulk.commit_group;");
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(_acc_n[0]), "=f"(_acc_n[1]), "=f"(_acc_n[2]), "=f"(_acc_n[3]), "=f"(_acc_n[4]), "=f"(_acc_n[5]), "=f"(_acc_n[6]), "=f"(_acc_n[7]), "=f"(_acc_n[8]), "=f"(_acc_n[9]), "=f"(_acc_n[10]), "=f"(_acc_n[11]), "=f"(_acc_n[12]), "=f"(_acc_n[13]), "=f"(_acc_n[14]), "=f"(_acc_n[15]), "=f"(_acc_n[16]), "=f"(_acc_n[17]), "=f"(_acc_n[18]), "=f"(_acc_n[19]), "=f"(_acc_n[20]), "=f"(_acc_n[21]), "=f"(_acc_n[22]), "=f"(_acc_n[23]), "=f"(_acc_n[24]), "=f"(_acc_n[25]), "=f"(_acc_n[26]), "=f"(_acc_n[27]), "=f"(_acc_n[28]), "=f"(_acc_n[29]), "=f"(_acc_n[30]), "=f"(_acc_n[31])
                    : "r"(tmem_row_n + 64));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                unsigned int slot_2 = smem_out_addr + 0 + (unsigned int)(epi_warp * 4096);
                if (epi_warp % 2 == 0) {
                    if (lane == 0) {
                        asm volatile("cp.async.bulk.wait_group.read 1;");
                    }
                }
                asm volatile("barrier.sync 1, 128;" ::: "memory");
                #pragma unroll
                for (int q_2 = 0; q_2 < 8; q_2++) {
                    int s0_2 = q_2 * 4;
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(slot_2 + row_off + (unsigned int)((q_2 ^ swz) * 16)), "r"(__float_as_uint(_acc_n[s0_2])), "r"(__float_as_uint(_acc_n[s0_2 + 1])), "r"(__float_as_uint(_acc_n[s0_2 + 2])), "r"(__float_as_uint(_acc_n[s0_2 + 3])));
                }
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("tcgen05.fence::before_thread_sync;");
                asm volatile("barrier.sync 1, 128;" ::: "memory");
                if (epi_warp % 2 == 0) {
                    if (lane == 0) {
                        tma_store_3d((&C32), (int)n_block_e * BLOCK_N + 64, row0_t, 0, slot_2);
                        tma_store_3d((&C32), (int)n_block_e * BLOCK_N + 64, row0_t + 32, 0, slot_2 + 4096);
                    }
                }
                asm volatile("cp.async.bulk.commit_group;");
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(_acc_n[0]), "=f"(_acc_n[1]), "=f"(_acc_n[2]), "=f"(_acc_n[3]), "=f"(_acc_n[4]), "=f"(_acc_n[5]), "=f"(_acc_n[6]), "=f"(_acc_n[7]), "=f"(_acc_n[8]), "=f"(_acc_n[9]), "=f"(_acc_n[10]), "=f"(_acc_n[11]), "=f"(_acc_n[12]), "=f"(_acc_n[13]), "=f"(_acc_n[14]), "=f"(_acc_n[15]), "=f"(_acc_n[16]), "=f"(_acc_n[17]), "=f"(_acc_n[18]), "=f"(_acc_n[19]), "=f"(_acc_n[20]), "=f"(_acc_n[21]), "=f"(_acc_n[22]), "=f"(_acc_n[23]), "=f"(_acc_n[24]), "=f"(_acc_n[25]), "=f"(_acc_n[26]), "=f"(_acc_n[27]), "=f"(_acc_n[28]), "=f"(_acc_n[29]), "=f"(_acc_n[30]), "=f"(_acc_n[31])
                    : "r"(tmem_row_n + 96));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                unsigned int slot_3 = smem_out_addr + 16384 + (unsigned int)(epi_warp * 4096);
                if (epi_warp % 2 == 0) {
                    if (lane == 0) {
                        asm volatile("cp.async.bulk.wait_group.read 1;");
                    }
                }
                asm volatile("barrier.sync 1, 128;" ::: "memory");
                #pragma unroll
                for (int q_3 = 0; q_3 < 8; q_3++) {
                    int s0_3 = q_3 * 4;
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(slot_3 + row_off + (unsigned int)((q_3 ^ swz) * 16)), "r"(__float_as_uint(_acc_n[s0_3])), "r"(__float_as_uint(_acc_n[s0_3 + 1])), "r"(__float_as_uint(_acc_n[s0_3 + 2])), "r"(__float_as_uint(_acc_n[s0_3 + 3])));
                }
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("tcgen05.fence::before_thread_sync;");
                asm volatile("barrier.sync 1, 128;" ::: "memory");
                if (epi_warp % 2 == 0) {
                    if (lane == 0) {
                        tma_store_3d((&C32), (int)n_block_e * BLOCK_N + 96, row0_t, 0, slot_3);
                        tma_store_3d((&C32), (int)n_block_e * BLOCK_N + 96, row0_t + 32, 0, slot_3 + 4096);
                    }
                }
                asm volatile("cp.async.bulk.commit_group;");
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(_acc_n[0]), "=f"(_acc_n[1]), "=f"(_acc_n[2]), "=f"(_acc_n[3]), "=f"(_acc_n[4]), "=f"(_acc_n[5]), "=f"(_acc_n[6]), "=f"(_acc_n[7]), "=f"(_acc_n[8]), "=f"(_acc_n[9]), "=f"(_acc_n[10]), "=f"(_acc_n[11]), "=f"(_acc_n[12]), "=f"(_acc_n[13]), "=f"(_acc_n[14]), "=f"(_acc_n[15]), "=f"(_acc_n[16]), "=f"(_acc_n[17]), "=f"(_acc_n[18]), "=f"(_acc_n[19]), "=f"(_acc_n[20]), "=f"(_acc_n[21]), "=f"(_acc_n[22]), "=f"(_acc_n[23]), "=f"(_acc_n[24]), "=f"(_acc_n[25]), "=f"(_acc_n[26]), "=f"(_acc_n[27]), "=f"(_acc_n[28]), "=f"(_acc_n[29]), "=f"(_acc_n[30]), "=f"(_acc_n[31])
                    : "r"(tmem_row_n + 128));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                unsigned int slot_4 = smem_out_addr + 0 + (unsigned int)(epi_warp * 4096);
                if (epi_warp % 2 == 0) {
                    if (lane == 0) {
                        asm volatile("cp.async.bulk.wait_group.read 1;");
                    }
                }
                asm volatile("barrier.sync 1, 128;" ::: "memory");
                #pragma unroll
                for (int q_4 = 0; q_4 < 8; q_4++) {
                    int s0_4 = q_4 * 4;
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(slot_4 + row_off + (unsigned int)((q_4 ^ swz) * 16)), "r"(__float_as_uint(_acc_n[s0_4])), "r"(__float_as_uint(_acc_n[s0_4 + 1])), "r"(__float_as_uint(_acc_n[s0_4 + 2])), "r"(__float_as_uint(_acc_n[s0_4 + 3])));
                }
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("tcgen05.fence::before_thread_sync;");
                asm volatile("barrier.sync 1, 128;" ::: "memory");
                if (epi_warp % 2 == 0) {
                    if (lane == 0) {
                        tma_store_3d((&C32), (int)n_block_e * BLOCK_N + 128, row0_t, 0, slot_4);
                        tma_store_3d((&C32), (int)n_block_e * BLOCK_N + 128, row0_t + 32, 0, slot_4 + 4096);
                    }
                }
                asm volatile("cp.async.bulk.commit_group;");
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(_acc_n[0]), "=f"(_acc_n[1]), "=f"(_acc_n[2]), "=f"(_acc_n[3]), "=f"(_acc_n[4]), "=f"(_acc_n[5]), "=f"(_acc_n[6]), "=f"(_acc_n[7]), "=f"(_acc_n[8]), "=f"(_acc_n[9]), "=f"(_acc_n[10]), "=f"(_acc_n[11]), "=f"(_acc_n[12]), "=f"(_acc_n[13]), "=f"(_acc_n[14]), "=f"(_acc_n[15]), "=f"(_acc_n[16]), "=f"(_acc_n[17]), "=f"(_acc_n[18]), "=f"(_acc_n[19]), "=f"(_acc_n[20]), "=f"(_acc_n[21]), "=f"(_acc_n[22]), "=f"(_acc_n[23]), "=f"(_acc_n[24]), "=f"(_acc_n[25]), "=f"(_acc_n[26]), "=f"(_acc_n[27]), "=f"(_acc_n[28]), "=f"(_acc_n[29]), "=f"(_acc_n[30]), "=f"(_acc_n[31])
                    : "r"(tmem_row_n + 160));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                unsigned int slot_5 = smem_out_addr + 16384 + (unsigned int)(epi_warp * 4096);
                if (epi_warp % 2 == 0) {
                    if (lane == 0) {
                        asm volatile("cp.async.bulk.wait_group.read 1;");
                    }
                }
                asm volatile("barrier.sync 1, 128;" ::: "memory");
                #pragma unroll
                for (int q_5 = 0; q_5 < 8; q_5++) {
                    int s0_5 = q_5 * 4;
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(slot_5 + row_off + (unsigned int)((q_5 ^ swz) * 16)), "r"(__float_as_uint(_acc_n[s0_5])), "r"(__float_as_uint(_acc_n[s0_5 + 1])), "r"(__float_as_uint(_acc_n[s0_5 + 2])), "r"(__float_as_uint(_acc_n[s0_5 + 3])));
                }
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("tcgen05.fence::before_thread_sync;");
                asm volatile("barrier.sync 1, 128;" ::: "memory");
                if (epi_warp % 2 == 0) {
                    if (lane == 0) {
                        tma_store_3d((&C32), (int)n_block_e * BLOCK_N + 160, row0_t, 0, slot_5);
                        tma_store_3d((&C32), (int)n_block_e * BLOCK_N + 160, row0_t + 32, 0, slot_5 + 4096);
                    }
                }
                asm volatile("cp.async.bulk.commit_group;");
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(_acc_n[0]), "=f"(_acc_n[1]), "=f"(_acc_n[2]), "=f"(_acc_n[3]), "=f"(_acc_n[4]), "=f"(_acc_n[5]), "=f"(_acc_n[6]), "=f"(_acc_n[7]), "=f"(_acc_n[8]), "=f"(_acc_n[9]), "=f"(_acc_n[10]), "=f"(_acc_n[11]), "=f"(_acc_n[12]), "=f"(_acc_n[13]), "=f"(_acc_n[14]), "=f"(_acc_n[15]), "=f"(_acc_n[16]), "=f"(_acc_n[17]), "=f"(_acc_n[18]), "=f"(_acc_n[19]), "=f"(_acc_n[20]), "=f"(_acc_n[21]), "=f"(_acc_n[22]), "=f"(_acc_n[23]), "=f"(_acc_n[24]), "=f"(_acc_n[25]), "=f"(_acc_n[26]), "=f"(_acc_n[27]), "=f"(_acc_n[28]), "=f"(_acc_n[29]), "=f"(_acc_n[30]), "=f"(_acc_n[31])
                    : "r"(tmem_row_n + 192));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                unsigned int slot_6 = smem_out_addr + 0 + (unsigned int)(epi_warp * 4096);
                if (epi_warp % 2 == 0) {
                    if (lane == 0) {
                        asm volatile("cp.async.bulk.wait_group.read 1;");
                    }
                }
                asm volatile("barrier.sync 1, 128;" ::: "memory");
                #pragma unroll
                for (int q_6 = 0; q_6 < 8; q_6++) {
                    int s0_6 = q_6 * 4;
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(slot_6 + row_off + (unsigned int)((q_6 ^ swz) * 16)), "r"(__float_as_uint(_acc_n[s0_6])), "r"(__float_as_uint(_acc_n[s0_6 + 1])), "r"(__float_as_uint(_acc_n[s0_6 + 2])), "r"(__float_as_uint(_acc_n[s0_6 + 3])));
                }
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("tcgen05.fence::before_thread_sync;");
                asm volatile("barrier.sync 1, 128;" ::: "memory");
                if (epi_warp % 2 == 0) {
                    if (lane == 0) {
                        tma_store_3d((&C32), (int)n_block_e * BLOCK_N + 192, row0_t, 0, slot_6);
                        tma_store_3d((&C32), (int)n_block_e * BLOCK_N + 192, row0_t + 32, 0, slot_6 + 4096);
                    }
                }
                asm volatile("cp.async.bulk.commit_group;");
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(_acc_n[0]), "=f"(_acc_n[1]), "=f"(_acc_n[2]), "=f"(_acc_n[3]), "=f"(_acc_n[4]), "=f"(_acc_n[5]), "=f"(_acc_n[6]), "=f"(_acc_n[7]), "=f"(_acc_n[8]), "=f"(_acc_n[9]), "=f"(_acc_n[10]), "=f"(_acc_n[11]), "=f"(_acc_n[12]), "=f"(_acc_n[13]), "=f"(_acc_n[14]), "=f"(_acc_n[15]), "=f"(_acc_n[16]), "=f"(_acc_n[17]), "=f"(_acc_n[18]), "=f"(_acc_n[19]), "=f"(_acc_n[20]), "=f"(_acc_n[21]), "=f"(_acc_n[22]), "=f"(_acc_n[23]), "=f"(_acc_n[24]), "=f"(_acc_n[25]), "=f"(_acc_n[26]), "=f"(_acc_n[27]), "=f"(_acc_n[28]), "=f"(_acc_n[29]), "=f"(_acc_n[30]), "=f"(_acc_n[31])
                    : "r"(tmem_row_n + 224));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                if (elect_sync()) {
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    asm volatile(
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                        :: "r"((epilogue_done_addr + (epi_stage) * 8) & 0xFEFFFFFF) : "memory");
                }
                epi_stage += 1;
                if (epi_stage == 2) { epi_stage = 0; _phase_mainloop_done ^= 1; }
                unsigned int slot_7 = smem_out_addr + 16384 + (unsigned int)(epi_warp * 4096);
                if (epi_warp % 2 == 0) {
                    if (lane == 0) {
                        asm volatile("cp.async.bulk.wait_group.read 1;");
                    }
                }
                asm volatile("barrier.sync 1, 128;" ::: "memory");
                #pragma unroll
                for (int q_7 = 0; q_7 < 8; q_7++) {
                    int s0_7 = q_7 * 4;
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(slot_7 + row_off + (unsigned int)((q_7 ^ swz) * 16)), "r"(__float_as_uint(_acc_n[s0_7])), "r"(__float_as_uint(_acc_n[s0_7 + 1])), "r"(__float_as_uint(_acc_n[s0_7 + 2])), "r"(__float_as_uint(_acc_n[s0_7 + 3])));
                }
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("tcgen05.fence::before_thread_sync;");
                asm volatile("barrier.sync 1, 128;" ::: "memory");
                if (epi_warp % 2 == 0) {
                    if (lane == 0) {
                        tma_store_3d((&C32), (int)n_block_e * BLOCK_N + 224, row0_t, 0, slot_7);
                        tma_store_3d((&C32), (int)n_block_e * BLOCK_N + 224, row0_t + 32, 0, slot_7 + 4096);
                    }
                }
                asm volatile("cp.async.bulk.commit_group;");
            }
            asm volatile("tcgen05.fence::before_thread_sync;");
            asm volatile("barrier.sync 1, 128;" ::: "memory");
            if (epi_warp == 0) {
                asm volatile("tcgen05.fence::after_thread_sync;");
                asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;" :: "r"(0), "r"(512));
            }
            if (epi_warp % 2 == 0) {
                if (lane == 0) {
                    asm volatile("cp.async.bulk.wait_group 0;");
                }
            }
        }
    }

    // Cleanup
}

} // extern "C"
