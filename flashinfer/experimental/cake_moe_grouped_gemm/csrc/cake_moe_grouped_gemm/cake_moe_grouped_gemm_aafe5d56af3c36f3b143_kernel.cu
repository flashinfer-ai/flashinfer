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

#define CAKE_INF CUDART_INF_F
#define TMEM_NCOLS 512
#define TMEM_ACCUM_OFFSET 0
#define NUM_TMA_PIPE_STAGES 7
#define NUM_MAINLOOP_PIPE_STAGES 2
#define NUM_WORK_PIPE_STAGES 4
#define SMEM_SMEM_A_OFF 1024
#define SMEM_SMEM_A_STAGE_BYTES 16384
#define SMEM_SMEM_A_STRIDE 32768
#define SMEM_SMEM_B_OFF 17408
#define SMEM_SMEM_B_STAGE_BYTES 16384
#define SMEM_SMEM_B_STRIDE 32768
#define SMEM_WORK_RING_OFF 230400
#define SMEM_WORK_RING_STAGE_BYTES 16
#define SMEM_WORK_RING_STRIDE 16
#define SMEM_PROBE_CELL_OFF 230464
#define SMEM_PROBE_CELL_STAGE_BYTES 16
#define SMEM_PROBE_CELL_STRIDE 16
#define SMEM_TOTAL 230528
#define BLOCK_M 128
#define BLOCK_N 256
#define BLOCK_K 64
#define MMA_K 16
#define CTA_GROUP 2
#define NUM_STAGES 7
#define NUM_EPILOGUE_WARPS 4
#define CLUSTER_M 256
#define EPI_CHUNK 16

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


__device__ __forceinline__ void mbarrier_arrive(int mbar_addr) {
    asm volatile(
        "mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];"
        :: "r"(mbar_addr) : "memory");
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


__device__ __forceinline__ void tma_4d_gmem2smem_cta2(
    int dst, const void *tmap_ptr, int x, int y, int z, int w, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.4d.shared::cluster.global"
        ".mbarrier::complete_tx::bytes.cta_group::2"
        " [%0], [%1, {%2, %3, %4, %5}], [%6];"
        :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y), "r"(z), "r"(w),
           "r"(mbar_addr) : "memory");
}



__device__ __forceinline__ void tmem_ld_x16(float* dst, int tmem_addr) {
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x16.b32"
        " {%0, %1, %2, %3, %4, %5, %6, %7,"
        "  %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
        : "=f"(dst[0]),  "=f"(dst[1]),  "=f"(dst[2]),  "=f"(dst[3]),
          "=f"(dst[4]),  "=f"(dst[5]),  "=f"(dst[6]),  "=f"(dst[7]),
          "=f"(dst[8]),  "=f"(dst[9]),  "=f"(dst[10]), "=f"(dst[11]),
          "=f"(dst[12]), "=f"(dst[13]), "=f"(dst[14]), "=f"(dst[15])
        : "r"(tmem_addr));
}


extern "C" {

__global__ __launch_bounds__(192) __cluster_dims__(2,1,1) void
kernel_cake_moe_grouped_gemm_aafe5d56af3c36f3b143(const __grid_constant__ CUtensorMap A, const __grid_constant__ CUtensorMap B, __nv_bfloat16* __restrict__ C, int* __restrict__ offs, int num_groups, int sum_m, int N, int K, int ldc, unsigned int* __restrict__ sched, int c0, int c1)
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
    #define mma_done_addr (mbar_base + 56)
    #define mainloop_done_addr (mbar_base + 112)
    #define epilogue_done_addr (mbar_base + 128)
    #define work_full_addr (mbar_base + 144)
    #define work_empty_addr (mbar_base + 176)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    const unsigned int clusters_x = gridDim.x / 2;
    const unsigned int cluster_id = ((blockIdx.z * gridDim.y + blockIdx.y) * clusters_x) + blockIdx.x / 2;
    const unsigned int num_clusters = clusters_x * gridDim.y * gridDim.z;

    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));

    // Kernel setup ops
    __nv_bfloat16* smem_a = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int smem_a_addr = smem + 1024;
    __nv_bfloat16* smem_b = reinterpret_cast<__nv_bfloat16*>(smem_raw + 17408);
    const int smem_b_addr = smem + 17408;
    unsigned int* work_ring = reinterpret_cast<unsigned int*>(smem_raw + 230400);
    const int work_ring_addr = smem + 230400;
    unsigned int* probe_cell = reinterpret_cast<unsigned int*>(smem_raw + 230464);
    const int probe_cell_addr = smem + 230464;

    // Mbarrier init (6 pipeline groups, 0 ordered-sequence groups, 26 barriers)
    // Mbarriers at smem_raw[0..208)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'tma_pipe' ---
            // tma_full: 7 barriers, init_count=2
            mbarrier_init(smem + 0, 2);
            mbarrier_init(smem + 8, 2);
            mbarrier_init(smem + 16, 2);
            mbarrier_init(smem + 24, 2);
            mbarrier_init(smem + 32, 2);
            mbarrier_init(smem + 40, 2);
            mbarrier_init(smem + 48, 2);
            // mma_done: 7 barriers, init_count=1
            mbarrier_init(smem + 56, 1);
            mbarrier_init(smem + 64, 1);
            mbarrier_init(smem + 72, 1);
            mbarrier_init(smem + 80, 1);
            mbarrier_init(smem + 88, 1);
            mbarrier_init(smem + 96, 1);
            mbarrier_init(smem + 104, 1);
            // --- pipeline 'mainloop_pipe' ---
            // mainloop_done: 2 barriers, init_count=1
            mbarrier_init(smem + 112, 1);
            mbarrier_init(smem + 120, 1);
            // epilogue_done: 2 barriers, init_count=8
            mbarrier_init(smem + 128, 8);
            mbarrier_init(smem + 136, 8);
            // --- pipeline 'work_pipe' ---
            // work_full: 4 barriers, init_count=1
            mbarrier_init(smem + 144, 1);
            mbarrier_init(smem + 152, 1);
            mbarrier_init(smem + 160, 1);
            mbarrier_init(smem + 168, 1);
            // work_empty: 4 barriers, init_count=290
            mbarrier_init(smem + 176, 290);
            mbarrier_init(smem + 184, 290);
            mbarrier_init(smem + 192, 290);
            mbarrier_init(smem + 200, 290);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();
    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");

    // TMEM alloc (512 columns, 512 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 208);
    if (warp == 0) {
        int _tmem_hold = smem + 208;
        asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(512) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_accum = taddr;

    // ---- Role: load ----
    if (warp == 0) {
        { // load_main
            unsigned int load_stage = 0;
            unsigned int work_stage = 0;
            int num_iters = K / BLOCK_K;
            int prank = cta_rank & 1;
            unsigned int grid_n_u = (unsigned int)(N / BLOCK_N);
            unsigned int rows = 0;
            int p_start = 0;
            #pragma unroll 1
            for (int gp = 0; gp < (unsigned int)num_groups; gp++) {
                int p_end = offs[gp];
                rows = rows + (unsigned int)((p_end - p_start + CLUSTER_M - 1) / CLUSTER_M);
                p_start = p_end;
            }
            unsigned int cl = (unsigned int)(c0 + c1);
            unsigned int w0 = grid_n_u * (unsigned int)c0;
            unsigned int w1 = grid_n_u * (unsigned int)c1;
            unsigned int t0 = rows * w0 / cl;
            unsigned int t1 = rows * grid_n_u - t0;
            unsigned int g_cur = 0;
            unsigned int g_row0 = 0;
            int g_start = 0;
            int g_end = offs[0];
            unsigned int g_rows = (unsigned int)((g_end + CLUSTER_M - 1) / CLUSTER_M);
            unsigned int near = 0;
            if (cta_rank == 0) {
                unsigned long long base = (unsigned long long)sched;
                unsigned int chunk_a = (unsigned int)((base + 8319 & 18446744073709543424) - base) >> 2;
                unsigned int word_a = chunk_a + (unsigned int)cluster_id % 32 * 32;
                unsigned long long lat_a = 4294967295;
                unsigned long long lat_b = 4294967295;
                #pragma unroll
                for (int _rnd = 0; _rnd < 2; _rnd++) {
                    unsigned long long t0_0;
                    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0_0));
                    if (lane == 0) {
                        unsigned int va = atomicAdd(&sched[word_a], 0);
                        asm volatile("st.shared.b32 [%0], %1;" :: "r"(probe_cell_addr), "r"(va) : "memory");
                    }
                    __syncwarp();
                    unsigned long long t1_1;
                    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1_1));
                    if (lane == 0) {
                        unsigned int vb = atomicAdd(&sched[word_a + 1024], 0);
                        asm volatile("st.shared.b32 [%0], %1;" :: "r"(probe_cell_addr + 4), "r"(vb) : "memory");
                    }
                    __syncwarp();
                    unsigned long long t2;
                    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t2));
                    if (lat_a > t1_1 - t0_0) {
                        lat_a = t1_1 - t0_0;
                    }
                    if (lat_b > t2 - t1_1) {
                        lat_b = t2 - t1_1;
                    }
                }
                unsigned int near_b = ((lat_b < lat_a) ? 1 : 0);
                near = near_b;
            }
            unsigned int _phase_work_empty = 1;
            unsigned int _phase_work_full = 0;
            unsigned int _phase_mma_done = 1;
            if (elect_sync()) {
                unsigned int queue = 0;
                unsigned int claim_ends = 0;
                if (cta_rank == 0) {
                    queue = near;
                    mbarrier_wait_cluster_hint(work_empty_addr + (work_stage) * 8, _phase_work_empty, 10000000);
                    unsigned int _atomic_old_0 = atomicAdd(&sched[queue], 1);
                    work_ring[work_stage * 4] = _atomic_old_0;
                    asm volatile(
                        "{\n\t"
                        ".reg .b32 remAddr32;\n\t"
                        "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                        "mbarrier.arrive.expect_tx.release.cluster.shared::cluster.b64 _, [remAddr32], %2;\n\t"
                        "}"
                        :: "r"(work_full_addr + (work_stage) * 8), "r"(1), "r"((uint32_t)(4)) : "memory");
                    uint32_t _cluster_mapa_0;
                    asm volatile(
                        "mapa.shared::cluster.u32 %0, %1, %2;"
                        : "=r"(_cluster_mapa_0) : "r"(work_ring_addr + work_stage * 16), "r"(1));
                    uint32_t _mbar_peer_1;
                    asm(
                        "mapa.shared::cluster.u32 %0, %1, %2;"
                        : "=r"(_mbar_peer_1) : "r"(work_full_addr + (work_stage) * 8), "r"(1));
                    asm volatile(
                        "st.async.shared::cluster.mbarrier::complete_tx::bytes.b32 [%0], %1, [%2];"
                        :: "r"(_cluster_mapa_0), "r"(_atomic_old_0), "r"(_mbar_peer_1) : "memory");
                    mbarrier_arrive(work_full_addr + (work_stage) * 8);
                    unsigned int ticket = _atomic_old_0;
                    unsigned int first = ticket;
                    if ((first & 2147483647) >= ((queue == 1) ? t1 : t0)) {
                        claim_ends = claim_ends + 1;
                        queue = queue ^ 1;
                    }
                }
                unsigned int ends = 0;
                unsigned int tiles_done = 0;
                #pragma unroll 1
                for (unsigned int _item = 0; _item < t0 + t1 + 2; _item++) {
                    mbarrier_wait(work_full_addr + (work_stage) * 8, _phase_work_full);
                    unsigned int ticket_1 = work_ring[work_stage * 4];
                    asm volatile(
                        "{\n\t"
                        ".reg .b32 remAddr32;\n\t"
                        "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                        "}"
                        :: "r"(work_empty_addr + (work_stage) * 8), "r"(0) : "memory");
                    work_stage += 1;
                    if (work_stage == 4) { work_stage = 0; _phase_work_empty ^= 1; _phase_work_full ^= 1; }
                    if (cta_rank == 0) {
                        if (claim_ends < 2) {
                            mbarrier_wait_cluster_hint(work_empty_addr + (work_stage) * 8, _phase_work_empty, 10000000);
                            unsigned int _atomic_old_1 = atomicAdd(&sched[queue], 1);
                            work_ring[work_stage * 4] = _atomic_old_1;
                            asm volatile(
                                "{\n\t"
                                ".reg .b32 remAddr32;\n\t"
                                "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                                "mbarrier.arrive.expect_tx.release.cluster.shared::cluster.b64 _, [remAddr32], %2;\n\t"
                                "}"
                                :: "r"(work_full_addr + (work_stage) * 8), "r"(1), "r"((uint32_t)(4)) : "memory");
                            uint32_t _cluster_mapa_2;
                            asm volatile(
                                "mapa.shared::cluster.u32 %0, %1, %2;"
                                : "=r"(_cluster_mapa_2) : "r"(work_ring_addr + work_stage * 16), "r"(1));
                            uint32_t _mbar_peer_3;
                            asm(
                                "mapa.shared::cluster.u32 %0, %1, %2;"
                                : "=r"(_mbar_peer_3) : "r"(work_full_addr + (work_stage) * 8), "r"(1));
                            asm volatile(
                                "st.async.shared::cluster.mbarrier::complete_tx::bytes.b32 [%0], %1, [%2];"
                                :: "r"(_cluster_mapa_2), "r"(_atomic_old_1), "r"(_mbar_peer_3) : "memory");
                            mbarrier_arrive(work_full_addr + (work_stage) * 8);
                            unsigned int ticket_0 = _atomic_old_1;
                            unsigned int nxt = ticket_0;
                            if ((nxt & 2147483647) >= ((queue == 1) ? t1 : t0)) {
                                claim_ends = claim_ends + 1;
                                queue = queue ^ 1;
                            }
                        }
                    }
                    unsigned int q = ticket_1 >> 31;
                    unsigned int k = ticket_1 & 2147483647;
                    unsigned int valid = ((q == 1) ? ((k < t1) ? 1 : 0) : ((k < t0) ? 1 : 0));
                    unsigned int w0s = ((w0 > 0) ? w0 : 1);
                    unsigned int w1s = ((w1 > 0) ? w1 : 1);
                    unsigned int r0 = ((k + 1) * cl - 1) / w0s;
                    unsigned int r1 = k * cl / w1s;
                    unsigned int n0 = k - r0 * w0s / cl;
                    unsigned int n1 = grid_n_u - 1 - (k - (r1 * w1s + cl - 1) / cl);
                    unsigned int row = ((q == 1) ? r1 : r0);
                    unsigned int item = ((q == 1) ? n1 : n0);
                    if (valid == 0) {
                        ends = ends + 1;
                    }
                    if (ends == 2) {
                        break;
                    }
                    if (valid == 1) {
                        if (row < g_row0) {
                            g_cur = 0;
                            g_row0 = 0;
                            g_start = 0;
                            g_end = offs[0];
                            g_rows = (unsigned int)((g_end + CLUSTER_M - 1) / CLUSTER_M);
                        }
                        while (row >= g_row0 + g_rows) {
                            g_row0 = g_row0 + g_rows;
                            g_start = g_end;
                            g_cur = g_cur + 1;
                            g_end = offs[g_cur];
                            g_rows = (unsigned int)((g_end - g_start + CLUSTER_M - 1) / CLUSTER_M);
                        }
                        int off_m = g_start + (int)(row - g_row0) * CLUSTER_M + prank * BLOCK_M;
                        int off_n = (int)item * BLOCK_N + prank * (BLOCK_N / CTA_GROUP);
                        #pragma unroll 1
                        for (int iter_k = 0; iter_k < num_iters; iter_k++) {
                            mbarrier_wait(mma_done_addr + (load_stage) * 8, _phase_mma_done);
                            tma_3d_gmem2smem_cta2(smem_a_addr + load_stage * 32768, (&A), 0, off_m, iter_k, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                            tma_4d_gmem2smem_cta2(smem_b_addr + load_stage * 32768, (&B), 0, off_n, iter_k, (int)g_cur, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                            asm volatile(
                                "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                                :: "r"((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF), "r"((uint32_t)(32768)) : "memory");
                            load_stage += 1;
                            if (load_stage == 7) { load_stage = 0; _phase_mma_done ^= 1; }
                        }
                        tiles_done = tiles_done + 1;
                    }
                }
                if (cta_rank == 0) {
                    __threadfence();
                    unsigned int retired = atomicAdd(&sched[2], 1);
                    if (retired == (unsigned int)num_clusters - 1) {
                        __threadfence();
                        sched[0] = 0;
                        sched[1] = 2147483648;
                        sched[2] = 0;
                    }
                }
                #pragma unroll
                for (int _ = 0; _ < NUM_STAGES; _++) {
                    mbarrier_wait(mma_done_addr + (load_stage) * 8, _phase_mma_done);
                    load_stage += 1;
                    if (load_stage == 7) { load_stage = 0; _phase_mma_done ^= 1; }
                }
            }
        }
    }
    // ---- Role: mma ----
    if (warp == 1) {
        { // mma_main
            unsigned int mma_tma_stage = 0;
            unsigned int mma_epi_stage = 0;
            unsigned int work_stage_m = 0;
            int num_iters_m = K / BLOCK_K;
            unsigned int grid_n_m = (unsigned int)(N / BLOCK_N);
            unsigned int rows_1 = 0;
            int p_start_1 = 0;
            #pragma unroll 1
            for (int gp_1 = 0; gp_1 < (unsigned int)num_groups; gp_1++) {
                int p_end_1 = offs[gp_1];
                rows_1 = rows_1 + (unsigned int)((p_end_1 - p_start_1 + CLUSTER_M - 1) / CLUSTER_M);
                p_start_1 = p_end_1;
            }
            unsigned int cl_1 = (unsigned int)(c0 + c1);
            unsigned int w0_1 = grid_n_m * (unsigned int)c0;
            unsigned int w1_1 = grid_n_m * (unsigned int)c1;
            unsigned int t0_1 = rows_1 * w0_1 / cl_1;
            unsigned int t1_2 = rows_1 * grid_n_m - t0_1;
            unsigned int _phase_work_full_1 = 0;
            unsigned int _phase_epilogue_done = 1;
            unsigned int _phase_tma_full = 0;
            if (cta_rank == 0) {
                unsigned int ends_m = 0;
                unsigned int tiles_done_m = 0;
                #pragma unroll 1
                for (unsigned int _item_m = 0; _item_m < t0_1 + t1_2 + 2; _item_m++) {
                    mbarrier_wait(work_full_addr + (work_stage_m) * 8, _phase_work_full_1);
                    unsigned int ticket_2 = work_ring[work_stage_m * 4];
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    asm volatile(
                        "{\n\t"
                        ".reg .b32 remAddr32;\n\t"
                        "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                        "}"
                        :: "r"(work_empty_addr + (work_stage_m) * 8), "r"(0) : "memory");
                    work_stage_m += 1;
                    if (work_stage_m == 4) { work_stage_m = 0; _phase_work_full_1 ^= 1; }
                    unsigned int q_1 = ticket_2 >> 31;
                    unsigned int k_1 = ticket_2 & 2147483647;
                    unsigned int valid_1 = ((q_1 == 1) ? ((k_1 < t1_2) ? 1 : 0) : ((k_1 < t0_1) ? 1 : 0));
                    unsigned int w0s_1 = ((w0_1 > 0) ? w0_1 : 1);
                    unsigned int w1s_1 = ((w1_1 > 0) ? w1_1 : 1);
                    unsigned int r0_1 = ((k_1 + 1) * cl_1 - 1) / w0s_1;
                    unsigned int r1_1 = k_1 * cl_1 / w1s_1;
                    unsigned int n0_1 = k_1 - r0_1 * w0s_1 / cl_1;
                    unsigned int n1_1 = grid_n_m - 1 - (k_1 - (r1_1 * w1s_1 + cl_1 - 1) / cl_1);
                    unsigned int row_1 = ((q_1 == 1) ? r1_1 : r0_1);
                    unsigned int item_1 = ((q_1 == 1) ? n1_1 : n0_1);
                    if (valid_1 == 0) {
                        ends_m = ends_m + 1;
                    }
                    if (ends_m == 2) {
                        break;
                    }
                    if (valid_1 == 1) {
                        mbarrier_wait(epilogue_done_addr + (mma_epi_stage) * 8, _phase_epilogue_done);
                        #pragma unroll 1
                        for (int iter_k_m = 0; iter_k_m < num_iters_m; iter_k_m++) {
                            mbarrier_wait(tma_full_addr + (mma_tma_stage) * 8, _phase_tma_full);
                            asm volatile("tcgen05.fence::after_thread_sync;");
                            int init_flag = ((iter_k_m == 0) ? 1 : 0);
                            int _mma_a_lo_0 = (((smem_a_addr) >> 4) & 0x3FFF) + (mma_tma_stage) * 2048;
                            int _mma_b_lo_0 = (((smem_b_addr) >> 4) & 0x3FFF) + (mma_tma_stage) * 2048;
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
                    "mov.b32 id, 272630928;\n\t"
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
                    :: "r"(_mma_a_lo_0), "r"(_mma_b_lo_0), "r"((tmem_accum + (mma_epi_stage * 256))), "r"(((init_flag) ? 0 : 1)) : "memory");
                            elect_commit_cg2_multicast(mma_done_addr + (mma_tma_stage) * 8, (uint16_t)(3));
                            mma_tma_stage += 1;
                            if (mma_tma_stage == 7) { mma_tma_stage = 0; _phase_tma_full ^= 1; }
                        }
                        elect_commit_cg2_multicast(mainloop_done_addr + (mma_epi_stage) * 8, (uint16_t)(3));
                        mma_epi_stage += 1;
                        if (mma_epi_stage == 2) { mma_epi_stage = 0; _phase_epilogue_done ^= 1; }
                        tiles_done_m = tiles_done_m + 1;
                    }
                }
            }
        }
    }
    // ---- Role: epilogue ----
    if (warp >= 2 && warp <= 5) {
        { // epilogue_main
            unsigned int epi_stage = 0;
            unsigned int work_stage_e = 0;
            const int epi_warp = warp % 4;
            const int epi_tid = epi_warp * 32 + lane;
            unsigned int grid_n_e = (unsigned int)(N / BLOCK_N);
            unsigned int rows_2 = 0;
            int p_start_2 = 0;
            #pragma unroll 1
            for (int gp_2 = 0; gp_2 < (unsigned int)num_groups; gp_2++) {
                int p_end_2 = offs[gp_2];
                rows_2 = rows_2 + (unsigned int)((p_end_2 - p_start_2 + CLUSTER_M - 1) / CLUSTER_M);
                p_start_2 = p_end_2;
            }
            unsigned int cl_2 = (unsigned int)(c0 + c1);
            unsigned int w0_2 = grid_n_e * (unsigned int)c0;
            unsigned int w1_2 = grid_n_e * (unsigned int)c1;
            unsigned int t0_2 = rows_2 * w0_2 / cl_2;
            unsigned int t1_3 = rows_2 * grid_n_e - t0_2;
            unsigned int g_cur_e = 0;
            unsigned int g_row0_e = 0;
            int g_start_e = 0;
            int g_end_e = offs[0];
            unsigned int g_rows_e = (unsigned int)((g_end_e + CLUSTER_M - 1) / CLUSTER_M);
            unsigned int ends_e = 0;
            unsigned int tiles_done_e = 0;
            unsigned int _phase_work_full_2 = 0;
            unsigned int _phase_mainloop_done = 0;
            #pragma unroll 1
            for (unsigned int _item_e = 0; _item_e < t0_2 + t1_3 + 2; _item_e++) {
                mbarrier_wait(work_full_addr + (work_stage_e) * 8, _phase_work_full_2);
                unsigned int ticket_3 = work_ring[work_stage_e * 4];
                asm volatile("tcgen05.fence::before_thread_sync;");
                asm volatile(
                    "{\n\t"
                    ".reg .b32 remAddr32;\n\t"
                    "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                    "}"
                    :: "r"(work_empty_addr + (work_stage_e) * 8), "r"(0) : "memory");
                work_stage_e += 1;
                if (work_stage_e == 4) { work_stage_e = 0; _phase_work_full_2 ^= 1; }
                unsigned int q_2 = ticket_3 >> 31;
                unsigned int k_2 = ticket_3 & 2147483647;
                unsigned int valid_2 = ((q_2 == 1) ? ((k_2 < t1_3) ? 1 : 0) : ((k_2 < t0_2) ? 1 : 0));
                unsigned int w0s_2 = ((w0_2 > 0) ? w0_2 : 1);
                unsigned int w1s_2 = ((w1_2 > 0) ? w1_2 : 1);
                unsigned int r0_2 = ((k_2 + 1) * cl_2 - 1) / w0s_2;
                unsigned int r1_2 = k_2 * cl_2 / w1s_2;
                unsigned int n0_2 = k_2 - r0_2 * w0s_2 / cl_2;
                unsigned int n1_2 = grid_n_e - 1 - (k_2 - (r1_2 * w1s_2 + cl_2 - 1) / cl_2);
                unsigned int row_2 = ((q_2 == 1) ? r1_2 : r0_2);
                unsigned int item_2 = ((q_2 == 1) ? n1_2 : n0_2);
                if (valid_2 == 0) {
                    ends_e = ends_e + 1;
                }
                if (ends_e == 2) {
                    break;
                }
                if (valid_2 == 1) {
                    if (row_2 < g_row0_e) {
                        g_cur_e = 0;
                        g_row0_e = 0;
                        g_start_e = 0;
                        g_end_e = offs[0];
                        g_rows_e = (unsigned int)((g_end_e + CLUSTER_M - 1) / CLUSTER_M);
                    }
                    while (row_2 >= g_row0_e + g_rows_e) {
                        g_row0_e = g_row0_e + g_rows_e;
                        g_start_e = g_end_e;
                        g_cur_e = g_cur_e + 1;
                        g_end_e = offs[g_cur_e];
                        g_rows_e = (unsigned int)((g_end_e - g_start_e + CLUSTER_M - 1) / CLUSTER_M);
                    }
                    int row_g = g_start_e + (int)(row_2 - g_row0_e) * CLUSTER_M + (cta_rank & 1) * BLOCK_M + epi_tid;
                    mbarrier_wait(mainloop_done_addr + (epi_stage) * 8, _phase_mainloop_done);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    #pragma unroll
                    for (int n_chunk = 0; n_chunk < BLOCK_N / EPI_CHUNK; n_chunk++) {
                        int row_0 = (cta_rank & 1) * 128 + epi_warp * 32;
                        int col = epi_stage * (unsigned int)BLOCK_N + (unsigned int)(n_chunk * EPI_CHUNK);
                        int tmem_addr = taddr + (unsigned int)(row_0 << 16) + (unsigned int)col;
                        float _tmem_load_0[16];
                        tmem_ld_x16(&_tmem_load_0[0], tmem_addr);
                        asm volatile("tcgen05.wait::ld.sync.aligned;");
                        if (row_g < g_end_e) {
                            unsigned int _packed_0[8];
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 0], _tmem_load_0[_lp*2+1 + 0]));
                                _packed_0[_lp] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 2], _tmem_load_0[_lp*2+1 + 2]));
                                _packed_0[_lp + 1] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 4], _tmem_load_0[_lp*2+1 + 4]));
                                _packed_0[_lp + 2] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 6], _tmem_load_0[_lp*2+1 + 6]));
                                _packed_0[_lp + 3] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 8], _tmem_load_0[_lp*2+1 + 8]));
                                _packed_0[_lp + 4] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 10], _tmem_load_0[_lp*2+1 + 10]));
                                _packed_0[_lp + 5] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 12], _tmem_load_0[_lp*2+1 + 12]));
                                _packed_0[_lp + 6] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 14], _tmem_load_0[_lp*2+1 + 14]));
                                _packed_0[_lp + 7] = *(uint32_t*)&_bf2;
                            }
                            {
                                const unsigned* _raw_stv8_0 = reinterpret_cast<const unsigned*>(_packed_0);
                                asm volatile(
                                    "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                    :: "l"((void*)(C + (row_g * ldc + ((int)item_2 * BLOCK_N + n_chunk * EPI_CHUNK)))), "r"(_raw_stv8_0[0]), "r"(_raw_stv8_0[1]), "r"(_raw_stv8_0[2]), "r"(_raw_stv8_0[3]), "r"(_raw_stv8_0[4]), "r"(_raw_stv8_0[5]), "r"(_raw_stv8_0[6]), "r"(_raw_stv8_0[7]) : "memory");
                            }
                        }
                    }
                    if (elect_sync()) {
                        asm volatile("tcgen05.fence::before_thread_sync;");
                        asm volatile(
                            "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                            :: "r"((epilogue_done_addr + (epi_stage) * 8) & 0xFEFFFFFF) : "memory");
                    }
                    epi_stage += 1;
                    if (epi_stage == 2) { epi_stage = 0; _phase_mainloop_done ^= 1; }
                    tiles_done_e = tiles_done_e + 1;
                }
            }
        }
    }

    // Cleanup
    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
    asm volatile("barrier.cluster.arrive.release.aligned;");
    asm volatile("barrier.cluster.wait.acquire.aligned;");

    if (warp == 0) {
        asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(512));
    }
}

} // extern "C"
