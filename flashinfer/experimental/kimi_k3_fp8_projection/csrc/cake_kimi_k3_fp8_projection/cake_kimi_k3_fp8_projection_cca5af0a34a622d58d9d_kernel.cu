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
#define TMEM_NCOLS 416
#define TMEM_ACCUM_OFFSET 0
#define TMEM_TMEM_SFA_OFFSET 384
#define TMEM_TMEM_SFB_OFFSET 392
#define NUM_TMA_PIPE_STAGES 5
#define NUM_MAINLOOP_PIPE_STAGES 1
#define NUM_WORK_PIPE_STAGES 4
#define SMEM_SMEM_OUT_OFF 206848
#define SMEM_SMEM_OUT_STAGE_BYTES 8192
#define SMEM_SMEM_OUT_STRIDE 8192
#define SMEM_SMEM_A_OFF 1024
#define SMEM_SMEM_A_STAGE_BYTES 16384
#define SMEM_SMEM_A_STRIDE 40960
#define SMEM_SMEM_B_OFF 17408
#define SMEM_SMEM_B_STAGE_BYTES 22528
#define SMEM_SMEM_B_STRIDE 40960
#define SMEM_SMEM_SFA_ALL_OFF 39936
#define SMEM_SMEM_SFA_ALL_STAGE_BYTES 512
#define SMEM_SMEM_SFA_ALL_STRIDE 40960
#define SMEM_SMEM_SFB_ALL_OFF 40448
#define SMEM_SMEM_SFB_ALL_STAGE_BYTES 1536
#define SMEM_SMEM_SFB_ALL_STRIDE 40960
#define SMEM_SMEM_SFB_P0_OFF 40448
#define SMEM_SMEM_SFB_P0_STAGE_BYTES 512
#define SMEM_SMEM_SFB_P0_STRIDE 40960
#define SMEM_SMEM_SFB_P1_OFF 40960
#define SMEM_SMEM_SFB_P1_STAGE_BYTES 512
#define SMEM_SMEM_SFB_P1_STRIDE 40960
#define SMEM_SMEM_SFB_P2_OFF 41472
#define SMEM_SMEM_SFB_P2_STAGE_BYTES 512
#define SMEM_SMEM_SFB_P2_STRIDE 40960
#define SMEM_WORK_RESPONSE_OFF 205824
#define SMEM_WORK_RESPONSE_STAGE_BYTES 16
#define SMEM_WORK_RESPONSE_STRIDE 16
#define SMEM_TOTAL 215040
#define THREADS 320
#define BLOCK_M 128
#define BLOCK_N 256
#define B_HALF_N 128
#define BLOCK_K 256
#define CTA_GROUP 2
#define NUM_STAGES 3
#define GROUP_M 32
#define WORK_STAGES 4

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



__device__ __forceinline__ void tcgen05_mma_mxf8_bs_cta2_leader(
    int taddr, uint64_t a_desc, uint64_t b_desc, uint32_t i_desc,
    int sfa_taddr, int sfb_taddr, int enable_input_d, uint32_t leader) {
    asm volatile(
        "{\n\t"
        ".reg .pred leader, p;\n\t"
        "setp.ne.b32 leader, %7, 0;\n\t"
        "setp.ne.b32 p, %6, 0;\n\t"
        "@leader tcgen05.mma.cta_group::2.kind::mxf8f6f4.block_scale [%0], %1, %2, %3, [%4], [%5], p;\n\t"
        "}\n"
        :: "r"(taddr), "l"(a_desc), "l"(b_desc), "r"(i_desc), "r"(sfa_taddr), "r"(sfb_taddr), "r"(enable_input_d), "r"(leader) : "memory");
}


__device__ __forceinline__ uint64_t desc_encode(uint64_t x) {
    return (x & 0x3FFFFULL) >> 4ULL;
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







__device__ __forceinline__ uint64_t make_sf_cp_desc_lo_sbo128(int lo) {
    const int SBO = 128;
    return (uint64_t)(uint32_t)lo
         | (desc_encode(SBO) << 32ULL)
         | (1ULL << 46ULL);
}



__device__ __forceinline__ void tcgen05_cp_32x128b_warpx4_cta2_leader(
    int taddr, uint64_t s_desc, uint32_t leader) {
    asm volatile(
        "{\n\t"
        ".reg .pred leader;\n\t"
        "setp.ne.b32 leader, %2, 0;\n\t"
        "@leader tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;\n\t"
        "}\n"
        :: "r"(taddr), "l"(s_desc), "r"(leader) : "memory");
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



__device__ __forceinline__ void tma_store_2d(
    const void *tmap, int x, int y, unsigned smem_addr) {
    asm volatile(
        "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group"
        " [%0, {%1, %2}], [%3];"
        :: "l"(tmap), "r"(x), "r"(y), "r"(smem_addr) : "memory");
}



extern "C" {

__global__ __launch_bounds__(320) __cluster_dims__(2,1,1) void
kernel_cake_kimi_k3_fp8_projection_cca5af0a34a622d58d9d(const __grid_constant__ CUtensorMap A, const __grid_constant__ CUtensorMap B, const __grid_constant__ CUtensorMap SFA, const __grid_constant__ CUtensorMap SFB, __nv_bfloat16* __restrict__ out, const __grid_constant__ CUtensorMap OUT, int M, int m_tiles, int n_tiles, int n_valid, int ldo, int store_vec, int num_k_iters, int sf_k_tiles, __nv_bfloat16* __restrict__ x, int K, float* __restrict__ sk_partials, unsigned int* __restrict__ sk_flags, int sk_pairs, int sk_rem, int sk_ksplit, int sk_dp)
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
    #define mma_done_addr (mbar_base + 40)
    #define mainloop_done_addr (mbar_base + 80)
    #define epilogue_done_addr (mbar_base + 88)
    #define work_full_addr (mbar_base + 96)
    #define work_empty_addr (mbar_base + 128)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    const unsigned int clusters_x = gridDim.x / 2;
    const unsigned int cluster_id = ((blockIdx.z * gridDim.y + blockIdx.y) * clusters_x) + blockIdx.x / 2;
    const unsigned int num_clusters = clusters_x * gridDim.y * gridDim.z;

    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));

    // Kernel setup ops
    __nv_bfloat16* smem_out = reinterpret_cast<__nv_bfloat16*>(smem_raw + 206848);
    const int smem_out_addr = smem + 206848;
    uint8_t* smem_a = reinterpret_cast<uint8_t*>(smem_raw + 1024);
    const int smem_a_addr = smem + 1024;
    uint8_t* smem_b = reinterpret_cast<uint8_t*>(smem_raw + 17408);
    const int smem_b_addr = smem + 17408;
    uint8_t* smem_sfa_all = reinterpret_cast<uint8_t*>(smem_raw + 39936);
    const int smem_sfa_all_addr = smem + 39936;
    uint8_t* smem_sfb_all = reinterpret_cast<uint8_t*>(smem_raw + 40448);
    const int smem_sfb_all_addr = smem + 40448;
    uint8_t* smem_sfb_p0 = reinterpret_cast<uint8_t*>(smem_raw + 40448);
    const int smem_sfb_p0_addr = smem + 40448;
    uint8_t* smem_sfb_p1 = reinterpret_cast<uint8_t*>(smem_raw + 40960);
    const int smem_sfb_p1_addr = smem + 40960;
    uint8_t* smem_sfb_p2 = reinterpret_cast<uint8_t*>(smem_raw + 41472);
    const int smem_sfb_p2_addr = smem + 41472;
    unsigned int* work_response = reinterpret_cast<unsigned int*>(smem_raw + 205824);
    const int work_response_addr = smem + 205824;

    // Mbarrier init (6 pipeline groups, 0 ordered-sequence groups, 20 barriers)
    // Mbarriers at smem_raw[0..160)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'tma_pipe' ---
            // tma_full: 5 barriers, init_count=2
            mbarrier_init(smem + 0, 2);
            mbarrier_init(smem + 8, 2);
            mbarrier_init(smem + 16, 2);
            mbarrier_init(smem + 24, 2);
            mbarrier_init(smem + 32, 2);
            // mma_done: 5 barriers, init_count=1
            mbarrier_init(smem + 40, 1);
            mbarrier_init(smem + 48, 1);
            mbarrier_init(smem + 56, 1);
            mbarrier_init(smem + 64, 1);
            mbarrier_init(smem + 72, 1);
            // --- pipeline 'mainloop_pipe' ---
            // mainloop_done: 1 barriers, init_count=1
            mbarrier_init(smem + 80, 1);
            // epilogue_done: 1 barriers, init_count=16
            mbarrier_init(smem + 88, 16);
            // --- pipeline 'work_pipe' ---
            // work_full: 4 barriers, init_count=1
            mbarrier_init(smem + 96, 1);
            mbarrier_init(smem + 104, 1);
            mbarrier_init(smem + 112, 1);
            mbarrier_init(smem + 120, 1);
            // work_empty: 4 barriers, init_count=546
            mbarrier_init(smem + 128, 546);
            mbarrier_init(smem + 136, 546);
            mbarrier_init(smem + 144, 546);
            mbarrier_init(smem + 152, 546);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();
    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");

    // TMEM alloc (512 columns, 416 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 160);
    if (warp == 0) {
        int _tmem_hold = smem + 160;
        asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(512) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_accum = taddr;
    const int tmem_tmem_sfa = taddr + 384;
    const int tmem_tmem_sfb = taddr + 392;

    // ---- Role: load ----
    if (warp == 0) {
        { // load_main
            int num_cluster_tiles = m_tiles / CTA_GROUP * n_tiles;
            unsigned int load_stage = 0;
            unsigned int work_stage = 0;
            asm volatile("griddepcontrol.wait;" ::: "memory");
            unsigned int _phase_work_empty = 1;
            unsigned int _phase_mma_done = 1;
            unsigned int _phase_work_full = 0;
            if (elect_sync()) {
                unsigned int this_bid = bid;
                #pragma unroll 1
                for (unsigned int _tile_iter = 0; _tile_iter < num_cluster_tiles; _tile_iter++) {
                    if (cta_rank == 0) {
                        mbarrier_wait_cluster_hint(work_empty_addr + (work_stage) * 8, _phase_work_empty, 10000000);
                        asm volatile(
                            "{\n\t"
                            ".reg .b32 remAddr32;\n\t"
                            "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                            "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [remAddr32], %2;\n\t"
                            "}"
                            :: "r"(work_full_addr + (work_stage) * 8), "r"(0), "r"((uint32_t)(16)) : "memory");
                        asm volatile(
                            "{\n\t"
                            ".reg .b32 remAddr32;\n\t"
                            "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                            "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [remAddr32], %2;\n\t"
                            "}"
                            :: "r"(work_full_addr + (work_stage) * 8), "r"(1), "r"((uint32_t)(16)) : "memory");
                        asm volatile(
                            "fence.proxy.async.shared::cta;\n\t"
                            "clusterlaunchcontrol.try_cancel.async.shared::cta"
                                ".mbarrier::complete_tx::bytes.multicast::cluster::all.b128"
                                " [%0], [%1];"
                            :: "r"(work_response_addr + work_stage * 16 + 0 * 16), "r"(work_full_addr + work_stage * 8)
                            : "memory");
                    }
                    int tiles_per_group = GROUP_M * n_tiles;
                    int group = this_bid / (unsigned int)tiles_per_group;
                    int first_m = group * GROUP_M;
                    int remaining = m_tiles - first_m;
                    int group_size = ((remaining >= GROUP_M) ? GROUP_M : remaining);
                    int local = this_bid % (unsigned int)tiles_per_group;
                    int bid_m = first_m + local % group_size;
                    int bid_n = local / group_size;
                    int off_m = bid_m * BLOCK_M;
                    int off_n = bid_n * 352;
                    int sfa_tile_row = bid_m * sf_k_tiles;
                    int sfb_tile_row = bid_n * sf_k_tiles;
                    #pragma unroll 1
                    for (int iter_k = 0; iter_k < num_k_iters; iter_k++) {
                        int k_group = iter_k * 2;
                        mbarrier_wait(mma_done_addr + (load_stage) * 8, _phase_mma_done);
                        tma_3d_gmem2smem_cta2(smem_a_addr + load_stage * 40960, (&A), 0, off_m, k_group + 0, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                        tma_3d_gmem2smem_cta2(smem_sfa_all_addr + load_stage * 40960, (&SFA), 0, 0, sfa_tile_row + k_group + 0, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                        tma_3d_gmem2smem_cta2(smem_b_addr + load_stage * 40960 + 0, (&B), 0, (off_n + cta_rank * 64 + 0) % 128, (off_n + cta_rank * 64 + 0) / 128 * (2 * num_k_iters) + k_group + 0, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                        tma_3d_gmem2smem_cta2(smem_b_addr + load_stage * 40960 + 2048, (&B), 0, (off_n + cta_rank * 64 + 16) % 128, (off_n + cta_rank * 64 + 16) / 128 * (2 * num_k_iters) + k_group + 0, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                        tma_3d_gmem2smem_cta2(smem_b_addr + load_stage * 40960 + 4096, (&B), 0, (off_n + cta_rank * 64 + 32) % 128, (off_n + cta_rank * 64 + 32) / 128 * (2 * num_k_iters) + k_group + 0, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                        tma_3d_gmem2smem_cta2(smem_b_addr + load_stage * 40960 + 6144, (&B), 0, (off_n + cta_rank * 64 + 48) % 128, (off_n + cta_rank * 64 + 48) / 128 * (2 * num_k_iters) + k_group + 0, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                        tma_3d_gmem2smem_cta2(smem_b_addr + load_stage * 40960 + 8192, (&B), 0, (off_n + cta_rank * 64 + 128) % 128, (off_n + cta_rank * 64 + 128) / 128 * (2 * num_k_iters) + k_group + 0, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                        tma_3d_gmem2smem_cta2(smem_b_addr + load_stage * 40960 + 10240, (&B), 0, (off_n + cta_rank * 64 + 144) % 128, (off_n + cta_rank * 64 + 144) / 128 * (2 * num_k_iters) + k_group + 0, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                        tma_3d_gmem2smem_cta2(smem_b_addr + load_stage * 40960 + 12288, (&B), 0, (off_n + cta_rank * 64 + 160) % 128, (off_n + cta_rank * 64 + 160) / 128 * (2 * num_k_iters) + k_group + 0, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                        tma_3d_gmem2smem_cta2(smem_b_addr + load_stage * 40960 + 14336, (&B), 0, (off_n + cta_rank * 64 + 176) % 128, (off_n + cta_rank * 64 + 176) / 128 * (2 * num_k_iters) + k_group + 0, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                        tma_3d_gmem2smem_cta2(smem_b_addr + load_stage * 40960 + 16384, (&B), 0, (off_n + cta_rank * 48 + 256) % 128, (off_n + cta_rank * 48 + 256) / 128 * (2 * num_k_iters) + k_group + 0, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                        tma_3d_gmem2smem_cta2(smem_b_addr + load_stage * 40960 + 18432, (&B), 0, (off_n + cta_rank * 48 + 272) % 128, (off_n + cta_rank * 48 + 272) / 128 * (2 * num_k_iters) + k_group + 0, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                        tma_3d_gmem2smem_cta2(smem_b_addr + load_stage * 40960 + 20480, (&B), 0, (off_n + cta_rank * 48 + 288) % 128, (off_n + cta_rank * 48 + 288) / 128 * (2 * num_k_iters) + k_group + 0, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                        tma_3d_gmem2smem_cta2(smem_sfb_all_addr + load_stage * 40960, (&SFB), 0, 0, sfb_tile_row + k_group + 0, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                        asm volatile(
                            "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                            :: "r"((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF), "r"((uint32_t)(40960)) : "memory");
                        load_stage += 1;
                        if (load_stage == 5) { load_stage = 0; _phase_mma_done ^= 1; }
                        mbarrier_wait(mma_done_addr + (load_stage) * 8, _phase_mma_done);
                        tma_3d_gmem2smem_cta2(smem_a_addr + load_stage * 40960, (&A), 0, off_m, k_group + 1, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                        tma_3d_gmem2smem_cta2(smem_sfa_all_addr + load_stage * 40960, (&SFA), 0, 0, sfa_tile_row + k_group + 1, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                        tma_3d_gmem2smem_cta2(smem_b_addr + load_stage * 40960 + 0, (&B), 0, (off_n + cta_rank * 64 + 0) % 128, (off_n + cta_rank * 64 + 0) / 128 * (2 * num_k_iters) + k_group + 1, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                        tma_3d_gmem2smem_cta2(smem_b_addr + load_stage * 40960 + 2048, (&B), 0, (off_n + cta_rank * 64 + 16) % 128, (off_n + cta_rank * 64 + 16) / 128 * (2 * num_k_iters) + k_group + 1, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                        tma_3d_gmem2smem_cta2(smem_b_addr + load_stage * 40960 + 4096, (&B), 0, (off_n + cta_rank * 64 + 32) % 128, (off_n + cta_rank * 64 + 32) / 128 * (2 * num_k_iters) + k_group + 1, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                        tma_3d_gmem2smem_cta2(smem_b_addr + load_stage * 40960 + 6144, (&B), 0, (off_n + cta_rank * 64 + 48) % 128, (off_n + cta_rank * 64 + 48) / 128 * (2 * num_k_iters) + k_group + 1, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                        tma_3d_gmem2smem_cta2(smem_b_addr + load_stage * 40960 + 8192, (&B), 0, (off_n + cta_rank * 64 + 128) % 128, (off_n + cta_rank * 64 + 128) / 128 * (2 * num_k_iters) + k_group + 1, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                        tma_3d_gmem2smem_cta2(smem_b_addr + load_stage * 40960 + 10240, (&B), 0, (off_n + cta_rank * 64 + 144) % 128, (off_n + cta_rank * 64 + 144) / 128 * (2 * num_k_iters) + k_group + 1, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                        tma_3d_gmem2smem_cta2(smem_b_addr + load_stage * 40960 + 12288, (&B), 0, (off_n + cta_rank * 64 + 160) % 128, (off_n + cta_rank * 64 + 160) / 128 * (2 * num_k_iters) + k_group + 1, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                        tma_3d_gmem2smem_cta2(smem_b_addr + load_stage * 40960 + 14336, (&B), 0, (off_n + cta_rank * 64 + 176) % 128, (off_n + cta_rank * 64 + 176) / 128 * (2 * num_k_iters) + k_group + 1, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                        tma_3d_gmem2smem_cta2(smem_b_addr + load_stage * 40960 + 16384, (&B), 0, (off_n + cta_rank * 48 + 256) % 128, (off_n + cta_rank * 48 + 256) / 128 * (2 * num_k_iters) + k_group + 1, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                        tma_3d_gmem2smem_cta2(smem_b_addr + load_stage * 40960 + 18432, (&B), 0, (off_n + cta_rank * 48 + 272) % 128, (off_n + cta_rank * 48 + 272) / 128 * (2 * num_k_iters) + k_group + 1, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                        tma_3d_gmem2smem_cta2(smem_b_addr + load_stage * 40960 + 20480, (&B), 0, (off_n + cta_rank * 48 + 288) % 128, (off_n + cta_rank * 48 + 288) / 128 * (2 * num_k_iters) + k_group + 1, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                        tma_3d_gmem2smem_cta2(smem_sfb_all_addr + load_stage * 40960, (&SFB), 0, 0, sfb_tile_row + k_group + 1, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                        asm volatile(
                            "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                            :: "r"((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF), "r"((uint32_t)(40960)) : "memory");
                        load_stage += 1;
                        if (load_stage == 5) { load_stage = 0; _phase_mma_done ^= 1; }
                    }
                    mbarrier_wait(work_full_addr + (work_stage) * 8, _phase_work_full);
                    uint32_t clc_valid = 0;
                    uint32_t clc_ctaid;
                    uint32_t clc_ctaid_y;
                    uint32_t clc_ctaid_z;
                    asm volatile(
                        "{\n\t"
                        ".reg .pred p1;\n\t"
                        ".reg .b128 clc_r;\n\t"
                        "ld.shared.b128 clc_r, [%4];\n\t"
                        "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                        "selp.u32 %3, 1, 0, p1;\n\t"
                        "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid.v4.b32.b128 {%0, %1, %2, _}, clc_r;\n\t"
                        "}\n"
                        : "=r"(clc_ctaid), "=r"(clc_ctaid_y), "=r"(clc_ctaid_z), "=r"(clc_valid)
                        : "r"(work_response_addr + work_stage * 16 + 0 * 16)
                        : "memory");
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    asm volatile(
                        "{\n\t"
                        ".reg .b32 remAddr32;\n\t"
                        "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                        "}"
                        :: "r"(work_empty_addr + (work_stage) * 8), "r"(0) : "memory");
                    work_stage += 1;
                    if (work_stage == 4) { work_stage = 0; _phase_work_empty ^= 1; _phase_work_full ^= 1; }
                    if (clc_valid == 0) {
                        break;
                    }
                    this_bid = clc_ctaid + (unsigned int)cta_rank;
                }
                asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
                #pragma unroll 1
                for (int _ = 0; _ < 5; _++) {
                    mbarrier_wait(mma_done_addr + (load_stage) * 8, _phase_mma_done);
                    load_stage += 1;
                    if (load_stage == 5) { load_stage = 0; _phase_mma_done ^= 1; }
                }
            }
        }
    }
    // ---- Role: mma ----
    if (warp == 1) {
        { // mma_main
            int num_cluster_tiles_1 = m_tiles / CTA_GROUP * n_tiles;
            unsigned int mma_tma_stage = 0;
            unsigned int work_stage_1 = 0;
            unsigned int _phase_epilogue_done = 1;
            unsigned int _phase_tma_full = 0;
            unsigned int _phase_work_full_1 = 0;
            if (cta_rank == 0) {
                #pragma unroll 1
                for (unsigned int _tile_iter_1 = 0; _tile_iter_1 < num_cluster_tiles_1; _tile_iter_1++) {
                    mbarrier_wait(epilogue_done_addr, _phase_epilogue_done);
                    #pragma unroll 1
                    for (int iter_k_1 = 0; iter_k_1 < num_k_iters; iter_k_1++) {
                        int init_flag = ((iter_k_1 == 0) ? 1 : 0);
                        mbarrier_wait(tma_full_addr + (mma_tma_stage) * 8, _phase_tma_full);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        uint32_t _elected_0 = elect_sync();
                        tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfa, make_sf_cp_desc_lo_sbo128((((smem_sfa_all_addr) >> 4) + (mma_tma_stage) * 2560)), _elected_0);
                        tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfb, make_sf_cp_desc_lo_sbo128((((smem_sfb_p0_addr) >> 4) + (mma_tma_stage) * 2560)), _elected_0);
                        tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfb + 4, make_sf_cp_desc_lo_sbo128((((smem_sfb_p1_addr) >> 4) + (mma_tma_stage) * 2560)), _elected_0);
                        tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfb + 8, make_sf_cp_desc_lo_sbo128((((smem_sfb_p2_addr) >> 4) + (mma_tma_stage) * 2560)), _elected_0);
                        int _smem_desc_lo_1 = (((smem_a_addr) >> 4) & 0x3FFF) + (mma_tma_stage) * 2560;
                        int _smem_desc_lo_2 = (((smem_b_addr) >> 4) & 0x3FFF) + (mma_tma_stage) * 2560;
                        {
                            uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_1) | ((uint64_t)0x40004040 << 32);
                            uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_2) | ((uint64_t)0x40004040 << 32);

                            tcgen05_mma_mxf8_bs_cta2_leader(tmem_accum, a_desc + 0, b_desc + 0,
                                0x10a00000U, tmem_tmem_sfa, tmem_tmem_sfb, ((init_flag) ? 0 : 1), _elected_0);
                            tcgen05_mma_mxf8_bs_cta2_leader(tmem_accum, a_desc + 2, b_desc + 2,
                                0x30a00010U, tmem_tmem_sfa, tmem_tmem_sfb, 1, _elected_0);
                            tcgen05_mma_mxf8_bs_cta2_leader(tmem_accum, a_desc + 4, b_desc + 4,
                                0x50a00020U, tmem_tmem_sfa, tmem_tmem_sfb, 1, _elected_0);
                            tcgen05_mma_mxf8_bs_cta2_leader(tmem_accum, a_desc + 6, b_desc + 6,
                                0x70a00030U, tmem_tmem_sfa, tmem_tmem_sfb, 1, _elected_0);
                        }
                        int _smem_desc_lo_3 = (((smem_a_addr) >> 4) & 0x3FFF) + (mma_tma_stage) * 2560;
                        int _smem_desc_lo_4 = (((smem_b_addr + 8192) >> 4) & 0x3FFF) + (mma_tma_stage) * 2560;
                        {
                            uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_3) | ((uint64_t)0x40004040 << 32);
                            uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_4) | ((uint64_t)0x40004040 << 32);

                            tcgen05_mma_mxf8_bs_cta2_leader((tmem_accum + (128)), a_desc + 0, b_desc + 0,
                                0x10a00000U, tmem_tmem_sfa, tmem_tmem_sfb + 4, ((init_flag) ? 0 : 1), _elected_0);
                            tcgen05_mma_mxf8_bs_cta2_leader((tmem_accum + (128)), a_desc + 2, b_desc + 2,
                                0x30a00010U, tmem_tmem_sfa, tmem_tmem_sfb + 4, 1, _elected_0);
                            tcgen05_mma_mxf8_bs_cta2_leader((tmem_accum + (128)), a_desc + 4, b_desc + 4,
                                0x50a00020U, tmem_tmem_sfa, tmem_tmem_sfb + 4, 1, _elected_0);
                            tcgen05_mma_mxf8_bs_cta2_leader((tmem_accum + (128)), a_desc + 6, b_desc + 6,
                                0x70a00030U, tmem_tmem_sfa, tmem_tmem_sfb + 4, 1, _elected_0);
                        }
                        int _smem_desc_lo_5 = (((smem_a_addr) >> 4) & 0x3FFF) + (mma_tma_stage) * 2560;
                        int _smem_desc_lo_6 = (((smem_b_addr + 16384) >> 4) & 0x3FFF) + (mma_tma_stage) * 2560;
                        {
                            uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_5) | ((uint64_t)0x40004040 << 32);
                            uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_6) | ((uint64_t)0x40004040 << 32);

                            tcgen05_mma_mxf8_bs_cta2_leader((tmem_accum + (256)), a_desc + 0, b_desc + 0,
                                0x10980000U, tmem_tmem_sfa, tmem_tmem_sfb + 8, ((init_flag) ? 0 : 1), _elected_0);
                            tcgen05_mma_mxf8_bs_cta2_leader((tmem_accum + (256)), a_desc + 2, b_desc + 2,
                                0x30980010U, tmem_tmem_sfa, tmem_tmem_sfb + 8, 1, _elected_0);
                            tcgen05_mma_mxf8_bs_cta2_leader((tmem_accum + (256)), a_desc + 4, b_desc + 4,
                                0x50980020U, tmem_tmem_sfa, tmem_tmem_sfb + 8, 1, _elected_0);
                            tcgen05_mma_mxf8_bs_cta2_leader((tmem_accum + (256)), a_desc + 6, b_desc + 6,
                                0x70980030U, tmem_tmem_sfa, tmem_tmem_sfb + 8, 1, _elected_0);
                        }
                        elect_commit_cg2_multicast(mma_done_addr + (mma_tma_stage) * 8, (uint16_t)(3));
                        mma_tma_stage += 1;
                        if (mma_tma_stage == 5) { mma_tma_stage = 0; _phase_tma_full ^= 1; }
                        mbarrier_wait(tma_full_addr + (mma_tma_stage) * 8, _phase_tma_full);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        uint32_t _elected_7 = elect_sync();
                        tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfa + 4, make_sf_cp_desc_lo_sbo128((((smem_sfa_all_addr) >> 4) + (mma_tma_stage) * 2560)), _elected_7);
                        tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfb + 12, make_sf_cp_desc_lo_sbo128((((smem_sfb_p0_addr) >> 4) + (mma_tma_stage) * 2560)), _elected_7);
                        tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfb + 16, make_sf_cp_desc_lo_sbo128((((smem_sfb_p1_addr) >> 4) + (mma_tma_stage) * 2560)), _elected_7);
                        tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfb + 20, make_sf_cp_desc_lo_sbo128((((smem_sfb_p2_addr) >> 4) + (mma_tma_stage) * 2560)), _elected_7);
                        int _smem_desc_lo_8 = (((smem_a_addr) >> 4) & 0x3FFF) + (mma_tma_stage) * 2560;
                        int _smem_desc_lo_9 = (((smem_b_addr) >> 4) & 0x3FFF) + (mma_tma_stage) * 2560;
                        {
                            uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_8) | ((uint64_t)0x40004040 << 32);
                            uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_9) | ((uint64_t)0x40004040 << 32);

                            tcgen05_mma_mxf8_bs_cta2_leader(tmem_accum, a_desc + 0, b_desc + 0,
                                0x10a00000U, tmem_tmem_sfa + 4, tmem_tmem_sfb + 12, 1, _elected_7);
                            tcgen05_mma_mxf8_bs_cta2_leader(tmem_accum, a_desc + 2, b_desc + 2,
                                0x30a00010U, tmem_tmem_sfa + 4, tmem_tmem_sfb + 12, 1, _elected_7);
                            tcgen05_mma_mxf8_bs_cta2_leader(tmem_accum, a_desc + 4, b_desc + 4,
                                0x50a00020U, tmem_tmem_sfa + 4, tmem_tmem_sfb + 12, 1, _elected_7);
                            tcgen05_mma_mxf8_bs_cta2_leader(tmem_accum, a_desc + 6, b_desc + 6,
                                0x70a00030U, tmem_tmem_sfa + 4, tmem_tmem_sfb + 12, 1, _elected_7);
                        }
                        int _smem_desc_lo_10 = (((smem_a_addr) >> 4) & 0x3FFF) + (mma_tma_stage) * 2560;
                        int _smem_desc_lo_11 = (((smem_b_addr + 8192) >> 4) & 0x3FFF) + (mma_tma_stage) * 2560;
                        {
                            uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_10) | ((uint64_t)0x40004040 << 32);
                            uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_11) | ((uint64_t)0x40004040 << 32);

                            tcgen05_mma_mxf8_bs_cta2_leader((tmem_accum + (128)), a_desc + 0, b_desc + 0,
                                0x10a00000U, tmem_tmem_sfa + 4, tmem_tmem_sfb + 16, 1, _elected_7);
                            tcgen05_mma_mxf8_bs_cta2_leader((tmem_accum + (128)), a_desc + 2, b_desc + 2,
                                0x30a00010U, tmem_tmem_sfa + 4, tmem_tmem_sfb + 16, 1, _elected_7);
                            tcgen05_mma_mxf8_bs_cta2_leader((tmem_accum + (128)), a_desc + 4, b_desc + 4,
                                0x50a00020U, tmem_tmem_sfa + 4, tmem_tmem_sfb + 16, 1, _elected_7);
                            tcgen05_mma_mxf8_bs_cta2_leader((tmem_accum + (128)), a_desc + 6, b_desc + 6,
                                0x70a00030U, tmem_tmem_sfa + 4, tmem_tmem_sfb + 16, 1, _elected_7);
                        }
                        int _smem_desc_lo_12 = (((smem_a_addr) >> 4) & 0x3FFF) + (mma_tma_stage) * 2560;
                        int _smem_desc_lo_13 = (((smem_b_addr + 16384) >> 4) & 0x3FFF) + (mma_tma_stage) * 2560;
                        {
                            uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_12) | ((uint64_t)0x40004040 << 32);
                            uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_13) | ((uint64_t)0x40004040 << 32);

                            tcgen05_mma_mxf8_bs_cta2_leader((tmem_accum + (256)), a_desc + 0, b_desc + 0,
                                0x10980000U, tmem_tmem_sfa + 4, tmem_tmem_sfb + 20, 1, _elected_7);
                            tcgen05_mma_mxf8_bs_cta2_leader((tmem_accum + (256)), a_desc + 2, b_desc + 2,
                                0x30980010U, tmem_tmem_sfa + 4, tmem_tmem_sfb + 20, 1, _elected_7);
                            tcgen05_mma_mxf8_bs_cta2_leader((tmem_accum + (256)), a_desc + 4, b_desc + 4,
                                0x50980020U, tmem_tmem_sfa + 4, tmem_tmem_sfb + 20, 1, _elected_7);
                            tcgen05_mma_mxf8_bs_cta2_leader((tmem_accum + (256)), a_desc + 6, b_desc + 6,
                                0x70980030U, tmem_tmem_sfa + 4, tmem_tmem_sfb + 20, 1, _elected_7);
                        }
                        elect_commit_cg2_multicast(mma_done_addr + (mma_tma_stage) * 8, (uint16_t)(3));
                        mma_tma_stage += 1;
                        if (mma_tma_stage == 5) { mma_tma_stage = 0; _phase_tma_full ^= 1; }
                    }
                    elect_commit_cg2_multicast(mainloop_done_addr, (uint16_t)(3));
                    _phase_epilogue_done ^= 1;
                    mbarrier_wait(work_full_addr + (work_stage_1) * 8, _phase_work_full_1);
                    uint32_t clc_valid_1 = 0;
                    uint32_t clc_ctaid_1;
                    uint32_t clc_ctaid_y_1;
                    uint32_t clc_ctaid_z_1;
                    asm volatile(
                        "{\n\t"
                        ".reg .pred p1;\n\t"
                        ".reg .b128 clc_r;\n\t"
                        "ld.shared.b128 clc_r, [%4];\n\t"
                        "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                        "selp.u32 %3, 1, 0, p1;\n\t"
                        "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid.v4.b32.b128 {%0, %1, %2, _}, clc_r;\n\t"
                        "}\n"
                        : "=r"(clc_ctaid_1), "=r"(clc_ctaid_y_1), "=r"(clc_ctaid_z_1), "=r"(clc_valid_1)
                        : "r"(work_response_addr + work_stage_1 * 16 + 0 * 16)
                        : "memory");
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    asm volatile(
                        "{\n\t"
                        ".reg .b32 remAddr32;\n\t"
                        "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                        "}"
                        :: "r"(work_empty_addr + (work_stage_1) * 8), "r"(0) : "memory");
                    work_stage_1 += 1;
                    if (work_stage_1 == 4) { work_stage_1 = 0; _phase_work_full_1 ^= 1; }
                    if (clc_valid_1 == 0) {
                        break;
                    }
                }
            }
        }
    }
    // ---- Role: epilogue ----
    if (warp >= 2 && warp <= 9) {
        { // epilogue_main
            int num_cluster_tiles_2 = m_tiles / CTA_GROUP * n_tiles;
            unsigned int work_stage_2 = 0;
            const int epi_warp = warp % 4;
            const int col_part = (warp - 2) / 4;
            const int local_row = epi_warp * 32 + lane;
            unsigned int this_bid_1 = bid;
            int sk_lane_addr = taddr + (unsigned int)(epi_warp * 32 << 16) + (unsigned int)(col_part * 176);
            unsigned int _phase_mainloop_done = 0;
            unsigned int _phase_work_full_2 = 0;
            #pragma unroll 1
            for (unsigned int _tile_iter_2 = 0; _tile_iter_2 < num_cluster_tiles_2; _tile_iter_2++) {
                int tiles_per_group_1 = GROUP_M * n_tiles;
                int group_1 = this_bid_1 / (unsigned int)tiles_per_group_1;
                int first_m_1 = group_1 * GROUP_M;
                int remaining_1 = m_tiles - first_m_1;
                int group_size_1 = ((remaining_1 >= GROUP_M) ? GROUP_M : remaining_1);
                int local_1 = this_bid_1 % (unsigned int)tiles_per_group_1;
                int bid_m_1 = first_m_1 + local_1 % group_size_1;
                int bid_n_1 = local_1 / group_size_1;
                int off_m_1 = bid_m_1 * BLOCK_M;
                int off_n_1 = bid_n_1 * 352;
                mbarrier_wait(mainloop_done_addr, _phase_mainloop_done);
                asm volatile("tcgen05.fence::after_thread_sync;");
                int lane_row = taddr + (unsigned int)(epi_warp * 32 << 16);
                float _tmem_load_0[64];
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(_tmem_load_0[0]), "=f"(_tmem_load_0[1]), "=f"(_tmem_load_0[2]), "=f"(_tmem_load_0[3]), "=f"(_tmem_load_0[4]), "=f"(_tmem_load_0[5]), "=f"(_tmem_load_0[6]), "=f"(_tmem_load_0[7]), "=f"(_tmem_load_0[8]), "=f"(_tmem_load_0[9]), "=f"(_tmem_load_0[10]), "=f"(_tmem_load_0[11]), "=f"(_tmem_load_0[12]), "=f"(_tmem_load_0[13]), "=f"(_tmem_load_0[14]), "=f"(_tmem_load_0[15]), "=f"(_tmem_load_0[16]), "=f"(_tmem_load_0[17]), "=f"(_tmem_load_0[18]), "=f"(_tmem_load_0[19]), "=f"(_tmem_load_0[20]), "=f"(_tmem_load_0[21]), "=f"(_tmem_load_0[22]), "=f"(_tmem_load_0[23]), "=f"(_tmem_load_0[24]), "=f"(_tmem_load_0[25]), "=f"(_tmem_load_0[26]), "=f"(_tmem_load_0[27]), "=f"(_tmem_load_0[28]), "=f"(_tmem_load_0[29]), "=f"(_tmem_load_0[30]), "=f"(_tmem_load_0[31])
                    : "r"(lane_row + col_part * 64));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(_tmem_load_0[32]), "=f"(_tmem_load_0[33]), "=f"(_tmem_load_0[34]), "=f"(_tmem_load_0[35]), "=f"(_tmem_load_0[36]), "=f"(_tmem_load_0[37]), "=f"(_tmem_load_0[38]), "=f"(_tmem_load_0[39]), "=f"(_tmem_load_0[40]), "=f"(_tmem_load_0[41]), "=f"(_tmem_load_0[42]), "=f"(_tmem_load_0[43]), "=f"(_tmem_load_0[44]), "=f"(_tmem_load_0[45]), "=f"(_tmem_load_0[46]), "=f"(_tmem_load_0[47]), "=f"(_tmem_load_0[48]), "=f"(_tmem_load_0[49]), "=f"(_tmem_load_0[50]), "=f"(_tmem_load_0[51]), "=f"(_tmem_load_0[52]), "=f"(_tmem_load_0[53]), "=f"(_tmem_load_0[54]), "=f"(_tmem_load_0[55]), "=f"(_tmem_load_0[56]), "=f"(_tmem_load_0[57]), "=f"(_tmem_load_0[58]), "=f"(_tmem_load_0[59]), "=f"(_tmem_load_0[60]), "=f"(_tmem_load_0[61]), "=f"(_tmem_load_0[62]), "=f"(_tmem_load_0[63])
                    : "r"(lane_row + col_part * 64 + 32));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                unsigned int _tmem_load_0_bf16[32];
                #pragma unroll
                for (int _lp = 0; _lp < 32; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 0], _tmem_load_0[_lp*2+1 + 0]));
                    _tmem_load_0_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                float _tmem_load_1[64];
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(_tmem_load_1[0]), "=f"(_tmem_load_1[1]), "=f"(_tmem_load_1[2]), "=f"(_tmem_load_1[3]), "=f"(_tmem_load_1[4]), "=f"(_tmem_load_1[5]), "=f"(_tmem_load_1[6]), "=f"(_tmem_load_1[7]), "=f"(_tmem_load_1[8]), "=f"(_tmem_load_1[9]), "=f"(_tmem_load_1[10]), "=f"(_tmem_load_1[11]), "=f"(_tmem_load_1[12]), "=f"(_tmem_load_1[13]), "=f"(_tmem_load_1[14]), "=f"(_tmem_load_1[15]), "=f"(_tmem_load_1[16]), "=f"(_tmem_load_1[17]), "=f"(_tmem_load_1[18]), "=f"(_tmem_load_1[19]), "=f"(_tmem_load_1[20]), "=f"(_tmem_load_1[21]), "=f"(_tmem_load_1[22]), "=f"(_tmem_load_1[23]), "=f"(_tmem_load_1[24]), "=f"(_tmem_load_1[25]), "=f"(_tmem_load_1[26]), "=f"(_tmem_load_1[27]), "=f"(_tmem_load_1[28]), "=f"(_tmem_load_1[29]), "=f"(_tmem_load_1[30]), "=f"(_tmem_load_1[31])
                    : "r"(lane_row + 128 + col_part * 64));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(_tmem_load_1[32]), "=f"(_tmem_load_1[33]), "=f"(_tmem_load_1[34]), "=f"(_tmem_load_1[35]), "=f"(_tmem_load_1[36]), "=f"(_tmem_load_1[37]), "=f"(_tmem_load_1[38]), "=f"(_tmem_load_1[39]), "=f"(_tmem_load_1[40]), "=f"(_tmem_load_1[41]), "=f"(_tmem_load_1[42]), "=f"(_tmem_load_1[43]), "=f"(_tmem_load_1[44]), "=f"(_tmem_load_1[45]), "=f"(_tmem_load_1[46]), "=f"(_tmem_load_1[47]), "=f"(_tmem_load_1[48]), "=f"(_tmem_load_1[49]), "=f"(_tmem_load_1[50]), "=f"(_tmem_load_1[51]), "=f"(_tmem_load_1[52]), "=f"(_tmem_load_1[53]), "=f"(_tmem_load_1[54]), "=f"(_tmem_load_1[55]), "=f"(_tmem_load_1[56]), "=f"(_tmem_load_1[57]), "=f"(_tmem_load_1[58]), "=f"(_tmem_load_1[59]), "=f"(_tmem_load_1[60]), "=f"(_tmem_load_1[61]), "=f"(_tmem_load_1[62]), "=f"(_tmem_load_1[63])
                    : "r"(lane_row + 128 + col_part * 64 + 32));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                unsigned int _tmem_load_1_bf16[32];
                #pragma unroll
                for (int _lp = 0; _lp < 32; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_1[_lp*2 + 0], _tmem_load_1[_lp*2+1 + 0]));
                    _tmem_load_1_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                float _tmem_load_2[64];
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(_tmem_load_2[0]), "=f"(_tmem_load_2[1]), "=f"(_tmem_load_2[2]), "=f"(_tmem_load_2[3]), "=f"(_tmem_load_2[4]), "=f"(_tmem_load_2[5]), "=f"(_tmem_load_2[6]), "=f"(_tmem_load_2[7]), "=f"(_tmem_load_2[8]), "=f"(_tmem_load_2[9]), "=f"(_tmem_load_2[10]), "=f"(_tmem_load_2[11]), "=f"(_tmem_load_2[12]), "=f"(_tmem_load_2[13]), "=f"(_tmem_load_2[14]), "=f"(_tmem_load_2[15]), "=f"(_tmem_load_2[16]), "=f"(_tmem_load_2[17]), "=f"(_tmem_load_2[18]), "=f"(_tmem_load_2[19]), "=f"(_tmem_load_2[20]), "=f"(_tmem_load_2[21]), "=f"(_tmem_load_2[22]), "=f"(_tmem_load_2[23]), "=f"(_tmem_load_2[24]), "=f"(_tmem_load_2[25]), "=f"(_tmem_load_2[26]), "=f"(_tmem_load_2[27]), "=f"(_tmem_load_2[28]), "=f"(_tmem_load_2[29]), "=f"(_tmem_load_2[30]), "=f"(_tmem_load_2[31])
                    : "r"(lane_row + 256 + col_part * 64));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                if (col_part == 0) {
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(_tmem_load_2[32]), "=f"(_tmem_load_2[33]), "=f"(_tmem_load_2[34]), "=f"(_tmem_load_2[35]), "=f"(_tmem_load_2[36]), "=f"(_tmem_load_2[37]), "=f"(_tmem_load_2[38]), "=f"(_tmem_load_2[39]), "=f"(_tmem_load_2[40]), "=f"(_tmem_load_2[41]), "=f"(_tmem_load_2[42]), "=f"(_tmem_load_2[43]), "=f"(_tmem_load_2[44]), "=f"(_tmem_load_2[45]), "=f"(_tmem_load_2[46]), "=f"(_tmem_load_2[47]), "=f"(_tmem_load_2[48]), "=f"(_tmem_load_2[49]), "=f"(_tmem_load_2[50]), "=f"(_tmem_load_2[51]), "=f"(_tmem_load_2[52]), "=f"(_tmem_load_2[53]), "=f"(_tmem_load_2[54]), "=f"(_tmem_load_2[55]), "=f"(_tmem_load_2[56]), "=f"(_tmem_load_2[57]), "=f"(_tmem_load_2[58]), "=f"(_tmem_load_2[59]), "=f"(_tmem_load_2[60]), "=f"(_tmem_load_2[61]), "=f"(_tmem_load_2[62]), "=f"(_tmem_load_2[63])
                        : "r"(lane_row + 256 + 32));
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                }
                if (elect_sync()) {
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    asm volatile(
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                        :: "r"((epilogue_done_addr) & 0xFEFFFFFF) : "memory");
                }
                _phase_mainloop_done ^= 1;
                unsigned int _tmem_load_2_bf16[32];
                #pragma unroll
                for (int _lp = 0; _lp < 16; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_2[_lp*2 + 0], _tmem_load_2[_lp*2+1 + 0]));
                    _tmem_load_2_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                if (col_part == 0) {
                    #pragma unroll
                    for (int _lp = 0; _lp < 16; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_2[_lp*2 + 32], _tmem_load_2[_lp*2+1 + 32]));
                        _tmem_load_2_bf16[_lp + 16] = *(uint32_t*)&_bf2;
                    }
                }
                unsigned int ts_base = smem_out_addr;
                unsigned int ts_row = ts_base + (unsigned int)(local_row * 64);
                int swz = local_row >> 1 & 3;
                #pragma unroll
                for (int c_0 = 0; c_0 < 2; c_0++) {
                    if (warp == 2) {
                        asm volatile("cp.async.bulk.wait_group.read 0;");
                    }
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    asm volatile("barrier.sync 1, 256;" ::: "memory");
                    if (col_part == 0) {
                        #pragma unroll
                        for (int q_0 = 0; q_0 < 4; q_0++) {
                            int s0_0 = c_0 * 16 + q_0 * 4;
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(ts_row + (unsigned int)((q_0 ^ swz) * 16)), "r"(_tmem_load_0_bf16[s0_0]), "r"(_tmem_load_0_bf16[s0_0 + 1]), "r"(_tmem_load_0_bf16[s0_0 + 2]), "r"(_tmem_load_0_bf16[s0_0 + 3]));
                        }
                    }
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    asm volatile("barrier.sync 1, 256;" ::: "memory");
                    if (warp == 2) {
                        if (elect_sync()) {
                            tma_store_2d((&OUT), off_n_1 + c_0 * 32, off_m_1, ts_base);
                            asm volatile("cp.async.bulk.commit_group;");
                        }
                    }
                    if (warp == 2) {
                        asm volatile("cp.async.bulk.wait_group.read 0;");
                    }
                    asm volatile("barrier.sync 1, 256;" ::: "memory");
                    if (col_part == 1) {
                        #pragma unroll
                        for (int q_1 = 0; q_1 < 4; q_1++) {
                            int s0_1 = c_0 * 16 + q_1 * 4;
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(ts_row + (unsigned int)((q_1 ^ swz) * 16)), "r"(_tmem_load_0_bf16[s0_1]), "r"(_tmem_load_0_bf16[s0_1 + 1]), "r"(_tmem_load_0_bf16[s0_1 + 2]), "r"(_tmem_load_0_bf16[s0_1 + 3]));
                        }
                    }
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    asm volatile("barrier.sync 1, 256;" ::: "memory");
                    if (warp == 2) {
                        if (elect_sync()) {
                            tma_store_2d((&OUT), off_n_1 + 64 + c_0 * 32, off_m_1, ts_base);
                            asm volatile("cp.async.bulk.commit_group;");
                        }
                    }
                }
                #pragma unroll
                for (int c_1 = 0; c_1 < 2; c_1++) {
                    if (warp == 2) {
                        asm volatile("cp.async.bulk.wait_group.read 0;");
                    }
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    asm volatile("barrier.sync 1, 256;" ::: "memory");
                    if (col_part == 0) {
                        #pragma unroll
                        for (int q_2 = 0; q_2 < 4; q_2++) {
                            int s0_2 = c_1 * 16 + q_2 * 4;
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(ts_row + (unsigned int)((q_2 ^ swz) * 16)), "r"(_tmem_load_1_bf16[s0_2]), "r"(_tmem_load_1_bf16[s0_2 + 1]), "r"(_tmem_load_1_bf16[s0_2 + 2]), "r"(_tmem_load_1_bf16[s0_2 + 3]));
                        }
                    }
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    asm volatile("barrier.sync 1, 256;" ::: "memory");
                    if (warp == 2) {
                        if (elect_sync()) {
                            tma_store_2d((&OUT), off_n_1 + 128 + c_1 * 32, off_m_1, ts_base);
                            asm volatile("cp.async.bulk.commit_group;");
                        }
                    }
                    if (warp == 2) {
                        asm volatile("cp.async.bulk.wait_group.read 0;");
                    }
                    asm volatile("barrier.sync 1, 256;" ::: "memory");
                    if (col_part == 1) {
                        #pragma unroll
                        for (int q_3 = 0; q_3 < 4; q_3++) {
                            int s0_3 = c_1 * 16 + q_3 * 4;
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(ts_row + (unsigned int)((q_3 ^ swz) * 16)), "r"(_tmem_load_1_bf16[s0_3]), "r"(_tmem_load_1_bf16[s0_3 + 1]), "r"(_tmem_load_1_bf16[s0_3 + 2]), "r"(_tmem_load_1_bf16[s0_3 + 3]));
                        }
                    }
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    asm volatile("barrier.sync 1, 256;" ::: "memory");
                    if (warp == 2) {
                        if (elect_sync()) {
                            tma_store_2d((&OUT), off_n_1 + 192 + c_1 * 32, off_m_1, ts_base);
                            asm volatile("cp.async.bulk.commit_group;");
                        }
                    }
                }
                #pragma unroll
                for (int c_2 = 0; c_2 < 2; c_2++) {
                    if (warp == 2) {
                        asm volatile("cp.async.bulk.wait_group.read 0;");
                    }
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    asm volatile("barrier.sync 1, 256;" ::: "memory");
                    if (col_part == 0) {
                        #pragma unroll
                        for (int q_4 = 0; q_4 < 4; q_4++) {
                            int s0_4 = c_2 * 16 + q_4 * 4;
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(ts_row + (unsigned int)((q_4 ^ swz) * 16)), "r"(_tmem_load_2_bf16[s0_4]), "r"(_tmem_load_2_bf16[s0_4 + 1]), "r"(_tmem_load_2_bf16[s0_4 + 2]), "r"(_tmem_load_2_bf16[s0_4 + 3]));
                        }
                    }
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    asm volatile("barrier.sync 1, 256;" ::: "memory");
                    if (warp == 2) {
                        if (elect_sync()) {
                            tma_store_2d((&OUT), off_n_1 + 256 + c_2 * 32, off_m_1, ts_base);
                            asm volatile("cp.async.bulk.commit_group;");
                        }
                    }
                    if (c_2 == 0) {
                        if (warp == 2) {
                            asm volatile("cp.async.bulk.wait_group.read 0;");
                        }
                        asm volatile("barrier.sync 1, 256;" ::: "memory");
                        if (col_part == 1) {
                            #pragma unroll
                            for (int q_5 = 0; q_5 < 4; q_5++) {
                                int s0_5 = q_5 * 4;
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(ts_row + (unsigned int)((q_5 ^ swz) * 16)), "r"(_tmem_load_2_bf16[s0_5]), "r"(_tmem_load_2_bf16[s0_5 + 1]), "r"(_tmem_load_2_bf16[s0_5 + 2]), "r"(_tmem_load_2_bf16[s0_5 + 3]));
                            }
                        }
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        asm volatile("barrier.sync 1, 256;" ::: "memory");
                        if (warp == 2) {
                            if (elect_sync()) {
                                tma_store_2d((&OUT), off_n_1 + 320, off_m_1, ts_base);
                                asm volatile("cp.async.bulk.commit_group;");
                            }
                        }
                    }
                }
                mbarrier_wait(work_full_addr + (work_stage_2) * 8, _phase_work_full_2);
                uint32_t clc_valid_2 = 0;
                uint32_t clc_ctaid_2;
                uint32_t clc_ctaid_y_2;
                uint32_t clc_ctaid_z_2;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%4];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "selp.u32 %3, 1, 0, p1;\n\t"
                    "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid.v4.b32.b128 {%0, %1, %2, _}, clc_r;\n\t"
                    "}\n"
                    : "=r"(clc_ctaid_2), "=r"(clc_ctaid_y_2), "=r"(clc_ctaid_z_2), "=r"(clc_valid_2)
                    : "r"(work_response_addr + work_stage_2 * 16 + 0 * 16)
                    : "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("tcgen05.fence::before_thread_sync;");
                asm volatile(
                    "{\n\t"
                    ".reg .b32 remAddr32;\n\t"
                    "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                    "}"
                    :: "r"(work_empty_addr + (work_stage_2) * 8), "r"(0) : "memory");
                work_stage_2 += 1;
                if (work_stage_2 == 4) { work_stage_2 = 0; _phase_work_full_2 ^= 1; }
                if (clc_valid_2 == 0) {
                    break;
                }
                this_bid_1 = clc_ctaid_2 + (unsigned int)cta_rank;
            }
            if ((warp - 2) % 4 == 0) {
                asm volatile("cp.async.bulk.wait_group 0;");
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
