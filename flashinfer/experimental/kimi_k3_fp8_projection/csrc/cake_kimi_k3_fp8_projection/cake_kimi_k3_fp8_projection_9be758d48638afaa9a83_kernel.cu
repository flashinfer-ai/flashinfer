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
#define TMEM_NCOLS 216
#define TMEM_ACCUM_OFFSET 0
#define TMEM_TMEM_SFA_OFFSET 192
#define TMEM_TMEM_SFB_OFFSET 200
#define NUM_TMA_PIPE_STAGES 3
#define NUM_MAINLOOP_PIPE_STAGES 1
#define NUM_WORK_PIPE_STAGES 4
#define SMEM_SMEM_OUT_OFF 183296
#define SMEM_SMEM_OUT_STAGE_BYTES 8192
#define SMEM_SMEM_OUT_STRIDE 8192
#define SMEM_SMEM_A_OFF 1024
#define SMEM_SMEM_A_STAGE_BYTES 32768
#define SMEM_SMEM_A_STRIDE 60416
#define SMEM_SMEM_B_OFF 33792
#define SMEM_SMEM_B_STAGE_BYTES 24576
#define SMEM_SMEM_B_STRIDE 60416
#define SMEM_SMEM_SFA_ALL_OFF 58368
#define SMEM_SMEM_SFA_ALL_STAGE_BYTES 1024
#define SMEM_SMEM_SFA_ALL_STRIDE 60416
#define SMEM_SMEM_SFB_ALL_OFF 59392
#define SMEM_SMEM_SFB_ALL_STAGE_BYTES 2048
#define SMEM_SMEM_SFB_ALL_STRIDE 60416
#define SMEM_SMEM_SFA0_OFF 58368
#define SMEM_SMEM_SFA0_STAGE_BYTES 512
#define SMEM_SMEM_SFA0_STRIDE 60416
#define SMEM_SMEM_SFA1_OFF 58880
#define SMEM_SMEM_SFA1_STAGE_BYTES 512
#define SMEM_SMEM_SFA1_STRIDE 60416
#define SMEM_SMEM_SFB0_OFF 59392
#define SMEM_SMEM_SFB0_STAGE_BYTES 1024
#define SMEM_SMEM_SFB0_STRIDE 60416
#define SMEM_SMEM_SFB1_OFF 60416
#define SMEM_SMEM_SFB1_STAGE_BYTES 1024
#define SMEM_SMEM_SFB1_STRIDE 60416
#define SMEM_WORK_RESPONSE_OFF 182272
#define SMEM_WORK_RESPONSE_STAGE_BYTES 16
#define SMEM_WORK_RESPONSE_STRIDE 16
#define SMEM_SMEM_XCH_OFF 1024
#define SMEM_SMEM_XCH_STAGE_BYTES 102400
#define SMEM_SMEM_XCH_STRIDE 102400
#define SMEM_TOTAL 199680
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


__device__ __forceinline__ void mbarrier_expect_tx(int mbar_addr, uint32_t bytes) {
    asm volatile(
        "mbarrier.expect_tx.relaxed.cta.shared::cta.b64 [%0], %1;"
        :: "r"(mbar_addr), "r"(bytes) : "memory");
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


__device__ __forceinline__ void mbarrier_arrive(int mbar_addr) {
    asm volatile(
        "mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];"
        :: "r"(mbar_addr) : "memory");
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

__global__ __launch_bounds__(320) __cluster_dims__(4,1,1) void
kernel_cake_kimi_k3_fp8_projection_9be758d48638afaa9a83(const __grid_constant__ CUtensorMap A, const __grid_constant__ CUtensorMap B, const __grid_constant__ CUtensorMap SFA, const __grid_constant__ CUtensorMap SFB, __nv_bfloat16* __restrict__ out, const __grid_constant__ CUtensorMap OUT, int M, int m_tiles, int n_tiles, int n_valid, int ldo, int store_vec, int num_k_iters, int sf_k_tiles, __nv_bfloat16* __restrict__ x, int K, float* __restrict__ sk_partials, unsigned int* __restrict__ sk_flags, int sk_pairs, int sk_rem, int sk_ksplit, int sk_dp)
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
    #define tma_full_h0_addr (mbar_base + 0)
    #define tma_full_h1_addr (mbar_base + 24)
    #define mma_done_h0_addr (mbar_base + 48)
    #define mma_done_h1_addr (mbar_base + 72)
    #define mainloop_done_addr (mbar_base + 96)
    #define epilogue_done_addr (mbar_base + 104)
    #define red_full_addr (mbar_base + 112)
    #define ring_dead_addr (mbar_base + 120)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    const unsigned int clusters_x = gridDim.x / 4;
    const unsigned int cluster_id = ((blockIdx.z * gridDim.y + blockIdx.y) * clusters_x) + blockIdx.x / 4;
    const unsigned int num_clusters = clusters_x * gridDim.y * gridDim.z;

    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));

    // Kernel setup ops
    __nv_bfloat16* smem_out = reinterpret_cast<__nv_bfloat16*>(smem_raw + 183296);
    const int smem_out_addr = smem + 183296;
    uint8_t* smem_a = reinterpret_cast<uint8_t*>(smem_raw + 1024);
    const int smem_a_addr = smem + 1024;
    uint8_t* smem_b = reinterpret_cast<uint8_t*>(smem_raw + 33792);
    const int smem_b_addr = smem + 33792;
    uint8_t* smem_sfa_all = reinterpret_cast<uint8_t*>(smem_raw + 58368);
    const int smem_sfa_all_addr = smem + 58368;
    uint8_t* smem_sfb_all = reinterpret_cast<uint8_t*>(smem_raw + 59392);
    const int smem_sfb_all_addr = smem + 59392;
    uint8_t* smem_sfa0 = reinterpret_cast<uint8_t*>(smem_raw + 58368);
    const int smem_sfa0_addr = smem + 58368;
    uint8_t* smem_sfa1 = reinterpret_cast<uint8_t*>(smem_raw + 58880);
    const int smem_sfa1_addr = smem + 58880;
    uint8_t* smem_sfb0 = reinterpret_cast<uint8_t*>(smem_raw + 59392);
    const int smem_sfb0_addr = smem + 59392;
    uint8_t* smem_sfb1 = reinterpret_cast<uint8_t*>(smem_raw + 60416);
    const int smem_sfb1_addr = smem + 60416;
    unsigned int* work_response = reinterpret_cast<unsigned int*>(smem_raw + 182272);
    const int work_response_addr = smem + 182272;
    float* smem_xch = reinterpret_cast<float*>(smem_raw + 1024);
    const int smem_xch_addr = smem + 1024;

    // Mbarrier init (8 pipeline groups, 0 ordered-sequence groups, 16 barriers)
    // Mbarriers at smem_raw[0..128)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'tma_pipe' ---
            // tma_full_h0: 3 barriers, init_count=2
            mbarrier_init(smem + 0, 2);
            mbarrier_init(smem + 8, 2);
            mbarrier_init(smem + 16, 2);
            // tma_full_h1: 3 barriers, init_count=2
            mbarrier_init(smem + 24, 2);
            mbarrier_init(smem + 32, 2);
            mbarrier_init(smem + 40, 2);
            // mma_done_h0: 3 barriers, init_count=1
            mbarrier_init(smem + 48, 1);
            mbarrier_init(smem + 56, 1);
            mbarrier_init(smem + 64, 1);
            // mma_done_h1: 3 barriers, init_count=1
            mbarrier_init(smem + 72, 1);
            mbarrier_init(smem + 80, 1);
            mbarrier_init(smem + 88, 1);
            // --- pipeline 'mainloop_pipe' ---
            // mainloop_done: 1 barriers, init_count=1
            mbarrier_init(smem + 96, 1);
            // epilogue_done: 1 barriers, init_count=16
            mbarrier_init(smem + 104, 16);
            // red_full: 1 barriers, init_count=1
            mbarrier_init(smem + 112, 1);
            // ring_dead: 1 barriers, init_count=1
            mbarrier_init(smem + 120, 1);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
            mbarrier_expect_tx(smem + 112, 51200);
        }
    }

    __syncwarp();
    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");

    // TMEM alloc (256 columns, 216 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 128);
    if (warp == 0) {
        int _tmem_hold = smem + 128;
        asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(256) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_accum = taddr;
    const int tmem_tmem_sfa = taddr + 192;
    const int tmem_tmem_sfb = taddr + 200;

    // ---- Role: load ----
    if (warp == 0) {
        { // load_main
            int num_cluster_tiles = m_tiles / CTA_GROUP * n_tiles;
            unsigned int load_stage = 0;
            unsigned int work_stage = 0;
            asm volatile("griddepcontrol.wait;" ::: "memory");
            unsigned int _phase_mma_done_h0 = 1;
            if (elect_sync()) {
                unsigned int this_bid = bid;
                unsigned int cs_tile_bid = (unsigned int)(bid / 4 * 2) + (unsigned int)(cta_rank & 1);
                int cs_p = cta_rank >> 1;
                int cs_k0 = num_k_iters * cs_p / 2;
                int cs_k1 = num_k_iters * (cs_p + 1) / 2;
                int tiles_per_group = GROUP_M * n_tiles;
                int group = cs_tile_bid / (unsigned int)tiles_per_group;
                int first_m = group * GROUP_M;
                int remaining = m_tiles - first_m;
                int group_size = ((remaining >= GROUP_M) ? GROUP_M : remaining);
                int local = cs_tile_bid % (unsigned int)tiles_per_group;
                int bid_m = first_m + local % group_size;
                int bid_n = local / group_size;
                int off_m = bid_m * BLOCK_M;
                int off_n = bid_n * 192;
                int weight_tile0 = (off_n + (cta_rank & 1) * 96) / 128 * (2 * num_k_iters);
                int b_row0 = off_n + (cta_rank & 1) * 96;
                int sfa_tile_row = bid_m * sf_k_tiles;
                int sfb_tile_row = bid_n * sf_k_tiles;
                #pragma unroll 1
                for (int iter_k = cs_k0; iter_k < cs_k1; iter_k++) {
                    int k_group = iter_k * 2;
                    mbarrier_wait(mma_done_h0_addr + (load_stage) * 8, _phase_mma_done_h0);
                    tma_3d_gmem2smem_cta2(smem_a_addr + load_stage * 60416, (&A), 0, off_m, k_group, ((tma_full_h0_addr + (load_stage) * 8) & 0xFEFFFFFF));
                    tma_3d_gmem2smem_cta2(smem_sfa_all_addr + load_stage * 60416, (&SFA), 0, 0, sfa_tile_row + k_group, ((tma_full_h0_addr + (load_stage) * 8) & 0xFEFFFFFF));
                    tma_3d_gmem2smem_cta2(smem_b_addr + load_stage * 60416, (&B), 0, b_row0 % 128, b_row0 / 128 * (2 * num_k_iters) + k_group, ((tma_full_h0_addr + (load_stage) * 8) & 0xFEFFFFFF));
                    tma_3d_gmem2smem_cta2(smem_b_addr + load_stage * 60416 + 4096, (&B), 0, (b_row0 + 32) % 128, (b_row0 + 32) / 128 * (2 * num_k_iters) + k_group, ((tma_full_h0_addr + (load_stage) * 8) & 0xFEFFFFFF));
                    tma_3d_gmem2smem_cta2(smem_b_addr + load_stage * 60416 + 8192, (&B), 0, (b_row0 + 64) % 128, (b_row0 + 64) / 128 * (2 * num_k_iters) + k_group, ((tma_full_h0_addr + (load_stage) * 8) & 0xFEFFFFFF));
                    tma_3d_gmem2smem_cta2(smem_sfb_all_addr + load_stage * 60416, (&SFB), 0, 0, sfb_tile_row + k_group, ((tma_full_h0_addr + (load_stage) * 8) & 0xFEFFFFFF));
                    asm volatile(
                        "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                        :: "r"((tma_full_h0_addr + (load_stage) * 8) & 0xFEFFFFFF), "r"((uint32_t)(30208)) : "memory");
                    mbarrier_wait(mma_done_h1_addr + (load_stage) * 8, _phase_mma_done_h0);
                    tma_3d_gmem2smem_cta2(smem_a_addr + load_stage * 60416 + 16384, (&A), 0, off_m, k_group + 1, ((tma_full_h1_addr + (load_stage) * 8) & 0xFEFFFFFF));
                    tma_3d_gmem2smem_cta2(smem_sfa_all_addr + load_stage * 60416 + 512, (&SFA), 0, 0, sfa_tile_row + k_group + 1, ((tma_full_h1_addr + (load_stage) * 8) & 0xFEFFFFFF));
                    tma_3d_gmem2smem_cta2(smem_b_addr + load_stage * 60416 + 12288, (&B), 0, b_row0 % 128, b_row0 / 128 * (2 * num_k_iters) + k_group + 1, ((tma_full_h1_addr + (load_stage) * 8) & 0xFEFFFFFF));
                    tma_3d_gmem2smem_cta2(smem_b_addr + load_stage * 60416 + 16384, (&B), 0, (b_row0 + 32) % 128, (b_row0 + 32) / 128 * (2 * num_k_iters) + k_group + 1, ((tma_full_h1_addr + (load_stage) * 8) & 0xFEFFFFFF));
                    tma_3d_gmem2smem_cta2(smem_b_addr + load_stage * 60416 + 20480, (&B), 0, (b_row0 + 64) % 128, (b_row0 + 64) / 128 * (2 * num_k_iters) + k_group + 1, ((tma_full_h1_addr + (load_stage) * 8) & 0xFEFFFFFF));
                    tma_3d_gmem2smem_cta2(smem_sfb_all_addr + load_stage * 60416 + 1024, (&SFB), 0, 0, sfb_tile_row + k_group + 1, ((tma_full_h1_addr + (load_stage) * 8) & 0xFEFFFFFF));
                    asm volatile(
                        "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                        :: "r"((tma_full_h1_addr + (load_stage) * 8) & 0xFEFFFFFF), "r"((uint32_t)(30208)) : "memory");
                    load_stage += 1;
                    if (load_stage == 3) { load_stage = 0; _phase_mma_done_h0 ^= 1; }
                }
                asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
                #pragma unroll 1
                for (int _ = 0; _ < 3; _++) {
                    mbarrier_wait(mma_done_h0_addr + (load_stage) * 8, _phase_mma_done_h0);
                    mbarrier_wait(mma_done_h1_addr + (load_stage) * 8, _phase_mma_done_h0);
                    load_stage += 1;
                    if (load_stage == 3) { load_stage = 0; _phase_mma_done_h0 ^= 1; }
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
            unsigned int _phase_tma_full_h0 = 0;
            if ((cta_rank & 1) == 0) {
                int cs_p_1 = cta_rank >> 1;
                int cs_k0_1 = num_k_iters * cs_p_1 / 2;
                int cs_k1_1 = num_k_iters * (cs_p_1 + 1) / 2;
                mbarrier_wait(epilogue_done_addr, _phase_epilogue_done);
                mbarrier_wait(tma_full_h0_addr + (mma_tma_stage) * 8, _phase_tma_full_h0);
                asm volatile("tcgen05.fence::after_thread_sync;");
                uint32_t _elected_0 = elect_sync();
                tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfa, make_sf_cp_desc_lo_sbo128((((smem_sfa0_addr) >> 4) + (mma_tma_stage) * 3776)), _elected_0);
                tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfb, make_sf_cp_desc_lo_sbo128((((smem_sfb0_addr) >> 4) + (mma_tma_stage) * 3776)), _elected_0);
                tcgen05_cp_32x128b_warpx4_cta2_leader((tmem_tmem_sfb + 4), make_sf_cp_desc_lo_sbo128((((smem_sfb0_addr) >> 4) + (mma_tma_stage) * 3776 + 32)), _elected_0);
                int _smem_desc_lo_1 = (((smem_a_addr) >> 4) & 0x3FFF) + (mma_tma_stage) * 3776;
                int _smem_desc_lo_2 = (((smem_b_addr) >> 4) & 0x3FFF) + (mma_tma_stage) * 3776;
                {
                    uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_1) | ((uint64_t)0x40004040 << 32);
                    uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_2) | ((uint64_t)0x40004040 << 32);

                    tcgen05_mma_mxf8_bs_cta2_leader(tmem_accum, a_desc + 0, b_desc + 0,
                        0x10b00000U, tmem_tmem_sfa, tmem_tmem_sfb, 0, _elected_0);
                    tcgen05_mma_mxf8_bs_cta2_leader(tmem_accum, a_desc + 2, b_desc + 2,
                        0x30b00010U, tmem_tmem_sfa, tmem_tmem_sfb, 1, _elected_0);
                    tcgen05_mma_mxf8_bs_cta2_leader(tmem_accum, a_desc + 4, b_desc + 4,
                        0x50b00020U, tmem_tmem_sfa, tmem_tmem_sfb, 1, _elected_0);
                    tcgen05_mma_mxf8_bs_cta2_leader(tmem_accum, a_desc + 6, b_desc + 6,
                        0x70b00030U, tmem_tmem_sfa, tmem_tmem_sfb, 1, _elected_0);
                }
                elect_commit_cg2_multicast(mma_done_h0_addr + (mma_tma_stage) * 8, (uint16_t)(3 << (cta_rank >> 1 << 1)));
                mbarrier_wait(tma_full_h1_addr + (mma_tma_stage) * 8, _phase_tma_full_h0);
                asm volatile("tcgen05.fence::after_thread_sync;");
                uint32_t _elected_3 = elect_sync();
                tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfa + 4, make_sf_cp_desc_lo_sbo128((((smem_sfa1_addr) >> 4) + (mma_tma_stage) * 3776)), _elected_3);
                tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfb + 8, make_sf_cp_desc_lo_sbo128((((smem_sfb1_addr) >> 4) + (mma_tma_stage) * 3776)), _elected_3);
                tcgen05_cp_32x128b_warpx4_cta2_leader((tmem_tmem_sfb + 8 + 4), make_sf_cp_desc_lo_sbo128((((smem_sfb1_addr) >> 4) + (mma_tma_stage) * 3776 + 32)), _elected_3);
                int _smem_desc_lo_4 = (((smem_a_addr + 16384) >> 4) & 0x3FFF) + (mma_tma_stage) * 3776;
                int _smem_desc_lo_5 = (((smem_b_addr + 12288) >> 4) & 0x3FFF) + (mma_tma_stage) * 3776;
                {
                    uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_4) | ((uint64_t)0x40004040 << 32);
                    uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_5) | ((uint64_t)0x40004040 << 32);

                    tcgen05_mma_mxf8_bs_cta2_leader(tmem_accum, a_desc + 0, b_desc + 0,
                        0x10b00000U, tmem_tmem_sfa + 4, tmem_tmem_sfb + 8, 1, _elected_3);
                    tcgen05_mma_mxf8_bs_cta2_leader(tmem_accum, a_desc + 2, b_desc + 2,
                        0x30b00010U, tmem_tmem_sfa + 4, tmem_tmem_sfb + 8, 1, _elected_3);
                    tcgen05_mma_mxf8_bs_cta2_leader(tmem_accum, a_desc + 4, b_desc + 4,
                        0x50b00020U, tmem_tmem_sfa + 4, tmem_tmem_sfb + 8, 1, _elected_3);
                    tcgen05_mma_mxf8_bs_cta2_leader(tmem_accum, a_desc + 6, b_desc + 6,
                        0x70b00030U, tmem_tmem_sfa + 4, tmem_tmem_sfb + 8, 1, _elected_3);
                }
                elect_commit_cg2_multicast(mma_done_h1_addr + (mma_tma_stage) * 8, (uint16_t)(3 << (cta_rank >> 1 << 1)));
                mma_tma_stage += 1;
                if (mma_tma_stage == 3) { mma_tma_stage = 0; _phase_tma_full_h0 ^= 1; }
                #pragma unroll 1
                for (int _iter_k = cs_k0_1 + 1; _iter_k < cs_k1_1; _iter_k++) {
                    mbarrier_wait(tma_full_h0_addr + (mma_tma_stage) * 8, _phase_tma_full_h0);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    uint32_t _elected_6 = elect_sync();
                    tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfa, make_sf_cp_desc_lo_sbo128((((smem_sfa0_addr) >> 4) + (mma_tma_stage) * 3776)), _elected_6);
                    tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfb, make_sf_cp_desc_lo_sbo128((((smem_sfb0_addr) >> 4) + (mma_tma_stage) * 3776)), _elected_6);
                    tcgen05_cp_32x128b_warpx4_cta2_leader((tmem_tmem_sfb + 4), make_sf_cp_desc_lo_sbo128((((smem_sfb0_addr) >> 4) + (mma_tma_stage) * 3776 + 32)), _elected_6);
                    int _smem_desc_lo_7 = (((smem_a_addr) >> 4) & 0x3FFF) + (mma_tma_stage) * 3776;
                    int _smem_desc_lo_8 = (((smem_b_addr) >> 4) & 0x3FFF) + (mma_tma_stage) * 3776;
                    {
                        uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_7) | ((uint64_t)0x40004040 << 32);
                        uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_8) | ((uint64_t)0x40004040 << 32);

                        tcgen05_mma_mxf8_bs_cta2_leader(tmem_accum, a_desc + 0, b_desc + 0,
                            0x10b00000U, tmem_tmem_sfa, tmem_tmem_sfb, 1, _elected_6);
                        tcgen05_mma_mxf8_bs_cta2_leader(tmem_accum, a_desc + 2, b_desc + 2,
                            0x30b00010U, tmem_tmem_sfa, tmem_tmem_sfb, 1, _elected_6);
                        tcgen05_mma_mxf8_bs_cta2_leader(tmem_accum, a_desc + 4, b_desc + 4,
                            0x50b00020U, tmem_tmem_sfa, tmem_tmem_sfb, 1, _elected_6);
                        tcgen05_mma_mxf8_bs_cta2_leader(tmem_accum, a_desc + 6, b_desc + 6,
                            0x70b00030U, tmem_tmem_sfa, tmem_tmem_sfb, 1, _elected_6);
                    }
                    elect_commit_cg2_multicast(mma_done_h0_addr + (mma_tma_stage) * 8, (uint16_t)(3 << (cta_rank >> 1 << 1)));
                    mbarrier_wait(tma_full_h1_addr + (mma_tma_stage) * 8, _phase_tma_full_h0);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    uint32_t _elected_9 = elect_sync();
                    tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfa + 4, make_sf_cp_desc_lo_sbo128((((smem_sfa1_addr) >> 4) + (mma_tma_stage) * 3776)), _elected_9);
                    tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfb + 8, make_sf_cp_desc_lo_sbo128((((smem_sfb1_addr) >> 4) + (mma_tma_stage) * 3776)), _elected_9);
                    tcgen05_cp_32x128b_warpx4_cta2_leader((tmem_tmem_sfb + 8 + 4), make_sf_cp_desc_lo_sbo128((((smem_sfb1_addr) >> 4) + (mma_tma_stage) * 3776 + 32)), _elected_9);
                    int _smem_desc_lo_10 = (((smem_a_addr + 16384) >> 4) & 0x3FFF) + (mma_tma_stage) * 3776;
                    int _smem_desc_lo_11 = (((smem_b_addr + 12288) >> 4) & 0x3FFF) + (mma_tma_stage) * 3776;
                    {
                        uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_10) | ((uint64_t)0x40004040 << 32);
                        uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_11) | ((uint64_t)0x40004040 << 32);

                        tcgen05_mma_mxf8_bs_cta2_leader(tmem_accum, a_desc + 0, b_desc + 0,
                            0x10b00000U, tmem_tmem_sfa + 4, tmem_tmem_sfb + 8, 1, _elected_9);
                        tcgen05_mma_mxf8_bs_cta2_leader(tmem_accum, a_desc + 2, b_desc + 2,
                            0x30b00010U, tmem_tmem_sfa + 4, tmem_tmem_sfb + 8, 1, _elected_9);
                        tcgen05_mma_mxf8_bs_cta2_leader(tmem_accum, a_desc + 4, b_desc + 4,
                            0x50b00020U, tmem_tmem_sfa + 4, tmem_tmem_sfb + 8, 1, _elected_9);
                        tcgen05_mma_mxf8_bs_cta2_leader(tmem_accum, a_desc + 6, b_desc + 6,
                            0x70b00030U, tmem_tmem_sfa + 4, tmem_tmem_sfb + 8, 1, _elected_9);
                    }
                    elect_commit_cg2_multicast(mma_done_h1_addr + (mma_tma_stage) * 8, (uint16_t)(3 << (cta_rank >> 1 << 1)));
                    mma_tma_stage += 1;
                    if (mma_tma_stage == 3) { mma_tma_stage = 0; _phase_tma_full_h0 ^= 1; }
                }
                elect_commit_cg2_multicast(mainloop_done_addr, (uint16_t)(3 << (cta_rank >> 1 << 1)));
                _phase_epilogue_done ^= 1;
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
            int sk_lane_addr = taddr + (unsigned int)(epi_warp * 32 << 16) + (unsigned int)(col_part * 96);
            unsigned int cs_tile_bid_1 = (unsigned int)(bid / 4 * 2) + (unsigned int)(cta_rank & 1);
            int tiles_per_group_1 = GROUP_M * n_tiles;
            int group_1 = cs_tile_bid_1 / (unsigned int)tiles_per_group_1;
            int first_m_1 = group_1 * GROUP_M;
            int remaining_1 = m_tiles - first_m_1;
            int group_size_1 = ((remaining_1 >= GROUP_M) ? GROUP_M : remaining_1);
            int local_1 = cs_tile_bid_1 % (unsigned int)tiles_per_group_1;
            int bid_m_1 = first_m_1 + local_1 % group_size_1;
            int bid_n_1 = local_1 / group_size_1;
            int off_m_1 = bid_m_1 * BLOCK_M;
            int off_n_1 = bid_n_1 * 192;
            int global_row = off_m_1 + local_row;
            int col0 = col_part * 96;
            int lane_addr = taddr + (unsigned int)(epi_warp * 32 << 16) + (unsigned int)col0;
            int col_base = off_n_1 + col0;
            unsigned long long row_base = (unsigned long long)global_row * (unsigned long long)ldo + (unsigned long long)col_base;
            unsigned int _phase_mainloop_done = 0;
            mbarrier_wait(mainloop_done_addr, _phase_mainloop_done);
            asm volatile("tcgen05.fence::after_thread_sync;");
            float _tmem_load_0[96];
            asm volatile(
                "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                : "=f"(_tmem_load_0[0]), "=f"(_tmem_load_0[1]), "=f"(_tmem_load_0[2]), "=f"(_tmem_load_0[3]), "=f"(_tmem_load_0[4]), "=f"(_tmem_load_0[5]), "=f"(_tmem_load_0[6]), "=f"(_tmem_load_0[7]), "=f"(_tmem_load_0[8]), "=f"(_tmem_load_0[9]), "=f"(_tmem_load_0[10]), "=f"(_tmem_load_0[11]), "=f"(_tmem_load_0[12]), "=f"(_tmem_load_0[13]), "=f"(_tmem_load_0[14]), "=f"(_tmem_load_0[15]), "=f"(_tmem_load_0[16]), "=f"(_tmem_load_0[17]), "=f"(_tmem_load_0[18]), "=f"(_tmem_load_0[19]), "=f"(_tmem_load_0[20]), "=f"(_tmem_load_0[21]), "=f"(_tmem_load_0[22]), "=f"(_tmem_load_0[23]), "=f"(_tmem_load_0[24]), "=f"(_tmem_load_0[25]), "=f"(_tmem_load_0[26]), "=f"(_tmem_load_0[27]), "=f"(_tmem_load_0[28]), "=f"(_tmem_load_0[29]), "=f"(_tmem_load_0[30]), "=f"(_tmem_load_0[31])
                : "r"(lane_addr));
            asm volatile("tcgen05.wait::ld.sync.aligned;");
            asm volatile(
                "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                : "=f"(_tmem_load_0[32]), "=f"(_tmem_load_0[33]), "=f"(_tmem_load_0[34]), "=f"(_tmem_load_0[35]), "=f"(_tmem_load_0[36]), "=f"(_tmem_load_0[37]), "=f"(_tmem_load_0[38]), "=f"(_tmem_load_0[39]), "=f"(_tmem_load_0[40]), "=f"(_tmem_load_0[41]), "=f"(_tmem_load_0[42]), "=f"(_tmem_load_0[43]), "=f"(_tmem_load_0[44]), "=f"(_tmem_load_0[45]), "=f"(_tmem_load_0[46]), "=f"(_tmem_load_0[47]), "=f"(_tmem_load_0[48]), "=f"(_tmem_load_0[49]), "=f"(_tmem_load_0[50]), "=f"(_tmem_load_0[51]), "=f"(_tmem_load_0[52]), "=f"(_tmem_load_0[53]), "=f"(_tmem_load_0[54]), "=f"(_tmem_load_0[55]), "=f"(_tmem_load_0[56]), "=f"(_tmem_load_0[57]), "=f"(_tmem_load_0[58]), "=f"(_tmem_load_0[59]), "=f"(_tmem_load_0[60]), "=f"(_tmem_load_0[61]), "=f"(_tmem_load_0[62]), "=f"(_tmem_load_0[63])
                : "r"(lane_addr + 32));
            asm volatile("tcgen05.wait::ld.sync.aligned;");
            asm volatile(
                "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                : "=f"(_tmem_load_0[64]), "=f"(_tmem_load_0[65]), "=f"(_tmem_load_0[66]), "=f"(_tmem_load_0[67]), "=f"(_tmem_load_0[68]), "=f"(_tmem_load_0[69]), "=f"(_tmem_load_0[70]), "=f"(_tmem_load_0[71]), "=f"(_tmem_load_0[72]), "=f"(_tmem_load_0[73]), "=f"(_tmem_load_0[74]), "=f"(_tmem_load_0[75]), "=f"(_tmem_load_0[76]), "=f"(_tmem_load_0[77]), "=f"(_tmem_load_0[78]), "=f"(_tmem_load_0[79]), "=f"(_tmem_load_0[80]), "=f"(_tmem_load_0[81]), "=f"(_tmem_load_0[82]), "=f"(_tmem_load_0[83]), "=f"(_tmem_load_0[84]), "=f"(_tmem_load_0[85]), "=f"(_tmem_load_0[86]), "=f"(_tmem_load_0[87]), "=f"(_tmem_load_0[88]), "=f"(_tmem_load_0[89]), "=f"(_tmem_load_0[90]), "=f"(_tmem_load_0[91]), "=f"(_tmem_load_0[92]), "=f"(_tmem_load_0[93]), "=f"(_tmem_load_0[94]), "=f"(_tmem_load_0[95])
                : "r"(lane_addr + 64));
            asm volatile("tcgen05.wait::ld.sync.aligned;");
            if (elect_sync()) {
                asm volatile("tcgen05.fence::before_thread_sync;");
                asm volatile(
                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                    :: "r"((epilogue_done_addr) & 0xFEFFFFFF) : "memory");
            }
            _phase_mainloop_done ^= 1;
            int cs_p_2 = cta_rank >> 1;
            int cs_r = cta_rank & 1;
            unsigned int xch_row = (unsigned int)(local_row * 400);
            if (tid == 64) {
                if (cs_r == 0) {
                    if (cs_p_2 == 0) {
                        asm volatile("tcgen05.fence::before_thread_sync;");
                        asm volatile(
                            "{\n\t"
                            ".reg .b32 remAddr32;\n\t"
                            "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                            "mbarrier.arrive.release.cluster.shared::cluster.b64 _, [remAddr32];\n\t"
                            "}"
                            :: "r"(ring_dead_addr), "r"(2) : "memory");
                    }
                    if (cs_p_2 == 1) {
                        asm volatile("tcgen05.fence::before_thread_sync;");
                        asm volatile(
                            "{\n\t"
                            ".reg .b32 remAddr32;\n\t"
                            "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                            "mbarrier.arrive.release.cluster.shared::cluster.b64 _, [remAddr32];\n\t"
                            "}"
                            :: "r"(ring_dead_addr), "r"(0) : "memory");
                    }
                }
                if (cs_r == 1) {
                    if (cs_p_2 == 0) {
                        asm volatile("tcgen05.fence::before_thread_sync;");
                        asm volatile(
                            "{\n\t"
                            ".reg .b32 remAddr32;\n\t"
                            "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                            "mbarrier.arrive.release.cluster.shared::cluster.b64 _, [remAddr32];\n\t"
                            "}"
                            :: "r"(ring_dead_addr), "r"(3) : "memory");
                    }
                    if (cs_p_2 == 1) {
                        asm volatile("tcgen05.fence::before_thread_sync;");
                        asm volatile(
                            "{\n\t"
                            ".reg .b32 remAddr32;\n\t"
                            "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                            "mbarrier.arrive.release.cluster.shared::cluster.b64 _, [remAddr32];\n\t"
                            "}"
                            :: "r"(ring_dead_addr), "r"(1) : "memory");
                    }
                }
            }
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            if (cs_p_2 == 0) {
                if (col_part == 1) {
                    unsigned int stg_b_1_0 = smem_xch_addr + 51200 + xch_row;
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(stg_b_1_0), "r"(__float_as_uint(_tmem_load_0[0])), "r"(__float_as_uint(_tmem_load_0[1])), "r"(__float_as_uint(_tmem_load_0[2])), "r"(__float_as_uint(_tmem_load_0[3])));
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(stg_b_1_0 + 16), "r"(__float_as_uint(_tmem_load_0[4])), "r"(__float_as_uint(_tmem_load_0[5])), "r"(__float_as_uint(_tmem_load_0[6])), "r"(__float_as_uint(_tmem_load_0[7])));
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(stg_b_1_0 + 32), "r"(__float_as_uint(_tmem_load_0[8])), "r"(__float_as_uint(_tmem_load_0[9])), "r"(__float_as_uint(_tmem_load_0[10])), "r"(__float_as_uint(_tmem_load_0[11])));
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(stg_b_1_0 + 48), "r"(__float_as_uint(_tmem_load_0[12])), "r"(__float_as_uint(_tmem_load_0[13])), "r"(__float_as_uint(_tmem_load_0[14])), "r"(__float_as_uint(_tmem_load_0[15])));
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(stg_b_1_0 + 64), "r"(__float_as_uint(_tmem_load_0[16])), "r"(__float_as_uint(_tmem_load_0[17])), "r"(__float_as_uint(_tmem_load_0[18])), "r"(__float_as_uint(_tmem_load_0[19])));
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(stg_b_1_0 + 80), "r"(__float_as_uint(_tmem_load_0[20])), "r"(__float_as_uint(_tmem_load_0[21])), "r"(__float_as_uint(_tmem_load_0[22])), "r"(__float_as_uint(_tmem_load_0[23])));
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(stg_b_1_0 + 96), "r"(__float_as_uint(_tmem_load_0[24])), "r"(__float_as_uint(_tmem_load_0[25])), "r"(__float_as_uint(_tmem_load_0[26])), "r"(__float_as_uint(_tmem_load_0[27])));
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(stg_b_1_0 + 112), "r"(__float_as_uint(_tmem_load_0[28])), "r"(__float_as_uint(_tmem_load_0[29])), "r"(__float_as_uint(_tmem_load_0[30])), "r"(__float_as_uint(_tmem_load_0[31])));
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(stg_b_1_0 + 128), "r"(__float_as_uint(_tmem_load_0[32])), "r"(__float_as_uint(_tmem_load_0[33])), "r"(__float_as_uint(_tmem_load_0[34])), "r"(__float_as_uint(_tmem_load_0[35])));
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(stg_b_1_0 + 144), "r"(__float_as_uint(_tmem_load_0[36])), "r"(__float_as_uint(_tmem_load_0[37])), "r"(__float_as_uint(_tmem_load_0[38])), "r"(__float_as_uint(_tmem_load_0[39])));
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(stg_b_1_0 + 160), "r"(__float_as_uint(_tmem_load_0[40])), "r"(__float_as_uint(_tmem_load_0[41])), "r"(__float_as_uint(_tmem_load_0[42])), "r"(__float_as_uint(_tmem_load_0[43])));
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(stg_b_1_0 + 176), "r"(__float_as_uint(_tmem_load_0[44])), "r"(__float_as_uint(_tmem_load_0[45])), "r"(__float_as_uint(_tmem_load_0[46])), "r"(__float_as_uint(_tmem_load_0[47])));
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(stg_b_1_0 + 192), "r"(__float_as_uint(_tmem_load_0[48])), "r"(__float_as_uint(_tmem_load_0[49])), "r"(__float_as_uint(_tmem_load_0[50])), "r"(__float_as_uint(_tmem_load_0[51])));
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(stg_b_1_0 + 208), "r"(__float_as_uint(_tmem_load_0[52])), "r"(__float_as_uint(_tmem_load_0[53])), "r"(__float_as_uint(_tmem_load_0[54])), "r"(__float_as_uint(_tmem_load_0[55])));
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(stg_b_1_0 + 224), "r"(__float_as_uint(_tmem_load_0[56])), "r"(__float_as_uint(_tmem_load_0[57])), "r"(__float_as_uint(_tmem_load_0[58])), "r"(__float_as_uint(_tmem_load_0[59])));
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(stg_b_1_0 + 240), "r"(__float_as_uint(_tmem_load_0[60])), "r"(__float_as_uint(_tmem_load_0[61])), "r"(__float_as_uint(_tmem_load_0[62])), "r"(__float_as_uint(_tmem_load_0[63])));
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(stg_b_1_0 + 256), "r"(__float_as_uint(_tmem_load_0[64])), "r"(__float_as_uint(_tmem_load_0[65])), "r"(__float_as_uint(_tmem_load_0[66])), "r"(__float_as_uint(_tmem_load_0[67])));
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(stg_b_1_0 + 272), "r"(__float_as_uint(_tmem_load_0[68])), "r"(__float_as_uint(_tmem_load_0[69])), "r"(__float_as_uint(_tmem_load_0[70])), "r"(__float_as_uint(_tmem_load_0[71])));
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(stg_b_1_0 + 288), "r"(__float_as_uint(_tmem_load_0[72])), "r"(__float_as_uint(_tmem_load_0[73])), "r"(__float_as_uint(_tmem_load_0[74])), "r"(__float_as_uint(_tmem_load_0[75])));
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(stg_b_1_0 + 304), "r"(__float_as_uint(_tmem_load_0[76])), "r"(__float_as_uint(_tmem_load_0[77])), "r"(__float_as_uint(_tmem_load_0[78])), "r"(__float_as_uint(_tmem_load_0[79])));
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(stg_b_1_0 + 320), "r"(__float_as_uint(_tmem_load_0[80])), "r"(__float_as_uint(_tmem_load_0[81])), "r"(__float_as_uint(_tmem_load_0[82])), "r"(__float_as_uint(_tmem_load_0[83])));
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(stg_b_1_0 + 336), "r"(__float_as_uint(_tmem_load_0[84])), "r"(__float_as_uint(_tmem_load_0[85])), "r"(__float_as_uint(_tmem_load_0[86])), "r"(__float_as_uint(_tmem_load_0[87])));
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(stg_b_1_0 + 352), "r"(__float_as_uint(_tmem_load_0[88])), "r"(__float_as_uint(_tmem_load_0[89])), "r"(__float_as_uint(_tmem_load_0[90])), "r"(__float_as_uint(_tmem_load_0[91])));
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(stg_b_1_0 + 368), "r"(__float_as_uint(_tmem_load_0[92])), "r"(__float_as_uint(_tmem_load_0[93])), "r"(__float_as_uint(_tmem_load_0[94])), "r"(__float_as_uint(_tmem_load_0[95])));
                }
            }
            if (cs_p_2 == 1) {
                if (col_part == 0) {
                    unsigned int stg_b_1_1 = smem_xch_addr + 51200 + xch_row;
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(stg_b_1_1), "r"(__float_as_uint(_tmem_load_0[0])), "r"(__float_as_uint(_tmem_load_0[1])), "r"(__float_as_uint(_tmem_load_0[2])), "r"(__float_as_uint(_tmem_load_0[3])));
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(stg_b_1_1 + 16), "r"(__float_as_uint(_tmem_load_0[4])), "r"(__float_as_uint(_tmem_load_0[5])), "r"(__float_as_uint(_tmem_load_0[6])), "r"(__float_as_uint(_tmem_load_0[7])));
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(stg_b_1_1 + 32), "r"(__float_as_uint(_tmem_load_0[8])), "r"(__float_as_uint(_tmem_load_0[9])), "r"(__float_as_uint(_tmem_load_0[10])), "r"(__float_as_uint(_tmem_load_0[11])));
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(stg_b_1_1 + 48), "r"(__float_as_uint(_tmem_load_0[12])), "r"(__float_as_uint(_tmem_load_0[13])), "r"(__float_as_uint(_tmem_load_0[14])), "r"(__float_as_uint(_tmem_load_0[15])));
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(stg_b_1_1 + 64), "r"(__float_as_uint(_tmem_load_0[16])), "r"(__float_as_uint(_tmem_load_0[17])), "r"(__float_as_uint(_tmem_load_0[18])), "r"(__float_as_uint(_tmem_load_0[19])));
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(stg_b_1_1 + 80), "r"(__float_as_uint(_tmem_load_0[20])), "r"(__float_as_uint(_tmem_load_0[21])), "r"(__float_as_uint(_tmem_load_0[22])), "r"(__float_as_uint(_tmem_load_0[23])));
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(stg_b_1_1 + 96), "r"(__float_as_uint(_tmem_load_0[24])), "r"(__float_as_uint(_tmem_load_0[25])), "r"(__float_as_uint(_tmem_load_0[26])), "r"(__float_as_uint(_tmem_load_0[27])));
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(stg_b_1_1 + 112), "r"(__float_as_uint(_tmem_load_0[28])), "r"(__float_as_uint(_tmem_load_0[29])), "r"(__float_as_uint(_tmem_load_0[30])), "r"(__float_as_uint(_tmem_load_0[31])));
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(stg_b_1_1 + 128), "r"(__float_as_uint(_tmem_load_0[32])), "r"(__float_as_uint(_tmem_load_0[33])), "r"(__float_as_uint(_tmem_load_0[34])), "r"(__float_as_uint(_tmem_load_0[35])));
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(stg_b_1_1 + 144), "r"(__float_as_uint(_tmem_load_0[36])), "r"(__float_as_uint(_tmem_load_0[37])), "r"(__float_as_uint(_tmem_load_0[38])), "r"(__float_as_uint(_tmem_load_0[39])));
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(stg_b_1_1 + 160), "r"(__float_as_uint(_tmem_load_0[40])), "r"(__float_as_uint(_tmem_load_0[41])), "r"(__float_as_uint(_tmem_load_0[42])), "r"(__float_as_uint(_tmem_load_0[43])));
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(stg_b_1_1 + 176), "r"(__float_as_uint(_tmem_load_0[44])), "r"(__float_as_uint(_tmem_load_0[45])), "r"(__float_as_uint(_tmem_load_0[46])), "r"(__float_as_uint(_tmem_load_0[47])));
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(stg_b_1_1 + 192), "r"(__float_as_uint(_tmem_load_0[48])), "r"(__float_as_uint(_tmem_load_0[49])), "r"(__float_as_uint(_tmem_load_0[50])), "r"(__float_as_uint(_tmem_load_0[51])));
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(stg_b_1_1 + 208), "r"(__float_as_uint(_tmem_load_0[52])), "r"(__float_as_uint(_tmem_load_0[53])), "r"(__float_as_uint(_tmem_load_0[54])), "r"(__float_as_uint(_tmem_load_0[55])));
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(stg_b_1_1 + 224), "r"(__float_as_uint(_tmem_load_0[56])), "r"(__float_as_uint(_tmem_load_0[57])), "r"(__float_as_uint(_tmem_load_0[58])), "r"(__float_as_uint(_tmem_load_0[59])));
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(stg_b_1_1 + 240), "r"(__float_as_uint(_tmem_load_0[60])), "r"(__float_as_uint(_tmem_load_0[61])), "r"(__float_as_uint(_tmem_load_0[62])), "r"(__float_as_uint(_tmem_load_0[63])));
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(stg_b_1_1 + 256), "r"(__float_as_uint(_tmem_load_0[64])), "r"(__float_as_uint(_tmem_load_0[65])), "r"(__float_as_uint(_tmem_load_0[66])), "r"(__float_as_uint(_tmem_load_0[67])));
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(stg_b_1_1 + 272), "r"(__float_as_uint(_tmem_load_0[68])), "r"(__float_as_uint(_tmem_load_0[69])), "r"(__float_as_uint(_tmem_load_0[70])), "r"(__float_as_uint(_tmem_load_0[71])));
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(stg_b_1_1 + 288), "r"(__float_as_uint(_tmem_load_0[72])), "r"(__float_as_uint(_tmem_load_0[73])), "r"(__float_as_uint(_tmem_load_0[74])), "r"(__float_as_uint(_tmem_load_0[75])));
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(stg_b_1_1 + 304), "r"(__float_as_uint(_tmem_load_0[76])), "r"(__float_as_uint(_tmem_load_0[77])), "r"(__float_as_uint(_tmem_load_0[78])), "r"(__float_as_uint(_tmem_load_0[79])));
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(stg_b_1_1 + 320), "r"(__float_as_uint(_tmem_load_0[80])), "r"(__float_as_uint(_tmem_load_0[81])), "r"(__float_as_uint(_tmem_load_0[82])), "r"(__float_as_uint(_tmem_load_0[83])));
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(stg_b_1_1 + 336), "r"(__float_as_uint(_tmem_load_0[84])), "r"(__float_as_uint(_tmem_load_0[85])), "r"(__float_as_uint(_tmem_load_0[86])), "r"(__float_as_uint(_tmem_load_0[87])));
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(stg_b_1_1 + 352), "r"(__float_as_uint(_tmem_load_0[88])), "r"(__float_as_uint(_tmem_load_0[89])), "r"(__float_as_uint(_tmem_load_0[90])), "r"(__float_as_uint(_tmem_load_0[91])));
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(stg_b_1_1 + 368), "r"(__float_as_uint(_tmem_load_0[92])), "r"(__float_as_uint(_tmem_load_0[93])), "r"(__float_as_uint(_tmem_load_0[94])), "r"(__float_as_uint(_tmem_load_0[95])));
                }
            }
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            asm volatile("tcgen05.fence::before_thread_sync;");
            asm volatile("barrier.sync 3, 256;" ::: "memory");
            if (warp == 2) {
                mbarrier_wait_cluster_hint(ring_dead_addr, 0, 10000000);
                if (elect_sync()) {
                    if (cs_r == 0) {
                        if (cs_p_2 == 0) {
                            uint32_t _mbar_peer_0;
                            asm(
                                "mapa.shared::cluster.u32 %0, %1, %2;"
                                : "=r"(_mbar_peer_0) : "r"(red_full_addr), "r"(2));
                            uint32_t _cluster_mapa_1;
                            asm volatile(
                                "mapa.shared::cluster.u32 %0, %1, %2;"
                                : "=r"(_cluster_mapa_1) : "r"(smem_xch_addr), "r"(2));
                            asm volatile(
                                "cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes"
                                " [%0], [%1], %2, [%3];"
                                :: "r"(_cluster_mapa_1), "r"(smem_xch_addr + 51200), "r"((uint32_t)(51200)), "r"(_mbar_peer_0)
                                : "memory");
                            asm volatile("cp.async.bulk.commit_group;");
                        }
                        if (cs_p_2 == 1) {
                            uint32_t _mbar_peer_2;
                            asm(
                                "mapa.shared::cluster.u32 %0, %1, %2;"
                                : "=r"(_mbar_peer_2) : "r"(red_full_addr), "r"(0));
                            uint32_t _cluster_mapa_3;
                            asm volatile(
                                "mapa.shared::cluster.u32 %0, %1, %2;"
                                : "=r"(_cluster_mapa_3) : "r"(smem_xch_addr), "r"(0));
                            asm volatile(
                                "cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes"
                                " [%0], [%1], %2, [%3];"
                                :: "r"(_cluster_mapa_3), "r"(smem_xch_addr + 51200), "r"((uint32_t)(51200)), "r"(_mbar_peer_2)
                                : "memory");
                            asm volatile("cp.async.bulk.commit_group;");
                        }
                    }
                    if (cs_r == 1) {
                        if (cs_p_2 == 0) {
                            uint32_t _mbar_peer_4;
                            asm(
                                "mapa.shared::cluster.u32 %0, %1, %2;"
                                : "=r"(_mbar_peer_4) : "r"(red_full_addr), "r"(3));
                            uint32_t _cluster_mapa_5;
                            asm volatile(
                                "mapa.shared::cluster.u32 %0, %1, %2;"
                                : "=r"(_cluster_mapa_5) : "r"(smem_xch_addr), "r"(3));
                            asm volatile(
                                "cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes"
                                " [%0], [%1], %2, [%3];"
                                :: "r"(_cluster_mapa_5), "r"(smem_xch_addr + 51200), "r"((uint32_t)(51200)), "r"(_mbar_peer_4)
                                : "memory");
                            asm volatile("cp.async.bulk.commit_group;");
                        }
                        if (cs_p_2 == 1) {
                            uint32_t _mbar_peer_6;
                            asm(
                                "mapa.shared::cluster.u32 %0, %1, %2;"
                                : "=r"(_mbar_peer_6) : "r"(red_full_addr), "r"(1));
                            uint32_t _cluster_mapa_7;
                            asm volatile(
                                "mapa.shared::cluster.u32 %0, %1, %2;"
                                : "=r"(_cluster_mapa_7) : "r"(smem_xch_addr), "r"(1));
                            asm volatile(
                                "cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes"
                                " [%0], [%1], %2, [%3];"
                                :: "r"(_cluster_mapa_7), "r"(smem_xch_addr + 51200), "r"((uint32_t)(51200)), "r"(_mbar_peer_6)
                                : "memory");
                            asm volatile("cp.async.bulk.commit_group;");
                        }
                    }
                    mbarrier_arrive(red_full_addr);
                }
            }
            mbarrier_wait_cluster_hint(red_full_addr, 0, 10000000);
            if (col_part == 0) {
                if (cs_p_2 == 0) {
                    float total[96];
                    total[0] = _tmem_load_0[0];
                    total[1] = _tmem_load_0[1];
                    total[2] = _tmem_load_0[2];
                    total[3] = _tmem_load_0[3];
                    total[4] = _tmem_load_0[4];
                    total[5] = _tmem_load_0[5];
                    total[6] = _tmem_load_0[6];
                    total[7] = _tmem_load_0[7];
                    total[8] = _tmem_load_0[8];
                    total[9] = _tmem_load_0[9];
                    total[10] = _tmem_load_0[10];
                    total[11] = _tmem_load_0[11];
                    total[12] = _tmem_load_0[12];
                    total[13] = _tmem_load_0[13];
                    total[14] = _tmem_load_0[14];
                    total[15] = _tmem_load_0[15];
                    total[16] = _tmem_load_0[16];
                    total[17] = _tmem_load_0[17];
                    total[18] = _tmem_load_0[18];
                    total[19] = _tmem_load_0[19];
                    total[20] = _tmem_load_0[20];
                    total[21] = _tmem_load_0[21];
                    total[22] = _tmem_load_0[22];
                    total[23] = _tmem_load_0[23];
                    total[24] = _tmem_load_0[24];
                    total[25] = _tmem_load_0[25];
                    total[26] = _tmem_load_0[26];
                    total[27] = _tmem_load_0[27];
                    total[28] = _tmem_load_0[28];
                    total[29] = _tmem_load_0[29];
                    total[30] = _tmem_load_0[30];
                    total[31] = _tmem_load_0[31];
                    total[32] = _tmem_load_0[32];
                    total[33] = _tmem_load_0[33];
                    total[34] = _tmem_load_0[34];
                    total[35] = _tmem_load_0[35];
                    total[36] = _tmem_load_0[36];
                    total[37] = _tmem_load_0[37];
                    total[38] = _tmem_load_0[38];
                    total[39] = _tmem_load_0[39];
                    total[40] = _tmem_load_0[40];
                    total[41] = _tmem_load_0[41];
                    total[42] = _tmem_load_0[42];
                    total[43] = _tmem_load_0[43];
                    total[44] = _tmem_load_0[44];
                    total[45] = _tmem_load_0[45];
                    total[46] = _tmem_load_0[46];
                    total[47] = _tmem_load_0[47];
                    total[48] = _tmem_load_0[48];
                    total[49] = _tmem_load_0[49];
                    total[50] = _tmem_load_0[50];
                    total[51] = _tmem_load_0[51];
                    total[52] = _tmem_load_0[52];
                    total[53] = _tmem_load_0[53];
                    total[54] = _tmem_load_0[54];
                    total[55] = _tmem_load_0[55];
                    total[56] = _tmem_load_0[56];
                    total[57] = _tmem_load_0[57];
                    total[58] = _tmem_load_0[58];
                    total[59] = _tmem_load_0[59];
                    total[60] = _tmem_load_0[60];
                    total[61] = _tmem_load_0[61];
                    total[62] = _tmem_load_0[62];
                    total[63] = _tmem_load_0[63];
                    total[64] = _tmem_load_0[64];
                    total[65] = _tmem_load_0[65];
                    total[66] = _tmem_load_0[66];
                    total[67] = _tmem_load_0[67];
                    total[68] = _tmem_load_0[68];
                    total[69] = _tmem_load_0[69];
                    total[70] = _tmem_load_0[70];
                    total[71] = _tmem_load_0[71];
                    total[72] = _tmem_load_0[72];
                    total[73] = _tmem_load_0[73];
                    total[74] = _tmem_load_0[74];
                    total[75] = _tmem_load_0[75];
                    total[76] = _tmem_load_0[76];
                    total[77] = _tmem_load_0[77];
                    total[78] = _tmem_load_0[78];
                    total[79] = _tmem_load_0[79];
                    total[80] = _tmem_load_0[80];
                    total[81] = _tmem_load_0[81];
                    total[82] = _tmem_load_0[82];
                    total[83] = _tmem_load_0[83];
                    total[84] = _tmem_load_0[84];
                    total[85] = _tmem_load_0[85];
                    total[86] = _tmem_load_0[86];
                    total[87] = _tmem_load_0[87];
                    total[88] = _tmem_load_0[88];
                    total[89] = _tmem_load_0[89];
                    total[90] = _tmem_load_0[90];
                    total[91] = _tmem_load_0[91];
                    total[92] = _tmem_load_0[92];
                    total[93] = _tmem_load_0[93];
                    total[94] = _tmem_load_0[94];
                    total[95] = _tmem_load_0[95];
                    float inb[96];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&inb[0])), "=r"(*reinterpret_cast<uint32_t*>(&inb[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&inb[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&inb[(0) + 3]))
                        : "r"(smem_xch_addr + xch_row));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&inb[4])), "=r"(*reinterpret_cast<uint32_t*>(&inb[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&inb[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&inb[(4) + 3]))
                        : "r"(smem_xch_addr + xch_row + 16));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&inb[8])), "=r"(*reinterpret_cast<uint32_t*>(&inb[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&inb[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&inb[(8) + 3]))
                        : "r"(smem_xch_addr + xch_row + 32));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&inb[12])), "=r"(*reinterpret_cast<uint32_t*>(&inb[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&inb[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&inb[(12) + 3]))
                        : "r"(smem_xch_addr + xch_row + 48));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&inb[16])), "=r"(*reinterpret_cast<uint32_t*>(&inb[(16) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&inb[(16) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&inb[(16) + 3]))
                        : "r"(smem_xch_addr + xch_row + 64));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&inb[20])), "=r"(*reinterpret_cast<uint32_t*>(&inb[(20) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&inb[(20) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&inb[(20) + 3]))
                        : "r"(smem_xch_addr + xch_row + 80));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&inb[24])), "=r"(*reinterpret_cast<uint32_t*>(&inb[(24) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&inb[(24) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&inb[(24) + 3]))
                        : "r"(smem_xch_addr + xch_row + 96));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&inb[28])), "=r"(*reinterpret_cast<uint32_t*>(&inb[(28) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&inb[(28) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&inb[(28) + 3]))
                        : "r"(smem_xch_addr + xch_row + 112));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&inb[32])), "=r"(*reinterpret_cast<uint32_t*>(&inb[(32) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&inb[(32) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&inb[(32) + 3]))
                        : "r"(smem_xch_addr + xch_row + 128));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&inb[36])), "=r"(*reinterpret_cast<uint32_t*>(&inb[(36) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&inb[(36) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&inb[(36) + 3]))
                        : "r"(smem_xch_addr + xch_row + 144));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&inb[40])), "=r"(*reinterpret_cast<uint32_t*>(&inb[(40) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&inb[(40) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&inb[(40) + 3]))
                        : "r"(smem_xch_addr + xch_row + 160));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&inb[44])), "=r"(*reinterpret_cast<uint32_t*>(&inb[(44) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&inb[(44) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&inb[(44) + 3]))
                        : "r"(smem_xch_addr + xch_row + 176));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&inb[48])), "=r"(*reinterpret_cast<uint32_t*>(&inb[(48) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&inb[(48) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&inb[(48) + 3]))
                        : "r"(smem_xch_addr + xch_row + 192));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&inb[52])), "=r"(*reinterpret_cast<uint32_t*>(&inb[(52) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&inb[(52) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&inb[(52) + 3]))
                        : "r"(smem_xch_addr + xch_row + 208));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&inb[56])), "=r"(*reinterpret_cast<uint32_t*>(&inb[(56) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&inb[(56) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&inb[(56) + 3]))
                        : "r"(smem_xch_addr + xch_row + 224));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&inb[60])), "=r"(*reinterpret_cast<uint32_t*>(&inb[(60) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&inb[(60) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&inb[(60) + 3]))
                        : "r"(smem_xch_addr + xch_row + 240));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&inb[64])), "=r"(*reinterpret_cast<uint32_t*>(&inb[(64) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&inb[(64) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&inb[(64) + 3]))
                        : "r"(smem_xch_addr + xch_row + 256));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&inb[68])), "=r"(*reinterpret_cast<uint32_t*>(&inb[(68) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&inb[(68) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&inb[(68) + 3]))
                        : "r"(smem_xch_addr + xch_row + 272));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&inb[72])), "=r"(*reinterpret_cast<uint32_t*>(&inb[(72) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&inb[(72) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&inb[(72) + 3]))
                        : "r"(smem_xch_addr + xch_row + 288));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&inb[76])), "=r"(*reinterpret_cast<uint32_t*>(&inb[(76) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&inb[(76) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&inb[(76) + 3]))
                        : "r"(smem_xch_addr + xch_row + 304));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&inb[80])), "=r"(*reinterpret_cast<uint32_t*>(&inb[(80) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&inb[(80) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&inb[(80) + 3]))
                        : "r"(smem_xch_addr + xch_row + 320));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&inb[84])), "=r"(*reinterpret_cast<uint32_t*>(&inb[(84) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&inb[(84) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&inb[(84) + 3]))
                        : "r"(smem_xch_addr + xch_row + 336));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&inb[88])), "=r"(*reinterpret_cast<uint32_t*>(&inb[(88) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&inb[(88) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&inb[(88) + 3]))
                        : "r"(smem_xch_addr + xch_row + 352));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&inb[92])), "=r"(*reinterpret_cast<uint32_t*>(&inb[(92) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&inb[(92) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&inb[(92) + 3]))
                        : "r"(smem_xch_addr + xch_row + 368));
                    total[0] = total[0] + inb[0];
                    total[1] = total[1] + inb[1];
                    total[2] = total[2] + inb[2];
                    total[3] = total[3] + inb[3];
                    total[4] = total[4] + inb[4];
                    total[5] = total[5] + inb[5];
                    total[6] = total[6] + inb[6];
                    total[7] = total[7] + inb[7];
                    total[8] = total[8] + inb[8];
                    total[9] = total[9] + inb[9];
                    total[10] = total[10] + inb[10];
                    total[11] = total[11] + inb[11];
                    total[12] = total[12] + inb[12];
                    total[13] = total[13] + inb[13];
                    total[14] = total[14] + inb[14];
                    total[15] = total[15] + inb[15];
                    total[16] = total[16] + inb[16];
                    total[17] = total[17] + inb[17];
                    total[18] = total[18] + inb[18];
                    total[19] = total[19] + inb[19];
                    total[20] = total[20] + inb[20];
                    total[21] = total[21] + inb[21];
                    total[22] = total[22] + inb[22];
                    total[23] = total[23] + inb[23];
                    total[24] = total[24] + inb[24];
                    total[25] = total[25] + inb[25];
                    total[26] = total[26] + inb[26];
                    total[27] = total[27] + inb[27];
                    total[28] = total[28] + inb[28];
                    total[29] = total[29] + inb[29];
                    total[30] = total[30] + inb[30];
                    total[31] = total[31] + inb[31];
                    total[32] = total[32] + inb[32];
                    total[33] = total[33] + inb[33];
                    total[34] = total[34] + inb[34];
                    total[35] = total[35] + inb[35];
                    total[36] = total[36] + inb[36];
                    total[37] = total[37] + inb[37];
                    total[38] = total[38] + inb[38];
                    total[39] = total[39] + inb[39];
                    total[40] = total[40] + inb[40];
                    total[41] = total[41] + inb[41];
                    total[42] = total[42] + inb[42];
                    total[43] = total[43] + inb[43];
                    total[44] = total[44] + inb[44];
                    total[45] = total[45] + inb[45];
                    total[46] = total[46] + inb[46];
                    total[47] = total[47] + inb[47];
                    total[48] = total[48] + inb[48];
                    total[49] = total[49] + inb[49];
                    total[50] = total[50] + inb[50];
                    total[51] = total[51] + inb[51];
                    total[52] = total[52] + inb[52];
                    total[53] = total[53] + inb[53];
                    total[54] = total[54] + inb[54];
                    total[55] = total[55] + inb[55];
                    total[56] = total[56] + inb[56];
                    total[57] = total[57] + inb[57];
                    total[58] = total[58] + inb[58];
                    total[59] = total[59] + inb[59];
                    total[60] = total[60] + inb[60];
                    total[61] = total[61] + inb[61];
                    total[62] = total[62] + inb[62];
                    total[63] = total[63] + inb[63];
                    total[64] = total[64] + inb[64];
                    total[65] = total[65] + inb[65];
                    total[66] = total[66] + inb[66];
                    total[67] = total[67] + inb[67];
                    total[68] = total[68] + inb[68];
                    total[69] = total[69] + inb[69];
                    total[70] = total[70] + inb[70];
                    total[71] = total[71] + inb[71];
                    total[72] = total[72] + inb[72];
                    total[73] = total[73] + inb[73];
                    total[74] = total[74] + inb[74];
                    total[75] = total[75] + inb[75];
                    total[76] = total[76] + inb[76];
                    total[77] = total[77] + inb[77];
                    total[78] = total[78] + inb[78];
                    total[79] = total[79] + inb[79];
                    total[80] = total[80] + inb[80];
                    total[81] = total[81] + inb[81];
                    total[82] = total[82] + inb[82];
                    total[83] = total[83] + inb[83];
                    total[84] = total[84] + inb[84];
                    total[85] = total[85] + inb[85];
                    total[86] = total[86] + inb[86];
                    total[87] = total[87] + inb[87];
                    total[88] = total[88] + inb[88];
                    total[89] = total[89] + inb[89];
                    total[90] = total[90] + inb[90];
                    total[91] = total[91] + inb[91];
                    total[92] = total[92] + inb[92];
                    total[93] = total[93] + inb[93];
                    total[94] = total[94] + inb[94];
                    total[95] = total[95] + inb[95];
                    unsigned int total_bf16[48];
                    #pragma unroll
                    for (int _lp = 0; _lp < 48; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(total[_lp*2 + 0], total[_lp*2+1 + 0]));
                        total_bf16[_lp] = *(uint32_t*)&_bf2;
                    }
                    unsigned int group_base = smem_out_addr + (unsigned int)(col_part * 8192);
                    int swz = local_row >> 1 & 3;
                    #pragma unroll
                    for (int c = 0; c < 3; c++) {
                        unsigned int ts_base = group_base;
                        unsigned int ts_row = ts_base + (unsigned int)(local_row * 64);
                        if ((warp - 2) % 4 == 0) {
                            asm volatile("cp.async.bulk.wait_group.read 0;");
                        }
                        asm volatile("barrier.sync %0, 128;" :: "r"(1 + col_part) : "memory");
                        #pragma unroll
                        for (int q = 0; q < 4; q++) {
                            int s0 = c * 16 + q * 4;
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(ts_row + (unsigned int)((q ^ swz) * 16)), "r"(total_bf16[s0]), "r"(total_bf16[s0 + 1]), "r"(total_bf16[s0 + 2]), "r"(total_bf16[s0 + 3]));
                        }
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        asm volatile("barrier.sync %0, 128;" :: "r"(1 + col_part) : "memory");
                        if ((warp - 2) % 4 == 0) {
                            if (elect_sync()) {
                                tma_store_2d((&OUT), col_base + c * 32, off_m_1, ts_base);
                                asm volatile("cp.async.bulk.commit_group;");
                            }
                        }
                    }
                }
            }
            if (col_part == 1) {
                if (cs_p_2 == 1) {
                    float total_1[96];
                    float inb_1[96];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&inb_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&inb_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&inb_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&inb_1[(0) + 3]))
                        : "r"(smem_xch_addr + xch_row));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&inb_1[4])), "=r"(*reinterpret_cast<uint32_t*>(&inb_1[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&inb_1[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&inb_1[(4) + 3]))
                        : "r"(smem_xch_addr + xch_row + 16));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&inb_1[8])), "=r"(*reinterpret_cast<uint32_t*>(&inb_1[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&inb_1[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&inb_1[(8) + 3]))
                        : "r"(smem_xch_addr + xch_row + 32));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&inb_1[12])), "=r"(*reinterpret_cast<uint32_t*>(&inb_1[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&inb_1[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&inb_1[(12) + 3]))
                        : "r"(smem_xch_addr + xch_row + 48));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&inb_1[16])), "=r"(*reinterpret_cast<uint32_t*>(&inb_1[(16) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&inb_1[(16) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&inb_1[(16) + 3]))
                        : "r"(smem_xch_addr + xch_row + 64));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&inb_1[20])), "=r"(*reinterpret_cast<uint32_t*>(&inb_1[(20) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&inb_1[(20) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&inb_1[(20) + 3]))
                        : "r"(smem_xch_addr + xch_row + 80));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&inb_1[24])), "=r"(*reinterpret_cast<uint32_t*>(&inb_1[(24) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&inb_1[(24) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&inb_1[(24) + 3]))
                        : "r"(smem_xch_addr + xch_row + 96));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&inb_1[28])), "=r"(*reinterpret_cast<uint32_t*>(&inb_1[(28) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&inb_1[(28) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&inb_1[(28) + 3]))
                        : "r"(smem_xch_addr + xch_row + 112));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&inb_1[32])), "=r"(*reinterpret_cast<uint32_t*>(&inb_1[(32) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&inb_1[(32) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&inb_1[(32) + 3]))
                        : "r"(smem_xch_addr + xch_row + 128));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&inb_1[36])), "=r"(*reinterpret_cast<uint32_t*>(&inb_1[(36) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&inb_1[(36) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&inb_1[(36) + 3]))
                        : "r"(smem_xch_addr + xch_row + 144));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&inb_1[40])), "=r"(*reinterpret_cast<uint32_t*>(&inb_1[(40) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&inb_1[(40) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&inb_1[(40) + 3]))
                        : "r"(smem_xch_addr + xch_row + 160));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&inb_1[44])), "=r"(*reinterpret_cast<uint32_t*>(&inb_1[(44) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&inb_1[(44) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&inb_1[(44) + 3]))
                        : "r"(smem_xch_addr + xch_row + 176));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&inb_1[48])), "=r"(*reinterpret_cast<uint32_t*>(&inb_1[(48) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&inb_1[(48) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&inb_1[(48) + 3]))
                        : "r"(smem_xch_addr + xch_row + 192));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&inb_1[52])), "=r"(*reinterpret_cast<uint32_t*>(&inb_1[(52) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&inb_1[(52) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&inb_1[(52) + 3]))
                        : "r"(smem_xch_addr + xch_row + 208));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&inb_1[56])), "=r"(*reinterpret_cast<uint32_t*>(&inb_1[(56) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&inb_1[(56) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&inb_1[(56) + 3]))
                        : "r"(smem_xch_addr + xch_row + 224));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&inb_1[60])), "=r"(*reinterpret_cast<uint32_t*>(&inb_1[(60) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&inb_1[(60) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&inb_1[(60) + 3]))
                        : "r"(smem_xch_addr + xch_row + 240));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&inb_1[64])), "=r"(*reinterpret_cast<uint32_t*>(&inb_1[(64) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&inb_1[(64) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&inb_1[(64) + 3]))
                        : "r"(smem_xch_addr + xch_row + 256));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&inb_1[68])), "=r"(*reinterpret_cast<uint32_t*>(&inb_1[(68) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&inb_1[(68) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&inb_1[(68) + 3]))
                        : "r"(smem_xch_addr + xch_row + 272));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&inb_1[72])), "=r"(*reinterpret_cast<uint32_t*>(&inb_1[(72) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&inb_1[(72) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&inb_1[(72) + 3]))
                        : "r"(smem_xch_addr + xch_row + 288));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&inb_1[76])), "=r"(*reinterpret_cast<uint32_t*>(&inb_1[(76) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&inb_1[(76) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&inb_1[(76) + 3]))
                        : "r"(smem_xch_addr + xch_row + 304));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&inb_1[80])), "=r"(*reinterpret_cast<uint32_t*>(&inb_1[(80) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&inb_1[(80) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&inb_1[(80) + 3]))
                        : "r"(smem_xch_addr + xch_row + 320));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&inb_1[84])), "=r"(*reinterpret_cast<uint32_t*>(&inb_1[(84) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&inb_1[(84) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&inb_1[(84) + 3]))
                        : "r"(smem_xch_addr + xch_row + 336));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&inb_1[88])), "=r"(*reinterpret_cast<uint32_t*>(&inb_1[(88) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&inb_1[(88) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&inb_1[(88) + 3]))
                        : "r"(smem_xch_addr + xch_row + 352));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&inb_1[92])), "=r"(*reinterpret_cast<uint32_t*>(&inb_1[(92) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&inb_1[(92) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&inb_1[(92) + 3]))
                        : "r"(smem_xch_addr + xch_row + 368));
                    total_1[0] = inb_1[0];
                    total_1[1] = inb_1[1];
                    total_1[2] = inb_1[2];
                    total_1[3] = inb_1[3];
                    total_1[4] = inb_1[4];
                    total_1[5] = inb_1[5];
                    total_1[6] = inb_1[6];
                    total_1[7] = inb_1[7];
                    total_1[8] = inb_1[8];
                    total_1[9] = inb_1[9];
                    total_1[10] = inb_1[10];
                    total_1[11] = inb_1[11];
                    total_1[12] = inb_1[12];
                    total_1[13] = inb_1[13];
                    total_1[14] = inb_1[14];
                    total_1[15] = inb_1[15];
                    total_1[16] = inb_1[16];
                    total_1[17] = inb_1[17];
                    total_1[18] = inb_1[18];
                    total_1[19] = inb_1[19];
                    total_1[20] = inb_1[20];
                    total_1[21] = inb_1[21];
                    total_1[22] = inb_1[22];
                    total_1[23] = inb_1[23];
                    total_1[24] = inb_1[24];
                    total_1[25] = inb_1[25];
                    total_1[26] = inb_1[26];
                    total_1[27] = inb_1[27];
                    total_1[28] = inb_1[28];
                    total_1[29] = inb_1[29];
                    total_1[30] = inb_1[30];
                    total_1[31] = inb_1[31];
                    total_1[32] = inb_1[32];
                    total_1[33] = inb_1[33];
                    total_1[34] = inb_1[34];
                    total_1[35] = inb_1[35];
                    total_1[36] = inb_1[36];
                    total_1[37] = inb_1[37];
                    total_1[38] = inb_1[38];
                    total_1[39] = inb_1[39];
                    total_1[40] = inb_1[40];
                    total_1[41] = inb_1[41];
                    total_1[42] = inb_1[42];
                    total_1[43] = inb_1[43];
                    total_1[44] = inb_1[44];
                    total_1[45] = inb_1[45];
                    total_1[46] = inb_1[46];
                    total_1[47] = inb_1[47];
                    total_1[48] = inb_1[48];
                    total_1[49] = inb_1[49];
                    total_1[50] = inb_1[50];
                    total_1[51] = inb_1[51];
                    total_1[52] = inb_1[52];
                    total_1[53] = inb_1[53];
                    total_1[54] = inb_1[54];
                    total_1[55] = inb_1[55];
                    total_1[56] = inb_1[56];
                    total_1[57] = inb_1[57];
                    total_1[58] = inb_1[58];
                    total_1[59] = inb_1[59];
                    total_1[60] = inb_1[60];
                    total_1[61] = inb_1[61];
                    total_1[62] = inb_1[62];
                    total_1[63] = inb_1[63];
                    total_1[64] = inb_1[64];
                    total_1[65] = inb_1[65];
                    total_1[66] = inb_1[66];
                    total_1[67] = inb_1[67];
                    total_1[68] = inb_1[68];
                    total_1[69] = inb_1[69];
                    total_1[70] = inb_1[70];
                    total_1[71] = inb_1[71];
                    total_1[72] = inb_1[72];
                    total_1[73] = inb_1[73];
                    total_1[74] = inb_1[74];
                    total_1[75] = inb_1[75];
                    total_1[76] = inb_1[76];
                    total_1[77] = inb_1[77];
                    total_1[78] = inb_1[78];
                    total_1[79] = inb_1[79];
                    total_1[80] = inb_1[80];
                    total_1[81] = inb_1[81];
                    total_1[82] = inb_1[82];
                    total_1[83] = inb_1[83];
                    total_1[84] = inb_1[84];
                    total_1[85] = inb_1[85];
                    total_1[86] = inb_1[86];
                    total_1[87] = inb_1[87];
                    total_1[88] = inb_1[88];
                    total_1[89] = inb_1[89];
                    total_1[90] = inb_1[90];
                    total_1[91] = inb_1[91];
                    total_1[92] = inb_1[92];
                    total_1[93] = inb_1[93];
                    total_1[94] = inb_1[94];
                    total_1[95] = inb_1[95];
                    total_1[0] = total_1[0] + _tmem_load_0[0];
                    total_1[1] = total_1[1] + _tmem_load_0[1];
                    total_1[2] = total_1[2] + _tmem_load_0[2];
                    total_1[3] = total_1[3] + _tmem_load_0[3];
                    total_1[4] = total_1[4] + _tmem_load_0[4];
                    total_1[5] = total_1[5] + _tmem_load_0[5];
                    total_1[6] = total_1[6] + _tmem_load_0[6];
                    total_1[7] = total_1[7] + _tmem_load_0[7];
                    total_1[8] = total_1[8] + _tmem_load_0[8];
                    total_1[9] = total_1[9] + _tmem_load_0[9];
                    total_1[10] = total_1[10] + _tmem_load_0[10];
                    total_1[11] = total_1[11] + _tmem_load_0[11];
                    total_1[12] = total_1[12] + _tmem_load_0[12];
                    total_1[13] = total_1[13] + _tmem_load_0[13];
                    total_1[14] = total_1[14] + _tmem_load_0[14];
                    total_1[15] = total_1[15] + _tmem_load_0[15];
                    total_1[16] = total_1[16] + _tmem_load_0[16];
                    total_1[17] = total_1[17] + _tmem_load_0[17];
                    total_1[18] = total_1[18] + _tmem_load_0[18];
                    total_1[19] = total_1[19] + _tmem_load_0[19];
                    total_1[20] = total_1[20] + _tmem_load_0[20];
                    total_1[21] = total_1[21] + _tmem_load_0[21];
                    total_1[22] = total_1[22] + _tmem_load_0[22];
                    total_1[23] = total_1[23] + _tmem_load_0[23];
                    total_1[24] = total_1[24] + _tmem_load_0[24];
                    total_1[25] = total_1[25] + _tmem_load_0[25];
                    total_1[26] = total_1[26] + _tmem_load_0[26];
                    total_1[27] = total_1[27] + _tmem_load_0[27];
                    total_1[28] = total_1[28] + _tmem_load_0[28];
                    total_1[29] = total_1[29] + _tmem_load_0[29];
                    total_1[30] = total_1[30] + _tmem_load_0[30];
                    total_1[31] = total_1[31] + _tmem_load_0[31];
                    total_1[32] = total_1[32] + _tmem_load_0[32];
                    total_1[33] = total_1[33] + _tmem_load_0[33];
                    total_1[34] = total_1[34] + _tmem_load_0[34];
                    total_1[35] = total_1[35] + _tmem_load_0[35];
                    total_1[36] = total_1[36] + _tmem_load_0[36];
                    total_1[37] = total_1[37] + _tmem_load_0[37];
                    total_1[38] = total_1[38] + _tmem_load_0[38];
                    total_1[39] = total_1[39] + _tmem_load_0[39];
                    total_1[40] = total_1[40] + _tmem_load_0[40];
                    total_1[41] = total_1[41] + _tmem_load_0[41];
                    total_1[42] = total_1[42] + _tmem_load_0[42];
                    total_1[43] = total_1[43] + _tmem_load_0[43];
                    total_1[44] = total_1[44] + _tmem_load_0[44];
                    total_1[45] = total_1[45] + _tmem_load_0[45];
                    total_1[46] = total_1[46] + _tmem_load_0[46];
                    total_1[47] = total_1[47] + _tmem_load_0[47];
                    total_1[48] = total_1[48] + _tmem_load_0[48];
                    total_1[49] = total_1[49] + _tmem_load_0[49];
                    total_1[50] = total_1[50] + _tmem_load_0[50];
                    total_1[51] = total_1[51] + _tmem_load_0[51];
                    total_1[52] = total_1[52] + _tmem_load_0[52];
                    total_1[53] = total_1[53] + _tmem_load_0[53];
                    total_1[54] = total_1[54] + _tmem_load_0[54];
                    total_1[55] = total_1[55] + _tmem_load_0[55];
                    total_1[56] = total_1[56] + _tmem_load_0[56];
                    total_1[57] = total_1[57] + _tmem_load_0[57];
                    total_1[58] = total_1[58] + _tmem_load_0[58];
                    total_1[59] = total_1[59] + _tmem_load_0[59];
                    total_1[60] = total_1[60] + _tmem_load_0[60];
                    total_1[61] = total_1[61] + _tmem_load_0[61];
                    total_1[62] = total_1[62] + _tmem_load_0[62];
                    total_1[63] = total_1[63] + _tmem_load_0[63];
                    total_1[64] = total_1[64] + _tmem_load_0[64];
                    total_1[65] = total_1[65] + _tmem_load_0[65];
                    total_1[66] = total_1[66] + _tmem_load_0[66];
                    total_1[67] = total_1[67] + _tmem_load_0[67];
                    total_1[68] = total_1[68] + _tmem_load_0[68];
                    total_1[69] = total_1[69] + _tmem_load_0[69];
                    total_1[70] = total_1[70] + _tmem_load_0[70];
                    total_1[71] = total_1[71] + _tmem_load_0[71];
                    total_1[72] = total_1[72] + _tmem_load_0[72];
                    total_1[73] = total_1[73] + _tmem_load_0[73];
                    total_1[74] = total_1[74] + _tmem_load_0[74];
                    total_1[75] = total_1[75] + _tmem_load_0[75];
                    total_1[76] = total_1[76] + _tmem_load_0[76];
                    total_1[77] = total_1[77] + _tmem_load_0[77];
                    total_1[78] = total_1[78] + _tmem_load_0[78];
                    total_1[79] = total_1[79] + _tmem_load_0[79];
                    total_1[80] = total_1[80] + _tmem_load_0[80];
                    total_1[81] = total_1[81] + _tmem_load_0[81];
                    total_1[82] = total_1[82] + _tmem_load_0[82];
                    total_1[83] = total_1[83] + _tmem_load_0[83];
                    total_1[84] = total_1[84] + _tmem_load_0[84];
                    total_1[85] = total_1[85] + _tmem_load_0[85];
                    total_1[86] = total_1[86] + _tmem_load_0[86];
                    total_1[87] = total_1[87] + _tmem_load_0[87];
                    total_1[88] = total_1[88] + _tmem_load_0[88];
                    total_1[89] = total_1[89] + _tmem_load_0[89];
                    total_1[90] = total_1[90] + _tmem_load_0[90];
                    total_1[91] = total_1[91] + _tmem_load_0[91];
                    total_1[92] = total_1[92] + _tmem_load_0[92];
                    total_1[93] = total_1[93] + _tmem_load_0[93];
                    total_1[94] = total_1[94] + _tmem_load_0[94];
                    total_1[95] = total_1[95] + _tmem_load_0[95];
                    unsigned int total_bf16_1[48];
                    #pragma unroll
                    for (int _lp = 0; _lp < 48; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(total_1[_lp*2 + 0], total_1[_lp*2+1 + 0]));
                        total_bf16_1[_lp] = *(uint32_t*)&_bf2;
                    }
                    unsigned int group_base_1 = smem_out_addr + (unsigned int)(col_part * 8192);
                    int swz_1 = local_row >> 1 & 3;
                    #pragma unroll
                    for (int c_1 = 0; c_1 < 3; c_1++) {
                        unsigned int ts_base_1 = group_base_1;
                        unsigned int ts_row_1 = ts_base_1 + (unsigned int)(local_row * 64);
                        if ((warp - 2) % 4 == 0) {
                            asm volatile("cp.async.bulk.wait_group.read 0;");
                        }
                        asm volatile("barrier.sync %0, 128;" :: "r"(1 + col_part) : "memory");
                        #pragma unroll
                        for (int q_1 = 0; q_1 < 4; q_1++) {
                            int s0_1 = c_1 * 16 + q_1 * 4;
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(ts_row_1 + (unsigned int)((q_1 ^ swz_1) * 16)), "r"(total_bf16_1[s0_1]), "r"(total_bf16_1[s0_1 + 1]), "r"(total_bf16_1[s0_1 + 2]), "r"(total_bf16_1[s0_1 + 3]));
                        }
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        asm volatile("barrier.sync %0, 128;" :: "r"(1 + col_part) : "memory");
                        if ((warp - 2) % 4 == 0) {
                            if (elect_sync()) {
                                tma_store_2d((&OUT), col_base + c_1 * 32, off_m_1, ts_base_1);
                                asm volatile("cp.async.bulk.commit_group;");
                            }
                        }
                    }
                }
            }
            if ((warp - 2) % 4 == 0) {
                asm volatile("cp.async.bulk.wait_group 0;");
            }
            if (warp == 2) {
                asm volatile("cp.async.bulk.wait_group 0;");
            }
        }
    }

    // Cleanup
    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
    asm volatile("barrier.cluster.arrive.release.aligned;");
    asm volatile("barrier.cluster.wait.acquire.aligned;");

    if (warp == 0) {
        asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(256));
    }
}

} // extern "C"
