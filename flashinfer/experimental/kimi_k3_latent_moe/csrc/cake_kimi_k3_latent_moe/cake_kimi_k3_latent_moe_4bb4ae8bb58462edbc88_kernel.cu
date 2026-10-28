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
#define SMEM_SMEM_A_OFF 1024
#define SMEM_SMEM_A_STAGE_BYTES 16384
#define SMEM_SMEM_A_STRIDE 32768
#define SMEM_SMEM_B_OFF 17408
#define SMEM_SMEM_B_STAGE_BYTES 16384
#define SMEM_SMEM_B_STRIDE 32768
#define SMEM_FIX_FLAG_OFF 230464
#define SMEM_FIX_FLAG_STAGE_BYTES 16
#define SMEM_FIX_FLAG_STRIDE 16
#define SMEM_TOTAL 230528
#define BLOCK_M 128
#define BLOCK_N 256
#define B_HALF_N 128
#define BLOCK_K 64
#define CTA_GROUP 2
#define NUM_STAGES 7
#define GROUP_M 16
#define WORK_STAGES 4
#define WORK_CONSUMERS 546
#define NUM_K_ITERS 112
#define R_TILES 4
#define L_TILES 14
#define S_TILES 6
#define N_TILES 24
#define I_LOCAL 768
#define NUM_EXPERTS 896
#define LATENT 3584
#define tiles_per_group (GROUP_M * N_TILES)

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





__device__ __forceinline__ void tmem_st_x16_f32(int tmem_addr, const float* src) {
    asm volatile(
        "tcgen05.st.sync.aligned.32x32b.x16.b32"
        " [%0], {%1, %2, %3, %4, %5, %6, %7, %8,"
        "  %9, %10, %11, %12, %13, %14, %15, %16};"
        :: "r"(tmem_addr),
           "f"(src[0]),  "f"(src[1]),  "f"(src[2]),  "f"(src[3]),
           "f"(src[4]),  "f"(src[5]),  "f"(src[6]),  "f"(src[7]),
           "f"(src[8]),  "f"(src[9]),  "f"(src[10]), "f"(src[11]),
           "f"(src[12]), "f"(src[13]), "f"(src[14]), "f"(src[15]));
}


__device__ __forceinline__ float approx_exp2(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}


__device__ __forceinline__ float approx_rcp(float x) {
    float y;
    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
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

__global__ __launch_bounds__(320) __cluster_dims__(2,1,1) void
kernel_cake_kimi_k3_latent_moe_4bb4ae8bb58462edbc88(const __grid_constant__ CUtensorMap A, const __grid_constant__ CUtensorMap WG, const __grid_constant__ CUtensorMap WD, const __grid_constant__ CUtensorMap WS, float* __restrict__ logits, __nv_bfloat16* __restrict__ latent, __nv_bfloat16* __restrict__ shared_act, float* __restrict__ ws, int* __restrict__ counters, int M, int m_tiles, int num_items, int full_items, int sk_ipc, int sk_max_seg, int sk_total)
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
    #define carry_full_addr (mbar_base + 144)

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
    int* fix_flag = reinterpret_cast<int*>(smem_raw + 230464);
    const int fix_flag_addr = smem + 230464;

    // Mbarrier init (5 pipeline groups, 0 ordered-sequence groups, 19 barriers)
    // Mbarriers at smem_raw[0..152)

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
            // epilogue_done: 2 barriers, init_count=16
            mbarrier_init(smem + 128, 16);
            mbarrier_init(smem + 136, 16);
            // carry_full: 1 barriers, init_count=16
            mbarrier_init(smem + 144, 16);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();
    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");

    // TMEM alloc (512 columns, 512 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 152);
    if (warp == 0) {
        int _tmem_hold = smem + 152;
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
            unsigned int _phase_mma_done = 1;
            if (elect_sync()) {
                unsigned int this_bid = bid;
                #pragma unroll 1
                for (unsigned int _tile_iter = 0; _tile_iter < num_items; _tile_iter++) {
                    int fd_cid = (int)this_bid / CTA_GROUP;
                    int rank_i = (int)this_bid % CTA_GROUP;
                    int lane_l = fd_cid / full_items;
                    int lane_r = fd_cid % full_items;
                    int lo = lane_l * sk_total / sk_ipc;
                    int hi = (lane_l + 1) * sk_total / sk_ipc;
                    int fd_pr = m_tiles / CTA_GROUP;
                    int fd_grp = fd_pr / full_items;
                    int fd_t0 = sk_max_seg / 2;
                    int _max_0 = ((fd_t0 % fd_pr - lane_r) > (0) ? (fd_t0 % fd_pr - lane_r) : (0));
                    int fd_h0 = (_max_0 + full_items - 1) / full_items;
                    int _min_1 = ((fd_h0) < (fd_grp) ? (fd_h0) : (fd_grp));
                    int su0 = fd_t0 / fd_pr * fd_grp + _min_1;
                    int fd_uh = (hi - 1) / NUM_K_ITERS;
                    int fd_ul = lo / NUM_K_ITERS;
                    int fd_rh = ((fd_uh >= su0 && (hi - 1) % NUM_K_ITERS >= NUM_K_ITERS / 2) ? 1 : 0);
                    int fd_rl = ((fd_ul >= su0 && lo % NUM_K_ITERS >= NUM_K_ITERS / 2) ? 1 : 0);
                    int _max_2 = ((fd_uh - su0) > (0) ? (fd_uh - su0) : (0));
                    int p_hi = fd_uh + _max_2 + fd_rh;
                    int _max_3 = ((fd_ul - su0) > (0) ? (fd_ul - su0) : (0));
                    int fd_plo = fd_ul + _max_3 + fd_rl;
                    int nsegs = p_hi - fd_plo + 1;
                    #pragma unroll 1
                    for (int seg_l = 0; seg_l < nsegs; seg_l++) {
                        int fs_p = p_hi - seg_l;
                        int fs_q = fs_p - su0;
                        int fs_sp = ((fs_q >= 0) ? 1 : 0);
                        int fs_u = ((fs_sp == 1) ? su0 + fs_q / 2 : fs_p);
                        int fs_kb = ((fs_sp == 1) ? fs_q % 2 * (NUM_K_ITERS / 2) : 0);
                        int fs_ke = ((fs_sp == 1) ? fs_kb + NUM_K_ITERS / 2 : NUM_K_ITERS);
                        int fs_ub = fs_u * NUM_K_ITERS;
                        int k_lo = ((fs_kb) > (lo - fs_ub) ? (fs_kb) : (lo - fs_ub));
                        int k_hi = ((fs_ke) < (hi - fs_ub) ? (fs_ke) : (hi - fs_ub));
                        int fs_pr = m_tiles / CTA_GROUP;
                        int fs_grp = fs_pr / full_items;
                        int pseudo = (fs_u / fs_grp * fs_pr + fs_u % fs_grp * full_items + lane_r) * CTA_GROUP + rank_i;
                        int group = pseudo / tiles_per_group;
                        int first_m = group * GROUP_M;
                        int remaining = m_tiles - first_m;
                        int group_size = ((remaining >= GROUP_M) ? GROUP_M : remaining);
                        int local = pseudo % tiles_per_group;
                        int bid_m = first_m + local % group_size;
                        int bid_n = local / group_size;
                        int off_m = bid_m * BLOCK_M;
                        int cls = ((bid_n < R_TILES) ? 0 : ((bid_n < R_TILES + L_TILES) ? 1 : 2));
                        int row_r = bid_n * BLOCK_N + cta_rank * B_HALF_N;
                        int row_l = (bid_n - R_TILES) * BLOCK_N + cta_rank * B_HALF_N;
                        int row_s = cta_rank * I_LOCAL + (bid_n - R_TILES - L_TILES) * B_HALF_N;
                        #pragma unroll 1
                        for (int iter_k = k_lo; iter_k < k_hi; iter_k++) {
                            mbarrier_wait(mma_done_addr + (load_stage) * 8, _phase_mma_done);
                            int off_k = iter_k * BLOCK_K;
                            tma_3d_gmem2smem_cta2(smem_a_addr + load_stage * 32768, (&A), 0, off_m, off_k / 64, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                            if (cls == 0) {
                                asm volatile(
                                    "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::2.L2::cache_hint"
                                    " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                    :: "r"(smem_b_addr + load_stage * 32768), "l"((&WG)), "r"(0), "r"(row_r), "r"(off_k / 64),
                                       "r"(((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF)), "l"(0x12F0000000000000ULL) : "memory");
                            }
                            if (cls == 1) {
                                asm volatile(
                                    "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::2.L2::cache_hint"
                                    " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                    :: "r"(smem_b_addr + load_stage * 32768), "l"((&WD)), "r"(0), "r"(row_l), "r"(off_k / 64),
                                       "r"(((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF)), "l"(0x12F0000000000000ULL) : "memory");
                            }
                            if (cls == 2) {
                                asm volatile(
                                    "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::2.L2::cache_hint"
                                    " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                    :: "r"(smem_b_addr + load_stage * 32768), "l"((&WS)), "r"(0), "r"(row_s), "r"(off_k / 64),
                                       "r"(((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF)), "l"(0x12F0000000000000ULL) : "memory");
                            }
                            asm volatile(
                                "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                                :: "r"((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF), "r"((uint32_t)(32768)) : "memory");
                            load_stage += 1;
                            if (load_stage == 7) { load_stage = 0; _phase_mma_done ^= 1; }
                        }
                    }
                    break;
                }
                #pragma unroll 1
                for (int _ = 0; _ < 7; _++) {
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
            unsigned int this_bid_m = bid;
            unsigned int _phase_epilogue_done = 1;
            unsigned int _phase_tma_full = 0;
            if (cta_rank == 0) {
                #pragma unroll 1
                for (unsigned int _tile_iter_1 = 0; _tile_iter_1 < num_items; _tile_iter_1++) {
                    int fd_cid_1 = (int)this_bid_m / CTA_GROUP;
                    int rank_i_1 = (int)this_bid_m % CTA_GROUP;
                    int lane_l_1 = fd_cid_1 / full_items;
                    int lane_r_1 = fd_cid_1 % full_items;
                    int lo_1 = lane_l_1 * sk_total / sk_ipc;
                    int hi_1 = (lane_l_1 + 1) * sk_total / sk_ipc;
                    int fd_pr_1 = m_tiles / CTA_GROUP;
                    int fd_grp_1 = fd_pr_1 / full_items;
                    int fd_t0_1 = sk_max_seg / 2;
                    int _max_0_1 = ((fd_t0_1 % fd_pr_1 - lane_r_1) > (0) ? (fd_t0_1 % fd_pr_1 - lane_r_1) : (0));
                    int fd_h0_1 = (_max_0_1 + full_items - 1) / full_items;
                    int _min_1_1 = ((fd_h0_1) < (fd_grp_1) ? (fd_h0_1) : (fd_grp_1));
                    int su0_1 = fd_t0_1 / fd_pr_1 * fd_grp_1 + _min_1_1;
                    int fd_uh_1 = (hi_1 - 1) / NUM_K_ITERS;
                    int fd_ul_1 = lo_1 / NUM_K_ITERS;
                    int fd_rh_1 = ((fd_uh_1 >= su0_1 && (hi_1 - 1) % NUM_K_ITERS >= NUM_K_ITERS / 2) ? 1 : 0);
                    int fd_rl_1 = ((fd_ul_1 >= su0_1 && lo_1 % NUM_K_ITERS >= NUM_K_ITERS / 2) ? 1 : 0);
                    int _max_2_1 = ((fd_uh_1 - su0_1) > (0) ? (fd_uh_1 - su0_1) : (0));
                    int p_hi_1 = fd_uh_1 + _max_2_1 + fd_rh_1;
                    int _max_3_1 = ((fd_ul_1 - su0_1) > (0) ? (fd_ul_1 - su0_1) : (0));
                    int fd_plo_1 = fd_ul_1 + _max_3_1 + fd_rl_1;
                    int nsegs_1 = p_hi_1 - fd_plo_1 + 1;
                    #pragma unroll 1
                    for (int seg_m = 0; seg_m < nsegs_1; seg_m++) {
                        mbarrier_wait(epilogue_done_addr + (mma_epi_stage) * 8, _phase_epilogue_done);
                        int fs_p_1 = p_hi_1 - seg_m;
                        int fs_q_1 = fs_p_1 - su0_1;
                        int fs_sp_1 = ((fs_q_1 >= 0) ? 1 : 0);
                        int fs_u_1 = ((fs_sp_1 == 1) ? su0_1 + fs_q_1 / 2 : fs_p_1);
                        int fs_kb_1 = ((fs_sp_1 == 1) ? fs_q_1 % 2 * (NUM_K_ITERS / 2) : 0);
                        int fs_ke_1 = ((fs_sp_1 == 1) ? fs_kb_1 + NUM_K_ITERS / 2 : NUM_K_ITERS);
                        int fs_ub_1 = fs_u_1 * NUM_K_ITERS;
                        int k_lo_1 = ((fs_kb_1) > (lo_1 - fs_ub_1) ? (fs_kb_1) : (lo_1 - fs_ub_1));
                        int k_hi_1 = ((fs_ke_1) < (hi_1 - fs_ub_1) ? (fs_ke_1) : (hi_1 - fs_ub_1));
                        int fs_pr_1 = m_tiles / CTA_GROUP;
                        int fs_grp_1 = fs_pr_1 / full_items;
                        int pseudo_1 = (fs_u_1 / fs_grp_1 * fs_pr_1 + fs_u_1 % fs_grp_1 * full_items + lane_r_1) * CTA_GROUP + rank_i_1;
                        int carried = ((k_lo_1 > fs_kb_1 && sk_max_seg % 2 == 0) ? 1 : 0);
                        if (carried == 1) {
                            mbarrier_wait(carry_full_addr, 0);
                        }
                        #pragma unroll 1
                        for (int iter_k_1 = k_lo_1; iter_k_1 < k_hi_1; iter_k_1++) {
                            mbarrier_wait(tma_full_addr + (mma_tma_stage) * 8, _phase_tma_full);
                            asm volatile("tcgen05.fence::after_thread_sync;");
                            int init_flag = ((iter_k_1 == k_lo_1 && carried == 0) ? 1 : 0);
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
                    }
                    break;
                }
            }
        }
    }
    // ---- Role: epilogue ----
    if (warp >= 2 && warp <= 9) {
        { // epilogue_main
            unsigned int epi_stage = 0;
            const int epi_warp = warp % 4;
            const int col_half = (warp - 2) / 4;
            const int local_row = epi_warp * 32 + lane;
            unsigned int this_bid_1 = bid;
            unsigned int _phase_mainloop_done = 0;
            #pragma unroll 1
            for (unsigned int _tile_iter_2 = 0; _tile_iter_2 < num_items; _tile_iter_2++) {
                if (_tile_iter_2 != 0) {
                    asm volatile("trap;" ::: "memory");
                }
                int fd_cid_2 = (int)this_bid_1 / CTA_GROUP;
                int rank_i_2 = (int)this_bid_1 % CTA_GROUP;
                int lane_l_2 = fd_cid_2 / full_items;
                int lane_r_2 = fd_cid_2 % full_items;
                int lo_2 = lane_l_2 * sk_total / sk_ipc;
                int hi_2 = (lane_l_2 + 1) * sk_total / sk_ipc;
                int fd_pr_2 = m_tiles / CTA_GROUP;
                int fd_grp_2 = fd_pr_2 / full_items;
                int fd_t0_2 = sk_max_seg / 2;
                int _max_0_2 = ((fd_t0_2 % fd_pr_2 - lane_r_2) > (0) ? (fd_t0_2 % fd_pr_2 - lane_r_2) : (0));
                int fd_h0_2 = (_max_0_2 + full_items - 1) / full_items;
                int _min_1_2 = ((fd_h0_2) < (fd_grp_2) ? (fd_h0_2) : (fd_grp_2));
                int su0_2 = fd_t0_2 / fd_pr_2 * fd_grp_2 + _min_1_2;
                int fd_uh_2 = (hi_2 - 1) / NUM_K_ITERS;
                int fd_ul_2 = lo_2 / NUM_K_ITERS;
                int fd_rh_2 = ((fd_uh_2 >= su0_2 && (hi_2 - 1) % NUM_K_ITERS >= NUM_K_ITERS / 2) ? 1 : 0);
                int fd_rl_2 = ((fd_ul_2 >= su0_2 && lo_2 % NUM_K_ITERS >= NUM_K_ITERS / 2) ? 1 : 0);
                int _max_2_2 = ((fd_uh_2 - su0_2) > (0) ? (fd_uh_2 - su0_2) : (0));
                int p_hi_2 = fd_uh_2 + _max_2_2 + fd_rh_2;
                int _max_3_2 = ((fd_ul_2 - su0_2) > (0) ? (fd_ul_2 - su0_2) : (0));
                int fd_plo_2 = fd_ul_2 + _max_3_2 + fd_rl_2;
                int nsegs_2 = p_hi_2 - fd_plo_2 + 1;
                int fl_p = p_hi_2 - (nsegs_2 - 1);
                int fl_q = fl_p - su0_2;
                int fl_sp = ((fl_q >= 0) ? 1 : 0);
                int fl_u = ((fl_sp == 1) ? su0_2 + fl_q / 2 : fl_p);
                int fl_kb = ((fl_sp == 1) ? fl_q % 2 * (NUM_K_ITERS / 2) : 0);
                int fl_ke = ((fl_sp == 1) ? fl_kb + NUM_K_ITERS / 2 : NUM_K_ITERS);
                int fl_ub = fl_u * NUM_K_ITERS;
                int fl_klo = ((fl_kb) > (lo_2 - fl_ub) ? (fl_kb) : (lo_2 - fl_ub));
                int fl_khi = ((fl_ke) < (hi_2 - fl_ub) ? (fl_ke) : (hi_2 - fl_ub));
                int fo_all = sk_max_seg % 2;
                int fo_half = NUM_K_ITERS / 2;
                int fo_na = sk_ipc * full_items * CTA_GROUP;
                int fo_me = (lane_l_2 * full_items + lane_r_2) * CTA_GROUP + cta_rank;
                int fo_prev = fo_me - full_items * CTA_GROUP;
                int fo_cin = ((fl_klo > fl_kb && fo_all == 0) ? 1 : 0);
                int fo_ti = ((fl_sp == 1 && fl_ke == NUM_K_ITERS && fl_khi == fl_ke && lo_2 >= fl_ub + fo_half) ? 1 : 0);
                int fo_lprev = (lane_l_2 - 1) * sk_total / sk_ipc;
                int fo_fa = ((fo_all == 1 && fl_klo > 0 && fl_khi == NUM_K_ITERS) ? 1 : 0);
                int fo_mid = ((fo_fa == 1 && fo_lprev > fl_ub) ? 1 : 0);
                int fo_pre = fo_ti + fo_fa;
                int fo_pslot = fo_prev;
                if (fo_ti == 1 && fl_klo > fo_half) {
                    fo_pslot = fo_prev + fo_na;
                }
                if (fo_mid == 1) {
                    fo_pslot = fo_prev - full_items * CTA_GROUP;
                }
                int fill_seg = ((nsegs_2 - 2) > (0) ? (nsegs_2 - 2) : (0));
                int fill_addr = taddr + (unsigned int)(epi_warp * 32 << 16) + (epi_stage + (unsigned int)(nsegs_2 - 1)) % 2 * (unsigned int)BLOCK_N;
                int pre_addr = taddr + (unsigned int)(epi_warp * 32 << 16) + (epi_stage + (unsigned int)nsegs_2) % 2 * (unsigned int)BLOCK_N;
                #pragma unroll 1
                for (int seg_i = 0; seg_i < nsegs_2; seg_i++) {
                    if (fo_cin == 1 && seg_i == fill_seg) {
                        if (warp == 2) {
                            if (elect_sync()) {
                                int32_t _fcar_ld_0;
                                asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_fcar_ld_0) : "l"(counters + fo_prev) : "memory");
                                int fcnt_c = _fcar_ld_0;
                                #pragma unroll 1
                                for (int _fspin_c = 0; _fspin_c < 4194304; _fspin_c++) {
                                    if (fcnt_c >= 1) {
                                        break;
                                    }
                                    int32_t _fcar_ld_1;
                                    asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_fcar_ld_1) : "l"(counters + fo_prev) : "memory");
                                    fcnt_c = _fcar_ld_1;
                                }
                                if (fcnt_c < 1) {
                                    asm volatile("trap;" ::: "memory");
                                }
                                asm volatile("fence.acquire.gpu;" ::: "memory");
                            }
                        }
                        asm volatile("barrier.sync 1, 256;" ::: "memory");
                        unsigned long long fcbase = (unsigned long long)fo_prev * (unsigned long long)(BLOCK_M * BLOCK_N) + (unsigned long long)local_row * 16;
                        float fcfill[64];
                        {
                            unsigned _ldv8_4_0;
                            unsigned _ldv8_4_1;
                            unsigned _ldv8_4_2;
                            unsigned _ldv8_4_3;
                            unsigned _ldv8_4_4;
                            unsigned _ldv8_4_5;
                            unsigned _ldv8_4_6;
                            unsigned _ldv8_4_7;
                            asm volatile(
                                "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                : "=r"(_ldv8_4_0), "=r"(_ldv8_4_1), "=r"(_ldv8_4_2), "=r"(_ldv8_4_3), "=r"(_ldv8_4_4), "=r"(_ldv8_4_5), "=r"(_ldv8_4_6), "=r"(_ldv8_4_7) : "l"((const void*)(ws + (fcbase + (unsigned long long)((col_half * B_HALF_N + 0 + 0) * BLOCK_M) + 0) + (0))) : "memory");
                            fcfill[0 + 0] = __uint_as_float(_ldv8_4_0);
                            fcfill[0 + 1] = __uint_as_float(_ldv8_4_1);
                            fcfill[0 + 2] = __uint_as_float(_ldv8_4_2);
                            fcfill[0 + 3] = __uint_as_float(_ldv8_4_3);
                            fcfill[0 + 4] = __uint_as_float(_ldv8_4_4);
                            fcfill[0 + 5] = __uint_as_float(_ldv8_4_5);
                            fcfill[0 + 6] = __uint_as_float(_ldv8_4_6);
                            fcfill[0 + 7] = __uint_as_float(_ldv8_4_7);
                        }
                        {
                            unsigned _ldv8_5_0;
                            unsigned _ldv8_5_1;
                            unsigned _ldv8_5_2;
                            unsigned _ldv8_5_3;
                            unsigned _ldv8_5_4;
                            unsigned _ldv8_5_5;
                            unsigned _ldv8_5_6;
                            unsigned _ldv8_5_7;
                            asm volatile(
                                "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                : "=r"(_ldv8_5_0), "=r"(_ldv8_5_1), "=r"(_ldv8_5_2), "=r"(_ldv8_5_3), "=r"(_ldv8_5_4), "=r"(_ldv8_5_5), "=r"(_ldv8_5_6), "=r"(_ldv8_5_7) : "l"((const void*)(ws + (fcbase + (unsigned long long)((col_half * B_HALF_N + 0 + 0) * BLOCK_M) + 8) + (0))) : "memory");
                            fcfill[8 + 0] = __uint_as_float(_ldv8_5_0);
                            fcfill[8 + 1] = __uint_as_float(_ldv8_5_1);
                            fcfill[8 + 2] = __uint_as_float(_ldv8_5_2);
                            fcfill[8 + 3] = __uint_as_float(_ldv8_5_3);
                            fcfill[8 + 4] = __uint_as_float(_ldv8_5_4);
                            fcfill[8 + 5] = __uint_as_float(_ldv8_5_5);
                            fcfill[8 + 6] = __uint_as_float(_ldv8_5_6);
                            fcfill[8 + 7] = __uint_as_float(_ldv8_5_7);
                        }
                        {
                            unsigned _ldv8_6_0;
                            unsigned _ldv8_6_1;
                            unsigned _ldv8_6_2;
                            unsigned _ldv8_6_3;
                            unsigned _ldv8_6_4;
                            unsigned _ldv8_6_5;
                            unsigned _ldv8_6_6;
                            unsigned _ldv8_6_7;
                            asm volatile(
                                "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                : "=r"(_ldv8_6_0), "=r"(_ldv8_6_1), "=r"(_ldv8_6_2), "=r"(_ldv8_6_3), "=r"(_ldv8_6_4), "=r"(_ldv8_6_5), "=r"(_ldv8_6_6), "=r"(_ldv8_6_7) : "l"((const void*)(ws + (fcbase + (unsigned long long)((col_half * B_HALF_N + 0 + 16) * BLOCK_M) + 0) + (0))) : "memory");
                            fcfill[16 + 0] = __uint_as_float(_ldv8_6_0);
                            fcfill[16 + 1] = __uint_as_float(_ldv8_6_1);
                            fcfill[16 + 2] = __uint_as_float(_ldv8_6_2);
                            fcfill[16 + 3] = __uint_as_float(_ldv8_6_3);
                            fcfill[16 + 4] = __uint_as_float(_ldv8_6_4);
                            fcfill[16 + 5] = __uint_as_float(_ldv8_6_5);
                            fcfill[16 + 6] = __uint_as_float(_ldv8_6_6);
                            fcfill[16 + 7] = __uint_as_float(_ldv8_6_7);
                        }
                        {
                            unsigned _ldv8_7_0;
                            unsigned _ldv8_7_1;
                            unsigned _ldv8_7_2;
                            unsigned _ldv8_7_3;
                            unsigned _ldv8_7_4;
                            unsigned _ldv8_7_5;
                            unsigned _ldv8_7_6;
                            unsigned _ldv8_7_7;
                            asm volatile(
                                "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                : "=r"(_ldv8_7_0), "=r"(_ldv8_7_1), "=r"(_ldv8_7_2), "=r"(_ldv8_7_3), "=r"(_ldv8_7_4), "=r"(_ldv8_7_5), "=r"(_ldv8_7_6), "=r"(_ldv8_7_7) : "l"((const void*)(ws + (fcbase + (unsigned long long)((col_half * B_HALF_N + 0 + 16) * BLOCK_M) + 8) + (0))) : "memory");
                            fcfill[24 + 0] = __uint_as_float(_ldv8_7_0);
                            fcfill[24 + 1] = __uint_as_float(_ldv8_7_1);
                            fcfill[24 + 2] = __uint_as_float(_ldv8_7_2);
                            fcfill[24 + 3] = __uint_as_float(_ldv8_7_3);
                            fcfill[24 + 4] = __uint_as_float(_ldv8_7_4);
                            fcfill[24 + 5] = __uint_as_float(_ldv8_7_5);
                            fcfill[24 + 6] = __uint_as_float(_ldv8_7_6);
                            fcfill[24 + 7] = __uint_as_float(_ldv8_7_7);
                        }
                        {
                            unsigned _ldv8_8_0;
                            unsigned _ldv8_8_1;
                            unsigned _ldv8_8_2;
                            unsigned _ldv8_8_3;
                            unsigned _ldv8_8_4;
                            unsigned _ldv8_8_5;
                            unsigned _ldv8_8_6;
                            unsigned _ldv8_8_7;
                            asm volatile(
                                "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                : "=r"(_ldv8_8_0), "=r"(_ldv8_8_1), "=r"(_ldv8_8_2), "=r"(_ldv8_8_3), "=r"(_ldv8_8_4), "=r"(_ldv8_8_5), "=r"(_ldv8_8_6), "=r"(_ldv8_8_7) : "l"((const void*)(ws + (fcbase + (unsigned long long)((col_half * B_HALF_N + 0 + 32) * BLOCK_M) + 0) + (0))) : "memory");
                            fcfill[32 + 0] = __uint_as_float(_ldv8_8_0);
                            fcfill[32 + 1] = __uint_as_float(_ldv8_8_1);
                            fcfill[32 + 2] = __uint_as_float(_ldv8_8_2);
                            fcfill[32 + 3] = __uint_as_float(_ldv8_8_3);
                            fcfill[32 + 4] = __uint_as_float(_ldv8_8_4);
                            fcfill[32 + 5] = __uint_as_float(_ldv8_8_5);
                            fcfill[32 + 6] = __uint_as_float(_ldv8_8_6);
                            fcfill[32 + 7] = __uint_as_float(_ldv8_8_7);
                        }
                        {
                            unsigned _ldv8_9_0;
                            unsigned _ldv8_9_1;
                            unsigned _ldv8_9_2;
                            unsigned _ldv8_9_3;
                            unsigned _ldv8_9_4;
                            unsigned _ldv8_9_5;
                            unsigned _ldv8_9_6;
                            unsigned _ldv8_9_7;
                            asm volatile(
                                "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                : "=r"(_ldv8_9_0), "=r"(_ldv8_9_1), "=r"(_ldv8_9_2), "=r"(_ldv8_9_3), "=r"(_ldv8_9_4), "=r"(_ldv8_9_5), "=r"(_ldv8_9_6), "=r"(_ldv8_9_7) : "l"((const void*)(ws + (fcbase + (unsigned long long)((col_half * B_HALF_N + 0 + 32) * BLOCK_M) + 8) + (0))) : "memory");
                            fcfill[40 + 0] = __uint_as_float(_ldv8_9_0);
                            fcfill[40 + 1] = __uint_as_float(_ldv8_9_1);
                            fcfill[40 + 2] = __uint_as_float(_ldv8_9_2);
                            fcfill[40 + 3] = __uint_as_float(_ldv8_9_3);
                            fcfill[40 + 4] = __uint_as_float(_ldv8_9_4);
                            fcfill[40 + 5] = __uint_as_float(_ldv8_9_5);
                            fcfill[40 + 6] = __uint_as_float(_ldv8_9_6);
                            fcfill[40 + 7] = __uint_as_float(_ldv8_9_7);
                        }
                        {
                            unsigned _ldv8_10_0;
                            unsigned _ldv8_10_1;
                            unsigned _ldv8_10_2;
                            unsigned _ldv8_10_3;
                            unsigned _ldv8_10_4;
                            unsigned _ldv8_10_5;
                            unsigned _ldv8_10_6;
                            unsigned _ldv8_10_7;
                            asm volatile(
                                "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                : "=r"(_ldv8_10_0), "=r"(_ldv8_10_1), "=r"(_ldv8_10_2), "=r"(_ldv8_10_3), "=r"(_ldv8_10_4), "=r"(_ldv8_10_5), "=r"(_ldv8_10_6), "=r"(_ldv8_10_7) : "l"((const void*)(ws + (fcbase + (unsigned long long)((col_half * B_HALF_N + 0 + 48) * BLOCK_M) + 0) + (0))) : "memory");
                            fcfill[48 + 0] = __uint_as_float(_ldv8_10_0);
                            fcfill[48 + 1] = __uint_as_float(_ldv8_10_1);
                            fcfill[48 + 2] = __uint_as_float(_ldv8_10_2);
                            fcfill[48 + 3] = __uint_as_float(_ldv8_10_3);
                            fcfill[48 + 4] = __uint_as_float(_ldv8_10_4);
                            fcfill[48 + 5] = __uint_as_float(_ldv8_10_5);
                            fcfill[48 + 6] = __uint_as_float(_ldv8_10_6);
                            fcfill[48 + 7] = __uint_as_float(_ldv8_10_7);
                        }
                        {
                            unsigned _ldv8_11_0;
                            unsigned _ldv8_11_1;
                            unsigned _ldv8_11_2;
                            unsigned _ldv8_11_3;
                            unsigned _ldv8_11_4;
                            unsigned _ldv8_11_5;
                            unsigned _ldv8_11_6;
                            unsigned _ldv8_11_7;
                            asm volatile(
                                "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                : "=r"(_ldv8_11_0), "=r"(_ldv8_11_1), "=r"(_ldv8_11_2), "=r"(_ldv8_11_3), "=r"(_ldv8_11_4), "=r"(_ldv8_11_5), "=r"(_ldv8_11_6), "=r"(_ldv8_11_7) : "l"((const void*)(ws + (fcbase + (unsigned long long)((col_half * B_HALF_N + 0 + 48) * BLOCK_M) + 8) + (0))) : "memory");
                            fcfill[56 + 0] = __uint_as_float(_ldv8_11_0);
                            fcfill[56 + 1] = __uint_as_float(_ldv8_11_1);
                            fcfill[56 + 2] = __uint_as_float(_ldv8_11_2);
                            fcfill[56 + 3] = __uint_as_float(_ldv8_11_3);
                            fcfill[56 + 4] = __uint_as_float(_ldv8_11_4);
                            fcfill[56 + 5] = __uint_as_float(_ldv8_11_5);
                            fcfill[56 + 6] = __uint_as_float(_ldv8_11_6);
                            fcfill[56 + 7] = __uint_as_float(_ldv8_11_7);
                        }
                        tmem_st_x16_f32(fill_addr + col_half * B_HALF_N + 0, fcfill);
                        tmem_st_x16_f32(fill_addr + col_half * B_HALF_N + 16, (fcfill + 16));
                        tmem_st_x16_f32(fill_addr + col_half * B_HALF_N + 32, (fcfill + 32));
                        tmem_st_x16_f32(fill_addr + col_half * B_HALF_N + 48, (fcfill + 48));
                        {
                            unsigned _ldv8_12_0;
                            unsigned _ldv8_12_1;
                            unsigned _ldv8_12_2;
                            unsigned _ldv8_12_3;
                            unsigned _ldv8_12_4;
                            unsigned _ldv8_12_5;
                            unsigned _ldv8_12_6;
                            unsigned _ldv8_12_7;
                            asm volatile(
                                "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                : "=r"(_ldv8_12_0), "=r"(_ldv8_12_1), "=r"(_ldv8_12_2), "=r"(_ldv8_12_3), "=r"(_ldv8_12_4), "=r"(_ldv8_12_5), "=r"(_ldv8_12_6), "=r"(_ldv8_12_7) : "l"((const void*)(ws + (fcbase + (unsigned long long)((col_half * B_HALF_N + 64 + 0) * BLOCK_M) + 0) + (0))) : "memory");
                            fcfill[0 + 0] = __uint_as_float(_ldv8_12_0);
                            fcfill[0 + 1] = __uint_as_float(_ldv8_12_1);
                            fcfill[0 + 2] = __uint_as_float(_ldv8_12_2);
                            fcfill[0 + 3] = __uint_as_float(_ldv8_12_3);
                            fcfill[0 + 4] = __uint_as_float(_ldv8_12_4);
                            fcfill[0 + 5] = __uint_as_float(_ldv8_12_5);
                            fcfill[0 + 6] = __uint_as_float(_ldv8_12_6);
                            fcfill[0 + 7] = __uint_as_float(_ldv8_12_7);
                        }
                        {
                            unsigned _ldv8_13_0;
                            unsigned _ldv8_13_1;
                            unsigned _ldv8_13_2;
                            unsigned _ldv8_13_3;
                            unsigned _ldv8_13_4;
                            unsigned _ldv8_13_5;
                            unsigned _ldv8_13_6;
                            unsigned _ldv8_13_7;
                            asm volatile(
                                "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                : "=r"(_ldv8_13_0), "=r"(_ldv8_13_1), "=r"(_ldv8_13_2), "=r"(_ldv8_13_3), "=r"(_ldv8_13_4), "=r"(_ldv8_13_5), "=r"(_ldv8_13_6), "=r"(_ldv8_13_7) : "l"((const void*)(ws + (fcbase + (unsigned long long)((col_half * B_HALF_N + 64 + 0) * BLOCK_M) + 8) + (0))) : "memory");
                            fcfill[8 + 0] = __uint_as_float(_ldv8_13_0);
                            fcfill[8 + 1] = __uint_as_float(_ldv8_13_1);
                            fcfill[8 + 2] = __uint_as_float(_ldv8_13_2);
                            fcfill[8 + 3] = __uint_as_float(_ldv8_13_3);
                            fcfill[8 + 4] = __uint_as_float(_ldv8_13_4);
                            fcfill[8 + 5] = __uint_as_float(_ldv8_13_5);
                            fcfill[8 + 6] = __uint_as_float(_ldv8_13_6);
                            fcfill[8 + 7] = __uint_as_float(_ldv8_13_7);
                        }
                        {
                            unsigned _ldv8_14_0;
                            unsigned _ldv8_14_1;
                            unsigned _ldv8_14_2;
                            unsigned _ldv8_14_3;
                            unsigned _ldv8_14_4;
                            unsigned _ldv8_14_5;
                            unsigned _ldv8_14_6;
                            unsigned _ldv8_14_7;
                            asm volatile(
                                "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                : "=r"(_ldv8_14_0), "=r"(_ldv8_14_1), "=r"(_ldv8_14_2), "=r"(_ldv8_14_3), "=r"(_ldv8_14_4), "=r"(_ldv8_14_5), "=r"(_ldv8_14_6), "=r"(_ldv8_14_7) : "l"((const void*)(ws + (fcbase + (unsigned long long)((col_half * B_HALF_N + 64 + 16) * BLOCK_M) + 0) + (0))) : "memory");
                            fcfill[16 + 0] = __uint_as_float(_ldv8_14_0);
                            fcfill[16 + 1] = __uint_as_float(_ldv8_14_1);
                            fcfill[16 + 2] = __uint_as_float(_ldv8_14_2);
                            fcfill[16 + 3] = __uint_as_float(_ldv8_14_3);
                            fcfill[16 + 4] = __uint_as_float(_ldv8_14_4);
                            fcfill[16 + 5] = __uint_as_float(_ldv8_14_5);
                            fcfill[16 + 6] = __uint_as_float(_ldv8_14_6);
                            fcfill[16 + 7] = __uint_as_float(_ldv8_14_7);
                        }
                        {
                            unsigned _ldv8_15_0;
                            unsigned _ldv8_15_1;
                            unsigned _ldv8_15_2;
                            unsigned _ldv8_15_3;
                            unsigned _ldv8_15_4;
                            unsigned _ldv8_15_5;
                            unsigned _ldv8_15_6;
                            unsigned _ldv8_15_7;
                            asm volatile(
                                "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                : "=r"(_ldv8_15_0), "=r"(_ldv8_15_1), "=r"(_ldv8_15_2), "=r"(_ldv8_15_3), "=r"(_ldv8_15_4), "=r"(_ldv8_15_5), "=r"(_ldv8_15_6), "=r"(_ldv8_15_7) : "l"((const void*)(ws + (fcbase + (unsigned long long)((col_half * B_HALF_N + 64 + 16) * BLOCK_M) + 8) + (0))) : "memory");
                            fcfill[24 + 0] = __uint_as_float(_ldv8_15_0);
                            fcfill[24 + 1] = __uint_as_float(_ldv8_15_1);
                            fcfill[24 + 2] = __uint_as_float(_ldv8_15_2);
                            fcfill[24 + 3] = __uint_as_float(_ldv8_15_3);
                            fcfill[24 + 4] = __uint_as_float(_ldv8_15_4);
                            fcfill[24 + 5] = __uint_as_float(_ldv8_15_5);
                            fcfill[24 + 6] = __uint_as_float(_ldv8_15_6);
                            fcfill[24 + 7] = __uint_as_float(_ldv8_15_7);
                        }
                        {
                            unsigned _ldv8_16_0;
                            unsigned _ldv8_16_1;
                            unsigned _ldv8_16_2;
                            unsigned _ldv8_16_3;
                            unsigned _ldv8_16_4;
                            unsigned _ldv8_16_5;
                            unsigned _ldv8_16_6;
                            unsigned _ldv8_16_7;
                            asm volatile(
                                "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                : "=r"(_ldv8_16_0), "=r"(_ldv8_16_1), "=r"(_ldv8_16_2), "=r"(_ldv8_16_3), "=r"(_ldv8_16_4), "=r"(_ldv8_16_5), "=r"(_ldv8_16_6), "=r"(_ldv8_16_7) : "l"((const void*)(ws + (fcbase + (unsigned long long)((col_half * B_HALF_N + 64 + 32) * BLOCK_M) + 0) + (0))) : "memory");
                            fcfill[32 + 0] = __uint_as_float(_ldv8_16_0);
                            fcfill[32 + 1] = __uint_as_float(_ldv8_16_1);
                            fcfill[32 + 2] = __uint_as_float(_ldv8_16_2);
                            fcfill[32 + 3] = __uint_as_float(_ldv8_16_3);
                            fcfill[32 + 4] = __uint_as_float(_ldv8_16_4);
                            fcfill[32 + 5] = __uint_as_float(_ldv8_16_5);
                            fcfill[32 + 6] = __uint_as_float(_ldv8_16_6);
                            fcfill[32 + 7] = __uint_as_float(_ldv8_16_7);
                        }
                        {
                            unsigned _ldv8_17_0;
                            unsigned _ldv8_17_1;
                            unsigned _ldv8_17_2;
                            unsigned _ldv8_17_3;
                            unsigned _ldv8_17_4;
                            unsigned _ldv8_17_5;
                            unsigned _ldv8_17_6;
                            unsigned _ldv8_17_7;
                            asm volatile(
                                "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                : "=r"(_ldv8_17_0), "=r"(_ldv8_17_1), "=r"(_ldv8_17_2), "=r"(_ldv8_17_3), "=r"(_ldv8_17_4), "=r"(_ldv8_17_5), "=r"(_ldv8_17_6), "=r"(_ldv8_17_7) : "l"((const void*)(ws + (fcbase + (unsigned long long)((col_half * B_HALF_N + 64 + 32) * BLOCK_M) + 8) + (0))) : "memory");
                            fcfill[40 + 0] = __uint_as_float(_ldv8_17_0);
                            fcfill[40 + 1] = __uint_as_float(_ldv8_17_1);
                            fcfill[40 + 2] = __uint_as_float(_ldv8_17_2);
                            fcfill[40 + 3] = __uint_as_float(_ldv8_17_3);
                            fcfill[40 + 4] = __uint_as_float(_ldv8_17_4);
                            fcfill[40 + 5] = __uint_as_float(_ldv8_17_5);
                            fcfill[40 + 6] = __uint_as_float(_ldv8_17_6);
                            fcfill[40 + 7] = __uint_as_float(_ldv8_17_7);
                        }
                        {
                            unsigned _ldv8_18_0;
                            unsigned _ldv8_18_1;
                            unsigned _ldv8_18_2;
                            unsigned _ldv8_18_3;
                            unsigned _ldv8_18_4;
                            unsigned _ldv8_18_5;
                            unsigned _ldv8_18_6;
                            unsigned _ldv8_18_7;
                            asm volatile(
                                "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                : "=r"(_ldv8_18_0), "=r"(_ldv8_18_1), "=r"(_ldv8_18_2), "=r"(_ldv8_18_3), "=r"(_ldv8_18_4), "=r"(_ldv8_18_5), "=r"(_ldv8_18_6), "=r"(_ldv8_18_7) : "l"((const void*)(ws + (fcbase + (unsigned long long)((col_half * B_HALF_N + 64 + 48) * BLOCK_M) + 0) + (0))) : "memory");
                            fcfill[48 + 0] = __uint_as_float(_ldv8_18_0);
                            fcfill[48 + 1] = __uint_as_float(_ldv8_18_1);
                            fcfill[48 + 2] = __uint_as_float(_ldv8_18_2);
                            fcfill[48 + 3] = __uint_as_float(_ldv8_18_3);
                            fcfill[48 + 4] = __uint_as_float(_ldv8_18_4);
                            fcfill[48 + 5] = __uint_as_float(_ldv8_18_5);
                            fcfill[48 + 6] = __uint_as_float(_ldv8_18_6);
                            fcfill[48 + 7] = __uint_as_float(_ldv8_18_7);
                        }
                        {
                            unsigned _ldv8_19_0;
                            unsigned _ldv8_19_1;
                            unsigned _ldv8_19_2;
                            unsigned _ldv8_19_3;
                            unsigned _ldv8_19_4;
                            unsigned _ldv8_19_5;
                            unsigned _ldv8_19_6;
                            unsigned _ldv8_19_7;
                            asm volatile(
                                "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                : "=r"(_ldv8_19_0), "=r"(_ldv8_19_1), "=r"(_ldv8_19_2), "=r"(_ldv8_19_3), "=r"(_ldv8_19_4), "=r"(_ldv8_19_5), "=r"(_ldv8_19_6), "=r"(_ldv8_19_7) : "l"((const void*)(ws + (fcbase + (unsigned long long)((col_half * B_HALF_N + 64 + 48) * BLOCK_M) + 8) + (0))) : "memory");
                            fcfill[56 + 0] = __uint_as_float(_ldv8_19_0);
                            fcfill[56 + 1] = __uint_as_float(_ldv8_19_1);
                            fcfill[56 + 2] = __uint_as_float(_ldv8_19_2);
                            fcfill[56 + 3] = __uint_as_float(_ldv8_19_3);
                            fcfill[56 + 4] = __uint_as_float(_ldv8_19_4);
                            fcfill[56 + 5] = __uint_as_float(_ldv8_19_5);
                            fcfill[56 + 6] = __uint_as_float(_ldv8_19_6);
                            fcfill[56 + 7] = __uint_as_float(_ldv8_19_7);
                        }
                        tmem_st_x16_f32(fill_addr + col_half * B_HALF_N + 64, fcfill);
                        tmem_st_x16_f32(fill_addr + col_half * B_HALF_N + 80, (fcfill + 16));
                        tmem_st_x16_f32(fill_addr + col_half * B_HALF_N + 96, (fcfill + 32));
                        tmem_st_x16_f32(fill_addr + col_half * B_HALF_N + 112, (fcfill + 48));
                        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                        asm volatile("barrier.sync 1, 256;" ::: "memory");
                        if (warp == 2) {
                            if (elect_sync()) {
                                counters[fo_prev] = 0;
                            }
                        }
                        if (elect_sync()) {
                            asm volatile("tcgen05.fence::before_thread_sync;");
                            asm volatile(
                                "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                                :: "r"((carry_full_addr) & 0xFEFFFFFF) : "memory");
                        }
                    }
                    if (fo_pre == 1 && seg_i == nsegs_2 - 1) {
                        if (warp == 2) {
                            if (elect_sync()) {
                                int32_t _fpre_ld_0;
                                asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_fpre_ld_0) : "l"(counters + fo_pslot) : "memory");
                                int fcnt_p = _fpre_ld_0;
                                #pragma unroll 1
                                for (int _fspin_p = 0; _fspin_p < 4194304; _fspin_p++) {
                                    if (fcnt_p >= 1) {
                                        break;
                                    }
                                    int32_t _fpre_ld_1;
                                    asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_fpre_ld_1) : "l"(counters + fo_pslot) : "memory");
                                    fcnt_p = _fpre_ld_1;
                                }
                                if (fcnt_p < 1) {
                                    asm volatile("trap;" ::: "memory");
                                }
                                asm volatile("fence.acquire.gpu;" ::: "memory");
                            }
                        }
                        asm volatile("barrier.sync 1, 256;" ::: "memory");
                        unsigned long long fpbase = (unsigned long long)fo_pslot * (unsigned long long)(BLOCK_M * BLOCK_N) + (unsigned long long)local_row * 16;
                        float fpfill[64];
                        {
                            unsigned _ldv8_20_0;
                            unsigned _ldv8_20_1;
                            unsigned _ldv8_20_2;
                            unsigned _ldv8_20_3;
                            unsigned _ldv8_20_4;
                            unsigned _ldv8_20_5;
                            unsigned _ldv8_20_6;
                            unsigned _ldv8_20_7;
                            asm volatile(
                                "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                : "=r"(_ldv8_20_0), "=r"(_ldv8_20_1), "=r"(_ldv8_20_2), "=r"(_ldv8_20_3), "=r"(_ldv8_20_4), "=r"(_ldv8_20_5), "=r"(_ldv8_20_6), "=r"(_ldv8_20_7) : "l"((const void*)(ws + (fpbase + (unsigned long long)((col_half * B_HALF_N + 0 + 0) * BLOCK_M) + 0) + (0))) : "memory");
                            fpfill[0 + 0] = __uint_as_float(_ldv8_20_0);
                            fpfill[0 + 1] = __uint_as_float(_ldv8_20_1);
                            fpfill[0 + 2] = __uint_as_float(_ldv8_20_2);
                            fpfill[0 + 3] = __uint_as_float(_ldv8_20_3);
                            fpfill[0 + 4] = __uint_as_float(_ldv8_20_4);
                            fpfill[0 + 5] = __uint_as_float(_ldv8_20_5);
                            fpfill[0 + 6] = __uint_as_float(_ldv8_20_6);
                            fpfill[0 + 7] = __uint_as_float(_ldv8_20_7);
                        }
                        {
                            unsigned _ldv8_21_0;
                            unsigned _ldv8_21_1;
                            unsigned _ldv8_21_2;
                            unsigned _ldv8_21_3;
                            unsigned _ldv8_21_4;
                            unsigned _ldv8_21_5;
                            unsigned _ldv8_21_6;
                            unsigned _ldv8_21_7;
                            asm volatile(
                                "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                : "=r"(_ldv8_21_0), "=r"(_ldv8_21_1), "=r"(_ldv8_21_2), "=r"(_ldv8_21_3), "=r"(_ldv8_21_4), "=r"(_ldv8_21_5), "=r"(_ldv8_21_6), "=r"(_ldv8_21_7) : "l"((const void*)(ws + (fpbase + (unsigned long long)((col_half * B_HALF_N + 0 + 0) * BLOCK_M) + 8) + (0))) : "memory");
                            fpfill[8 + 0] = __uint_as_float(_ldv8_21_0);
                            fpfill[8 + 1] = __uint_as_float(_ldv8_21_1);
                            fpfill[8 + 2] = __uint_as_float(_ldv8_21_2);
                            fpfill[8 + 3] = __uint_as_float(_ldv8_21_3);
                            fpfill[8 + 4] = __uint_as_float(_ldv8_21_4);
                            fpfill[8 + 5] = __uint_as_float(_ldv8_21_5);
                            fpfill[8 + 6] = __uint_as_float(_ldv8_21_6);
                            fpfill[8 + 7] = __uint_as_float(_ldv8_21_7);
                        }
                        {
                            unsigned _ldv8_22_0;
                            unsigned _ldv8_22_1;
                            unsigned _ldv8_22_2;
                            unsigned _ldv8_22_3;
                            unsigned _ldv8_22_4;
                            unsigned _ldv8_22_5;
                            unsigned _ldv8_22_6;
                            unsigned _ldv8_22_7;
                            asm volatile(
                                "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                : "=r"(_ldv8_22_0), "=r"(_ldv8_22_1), "=r"(_ldv8_22_2), "=r"(_ldv8_22_3), "=r"(_ldv8_22_4), "=r"(_ldv8_22_5), "=r"(_ldv8_22_6), "=r"(_ldv8_22_7) : "l"((const void*)(ws + (fpbase + (unsigned long long)((col_half * B_HALF_N + 0 + 16) * BLOCK_M) + 0) + (0))) : "memory");
                            fpfill[16 + 0] = __uint_as_float(_ldv8_22_0);
                            fpfill[16 + 1] = __uint_as_float(_ldv8_22_1);
                            fpfill[16 + 2] = __uint_as_float(_ldv8_22_2);
                            fpfill[16 + 3] = __uint_as_float(_ldv8_22_3);
                            fpfill[16 + 4] = __uint_as_float(_ldv8_22_4);
                            fpfill[16 + 5] = __uint_as_float(_ldv8_22_5);
                            fpfill[16 + 6] = __uint_as_float(_ldv8_22_6);
                            fpfill[16 + 7] = __uint_as_float(_ldv8_22_7);
                        }
                        {
                            unsigned _ldv8_23_0;
                            unsigned _ldv8_23_1;
                            unsigned _ldv8_23_2;
                            unsigned _ldv8_23_3;
                            unsigned _ldv8_23_4;
                            unsigned _ldv8_23_5;
                            unsigned _ldv8_23_6;
                            unsigned _ldv8_23_7;
                            asm volatile(
                                "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                : "=r"(_ldv8_23_0), "=r"(_ldv8_23_1), "=r"(_ldv8_23_2), "=r"(_ldv8_23_3), "=r"(_ldv8_23_4), "=r"(_ldv8_23_5), "=r"(_ldv8_23_6), "=r"(_ldv8_23_7) : "l"((const void*)(ws + (fpbase + (unsigned long long)((col_half * B_HALF_N + 0 + 16) * BLOCK_M) + 8) + (0))) : "memory");
                            fpfill[24 + 0] = __uint_as_float(_ldv8_23_0);
                            fpfill[24 + 1] = __uint_as_float(_ldv8_23_1);
                            fpfill[24 + 2] = __uint_as_float(_ldv8_23_2);
                            fpfill[24 + 3] = __uint_as_float(_ldv8_23_3);
                            fpfill[24 + 4] = __uint_as_float(_ldv8_23_4);
                            fpfill[24 + 5] = __uint_as_float(_ldv8_23_5);
                            fpfill[24 + 6] = __uint_as_float(_ldv8_23_6);
                            fpfill[24 + 7] = __uint_as_float(_ldv8_23_7);
                        }
                        {
                            unsigned _ldv8_24_0;
                            unsigned _ldv8_24_1;
                            unsigned _ldv8_24_2;
                            unsigned _ldv8_24_3;
                            unsigned _ldv8_24_4;
                            unsigned _ldv8_24_5;
                            unsigned _ldv8_24_6;
                            unsigned _ldv8_24_7;
                            asm volatile(
                                "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                : "=r"(_ldv8_24_0), "=r"(_ldv8_24_1), "=r"(_ldv8_24_2), "=r"(_ldv8_24_3), "=r"(_ldv8_24_4), "=r"(_ldv8_24_5), "=r"(_ldv8_24_6), "=r"(_ldv8_24_7) : "l"((const void*)(ws + (fpbase + (unsigned long long)((col_half * B_HALF_N + 0 + 32) * BLOCK_M) + 0) + (0))) : "memory");
                            fpfill[32 + 0] = __uint_as_float(_ldv8_24_0);
                            fpfill[32 + 1] = __uint_as_float(_ldv8_24_1);
                            fpfill[32 + 2] = __uint_as_float(_ldv8_24_2);
                            fpfill[32 + 3] = __uint_as_float(_ldv8_24_3);
                            fpfill[32 + 4] = __uint_as_float(_ldv8_24_4);
                            fpfill[32 + 5] = __uint_as_float(_ldv8_24_5);
                            fpfill[32 + 6] = __uint_as_float(_ldv8_24_6);
                            fpfill[32 + 7] = __uint_as_float(_ldv8_24_7);
                        }
                        {
                            unsigned _ldv8_25_0;
                            unsigned _ldv8_25_1;
                            unsigned _ldv8_25_2;
                            unsigned _ldv8_25_3;
                            unsigned _ldv8_25_4;
                            unsigned _ldv8_25_5;
                            unsigned _ldv8_25_6;
                            unsigned _ldv8_25_7;
                            asm volatile(
                                "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                : "=r"(_ldv8_25_0), "=r"(_ldv8_25_1), "=r"(_ldv8_25_2), "=r"(_ldv8_25_3), "=r"(_ldv8_25_4), "=r"(_ldv8_25_5), "=r"(_ldv8_25_6), "=r"(_ldv8_25_7) : "l"((const void*)(ws + (fpbase + (unsigned long long)((col_half * B_HALF_N + 0 + 32) * BLOCK_M) + 8) + (0))) : "memory");
                            fpfill[40 + 0] = __uint_as_float(_ldv8_25_0);
                            fpfill[40 + 1] = __uint_as_float(_ldv8_25_1);
                            fpfill[40 + 2] = __uint_as_float(_ldv8_25_2);
                            fpfill[40 + 3] = __uint_as_float(_ldv8_25_3);
                            fpfill[40 + 4] = __uint_as_float(_ldv8_25_4);
                            fpfill[40 + 5] = __uint_as_float(_ldv8_25_5);
                            fpfill[40 + 6] = __uint_as_float(_ldv8_25_6);
                            fpfill[40 + 7] = __uint_as_float(_ldv8_25_7);
                        }
                        {
                            unsigned _ldv8_26_0;
                            unsigned _ldv8_26_1;
                            unsigned _ldv8_26_2;
                            unsigned _ldv8_26_3;
                            unsigned _ldv8_26_4;
                            unsigned _ldv8_26_5;
                            unsigned _ldv8_26_6;
                            unsigned _ldv8_26_7;
                            asm volatile(
                                "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                : "=r"(_ldv8_26_0), "=r"(_ldv8_26_1), "=r"(_ldv8_26_2), "=r"(_ldv8_26_3), "=r"(_ldv8_26_4), "=r"(_ldv8_26_5), "=r"(_ldv8_26_6), "=r"(_ldv8_26_7) : "l"((const void*)(ws + (fpbase + (unsigned long long)((col_half * B_HALF_N + 0 + 48) * BLOCK_M) + 0) + (0))) : "memory");
                            fpfill[48 + 0] = __uint_as_float(_ldv8_26_0);
                            fpfill[48 + 1] = __uint_as_float(_ldv8_26_1);
                            fpfill[48 + 2] = __uint_as_float(_ldv8_26_2);
                            fpfill[48 + 3] = __uint_as_float(_ldv8_26_3);
                            fpfill[48 + 4] = __uint_as_float(_ldv8_26_4);
                            fpfill[48 + 5] = __uint_as_float(_ldv8_26_5);
                            fpfill[48 + 6] = __uint_as_float(_ldv8_26_6);
                            fpfill[48 + 7] = __uint_as_float(_ldv8_26_7);
                        }
                        {
                            unsigned _ldv8_27_0;
                            unsigned _ldv8_27_1;
                            unsigned _ldv8_27_2;
                            unsigned _ldv8_27_3;
                            unsigned _ldv8_27_4;
                            unsigned _ldv8_27_5;
                            unsigned _ldv8_27_6;
                            unsigned _ldv8_27_7;
                            asm volatile(
                                "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                : "=r"(_ldv8_27_0), "=r"(_ldv8_27_1), "=r"(_ldv8_27_2), "=r"(_ldv8_27_3), "=r"(_ldv8_27_4), "=r"(_ldv8_27_5), "=r"(_ldv8_27_6), "=r"(_ldv8_27_7) : "l"((const void*)(ws + (fpbase + (unsigned long long)((col_half * B_HALF_N + 0 + 48) * BLOCK_M) + 8) + (0))) : "memory");
                            fpfill[56 + 0] = __uint_as_float(_ldv8_27_0);
                            fpfill[56 + 1] = __uint_as_float(_ldv8_27_1);
                            fpfill[56 + 2] = __uint_as_float(_ldv8_27_2);
                            fpfill[56 + 3] = __uint_as_float(_ldv8_27_3);
                            fpfill[56 + 4] = __uint_as_float(_ldv8_27_4);
                            fpfill[56 + 5] = __uint_as_float(_ldv8_27_5);
                            fpfill[56 + 6] = __uint_as_float(_ldv8_27_6);
                            fpfill[56 + 7] = __uint_as_float(_ldv8_27_7);
                        }
                        tmem_st_x16_f32(pre_addr + col_half * B_HALF_N + 0, fpfill);
                        tmem_st_x16_f32(pre_addr + col_half * B_HALF_N + 16, (fpfill + 16));
                        tmem_st_x16_f32(pre_addr + col_half * B_HALF_N + 32, (fpfill + 32));
                        tmem_st_x16_f32(pre_addr + col_half * B_HALF_N + 48, (fpfill + 48));
                        {
                            unsigned _ldv8_28_0;
                            unsigned _ldv8_28_1;
                            unsigned _ldv8_28_2;
                            unsigned _ldv8_28_3;
                            unsigned _ldv8_28_4;
                            unsigned _ldv8_28_5;
                            unsigned _ldv8_28_6;
                            unsigned _ldv8_28_7;
                            asm volatile(
                                "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                : "=r"(_ldv8_28_0), "=r"(_ldv8_28_1), "=r"(_ldv8_28_2), "=r"(_ldv8_28_3), "=r"(_ldv8_28_4), "=r"(_ldv8_28_5), "=r"(_ldv8_28_6), "=r"(_ldv8_28_7) : "l"((const void*)(ws + (fpbase + (unsigned long long)((col_half * B_HALF_N + 64 + 0) * BLOCK_M) + 0) + (0))) : "memory");
                            fpfill[0 + 0] = __uint_as_float(_ldv8_28_0);
                            fpfill[0 + 1] = __uint_as_float(_ldv8_28_1);
                            fpfill[0 + 2] = __uint_as_float(_ldv8_28_2);
                            fpfill[0 + 3] = __uint_as_float(_ldv8_28_3);
                            fpfill[0 + 4] = __uint_as_float(_ldv8_28_4);
                            fpfill[0 + 5] = __uint_as_float(_ldv8_28_5);
                            fpfill[0 + 6] = __uint_as_float(_ldv8_28_6);
                            fpfill[0 + 7] = __uint_as_float(_ldv8_28_7);
                        }
                        {
                            unsigned _ldv8_29_0;
                            unsigned _ldv8_29_1;
                            unsigned _ldv8_29_2;
                            unsigned _ldv8_29_3;
                            unsigned _ldv8_29_4;
                            unsigned _ldv8_29_5;
                            unsigned _ldv8_29_6;
                            unsigned _ldv8_29_7;
                            asm volatile(
                                "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                : "=r"(_ldv8_29_0), "=r"(_ldv8_29_1), "=r"(_ldv8_29_2), "=r"(_ldv8_29_3), "=r"(_ldv8_29_4), "=r"(_ldv8_29_5), "=r"(_ldv8_29_6), "=r"(_ldv8_29_7) : "l"((const void*)(ws + (fpbase + (unsigned long long)((col_half * B_HALF_N + 64 + 0) * BLOCK_M) + 8) + (0))) : "memory");
                            fpfill[8 + 0] = __uint_as_float(_ldv8_29_0);
                            fpfill[8 + 1] = __uint_as_float(_ldv8_29_1);
                            fpfill[8 + 2] = __uint_as_float(_ldv8_29_2);
                            fpfill[8 + 3] = __uint_as_float(_ldv8_29_3);
                            fpfill[8 + 4] = __uint_as_float(_ldv8_29_4);
                            fpfill[8 + 5] = __uint_as_float(_ldv8_29_5);
                            fpfill[8 + 6] = __uint_as_float(_ldv8_29_6);
                            fpfill[8 + 7] = __uint_as_float(_ldv8_29_7);
                        }
                        {
                            unsigned _ldv8_30_0;
                            unsigned _ldv8_30_1;
                            unsigned _ldv8_30_2;
                            unsigned _ldv8_30_3;
                            unsigned _ldv8_30_4;
                            unsigned _ldv8_30_5;
                            unsigned _ldv8_30_6;
                            unsigned _ldv8_30_7;
                            asm volatile(
                                "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                : "=r"(_ldv8_30_0), "=r"(_ldv8_30_1), "=r"(_ldv8_30_2), "=r"(_ldv8_30_3), "=r"(_ldv8_30_4), "=r"(_ldv8_30_5), "=r"(_ldv8_30_6), "=r"(_ldv8_30_7) : "l"((const void*)(ws + (fpbase + (unsigned long long)((col_half * B_HALF_N + 64 + 16) * BLOCK_M) + 0) + (0))) : "memory");
                            fpfill[16 + 0] = __uint_as_float(_ldv8_30_0);
                            fpfill[16 + 1] = __uint_as_float(_ldv8_30_1);
                            fpfill[16 + 2] = __uint_as_float(_ldv8_30_2);
                            fpfill[16 + 3] = __uint_as_float(_ldv8_30_3);
                            fpfill[16 + 4] = __uint_as_float(_ldv8_30_4);
                            fpfill[16 + 5] = __uint_as_float(_ldv8_30_5);
                            fpfill[16 + 6] = __uint_as_float(_ldv8_30_6);
                            fpfill[16 + 7] = __uint_as_float(_ldv8_30_7);
                        }
                        {
                            unsigned _ldv8_31_0;
                            unsigned _ldv8_31_1;
                            unsigned _ldv8_31_2;
                            unsigned _ldv8_31_3;
                            unsigned _ldv8_31_4;
                            unsigned _ldv8_31_5;
                            unsigned _ldv8_31_6;
                            unsigned _ldv8_31_7;
                            asm volatile(
                                "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                : "=r"(_ldv8_31_0), "=r"(_ldv8_31_1), "=r"(_ldv8_31_2), "=r"(_ldv8_31_3), "=r"(_ldv8_31_4), "=r"(_ldv8_31_5), "=r"(_ldv8_31_6), "=r"(_ldv8_31_7) : "l"((const void*)(ws + (fpbase + (unsigned long long)((col_half * B_HALF_N + 64 + 16) * BLOCK_M) + 8) + (0))) : "memory");
                            fpfill[24 + 0] = __uint_as_float(_ldv8_31_0);
                            fpfill[24 + 1] = __uint_as_float(_ldv8_31_1);
                            fpfill[24 + 2] = __uint_as_float(_ldv8_31_2);
                            fpfill[24 + 3] = __uint_as_float(_ldv8_31_3);
                            fpfill[24 + 4] = __uint_as_float(_ldv8_31_4);
                            fpfill[24 + 5] = __uint_as_float(_ldv8_31_5);
                            fpfill[24 + 6] = __uint_as_float(_ldv8_31_6);
                            fpfill[24 + 7] = __uint_as_float(_ldv8_31_7);
                        }
                        {
                            unsigned _ldv8_32_0;
                            unsigned _ldv8_32_1;
                            unsigned _ldv8_32_2;
                            unsigned _ldv8_32_3;
                            unsigned _ldv8_32_4;
                            unsigned _ldv8_32_5;
                            unsigned _ldv8_32_6;
                            unsigned _ldv8_32_7;
                            asm volatile(
                                "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                : "=r"(_ldv8_32_0), "=r"(_ldv8_32_1), "=r"(_ldv8_32_2), "=r"(_ldv8_32_3), "=r"(_ldv8_32_4), "=r"(_ldv8_32_5), "=r"(_ldv8_32_6), "=r"(_ldv8_32_7) : "l"((const void*)(ws + (fpbase + (unsigned long long)((col_half * B_HALF_N + 64 + 32) * BLOCK_M) + 0) + (0))) : "memory");
                            fpfill[32 + 0] = __uint_as_float(_ldv8_32_0);
                            fpfill[32 + 1] = __uint_as_float(_ldv8_32_1);
                            fpfill[32 + 2] = __uint_as_float(_ldv8_32_2);
                            fpfill[32 + 3] = __uint_as_float(_ldv8_32_3);
                            fpfill[32 + 4] = __uint_as_float(_ldv8_32_4);
                            fpfill[32 + 5] = __uint_as_float(_ldv8_32_5);
                            fpfill[32 + 6] = __uint_as_float(_ldv8_32_6);
                            fpfill[32 + 7] = __uint_as_float(_ldv8_32_7);
                        }
                        {
                            unsigned _ldv8_33_0;
                            unsigned _ldv8_33_1;
                            unsigned _ldv8_33_2;
                            unsigned _ldv8_33_3;
                            unsigned _ldv8_33_4;
                            unsigned _ldv8_33_5;
                            unsigned _ldv8_33_6;
                            unsigned _ldv8_33_7;
                            asm volatile(
                                "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                : "=r"(_ldv8_33_0), "=r"(_ldv8_33_1), "=r"(_ldv8_33_2), "=r"(_ldv8_33_3), "=r"(_ldv8_33_4), "=r"(_ldv8_33_5), "=r"(_ldv8_33_6), "=r"(_ldv8_33_7) : "l"((const void*)(ws + (fpbase + (unsigned long long)((col_half * B_HALF_N + 64 + 32) * BLOCK_M) + 8) + (0))) : "memory");
                            fpfill[40 + 0] = __uint_as_float(_ldv8_33_0);
                            fpfill[40 + 1] = __uint_as_float(_ldv8_33_1);
                            fpfill[40 + 2] = __uint_as_float(_ldv8_33_2);
                            fpfill[40 + 3] = __uint_as_float(_ldv8_33_3);
                            fpfill[40 + 4] = __uint_as_float(_ldv8_33_4);
                            fpfill[40 + 5] = __uint_as_float(_ldv8_33_5);
                            fpfill[40 + 6] = __uint_as_float(_ldv8_33_6);
                            fpfill[40 + 7] = __uint_as_float(_ldv8_33_7);
                        }
                        {
                            unsigned _ldv8_34_0;
                            unsigned _ldv8_34_1;
                            unsigned _ldv8_34_2;
                            unsigned _ldv8_34_3;
                            unsigned _ldv8_34_4;
                            unsigned _ldv8_34_5;
                            unsigned _ldv8_34_6;
                            unsigned _ldv8_34_7;
                            asm volatile(
                                "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                : "=r"(_ldv8_34_0), "=r"(_ldv8_34_1), "=r"(_ldv8_34_2), "=r"(_ldv8_34_3), "=r"(_ldv8_34_4), "=r"(_ldv8_34_5), "=r"(_ldv8_34_6), "=r"(_ldv8_34_7) : "l"((const void*)(ws + (fpbase + (unsigned long long)((col_half * B_HALF_N + 64 + 48) * BLOCK_M) + 0) + (0))) : "memory");
                            fpfill[48 + 0] = __uint_as_float(_ldv8_34_0);
                            fpfill[48 + 1] = __uint_as_float(_ldv8_34_1);
                            fpfill[48 + 2] = __uint_as_float(_ldv8_34_2);
                            fpfill[48 + 3] = __uint_as_float(_ldv8_34_3);
                            fpfill[48 + 4] = __uint_as_float(_ldv8_34_4);
                            fpfill[48 + 5] = __uint_as_float(_ldv8_34_5);
                            fpfill[48 + 6] = __uint_as_float(_ldv8_34_6);
                            fpfill[48 + 7] = __uint_as_float(_ldv8_34_7);
                        }
                        {
                            unsigned _ldv8_35_0;
                            unsigned _ldv8_35_1;
                            unsigned _ldv8_35_2;
                            unsigned _ldv8_35_3;
                            unsigned _ldv8_35_4;
                            unsigned _ldv8_35_5;
                            unsigned _ldv8_35_6;
                            unsigned _ldv8_35_7;
                            asm volatile(
                                "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                : "=r"(_ldv8_35_0), "=r"(_ldv8_35_1), "=r"(_ldv8_35_2), "=r"(_ldv8_35_3), "=r"(_ldv8_35_4), "=r"(_ldv8_35_5), "=r"(_ldv8_35_6), "=r"(_ldv8_35_7) : "l"((const void*)(ws + (fpbase + (unsigned long long)((col_half * B_HALF_N + 64 + 48) * BLOCK_M) + 8) + (0))) : "memory");
                            fpfill[56 + 0] = __uint_as_float(_ldv8_35_0);
                            fpfill[56 + 1] = __uint_as_float(_ldv8_35_1);
                            fpfill[56 + 2] = __uint_as_float(_ldv8_35_2);
                            fpfill[56 + 3] = __uint_as_float(_ldv8_35_3);
                            fpfill[56 + 4] = __uint_as_float(_ldv8_35_4);
                            fpfill[56 + 5] = __uint_as_float(_ldv8_35_5);
                            fpfill[56 + 6] = __uint_as_float(_ldv8_35_6);
                            fpfill[56 + 7] = __uint_as_float(_ldv8_35_7);
                        }
                        tmem_st_x16_f32(pre_addr + col_half * B_HALF_N + 64, fpfill);
                        tmem_st_x16_f32(pre_addr + col_half * B_HALF_N + 80, (fpfill + 16));
                        tmem_st_x16_f32(pre_addr + col_half * B_HALF_N + 96, (fpfill + 32));
                        tmem_st_x16_f32(pre_addr + col_half * B_HALF_N + 112, (fpfill + 48));
                        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                        asm volatile("barrier.sync 1, 256;" ::: "memory");
                        if (warp == 2) {
                            if (elect_sync()) {
                                counters[fo_pslot] = 0;
                            }
                        }
                    }
                    mbarrier_wait(mainloop_done_addr + (epi_stage) * 8, _phase_mainloop_done);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int fs_p_2 = p_hi_2 - seg_i;
                    int fs_q_2 = fs_p_2 - su0_2;
                    int fs_sp_2 = ((fs_q_2 >= 0) ? 1 : 0);
                    int fs_u_2 = ((fs_sp_2 == 1) ? su0_2 + fs_q_2 / 2 : fs_p_2);
                    int fs_kb_2 = ((fs_sp_2 == 1) ? fs_q_2 % 2 * (NUM_K_ITERS / 2) : 0);
                    int fs_ke_2 = ((fs_sp_2 == 1) ? fs_kb_2 + NUM_K_ITERS / 2 : NUM_K_ITERS);
                    int fs_ub_2 = fs_u_2 * NUM_K_ITERS;
                    int k_lo_2 = ((fs_kb_2) > (lo_2 - fs_ub_2) ? (fs_kb_2) : (lo_2 - fs_ub_2));
                    int k_hi_2 = ((fs_ke_2) < (hi_2 - fs_ub_2) ? (fs_ke_2) : (hi_2 - fs_ub_2));
                    int fs_pr_2 = m_tiles / CTA_GROUP;
                    int fs_grp_2 = fs_pr_2 / full_items;
                    int pseudo_2 = (fs_u_2 / fs_grp_2 * fs_pr_2 + fs_u_2 % fs_grp_2 * full_items + lane_r_2) * CTA_GROUP + rank_i_2;
                    int group_1 = pseudo_2 / tiles_per_group;
                    int first_m_1 = group_1 * GROUP_M;
                    int remaining_1 = m_tiles - first_m_1;
                    int group_size_1 = ((remaining_1 >= GROUP_M) ? GROUP_M : remaining_1);
                    int local_1 = pseudo_2 % tiles_per_group;
                    int bid_m_1 = first_m_1 + local_1 % group_size_1;
                    int bid_n_1 = local_1 / group_size_1;
                    int off_m_1 = bid_m_1 * BLOCK_M;
                    int cls_1 = ((bid_n_1 < R_TILES) ? 0 : ((bid_n_1 < R_TILES + L_TILES) ? 1 : 2));
                    int global_row = off_m_1 + local_row;
                    int lane_addr = taddr + (unsigned int)(epi_warp * 32 << 16) + epi_stage * (unsigned int)BLOCK_N;
                    int oth_addr = taddr + (unsigned int)(epi_warp * 32 << 16) + (epi_stage + 1) % 2 * (unsigned int)BLOCK_N;
                    int row_ok = ((global_row < M) ? 1 : 0);
                    int fo_cmp = ((k_hi_2 == fs_ke_2) ? 1 : 0);
                    int fo_p1in = ((fs_sp_2 == 1 && fs_ke_2 == NUM_K_ITERS && fo_cmp == 1 && lo_2 < fs_ub_2 + fo_half) ? 1 : 0);
                    int fo_p0f = ((fs_sp_2 == 1 && fs_ke_2 == fo_half && seg_i >= 1 && hi_2 >= fs_ub_2 + NUM_K_ITERS) ? 1 : 0);
                    int fo_st = ((fo_cmp == 0) ? 1 : 0);
                    if (fs_sp_2 == 1 && fs_ke_2 == fo_half && fo_p0f == 0) {
                        fo_st = ((seg_i == 0) ? 1 : 2);
                    }
                    if (fo_p1in == 1 && seg_i + 1 < nsegs_2 - 1) {
                        fo_st = 2;
                    }
                    int fo_keep = ((fo_p1in == 1 && seg_i + 1 == nsegs_2 - 1) ? 1 : 0);
                    int fo_fah = ((fs_sp_2 == 0 && fo_cmp == 1 && fo_all == 1 && k_lo_2 > 0) ? 1 : 0);
                    int fo_tih = ((fs_sp_2 == 1 && fs_ke_2 == NUM_K_ITERS && fo_cmp == 1 && lo_2 >= fs_ub_2 + fo_half) ? 1 : 0);
                    int fo_fold = fo_fah + fo_tih + fo_p0f;
                    if (fo_st > 0) {
                        int fo_sslot = ((fo_st == 1) ? fo_me : fo_me + fo_na);
                        unsigned long long fsbase = (unsigned long long)fo_sslot * (unsigned long long)(BLOCK_M * BLOCK_N) + (unsigned long long)local_row * 16;
                        float fstore[64];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                            : "=f"(fstore[0]), "=f"(fstore[1]), "=f"(fstore[2]), "=f"(fstore[3]), "=f"(fstore[4]), "=f"(fstore[5]), "=f"(fstore[6]), "=f"(fstore[7]), "=f"(fstore[8]), "=f"(fstore[9]), "=f"(fstore[10]), "=f"(fstore[11]), "=f"(fstore[12]), "=f"(fstore[13]), "=f"(fstore[14]), "=f"(fstore[15]), "=f"(fstore[16]), "=f"(fstore[17]), "=f"(fstore[18]), "=f"(fstore[19]), "=f"(fstore[20]), "=f"(fstore[21]), "=f"(fstore[22]), "=f"(fstore[23]), "=f"(fstore[24]), "=f"(fstore[25]), "=f"(fstore[26]), "=f"(fstore[27]), "=f"(fstore[28]), "=f"(fstore[29]), "=f"(fstore[30]), "=f"(fstore[31])
                            : "r"(lane_addr + col_half * B_HALF_N + 0));
                        asm volatile(
                            "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                            : "=f"(fstore[32]), "=f"(fstore[33]), "=f"(fstore[34]), "=f"(fstore[35]), "=f"(fstore[36]), "=f"(fstore[37]), "=f"(fstore[38]), "=f"(fstore[39]), "=f"(fstore[40]), "=f"(fstore[41]), "=f"(fstore[42]), "=f"(fstore[43]), "=f"(fstore[44]), "=f"(fstore[45]), "=f"(fstore[46]), "=f"(fstore[47]), "=f"(fstore[48]), "=f"(fstore[49]), "=f"(fstore[50]), "=f"(fstore[51]), "=f"(fstore[52]), "=f"(fstore[53]), "=f"(fstore[54]), "=f"(fstore[55]), "=f"(fstore[56]), "=f"(fstore[57]), "=f"(fstore[58]), "=f"(fstore[59]), "=f"(fstore[60]), "=f"(fstore[61]), "=f"(fstore[62]), "=f"(fstore[63])
                            : "r"(lane_addr + col_half * B_HALF_N + 32));
                        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                        {
                            unsigned _stv8_36_0 = __float_as_uint(fstore[0 + 0]);
                            unsigned _stv8_36_1 = __float_as_uint(fstore[0 + 1]);
                            unsigned _stv8_36_2 = __float_as_uint(fstore[0 + 2]);
                            unsigned _stv8_36_3 = __float_as_uint(fstore[0 + 3]);
                            unsigned _stv8_36_4 = __float_as_uint(fstore[0 + 4]);
                            unsigned _stv8_36_5 = __float_as_uint(fstore[0 + 5]);
                            unsigned _stv8_36_6 = __float_as_uint(fstore[0 + 6]);
                            unsigned _stv8_36_7 = __float_as_uint(fstore[0 + 7]);
                            asm volatile(
                                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                :: "l"((void*)(ws + (fsbase + (unsigned long long)((col_half * B_HALF_N + 0) * BLOCK_M)) + (0))), "r"(_stv8_36_0), "r"(_stv8_36_1), "r"(_stv8_36_2), "r"(_stv8_36_3), "r"(_stv8_36_4), "r"(_stv8_36_5), "r"(_stv8_36_6), "r"(_stv8_36_7) : "memory");
                        }
                        {
                            unsigned _stv8_37_0 = __float_as_uint(fstore[8 + 0]);
                            unsigned _stv8_37_1 = __float_as_uint(fstore[8 + 1]);
                            unsigned _stv8_37_2 = __float_as_uint(fstore[8 + 2]);
                            unsigned _stv8_37_3 = __float_as_uint(fstore[8 + 3]);
                            unsigned _stv8_37_4 = __float_as_uint(fstore[8 + 4]);
                            unsigned _stv8_37_5 = __float_as_uint(fstore[8 + 5]);
                            unsigned _stv8_37_6 = __float_as_uint(fstore[8 + 6]);
                            unsigned _stv8_37_7 = __float_as_uint(fstore[8 + 7]);
                            asm volatile(
                                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                :: "l"((void*)(ws + (fsbase + (unsigned long long)((col_half * B_HALF_N + 0) * BLOCK_M) + 8) + (0))), "r"(_stv8_37_0), "r"(_stv8_37_1), "r"(_stv8_37_2), "r"(_stv8_37_3), "r"(_stv8_37_4), "r"(_stv8_37_5), "r"(_stv8_37_6), "r"(_stv8_37_7) : "memory");
                        }
                        {
                            unsigned _stv8_38_0 = __float_as_uint(fstore[16 + 0]);
                            unsigned _stv8_38_1 = __float_as_uint(fstore[16 + 1]);
                            unsigned _stv8_38_2 = __float_as_uint(fstore[16 + 2]);
                            unsigned _stv8_38_3 = __float_as_uint(fstore[16 + 3]);
                            unsigned _stv8_38_4 = __float_as_uint(fstore[16 + 4]);
                            unsigned _stv8_38_5 = __float_as_uint(fstore[16 + 5]);
                            unsigned _stv8_38_6 = __float_as_uint(fstore[16 + 6]);
                            unsigned _stv8_38_7 = __float_as_uint(fstore[16 + 7]);
                            asm volatile(
                                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                :: "l"((void*)(ws + (fsbase + (unsigned long long)((col_half * B_HALF_N + 16) * BLOCK_M)) + (0))), "r"(_stv8_38_0), "r"(_stv8_38_1), "r"(_stv8_38_2), "r"(_stv8_38_3), "r"(_stv8_38_4), "r"(_stv8_38_5), "r"(_stv8_38_6), "r"(_stv8_38_7) : "memory");
                        }
                        {
                            unsigned _stv8_39_0 = __float_as_uint(fstore[24 + 0]);
                            unsigned _stv8_39_1 = __float_as_uint(fstore[24 + 1]);
                            unsigned _stv8_39_2 = __float_as_uint(fstore[24 + 2]);
                            unsigned _stv8_39_3 = __float_as_uint(fstore[24 + 3]);
                            unsigned _stv8_39_4 = __float_as_uint(fstore[24 + 4]);
                            unsigned _stv8_39_5 = __float_as_uint(fstore[24 + 5]);
                            unsigned _stv8_39_6 = __float_as_uint(fstore[24 + 6]);
                            unsigned _stv8_39_7 = __float_as_uint(fstore[24 + 7]);
                            asm volatile(
                                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                :: "l"((void*)(ws + (fsbase + (unsigned long long)((col_half * B_HALF_N + 16) * BLOCK_M) + 8) + (0))), "r"(_stv8_39_0), "r"(_stv8_39_1), "r"(_stv8_39_2), "r"(_stv8_39_3), "r"(_stv8_39_4), "r"(_stv8_39_5), "r"(_stv8_39_6), "r"(_stv8_39_7) : "memory");
                        }
                        {
                            unsigned _stv8_40_0 = __float_as_uint(fstore[32 + 0]);
                            unsigned _stv8_40_1 = __float_as_uint(fstore[32 + 1]);
                            unsigned _stv8_40_2 = __float_as_uint(fstore[32 + 2]);
                            unsigned _stv8_40_3 = __float_as_uint(fstore[32 + 3]);
                            unsigned _stv8_40_4 = __float_as_uint(fstore[32 + 4]);
                            unsigned _stv8_40_5 = __float_as_uint(fstore[32 + 5]);
                            unsigned _stv8_40_6 = __float_as_uint(fstore[32 + 6]);
                            unsigned _stv8_40_7 = __float_as_uint(fstore[32 + 7]);
                            asm volatile(
                                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                :: "l"((void*)(ws + (fsbase + (unsigned long long)((col_half * B_HALF_N + 32) * BLOCK_M)) + (0))), "r"(_stv8_40_0), "r"(_stv8_40_1), "r"(_stv8_40_2), "r"(_stv8_40_3), "r"(_stv8_40_4), "r"(_stv8_40_5), "r"(_stv8_40_6), "r"(_stv8_40_7) : "memory");
                        }
                        {
                            unsigned _stv8_41_0 = __float_as_uint(fstore[40 + 0]);
                            unsigned _stv8_41_1 = __float_as_uint(fstore[40 + 1]);
                            unsigned _stv8_41_2 = __float_as_uint(fstore[40 + 2]);
                            unsigned _stv8_41_3 = __float_as_uint(fstore[40 + 3]);
                            unsigned _stv8_41_4 = __float_as_uint(fstore[40 + 4]);
                            unsigned _stv8_41_5 = __float_as_uint(fstore[40 + 5]);
                            unsigned _stv8_41_6 = __float_as_uint(fstore[40 + 6]);
                            unsigned _stv8_41_7 = __float_as_uint(fstore[40 + 7]);
                            asm volatile(
                                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                :: "l"((void*)(ws + (fsbase + (unsigned long long)((col_half * B_HALF_N + 32) * BLOCK_M) + 8) + (0))), "r"(_stv8_41_0), "r"(_stv8_41_1), "r"(_stv8_41_2), "r"(_stv8_41_3), "r"(_stv8_41_4), "r"(_stv8_41_5), "r"(_stv8_41_6), "r"(_stv8_41_7) : "memory");
                        }
                        {
                            unsigned _stv8_42_0 = __float_as_uint(fstore[48 + 0]);
                            unsigned _stv8_42_1 = __float_as_uint(fstore[48 + 1]);
                            unsigned _stv8_42_2 = __float_as_uint(fstore[48 + 2]);
                            unsigned _stv8_42_3 = __float_as_uint(fstore[48 + 3]);
                            unsigned _stv8_42_4 = __float_as_uint(fstore[48 + 4]);
                            unsigned _stv8_42_5 = __float_as_uint(fstore[48 + 5]);
                            unsigned _stv8_42_6 = __float_as_uint(fstore[48 + 6]);
                            unsigned _stv8_42_7 = __float_as_uint(fstore[48 + 7]);
                            asm volatile(
                                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                :: "l"((void*)(ws + (fsbase + (unsigned long long)((col_half * B_HALF_N + 48) * BLOCK_M)) + (0))), "r"(_stv8_42_0), "r"(_stv8_42_1), "r"(_stv8_42_2), "r"(_stv8_42_3), "r"(_stv8_42_4), "r"(_stv8_42_5), "r"(_stv8_42_6), "r"(_stv8_42_7) : "memory");
                        }
                        {
                            unsigned _stv8_43_0 = __float_as_uint(fstore[56 + 0]);
                            unsigned _stv8_43_1 = __float_as_uint(fstore[56 + 1]);
                            unsigned _stv8_43_2 = __float_as_uint(fstore[56 + 2]);
                            unsigned _stv8_43_3 = __float_as_uint(fstore[56 + 3]);
                            unsigned _stv8_43_4 = __float_as_uint(fstore[56 + 4]);
                            unsigned _stv8_43_5 = __float_as_uint(fstore[56 + 5]);
                            unsigned _stv8_43_6 = __float_as_uint(fstore[56 + 6]);
                            unsigned _stv8_43_7 = __float_as_uint(fstore[56 + 7]);
                            asm volatile(
                                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                :: "l"((void*)(ws + (fsbase + (unsigned long long)((col_half * B_HALF_N + 48) * BLOCK_M) + 8) + (0))), "r"(_stv8_43_0), "r"(_stv8_43_1), "r"(_stv8_43_2), "r"(_stv8_43_3), "r"(_stv8_43_4), "r"(_stv8_43_5), "r"(_stv8_43_6), "r"(_stv8_43_7) : "memory");
                        }
                        asm volatile(
                            "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                            : "=f"(fstore[0]), "=f"(fstore[1]), "=f"(fstore[2]), "=f"(fstore[3]), "=f"(fstore[4]), "=f"(fstore[5]), "=f"(fstore[6]), "=f"(fstore[7]), "=f"(fstore[8]), "=f"(fstore[9]), "=f"(fstore[10]), "=f"(fstore[11]), "=f"(fstore[12]), "=f"(fstore[13]), "=f"(fstore[14]), "=f"(fstore[15]), "=f"(fstore[16]), "=f"(fstore[17]), "=f"(fstore[18]), "=f"(fstore[19]), "=f"(fstore[20]), "=f"(fstore[21]), "=f"(fstore[22]), "=f"(fstore[23]), "=f"(fstore[24]), "=f"(fstore[25]), "=f"(fstore[26]), "=f"(fstore[27]), "=f"(fstore[28]), "=f"(fstore[29]), "=f"(fstore[30]), "=f"(fstore[31])
                            : "r"(lane_addr + col_half * B_HALF_N + 64));
                        asm volatile(
                            "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                            : "=f"(fstore[32]), "=f"(fstore[33]), "=f"(fstore[34]), "=f"(fstore[35]), "=f"(fstore[36]), "=f"(fstore[37]), "=f"(fstore[38]), "=f"(fstore[39]), "=f"(fstore[40]), "=f"(fstore[41]), "=f"(fstore[42]), "=f"(fstore[43]), "=f"(fstore[44]), "=f"(fstore[45]), "=f"(fstore[46]), "=f"(fstore[47]), "=f"(fstore[48]), "=f"(fstore[49]), "=f"(fstore[50]), "=f"(fstore[51]), "=f"(fstore[52]), "=f"(fstore[53]), "=f"(fstore[54]), "=f"(fstore[55]), "=f"(fstore[56]), "=f"(fstore[57]), "=f"(fstore[58]), "=f"(fstore[59]), "=f"(fstore[60]), "=f"(fstore[61]), "=f"(fstore[62]), "=f"(fstore[63])
                            : "r"(lane_addr + col_half * B_HALF_N + 96));
                        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                        {
                            unsigned _stv8_44_0 = __float_as_uint(fstore[0 + 0]);
                            unsigned _stv8_44_1 = __float_as_uint(fstore[0 + 1]);
                            unsigned _stv8_44_2 = __float_as_uint(fstore[0 + 2]);
                            unsigned _stv8_44_3 = __float_as_uint(fstore[0 + 3]);
                            unsigned _stv8_44_4 = __float_as_uint(fstore[0 + 4]);
                            unsigned _stv8_44_5 = __float_as_uint(fstore[0 + 5]);
                            unsigned _stv8_44_6 = __float_as_uint(fstore[0 + 6]);
                            unsigned _stv8_44_7 = __float_as_uint(fstore[0 + 7]);
                            asm volatile(
                                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                :: "l"((void*)(ws + (fsbase + (unsigned long long)((col_half * B_HALF_N + 64) * BLOCK_M)) + (0))), "r"(_stv8_44_0), "r"(_stv8_44_1), "r"(_stv8_44_2), "r"(_stv8_44_3), "r"(_stv8_44_4), "r"(_stv8_44_5), "r"(_stv8_44_6), "r"(_stv8_44_7) : "memory");
                        }
                        {
                            unsigned _stv8_45_0 = __float_as_uint(fstore[8 + 0]);
                            unsigned _stv8_45_1 = __float_as_uint(fstore[8 + 1]);
                            unsigned _stv8_45_2 = __float_as_uint(fstore[8 + 2]);
                            unsigned _stv8_45_3 = __float_as_uint(fstore[8 + 3]);
                            unsigned _stv8_45_4 = __float_as_uint(fstore[8 + 4]);
                            unsigned _stv8_45_5 = __float_as_uint(fstore[8 + 5]);
                            unsigned _stv8_45_6 = __float_as_uint(fstore[8 + 6]);
                            unsigned _stv8_45_7 = __float_as_uint(fstore[8 + 7]);
                            asm volatile(
                                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                :: "l"((void*)(ws + (fsbase + (unsigned long long)((col_half * B_HALF_N + 64) * BLOCK_M) + 8) + (0))), "r"(_stv8_45_0), "r"(_stv8_45_1), "r"(_stv8_45_2), "r"(_stv8_45_3), "r"(_stv8_45_4), "r"(_stv8_45_5), "r"(_stv8_45_6), "r"(_stv8_45_7) : "memory");
                        }
                        {
                            unsigned _stv8_46_0 = __float_as_uint(fstore[16 + 0]);
                            unsigned _stv8_46_1 = __float_as_uint(fstore[16 + 1]);
                            unsigned _stv8_46_2 = __float_as_uint(fstore[16 + 2]);
                            unsigned _stv8_46_3 = __float_as_uint(fstore[16 + 3]);
                            unsigned _stv8_46_4 = __float_as_uint(fstore[16 + 4]);
                            unsigned _stv8_46_5 = __float_as_uint(fstore[16 + 5]);
                            unsigned _stv8_46_6 = __float_as_uint(fstore[16 + 6]);
                            unsigned _stv8_46_7 = __float_as_uint(fstore[16 + 7]);
                            asm volatile(
                                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                :: "l"((void*)(ws + (fsbase + (unsigned long long)((col_half * B_HALF_N + 80) * BLOCK_M)) + (0))), "r"(_stv8_46_0), "r"(_stv8_46_1), "r"(_stv8_46_2), "r"(_stv8_46_3), "r"(_stv8_46_4), "r"(_stv8_46_5), "r"(_stv8_46_6), "r"(_stv8_46_7) : "memory");
                        }
                        {
                            unsigned _stv8_47_0 = __float_as_uint(fstore[24 + 0]);
                            unsigned _stv8_47_1 = __float_as_uint(fstore[24 + 1]);
                            unsigned _stv8_47_2 = __float_as_uint(fstore[24 + 2]);
                            unsigned _stv8_47_3 = __float_as_uint(fstore[24 + 3]);
                            unsigned _stv8_47_4 = __float_as_uint(fstore[24 + 4]);
                            unsigned _stv8_47_5 = __float_as_uint(fstore[24 + 5]);
                            unsigned _stv8_47_6 = __float_as_uint(fstore[24 + 6]);
                            unsigned _stv8_47_7 = __float_as_uint(fstore[24 + 7]);
                            asm volatile(
                                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                :: "l"((void*)(ws + (fsbase + (unsigned long long)((col_half * B_HALF_N + 80) * BLOCK_M) + 8) + (0))), "r"(_stv8_47_0), "r"(_stv8_47_1), "r"(_stv8_47_2), "r"(_stv8_47_3), "r"(_stv8_47_4), "r"(_stv8_47_5), "r"(_stv8_47_6), "r"(_stv8_47_7) : "memory");
                        }
                        {
                            unsigned _stv8_48_0 = __float_as_uint(fstore[32 + 0]);
                            unsigned _stv8_48_1 = __float_as_uint(fstore[32 + 1]);
                            unsigned _stv8_48_2 = __float_as_uint(fstore[32 + 2]);
                            unsigned _stv8_48_3 = __float_as_uint(fstore[32 + 3]);
                            unsigned _stv8_48_4 = __float_as_uint(fstore[32 + 4]);
                            unsigned _stv8_48_5 = __float_as_uint(fstore[32 + 5]);
                            unsigned _stv8_48_6 = __float_as_uint(fstore[32 + 6]);
                            unsigned _stv8_48_7 = __float_as_uint(fstore[32 + 7]);
                            asm volatile(
                                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                :: "l"((void*)(ws + (fsbase + (unsigned long long)((col_half * B_HALF_N + 96) * BLOCK_M)) + (0))), "r"(_stv8_48_0), "r"(_stv8_48_1), "r"(_stv8_48_2), "r"(_stv8_48_3), "r"(_stv8_48_4), "r"(_stv8_48_5), "r"(_stv8_48_6), "r"(_stv8_48_7) : "memory");
                        }
                        {
                            unsigned _stv8_49_0 = __float_as_uint(fstore[40 + 0]);
                            unsigned _stv8_49_1 = __float_as_uint(fstore[40 + 1]);
                            unsigned _stv8_49_2 = __float_as_uint(fstore[40 + 2]);
                            unsigned _stv8_49_3 = __float_as_uint(fstore[40 + 3]);
                            unsigned _stv8_49_4 = __float_as_uint(fstore[40 + 4]);
                            unsigned _stv8_49_5 = __float_as_uint(fstore[40 + 5]);
                            unsigned _stv8_49_6 = __float_as_uint(fstore[40 + 6]);
                            unsigned _stv8_49_7 = __float_as_uint(fstore[40 + 7]);
                            asm volatile(
                                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                :: "l"((void*)(ws + (fsbase + (unsigned long long)((col_half * B_HALF_N + 96) * BLOCK_M) + 8) + (0))), "r"(_stv8_49_0), "r"(_stv8_49_1), "r"(_stv8_49_2), "r"(_stv8_49_3), "r"(_stv8_49_4), "r"(_stv8_49_5), "r"(_stv8_49_6), "r"(_stv8_49_7) : "memory");
                        }
                        {
                            unsigned _stv8_50_0 = __float_as_uint(fstore[48 + 0]);
                            unsigned _stv8_50_1 = __float_as_uint(fstore[48 + 1]);
                            unsigned _stv8_50_2 = __float_as_uint(fstore[48 + 2]);
                            unsigned _stv8_50_3 = __float_as_uint(fstore[48 + 3]);
                            unsigned _stv8_50_4 = __float_as_uint(fstore[48 + 4]);
                            unsigned _stv8_50_5 = __float_as_uint(fstore[48 + 5]);
                            unsigned _stv8_50_6 = __float_as_uint(fstore[48 + 6]);
                            unsigned _stv8_50_7 = __float_as_uint(fstore[48 + 7]);
                            asm volatile(
                                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                :: "l"((void*)(ws + (fsbase + (unsigned long long)((col_half * B_HALF_N + 112) * BLOCK_M)) + (0))), "r"(_stv8_50_0), "r"(_stv8_50_1), "r"(_stv8_50_2), "r"(_stv8_50_3), "r"(_stv8_50_4), "r"(_stv8_50_5), "r"(_stv8_50_6), "r"(_stv8_50_7) : "memory");
                        }
                        {
                            unsigned _stv8_51_0 = __float_as_uint(fstore[56 + 0]);
                            unsigned _stv8_51_1 = __float_as_uint(fstore[56 + 1]);
                            unsigned _stv8_51_2 = __float_as_uint(fstore[56 + 2]);
                            unsigned _stv8_51_3 = __float_as_uint(fstore[56 + 3]);
                            unsigned _stv8_51_4 = __float_as_uint(fstore[56 + 4]);
                            unsigned _stv8_51_5 = __float_as_uint(fstore[56 + 5]);
                            unsigned _stv8_51_6 = __float_as_uint(fstore[56 + 6]);
                            unsigned _stv8_51_7 = __float_as_uint(fstore[56 + 7]);
                            asm volatile(
                                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                :: "l"((void*)(ws + (fsbase + (unsigned long long)((col_half * B_HALF_N + 112) * BLOCK_M) + 8) + (0))), "r"(_stv8_51_0), "r"(_stv8_51_1), "r"(_stv8_51_2), "r"(_stv8_51_3), "r"(_stv8_51_4), "r"(_stv8_51_5), "r"(_stv8_51_6), "r"(_stv8_51_7) : "memory");
                        }
                        asm volatile("tcgen05.fence::before_thread_sync;");
                        asm volatile("barrier.sync 1, 256;" ::: "memory");
                        if (warp == 2) {
                            if (elect_sync()) {
                                asm volatile("fence.release.gpu;" ::: "memory");
                                int _fst_old = atomicAdd(&counters[fo_sslot], 1);
                            }
                        }
                    }
                    if (fo_fold > 0) {
                        int fo_a0 = ((fo_p0f == 1) ? lane_addr : oth_addr);
                        int fo_a2 = ((fo_p0f == 1) ? oth_addr : lane_addr);
                        int fo_mlast = ((seg_i < nsegs_2 - 1) ? 1 : 0);
                        int fo_t2 = 1 - fo_p0f * fo_mlast;
                        int fo_ws = fo_p0f * fo_mlast + fo_fah * fo_mid;
                        int fo_wslot = ((fo_p0f == 1) ? fo_me + fo_na : fo_prev);
                        if (fo_ws == 1) {
                            if (warp == 2) {
                                if (elect_sync()) {
                                    int32_t _fws_ld_0;
                                    asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_fws_ld_0) : "l"(counters + fo_wslot) : "memory");
                                    int fcnt_w = _fws_ld_0;
                                    #pragma unroll 1
                                    for (int _fspin_w = 0; _fspin_w < 4194304; _fspin_w++) {
                                        if (fcnt_w >= 1) {
                                            break;
                                        }
                                        int32_t _fws_ld_1;
                                        asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_fws_ld_1) : "l"(counters + fo_wslot) : "memory");
                                        fcnt_w = _fws_ld_1;
                                    }
                                    if (fcnt_w < 1) {
                                        asm volatile("trap;" ::: "memory");
                                    }
                                    asm volatile("fence.acquire.gpu;" ::: "memory");
                                }
                            }
                            asm volatile("barrier.sync 1, 256;" ::: "memory");
                        }
                        unsigned long long fwbase = (unsigned long long)fo_wslot * (unsigned long long)(BLOCK_M * BLOCK_N) + (unsigned long long)local_row * 16;
                        if (cls_1 < 2) {
                            int col_base_rf = bid_n_1 * BLOCK_N + col_half * B_HALF_N;
                            unsigned long long row_out_rf = (unsigned long long)global_row * (unsigned long long)NUM_EXPERTS + (unsigned long long)col_base_rf;
                            int store_rf = ((col_base_rf < NUM_EXPERTS) ? 1 : 0);
                            int col_base_lf = (bid_n_1 - R_TILES) * BLOCK_N + col_half * B_HALF_N;
                            unsigned long long row_out_lf = (unsigned long long)global_row * (unsigned long long)LATENT + (unsigned long long)col_base_lf;
                            #pragma unroll
                            for (int fh = 0; fh < 2; fh++) {
                                int colf = col_half * B_HALF_N + fh * 64;
                                float totf[64];
                                #pragma unroll
                                for (int e = 0; e < 64; e++) {
                                    totf[e] = 0.0f;
                                }
                                float fld0[64];
                                asm volatile(
                                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                                    : "=f"(fld0[0]), "=f"(fld0[1]), "=f"(fld0[2]), "=f"(fld0[3]), "=f"(fld0[4]), "=f"(fld0[5]), "=f"(fld0[6]), "=f"(fld0[7]), "=f"(fld0[8]), "=f"(fld0[9]), "=f"(fld0[10]), "=f"(fld0[11]), "=f"(fld0[12]), "=f"(fld0[13]), "=f"(fld0[14]), "=f"(fld0[15]), "=f"(fld0[16]), "=f"(fld0[17]), "=f"(fld0[18]), "=f"(fld0[19]), "=f"(fld0[20]), "=f"(fld0[21]), "=f"(fld0[22]), "=f"(fld0[23]), "=f"(fld0[24]), "=f"(fld0[25]), "=f"(fld0[26]), "=f"(fld0[27]), "=f"(fld0[28]), "=f"(fld0[29]), "=f"(fld0[30]), "=f"(fld0[31])
                                    : "r"(fo_a0 + colf));
                                asm volatile("tcgen05.wait::ld.sync.aligned;");
                                asm volatile(
                                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                                    : "=f"(fld0[32]), "=f"(fld0[33]), "=f"(fld0[34]), "=f"(fld0[35]), "=f"(fld0[36]), "=f"(fld0[37]), "=f"(fld0[38]), "=f"(fld0[39]), "=f"(fld0[40]), "=f"(fld0[41]), "=f"(fld0[42]), "=f"(fld0[43]), "=f"(fld0[44]), "=f"(fld0[45]), "=f"(fld0[46]), "=f"(fld0[47]), "=f"(fld0[48]), "=f"(fld0[49]), "=f"(fld0[50]), "=f"(fld0[51]), "=f"(fld0[52]), "=f"(fld0[53]), "=f"(fld0[54]), "=f"(fld0[55]), "=f"(fld0[56]), "=f"(fld0[57]), "=f"(fld0[58]), "=f"(fld0[59]), "=f"(fld0[60]), "=f"(fld0[61]), "=f"(fld0[62]), "=f"(fld0[63])
                                    : "r"(fo_a0 + colf + 32));
                                asm volatile("tcgen05.wait::ld.sync.aligned;");
                                #pragma unroll
                                for (int e_1 = 0; e_1 < 64; e_1++) {
                                    totf[e_1] = totf[e_1] + fld0[e_1];
                                }
                                if (fo_ws == 1) {
                                    #pragma unroll
                                    for (int q = 0; q < 8; q++) {
                                        float fwv[8];
                                        {
                                            unsigned _ldv8_52_0;
                                            unsigned _ldv8_52_1;
                                            unsigned _ldv8_52_2;
                                            unsigned _ldv8_52_3;
                                            unsigned _ldv8_52_4;
                                            unsigned _ldv8_52_5;
                                            unsigned _ldv8_52_6;
                                            unsigned _ldv8_52_7;
                                            asm volatile(
                                                "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                                : "=r"(_ldv8_52_0), "=r"(_ldv8_52_1), "=r"(_ldv8_52_2), "=r"(_ldv8_52_3), "=r"(_ldv8_52_4), "=r"(_ldv8_52_5), "=r"(_ldv8_52_6), "=r"(_ldv8_52_7) : "l"((const void*)(ws + (fwbase + (unsigned long long)(colf * BLOCK_M) + (unsigned long long)(q / 2 * (16 * BLOCK_M)) + (unsigned long long)(q % 2 * 8)) + (0))) : "memory");
                                            fwv[0 + 0] = __uint_as_float(_ldv8_52_0);
                                            fwv[0 + 1] = __uint_as_float(_ldv8_52_1);
                                            fwv[0 + 2] = __uint_as_float(_ldv8_52_2);
                                            fwv[0 + 3] = __uint_as_float(_ldv8_52_3);
                                            fwv[0 + 4] = __uint_as_float(_ldv8_52_4);
                                            fwv[0 + 5] = __uint_as_float(_ldv8_52_5);
                                            fwv[0 + 6] = __uint_as_float(_ldv8_52_6);
                                            fwv[0 + 7] = __uint_as_float(_ldv8_52_7);
                                        }
                                        #pragma unroll
                                        for (int e_2 = 0; e_2 < 8; e_2++) {
                                            totf[q * 8 + e_2] = totf[q * 8 + e_2] + fwv[e_2];
                                        }
                                    }
                                }
                                if (fo_t2 == 1) {
                                    float fld2[64];
                                    asm volatile(
                                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                                        : "=f"(fld2[0]), "=f"(fld2[1]), "=f"(fld2[2]), "=f"(fld2[3]), "=f"(fld2[4]), "=f"(fld2[5]), "=f"(fld2[6]), "=f"(fld2[7]), "=f"(fld2[8]), "=f"(fld2[9]), "=f"(fld2[10]), "=f"(fld2[11]), "=f"(fld2[12]), "=f"(fld2[13]), "=f"(fld2[14]), "=f"(fld2[15]), "=f"(fld2[16]), "=f"(fld2[17]), "=f"(fld2[18]), "=f"(fld2[19]), "=f"(fld2[20]), "=f"(fld2[21]), "=f"(fld2[22]), "=f"(fld2[23]), "=f"(fld2[24]), "=f"(fld2[25]), "=f"(fld2[26]), "=f"(fld2[27]), "=f"(fld2[28]), "=f"(fld2[29]), "=f"(fld2[30]), "=f"(fld2[31])
                                        : "r"(fo_a2 + colf));
                                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                                    asm volatile(
                                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                                        : "=f"(fld2[32]), "=f"(fld2[33]), "=f"(fld2[34]), "=f"(fld2[35]), "=f"(fld2[36]), "=f"(fld2[37]), "=f"(fld2[38]), "=f"(fld2[39]), "=f"(fld2[40]), "=f"(fld2[41]), "=f"(fld2[42]), "=f"(fld2[43]), "=f"(fld2[44]), "=f"(fld2[45]), "=f"(fld2[46]), "=f"(fld2[47]), "=f"(fld2[48]), "=f"(fld2[49]), "=f"(fld2[50]), "=f"(fld2[51]), "=f"(fld2[52]), "=f"(fld2[53]), "=f"(fld2[54]), "=f"(fld2[55]), "=f"(fld2[56]), "=f"(fld2[57]), "=f"(fld2[58]), "=f"(fld2[59]), "=f"(fld2[60]), "=f"(fld2[61]), "=f"(fld2[62]), "=f"(fld2[63])
                                        : "r"(fo_a2 + colf + 32));
                                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                                    #pragma unroll
                                    for (int e_3 = 0; e_3 < 64; e_3++) {
                                        totf[e_3] = totf[e_3] + fld2[e_3];
                                    }
                                }
                                #pragma unroll
                                for (int q_1 = 0; q_1 < 4; q_1++) {
                                    float outf[16];
                                    #pragma unroll
                                    for (int e_4 = 0; e_4 < 16; e_4++) {
                                        outf[e_4] = totf[q_1 * 16 + e_4];
                                    }
                                    if (cls_1 == 0) {
                                        if (row_ok + store_rf == 2) {
                                            {
                                                unsigned _stv8_53_0 = __float_as_uint(outf[0 + 0]);
                                                unsigned _stv8_53_1 = __float_as_uint(outf[0 + 1]);
                                                unsigned _stv8_53_2 = __float_as_uint(outf[0 + 2]);
                                                unsigned _stv8_53_3 = __float_as_uint(outf[0 + 3]);
                                                unsigned _stv8_53_4 = __float_as_uint(outf[0 + 4]);
                                                unsigned _stv8_53_5 = __float_as_uint(outf[0 + 5]);
                                                unsigned _stv8_53_6 = __float_as_uint(outf[0 + 6]);
                                                unsigned _stv8_53_7 = __float_as_uint(outf[0 + 7]);
                                                asm volatile(
                                                    "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                                    :: "l"((void*)(logits + (row_out_rf + (unsigned long long)(fh * 64) + (unsigned long long)(q_1 * 16)) + (0))), "r"(_stv8_53_0), "r"(_stv8_53_1), "r"(_stv8_53_2), "r"(_stv8_53_3), "r"(_stv8_53_4), "r"(_stv8_53_5), "r"(_stv8_53_6), "r"(_stv8_53_7) : "memory");
                                            }
                                            {
                                                unsigned _stv8_54_0 = __float_as_uint(outf[8 + 0]);
                                                unsigned _stv8_54_1 = __float_as_uint(outf[8 + 1]);
                                                unsigned _stv8_54_2 = __float_as_uint(outf[8 + 2]);
                                                unsigned _stv8_54_3 = __float_as_uint(outf[8 + 3]);
                                                unsigned _stv8_54_4 = __float_as_uint(outf[8 + 4]);
                                                unsigned _stv8_54_5 = __float_as_uint(outf[8 + 5]);
                                                unsigned _stv8_54_6 = __float_as_uint(outf[8 + 6]);
                                                unsigned _stv8_54_7 = __float_as_uint(outf[8 + 7]);
                                                asm volatile(
                                                    "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                                    :: "l"((void*)(logits + (row_out_rf + (unsigned long long)(fh * 64) + (unsigned long long)(q_1 * 16) + 8) + (0))), "r"(_stv8_54_0), "r"(_stv8_54_1), "r"(_stv8_54_2), "r"(_stv8_54_3), "r"(_stv8_54_4), "r"(_stv8_54_5), "r"(_stv8_54_6), "r"(_stv8_54_7) : "memory");
                                            }
                                        }
                                    }
                                    if (cls_1 == 1) {
                                        if (row_ok == 1) {
                                            unsigned int fpack[8];
                                            #pragma unroll
                                            for (int _lp = 0; _lp < 1; _lp++) {
                                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(outf[_lp*2 + 0], outf[_lp*2+1 + 0]));
                                                fpack[_lp] = *(uint32_t*)&_bf2;
                                            }
                                            #pragma unroll
                                            for (int _lp = 0; _lp < 1; _lp++) {
                                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(outf[_lp*2 + 2], outf[_lp*2+1 + 2]));
                                                fpack[_lp + 1] = *(uint32_t*)&_bf2;
                                            }
                                            #pragma unroll
                                            for (int _lp = 0; _lp < 1; _lp++) {
                                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(outf[_lp*2 + 4], outf[_lp*2+1 + 4]));
                                                fpack[_lp + 2] = *(uint32_t*)&_bf2;
                                            }
                                            #pragma unroll
                                            for (int _lp = 0; _lp < 1; _lp++) {
                                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(outf[_lp*2 + 6], outf[_lp*2+1 + 6]));
                                                fpack[_lp + 3] = *(uint32_t*)&_bf2;
                                            }
                                            #pragma unroll
                                            for (int _lp = 0; _lp < 1; _lp++) {
                                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(outf[_lp*2 + 8], outf[_lp*2+1 + 8]));
                                                fpack[_lp + 4] = *(uint32_t*)&_bf2;
                                            }
                                            #pragma unroll
                                            for (int _lp = 0; _lp < 1; _lp++) {
                                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(outf[_lp*2 + 10], outf[_lp*2+1 + 10]));
                                                fpack[_lp + 5] = *(uint32_t*)&_bf2;
                                            }
                                            #pragma unroll
                                            for (int _lp = 0; _lp < 1; _lp++) {
                                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(outf[_lp*2 + 12], outf[_lp*2+1 + 12]));
                                                fpack[_lp + 6] = *(uint32_t*)&_bf2;
                                            }
                                            #pragma unroll
                                            for (int _lp = 0; _lp < 1; _lp++) {
                                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(outf[_lp*2 + 14], outf[_lp*2+1 + 14]));
                                                fpack[_lp + 7] = *(uint32_t*)&_bf2;
                                            }
                                            {
                                                const unsigned* _raw_stv8_55 = reinterpret_cast<const unsigned*>(fpack);
                                                asm volatile(
                                                    "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                                    :: "l"((void*)(latent + (row_out_lf + (unsigned long long)(fh * 64) + (unsigned long long)(q_1 * 16)))), "r"(_raw_stv8_55[0]), "r"(_raw_stv8_55[1]), "r"(_raw_stv8_55[2]), "r"(_raw_stv8_55[3]), "r"(_raw_stv8_55[4]), "r"(_raw_stv8_55[5]), "r"(_raw_stv8_55[6]), "r"(_raw_stv8_55[7]) : "memory");
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        if (cls_1 == 2) {
                            int col_base_sf = (bid_n_1 - R_TILES - L_TILES) * B_HALF_N + col_half * (B_HALF_N / 2);
                            unsigned long long row_out_sf = (unsigned long long)global_row * (unsigned long long)I_LOCAL + (unsigned long long)col_base_sf;
                            #pragma unroll
                            for (int fh_1 = 0; fh_1 < 2; fh_1++) {
                                int colgf = col_half * (B_HALF_N / 2) + fh_1 * 32;
                                float totgf[32];
                                float totuf[32];
                                #pragma unroll
                                for (int e_5 = 0; e_5 < 32; e_5++) {
                                    totgf[e_5] = 0.0f;
                                    totuf[e_5] = 0.0f;
                                }
                                float fg0[32];
                                asm volatile(
                                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                                    : "=f"(fg0[0]), "=f"(fg0[1]), "=f"(fg0[2]), "=f"(fg0[3]), "=f"(fg0[4]), "=f"(fg0[5]), "=f"(fg0[6]), "=f"(fg0[7]), "=f"(fg0[8]), "=f"(fg0[9]), "=f"(fg0[10]), "=f"(fg0[11]), "=f"(fg0[12]), "=f"(fg0[13]), "=f"(fg0[14]), "=f"(fg0[15]), "=f"(fg0[16]), "=f"(fg0[17]), "=f"(fg0[18]), "=f"(fg0[19]), "=f"(fg0[20]), "=f"(fg0[21]), "=f"(fg0[22]), "=f"(fg0[23]), "=f"(fg0[24]), "=f"(fg0[25]), "=f"(fg0[26]), "=f"(fg0[27]), "=f"(fg0[28]), "=f"(fg0[29]), "=f"(fg0[30]), "=f"(fg0[31])
                                    : "r"(fo_a0 + colgf));
                                asm volatile("tcgen05.wait::ld.sync.aligned;");
                                float fu0[32];
                                asm volatile(
                                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                                    : "=f"(fu0[0]), "=f"(fu0[1]), "=f"(fu0[2]), "=f"(fu0[3]), "=f"(fu0[4]), "=f"(fu0[5]), "=f"(fu0[6]), "=f"(fu0[7]), "=f"(fu0[8]), "=f"(fu0[9]), "=f"(fu0[10]), "=f"(fu0[11]), "=f"(fu0[12]), "=f"(fu0[13]), "=f"(fu0[14]), "=f"(fu0[15]), "=f"(fu0[16]), "=f"(fu0[17]), "=f"(fu0[18]), "=f"(fu0[19]), "=f"(fu0[20]), "=f"(fu0[21]), "=f"(fu0[22]), "=f"(fu0[23]), "=f"(fu0[24]), "=f"(fu0[25]), "=f"(fu0[26]), "=f"(fu0[27]), "=f"(fu0[28]), "=f"(fu0[29]), "=f"(fu0[30]), "=f"(fu0[31])
                                    : "r"(fo_a0 + B_HALF_N + colgf));
                                asm volatile("tcgen05.wait::ld.sync.aligned;");
                                #pragma unroll
                                for (int e_6 = 0; e_6 < 32; e_6++) {
                                    totgf[e_6] = totgf[e_6] + fg0[e_6];
                                    totuf[e_6] = totuf[e_6] + fu0[e_6];
                                }
                                if (fo_ws == 1) {
                                    #pragma unroll
                                    for (int q_2 = 0; q_2 < 4; q_2++) {
                                        float fwg[8];
                                        {
                                            unsigned _ldv8_56_0;
                                            unsigned _ldv8_56_1;
                                            unsigned _ldv8_56_2;
                                            unsigned _ldv8_56_3;
                                            unsigned _ldv8_56_4;
                                            unsigned _ldv8_56_5;
                                            unsigned _ldv8_56_6;
                                            unsigned _ldv8_56_7;
                                            asm volatile(
                                                "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                                : "=r"(_ldv8_56_0), "=r"(_ldv8_56_1), "=r"(_ldv8_56_2), "=r"(_ldv8_56_3), "=r"(_ldv8_56_4), "=r"(_ldv8_56_5), "=r"(_ldv8_56_6), "=r"(_ldv8_56_7) : "l"((const void*)(ws + (fwbase + (unsigned long long)(colgf * BLOCK_M) + (unsigned long long)(q_2 / 2 * (16 * BLOCK_M)) + (unsigned long long)(q_2 % 2 * 8)) + (0))) : "memory");
                                            fwg[0 + 0] = __uint_as_float(_ldv8_56_0);
                                            fwg[0 + 1] = __uint_as_float(_ldv8_56_1);
                                            fwg[0 + 2] = __uint_as_float(_ldv8_56_2);
                                            fwg[0 + 3] = __uint_as_float(_ldv8_56_3);
                                            fwg[0 + 4] = __uint_as_float(_ldv8_56_4);
                                            fwg[0 + 5] = __uint_as_float(_ldv8_56_5);
                                            fwg[0 + 6] = __uint_as_float(_ldv8_56_6);
                                            fwg[0 + 7] = __uint_as_float(_ldv8_56_7);
                                        }
                                        float fwu[8];
                                        {
                                            unsigned _ldv8_57_0;
                                            unsigned _ldv8_57_1;
                                            unsigned _ldv8_57_2;
                                            unsigned _ldv8_57_3;
                                            unsigned _ldv8_57_4;
                                            unsigned _ldv8_57_5;
                                            unsigned _ldv8_57_6;
                                            unsigned _ldv8_57_7;
                                            asm volatile(
                                                "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                                : "=r"(_ldv8_57_0), "=r"(_ldv8_57_1), "=r"(_ldv8_57_2), "=r"(_ldv8_57_3), "=r"(_ldv8_57_4), "=r"(_ldv8_57_5), "=r"(_ldv8_57_6), "=r"(_ldv8_57_7) : "l"((const void*)(ws + (fwbase + (unsigned long long)(B_HALF_N * BLOCK_M) + (unsigned long long)(colgf * BLOCK_M) + (unsigned long long)(q_2 / 2 * (16 * BLOCK_M)) + (unsigned long long)(q_2 % 2 * 8)) + (0))) : "memory");
                                            fwu[0 + 0] = __uint_as_float(_ldv8_57_0);
                                            fwu[0 + 1] = __uint_as_float(_ldv8_57_1);
                                            fwu[0 + 2] = __uint_as_float(_ldv8_57_2);
                                            fwu[0 + 3] = __uint_as_float(_ldv8_57_3);
                                            fwu[0 + 4] = __uint_as_float(_ldv8_57_4);
                                            fwu[0 + 5] = __uint_as_float(_ldv8_57_5);
                                            fwu[0 + 6] = __uint_as_float(_ldv8_57_6);
                                            fwu[0 + 7] = __uint_as_float(_ldv8_57_7);
                                        }
                                        #pragma unroll
                                        for (int e_7 = 0; e_7 < 8; e_7++) {
                                            totgf[q_2 * 8 + e_7] = totgf[q_2 * 8 + e_7] + fwg[e_7];
                                            totuf[q_2 * 8 + e_7] = totuf[q_2 * 8 + e_7] + fwu[e_7];
                                        }
                                    }
                                }
                                if (fo_t2 == 1) {
                                    float fg2[32];
                                    asm volatile(
                                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                                        : "=f"(fg2[0]), "=f"(fg2[1]), "=f"(fg2[2]), "=f"(fg2[3]), "=f"(fg2[4]), "=f"(fg2[5]), "=f"(fg2[6]), "=f"(fg2[7]), "=f"(fg2[8]), "=f"(fg2[9]), "=f"(fg2[10]), "=f"(fg2[11]), "=f"(fg2[12]), "=f"(fg2[13]), "=f"(fg2[14]), "=f"(fg2[15]), "=f"(fg2[16]), "=f"(fg2[17]), "=f"(fg2[18]), "=f"(fg2[19]), "=f"(fg2[20]), "=f"(fg2[21]), "=f"(fg2[22]), "=f"(fg2[23]), "=f"(fg2[24]), "=f"(fg2[25]), "=f"(fg2[26]), "=f"(fg2[27]), "=f"(fg2[28]), "=f"(fg2[29]), "=f"(fg2[30]), "=f"(fg2[31])
                                        : "r"(fo_a2 + colgf));
                                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                                    float fu2[32];
                                    asm volatile(
                                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                                        : "=f"(fu2[0]), "=f"(fu2[1]), "=f"(fu2[2]), "=f"(fu2[3]), "=f"(fu2[4]), "=f"(fu2[5]), "=f"(fu2[6]), "=f"(fu2[7]), "=f"(fu2[8]), "=f"(fu2[9]), "=f"(fu2[10]), "=f"(fu2[11]), "=f"(fu2[12]), "=f"(fu2[13]), "=f"(fu2[14]), "=f"(fu2[15]), "=f"(fu2[16]), "=f"(fu2[17]), "=f"(fu2[18]), "=f"(fu2[19]), "=f"(fu2[20]), "=f"(fu2[21]), "=f"(fu2[22]), "=f"(fu2[23]), "=f"(fu2[24]), "=f"(fu2[25]), "=f"(fu2[26]), "=f"(fu2[27]), "=f"(fu2[28]), "=f"(fu2[29]), "=f"(fu2[30]), "=f"(fu2[31])
                                        : "r"(fo_a2 + B_HALF_N + colgf));
                                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                                    #pragma unroll
                                    for (int e_8 = 0; e_8 < 32; e_8++) {
                                        totgf[e_8] = totgf[e_8] + fg2[e_8];
                                        totuf[e_8] = totuf[e_8] + fu2[e_8];
                                    }
                                }
                                #pragma unroll
                                for (int q_3 = 0; q_3 < 2; q_3++) {
                                    float out_f[16];
                                    #pragma unroll
                                    for (int j_1 = 0; j_1 < 16; j_1++) {
                                        float gf = (float)(__nv_bfloat16)totgf[q_3 * 16 + j_1];
                                        float uf = (float)(__nv_bfloat16)totuf[q_3 * 16 + j_1];
                                        float _exp2_58 = approx_exp2((-gf) * 1.442695f);
                                        float sigf = approx_rcp(1.0f + _exp2_58);
                                        float _tanh_59 = tanhf(gf * 0.25f);
                                        float af = 4.0f * _tanh_59 * sigf;
                                        float _tanh_60 = tanhf(uf * 0.04f);
                                        float bf = 25.0f * _tanh_60;
                                        out_f[j_1] = af * bf;
                                    }
                                    if (row_ok == 1) {
                                        unsigned int fpack_s[8];
                                        #pragma unroll
                                        for (int _lp = 0; _lp < 1; _lp++) {
                                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_f[_lp*2 + 0], out_f[_lp*2+1 + 0]));
                                            fpack_s[_lp] = *(uint32_t*)&_bf2;
                                        }
                                        #pragma unroll
                                        for (int _lp = 0; _lp < 1; _lp++) {
                                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_f[_lp*2 + 2], out_f[_lp*2+1 + 2]));
                                            fpack_s[_lp + 1] = *(uint32_t*)&_bf2;
                                        }
                                        #pragma unroll
                                        for (int _lp = 0; _lp < 1; _lp++) {
                                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_f[_lp*2 + 4], out_f[_lp*2+1 + 4]));
                                            fpack_s[_lp + 2] = *(uint32_t*)&_bf2;
                                        }
                                        #pragma unroll
                                        for (int _lp = 0; _lp < 1; _lp++) {
                                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_f[_lp*2 + 6], out_f[_lp*2+1 + 6]));
                                            fpack_s[_lp + 3] = *(uint32_t*)&_bf2;
                                        }
                                        #pragma unroll
                                        for (int _lp = 0; _lp < 1; _lp++) {
                                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_f[_lp*2 + 8], out_f[_lp*2+1 + 8]));
                                            fpack_s[_lp + 4] = *(uint32_t*)&_bf2;
                                        }
                                        #pragma unroll
                                        for (int _lp = 0; _lp < 1; _lp++) {
                                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_f[_lp*2 + 10], out_f[_lp*2+1 + 10]));
                                            fpack_s[_lp + 5] = *(uint32_t*)&_bf2;
                                        }
                                        #pragma unroll
                                        for (int _lp = 0; _lp < 1; _lp++) {
                                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_f[_lp*2 + 12], out_f[_lp*2+1 + 12]));
                                            fpack_s[_lp + 6] = *(uint32_t*)&_bf2;
                                        }
                                        #pragma unroll
                                        for (int _lp = 0; _lp < 1; _lp++) {
                                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_f[_lp*2 + 14], out_f[_lp*2+1 + 14]));
                                            fpack_s[_lp + 7] = *(uint32_t*)&_bf2;
                                        }
                                        {
                                            const unsigned* _raw_stv8_61 = reinterpret_cast<const unsigned*>(fpack_s);
                                            asm volatile(
                                                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                                :: "l"((void*)(shared_act + (row_out_sf + (unsigned long long)(fh_1 * 32) + (unsigned long long)(q_3 * 16)))), "r"(_raw_stv8_61[0]), "r"(_raw_stv8_61[1]), "r"(_raw_stv8_61[2]), "r"(_raw_stv8_61[3]), "r"(_raw_stv8_61[4]), "r"(_raw_stv8_61[5]), "r"(_raw_stv8_61[6]), "r"(_raw_stv8_61[7]) : "memory");
                                        }
                                    }
                                }
                            }
                        }
                        if (fo_ws == 1) {
                            asm volatile("barrier.sync 1, 256;" ::: "memory");
                            if (warp == 2) {
                                if (elect_sync()) {
                                    counters[fo_wslot] = 0;
                                }
                            }
                        }
                    }
                    if (fo_st == 0 && fo_keep == 0 && fo_fold == 0) {
                        if (cls_1 == 0) {
                            int col_base_r = bid_n_1 * BLOCK_N + col_half * B_HALF_N;
                            unsigned long long row_out_r = (unsigned long long)global_row * (unsigned long long)NUM_EXPERTS + (unsigned long long)col_base_r;
                            int store_r = ((col_base_r < NUM_EXPERTS) ? 1 : 0);
                            #pragma unroll 1
                            for (int n_chunk = 0; n_chunk < B_HALF_N / 16; n_chunk++) {
                                int col = n_chunk * 16;
                                float _tmem_load_10[16];
                                tmem_ld_x16(&_tmem_load_10[0], lane_addr + col_half * B_HALF_N + col);
                                asm volatile("tcgen05.wait::ld.sync.aligned;");
                                if (row_ok + store_r == 2) {
                                    {
                                        unsigned _stv8_62_0 = __float_as_uint(_tmem_load_10[0 + 0]);
                                        unsigned _stv8_62_1 = __float_as_uint(_tmem_load_10[0 + 1]);
                                        unsigned _stv8_62_2 = __float_as_uint(_tmem_load_10[0 + 2]);
                                        unsigned _stv8_62_3 = __float_as_uint(_tmem_load_10[0 + 3]);
                                        unsigned _stv8_62_4 = __float_as_uint(_tmem_load_10[0 + 4]);
                                        unsigned _stv8_62_5 = __float_as_uint(_tmem_load_10[0 + 5]);
                                        unsigned _stv8_62_6 = __float_as_uint(_tmem_load_10[0 + 6]);
                                        unsigned _stv8_62_7 = __float_as_uint(_tmem_load_10[0 + 7]);
                                        asm volatile(
                                            "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                            :: "l"((void*)(logits + (row_out_r + (unsigned long long)col) + (0))), "r"(_stv8_62_0), "r"(_stv8_62_1), "r"(_stv8_62_2), "r"(_stv8_62_3), "r"(_stv8_62_4), "r"(_stv8_62_5), "r"(_stv8_62_6), "r"(_stv8_62_7) : "memory");
                                    }
                                    {
                                        unsigned _stv8_63_0 = __float_as_uint(_tmem_load_10[8 + 0]);
                                        unsigned _stv8_63_1 = __float_as_uint(_tmem_load_10[8 + 1]);
                                        unsigned _stv8_63_2 = __float_as_uint(_tmem_load_10[8 + 2]);
                                        unsigned _stv8_63_3 = __float_as_uint(_tmem_load_10[8 + 3]);
                                        unsigned _stv8_63_4 = __float_as_uint(_tmem_load_10[8 + 4]);
                                        unsigned _stv8_63_5 = __float_as_uint(_tmem_load_10[8 + 5]);
                                        unsigned _stv8_63_6 = __float_as_uint(_tmem_load_10[8 + 6]);
                                        unsigned _stv8_63_7 = __float_as_uint(_tmem_load_10[8 + 7]);
                                        asm volatile(
                                            "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                            :: "l"((void*)(logits + (row_out_r + (unsigned long long)col + 8) + (0))), "r"(_stv8_63_0), "r"(_stv8_63_1), "r"(_stv8_63_2), "r"(_stv8_63_3), "r"(_stv8_63_4), "r"(_stv8_63_5), "r"(_stv8_63_6), "r"(_stv8_63_7) : "memory");
                                    }
                                }
                            }
                        }
                        if (cls_1 == 1) {
                            int col_base_l = (bid_n_1 - R_TILES) * BLOCK_N + col_half * B_HALF_N;
                            unsigned long long row_out_l = (unsigned long long)global_row * (unsigned long long)LATENT + (unsigned long long)col_base_l;
                            #pragma unroll 1
                            for (int n_chunk_1 = 0; n_chunk_1 < B_HALF_N / 16; n_chunk_1++) {
                                int col_1 = n_chunk_1 * 16;
                                float _tmem_load_11[16];
                                tmem_ld_x16(&_tmem_load_11[0], lane_addr + col_half * B_HALF_N + col_1);
                                asm volatile("tcgen05.wait::ld.sync.aligned;");
                                if (row_ok == 1) {
                                    unsigned int _packed_4[8];
                                    #pragma unroll
                                    for (int _lp = 0; _lp < 1; _lp++) {
                                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_11[_lp*2 + 0], _tmem_load_11[_lp*2+1 + 0]));
                                        _packed_4[_lp] = *(uint32_t*)&_bf2;
                                    }
                                    #pragma unroll
                                    for (int _lp = 0; _lp < 1; _lp++) {
                                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_11[_lp*2 + 2], _tmem_load_11[_lp*2+1 + 2]));
                                        _packed_4[_lp + 1] = *(uint32_t*)&_bf2;
                                    }
                                    #pragma unroll
                                    for (int _lp = 0; _lp < 1; _lp++) {
                                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_11[_lp*2 + 4], _tmem_load_11[_lp*2+1 + 4]));
                                        _packed_4[_lp + 2] = *(uint32_t*)&_bf2;
                                    }
                                    #pragma unroll
                                    for (int _lp = 0; _lp < 1; _lp++) {
                                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_11[_lp*2 + 6], _tmem_load_11[_lp*2+1 + 6]));
                                        _packed_4[_lp + 3] = *(uint32_t*)&_bf2;
                                    }
                                    #pragma unroll
                                    for (int _lp = 0; _lp < 1; _lp++) {
                                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_11[_lp*2 + 8], _tmem_load_11[_lp*2+1 + 8]));
                                        _packed_4[_lp + 4] = *(uint32_t*)&_bf2;
                                    }
                                    #pragma unroll
                                    for (int _lp = 0; _lp < 1; _lp++) {
                                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_11[_lp*2 + 10], _tmem_load_11[_lp*2+1 + 10]));
                                        _packed_4[_lp + 5] = *(uint32_t*)&_bf2;
                                    }
                                    #pragma unroll
                                    for (int _lp = 0; _lp < 1; _lp++) {
                                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_11[_lp*2 + 12], _tmem_load_11[_lp*2+1 + 12]));
                                        _packed_4[_lp + 6] = *(uint32_t*)&_bf2;
                                    }
                                    #pragma unroll
                                    for (int _lp = 0; _lp < 1; _lp++) {
                                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_11[_lp*2 + 14], _tmem_load_11[_lp*2+1 + 14]));
                                        _packed_4[_lp + 7] = *(uint32_t*)&_bf2;
                                    }
                                    {
                                        const unsigned* _raw_stv8_64 = reinterpret_cast<const unsigned*>(_packed_4);
                                        asm volatile(
                                            "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                            :: "l"((void*)(latent + (row_out_l + (unsigned long long)col_1))), "r"(_raw_stv8_64[0]), "r"(_raw_stv8_64[1]), "r"(_raw_stv8_64[2]), "r"(_raw_stv8_64[3]), "r"(_raw_stv8_64[4]), "r"(_raw_stv8_64[5]), "r"(_raw_stv8_64[6]), "r"(_raw_stv8_64[7]) : "memory");
                                    }
                                }
                            }
                        }
                        if (cls_1 == 2) {
                            int col_base_s = (bid_n_1 - R_TILES - L_TILES) * B_HALF_N + col_half * (B_HALF_N / 2);
                            unsigned long long row_out_s = (unsigned long long)global_row * (unsigned long long)I_LOCAL + (unsigned long long)col_base_s;
                            #pragma unroll 1
                            for (int n_chunk_2 = 0; n_chunk_2 < B_HALF_N / 32; n_chunk_2++) {
                                int col_2 = n_chunk_2 * 16;
                                float _tmem_load_12[16];
                                tmem_ld_x16(&_tmem_load_12[0], lane_addr + col_half * (B_HALF_N / 2) + col_2);
                                asm volatile("tcgen05.wait::ld.sync.aligned;");
                                float _tmem_load_13[16];
                                tmem_ld_x16(&_tmem_load_13[0], lane_addr + B_HALF_N + col_half * (B_HALF_N / 2) + col_2);
                                asm volatile("tcgen05.wait::ld.sync.aligned;");
                                float out_vals[16];
                                #pragma unroll
                                for (int j_1_1 = 0; j_1_1 < 16; j_1_1++) {
                                    float g = (float)(__nv_bfloat16)_tmem_load_12[j_1_1];
                                    float u = (float)(__nv_bfloat16)_tmem_load_13[j_1_1];
                                    float _exp2_65 = approx_exp2((-g) * 1.442695f);
                                    float sig = approx_rcp(1.0f + _exp2_65);
                                    float _tanh_66 = tanhf(g * 0.25f);
                                    float a = 4.0f * _tanh_66 * sig;
                                    float _tanh_67 = tanhf(u * 0.04f);
                                    float b = 25.0f * _tanh_67;
                                    out_vals[j_1_1] = a * b;
                                }
                                if (row_ok == 1) {
                                    unsigned int _packed_5[8];
                                    #pragma unroll
                                    for (int _lp = 0; _lp < 1; _lp++) {
                                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_vals[_lp*2 + 0], out_vals[_lp*2+1 + 0]));
                                        _packed_5[_lp] = *(uint32_t*)&_bf2;
                                    }
                                    #pragma unroll
                                    for (int _lp = 0; _lp < 1; _lp++) {
                                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_vals[_lp*2 + 2], out_vals[_lp*2+1 + 2]));
                                        _packed_5[_lp + 1] = *(uint32_t*)&_bf2;
                                    }
                                    #pragma unroll
                                    for (int _lp = 0; _lp < 1; _lp++) {
                                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_vals[_lp*2 + 4], out_vals[_lp*2+1 + 4]));
                                        _packed_5[_lp + 2] = *(uint32_t*)&_bf2;
                                    }
                                    #pragma unroll
                                    for (int _lp = 0; _lp < 1; _lp++) {
                                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_vals[_lp*2 + 6], out_vals[_lp*2+1 + 6]));
                                        _packed_5[_lp + 3] = *(uint32_t*)&_bf2;
                                    }
                                    #pragma unroll
                                    for (int _lp = 0; _lp < 1; _lp++) {
                                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_vals[_lp*2 + 8], out_vals[_lp*2+1 + 8]));
                                        _packed_5[_lp + 4] = *(uint32_t*)&_bf2;
                                    }
                                    #pragma unroll
                                    for (int _lp = 0; _lp < 1; _lp++) {
                                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_vals[_lp*2 + 10], out_vals[_lp*2+1 + 10]));
                                        _packed_5[_lp + 5] = *(uint32_t*)&_bf2;
                                    }
                                    #pragma unroll
                                    for (int _lp = 0; _lp < 1; _lp++) {
                                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_vals[_lp*2 + 12], out_vals[_lp*2+1 + 12]));
                                        _packed_5[_lp + 6] = *(uint32_t*)&_bf2;
                                    }
                                    #pragma unroll
                                    for (int _lp = 0; _lp < 1; _lp++) {
                                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_vals[_lp*2 + 14], out_vals[_lp*2+1 + 14]));
                                        _packed_5[_lp + 7] = *(uint32_t*)&_bf2;
                                    }
                                    {
                                        const unsigned* _raw_stv8_68 = reinterpret_cast<const unsigned*>(_packed_5);
                                        asm volatile(
                                            "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                            :: "l"((void*)(shared_act + (row_out_s + (unsigned long long)col_2))), "r"(_raw_stv8_68[0]), "r"(_raw_stv8_68[1]), "r"(_raw_stv8_68[2]), "r"(_raw_stv8_68[3]), "r"(_raw_stv8_68[4]), "r"(_raw_stv8_68[5]), "r"(_raw_stv8_68[6]), "r"(_raw_stv8_68[7]) : "memory");
                                    }
                                }
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
                }
                break;
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
