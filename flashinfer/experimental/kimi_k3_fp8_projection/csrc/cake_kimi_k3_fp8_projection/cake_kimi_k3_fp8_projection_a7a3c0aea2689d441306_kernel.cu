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
#define TMEM_NCOLS 504
#define TMEM_ACCUM_OFFSET 0
#define TMEM_SK_PRE_OFFSET 256
#define TMEM_TMEM_SFA_OFFSET 480
#define TMEM_TMEM_SFB_OFFSET 488
#define NUM_TMA_PIPE_STAGES 3
#define NUM_MAINLOOP_PIPE_STAGES 1
#define NUM_WORK_PIPE_STAGES 4
#define SMEM_SMEM_OUT_OFF 207872
#define SMEM_SMEM_OUT_STAGE_BYTES 8192
#define SMEM_SMEM_OUT_STRIDE 8192
#define SMEM_SMEM_A_OFF 1024
#define SMEM_SMEM_A_STAGE_BYTES 32768
#define SMEM_SMEM_A_STRIDE 68608
#define SMEM_SMEM_B_OFF 33792
#define SMEM_SMEM_B_STAGE_BYTES 32768
#define SMEM_SMEM_B_STRIDE 68608
#define SMEM_SMEM_SFA_ALL_OFF 66560
#define SMEM_SMEM_SFA_ALL_STAGE_BYTES 1024
#define SMEM_SMEM_SFA_ALL_STRIDE 68608
#define SMEM_SMEM_SFB_ALL_OFF 67584
#define SMEM_SMEM_SFB_ALL_STAGE_BYTES 2048
#define SMEM_SMEM_SFB_ALL_STRIDE 68608
#define SMEM_SMEM_SFA0_OFF 66560
#define SMEM_SMEM_SFA0_STAGE_BYTES 512
#define SMEM_SMEM_SFA0_STRIDE 68608
#define SMEM_SMEM_SFA1_OFF 67072
#define SMEM_SMEM_SFA1_STAGE_BYTES 512
#define SMEM_SMEM_SFA1_STRIDE 68608
#define SMEM_SMEM_SFB0_OFF 67584
#define SMEM_SMEM_SFB0_STAGE_BYTES 1024
#define SMEM_SMEM_SFB0_STRIDE 68608
#define SMEM_SMEM_SFB1_OFF 68608
#define SMEM_SMEM_SFB1_STAGE_BYTES 1024
#define SMEM_SMEM_SFB1_STRIDE 68608
#define SMEM_WORK_RESPONSE_OFF 206848
#define SMEM_WORK_RESPONSE_STAGE_BYTES 16
#define SMEM_WORK_RESPONSE_STRIDE 16
#define SMEM_TOTAL 224256
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


__device__ __forceinline__ void tcgen05_mma_mxf8_bs_cta2(
    int taddr, uint64_t a_desc, uint64_t b_desc, uint32_t i_desc,
    int sfa_taddr, int sfb_taddr, int enable_input_d) {
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, %6, 0;\n\t"
        "tcgen05.mma.cta_group::2.kind::mxf8f6f4.block_scale"
        " [%0], %1, %2, %3, [%4], [%5], p;\n\t"
        "}\n"
        :: "r"(taddr), "l"(a_desc), "l"(b_desc),
           "r"(i_desc), "r"(sfa_taddr), "r"(sfb_taddr),
           "r"(enable_input_d));
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




__device__ __forceinline__ uint64_t make_sf_cp_desc_lo_sbo128(int lo) {
    const int SBO = 128;
    return (uint64_t)(uint32_t)lo
         | (desc_encode(SBO) << 32ULL)
         | (1ULL << 46ULL);
}


__device__ __forceinline__ void tcgen05_cp_32x128b_warpx4_cta2(
    int taddr, uint64_t s_desc) {
    asm volatile(
        "tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;"
        :: "r"(taddr), "l"(s_desc));
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
kernel_cake_kimi_k3_fp8_projection_a7a3c0aea2689d441306(const __grid_constant__ CUtensorMap A, const __grid_constant__ CUtensorMap B, const __grid_constant__ CUtensorMap SFA, const __grid_constant__ CUtensorMap SFB, __nv_bfloat16* __restrict__ out, const __grid_constant__ CUtensorMap OUT, int M, int m_tiles, int n_tiles, int n_valid, int ldo, int store_vec, int num_k_iters, int sf_k_tiles, __nv_bfloat16* __restrict__ x, int K, float* __restrict__ sk_partials, unsigned int* __restrict__ sk_flags, int sk_pairs, int sk_rem, int sk_ksplit, int sk_dp, int sk_l, int sk_slack, int sk_b)
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
    #define mma_done_addr (mbar_base + 24)
    #define mainloop_done_addr (mbar_base + 48)
    #define epilogue_done_addr (mbar_base + 56)
    #define work_full_addr (mbar_base + 64)
    #define work_empty_addr (mbar_base + 96)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    const unsigned int clusters_x = gridDim.x / 2;
    const unsigned int cluster_id = ((blockIdx.z * gridDim.y + blockIdx.y) * clusters_x) + blockIdx.x / 2;
    const unsigned int num_clusters = clusters_x * gridDim.y * gridDim.z;

    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));

    // Kernel setup ops
    __nv_bfloat16* smem_out = reinterpret_cast<__nv_bfloat16*>(smem_raw + 207872);
    const int smem_out_addr = smem + 207872;
    uint8_t* smem_a = reinterpret_cast<uint8_t*>(smem_raw + 1024);
    const int smem_a_addr = smem + 1024;
    uint8_t* smem_b = reinterpret_cast<uint8_t*>(smem_raw + 33792);
    const int smem_b_addr = smem + 33792;
    uint8_t* smem_sfa_all = reinterpret_cast<uint8_t*>(smem_raw + 66560);
    const int smem_sfa_all_addr = smem + 66560;
    uint8_t* smem_sfb_all = reinterpret_cast<uint8_t*>(smem_raw + 67584);
    const int smem_sfb_all_addr = smem + 67584;
    uint8_t* smem_sfa0 = reinterpret_cast<uint8_t*>(smem_raw + 66560);
    const int smem_sfa0_addr = smem + 66560;
    uint8_t* smem_sfa1 = reinterpret_cast<uint8_t*>(smem_raw + 67072);
    const int smem_sfa1_addr = smem + 67072;
    uint8_t* smem_sfb0 = reinterpret_cast<uint8_t*>(smem_raw + 67584);
    const int smem_sfb0_addr = smem + 67584;
    uint8_t* smem_sfb1 = reinterpret_cast<uint8_t*>(smem_raw + 68608);
    const int smem_sfb1_addr = smem + 68608;
    unsigned int* work_response = reinterpret_cast<unsigned int*>(smem_raw + 206848);
    const int work_response_addr = smem + 206848;

    // Mbarrier init (6 pipeline groups, 0 ordered-sequence groups, 16 barriers)
    // Mbarriers at smem_raw[0..128)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'tma_pipe' ---
            // tma_full: 3 barriers, init_count=2
            mbarrier_init(smem + 0, 2);
            mbarrier_init(smem + 8, 2);
            mbarrier_init(smem + 16, 2);
            // mma_done: 3 barriers, init_count=1
            mbarrier_init(smem + 24, 1);
            mbarrier_init(smem + 32, 1);
            mbarrier_init(smem + 40, 1);
            // --- pipeline 'mainloop_pipe' ---
            // mainloop_done: 1 barriers, init_count=1
            mbarrier_init(smem + 48, 1);
            // epilogue_done: 1 barriers, init_count=16
            mbarrier_init(smem + 56, 16);
            // --- pipeline 'work_pipe' ---
            // work_full: 4 barriers, init_count=1
            mbarrier_init(smem + 64, 1);
            mbarrier_init(smem + 72, 1);
            mbarrier_init(smem + 80, 1);
            mbarrier_init(smem + 88, 1);
            // work_empty: 4 barriers, init_count=546
            mbarrier_init(smem + 96, 546);
            mbarrier_init(smem + 104, 546);
            mbarrier_init(smem + 112, 546);
            mbarrier_init(smem + 120, 546);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();
    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");

    // TMEM alloc (512 columns, 504 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 128);
    if (warp == 0) {
        int _tmem_hold = smem + 128;
        asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(512) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_accum = taddr;
    const int tmem_sk_pre = taddr + 256;
    const int tmem_tmem_sfa = taddr + 480;
    const int tmem_tmem_sfb = taddr + 488;

    // ---- Role: load ----
    if (warp == 0) {
        { // load_main
            int num_cluster_tiles = m_tiles / CTA_GROUP * n_tiles;
            unsigned int load_stage = 0;
            unsigned int work_stage = 0;
            asm volatile("griddepcontrol.wait;" ::: "memory");
            unsigned int _phase_mma_done = 1;
            unsigned int _phase_work_empty = 1;
            unsigned int _phase_work_full = 0;
            if (elect_sync()) {
                unsigned int this_bid = bid;
                int skp_p = (int)(bid / 2);
                int skp_early = 0;
                int skp_et = 0;
                int skp_ek0 = 0;
                int skp_ek1 = 0;
                int skp_eord = 0;
                int skp_late = 0;
                int skp_lt = 0;
                int skp_lk0 = 0;
                int skp_lnc = 0;
                int skp_cap = 2147483647;
                if (sk_dp > 0) {
                    int skp_ne = sk_rem * (sk_l - 1);
                    if (skp_p < skp_ne) {
                        skp_early = 1;
                        skp_et = skp_p / (sk_l - 1);
                        skp_eord = skp_p % (sk_l - 1);
                        skp_ek0 = skp_eord * num_k_iters / sk_l;
                        skp_ek1 = (skp_eord + 1) * num_k_iters / sk_l;
                    }
                    if (skp_p >= sk_pairs - sk_rem && skp_p < sk_pairs) {
                        skp_late = 1;
                        skp_lt = sk_pairs - 1 - skp_p;
                        skp_lk0 = (sk_l - 1) * num_k_iters / sk_l;
                        skp_lnc = sk_l - 1;
                        skp_cap = sk_slack;
                    }
                } else {
                    int skp_g = skp_p / sk_b;
                    int skp_j = skp_p % sk_b;
                    int skp_total = sk_rem * num_k_iters;
                    int skp_lo = 0;
                    int skp_hi = 0;
                    int skp_own = 0;
                    int skp_tl = 0;
                    #pragma unroll 1
                    for (int skp_jj = 0; skp_jj < skp_j + 1; skp_jj++) {
                        skp_lo = skp_hi;
                        int skp_t = skp_lo / num_k_iters;
                        if (skp_t > skp_tl) {
                            skp_own = ((skp_lo == skp_t * num_k_iters) ? skp_jj : skp_jj - 1);
                        }
                        skp_tl = skp_t;
                        int skp_tend = (skp_t + 1) * num_k_iters;
                        int skp_len = ((skp_tend <= skp_lo + sk_l) ? sk_l : sk_l - sk_slack);
                        int _min_0 = ((skp_lo + skp_len) < (skp_total) ? (skp_lo + skp_len) : (skp_total));
                        skp_hi = _min_0;
                    }
                    int skp_th = (skp_hi - 1) / num_k_iters;
                    int skp_tbase = skp_g * sk_rem;
                    if (skp_th > skp_tl) {
                        skp_early = 1;
                        skp_et = skp_tbase + skp_th;
                        skp_ek1 = skp_hi - skp_th * num_k_iters;
                        skp_late = 1;
                        skp_lt = skp_tbase + skp_tl;
                        skp_lk0 = skp_lo - skp_tl * num_k_iters;
                        skp_lnc = skp_j - skp_own;
                    } else if (skp_hi == (skp_tl + 1) * num_k_iters) {
                        skp_late = 1;
                        skp_lt = skp_tbase + skp_tl;
                        skp_lk0 = skp_lo - skp_tl * num_k_iters;
                        skp_lnc = skp_j - skp_own;
                    } else {
                        skp_early = 1;
                        skp_et = skp_tbase + skp_tl;
                        skp_ek0 = skp_lo - skp_tl * num_k_iters;
                        skp_ek1 = skp_hi - skp_tl * num_k_iters;
                        skp_eord = skp_j - skp_own;
                    }
                }
                if (skp_early == 1) {
                    unsigned int skp_ebid = (unsigned int)((sk_dp + skp_et) * 2) + (unsigned int)cta_rank;
                    int tiles_per_group = GROUP_M * n_tiles;
                    int group = skp_ebid / (unsigned int)tiles_per_group;
                    int first_m = group * GROUP_M;
                    int remaining = m_tiles - first_m;
                    int group_size = ((remaining >= GROUP_M) ? GROUP_M : remaining);
                    int local = skp_ebid % (unsigned int)tiles_per_group;
                    int bid_m = first_m + local % group_size;
                    int bid_n = local / group_size;
                    int off_m = bid_m * BLOCK_M;
                    int off_n = bid_n * 256;
                    int weight_tile0 = (off_n + cta_rank * 128) / 128 * (2 * num_k_iters);
                    int sfa_tile_row = bid_m * sf_k_tiles;
                    int sfb_tile_row = bid_n * sf_k_tiles;
                    #pragma unroll 1
                    for (int iter_k = skp_ek0; iter_k < skp_ek1; iter_k++) {
                        mbarrier_wait(mma_done_addr + (load_stage) * 8, _phase_mma_done);
                        int k_group = iter_k * 2;
                        tma_3d_gmem2smem_cta2(smem_a_addr + load_stage * 68608, (&A), 0, off_m, k_group, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                        tma_3d_gmem2smem_cta2(smem_sfa_all_addr + load_stage * 68608, (&SFA), 0, 0, sfa_tile_row + k_group, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                        tma_3d_gmem2smem_cta2(smem_b_addr + load_stage * 68608, (&B), 0, 0, weight_tile0 + k_group, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                        tma_3d_gmem2smem_cta2(smem_sfb_all_addr + load_stage * 68608, (&SFB), 0, 0, sfb_tile_row + k_group, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                        asm volatile(
                            "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                            :: "r"((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF), "r"((uint32_t)(68608)) : "memory");
                        load_stage += 1;
                        if (load_stage == 3) { load_stage = 0; _phase_mma_done ^= 1; }
                    }
                }
                if (sk_dp > 0) {
                    #pragma unroll 1
                    for (int skp_done = 0; skp_done < num_cluster_tiles; skp_done++) {
                        int skp_stop = ((skp_done + 1 == skp_cap) ? 1 : 0);
                        if (cta_rank == 0) {
                            if (skp_stop == 0) {
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
                        }
                        int tiles_per_group_1 = GROUP_M * n_tiles;
                        int group_1 = this_bid / (unsigned int)tiles_per_group_1;
                        int first_m_1 = group_1 * GROUP_M;
                        int remaining_1 = m_tiles - first_m_1;
                        int group_size_1 = ((remaining_1 >= GROUP_M) ? GROUP_M : remaining_1);
                        int local_1 = this_bid % (unsigned int)tiles_per_group_1;
                        int bid_m_1 = first_m_1 + local_1 % group_size_1;
                        int bid_n_1 = local_1 / group_size_1;
                        int off_m_1 = bid_m_1 * BLOCK_M;
                        int off_n_1 = bid_n_1 * 256;
                        int weight_tile0_1 = (off_n_1 + cta_rank * 128) / 128 * (2 * num_k_iters);
                        int sfa_tile_row_1 = bid_m_1 * sf_k_tiles;
                        int sfb_tile_row_1 = bid_n_1 * sf_k_tiles;
                        #pragma unroll 1
                        for (int iter_k_1 = 0; iter_k_1 < num_k_iters; iter_k_1++) {
                            mbarrier_wait(mma_done_addr + (load_stage) * 8, _phase_mma_done);
                            int k_group_1 = iter_k_1 * 2;
                            tma_3d_gmem2smem_cta2(smem_a_addr + load_stage * 68608, (&A), 0, off_m_1, k_group_1, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                            tma_3d_gmem2smem_cta2(smem_sfa_all_addr + load_stage * 68608, (&SFA), 0, 0, sfa_tile_row_1 + k_group_1, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                            tma_3d_gmem2smem_cta2(smem_b_addr + load_stage * 68608, (&B), 0, 0, weight_tile0_1 + k_group_1, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                            tma_3d_gmem2smem_cta2(smem_sfb_all_addr + load_stage * 68608, (&SFB), 0, 0, sfb_tile_row_1 + k_group_1, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                            asm volatile(
                                "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                                :: "r"((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF), "r"((uint32_t)(68608)) : "memory");
                            load_stage += 1;
                            if (load_stage == 3) { load_stage = 0; _phase_mma_done ^= 1; }
                        }
                        if (skp_stop == 1) {
                            break;
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
                }
                if (skp_late == 1) {
                    unsigned int skp_lbid = (unsigned int)((sk_dp + skp_lt) * 2) + (unsigned int)cta_rank;
                    int tiles_per_group_2 = GROUP_M * n_tiles;
                    int group_2 = skp_lbid / (unsigned int)tiles_per_group_2;
                    int first_m_2 = group_2 * GROUP_M;
                    int remaining_2 = m_tiles - first_m_2;
                    int group_size_2 = ((remaining_2 >= GROUP_M) ? GROUP_M : remaining_2);
                    int local_2 = skp_lbid % (unsigned int)tiles_per_group_2;
                    int bid_m_2 = first_m_2 + local_2 % group_size_2;
                    int bid_n_2 = local_2 / group_size_2;
                    int off_m_2 = bid_m_2 * BLOCK_M;
                    int off_n_2 = bid_n_2 * 256;
                    int weight_tile0_2 = (off_n_2 + cta_rank * 128) / 128 * (2 * num_k_iters);
                    int sfa_tile_row_2 = bid_m_2 * sf_k_tiles;
                    int sfb_tile_row_2 = bid_n_2 * sf_k_tiles;
                    #pragma unroll 1
                    for (int iter_k_2 = skp_lk0; iter_k_2 < num_k_iters; iter_k_2++) {
                        mbarrier_wait(mma_done_addr + (load_stage) * 8, _phase_mma_done);
                        int k_group_2 = iter_k_2 * 2;
                        tma_3d_gmem2smem_cta2(smem_a_addr + load_stage * 68608, (&A), 0, off_m_2, k_group_2, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                        tma_3d_gmem2smem_cta2(smem_sfa_all_addr + load_stage * 68608, (&SFA), 0, 0, sfa_tile_row_2 + k_group_2, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                        tma_3d_gmem2smem_cta2(smem_b_addr + load_stage * 68608, (&B), 0, 0, weight_tile0_2 + k_group_2, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                        tma_3d_gmem2smem_cta2(smem_sfb_all_addr + load_stage * 68608, (&SFB), 0, 0, sfb_tile_row_2 + k_group_2, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                        asm volatile(
                            "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                            :: "r"((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF), "r"((uint32_t)(68608)) : "memory");
                        load_stage += 1;
                        if (load_stage == 3) { load_stage = 0; _phase_mma_done ^= 1; }
                    }
                }
                asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
                #pragma unroll 1
                for (int _ = 0; _ < 3; _++) {
                    mbarrier_wait(mma_done_addr + (load_stage) * 8, _phase_mma_done);
                    load_stage += 1;
                    if (load_stage == 3) { load_stage = 0; _phase_mma_done ^= 1; }
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
                int skp_p_1 = (int)(bid / 2);
                int skp_early_1 = 0;
                int skp_et_1 = 0;
                int skp_ek0_1 = 0;
                int skp_ek1_1 = 0;
                int skp_eord_1 = 0;
                int skp_late_1 = 0;
                int skp_lt_1 = 0;
                int skp_lk0_1 = 0;
                int skp_lnc_1 = 0;
                int skp_cap_1 = 2147483647;
                if (sk_dp > 0) {
                    int skp_ne_1 = sk_rem * (sk_l - 1);
                    if (skp_p_1 < skp_ne_1) {
                        skp_early_1 = 1;
                        skp_et_1 = skp_p_1 / (sk_l - 1);
                        skp_eord_1 = skp_p_1 % (sk_l - 1);
                        skp_ek0_1 = skp_eord_1 * num_k_iters / sk_l;
                        skp_ek1_1 = (skp_eord_1 + 1) * num_k_iters / sk_l;
                    }
                    if (skp_p_1 >= sk_pairs - sk_rem && skp_p_1 < sk_pairs) {
                        skp_late_1 = 1;
                        skp_lt_1 = sk_pairs - 1 - skp_p_1;
                        skp_lk0_1 = (sk_l - 1) * num_k_iters / sk_l;
                        skp_lnc_1 = sk_l - 1;
                        skp_cap_1 = sk_slack;
                    }
                } else {
                    int skp_g_1 = skp_p_1 / sk_b;
                    int skp_j_1 = skp_p_1 % sk_b;
                    int skp_total_1 = sk_rem * num_k_iters;
                    int skp_lo_1 = 0;
                    int skp_hi_1 = 0;
                    int skp_own_1 = 0;
                    int skp_tl_1 = 0;
                    #pragma unroll 1
                    for (int skp_jj_1 = 0; skp_jj_1 < skp_j_1 + 1; skp_jj_1++) {
                        skp_lo_1 = skp_hi_1;
                        int skp_t_1 = skp_lo_1 / num_k_iters;
                        if (skp_t_1 > skp_tl_1) {
                            skp_own_1 = ((skp_lo_1 == skp_t_1 * num_k_iters) ? skp_jj_1 : skp_jj_1 - 1);
                        }
                        skp_tl_1 = skp_t_1;
                        int skp_tend_1 = (skp_t_1 + 1) * num_k_iters;
                        int skp_len_1 = ((skp_tend_1 <= skp_lo_1 + sk_l) ? sk_l : sk_l - sk_slack);
                        int _min_0_1 = ((skp_lo_1 + skp_len_1) < (skp_total_1) ? (skp_lo_1 + skp_len_1) : (skp_total_1));
                        skp_hi_1 = _min_0_1;
                    }
                    int skp_th_1 = (skp_hi_1 - 1) / num_k_iters;
                    int skp_tbase_1 = skp_g_1 * sk_rem;
                    if (skp_th_1 > skp_tl_1) {
                        skp_early_1 = 1;
                        skp_et_1 = skp_tbase_1 + skp_th_1;
                        skp_ek1_1 = skp_hi_1 - skp_th_1 * num_k_iters;
                        skp_late_1 = 1;
                        skp_lt_1 = skp_tbase_1 + skp_tl_1;
                        skp_lk0_1 = skp_lo_1 - skp_tl_1 * num_k_iters;
                        skp_lnc_1 = skp_j_1 - skp_own_1;
                    } else if (skp_hi_1 == (skp_tl_1 + 1) * num_k_iters) {
                        skp_late_1 = 1;
                        skp_lt_1 = skp_tbase_1 + skp_tl_1;
                        skp_lk0_1 = skp_lo_1 - skp_tl_1 * num_k_iters;
                        skp_lnc_1 = skp_j_1 - skp_own_1;
                    } else {
                        skp_early_1 = 1;
                        skp_et_1 = skp_tbase_1 + skp_tl_1;
                        skp_ek0_1 = skp_lo_1 - skp_tl_1 * num_k_iters;
                        skp_ek1_1 = skp_hi_1 - skp_tl_1 * num_k_iters;
                        skp_eord_1 = skp_j_1 - skp_own_1;
                    }
                }
                if (skp_early_1 == 1) {
                    mbarrier_wait(epilogue_done_addr, _phase_epilogue_done);
                    #pragma unroll 1
                    for (int iter_k_3 = skp_ek0_1; iter_k_3 < skp_ek1_1; iter_k_3++) {
                        mbarrier_wait(tma_full_addr + (mma_tma_stage) * 8, _phase_tma_full);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int init_flag = ((iter_k_3 == skp_ek0_1) ? 1 : 0);
                        if (elect_sync()) {
                            init_flag = init_flag;
                            tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfa, make_sf_cp_desc_lo_sbo128((((smem_sfa0_addr) >> 4) + (mma_tma_stage) * 4288)));
                            tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfb, make_sf_cp_desc_lo_sbo128((((smem_sfb0_addr) >> 4) + (mma_tma_stage) * 4288)));
                            tcgen05_cp_32x128b_warpx4_cta2((tmem_tmem_sfb + 4), make_sf_cp_desc_lo_sbo128((((smem_sfb0_addr) >> 4) + (mma_tma_stage) * 4288 + 32)));
                            tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfa + 4, make_sf_cp_desc_lo_sbo128((((smem_sfa1_addr) >> 4) + (mma_tma_stage) * 4288)));
                            tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfb + 8, make_sf_cp_desc_lo_sbo128((((smem_sfb1_addr) >> 4) + (mma_tma_stage) * 4288)));
                            tcgen05_cp_32x128b_warpx4_cta2((tmem_tmem_sfb + 8 + 4), make_sf_cp_desc_lo_sbo128((((smem_sfb1_addr) >> 4) + (mma_tma_stage) * 4288 + 32)));
                            int _smem_desc_lo_1 = (((smem_a_addr) >> 4) & 0x3FFF) + (mma_tma_stage) * 4288;
                            int _smem_desc_lo_2 = (((smem_b_addr) >> 4) & 0x3FFF) + (mma_tma_stage) * 4288;
                            {
                                uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_1) | ((uint64_t)0x40004040 << 32);
                                uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_2) | ((uint64_t)0x40004040 << 32);

                                tcgen05_mma_mxf8_bs_cta2(tmem_accum, a_desc + 0, b_desc + 0,
                                    0x10c00000U, tmem_tmem_sfa, tmem_tmem_sfb, ((init_flag) ? 0 : 1));
                                tcgen05_mma_mxf8_bs_cta2(tmem_accum, a_desc + 2, b_desc + 2,
                                    0x30c00010U, tmem_tmem_sfa, tmem_tmem_sfb, 1);
                                tcgen05_mma_mxf8_bs_cta2(tmem_accum, a_desc + 4, b_desc + 4,
                                    0x50c00020U, tmem_tmem_sfa, tmem_tmem_sfb, 1);
                                tcgen05_mma_mxf8_bs_cta2(tmem_accum, a_desc + 6, b_desc + 6,
                                    0x70c00030U, tmem_tmem_sfa, tmem_tmem_sfb, 1);
                            }
                            int _smem_desc_lo_3 = (((smem_a_addr + 16384) >> 4) & 0x3FFF) + (mma_tma_stage) * 4288;
                            int _smem_desc_lo_4 = (((smem_b_addr + 16384) >> 4) & 0x3FFF) + (mma_tma_stage) * 4288;
                            {
                                uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_3) | ((uint64_t)0x40004040 << 32);
                                uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_4) | ((uint64_t)0x40004040 << 32);

                                tcgen05_mma_mxf8_bs_cta2(tmem_accum, a_desc + 0, b_desc + 0,
                                    0x10c00000U, tmem_tmem_sfa + 4, tmem_tmem_sfb + 8, 1);
                                tcgen05_mma_mxf8_bs_cta2(tmem_accum, a_desc + 2, b_desc + 2,
                                    0x30c00010U, tmem_tmem_sfa + 4, tmem_tmem_sfb + 8, 1);
                                tcgen05_mma_mxf8_bs_cta2(tmem_accum, a_desc + 4, b_desc + 4,
                                    0x50c00020U, tmem_tmem_sfa + 4, tmem_tmem_sfb + 8, 1);
                                tcgen05_mma_mxf8_bs_cta2(tmem_accum, a_desc + 6, b_desc + 6,
                                    0x70c00030U, tmem_tmem_sfa + 4, tmem_tmem_sfb + 8, 1);
                            }
                        }
                        elect_commit_cg2_multicast(mma_done_addr + (mma_tma_stage) * 8, (uint16_t)(3));
                        mma_tma_stage += 1;
                        if (mma_tma_stage == 3) { mma_tma_stage = 0; _phase_tma_full ^= 1; }
                    }
                    elect_commit_cg2_multicast(mainloop_done_addr, (uint16_t)(3));
                    _phase_epilogue_done ^= 1;
                }
                if (sk_dp > 0) {
                    #pragma unroll 1
                    for (int skp_done_1 = 0; skp_done_1 < num_cluster_tiles_1; skp_done_1++) {
                        mbarrier_wait(epilogue_done_addr, _phase_epilogue_done);
                        #pragma unroll 1
                        for (int iter_k_4 = 0; iter_k_4 < num_k_iters; iter_k_4++) {
                            mbarrier_wait(tma_full_addr + (mma_tma_stage) * 8, _phase_tma_full);
                            asm volatile("tcgen05.fence::after_thread_sync;");
                            int init_flag_1 = ((iter_k_4 == 0) ? 1 : 0);
                            if (elect_sync()) {
                                init_flag_1 = init_flag_1;
                                tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfa, make_sf_cp_desc_lo_sbo128((((smem_sfa0_addr) >> 4) + (mma_tma_stage) * 4288)));
                                tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfb, make_sf_cp_desc_lo_sbo128((((smem_sfb0_addr) >> 4) + (mma_tma_stage) * 4288)));
                                tcgen05_cp_32x128b_warpx4_cta2((tmem_tmem_sfb + 4), make_sf_cp_desc_lo_sbo128((((smem_sfb0_addr) >> 4) + (mma_tma_stage) * 4288 + 32)));
                                tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfa + 4, make_sf_cp_desc_lo_sbo128((((smem_sfa1_addr) >> 4) + (mma_tma_stage) * 4288)));
                                tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfb + 8, make_sf_cp_desc_lo_sbo128((((smem_sfb1_addr) >> 4) + (mma_tma_stage) * 4288)));
                                tcgen05_cp_32x128b_warpx4_cta2((tmem_tmem_sfb + 8 + 4), make_sf_cp_desc_lo_sbo128((((smem_sfb1_addr) >> 4) + (mma_tma_stage) * 4288 + 32)));
                                int _smem_desc_lo_5 = (((smem_a_addr) >> 4) & 0x3FFF) + (mma_tma_stage) * 4288;
                                int _smem_desc_lo_6 = (((smem_b_addr) >> 4) & 0x3FFF) + (mma_tma_stage) * 4288;
                                {
                                    uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_5) | ((uint64_t)0x40004040 << 32);
                                    uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_6) | ((uint64_t)0x40004040 << 32);

                                    tcgen05_mma_mxf8_bs_cta2(tmem_accum, a_desc + 0, b_desc + 0,
                                        0x10c00000U, tmem_tmem_sfa, tmem_tmem_sfb, ((init_flag_1) ? 0 : 1));
                                    tcgen05_mma_mxf8_bs_cta2(tmem_accum, a_desc + 2, b_desc + 2,
                                        0x30c00010U, tmem_tmem_sfa, tmem_tmem_sfb, 1);
                                    tcgen05_mma_mxf8_bs_cta2(tmem_accum, a_desc + 4, b_desc + 4,
                                        0x50c00020U, tmem_tmem_sfa, tmem_tmem_sfb, 1);
                                    tcgen05_mma_mxf8_bs_cta2(tmem_accum, a_desc + 6, b_desc + 6,
                                        0x70c00030U, tmem_tmem_sfa, tmem_tmem_sfb, 1);
                                }
                                int _smem_desc_lo_7 = (((smem_a_addr + 16384) >> 4) & 0x3FFF) + (mma_tma_stage) * 4288;
                                int _smem_desc_lo_8 = (((smem_b_addr + 16384) >> 4) & 0x3FFF) + (mma_tma_stage) * 4288;
                                {
                                    uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_7) | ((uint64_t)0x40004040 << 32);
                                    uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_8) | ((uint64_t)0x40004040 << 32);

                                    tcgen05_mma_mxf8_bs_cta2(tmem_accum, a_desc + 0, b_desc + 0,
                                        0x10c00000U, tmem_tmem_sfa + 4, tmem_tmem_sfb + 8, 1);
                                    tcgen05_mma_mxf8_bs_cta2(tmem_accum, a_desc + 2, b_desc + 2,
                                        0x30c00010U, tmem_tmem_sfa + 4, tmem_tmem_sfb + 8, 1);
                                    tcgen05_mma_mxf8_bs_cta2(tmem_accum, a_desc + 4, b_desc + 4,
                                        0x50c00020U, tmem_tmem_sfa + 4, tmem_tmem_sfb + 8, 1);
                                    tcgen05_mma_mxf8_bs_cta2(tmem_accum, a_desc + 6, b_desc + 6,
                                        0x70c00030U, tmem_tmem_sfa + 4, tmem_tmem_sfb + 8, 1);
                                }
                            }
                            elect_commit_cg2_multicast(mma_done_addr + (mma_tma_stage) * 8, (uint16_t)(3));
                            mma_tma_stage += 1;
                            if (mma_tma_stage == 3) { mma_tma_stage = 0; _phase_tma_full ^= 1; }
                        }
                        elect_commit_cg2_multicast(mainloop_done_addr, (uint16_t)(3));
                        _phase_epilogue_done ^= 1;
                        if (skp_done_1 + 1 == skp_cap_1) {
                            break;
                        }
                        mbarrier_wait(work_full_addr + (work_stage_1) * 8, _phase_work_full_1);
                        uint32_t _clc_valid_1 = 0;
                        uint32_t _clc_ctaid_1;
                        uint32_t _clc_ctaid_y_1;
                        uint32_t _clc_ctaid_z_1;
                        asm volatile(
                            "{\n\t"
                            ".reg .pred p1;\n\t"
                            ".reg .b128 clc_r;\n\t"
                            "ld.shared.b128 clc_r, [%4];\n\t"
                            "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                            "selp.u32 %3, 1, 0, p1;\n\t"
                            "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid.v4.b32.b128 {%0, %1, %2, _}, clc_r;\n\t"
                            "}\n"
                            : "=r"(_clc_ctaid_1), "=r"(_clc_ctaid_y_1), "=r"(_clc_ctaid_z_1), "=r"(_clc_valid_1)
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
                        if (_clc_valid_1 == 0) {
                            break;
                        }
                    }
                }
                if (skp_late_1 == 1) {
                    mbarrier_wait(epilogue_done_addr, _phase_epilogue_done);
                    #pragma unroll 1
                    for (int iter_k_5 = skp_lk0_1; iter_k_5 < num_k_iters; iter_k_5++) {
                        mbarrier_wait(tma_full_addr + (mma_tma_stage) * 8, _phase_tma_full);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int init_flag_2 = ((iter_k_5 == skp_lk0_1) ? 1 : 0);
                        if (elect_sync()) {
                            init_flag_2 = init_flag_2;
                            tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfa, make_sf_cp_desc_lo_sbo128((((smem_sfa0_addr) >> 4) + (mma_tma_stage) * 4288)));
                            tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfb, make_sf_cp_desc_lo_sbo128((((smem_sfb0_addr) >> 4) + (mma_tma_stage) * 4288)));
                            tcgen05_cp_32x128b_warpx4_cta2((tmem_tmem_sfb + 4), make_sf_cp_desc_lo_sbo128((((smem_sfb0_addr) >> 4) + (mma_tma_stage) * 4288 + 32)));
                            tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfa + 4, make_sf_cp_desc_lo_sbo128((((smem_sfa1_addr) >> 4) + (mma_tma_stage) * 4288)));
                            tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfb + 8, make_sf_cp_desc_lo_sbo128((((smem_sfb1_addr) >> 4) + (mma_tma_stage) * 4288)));
                            tcgen05_cp_32x128b_warpx4_cta2((tmem_tmem_sfb + 8 + 4), make_sf_cp_desc_lo_sbo128((((smem_sfb1_addr) >> 4) + (mma_tma_stage) * 4288 + 32)));
                            int _smem_desc_lo_9 = (((smem_a_addr) >> 4) & 0x3FFF) + (mma_tma_stage) * 4288;
                            int _smem_desc_lo_10 = (((smem_b_addr) >> 4) & 0x3FFF) + (mma_tma_stage) * 4288;
                            {
                                uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_9) | ((uint64_t)0x40004040 << 32);
                                uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_10) | ((uint64_t)0x40004040 << 32);

                                tcgen05_mma_mxf8_bs_cta2(tmem_accum, a_desc + 0, b_desc + 0,
                                    0x10c00000U, tmem_tmem_sfa, tmem_tmem_sfb, ((init_flag_2) ? 0 : 1));
                                tcgen05_mma_mxf8_bs_cta2(tmem_accum, a_desc + 2, b_desc + 2,
                                    0x30c00010U, tmem_tmem_sfa, tmem_tmem_sfb, 1);
                                tcgen05_mma_mxf8_bs_cta2(tmem_accum, a_desc + 4, b_desc + 4,
                                    0x50c00020U, tmem_tmem_sfa, tmem_tmem_sfb, 1);
                                tcgen05_mma_mxf8_bs_cta2(tmem_accum, a_desc + 6, b_desc + 6,
                                    0x70c00030U, tmem_tmem_sfa, tmem_tmem_sfb, 1);
                            }
                            int _smem_desc_lo_11 = (((smem_a_addr + 16384) >> 4) & 0x3FFF) + (mma_tma_stage) * 4288;
                            int _smem_desc_lo_12 = (((smem_b_addr + 16384) >> 4) & 0x3FFF) + (mma_tma_stage) * 4288;
                            {
                                uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_11) | ((uint64_t)0x40004040 << 32);
                                uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_12) | ((uint64_t)0x40004040 << 32);

                                tcgen05_mma_mxf8_bs_cta2(tmem_accum, a_desc + 0, b_desc + 0,
                                    0x10c00000U, tmem_tmem_sfa + 4, tmem_tmem_sfb + 8, 1);
                                tcgen05_mma_mxf8_bs_cta2(tmem_accum, a_desc + 2, b_desc + 2,
                                    0x30c00010U, tmem_tmem_sfa + 4, tmem_tmem_sfb + 8, 1);
                                tcgen05_mma_mxf8_bs_cta2(tmem_accum, a_desc + 4, b_desc + 4,
                                    0x50c00020U, tmem_tmem_sfa + 4, tmem_tmem_sfb + 8, 1);
                                tcgen05_mma_mxf8_bs_cta2(tmem_accum, a_desc + 6, b_desc + 6,
                                    0x70c00030U, tmem_tmem_sfa + 4, tmem_tmem_sfb + 8, 1);
                            }
                        }
                        elect_commit_cg2_multicast(mma_done_addr + (mma_tma_stage) * 8, (uint16_t)(3));
                        mma_tma_stage += 1;
                        if (mma_tma_stage == 3) { mma_tma_stage = 0; _phase_tma_full ^= 1; }
                    }
                    elect_commit_cg2_multicast(mainloop_done_addr, (uint16_t)(3));
                    _phase_epilogue_done ^= 1;
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
            int sk_lane_addr = taddr + (unsigned int)(epi_warp * 32 << 16) + (unsigned int)(col_part * 128);
            int skp_rank = (int)cta_rank;
            int skp_wvec = (skp_rank * 8 + (warp - 2)) * 32;
            int skp_pre_addr = taddr + (unsigned int)(epi_warp * 32 << 16) + (unsigned int)(256 + col_part * 112);
            int skp_p_2 = (int)(bid / 2);
            int skp_early_2 = 0;
            int skp_et_2 = 0;
            int skp_ek0_2 = 0;
            int skp_ek1_2 = 0;
            int skp_eord_2 = 0;
            int skp_late_2 = 0;
            int skp_lt_2 = 0;
            int skp_lk0_2 = 0;
            int skp_lnc_2 = 0;
            int skp_cap_2 = 2147483647;
            if (sk_dp > 0) {
                int skp_ne_2 = sk_rem * (sk_l - 1);
                if (skp_p_2 < skp_ne_2) {
                    skp_early_2 = 1;
                    skp_et_2 = skp_p_2 / (sk_l - 1);
                    skp_eord_2 = skp_p_2 % (sk_l - 1);
                    skp_ek0_2 = skp_eord_2 * num_k_iters / sk_l;
                    skp_ek1_2 = (skp_eord_2 + 1) * num_k_iters / sk_l;
                }
                if (skp_p_2 >= sk_pairs - sk_rem && skp_p_2 < sk_pairs) {
                    skp_late_2 = 1;
                    skp_lt_2 = sk_pairs - 1 - skp_p_2;
                    skp_lk0_2 = (sk_l - 1) * num_k_iters / sk_l;
                    skp_lnc_2 = sk_l - 1;
                    skp_cap_2 = sk_slack;
                }
            } else {
                int skp_g_2 = skp_p_2 / sk_b;
                int skp_j_2 = skp_p_2 % sk_b;
                int skp_total_2 = sk_rem * num_k_iters;
                int skp_lo_2 = 0;
                int skp_hi_2 = 0;
                int skp_own_2 = 0;
                int skp_tl_2 = 0;
                #pragma unroll 1
                for (int skp_jj_2 = 0; skp_jj_2 < skp_j_2 + 1; skp_jj_2++) {
                    skp_lo_2 = skp_hi_2;
                    int skp_t_2 = skp_lo_2 / num_k_iters;
                    if (skp_t_2 > skp_tl_2) {
                        skp_own_2 = ((skp_lo_2 == skp_t_2 * num_k_iters) ? skp_jj_2 : skp_jj_2 - 1);
                    }
                    skp_tl_2 = skp_t_2;
                    int skp_tend_2 = (skp_t_2 + 1) * num_k_iters;
                    int skp_len_2 = ((skp_tend_2 <= skp_lo_2 + sk_l) ? sk_l : sk_l - sk_slack);
                    int _min_0_2 = ((skp_lo_2 + skp_len_2) < (skp_total_2) ? (skp_lo_2 + skp_len_2) : (skp_total_2));
                    skp_hi_2 = _min_0_2;
                }
                int skp_th_2 = (skp_hi_2 - 1) / num_k_iters;
                int skp_tbase_2 = skp_g_2 * sk_rem;
                if (skp_th_2 > skp_tl_2) {
                    skp_early_2 = 1;
                    skp_et_2 = skp_tbase_2 + skp_th_2;
                    skp_ek1_2 = skp_hi_2 - skp_th_2 * num_k_iters;
                    skp_late_2 = 1;
                    skp_lt_2 = skp_tbase_2 + skp_tl_2;
                    skp_lk0_2 = skp_lo_2 - skp_tl_2 * num_k_iters;
                    skp_lnc_2 = skp_j_2 - skp_own_2;
                } else if (skp_hi_2 == (skp_tl_2 + 1) * num_k_iters) {
                    skp_late_2 = 1;
                    skp_lt_2 = skp_tbase_2 + skp_tl_2;
                    skp_lk0_2 = skp_lo_2 - skp_tl_2 * num_k_iters;
                    skp_lnc_2 = skp_j_2 - skp_own_2;
                } else {
                    skp_early_2 = 1;
                    skp_et_2 = skp_tbase_2 + skp_tl_2;
                    skp_ek0_2 = skp_lo_2 - skp_tl_2 * num_k_iters;
                    skp_ek1_2 = skp_hi_2 - skp_tl_2 * num_k_iters;
                    skp_eord_2 = skp_j_2 - skp_own_2;
                }
            }
            unsigned int _phase_mainloop_done = 0;
            if (skp_early_2 == 1) {
                int skp_eslot = skp_et_2 * sk_ksplit + skp_eord_2;
                mbarrier_wait(mainloop_done_addr, _phase_mainloop_done);
                asm volatile("tcgen05.fence::after_thread_sync;");
                float _tmem_load_0[128];
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x64.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, [%64];"
                    : "=f"(_tmem_load_0[0]), "=f"(_tmem_load_0[1]), "=f"(_tmem_load_0[2]), "=f"(_tmem_load_0[3]), "=f"(_tmem_load_0[4]), "=f"(_tmem_load_0[5]), "=f"(_tmem_load_0[6]), "=f"(_tmem_load_0[7]), "=f"(_tmem_load_0[8]), "=f"(_tmem_load_0[9]), "=f"(_tmem_load_0[10]), "=f"(_tmem_load_0[11]), "=f"(_tmem_load_0[12]), "=f"(_tmem_load_0[13]), "=f"(_tmem_load_0[14]), "=f"(_tmem_load_0[15]), "=f"(_tmem_load_0[16]), "=f"(_tmem_load_0[17]), "=f"(_tmem_load_0[18]), "=f"(_tmem_load_0[19]), "=f"(_tmem_load_0[20]), "=f"(_tmem_load_0[21]), "=f"(_tmem_load_0[22]), "=f"(_tmem_load_0[23]), "=f"(_tmem_load_0[24]), "=f"(_tmem_load_0[25]), "=f"(_tmem_load_0[26]), "=f"(_tmem_load_0[27]), "=f"(_tmem_load_0[28]), "=f"(_tmem_load_0[29]), "=f"(_tmem_load_0[30]), "=f"(_tmem_load_0[31]), "=f"(_tmem_load_0[32]), "=f"(_tmem_load_0[33]), "=f"(_tmem_load_0[34]), "=f"(_tmem_load_0[35]), "=f"(_tmem_load_0[36]), "=f"(_tmem_load_0[37]), "=f"(_tmem_load_0[38]), "=f"(_tmem_load_0[39]), "=f"(_tmem_load_0[40]), "=f"(_tmem_load_0[41]), "=f"(_tmem_load_0[42]), "=f"(_tmem_load_0[43]), "=f"(_tmem_load_0[44]), "=f"(_tmem_load_0[45]), "=f"(_tmem_load_0[46]), "=f"(_tmem_load_0[47]), "=f"(_tmem_load_0[48]), "=f"(_tmem_load_0[49]), "=f"(_tmem_load_0[50]), "=f"(_tmem_load_0[51]), "=f"(_tmem_load_0[52]), "=f"(_tmem_load_0[53]), "=f"(_tmem_load_0[54]), "=f"(_tmem_load_0[55]), "=f"(_tmem_load_0[56]), "=f"(_tmem_load_0[57]), "=f"(_tmem_load_0[58]), "=f"(_tmem_load_0[59]), "=f"(_tmem_load_0[60]), "=f"(_tmem_load_0[61]), "=f"(_tmem_load_0[62]), "=f"(_tmem_load_0[63])
                    : "r"(sk_lane_addr));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x64.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, [%64];"
                    : "=f"(_tmem_load_0[64]), "=f"(_tmem_load_0[65]), "=f"(_tmem_load_0[66]), "=f"(_tmem_load_0[67]), "=f"(_tmem_load_0[68]), "=f"(_tmem_load_0[69]), "=f"(_tmem_load_0[70]), "=f"(_tmem_load_0[71]), "=f"(_tmem_load_0[72]), "=f"(_tmem_load_0[73]), "=f"(_tmem_load_0[74]), "=f"(_tmem_load_0[75]), "=f"(_tmem_load_0[76]), "=f"(_tmem_load_0[77]), "=f"(_tmem_load_0[78]), "=f"(_tmem_load_0[79]), "=f"(_tmem_load_0[80]), "=f"(_tmem_load_0[81]), "=f"(_tmem_load_0[82]), "=f"(_tmem_load_0[83]), "=f"(_tmem_load_0[84]), "=f"(_tmem_load_0[85]), "=f"(_tmem_load_0[86]), "=f"(_tmem_load_0[87]), "=f"(_tmem_load_0[88]), "=f"(_tmem_load_0[89]), "=f"(_tmem_load_0[90]), "=f"(_tmem_load_0[91]), "=f"(_tmem_load_0[92]), "=f"(_tmem_load_0[93]), "=f"(_tmem_load_0[94]), "=f"(_tmem_load_0[95]), "=f"(_tmem_load_0[96]), "=f"(_tmem_load_0[97]), "=f"(_tmem_load_0[98]), "=f"(_tmem_load_0[99]), "=f"(_tmem_load_0[100]), "=f"(_tmem_load_0[101]), "=f"(_tmem_load_0[102]), "=f"(_tmem_load_0[103]), "=f"(_tmem_load_0[104]), "=f"(_tmem_load_0[105]), "=f"(_tmem_load_0[106]), "=f"(_tmem_load_0[107]), "=f"(_tmem_load_0[108]), "=f"(_tmem_load_0[109]), "=f"(_tmem_load_0[110]), "=f"(_tmem_load_0[111]), "=f"(_tmem_load_0[112]), "=f"(_tmem_load_0[113]), "=f"(_tmem_load_0[114]), "=f"(_tmem_load_0[115]), "=f"(_tmem_load_0[116]), "=f"(_tmem_load_0[117]), "=f"(_tmem_load_0[118]), "=f"(_tmem_load_0[119]), "=f"(_tmem_load_0[120]), "=f"(_tmem_load_0[121]), "=f"(_tmem_load_0[122]), "=f"(_tmem_load_0[123]), "=f"(_tmem_load_0[124]), "=f"(_tmem_load_0[125]), "=f"(_tmem_load_0[126]), "=f"(_tmem_load_0[127])
                    : "r"(sk_lane_addr + 64));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                if (elect_sync()) {
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    asm volatile(
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                        :: "r"((epilogue_done_addr) & 0xFEFFFFFF) : "memory");
                }
                _phase_mainloop_done ^= 1;
                int skp_cvi0 = (skp_eslot * 2 * 8 * 32 + skp_wvec) * 32 + lane;
                #pragma unroll
                for (int v = 0; v < 32; v++) {
                    float quad_p[4];
                    #pragma unroll
                    for (int j = 0; j < 4; j++) {
                        quad_p[j] = _tmem_load_0[4 * v + j];
                    }
                    int skp_cvi = skp_cvi0 + v * 32;
                    {
                        float4 _v4 = make_float4(quad_p[0 + 0], quad_p[0 + 1], quad_p[0 + 2], quad_p[0 + 3]);
                        *reinterpret_cast<float4*>((sk_partials + ((unsigned long long)skp_cvi * 4)) + 0) = _v4;
                    }
                }
                asm volatile("tcgen05.fence::before_thread_sync;");
                asm volatile("barrier.sync 3, 256;" ::: "memory");
                if (tid == 64) {
                    asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(sk_flags) + (skp_eslot * 2 + skp_rank))), "r"(static_cast<unsigned int>(1)) : "memory");
                }
            }
            unsigned int _phase_work_full_2 = 0;
            if (sk_dp > 0) {
                #pragma unroll 1
                for (int skp_done_2 = 0; skp_done_2 < num_cluster_tiles_2; skp_done_2++) {
                    int tiles_per_group_3 = GROUP_M * n_tiles;
                    int group_3 = this_bid_1 / (unsigned int)tiles_per_group_3;
                    int first_m_3 = group_3 * GROUP_M;
                    int remaining_3 = m_tiles - first_m_3;
                    int group_size_3 = ((remaining_3 >= GROUP_M) ? GROUP_M : remaining_3);
                    int local_3 = this_bid_1 % (unsigned int)tiles_per_group_3;
                    int bid_m_3 = first_m_3 + local_3 % group_size_3;
                    int bid_n_3 = local_3 / group_size_3;
                    int off_m_3 = bid_m_3 * BLOCK_M;
                    int off_n_3 = bid_n_3 * 256;
                    int global_row = off_m_3 + local_row;
                    int col0 = col_part * 128;
                    int lane_addr = taddr + (unsigned int)(epi_warp * 32 << 16) + (unsigned int)col0;
                    int col_base = off_n_3 + col0;
                    unsigned long long row_base = (unsigned long long)global_row * (unsigned long long)ldo + (unsigned long long)col_base;
                    mbarrier_wait(mainloop_done_addr, _phase_mainloop_done);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    float tmem_load[128];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x64.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, [%64];"
                        : "=f"(tmem_load[0]), "=f"(tmem_load[1]), "=f"(tmem_load[2]), "=f"(tmem_load[3]), "=f"(tmem_load[4]), "=f"(tmem_load[5]), "=f"(tmem_load[6]), "=f"(tmem_load[7]), "=f"(tmem_load[8]), "=f"(tmem_load[9]), "=f"(tmem_load[10]), "=f"(tmem_load[11]), "=f"(tmem_load[12]), "=f"(tmem_load[13]), "=f"(tmem_load[14]), "=f"(tmem_load[15]), "=f"(tmem_load[16]), "=f"(tmem_load[17]), "=f"(tmem_load[18]), "=f"(tmem_load[19]), "=f"(tmem_load[20]), "=f"(tmem_load[21]), "=f"(tmem_load[22]), "=f"(tmem_load[23]), "=f"(tmem_load[24]), "=f"(tmem_load[25]), "=f"(tmem_load[26]), "=f"(tmem_load[27]), "=f"(tmem_load[28]), "=f"(tmem_load[29]), "=f"(tmem_load[30]), "=f"(tmem_load[31]), "=f"(tmem_load[32]), "=f"(tmem_load[33]), "=f"(tmem_load[34]), "=f"(tmem_load[35]), "=f"(tmem_load[36]), "=f"(tmem_load[37]), "=f"(tmem_load[38]), "=f"(tmem_load[39]), "=f"(tmem_load[40]), "=f"(tmem_load[41]), "=f"(tmem_load[42]), "=f"(tmem_load[43]), "=f"(tmem_load[44]), "=f"(tmem_load[45]), "=f"(tmem_load[46]), "=f"(tmem_load[47]), "=f"(tmem_load[48]), "=f"(tmem_load[49]), "=f"(tmem_load[50]), "=f"(tmem_load[51]), "=f"(tmem_load[52]), "=f"(tmem_load[53]), "=f"(tmem_load[54]), "=f"(tmem_load[55]), "=f"(tmem_load[56]), "=f"(tmem_load[57]), "=f"(tmem_load[58]), "=f"(tmem_load[59]), "=f"(tmem_load[60]), "=f"(tmem_load[61]), "=f"(tmem_load[62]), "=f"(tmem_load[63])
                        : "r"(lane_addr));
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x64.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, [%64];"
                        : "=f"(tmem_load[64]), "=f"(tmem_load[65]), "=f"(tmem_load[66]), "=f"(tmem_load[67]), "=f"(tmem_load[68]), "=f"(tmem_load[69]), "=f"(tmem_load[70]), "=f"(tmem_load[71]), "=f"(tmem_load[72]), "=f"(tmem_load[73]), "=f"(tmem_load[74]), "=f"(tmem_load[75]), "=f"(tmem_load[76]), "=f"(tmem_load[77]), "=f"(tmem_load[78]), "=f"(tmem_load[79]), "=f"(tmem_load[80]), "=f"(tmem_load[81]), "=f"(tmem_load[82]), "=f"(tmem_load[83]), "=f"(tmem_load[84]), "=f"(tmem_load[85]), "=f"(tmem_load[86]), "=f"(tmem_load[87]), "=f"(tmem_load[88]), "=f"(tmem_load[89]), "=f"(tmem_load[90]), "=f"(tmem_load[91]), "=f"(tmem_load[92]), "=f"(tmem_load[93]), "=f"(tmem_load[94]), "=f"(tmem_load[95]), "=f"(tmem_load[96]), "=f"(tmem_load[97]), "=f"(tmem_load[98]), "=f"(tmem_load[99]), "=f"(tmem_load[100]), "=f"(tmem_load[101]), "=f"(tmem_load[102]), "=f"(tmem_load[103]), "=f"(tmem_load[104]), "=f"(tmem_load[105]), "=f"(tmem_load[106]), "=f"(tmem_load[107]), "=f"(tmem_load[108]), "=f"(tmem_load[109]), "=f"(tmem_load[110]), "=f"(tmem_load[111]), "=f"(tmem_load[112]), "=f"(tmem_load[113]), "=f"(tmem_load[114]), "=f"(tmem_load[115]), "=f"(tmem_load[116]), "=f"(tmem_load[117]), "=f"(tmem_load[118]), "=f"(tmem_load[119]), "=f"(tmem_load[120]), "=f"(tmem_load[121]), "=f"(tmem_load[122]), "=f"(tmem_load[123]), "=f"(tmem_load[124]), "=f"(tmem_load[125]), "=f"(tmem_load[126]), "=f"(tmem_load[127])
                        : "r"(lane_addr + 64));
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    if (elect_sync()) {
                        asm volatile("tcgen05.fence::before_thread_sync;");
                        asm volatile(
                            "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                            :: "r"((epilogue_done_addr) & 0xFEFFFFFF) : "memory");
                    }
                    _phase_mainloop_done ^= 1;
                    unsigned int tmem_load_0_bf16[64];
                    #pragma unroll
                    for (int _lp = 0; _lp < 64; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(tmem_load[_lp*2 + 0], tmem_load[_lp*2+1 + 0]));
                        tmem_load_0_bf16[_lp] = *(uint32_t*)&_bf2;
                    }
                    unsigned int group_base = smem_out_addr + (unsigned int)(col_part * 8192);
                    int swz = local_row >> 1 & 3;
                    #pragma unroll
                    for (int c = 0; c < 4; c++) {
                        unsigned int ts_base = group_base;
                        unsigned int ts_row = ts_base + (unsigned int)(local_row * 64);
                        if ((warp - 2) % 4 == 0) {
                            asm volatile("cp.async.bulk.wait_group.read 0;");
                        }
                        asm volatile("tcgen05.fence::before_thread_sync;");
                        asm volatile("barrier.sync %0, 128;" :: "r"(1 + col_part) : "memory");
                        #pragma unroll
                        for (int q = 0; q < 4; q++) {
                            int s0 = c * 16 + q * 4;
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(ts_row + (unsigned int)((q ^ swz) * 16)), "r"(tmem_load_0_bf16[s0]), "r"(tmem_load_0_bf16[s0 + 1]), "r"(tmem_load_0_bf16[s0 + 2]), "r"(tmem_load_0_bf16[s0 + 3]));
                        }
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        asm volatile("barrier.sync %0, 128;" :: "r"(1 + col_part) : "memory");
                        if ((warp - 2) % 4 == 0) {
                            if (elect_sync()) {
                                tma_store_2d((&OUT), col_base + c * 32, off_m_3, ts_base);
                                asm volatile("cp.async.bulk.commit_group;");
                            }
                        }
                    }
                    if (skp_done_2 + 1 == skp_cap_2) {
                        break;
                    }
                    mbarrier_wait(work_full_addr + (work_stage_2) * 8, _phase_work_full_2);
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
                    if (clc_valid_1 == 0) {
                        break;
                    }
                    this_bid_1 = clc_ctaid_1 + (unsigned int)cta_rank;
                }
            }
            if (skp_late_2 == 1) {
                unsigned int skp_lbid_1 = (unsigned int)((sk_dp + skp_lt_2) * 2) + (unsigned int)cta_rank;
                int tiles_per_group_4 = GROUP_M * n_tiles;
                int group_4 = skp_lbid_1 / (unsigned int)tiles_per_group_4;
                int first_m_4 = group_4 * GROUP_M;
                int remaining_4 = m_tiles - first_m_4;
                int group_size_4 = ((remaining_4 >= GROUP_M) ? GROUP_M : remaining_4);
                int local_4 = skp_lbid_1 % (unsigned int)tiles_per_group_4;
                int bid_m_4 = first_m_4 + local_4 % group_size_4;
                int bid_n_4 = local_4 / group_size_4;
                int off_m_4 = bid_m_4 * BLOCK_M;
                int off_n_4 = bid_n_4 * 256;
                int global_row_1 = off_m_4 + local_row;
                int col0_1 = col_part * 128;
                int lane_addr_1 = taddr + (unsigned int)(epi_warp * 32 << 16) + (unsigned int)col0_1;
                int col_base_1 = off_n_4 + col0_1;
                unsigned long long row_base_1 = (unsigned long long)global_row_1 * (unsigned long long)ldo + (unsigned long long)col_base_1;
                if (skp_lnc_2 > 0) {
                    if (tid == 64) {
                        if (skp_lnc_2 > 0) {
                            {
                            unsigned int _acquire_observed;
                            do {
                            asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(_acquire_observed) : "l"((reinterpret_cast<unsigned int*>(sk_flags) + (skp_lt_2 * sk_ksplit * 2 + skp_rank))) : "memory");
                            } while (static_cast<unsigned int>(_acquire_observed - static_cast<unsigned int>(1)) >= static_cast<unsigned int>(1));
                            }
                        }
                        if (skp_lnc_2 > 1) {
                            {
                            unsigned int _acquire_observed;
                            do {
                            asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(_acquire_observed) : "l"((reinterpret_cast<unsigned int*>(sk_flags) + ((skp_lt_2 * sk_ksplit + 1) * 2 + skp_rank))) : "memory");
                            } while (static_cast<unsigned int>(_acquire_observed - static_cast<unsigned int>(1)) >= static_cast<unsigned int>(1));
                            }
                        }
                        if (skp_lnc_2 > 2) {
                            {
                            unsigned int _acquire_observed;
                            do {
                            asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(_acquire_observed) : "l"((reinterpret_cast<unsigned int*>(sk_flags) + ((skp_lt_2 * sk_ksplit + 2) * 2 + skp_rank))) : "memory");
                            } while (static_cast<unsigned int>(_acquire_observed - static_cast<unsigned int>(1)) >= static_cast<unsigned int>(1));
                            }
                        }
                        if (skp_lnc_2 > 3) {
                            {
                            unsigned int _acquire_observed;
                            do {
                            asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(_acquire_observed) : "l"((reinterpret_cast<unsigned int*>(sk_flags) + ((skp_lt_2 * sk_ksplit + 3) * 2 + skp_rank))) : "memory");
                            } while (static_cast<unsigned int>(_acquire_observed - static_cast<unsigned int>(1)) >= static_cast<unsigned int>(1));
                            }
                        }
                    }
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    asm volatile("barrier.sync 3, 256;" ::: "memory");
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    float pre[32];
                    if (skp_lnc_2 > 0) {
                        int skp_vi0 = (skp_lt_2 * sk_ksplit * 2 * 8 * 32 + skp_wvec) * 32 + lane;
                        float parts_q[32];
                        #pragma unroll
                        for (int skp_v = 0; skp_v < 8; skp_v++) {
                            int skp_vi = skp_vi0 + skp_v * 32;
                            {
                                float4 _v4 = *reinterpret_cast<const float4*>(sk_partials + ((unsigned long long)skp_vi * 4) + 0);
                                parts_q[4 * skp_v + 0] = _v4.x;
                                parts_q[4 * skp_v + 1] = _v4.y;
                                parts_q[4 * skp_v + 2] = _v4.z;
                                parts_q[4 * skp_v + 3] = _v4.w;
                            }
                        }
                        #pragma unroll
                        for (int i = 0; i < 32; i++) {
                            pre[i] = parts_q[i];
                        }
                    }
                    if (skp_lnc_2 > 1) {
                        int skp_vi0_0 = ((skp_lt_2 * sk_ksplit + 1) * 2 * 8 * 32 + skp_wvec) * 32 + lane;
                        float parts_q_0[32];
                        #pragma unroll
                        for (int skp_v_1 = 0; skp_v_1 < 8; skp_v_1++) {
                            int skp_vi_1 = skp_vi0_0 + skp_v_1 * 32;
                            {
                                float4 _v4 = *reinterpret_cast<const float4*>(sk_partials + ((unsigned long long)skp_vi_1 * 4) + 0);
                                parts_q_0[4 * skp_v_1 + 0] = _v4.x;
                                parts_q_0[4 * skp_v_1 + 1] = _v4.y;
                                parts_q_0[4 * skp_v_1 + 2] = _v4.z;
                                parts_q_0[4 * skp_v_1 + 3] = _v4.w;
                            }
                        }
                        #pragma unroll
                        for (int i_0 = 0; i_0 < 32; i_0++) {
                            pre[i_0] = pre[i_0] + parts_q_0[i_0];
                        }
                    }
                    if (skp_lnc_2 > 2) {
                        int skp_vi0_1 = ((skp_lt_2 * sk_ksplit + 2) * 2 * 8 * 32 + skp_wvec) * 32 + lane;
                        float parts_q_1[32];
                        #pragma unroll
                        for (int skp_v_2 = 0; skp_v_2 < 8; skp_v_2++) {
                            int skp_vi_2 = skp_vi0_1 + skp_v_2 * 32;
                            {
                                float4 _v4 = *reinterpret_cast<const float4*>(sk_partials + ((unsigned long long)skp_vi_2 * 4) + 0);
                                parts_q_1[4 * skp_v_2 + 0] = _v4.x;
                                parts_q_1[4 * skp_v_2 + 1] = _v4.y;
                                parts_q_1[4 * skp_v_2 + 2] = _v4.z;
                                parts_q_1[4 * skp_v_2 + 3] = _v4.w;
                            }
                        }
                        #pragma unroll
                        for (int i_1 = 0; i_1 < 32; i_1++) {
                            pre[i_1] = pre[i_1] + parts_q_1[i_1];
                        }
                    }
                    if (skp_lnc_2 > 3) {
                        int skp_vi0_2 = ((skp_lt_2 * sk_ksplit + 3) * 2 * 8 * 32 + skp_wvec) * 32 + lane;
                        float parts_q_2[32];
                        #pragma unroll
                        for (int skp_v_3 = 0; skp_v_3 < 8; skp_v_3++) {
                            int skp_vi_3 = skp_vi0_2 + skp_v_3 * 32;
                            {
                                float4 _v4 = *reinterpret_cast<const float4*>(sk_partials + ((unsigned long long)skp_vi_3 * 4) + 0);
                                parts_q_2[4 * skp_v_3 + 0] = _v4.x;
                                parts_q_2[4 * skp_v_3 + 1] = _v4.y;
                                parts_q_2[4 * skp_v_3 + 2] = _v4.z;
                                parts_q_2[4 * skp_v_3 + 3] = _v4.w;
                            }
                        }
                        #pragma unroll
                        for (int i_2 = 0; i_2 < 32; i_2++) {
                            pre[i_2] = pre[i_2] + parts_q_2[i_2];
                        }
                    }
                    tmem_st_x16_f32(skp_pre_addr, pre);
                    tmem_st_x16_f32(skp_pre_addr + 16, (pre + 16));
                    float pre_0[32];
                    if (skp_lnc_2 > 0) {
                        int skp_vi0_3 = (skp_lt_2 * sk_ksplit * 2 * 8 * 32 + skp_wvec) * 32 + lane;
                        float parts_q_3[32];
                        #pragma unroll
                        for (int skp_v_4 = 0; skp_v_4 < 8; skp_v_4++) {
                            int skp_vi_4 = skp_vi0_3 + (8 + skp_v_4) * 32;
                            {
                                float4 _v4 = *reinterpret_cast<const float4*>(sk_partials + ((unsigned long long)skp_vi_4 * 4) + 0);
                                parts_q_3[4 * skp_v_4 + 0] = _v4.x;
                                parts_q_3[4 * skp_v_4 + 1] = _v4.y;
                                parts_q_3[4 * skp_v_4 + 2] = _v4.z;
                                parts_q_3[4 * skp_v_4 + 3] = _v4.w;
                            }
                        }
                        #pragma unroll
                        for (int i_3 = 0; i_3 < 32; i_3++) {
                            pre_0[i_3] = parts_q_3[i_3];
                        }
                    }
                    if (skp_lnc_2 > 1) {
                        int skp_vi0_4 = ((skp_lt_2 * sk_ksplit + 1) * 2 * 8 * 32 + skp_wvec) * 32 + lane;
                        float parts_q_4[32];
                        #pragma unroll
                        for (int skp_v_5 = 0; skp_v_5 < 8; skp_v_5++) {
                            int skp_vi_5 = skp_vi0_4 + (8 + skp_v_5) * 32;
                            {
                                float4 _v4 = *reinterpret_cast<const float4*>(sk_partials + ((unsigned long long)skp_vi_5 * 4) + 0);
                                parts_q_4[4 * skp_v_5 + 0] = _v4.x;
                                parts_q_4[4 * skp_v_5 + 1] = _v4.y;
                                parts_q_4[4 * skp_v_5 + 2] = _v4.z;
                                parts_q_4[4 * skp_v_5 + 3] = _v4.w;
                            }
                        }
                        #pragma unroll
                        for (int i_4 = 0; i_4 < 32; i_4++) {
                            pre_0[i_4] = pre_0[i_4] + parts_q_4[i_4];
                        }
                    }
                    if (skp_lnc_2 > 2) {
                        int skp_vi0_5 = ((skp_lt_2 * sk_ksplit + 2) * 2 * 8 * 32 + skp_wvec) * 32 + lane;
                        float parts_q_5[32];
                        #pragma unroll
                        for (int skp_v_6 = 0; skp_v_6 < 8; skp_v_6++) {
                            int skp_vi_6 = skp_vi0_5 + (8 + skp_v_6) * 32;
                            {
                                float4 _v4 = *reinterpret_cast<const float4*>(sk_partials + ((unsigned long long)skp_vi_6 * 4) + 0);
                                parts_q_5[4 * skp_v_6 + 0] = _v4.x;
                                parts_q_5[4 * skp_v_6 + 1] = _v4.y;
                                parts_q_5[4 * skp_v_6 + 2] = _v4.z;
                                parts_q_5[4 * skp_v_6 + 3] = _v4.w;
                            }
                        }
                        #pragma unroll
                        for (int i_5 = 0; i_5 < 32; i_5++) {
                            pre_0[i_5] = pre_0[i_5] + parts_q_5[i_5];
                        }
                    }
                    if (skp_lnc_2 > 3) {
                        int skp_vi0_6 = ((skp_lt_2 * sk_ksplit + 3) * 2 * 8 * 32 + skp_wvec) * 32 + lane;
                        float parts_q_6[32];
                        #pragma unroll
                        for (int skp_v_7 = 0; skp_v_7 < 8; skp_v_7++) {
                            int skp_vi_7 = skp_vi0_6 + (8 + skp_v_7) * 32;
                            {
                                float4 _v4 = *reinterpret_cast<const float4*>(sk_partials + ((unsigned long long)skp_vi_7 * 4) + 0);
                                parts_q_6[4 * skp_v_7 + 0] = _v4.x;
                                parts_q_6[4 * skp_v_7 + 1] = _v4.y;
                                parts_q_6[4 * skp_v_7 + 2] = _v4.z;
                                parts_q_6[4 * skp_v_7 + 3] = _v4.w;
                            }
                        }
                        #pragma unroll
                        for (int i_6 = 0; i_6 < 32; i_6++) {
                            pre_0[i_6] = pre_0[i_6] + parts_q_6[i_6];
                        }
                    }
                    tmem_st_x16_f32(skp_pre_addr + 32, pre_0);
                    tmem_st_x16_f32(skp_pre_addr + 32 + 16, (pre_0 + 16));
                    float pre_1[32];
                    if (skp_lnc_2 > 0) {
                        int skp_vi0_7 = (skp_lt_2 * sk_ksplit * 2 * 8 * 32 + skp_wvec) * 32 + lane;
                        float parts_q_7[32];
                        #pragma unroll
                        for (int skp_v_8 = 0; skp_v_8 < 8; skp_v_8++) {
                            int skp_vi_8 = skp_vi0_7 + (16 + skp_v_8) * 32;
                            {
                                float4 _v4 = *reinterpret_cast<const float4*>(sk_partials + ((unsigned long long)skp_vi_8 * 4) + 0);
                                parts_q_7[4 * skp_v_8 + 0] = _v4.x;
                                parts_q_7[4 * skp_v_8 + 1] = _v4.y;
                                parts_q_7[4 * skp_v_8 + 2] = _v4.z;
                                parts_q_7[4 * skp_v_8 + 3] = _v4.w;
                            }
                        }
                        #pragma unroll
                        for (int i_7 = 0; i_7 < 32; i_7++) {
                            pre_1[i_7] = parts_q_7[i_7];
                        }
                    }
                    if (skp_lnc_2 > 1) {
                        int skp_vi0_3_1 = ((skp_lt_2 * sk_ksplit + 1) * 2 * 8 * 32 + skp_wvec) * 32 + lane;
                        float parts_q_3_1[32];
                        #pragma unroll
                        for (int skp_v_9 = 0; skp_v_9 < 8; skp_v_9++) {
                            int skp_vi_9 = skp_vi0_3_1 + (16 + skp_v_9) * 32;
                            {
                                float4 _v4 = *reinterpret_cast<const float4*>(sk_partials + ((unsigned long long)skp_vi_9 * 4) + 0);
                                parts_q_3_1[4 * skp_v_9 + 0] = _v4.x;
                                parts_q_3_1[4 * skp_v_9 + 1] = _v4.y;
                                parts_q_3_1[4 * skp_v_9 + 2] = _v4.z;
                                parts_q_3_1[4 * skp_v_9 + 3] = _v4.w;
                            }
                        }
                        #pragma unroll
                        for (int i_3_1 = 0; i_3_1 < 32; i_3_1++) {
                            pre_1[i_3_1] = pre_1[i_3_1] + parts_q_3_1[i_3_1];
                        }
                    }
                    if (skp_lnc_2 > 2) {
                        int skp_vi0_4_1 = ((skp_lt_2 * sk_ksplit + 2) * 2 * 8 * 32 + skp_wvec) * 32 + lane;
                        float parts_q_4_1[32];
                        #pragma unroll
                        for (int skp_v_10 = 0; skp_v_10 < 8; skp_v_10++) {
                            int skp_vi_10 = skp_vi0_4_1 + (16 + skp_v_10) * 32;
                            {
                                float4 _v4 = *reinterpret_cast<const float4*>(sk_partials + ((unsigned long long)skp_vi_10 * 4) + 0);
                                parts_q_4_1[4 * skp_v_10 + 0] = _v4.x;
                                parts_q_4_1[4 * skp_v_10 + 1] = _v4.y;
                                parts_q_4_1[4 * skp_v_10 + 2] = _v4.z;
                                parts_q_4_1[4 * skp_v_10 + 3] = _v4.w;
                            }
                        }
                        #pragma unroll
                        for (int i_4_1 = 0; i_4_1 < 32; i_4_1++) {
                            pre_1[i_4_1] = pre_1[i_4_1] + parts_q_4_1[i_4_1];
                        }
                    }
                    if (skp_lnc_2 > 3) {
                        int skp_vi0_5_1 = ((skp_lt_2 * sk_ksplit + 3) * 2 * 8 * 32 + skp_wvec) * 32 + lane;
                        float parts_q_5_1[32];
                        #pragma unroll
                        for (int skp_v_11 = 0; skp_v_11 < 8; skp_v_11++) {
                            int skp_vi_11 = skp_vi0_5_1 + (16 + skp_v_11) * 32;
                            {
                                float4 _v4 = *reinterpret_cast<const float4*>(sk_partials + ((unsigned long long)skp_vi_11 * 4) + 0);
                                parts_q_5_1[4 * skp_v_11 + 0] = _v4.x;
                                parts_q_5_1[4 * skp_v_11 + 1] = _v4.y;
                                parts_q_5_1[4 * skp_v_11 + 2] = _v4.z;
                                parts_q_5_1[4 * skp_v_11 + 3] = _v4.w;
                            }
                        }
                        #pragma unroll
                        for (int i_5_1 = 0; i_5_1 < 32; i_5_1++) {
                            pre_1[i_5_1] = pre_1[i_5_1] + parts_q_5_1[i_5_1];
                        }
                    }
                    tmem_st_x16_f32(skp_pre_addr + 64, pre_1);
                    tmem_st_x16_f32(skp_pre_addr + 64 + 16, (pre_1 + 16));
                    float pre_2[16];
                    if (skp_lnc_2 > 0) {
                        int skp_vi0_8 = (skp_lt_2 * sk_ksplit * 2 * 8 * 32 + skp_wvec) * 32 + lane;
                        float parts_q_8[16];
                        #pragma unroll
                        for (int skp_v_12 = 0; skp_v_12 < 4; skp_v_12++) {
                            int skp_vi_12 = skp_vi0_8 + (24 + skp_v_12) * 32;
                            {
                                float4 _v4 = *reinterpret_cast<const float4*>(sk_partials + ((unsigned long long)skp_vi_12 * 4) + 0);
                                parts_q_8[4 * skp_v_12 + 0] = _v4.x;
                                parts_q_8[4 * skp_v_12 + 1] = _v4.y;
                                parts_q_8[4 * skp_v_12 + 2] = _v4.z;
                                parts_q_8[4 * skp_v_12 + 3] = _v4.w;
                            }
                        }
                        #pragma unroll
                        for (int i_8 = 0; i_8 < 16; i_8++) {
                            pre_2[i_8] = parts_q_8[i_8];
                        }
                    }
                    if (skp_lnc_2 > 1) {
                        int skp_vi0_6_1 = ((skp_lt_2 * sk_ksplit + 1) * 2 * 8 * 32 + skp_wvec) * 32 + lane;
                        float parts_q_6_1[16];
                        #pragma unroll
                        for (int skp_v_13 = 0; skp_v_13 < 4; skp_v_13++) {
                            int skp_vi_13 = skp_vi0_6_1 + (24 + skp_v_13) * 32;
                            {
                                float4 _v4 = *reinterpret_cast<const float4*>(sk_partials + ((unsigned long long)skp_vi_13 * 4) + 0);
                                parts_q_6_1[4 * skp_v_13 + 0] = _v4.x;
                                parts_q_6_1[4 * skp_v_13 + 1] = _v4.y;
                                parts_q_6_1[4 * skp_v_13 + 2] = _v4.z;
                                parts_q_6_1[4 * skp_v_13 + 3] = _v4.w;
                            }
                        }
                        #pragma unroll
                        for (int i_6_1 = 0; i_6_1 < 16; i_6_1++) {
                            pre_2[i_6_1] = pre_2[i_6_1] + parts_q_6_1[i_6_1];
                        }
                    }
                    if (skp_lnc_2 > 2) {
                        int skp_vi0_7_1 = ((skp_lt_2 * sk_ksplit + 2) * 2 * 8 * 32 + skp_wvec) * 32 + lane;
                        float parts_q_7_1[16];
                        #pragma unroll
                        for (int skp_v_14 = 0; skp_v_14 < 4; skp_v_14++) {
                            int skp_vi_14 = skp_vi0_7_1 + (24 + skp_v_14) * 32;
                            {
                                float4 _v4 = *reinterpret_cast<const float4*>(sk_partials + ((unsigned long long)skp_vi_14 * 4) + 0);
                                parts_q_7_1[4 * skp_v_14 + 0] = _v4.x;
                                parts_q_7_1[4 * skp_v_14 + 1] = _v4.y;
                                parts_q_7_1[4 * skp_v_14 + 2] = _v4.z;
                                parts_q_7_1[4 * skp_v_14 + 3] = _v4.w;
                            }
                        }
                        #pragma unroll
                        for (int i_7_1 = 0; i_7_1 < 16; i_7_1++) {
                            pre_2[i_7_1] = pre_2[i_7_1] + parts_q_7_1[i_7_1];
                        }
                    }
                    if (skp_lnc_2 > 3) {
                        int skp_vi0_8_1 = ((skp_lt_2 * sk_ksplit + 3) * 2 * 8 * 32 + skp_wvec) * 32 + lane;
                        float parts_q_8_1[16];
                        #pragma unroll
                        for (int skp_v_15 = 0; skp_v_15 < 4; skp_v_15++) {
                            int skp_vi_15 = skp_vi0_8_1 + (24 + skp_v_15) * 32;
                            {
                                float4 _v4 = *reinterpret_cast<const float4*>(sk_partials + ((unsigned long long)skp_vi_15 * 4) + 0);
                                parts_q_8_1[4 * skp_v_15 + 0] = _v4.x;
                                parts_q_8_1[4 * skp_v_15 + 1] = _v4.y;
                                parts_q_8_1[4 * skp_v_15 + 2] = _v4.z;
                                parts_q_8_1[4 * skp_v_15 + 3] = _v4.w;
                            }
                        }
                        #pragma unroll
                        for (int i_8_1 = 0; i_8_1 < 16; i_8_1++) {
                            pre_2[i_8_1] = pre_2[i_8_1] + parts_q_8_1[i_8_1];
                        }
                    }
                    tmem_st_x16_f32(skp_pre_addr + 96, pre_2);
                    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    asm volatile("barrier.sync 3, 256;" ::: "memory");
                    if (tid == 64) {
                        if (skp_lnc_2 > 0) {
                            asm volatile("st.release.gpu.global.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(sk_flags) + (skp_lt_2 * sk_ksplit * 2 + skp_rank))), "r"(static_cast<unsigned int>(0)) : "memory");
                        }
                        if (skp_lnc_2 > 1) {
                            asm volatile("st.release.gpu.global.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(sk_flags) + ((skp_lt_2 * sk_ksplit + 1) * 2 + skp_rank))), "r"(static_cast<unsigned int>(0)) : "memory");
                        }
                        if (skp_lnc_2 > 2) {
                            asm volatile("st.release.gpu.global.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(sk_flags) + ((skp_lt_2 * sk_ksplit + 2) * 2 + skp_rank))), "r"(static_cast<unsigned int>(0)) : "memory");
                        }
                        if (skp_lnc_2 > 3) {
                            asm volatile("st.release.gpu.global.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(sk_flags) + ((skp_lt_2 * sk_ksplit + 3) * 2 + skp_rank))), "r"(static_cast<unsigned int>(0)) : "memory");
                        }
                    }
                }
                mbarrier_wait(mainloop_done_addr, _phase_mainloop_done);
                asm volatile("tcgen05.fence::after_thread_sync;");
                float _tmem_load_2[128];
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x64.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, [%64];"
                    : "=f"(_tmem_load_2[0]), "=f"(_tmem_load_2[1]), "=f"(_tmem_load_2[2]), "=f"(_tmem_load_2[3]), "=f"(_tmem_load_2[4]), "=f"(_tmem_load_2[5]), "=f"(_tmem_load_2[6]), "=f"(_tmem_load_2[7]), "=f"(_tmem_load_2[8]), "=f"(_tmem_load_2[9]), "=f"(_tmem_load_2[10]), "=f"(_tmem_load_2[11]), "=f"(_tmem_load_2[12]), "=f"(_tmem_load_2[13]), "=f"(_tmem_load_2[14]), "=f"(_tmem_load_2[15]), "=f"(_tmem_load_2[16]), "=f"(_tmem_load_2[17]), "=f"(_tmem_load_2[18]), "=f"(_tmem_load_2[19]), "=f"(_tmem_load_2[20]), "=f"(_tmem_load_2[21]), "=f"(_tmem_load_2[22]), "=f"(_tmem_load_2[23]), "=f"(_tmem_load_2[24]), "=f"(_tmem_load_2[25]), "=f"(_tmem_load_2[26]), "=f"(_tmem_load_2[27]), "=f"(_tmem_load_2[28]), "=f"(_tmem_load_2[29]), "=f"(_tmem_load_2[30]), "=f"(_tmem_load_2[31]), "=f"(_tmem_load_2[32]), "=f"(_tmem_load_2[33]), "=f"(_tmem_load_2[34]), "=f"(_tmem_load_2[35]), "=f"(_tmem_load_2[36]), "=f"(_tmem_load_2[37]), "=f"(_tmem_load_2[38]), "=f"(_tmem_load_2[39]), "=f"(_tmem_load_2[40]), "=f"(_tmem_load_2[41]), "=f"(_tmem_load_2[42]), "=f"(_tmem_load_2[43]), "=f"(_tmem_load_2[44]), "=f"(_tmem_load_2[45]), "=f"(_tmem_load_2[46]), "=f"(_tmem_load_2[47]), "=f"(_tmem_load_2[48]), "=f"(_tmem_load_2[49]), "=f"(_tmem_load_2[50]), "=f"(_tmem_load_2[51]), "=f"(_tmem_load_2[52]), "=f"(_tmem_load_2[53]), "=f"(_tmem_load_2[54]), "=f"(_tmem_load_2[55]), "=f"(_tmem_load_2[56]), "=f"(_tmem_load_2[57]), "=f"(_tmem_load_2[58]), "=f"(_tmem_load_2[59]), "=f"(_tmem_load_2[60]), "=f"(_tmem_load_2[61]), "=f"(_tmem_load_2[62]), "=f"(_tmem_load_2[63])
                    : "r"(lane_addr_1));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x64.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, [%64];"
                    : "=f"(_tmem_load_2[64]), "=f"(_tmem_load_2[65]), "=f"(_tmem_load_2[66]), "=f"(_tmem_load_2[67]), "=f"(_tmem_load_2[68]), "=f"(_tmem_load_2[69]), "=f"(_tmem_load_2[70]), "=f"(_tmem_load_2[71]), "=f"(_tmem_load_2[72]), "=f"(_tmem_load_2[73]), "=f"(_tmem_load_2[74]), "=f"(_tmem_load_2[75]), "=f"(_tmem_load_2[76]), "=f"(_tmem_load_2[77]), "=f"(_tmem_load_2[78]), "=f"(_tmem_load_2[79]), "=f"(_tmem_load_2[80]), "=f"(_tmem_load_2[81]), "=f"(_tmem_load_2[82]), "=f"(_tmem_load_2[83]), "=f"(_tmem_load_2[84]), "=f"(_tmem_load_2[85]), "=f"(_tmem_load_2[86]), "=f"(_tmem_load_2[87]), "=f"(_tmem_load_2[88]), "=f"(_tmem_load_2[89]), "=f"(_tmem_load_2[90]), "=f"(_tmem_load_2[91]), "=f"(_tmem_load_2[92]), "=f"(_tmem_load_2[93]), "=f"(_tmem_load_2[94]), "=f"(_tmem_load_2[95]), "=f"(_tmem_load_2[96]), "=f"(_tmem_load_2[97]), "=f"(_tmem_load_2[98]), "=f"(_tmem_load_2[99]), "=f"(_tmem_load_2[100]), "=f"(_tmem_load_2[101]), "=f"(_tmem_load_2[102]), "=f"(_tmem_load_2[103]), "=f"(_tmem_load_2[104]), "=f"(_tmem_load_2[105]), "=f"(_tmem_load_2[106]), "=f"(_tmem_load_2[107]), "=f"(_tmem_load_2[108]), "=f"(_tmem_load_2[109]), "=f"(_tmem_load_2[110]), "=f"(_tmem_load_2[111]), "=f"(_tmem_load_2[112]), "=f"(_tmem_load_2[113]), "=f"(_tmem_load_2[114]), "=f"(_tmem_load_2[115]), "=f"(_tmem_load_2[116]), "=f"(_tmem_load_2[117]), "=f"(_tmem_load_2[118]), "=f"(_tmem_load_2[119]), "=f"(_tmem_load_2[120]), "=f"(_tmem_load_2[121]), "=f"(_tmem_load_2[122]), "=f"(_tmem_load_2[123]), "=f"(_tmem_load_2[124]), "=f"(_tmem_load_2[125]), "=f"(_tmem_load_2[126]), "=f"(_tmem_load_2[127])
                    : "r"(lane_addr_1 + 64));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                if (elect_sync()) {
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    asm volatile(
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                        :: "r"((epilogue_done_addr) & 0xFEFFFFFF) : "memory");
                }
                _phase_mainloop_done ^= 1;
                if (skp_lnc_2 > 0) {
                    float _tmem_load_3[32];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(_tmem_load_3[0]), "=f"(_tmem_load_3[1]), "=f"(_tmem_load_3[2]), "=f"(_tmem_load_3[3]), "=f"(_tmem_load_3[4]), "=f"(_tmem_load_3[5]), "=f"(_tmem_load_3[6]), "=f"(_tmem_load_3[7]), "=f"(_tmem_load_3[8]), "=f"(_tmem_load_3[9]), "=f"(_tmem_load_3[10]), "=f"(_tmem_load_3[11]), "=f"(_tmem_load_3[12]), "=f"(_tmem_load_3[13]), "=f"(_tmem_load_3[14]), "=f"(_tmem_load_3[15]), "=f"(_tmem_load_3[16]), "=f"(_tmem_load_3[17]), "=f"(_tmem_load_3[18]), "=f"(_tmem_load_3[19]), "=f"(_tmem_load_3[20]), "=f"(_tmem_load_3[21]), "=f"(_tmem_load_3[22]), "=f"(_tmem_load_3[23]), "=f"(_tmem_load_3[24]), "=f"(_tmem_load_3[25]), "=f"(_tmem_load_3[26]), "=f"(_tmem_load_3[27]), "=f"(_tmem_load_3[28]), "=f"(_tmem_load_3[29]), "=f"(_tmem_load_3[30]), "=f"(_tmem_load_3[31])
                        : "r"(skp_pre_addr));
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    #pragma unroll
                    for (int i_9 = 0; i_9 < 32; i_9++) {
                        _tmem_load_2[i_9] = _tmem_load_3[i_9] + _tmem_load_2[i_9];
                    }
                    float _tmem_load_4[32];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(_tmem_load_4[0]), "=f"(_tmem_load_4[1]), "=f"(_tmem_load_4[2]), "=f"(_tmem_load_4[3]), "=f"(_tmem_load_4[4]), "=f"(_tmem_load_4[5]), "=f"(_tmem_load_4[6]), "=f"(_tmem_load_4[7]), "=f"(_tmem_load_4[8]), "=f"(_tmem_load_4[9]), "=f"(_tmem_load_4[10]), "=f"(_tmem_load_4[11]), "=f"(_tmem_load_4[12]), "=f"(_tmem_load_4[13]), "=f"(_tmem_load_4[14]), "=f"(_tmem_load_4[15]), "=f"(_tmem_load_4[16]), "=f"(_tmem_load_4[17]), "=f"(_tmem_load_4[18]), "=f"(_tmem_load_4[19]), "=f"(_tmem_load_4[20]), "=f"(_tmem_load_4[21]), "=f"(_tmem_load_4[22]), "=f"(_tmem_load_4[23]), "=f"(_tmem_load_4[24]), "=f"(_tmem_load_4[25]), "=f"(_tmem_load_4[26]), "=f"(_tmem_load_4[27]), "=f"(_tmem_load_4[28]), "=f"(_tmem_load_4[29]), "=f"(_tmem_load_4[30]), "=f"(_tmem_load_4[31])
                        : "r"(skp_pre_addr + 32));
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    #pragma unroll
                    for (int i_10 = 0; i_10 < 32; i_10++) {
                        _tmem_load_2[32 + i_10] = _tmem_load_4[i_10] + _tmem_load_2[32 + i_10];
                    }
                    float _tmem_load_5[32];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(_tmem_load_5[0]), "=f"(_tmem_load_5[1]), "=f"(_tmem_load_5[2]), "=f"(_tmem_load_5[3]), "=f"(_tmem_load_5[4]), "=f"(_tmem_load_5[5]), "=f"(_tmem_load_5[6]), "=f"(_tmem_load_5[7]), "=f"(_tmem_load_5[8]), "=f"(_tmem_load_5[9]), "=f"(_tmem_load_5[10]), "=f"(_tmem_load_5[11]), "=f"(_tmem_load_5[12]), "=f"(_tmem_load_5[13]), "=f"(_tmem_load_5[14]), "=f"(_tmem_load_5[15]), "=f"(_tmem_load_5[16]), "=f"(_tmem_load_5[17]), "=f"(_tmem_load_5[18]), "=f"(_tmem_load_5[19]), "=f"(_tmem_load_5[20]), "=f"(_tmem_load_5[21]), "=f"(_tmem_load_5[22]), "=f"(_tmem_load_5[23]), "=f"(_tmem_load_5[24]), "=f"(_tmem_load_5[25]), "=f"(_tmem_load_5[26]), "=f"(_tmem_load_5[27]), "=f"(_tmem_load_5[28]), "=f"(_tmem_load_5[29]), "=f"(_tmem_load_5[30]), "=f"(_tmem_load_5[31])
                        : "r"(skp_pre_addr + 64));
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    #pragma unroll
                    for (int i_11 = 0; i_11 < 32; i_11++) {
                        _tmem_load_2[64 + i_11] = _tmem_load_5[i_11] + _tmem_load_2[64 + i_11];
                    }
                    float _tmem_load_6[16];
                    tmem_ld_x16(&_tmem_load_6[0], skp_pre_addr + 96);
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    #pragma unroll
                    for (int i_12 = 0; i_12 < 16; i_12++) {
                        _tmem_load_2[96 + i_12] = _tmem_load_6[i_12] + _tmem_load_2[96 + i_12];
                    }
                    float post[16];
                    if (skp_lnc_2 > 0) {
                        int skp_pvi0 = (skp_lt_2 * sk_ksplit * 2 * 8 * 32 + skp_wvec) * 32 + lane;
                        float parts_p[16];
                        #pragma unroll
                        for (int skp_v_16 = 0; skp_v_16 < 4; skp_v_16++) {
                            int skp_vi_16 = skp_pvi0 + (28 + skp_v_16) * 32;
                            {
                                float4 _v4 = *reinterpret_cast<const float4*>(sk_partials + ((unsigned long long)skp_vi_16 * 4) + 0);
                                parts_p[4 * skp_v_16 + 0] = _v4.x;
                                parts_p[4 * skp_v_16 + 1] = _v4.y;
                                parts_p[4 * skp_v_16 + 2] = _v4.z;
                                parts_p[4 * skp_v_16 + 3] = _v4.w;
                            }
                        }
                        #pragma unroll
                        for (int i_13 = 0; i_13 < 16; i_13++) {
                            post[i_13] = parts_p[i_13];
                        }
                    }
                    if (skp_lnc_2 > 1) {
                        int skp_pvi0_0 = ((skp_lt_2 * sk_ksplit + 1) * 2 * 8 * 32 + skp_wvec) * 32 + lane;
                        float parts_p_0[16];
                        #pragma unroll
                        for (int skp_v_17 = 0; skp_v_17 < 4; skp_v_17++) {
                            int skp_vi_17 = skp_pvi0_0 + (28 + skp_v_17) * 32;
                            {
                                float4 _v4 = *reinterpret_cast<const float4*>(sk_partials + ((unsigned long long)skp_vi_17 * 4) + 0);
                                parts_p_0[4 * skp_v_17 + 0] = _v4.x;
                                parts_p_0[4 * skp_v_17 + 1] = _v4.y;
                                parts_p_0[4 * skp_v_17 + 2] = _v4.z;
                                parts_p_0[4 * skp_v_17 + 3] = _v4.w;
                            }
                        }
                        #pragma unroll
                        for (int i_9_1 = 0; i_9_1 < 16; i_9_1++) {
                            post[i_9_1] = post[i_9_1] + parts_p_0[i_9_1];
                        }
                    }
                    if (skp_lnc_2 > 2) {
                        int skp_pvi0_1 = ((skp_lt_2 * sk_ksplit + 2) * 2 * 8 * 32 + skp_wvec) * 32 + lane;
                        float parts_p_1[16];
                        #pragma unroll
                        for (int skp_v_18 = 0; skp_v_18 < 4; skp_v_18++) {
                            int skp_vi_18 = skp_pvi0_1 + (28 + skp_v_18) * 32;
                            {
                                float4 _v4 = *reinterpret_cast<const float4*>(sk_partials + ((unsigned long long)skp_vi_18 * 4) + 0);
                                parts_p_1[4 * skp_v_18 + 0] = _v4.x;
                                parts_p_1[4 * skp_v_18 + 1] = _v4.y;
                                parts_p_1[4 * skp_v_18 + 2] = _v4.z;
                                parts_p_1[4 * skp_v_18 + 3] = _v4.w;
                            }
                        }
                        #pragma unroll
                        for (int i_10_1 = 0; i_10_1 < 16; i_10_1++) {
                            post[i_10_1] = post[i_10_1] + parts_p_1[i_10_1];
                        }
                    }
                    if (skp_lnc_2 > 3) {
                        int skp_pvi0_2 = ((skp_lt_2 * sk_ksplit + 3) * 2 * 8 * 32 + skp_wvec) * 32 + lane;
                        float parts_p_2[16];
                        #pragma unroll
                        for (int skp_v_19 = 0; skp_v_19 < 4; skp_v_19++) {
                            int skp_vi_19 = skp_pvi0_2 + (28 + skp_v_19) * 32;
                            {
                                float4 _v4 = *reinterpret_cast<const float4*>(sk_partials + ((unsigned long long)skp_vi_19 * 4) + 0);
                                parts_p_2[4 * skp_v_19 + 0] = _v4.x;
                                parts_p_2[4 * skp_v_19 + 1] = _v4.y;
                                parts_p_2[4 * skp_v_19 + 2] = _v4.z;
                                parts_p_2[4 * skp_v_19 + 3] = _v4.w;
                            }
                        }
                        #pragma unroll
                        for (int i_11_1 = 0; i_11_1 < 16; i_11_1++) {
                            post[i_11_1] = post[i_11_1] + parts_p_2[i_11_1];
                        }
                    }
                    #pragma unroll
                    for (int i_14 = 0; i_14 < 16; i_14++) {
                        _tmem_load_2[112 + i_14] = post[i_14] + _tmem_load_2[112 + i_14];
                    }
                }
                unsigned int tmem_load_0_bf16_1[64];
                #pragma unroll
                for (int _lp = 0; _lp < 64; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_2[_lp*2 + 0], _tmem_load_2[_lp*2+1 + 0]));
                    tmem_load_0_bf16_1[_lp] = *(uint32_t*)&_bf2;
                }
                unsigned int group_base_1 = smem_out_addr + (unsigned int)(col_part * 8192);
                int swz_1 = local_row >> 1 & 3;
                #pragma unroll
                for (int c_1 = 0; c_1 < 4; c_1++) {
                    unsigned int ts_base_1 = group_base_1;
                    unsigned int ts_row_1 = ts_base_1 + (unsigned int)(local_row * 64);
                    if ((warp - 2) % 4 == 0) {
                        asm volatile("cp.async.bulk.wait_group.read 0;");
                    }
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    asm volatile("barrier.sync %0, 128;" :: "r"(1 + col_part) : "memory");
                    #pragma unroll
                    for (int q_1 = 0; q_1 < 4; q_1++) {
                        int s0_1 = c_1 * 16 + q_1 * 4;
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(ts_row_1 + (unsigned int)((q_1 ^ swz_1) * 16)), "r"(tmem_load_0_bf16_1[s0_1]), "r"(tmem_load_0_bf16_1[s0_1 + 1]), "r"(tmem_load_0_bf16_1[s0_1 + 2]), "r"(tmem_load_0_bf16_1[s0_1 + 3]));
                    }
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    asm volatile("barrier.sync %0, 128;" :: "r"(1 + col_part) : "memory");
                    if ((warp - 2) % 4 == 0) {
                        if (elect_sync()) {
                            tma_store_2d((&OUT), col_base_1 + c_1 * 32, off_m_4, ts_base_1);
                            asm volatile("cp.async.bulk.commit_group;");
                        }
                    }
                }
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
