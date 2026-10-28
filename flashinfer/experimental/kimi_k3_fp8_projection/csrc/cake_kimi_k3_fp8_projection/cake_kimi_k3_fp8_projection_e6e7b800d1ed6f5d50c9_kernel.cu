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
#define TMEM_NCOLS 280
#define TMEM_ACCUM_OFFSET 0
#define TMEM_TMEM_SFA_OFFSET 256
#define TMEM_TMEM_SFB_OFFSET 264
#define NUM_TMA_PIPE_STAGES 3
#define NUM_MAINLOOP_PIPE_STAGES 1
#define NUM_WORK_PIPE_STAGES 4
#define NUM_SEG_PIPE_STAGES 1
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
#define SMEM_SMEM_RA_OFF 1024
#define SMEM_SMEM_RA_STAGE_BYTES 16384
#define SMEM_SMEM_RA_STRIDE 16384
#define SMEM_SMEM_RSFA_OFF 66560
#define SMEM_SMEM_RSFA_STAGE_BYTES 512
#define SMEM_SMEM_RSFA_STRIDE 512
#define SMEM_SMEM_RB_OFF 68608
#define SMEM_SMEM_RB_STAGE_BYTES 32768
#define SMEM_SMEM_RB_STRIDE 34816
#define SMEM_SMEM_RSFB_OFF 101376
#define SMEM_SMEM_RSFB_STAGE_BYTES 2048
#define SMEM_SMEM_RSFB_STRIDE 34816
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





__device__ __forceinline__ uint64_t make_sf_cp_desc_sbo128(int addr) {
    const int SBO = 128;
    return desc_encode(addr)
         | (desc_encode(SBO) << 32ULL)
         | (1ULL << 46ULL);
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
kernel_cake_kimi_k3_fp8_projection_e6e7b800d1ed6f5d50c9(const __grid_constant__ CUtensorMap A, const __grid_constant__ CUtensorMap B, const __grid_constant__ CUtensorMap SFA, const __grid_constant__ CUtensorMap SFB, __nv_bfloat16* __restrict__ out, const __grid_constant__ CUtensorMap OUT, int M, int m_tiles, int n_tiles, int n_valid, int ldo, int store_vec, int num_k_iters, int sf_k_tiles, __nv_bfloat16* __restrict__ x, int K, float* __restrict__ sk_partials, unsigned int* __restrict__ sk_flags, int sk_pairs, int sk_rem, int sk_ksplit, int sk_dp)
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
    #define seg_full_addr (mbar_base + 112)
    #define seg_empty_addr (mbar_base + 120)

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
    uint8_t* smem_ra = reinterpret_cast<uint8_t*>(smem_raw + 1024);
    const int smem_ra_addr = smem + 1024;
    uint8_t* smem_rsfa = reinterpret_cast<uint8_t*>(smem_raw + 66560);
    const int smem_rsfa_addr = smem + 66560;
    uint8_t* smem_rb = reinterpret_cast<uint8_t*>(smem_raw + 68608);
    const int smem_rb_addr = smem + 68608;
    uint8_t* smem_rsfb = reinterpret_cast<uint8_t*>(smem_raw + 101376);
    const int smem_rsfb_addr = smem + 101376;

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
            // --- pipeline 'seg_pipe' ---
            // seg_full: 1 barriers, init_count=2
            mbarrier_init(smem + 112, 2);
            // seg_empty: 1 barriers, init_count=1
            mbarrier_init(smem + 120, 1);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();
    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");

    // TMEM alloc (512 columns, 280 used)
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
    const int tmem_tmem_sfa = taddr + 256;
    const int tmem_tmem_sfb = taddr + 264;

    // ---- Role: load ----
    if (warp == 0) {
        { // load_main
            int num_cluster_tiles = m_tiles / CTA_GROUP * n_tiles;
            unsigned int load_stage = 0;
            unsigned int work_stage = 0;
            asm volatile("griddepcontrol.wait;" ::: "memory");
            unsigned int _phase_seg_empty = 1;
            unsigned int _phase_mma_done_h0 = 1;
            if (elect_sync()) {
                unsigned int this_bid = bid;
                int ars_p = (int)(bid / CTA_GROUP);
                int ars_lo = ars_p * num_cluster_tiles / sk_pairs;
                int ars_hi = (ars_p + 1) * num_cluster_tiles / sk_pairs;
                unsigned int ars_seg_tx = (unsigned int)(num_k_iters * 33792);
                #pragma unroll 1
                for (int ars_s = ars_lo / n_tiles; ars_s < (ars_hi + n_tiles - 1) / n_tiles; ars_s++) {
                    int ars_tb = ((ars_lo > ars_s * n_tiles) ? ars_lo : ars_s * n_tiles);
                    int ars_te = ((ars_hi < (ars_s + 1) * n_tiles) ? ars_hi : (ars_s + 1) * n_tiles);
                    int ars_off_m = (ars_s * 2 + cta_rank) * BLOCK_M;
                    int ars_sfa_row = (ars_s * 2 + cta_rank) * sf_k_tiles;
                    mbarrier_wait(seg_empty_addr, _phase_seg_empty);
                    #pragma unroll 1
                    for (int ars_kk = 0; ars_kk < 2 * num_k_iters; ars_kk++) {
                        tma_3d_gmem2smem_cta2(smem_ra_addr + (0 * 4 + (unsigned int)ars_kk) * 16384, (&A), 0, ars_off_m, ars_kk, ((seg_full_addr) & 0xFEFFFFFF));
                        tma_3d_gmem2smem_cta2(smem_rsfa_addr + (0 * 4 + (unsigned int)ars_kk) * 512, (&SFA), 0, 0, ars_sfa_row + ars_kk, ((seg_full_addr) & 0xFEFFFFFF));
                    }
                    asm volatile(
                        "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                        :: "r"((seg_full_addr) & 0xFEFFFFFF), "r"((uint32_t)(ars_seg_tx)) : "memory");
                    _phase_seg_empty ^= 1;
                    #pragma unroll 1
                    for (int ars_t = ars_tb; ars_t < ars_te; ars_t++) {
                        int ars_n = ars_t - ars_s * n_tiles;
                        int ars_w0 = (ars_n * 2 + cta_rank) * (2 * num_k_iters);
                        int ars_sfb_row = ars_n * sf_k_tiles;
                        #pragma unroll 1
                        for (int ars_ik = 0; ars_ik < num_k_iters; ars_ik++) {
                            mbarrier_wait(mma_done_h0_addr + (load_stage) * 8, _phase_mma_done_h0);
                            tma_3d_gmem2smem_cta2(smem_rb_addr + load_stage * 34816 + 0, (&B), 0, 0, ars_w0 + ars_ik * 2 + 0, ((tma_full_h0_addr + (load_stage) * 8) & 0xFEFFFFFF));
                            tma_3d_gmem2smem_cta2(smem_rsfb_addr + load_stage * 34816 + 0, (&SFB), 0, 0, ars_sfb_row + ars_ik * 2 + 0, ((tma_full_h0_addr + (load_stage) * 8) & 0xFEFFFFFF));
                            asm volatile(
                                "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                                :: "r"((tma_full_h0_addr + (load_stage) * 8) & 0xFEFFFFFF), "r"((uint32_t)(17408)) : "memory");
                            mbarrier_wait(mma_done_h1_addr + (load_stage) * 8, _phase_mma_done_h0);
                            tma_3d_gmem2smem_cta2(smem_rb_addr + load_stage * 34816 + 16384, (&B), 0, 0, ars_w0 + ars_ik * 2 + 1, ((tma_full_h1_addr + (load_stage) * 8) & 0xFEFFFFFF));
                            tma_3d_gmem2smem_cta2(smem_rsfb_addr + load_stage * 34816 + 1024, (&SFB), 0, 0, ars_sfb_row + ars_ik * 2 + 1, ((tma_full_h1_addr + (load_stage) * 8) & 0xFEFFFFFF));
                            asm volatile(
                                "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                                :: "r"((tma_full_h1_addr + (load_stage) * 8) & 0xFEFFFFFF), "r"((uint32_t)(17408)) : "memory");
                            load_stage += 1;
                            if (load_stage == 3) { load_stage = 0; _phase_mma_done_h0 ^= 1; }
                        }
                    }
                }
                asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
                #pragma unroll 1
                for (int _ = 0; _ < 3; _++) {
                    mbarrier_wait(mma_done_h0_addr + (load_stage) * 8, _phase_mma_done_h0);
                    mbarrier_wait(mma_done_h1_addr + (load_stage) * 8, _phase_mma_done_h0);
                    load_stage += 1;
                    if (load_stage == 3) { load_stage = 0; _phase_mma_done_h0 ^= 1; }
                }
                mbarrier_wait(seg_empty_addr, _phase_seg_empty);
            }
        }
    }
    // ---- Role: mma ----
    if (warp == 1) {
        { // mma_main
            int num_cluster_tiles_1 = m_tiles / CTA_GROUP * n_tiles;
            unsigned int mma_tma_stage = 0;
            unsigned int work_stage_1 = 0;
            unsigned int _phase_seg_full = 0;
            unsigned int _phase_epilogue_done = 1;
            unsigned int _phase_tma_full_h0 = 0;
            if (cta_rank == 0) {
                int ars_p_1 = (int)(bid / CTA_GROUP);
                int ars_lo_1 = ars_p_1 * num_cluster_tiles_1 / sk_pairs;
                int ars_hi_1 = (ars_p_1 + 1) * num_cluster_tiles_1 / sk_pairs;
                #pragma unroll 1
                for (int ars_s_1 = ars_lo_1 / n_tiles; ars_s_1 < (ars_hi_1 + n_tiles - 1) / n_tiles; ars_s_1++) {
                    int ars_tb_1 = ((ars_lo_1 > ars_s_1 * n_tiles) ? ars_lo_1 : ars_s_1 * n_tiles);
                    int ars_te_1 = ((ars_hi_1 < (ars_s_1 + 1) * n_tiles) ? ars_hi_1 : (ars_s_1 + 1) * n_tiles);
                    mbarrier_wait(seg_full_addr, _phase_seg_full);
                    #pragma unroll 1
                    for (int _ars_t = ars_tb_1; _ars_t < ars_te_1; _ars_t++) {
                        mbarrier_wait(epilogue_done_addr, _phase_epilogue_done);
                        #pragma unroll 1
                        for (int ars_ik_1 = 0; ars_ik_1 < num_k_iters; ars_ik_1++) {
                            int ars_init = ((ars_ik_1 == 0) ? 1 : 0);
                            mbarrier_wait(tma_full_h0_addr + (mma_tma_stage) * 8, _phase_tma_full_h0);
                            asm volatile("tcgen05.fence::after_thread_sync;");
                            uint32_t _elected_0 = elect_sync();
                            tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfa, make_sf_cp_desc_sbo128(smem_rsfa_addr + (unsigned int)((ars_ik_1 * 2 + 0) * 512)), _elected_0);
                            tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfb, make_sf_cp_desc_lo_sbo128((((smem_rsfb_addr) >> 4) + (mma_tma_stage) * 2176)), _elected_0);
                            tcgen05_cp_32x128b_warpx4_cta2_leader((tmem_tmem_sfb + 4), make_sf_cp_desc_lo_sbo128((((smem_rsfb_addr) >> 4) + (mma_tma_stage) * 2176 + 32)), _elected_0);
                            int _smem_desc_lo_1 = (((smem_ra_addr) >> 4) & 0x3FFF) + (ars_ik_1 * 2 + 0) * 1024;
                            int _smem_desc_lo_2 = (((smem_rb_addr) >> 4) & 0x3FFF) + (mma_tma_stage) * 2176;
                            {
                                uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_1) | ((uint64_t)0x40004040 << 32);
                                uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_2) | ((uint64_t)0x40004040 << 32);

                                tcgen05_mma_mxf8_bs_cta2_leader(tmem_accum, a_desc + 0, b_desc + 0,
                                    0x10c00000U, tmem_tmem_sfa, tmem_tmem_sfb, ((ars_init) ? 0 : 1), _elected_0);
                                tcgen05_mma_mxf8_bs_cta2_leader(tmem_accum, a_desc + 2, b_desc + 2,
                                    0x30c00010U, tmem_tmem_sfa, tmem_tmem_sfb, 1, _elected_0);
                                tcgen05_mma_mxf8_bs_cta2_leader(tmem_accum, a_desc + 4, b_desc + 4,
                                    0x50c00020U, tmem_tmem_sfa, tmem_tmem_sfb, 1, _elected_0);
                                tcgen05_mma_mxf8_bs_cta2_leader(tmem_accum, a_desc + 6, b_desc + 6,
                                    0x70c00030U, tmem_tmem_sfa, tmem_tmem_sfb, 1, _elected_0);
                            }
                            elect_commit_cg2_multicast(mma_done_h0_addr + (mma_tma_stage) * 8, (uint16_t)(3));
                            mbarrier_wait(tma_full_h1_addr + (mma_tma_stage) * 8, _phase_tma_full_h0);
                            asm volatile("tcgen05.fence::after_thread_sync;");
                            uint32_t _elected_3 = elect_sync();
                            tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfa + 4, make_sf_cp_desc_sbo128(smem_rsfa_addr + (unsigned int)((ars_ik_1 * 2 + 1) * 512)), _elected_3);
                            tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfb + 8, make_sf_cp_desc_lo_sbo128((((smem_rsfb_addr) >> 4) + (mma_tma_stage) * 2176 + 64)), _elected_3);
                            tcgen05_cp_32x128b_warpx4_cta2_leader((tmem_tmem_sfb + 8 + 4), make_sf_cp_desc_lo_sbo128((((smem_rsfb_addr) >> 4) + (mma_tma_stage) * 2176 + 64 + 32)), _elected_3);
                            int _smem_desc_lo_4 = (((smem_ra_addr) >> 4) & 0x3FFF) + (ars_ik_1 * 2 + 1) * 1024;
                            int _smem_desc_lo_5 = (((smem_rb_addr + 16384) >> 4) & 0x3FFF) + (mma_tma_stage) * 2176;
                            {
                                uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_4) | ((uint64_t)0x40004040 << 32);
                                uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_5) | ((uint64_t)0x40004040 << 32);

                                tcgen05_mma_mxf8_bs_cta2_leader(tmem_accum, a_desc + 0, b_desc + 0,
                                    0x10c00000U, tmem_tmem_sfa + 4, tmem_tmem_sfb + 8, 1, _elected_3);
                                tcgen05_mma_mxf8_bs_cta2_leader(tmem_accum, a_desc + 2, b_desc + 2,
                                    0x30c00010U, tmem_tmem_sfa + 4, tmem_tmem_sfb + 8, 1, _elected_3);
                                tcgen05_mma_mxf8_bs_cta2_leader(tmem_accum, a_desc + 4, b_desc + 4,
                                    0x50c00020U, tmem_tmem_sfa + 4, tmem_tmem_sfb + 8, 1, _elected_3);
                                tcgen05_mma_mxf8_bs_cta2_leader(tmem_accum, a_desc + 6, b_desc + 6,
                                    0x70c00030U, tmem_tmem_sfa + 4, tmem_tmem_sfb + 8, 1, _elected_3);
                            }
                            elect_commit_cg2_multicast(mma_done_h1_addr + (mma_tma_stage) * 8, (uint16_t)(3));
                            mma_tma_stage += 1;
                            if (mma_tma_stage == 3) { mma_tma_stage = 0; _phase_tma_full_h0 ^= 1; }
                        }
                        elect_commit_cg2_multicast(mainloop_done_addr, (uint16_t)(3));
                        _phase_epilogue_done ^= 1;
                    }
                    elect_commit_cg2_multicast(seg_empty_addr, (uint16_t)(3));
                    _phase_seg_full ^= 1;
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
            int ars_p_2 = (int)(bid / CTA_GROUP);
            int ars_lo_2 = ars_p_2 * num_cluster_tiles_2 / sk_pairs;
            int ars_hi_2 = (ars_p_2 + 1) * num_cluster_tiles_2 / sk_pairs;
            unsigned int _phase_mainloop_done = 0;
            #pragma unroll 1
            for (int ars_s_2 = ars_lo_2 / n_tiles; ars_s_2 < (ars_hi_2 + n_tiles - 1) / n_tiles; ars_s_2++) {
                int ars_tb_2 = ((ars_lo_2 > ars_s_2 * n_tiles) ? ars_lo_2 : ars_s_2 * n_tiles);
                int ars_te_2 = ((ars_hi_2 < (ars_s_2 + 1) * n_tiles) ? ars_hi_2 : (ars_s_2 + 1) * n_tiles);
                #pragma unroll 1
                for (int ars_t_1 = ars_tb_2; ars_t_1 < ars_te_2; ars_t_1++) {
                    unsigned int ars_bid = (unsigned int)(ars_t_1 * 2) + (unsigned int)cta_rank;
                    int tiles_per_group = 2 * n_tiles;
                    int group = ars_bid / (unsigned int)tiles_per_group;
                    int first_m = group * 2;
                    int remaining = m_tiles - first_m;
                    int group_size = ((remaining >= 2) ? 2 : remaining);
                    int local = ars_bid % (unsigned int)tiles_per_group;
                    int bid_m = first_m + local % group_size;
                    int bid_n = local / group_size;
                    int off_m = bid_m * BLOCK_M;
                    int off_n = bid_n * 256;
                    int global_row = off_m + local_row;
                    int col0 = col_part * 128;
                    int lane_addr = taddr + (unsigned int)(epi_warp * 32 << 16) + (unsigned int)col0;
                    int col_base = off_n + col0;
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
                                tma_store_2d((&OUT), col_base + c * 32, off_m, ts_base);
                                asm volatile("cp.async.bulk.commit_group;");
                            }
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
