/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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
// Portions derived from DeepGEMM, Copyright (c) 2025 DeepSeek.
// DeepGEMM portions are licensed under MIT; see DEEPGEMM_NOTICE.txt in this directory.

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
#define TMEM_NCOLS 256
#define TMEM_ACC_OFFSET 0
#define TMEM_SF_A_OFFSET 128
#define TMEM_SF_B_OFFSET 132
#define NUM_MAIN_PIPE_STAGES 10
#define NUM_ACC_PIPE_STAGES 2
#define SMEM_CD_OFF 0
#define SMEM_CD_STAGE_BYTES 4096
#define SMEM_CD_STRIDE 4096
#define SMEM_A_OFF 8192
#define SMEM_A_STAGE_BYTES 4096
#define SMEM_A_STRIDE 4096
#define SMEM_B_OFF 49152
#define SMEM_B_STAGE_BYTES 16384
#define SMEM_B_STRIDE 16384
#define SMEM_SFA_OFF 212992
#define SMEM_SFA_STAGE_BYTES 512
#define SMEM_SFA_STRIDE 512
#define SMEM_SFB_OFF 218112
#define SMEM_SFB_STAGE_BYTES 512
#define SMEM_SFB_STRIDE 512
#define SMEM_TOTAL 223616
#define THREADS 256

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



__device__ __forceinline__ void mbarrier_arrive_expect_tx(int mbar_addr, uint32_t bytes) {
    asm volatile(
        "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
        :: "r"(mbar_addr), "r"(bytes) : "memory");
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


__device__ __forceinline__ void tma_2d_gmem2smem(
    int dst, const void *tmap_ptr, int x, int y, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global"
        ".mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3}], [%4];"
        :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y),
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

__global__ __launch_bounds__(256, 1) __cluster_dims__(2,1,1) void
kernel_cake_deepgemm_batched_gemm_c1767ea2214a0b0df04b(const __grid_constant__ CUtensorMap A, const __grid_constant__ CUtensorMap B, const __grid_constant__ CUtensorMap SFA, const __grid_constant__ CUtensorMap SFB, const __grid_constant__ CUtensorMap D, unsigned int* __restrict__ SFD, unsigned int M, unsigned int grid_m, unsigned long long sfd_stride, float alpha)
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

    const int mbar_base = smem + 223232;
    #define full_addr (mbar_base + 0)
    #define sf_full_addr (mbar_base + 80)
    #define empty_addr (mbar_base + 160)
    #define acc_full_addr (mbar_base + 240)
    #define acc_empty_addr (mbar_base + 256)
    #define overlap_empty_addr (mbar_base + 272)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    const unsigned int clusters_x = gridDim.x / 2;
    const unsigned int cluster_id = ((blockIdx.z * gridDim.y + blockIdx.y) * clusters_x) + blockIdx.x / 2;
    const unsigned int num_clusters = clusters_x * gridDim.y * gridDim.z;

    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));

    // Kernel setup ops
    __nv_bfloat16* cd = reinterpret_cast<__nv_bfloat16*>(smem_raw + 0);
    const int cd_addr = smem + 0;
    uint8_t* a = reinterpret_cast<uint8_t*>(smem_raw + 8192);
    const int a_addr = smem + 8192;
    uint8_t* b = reinterpret_cast<uint8_t*>(smem_raw + 49152);
    const int b_addr = smem + 49152;
    unsigned int* sfa = reinterpret_cast<unsigned int*>(smem_raw + 212992);
    const int sfa_addr = smem + 212992;
    unsigned int* sfb = reinterpret_cast<unsigned int*>(smem_raw + 218112);
    const int sfb_addr = smem + 218112;
    asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    if (warp == 0) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&A))) : "memory"); }
    if (warp == 0) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&B))) : "memory"); }
    if (warp == 0) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&SFA))) : "memory"); }
    if (warp == 0) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&SFB))) : "memory"); }
    if (warp == 0) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&D))) : "memory"); }
    if (warp == 0) {
        if (elect_sync()) {
            if ((unsigned int)bid < 8 * grid_m * 8) {
                unsigned int head = (unsigned int)bid / (grid_m * 8);
                unsigned int local = (unsigned int)bid % (grid_m * 8);
                unsigned int n_pair = local / (2 * grid_m);
                unsigned int off_m = local / 2 % grid_m * 64;
                unsigned int off_n = (n_pair * 2 + local % 2) * 128;
                asm volatile("cp.async.bulk.prefetch.tensor.3d.L2.global.tile [%0, {%1, %2, %3}];" :: "l"((uint64_t)((&A))), "r"((int)(0)), "r"((int)(off_m + (unsigned int)(cta_rank * 32))), "r"((int)(head)) : "memory");
                asm volatile("cp.async.bulk.prefetch.tensor.3d.L2.global.tile [%0, {%1, %2, %3}];" :: "l"((uint64_t)((&B))), "r"((int)(0)), "r"((int)(off_n)), "r"((int)(head)) : "memory");
                asm volatile("cp.async.bulk.prefetch.tensor.3d.L2.global.tile [%0, {%1, %2, %3}];" :: "l"((uint64_t)((&A))), "r"((int)(128)), "r"((int)(off_m + (unsigned int)(cta_rank * 32))), "r"((int)(head)) : "memory");
                asm volatile("cp.async.bulk.prefetch.tensor.3d.L2.global.tile [%0, {%1, %2, %3}];" :: "l"((uint64_t)((&B))), "r"((int)(128)), "r"((int)(off_n)), "r"((int)(head)) : "memory");
                asm volatile("cp.async.bulk.prefetch.tensor.3d.L2.global.tile [%0, {%1, %2, %3}];" :: "l"((uint64_t)((&A))), "r"((int)(256)), "r"((int)(off_m + (unsigned int)(cta_rank * 32))), "r"((int)(head)) : "memory");
                asm volatile("cp.async.bulk.prefetch.tensor.3d.L2.global.tile [%0, {%1, %2, %3}];" :: "l"((uint64_t)((&B))), "r"((int)(256)), "r"((int)(off_n)), "r"((int)(head)) : "memory");
                asm volatile("cp.async.bulk.prefetch.tensor.3d.L2.global.tile [%0, {%1, %2, %3}];" :: "l"((uint64_t)((&A))), "r"((int)(384)), "r"((int)(off_m + (unsigned int)(cta_rank * 32))), "r"((int)(head)) : "memory");
                asm volatile("cp.async.bulk.prefetch.tensor.3d.L2.global.tile [%0, {%1, %2, %3}];" :: "l"((uint64_t)((&B))), "r"((int)(384)), "r"((int)(off_n)), "r"((int)(head)) : "memory");
            }
        }
    }

    // Mbarrier init (6 pipeline groups, 0 ordered-sequence groups, 36 barriers)
    // Mbarriers at smem_raw[223232..223520)

    if (warp == 1) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'main_pipe' ---
            // full: 10 barriers, init_count=66
            mbarrier_init(smem + 223232, 66);
            mbarrier_init(smem + 223240, 66);
            mbarrier_init(smem + 223248, 66);
            mbarrier_init(smem + 223256, 66);
            mbarrier_init(smem + 223264, 66);
            mbarrier_init(smem + 223272, 66);
            mbarrier_init(smem + 223280, 66);
            mbarrier_init(smem + 223288, 66);
            mbarrier_init(smem + 223296, 66);
            mbarrier_init(smem + 223304, 66);
            // sf_full: 10 barriers, init_count=1
            mbarrier_init(smem + 223312, 1);
            mbarrier_init(smem + 223320, 1);
            mbarrier_init(smem + 223328, 1);
            mbarrier_init(smem + 223336, 1);
            mbarrier_init(smem + 223344, 1);
            mbarrier_init(smem + 223352, 1);
            mbarrier_init(smem + 223360, 1);
            mbarrier_init(smem + 223368, 1);
            mbarrier_init(smem + 223376, 1);
            mbarrier_init(smem + 223384, 1);
            // empty: 10 barriers, init_count=1
            mbarrier_init(smem + 223392, 1);
            mbarrier_init(smem + 223400, 1);
            mbarrier_init(smem + 223408, 1);
            mbarrier_init(smem + 223416, 1);
            mbarrier_init(smem + 223424, 1);
            mbarrier_init(smem + 223432, 1);
            mbarrier_init(smem + 223440, 1);
            mbarrier_init(smem + 223448, 1);
            mbarrier_init(smem + 223456, 1);
            mbarrier_init(smem + 223464, 1);
            // --- pipeline 'acc_pipe' ---
            // acc_full: 2 barriers, init_count=1
            mbarrier_init(smem + 223472, 1);
            mbarrier_init(smem + 223480, 1);
            // acc_empty: 2 barriers, init_count=256
            mbarrier_init(smem + 223488, 256);
            mbarrier_init(smem + 223496, 256);
            // overlap_empty: 2 barriers, init_count=256
            mbarrier_init(smem + 223504, 256);
            mbarrier_init(smem + 223512, 256);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    // TMEM alloc (256 columns, 136 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 223520);
    if (warp == 2) {
        int _tmem_hold = smem + 223520;
        asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(256) : "memory");
        __syncwarp();
    }

    __syncthreads();
    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_acc = taddr;
    const int tmem_sf_a = taddr + 128;
    const int tmem_sf_b = taddr + 132;
    asm volatile("griddepcontrol.wait;" ::: "memory");

    // ---- Role: load ----
    if (warp == 0) {
        { // load_main
            unsigned int stage = 0;
            unsigned int _phase_empty = 1;
            #pragma unroll 1
            for (int tile = bid; tile < 8 * grid_m * 8; tile += num_bids) {
                unsigned int head_1 = (unsigned int)tile / (grid_m * 8);
                unsigned int local_1 = (unsigned int)tile % (grid_m * 8);
                unsigned int n_pair_1 = local_1 / (2 * grid_m);
                unsigned int off_m_1 = local_1 / 2 % grid_m * 64;
                unsigned int off_n_1 = (n_pair_1 * 2 + local_1 % 2) * 128;
                #pragma unroll 4
                for (int kt = 0; kt < 32; kt++) {
                    mbarrier_wait(empty_addr + (stage) * 8, _phase_empty);
                    if (elect_sync()) {
                        unsigned int sf_bytes = 0;
                        if (kt % 4 == 0) {
                            tma_2d_gmem2smem(sfa_addr + stage * 512, (&SFA), off_m_1, head_1 * 8 + (unsigned int)(kt / 4), sf_full_addr + (stage) * 8);
                            tma_2d_gmem2smem(sfb_addr + stage * 512, (&SFB), off_n_1, head_1 * 8 + (unsigned int)(kt / 4), sf_full_addr + (stage) * 8);
                            sf_bytes = 1024;
                        }
                        mbarrier_arrive_expect_tx(sf_full_addr + (stage) * 8, sf_bytes);
                        tma_3d_gmem2smem_cta2(a_addr + stage * 4096, (&A), kt * 128, off_m_1 + (unsigned int)(cta_rank * 32), head_1, ((full_addr + (stage) * 8) & 0xFEFFFFFF));
                        tma_3d_gmem2smem_cta2(b_addr + stage * 16384, (&B), kt * 128, off_n_1, head_1, ((full_addr + (stage) * 8) & 0xFEFFFFFF));
                        if (cta_rank == 0) {
                            mbarrier_arrive_expect_tx(full_addr + (stage) * 8, 40960);
                        } else {
                            asm volatile(
                                "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                                :: "r"((full_addr + (stage) * 8) & 0xFEFFFFFF) : "memory");
                        }
                    }
                    stage += 1;
                    if (stage == 10) { stage = 0; _phase_empty ^= 1; }
                }
            }
        }
    }
    // ---- Role: mma ----
    if (warp == 1) {
        { // mma_main
            unsigned int stage_1 = 0;
            unsigned int acc_stage = 0;
            unsigned int completed = 0;
            unsigned int _phase_acc_empty = 1;
            unsigned int _phase_full = 0;
            if (cta_rank == 0) {
                #pragma unroll 1
                for (int tile_1 = bid; tile_1 < 8 * grid_m * 8; tile_1 += num_bids) {
                    mbarrier_wait(acc_empty_addr + (acc_stage) * 8, _phase_acc_empty);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    #pragma unroll 4
                    for (int kt_1 = 0; kt_1 < 32; kt_1++) {
                        mbarrier_wait(full_addr + (stage_1) * 8, _phase_full);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        if (elect_sync()) {
                            if (kt_1 % 4 == 0) {
                                tcgen05_cp_32x128b_warpx4_cta2(tmem_sf_a, make_sf_cp_desc_lo_sbo128((((sfa_addr) >> 4) + (stage_1) * 32)));
                                tcgen05_cp_32x128b_warpx4_cta2(tmem_sf_b, make_sf_cp_desc_lo_sbo128((((sfb_addr) >> 4) + (stage_1) * 32)));
                            }
                        }
                        __syncwarp();
                        if (elect_sync()) {
                            int _mma_a_lo_0 = (((b_addr) >> 4) & 0x3FFF) + (stage_1) * 1024;
                            int _mma_b_lo_0 = (((a_addr) >> 4) & 0x3FFF) + (stage_1) * 256;
                            {
                                uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_0) | ((uint64_t)0x40004040 << 32);
                                uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_0) | ((uint64_t)0x40004040 << 32);

                                tcgen05_mma_mxf8_bs_cta2((tmem_acc + (acc_stage * 64)), a_desc + 0, b_desc + 0,
                                    (0x10900000U | ((((uint32_t)(kt_1) % 4U)) << 29) | ((((uint32_t)(kt_1) % 4U)) << 4)), tmem_sf_b, tmem_sf_a, ((kt_1 == 0) ? 0 : 1));
                                tcgen05_mma_mxf8_bs_cta2((tmem_acc + (acc_stage * 64)), a_desc + 2, b_desc + 2,
                                    (0x10900000U | ((((uint32_t)(kt_1) % 4U)) << 29) | ((((uint32_t)(kt_1) % 4U)) << 4)), tmem_sf_b, tmem_sf_a, 1);
                                tcgen05_mma_mxf8_bs_cta2((tmem_acc + (acc_stage * 64)), a_desc + 4, b_desc + 4,
                                    (0x10900000U | ((((uint32_t)(kt_1) % 4U)) << 29) | ((((uint32_t)(kt_1) % 4U)) << 4)), tmem_sf_b, tmem_sf_a, 1);
                                tcgen05_mma_mxf8_bs_cta2((tmem_acc + (acc_stage * 64)), a_desc + 6, b_desc + 6,
                                    (0x10900000U | ((((uint32_t)(kt_1) % 4U)) << 29) | ((((uint32_t)(kt_1) % 4U)) << 4)), tmem_sf_b, tmem_sf_a, 1);
                            }
                        }
                        __syncwarp();
                        elect_commit_cg2_multicast(empty_addr + (stage_1) * 8, (uint16_t)(3));
                        if (kt_1 == 31) {
                            elect_commit_cg2_multicast(acc_full_addr + (acc_stage) * 8, (uint16_t)(3));
                        }
                        __syncwarp();
                        stage_1 += 1;
                        if (stage_1 == 10) { stage_1 = 0; _phase_full ^= 1; }
                    }
                    acc_stage += 1;
                    if (acc_stage == 2) { acc_stage = 0; _phase_acc_empty ^= 1; }
                    completed += 1;
                }
                if (completed > 0) {
                    mbarrier_wait(acc_empty_addr + ((completed - 1) % 2) * 8, (completed - 1) / 2 & 1);
                }
            }
        }
    }
    // ---- Role: transpose ----
    if (warp == 2) {
        { // transpose_main
            unsigned int stage_2 = 0;
            unsigned int _phase_sf_full = 0;
            #pragma unroll 1
            for (int tile_2 = bid; tile_2 < 8 * grid_m * 8; tile_2 += num_bids) {
                #pragma unroll 1
                for (int kt_2 = 0; kt_2 < 32; kt_2++) {
                    mbarrier_wait(sf_full_addr + (stage_2) * 8, _phase_sf_full);
                    if (kt_2 % 4 == 0) {
                        unsigned int _sf_v[4];
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&_sf_v[0])) : "r"(sfa_addr + stage_2 * 512 + (unsigned int)(((lane >> 3 ^ 0) * 32 + lane) * 4)));
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&_sf_v[1])) : "r"(sfa_addr + stage_2 * 512 + (unsigned int)(((lane >> 3 ^ 1) * 32 + lane) * 4)));
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&_sf_v[2])) : "r"(sfa_addr + stage_2 * 512 + (unsigned int)(((lane >> 3 ^ 2) * 32 + lane) * 4)));
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&_sf_v[3])) : "r"(sfa_addr + stage_2 * 512 + (unsigned int)(((lane >> 3 ^ 3) * 32 + lane) * 4)));
                        __syncwarp();
                        asm volatile("st.shared.b32 [%0], %1;" :: "r"(sfa_addr + stage_2 * 512 + (unsigned int)((lane * 4 + (lane >> 3 ^ 0)) * 4)), "r"((_sf_v[0])));
                        asm volatile("st.shared.b32 [%0], %1;" :: "r"(sfa_addr + stage_2 * 512 + (unsigned int)((lane * 4 + (lane >> 3 ^ 1)) * 4)), "r"((_sf_v[1])));
                        asm volatile("st.shared.b32 [%0], %1;" :: "r"(sfa_addr + stage_2 * 512 + (unsigned int)((lane * 4 + (lane >> 3 ^ 2)) * 4)), "r"((_sf_v[2])));
                        asm volatile("st.shared.b32 [%0], %1;" :: "r"(sfa_addr + stage_2 * 512 + (unsigned int)((lane * 4 + (lane >> 3 ^ 3)) * 4)), "r"((_sf_v[3])));
                        unsigned int _sf_v_0[4];
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&_sf_v_0[0])) : "r"(sfb_addr + stage_2 * 512 + (unsigned int)(((lane >> 3 ^ 0) * 32 + lane) * 4)));
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&_sf_v_0[1])) : "r"(sfb_addr + stage_2 * 512 + (unsigned int)(((lane >> 3 ^ 1) * 32 + lane) * 4)));
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&_sf_v_0[2])) : "r"(sfb_addr + stage_2 * 512 + (unsigned int)(((lane >> 3 ^ 2) * 32 + lane) * 4)));
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&_sf_v_0[3])) : "r"(sfb_addr + stage_2 * 512 + (unsigned int)(((lane >> 3 ^ 3) * 32 + lane) * 4)));
                        __syncwarp();
                        asm volatile("st.shared.b32 [%0], %1;" :: "r"(sfb_addr + stage_2 * 512 + (unsigned int)((lane * 4 + (lane >> 3 ^ 0)) * 4)), "r"((_sf_v_0[0])));
                        asm volatile("st.shared.b32 [%0], %1;" :: "r"(sfb_addr + stage_2 * 512 + (unsigned int)((lane * 4 + (lane >> 3 ^ 1)) * 4)), "r"((_sf_v_0[1])));
                        asm volatile("st.shared.b32 [%0], %1;" :: "r"(sfb_addr + stage_2 * 512 + (unsigned int)((lane * 4 + (lane >> 3 ^ 2)) * 4)), "r"((_sf_v_0[2])));
                        asm volatile("st.shared.b32 [%0], %1;" :: "r"(sfb_addr + stage_2 * 512 + (unsigned int)((lane * 4 + (lane >> 3 ^ 3)) * 4)), "r"((_sf_v_0[3])));
                    }
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    asm volatile(
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                        :: "r"((full_addr + (stage_2) * 8) & 0xFEFFFFFF) : "memory");
                    stage_2 += 1;
                    if (stage_2 == 10) { stage_2 = 0; _phase_sf_full ^= 1; }
                }
            }
        }
    }
    // ---- Role: store ----
    if (warp >= 4 && warp <= 7) {
        { // store_main
            unsigned int acc_stage_1 = 0;
            unsigned int store_stage = 0;
            unsigned int warp_0 = warp - 4;
            unsigned int _phase_acc_full = 0;
            #pragma unroll 1
            for (int tile_3 = bid; tile_3 < 8 * grid_m * 8; tile_3 += num_bids) {
                unsigned int head_2 = (unsigned int)tile_3 / (grid_m * 8);
                unsigned int local_2 = (unsigned int)tile_3 % (grid_m * 8);
                unsigned int n_pair_2 = local_2 / (2 * grid_m);
                unsigned int off_m_2 = local_2 / 2 % grid_m * 64;
                unsigned int off_n_2 = (n_pair_2 * 2 + local_2 % 2) * 128;
                mbarrier_wait(acc_full_addr + (acc_stage_1) * 8, _phase_acc_full);
                asm volatile("tcgen05.fence::after_thread_sync;");
                #pragma unroll
                for (int store_idx = 0; store_idx < 4; store_idx++) {
                    if (warp_0 == 0) {
                        asm volatile("cp.async.bulk.wait_group 1;");
                    }
                    asm volatile("barrier.sync 0, 128;" ::: "memory");
                    #pragma unroll
                    for (int load_idx = 0; load_idx < 2; load_idx++) {
                        unsigned int address = taddr + acc_stage_1 * 64 + (unsigned int)(store_idx * 16) + (unsigned int)(load_idx * 8);
                        float _tmem_load_0[4];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x256b.x1.b32"
                            " {%0, %1, %2, %3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[3]))
                            : "r"(address));
                        float _tmem_load_1[4];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x256b.x1.b32"
                            " {%0, %1, %2, %3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[3]))
                            : "r"(address | 1048576));
                        asm volatile("tcgen05.wait::ld.sync.aligned;");
                        if (store_idx == 3 && load_idx == 1) {
                            asm volatile("tcgen05.fence::before_thread_sync;");
                            asm volatile(
                                "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                                :: "r"((acc_empty_addr + (acc_stage_1) * 8) & 0xFEFFFFFF) : "memory");
                        }
                        uint32_t _tmem_load_0_bf16[2];
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 0], _tmem_load_0[_lp*2+1 + 0]));
                            _tmem_load_0_bf16[_lp] = *(uint32_t*)&_bf2;
                        }
                        uint32_t _tmem_load_1_bf16[2];
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_1[_lp*2 + 0], _tmem_load_1[_lp*2+1 + 0]));
                            _tmem_load_1_bf16[_lp] = *(uint32_t*)&_bf2;
                        }
                        unsigned int row = lane % 8;
                        unsigned int col = warp_0 % 2 * 4 + (unsigned int)(lane / 8);
                        unsigned int write_addr = cd_addr + store_stage * 4096 + warp_0 / 2 * 16 * 128 + (unsigned int)(load_idx * 8 * 128) + row * 128 + (col ^ row) * 16;
                        uint32_t _stmatrix_addr_0 = static_cast<uint32_t>(write_addr);
                        asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                            :: "r"(_stmatrix_addr_0), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_bf16[0])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_bf16[1])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_bf16[0])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_bf16[1]))
                            : "memory");
                    }
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    asm volatile("barrier.sync 0, 128;" ::: "memory");
                    if (warp == 4) {
                        if (elect_sync()) {
                            tma_store_3d((&D), off_n_2, off_m_2 + (unsigned int)(store_idx * 16), head_2, cd_addr + store_stage * 4096);
                            tma_store_3d((&D), off_n_2 + 64, off_m_2 + (unsigned int)(store_idx * 16), head_2, cd_addr + store_stage * 4096 + 2048);
                            asm volatile("cp.async.bulk.commit_group;");
                        }
                    }
                    store_stage = (store_stage + 1) % 2;
                    __syncwarp();
                }
                acc_stage_1 += 1;
                if (acc_stage_1 == 2) { acc_stage_1 = 0; _phase_acc_full ^= 1; }
            }
        }
    }

    // Kernel teardown ops
    asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    if (warp == 0) {
        asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;" :: "r"(0), "r"(256));
    }

    // Cleanup
}

} // extern "C"
