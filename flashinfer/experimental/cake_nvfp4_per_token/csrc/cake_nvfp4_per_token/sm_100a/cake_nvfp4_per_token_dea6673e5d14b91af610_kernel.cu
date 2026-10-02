/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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
#define TMEM_ACC_OFFSET 0
#define TMEM_TMEM_SFA_OFFSET 192
#define TMEM_TMEM_SFB_OFFSET 288
#define NUM_TMA_PIPE_STAGES 6
#define NUM_ACC_PIPE_STAGES 1
#define SMEM_SMEM_A_OFF 1024
#define SMEM_SMEM_A_STAGE_BYTES 16384
#define SMEM_SMEM_A_STRIDE 34816
#define SMEM_SMEM_B_OFF 17408
#define SMEM_SMEM_B_STAGE_BYTES 12288
#define SMEM_SMEM_B_STRIDE 34816
#define SMEM_SMEM_SFA_OFF 29696
#define SMEM_SMEM_SFA_STAGE_BYTES 2048
#define SMEM_SMEM_SFA_STRIDE 34816
#define SMEM_SMEM_V3_OFF 31744
#define SMEM_SMEM_V3_STAGE_BYTES 1024
#define SMEM_SMEM_V3_STRIDE 34816
#define SMEM_SMEM_V4_OFF 32768
#define SMEM_SMEM_V4_STAGE_BYTES 1024
#define SMEM_SMEM_V4_STRIDE 34816
#define SMEM_SMEM_V5_OFF 33792
#define SMEM_SMEM_V5_STAGE_BYTES 1024
#define SMEM_SMEM_V5_STRIDE 34816
#define SMEM_SMEM_V6_OFF 34816
#define SMEM_SMEM_V6_STAGE_BYTES 1024
#define SMEM_SMEM_V6_STRIDE 34816
#define SMEM_SMEM_V7_OFF 31744
#define SMEM_SMEM_V7_STAGE_BYTES 2048
#define SMEM_SMEM_V7_STRIDE 34816
#define SMEM_SMEM_V8_OFF 33792
#define SMEM_SMEM_V8_STAGE_BYTES 2048
#define SMEM_SMEM_V8_STRIDE 34816
#define SMEM_SMEM_OUT_OFF 209920
#define SMEM_SMEM_OUT_STAGE_BYTES 8192
#define SMEM_SMEM_OUT_STRIDE 8192
#define SMEM_TOTAL 226304
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

// Source-faithful relaxed CTA wait used only by a typed protocol that does
// not attach the PTX acquire qualifier, such as FA4's interior P-ready edge.
// Exact source ports may request the PTX suspendTimeHint operand explicitly.
// The hint is expressed in nanoseconds and is kept separate from the canonical
// no-hint CTA helper so unrelated schedules retain their existing retry path.
// Exact unqualified CTA wait used by source schedules whose PTX intentionally
// omits the acquire qualifier while retaining a typed suspendTimeHint operand.

__device__ __forceinline__ void tcgen05_mma_mxf4nvf4_bs_cta2(
    int taddr, uint64_t a_desc, uint64_t b_desc, uint32_t i_desc,
    int sfa_taddr, int sfb_taddr, int enable_input_d) {
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, %6, 0;\n\t"
        "tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::4X"
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


extern "C" {

__global__ __launch_bounds__(256, 1) __cluster_dims__(2,1,1) void
kernel_cake_nvfp4_per_token_dea6673e5d14b91af610(const __grid_constant__ CUtensorMap A, const __grid_constant__ CUtensorMap B, const __grid_constant__ CUtensorMap SFA, const __grid_constant__ CUtensorMap SFB, uint8_t* __restrict__ SFA_RAW, uint8_t* __restrict__ SFB_RAW, float* __restrict__ alpha, const __grid_constant__ CUtensorMap out, int M, int N, int K_tiles, int tok_tiles, int num_tiles, float* __restrict__ red_ws, unsigned int* __restrict__ red_flags, unsigned int* __restrict__ red_gen, int sk_tiles, int sk_pairs)
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
    #define tma_empty_addr (mbar_base + 48)
    #define acc_full_addr (mbar_base + 96)
    #define acc_empty_addr (mbar_base + 104)
    #define tmem_dealloc_bar_addr (mbar_base + 112)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    const unsigned int clusters_x = gridDim.x / 2;
    const unsigned int cluster_id = ((blockIdx.z * gridDim.y + blockIdx.y) * clusters_x) + blockIdx.x / 2;
    const unsigned int num_clusters = clusters_x * gridDim.y * gridDim.z;

    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));

    // Kernel setup ops
    uint8_t* smem_a = reinterpret_cast<uint8_t*>(smem_raw + 1024);
    const int smem_a_addr = smem + 1024;
    uint8_t* smem_b = reinterpret_cast<uint8_t*>(smem_raw + 17408);
    const int smem_b_addr = smem + 17408;
    uint8_t* smem_sfa = reinterpret_cast<uint8_t*>(smem_raw + 29696);
    const int smem_sfa_addr = smem + 29696;
    uint8_t* smem_v3 = reinterpret_cast<uint8_t*>(smem_raw + 31744);
    const int smem_v3_addr = smem + 31744;
    uint8_t* smem_v4 = reinterpret_cast<uint8_t*>(smem_raw + 32768);
    const int smem_v4_addr = smem + 32768;
    uint8_t* smem_v5 = reinterpret_cast<uint8_t*>(smem_raw + 33792);
    const int smem_v5_addr = smem + 33792;
    uint8_t* smem_v6 = reinterpret_cast<uint8_t*>(smem_raw + 34816);
    const int smem_v6_addr = smem + 34816;
    uint8_t* smem_v7 = reinterpret_cast<uint8_t*>(smem_raw + 31744);
    const int smem_v7_addr = smem + 31744;
    uint8_t* smem_v8 = reinterpret_cast<uint8_t*>(smem_raw + 33792);
    const int smem_v8_addr = smem + 33792;
    __nv_bfloat16* smem_out = reinterpret_cast<__nv_bfloat16*>(smem_raw + 209920);
    const int smem_out_addr = smem + 209920;
    if (warp == 2) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&A))) : "memory"); }
    if (warp == 2) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&B))) : "memory"); }
    if (warp == 2) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&SFA))) : "memory"); }
    if (warp == 2) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&SFB))) : "memory"); }
    if (warp == 2) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&out))) : "memory"); }

    // Mbarrier init (5 pipeline groups, 0 ordered-sequence groups, 15 barriers)
    // Mbarriers at smem_raw[0..120)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'tma_pipe' ---
            // tma_full: 6 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            mbarrier_init(smem + 8, 1);
            mbarrier_init(smem + 16, 1);
            mbarrier_init(smem + 24, 1);
            mbarrier_init(smem + 32, 1);
            mbarrier_init(smem + 40, 1);
            // tma_empty: 6 barriers, init_count=1
            mbarrier_init(smem + 48, 1);
            mbarrier_init(smem + 56, 1);
            mbarrier_init(smem + 64, 1);
            mbarrier_init(smem + 72, 1);
            mbarrier_init(smem + 80, 1);
            mbarrier_init(smem + 88, 1);
            // --- pipeline 'acc_pipe' ---
            // acc_full: 1 barriers, init_count=1
            mbarrier_init(smem + 96, 1);
            // acc_empty: 1 barriers, init_count=256
            mbarrier_init(smem + 104, 256);
            // tmem_dealloc_bar: 1 barriers, init_count=32
            mbarrier_init(smem + 112, 32);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");

    // TMEM alloc (512 columns, 480 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 120);
    if (warp == 2) {
        int _tmem_hold = smem + 120;
        asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(512) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;");
    }

    // Partial post-allocation TMEM rendezvous (192 threads on named barrier 2)
    if (warp == 1 || warp == 2 || warp == 4 || warp == 5 || warp == 6 || warp == 7) {
        asm volatile("barrier.sync.aligned %0, %1;" :: "r"(2), "r"(192) : "memory");
        asm volatile("tcgen05.fence::after_thread_sync;");
    }

    const int taddr = (warp == 1 || warp == 2 || warp == 4 || warp == 5 || warp == 6 || warp == 7) ? tmem_addr_storage[0] : 0;

    // Kernel post-init ops
    const int tmem_acc = taddr;
    const int tmem_tmem_sfa = taddr + 192;
    const int tmem_tmem_sfb = taddr + 288;

    // ---- Role: load ----
    if (warp == 0) {
        { // load_main
            unsigned int load_stage = 0;
            asm volatile("griddepcontrol.wait;" ::: "memory");
            int ld_first = num_tiles - sk_tiles;
            unsigned int _phase_tma_empty = 1;
            #pragma unroll 1
            for (unsigned int tile = cluster_id; tile < ld_first; tile += num_clusters) {
                int w_idx = tile / (unsigned int)tok_tiles;
                int tok_idx = tile - (unsigned int)(w_idx * tok_tiles);
                int a_rows = tok_idx * 256 + cta_rank * 128;
                int b_rows = w_idx * 192 + cta_rank * 96;
                int a_atom = a_rows / 128;
                int b_atom = w_idx * 192 / 128;
                #pragma unroll 1
                for (unsigned int k_tile = 0; k_tile < K_tiles; k_tile++) {
                    mbarrier_wait(tma_empty_addr + (load_stage) * 8, _phase_tma_empty);
                    if (cta_rank == 0) {
                        if (elect_sync()) {
                            asm volatile(
                                "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                                :: "r"((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF), "r"((uint32_t)(69632)) : "memory");
                        }
                    }
                    if (elect_sync()) {
                        tma_3d_gmem2smem_cta2(smem_a_addr + load_stage * 34816, (&A), 0, a_rows, k_tile, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                        tma_3d_gmem2smem_cta2(smem_sfa_addr + load_stage * 34816, (&SFA), 0, 4 * k_tile, a_atom, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                        asm volatile(
                            "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::2.L2::cache_hint"
                            " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                            :: "r"(smem_b_addr + load_stage * 34816), "l"((&B)), "r"(0), "r"(b_rows), "r"(k_tile),
                               "r"(((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF)), "l"(0x12F0000000000000ULL) : "memory");
                    }
                    if (cta_rank == 0) {
                        if (elect_sync()) {
                            asm volatile(
                                "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2.L2::cache_hint"
                                " [%0], [%1, {%2, %3, %4}], [%5], %6, %7;"
                                :: "r"(smem_v3_addr + load_stage * 34816), "l"((&SFB)), "r"(0), "r"(4 * k_tile), "r"(b_atom),
                                   "r"(((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(3)), "l"(0x12F0000000000000ULL) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2.L2::cache_hint"
                                " [%0], [%1, {%2, %3, %4}], [%5], %6, %7;"
                                :: "r"(smem_v4_addr + load_stage * 34816), "l"((&SFB)), "r"(0), "r"(4 * k_tile + 1), "r"(b_atom),
                                   "r"(((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(3)), "l"(0x12F0000000000000ULL) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2.L2::cache_hint"
                                " [%0], [%1, {%2, %3, %4}], [%5], %6, %7;"
                                :: "r"(smem_v5_addr + load_stage * 34816), "l"((&SFB)), "r"(0), "r"(4 * k_tile + 2), "r"(b_atom),
                                   "r"(((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(3)), "l"(0x12F0000000000000ULL) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2.L2::cache_hint"
                                " [%0], [%1, {%2, %3, %4}], [%5], %6, %7;"
                                :: "r"(smem_v6_addr + load_stage * 34816), "l"((&SFB)), "r"(0), "r"(4 * k_tile + 3), "r"(b_atom),
                                   "r"(((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(3)), "l"(0x12F0000000000000ULL) : "memory");
                        }
                    }
                    load_stage += 1;
                    if (load_stage == 6) { load_stage = 0; _phase_tma_empty ^= 1; }
                }
            }
            if (cluster_id < (unsigned int)sk_pairs) {
                int ld_t = (int)cluster_id / 6;
                int ld_j = (int)cluster_id - ld_t * 6;
                int ld_kb = ld_j * K_tiles / 6;
                int ld_ke = (ld_j + 1) * K_tiles / 6;
                int w_idx_1 = (ld_first + ld_t) / tok_tiles;
                int tok_idx_1 = ld_first + ld_t - w_idx_1 * tok_tiles;
                int a_rows_1 = tok_idx_1 * 256 + cta_rank * 128;
                int b_rows_1 = w_idx_1 * 192 + cta_rank * 96;
                int a_atom_1 = a_rows_1 / 128;
                int b_atom_1 = w_idx_1 * 192 / 128;
                #pragma unroll 1
                for (unsigned int k_tile_1 = ld_kb; k_tile_1 < ld_ke; k_tile_1++) {
                    mbarrier_wait(tma_empty_addr + (load_stage) * 8, _phase_tma_empty);
                    if (cta_rank == 0) {
                        if (elect_sync()) {
                            asm volatile(
                                "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                                :: "r"((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF), "r"((uint32_t)(69632)) : "memory");
                        }
                    }
                    if (elect_sync()) {
                        tma_3d_gmem2smem_cta2(smem_a_addr + load_stage * 34816, (&A), 0, a_rows_1, k_tile_1, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                        tma_3d_gmem2smem_cta2(smem_sfa_addr + load_stage * 34816, (&SFA), 0, 4 * k_tile_1, a_atom_1, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                        asm volatile(
                            "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::2.L2::cache_hint"
                            " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                            :: "r"(smem_b_addr + load_stage * 34816), "l"((&B)), "r"(0), "r"(b_rows_1), "r"(k_tile_1),
                               "r"(((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF)), "l"(0x12F0000000000000ULL) : "memory");
                    }
                    if (cta_rank == 0) {
                        if (elect_sync()) {
                            asm volatile(
                                "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2.L2::cache_hint"
                                " [%0], [%1, {%2, %3, %4}], [%5], %6, %7;"
                                :: "r"(smem_v3_addr + load_stage * 34816), "l"((&SFB)), "r"(0), "r"(4 * k_tile_1), "r"(b_atom_1),
                                   "r"(((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(3)), "l"(0x12F0000000000000ULL) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2.L2::cache_hint"
                                " [%0], [%1, {%2, %3, %4}], [%5], %6, %7;"
                                :: "r"(smem_v4_addr + load_stage * 34816), "l"((&SFB)), "r"(0), "r"(4 * k_tile_1 + 1), "r"(b_atom_1),
                                   "r"(((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(3)), "l"(0x12F0000000000000ULL) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2.L2::cache_hint"
                                " [%0], [%1, {%2, %3, %4}], [%5], %6, %7;"
                                :: "r"(smem_v5_addr + load_stage * 34816), "l"((&SFB)), "r"(0), "r"(4 * k_tile_1 + 2), "r"(b_atom_1),
                                   "r"(((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(3)), "l"(0x12F0000000000000ULL) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2.L2::cache_hint"
                                " [%0], [%1, {%2, %3, %4}], [%5], %6, %7;"
                                :: "r"(smem_v6_addr + load_stage * 34816), "l"((&SFB)), "r"(0), "r"(4 * k_tile_1 + 3), "r"(b_atom_1),
                                   "r"(((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(3)), "l"(0x12F0000000000000ULL) : "memory");
                        }
                    }
                    load_stage += 1;
                    if (load_stage == 6) { load_stage = 0; _phase_tma_empty ^= 1; }
                }
            }
        }
    }
    // ---- Role: mma ----
    if (warp == 1) {
        { // mma_main
            unsigned int mma_stage = 0;
            unsigned int acc_stage = 0;
            unsigned int completed = 0;
            unsigned int _phase_acc_empty = 1;
            unsigned int _phase_tma_full = 0;
            if (cta_rank == 0) {
                int mm_first = num_tiles - sk_tiles;
                #pragma unroll 1
                for (unsigned int tile_1 = cluster_id; tile_1 < mm_first; tile_1 += num_clusters) {
                    int acc_base = (int)acc_stage * 192;
                    int w_idx_mma = tile_1 / (unsigned int)tok_tiles;
                    int sfb_shift = w_idx_mma * 192 % 128 / 32;
                    mbarrier_wait(acc_empty_addr + (acc_stage) * 8, _phase_acc_empty);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    #pragma unroll 1
                    for (unsigned int k_tile_2 = 0; k_tile_2 < K_tiles; k_tile_2++) {
                        mbarrier_wait(tma_full_addr + (mma_stage) * 8, _phase_tma_full);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int sfa_col = (int)mma_stage * 16;
                        int sfb_col = (int)mma_stage * 32;
                        if (elect_sync()) {
                            tcgen05_cp_32x128b_warpx4_cta2((unsigned int)tmem_tmem_sfa + mma_stage * 16, make_sf_cp_desc_lo_sbo128((((smem_sfa_addr) >> 4) + (mma_stage) * 2176)));
                            tcgen05_cp_32x128b_warpx4_cta2(((unsigned int)tmem_tmem_sfa + mma_stage * 16 + 4), make_sf_cp_desc_lo_sbo128((((smem_sfa_addr) >> 4) + (mma_stage) * 2176 + 32)));
                            tcgen05_cp_32x128b_warpx4_cta2(((unsigned int)tmem_tmem_sfa + mma_stage * 16 + 8), make_sf_cp_desc_lo_sbo128((((smem_sfa_addr) >> 4) + (mma_stage) * 2176 + 64)));
                            tcgen05_cp_32x128b_warpx4_cta2(((unsigned int)tmem_tmem_sfa + mma_stage * 16 + 12), make_sf_cp_desc_lo_sbo128((((smem_sfa_addr) >> 4) + (mma_stage) * 2176 + 96)));
                        }
                        if (elect_sync()) {
                            tcgen05_cp_32x128b_warpx4_cta2((unsigned int)tmem_tmem_sfb + mma_stage * 2 * 16, make_sf_cp_desc_lo_sbo128((((smem_v7_addr) >> 4) + (mma_stage) * 2176)));
                            tcgen05_cp_32x128b_warpx4_cta2(((unsigned int)tmem_tmem_sfb + mma_stage * 2 * 16 + 4), make_sf_cp_desc_lo_sbo128((((smem_v7_addr) >> 4) + (mma_stage) * 2176 + 32)));
                            tcgen05_cp_32x128b_warpx4_cta2(((unsigned int)tmem_tmem_sfb + mma_stage * 2 * 16 + 8), make_sf_cp_desc_lo_sbo128((((smem_v7_addr) >> 4) + (mma_stage) * 2176 + 64)));
                            tcgen05_cp_32x128b_warpx4_cta2(((unsigned int)tmem_tmem_sfb + mma_stage * 2 * 16 + 12), make_sf_cp_desc_lo_sbo128((((smem_v7_addr) >> 4) + (mma_stage) * 2176 + 96)));
                        }
                        if (elect_sync()) {
                            tcgen05_cp_32x128b_warpx4_cta2((unsigned int)tmem_tmem_sfb + (mma_stage * 2 + 1) * 16, make_sf_cp_desc_lo_sbo128((((smem_v8_addr) >> 4) + (mma_stage) * 2176)));
                            tcgen05_cp_32x128b_warpx4_cta2(((unsigned int)tmem_tmem_sfb + (mma_stage * 2 + 1) * 16 + 4), make_sf_cp_desc_lo_sbo128((((smem_v8_addr) >> 4) + (mma_stage) * 2176 + 32)));
                            tcgen05_cp_32x128b_warpx4_cta2(((unsigned int)tmem_tmem_sfb + (mma_stage * 2 + 1) * 16 + 8), make_sf_cp_desc_lo_sbo128((((smem_v8_addr) >> 4) + (mma_stage) * 2176 + 64)));
                            tcgen05_cp_32x128b_warpx4_cta2(((unsigned int)tmem_tmem_sfb + (mma_stage * 2 + 1) * 16 + 12), make_sf_cp_desc_lo_sbo128((((smem_v8_addr) >> 4) + (mma_stage) * 2176 + 96)));
                        }
                        int init_flag = ((k_tile_2 == 0) ? 1 : 0);
                        int _mma_a_lo_0 = (((smem_a_addr) >> 4) & 0x3FFF) + (mma_stage) * 2176;
                        int _mma_b_lo_0 = (((smem_b_addr) >> 4) & 0x3FFF) + (mma_stage) * 2176;
                        if (elect_sync()) {
                            {
                                uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_0) | ((uint64_t)0x40004040 << 32);
                                uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_0) | ((uint64_t)0x40004040 << 32);

                                tcgen05_mma_mxf4nvf4_bs_cta2((tmem_acc + (acc_base)), a_desc + 0, b_desc + 0,
                                    0x10300480U, tmem_tmem_sfa + sfa_col + 0, tmem_tmem_sfb + (sfb_col + sfb_shift) + 0, ((((1) ? init_flag : 0)) ? 0 : 1));
                            }
                        }
                        int _mma_a_lo_1 = (((smem_a_addr + 32) >> 4) & 0x3FFF) + (mma_stage) * 2176;
                        int _mma_b_lo_1 = (((smem_b_addr + 32) >> 4) & 0x3FFF) + (mma_stage) * 2176;
                        if (elect_sync()) {
                            {
                                uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_1) | ((uint64_t)0x40004040 << 32);
                                uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_1) | ((uint64_t)0x40004040 << 32);

                                tcgen05_mma_mxf4nvf4_bs_cta2((tmem_acc + (acc_base)), a_desc + 0, b_desc + 0,
                                    0x10300480U, tmem_tmem_sfa + (sfa_col + 4) + 0, tmem_tmem_sfb + (sfb_col + 8 + sfb_shift) + 0, ((((0) ? init_flag : 0)) ? 0 : 1));
                            }
                        }
                        int _mma_a_lo_2 = (((smem_a_addr + 64) >> 4) & 0x3FFF) + (mma_stage) * 2176;
                        int _mma_b_lo_2 = (((smem_b_addr + 64) >> 4) & 0x3FFF) + (mma_stage) * 2176;
                        if (elect_sync()) {
                            {
                                uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_2) | ((uint64_t)0x40004040 << 32);
                                uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_2) | ((uint64_t)0x40004040 << 32);

                                tcgen05_mma_mxf4nvf4_bs_cta2((tmem_acc + (acc_base)), a_desc + 0, b_desc + 0,
                                    0x10300480U, tmem_tmem_sfa + (sfa_col + 8) + 0, tmem_tmem_sfb + (sfb_col + 16 + sfb_shift) + 0, ((((0) ? init_flag : 0)) ? 0 : 1));
                            }
                        }
                        int _mma_a_lo_3 = (((smem_a_addr + 96) >> 4) & 0x3FFF) + (mma_stage) * 2176;
                        int _mma_b_lo_3 = (((smem_b_addr + 96) >> 4) & 0x3FFF) + (mma_stage) * 2176;
                        if (elect_sync()) {
                            {
                                uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_3) | ((uint64_t)0x40004040 << 32);
                                uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_3) | ((uint64_t)0x40004040 << 32);

                                tcgen05_mma_mxf4nvf4_bs_cta2((tmem_acc + (acc_base)), a_desc + 0, b_desc + 0,
                                    0x10300480U, tmem_tmem_sfa + (sfa_col + 12) + 0, tmem_tmem_sfb + (sfb_col + 24 + sfb_shift) + 0, ((((0) ? init_flag : 0)) ? 0 : 1));
                            }
                        }
                        elect_commit_cg2_multicast(tma_empty_addr + (mma_stage) * 8, (uint16_t)(3));
                        mma_stage += 1;
                        if (mma_stage == 6) { mma_stage = 0; _phase_tma_full ^= 1; }
                    }
                    elect_commit_cg2_multicast(acc_full_addr + (acc_stage) * 8, (uint16_t)(3));
                    _phase_acc_empty ^= 1;
                    completed += 1;
                }
                if (cluster_id < (unsigned int)sk_pairs) {
                    int mm_t = (int)cluster_id / 6;
                    int mm_j = (int)cluster_id - mm_t * 6;
                    int mm_kb = mm_j * K_tiles / 6;
                    int mm_ke = (mm_j + 1) * K_tiles / 6;
                    int acc_base_1 = (int)acc_stage * 192;
                    int w_idx_mma_1 = (mm_first + mm_t) / tok_tiles;
                    int sfb_shift_1 = w_idx_mma_1 * 192 % 128 / 32;
                    mbarrier_wait(acc_empty_addr + (acc_stage) * 8, _phase_acc_empty);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    #pragma unroll 1
                    for (unsigned int k_tile_3 = mm_kb; k_tile_3 < mm_ke; k_tile_3++) {
                        mbarrier_wait(tma_full_addr + (mma_stage) * 8, _phase_tma_full);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int sfa_col_1 = (int)mma_stage * 16;
                        int sfb_col_1 = (int)mma_stage * 32;
                        if (elect_sync()) {
                            tcgen05_cp_32x128b_warpx4_cta2((unsigned int)tmem_tmem_sfa + mma_stage * 16, make_sf_cp_desc_lo_sbo128((((smem_sfa_addr) >> 4) + (mma_stage) * 2176)));
                            tcgen05_cp_32x128b_warpx4_cta2(((unsigned int)tmem_tmem_sfa + mma_stage * 16 + 4), make_sf_cp_desc_lo_sbo128((((smem_sfa_addr) >> 4) + (mma_stage) * 2176 + 32)));
                            tcgen05_cp_32x128b_warpx4_cta2(((unsigned int)tmem_tmem_sfa + mma_stage * 16 + 8), make_sf_cp_desc_lo_sbo128((((smem_sfa_addr) >> 4) + (mma_stage) * 2176 + 64)));
                            tcgen05_cp_32x128b_warpx4_cta2(((unsigned int)tmem_tmem_sfa + mma_stage * 16 + 12), make_sf_cp_desc_lo_sbo128((((smem_sfa_addr) >> 4) + (mma_stage) * 2176 + 96)));
                        }
                        if (elect_sync()) {
                            tcgen05_cp_32x128b_warpx4_cta2((unsigned int)tmem_tmem_sfb + mma_stage * 2 * 16, make_sf_cp_desc_lo_sbo128((((smem_v7_addr) >> 4) + (mma_stage) * 2176)));
                            tcgen05_cp_32x128b_warpx4_cta2(((unsigned int)tmem_tmem_sfb + mma_stage * 2 * 16 + 4), make_sf_cp_desc_lo_sbo128((((smem_v7_addr) >> 4) + (mma_stage) * 2176 + 32)));
                            tcgen05_cp_32x128b_warpx4_cta2(((unsigned int)tmem_tmem_sfb + mma_stage * 2 * 16 + 8), make_sf_cp_desc_lo_sbo128((((smem_v7_addr) >> 4) + (mma_stage) * 2176 + 64)));
                            tcgen05_cp_32x128b_warpx4_cta2(((unsigned int)tmem_tmem_sfb + mma_stage * 2 * 16 + 12), make_sf_cp_desc_lo_sbo128((((smem_v7_addr) >> 4) + (mma_stage) * 2176 + 96)));
                        }
                        if (elect_sync()) {
                            tcgen05_cp_32x128b_warpx4_cta2((unsigned int)tmem_tmem_sfb + (mma_stage * 2 + 1) * 16, make_sf_cp_desc_lo_sbo128((((smem_v8_addr) >> 4) + (mma_stage) * 2176)));
                            tcgen05_cp_32x128b_warpx4_cta2(((unsigned int)tmem_tmem_sfb + (mma_stage * 2 + 1) * 16 + 4), make_sf_cp_desc_lo_sbo128((((smem_v8_addr) >> 4) + (mma_stage) * 2176 + 32)));
                            tcgen05_cp_32x128b_warpx4_cta2(((unsigned int)tmem_tmem_sfb + (mma_stage * 2 + 1) * 16 + 8), make_sf_cp_desc_lo_sbo128((((smem_v8_addr) >> 4) + (mma_stage) * 2176 + 64)));
                            tcgen05_cp_32x128b_warpx4_cta2(((unsigned int)tmem_tmem_sfb + (mma_stage * 2 + 1) * 16 + 12), make_sf_cp_desc_lo_sbo128((((smem_v8_addr) >> 4) + (mma_stage) * 2176 + 96)));
                        }
                        int init_flag_1 = ((k_tile_3 == (unsigned int)mm_kb) ? 1 : 0);
                        int _mma_a_lo_4 = (((smem_a_addr) >> 4) & 0x3FFF) + (mma_stage) * 2176;
                        int _mma_b_lo_4 = (((smem_b_addr) >> 4) & 0x3FFF) + (mma_stage) * 2176;
                        if (elect_sync()) {
                            {
                                uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_4) | ((uint64_t)0x40004040 << 32);
                                uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_4) | ((uint64_t)0x40004040 << 32);

                                tcgen05_mma_mxf4nvf4_bs_cta2((tmem_acc + (acc_base_1)), a_desc + 0, b_desc + 0,
                                    0x10300480U, tmem_tmem_sfa + sfa_col_1 + 0, tmem_tmem_sfb + (sfb_col_1 + sfb_shift_1) + 0, ((((1) ? init_flag_1 : 0)) ? 0 : 1));
                            }
                        }
                        int _mma_a_lo_5 = (((smem_a_addr + 32) >> 4) & 0x3FFF) + (mma_stage) * 2176;
                        int _mma_b_lo_5 = (((smem_b_addr + 32) >> 4) & 0x3FFF) + (mma_stage) * 2176;
                        if (elect_sync()) {
                            {
                                uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_5) | ((uint64_t)0x40004040 << 32);
                                uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_5) | ((uint64_t)0x40004040 << 32);

                                tcgen05_mma_mxf4nvf4_bs_cta2((tmem_acc + (acc_base_1)), a_desc + 0, b_desc + 0,
                                    0x10300480U, tmem_tmem_sfa + (sfa_col_1 + 4) + 0, tmem_tmem_sfb + (sfb_col_1 + 8 + sfb_shift_1) + 0, ((((0) ? init_flag_1 : 0)) ? 0 : 1));
                            }
                        }
                        int _mma_a_lo_6 = (((smem_a_addr + 64) >> 4) & 0x3FFF) + (mma_stage) * 2176;
                        int _mma_b_lo_6 = (((smem_b_addr + 64) >> 4) & 0x3FFF) + (mma_stage) * 2176;
                        if (elect_sync()) {
                            {
                                uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_6) | ((uint64_t)0x40004040 << 32);
                                uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_6) | ((uint64_t)0x40004040 << 32);

                                tcgen05_mma_mxf4nvf4_bs_cta2((tmem_acc + (acc_base_1)), a_desc + 0, b_desc + 0,
                                    0x10300480U, tmem_tmem_sfa + (sfa_col_1 + 8) + 0, tmem_tmem_sfb + (sfb_col_1 + 16 + sfb_shift_1) + 0, ((((0) ? init_flag_1 : 0)) ? 0 : 1));
                            }
                        }
                        int _mma_a_lo_7 = (((smem_a_addr + 96) >> 4) & 0x3FFF) + (mma_stage) * 2176;
                        int _mma_b_lo_7 = (((smem_b_addr + 96) >> 4) & 0x3FFF) + (mma_stage) * 2176;
                        if (elect_sync()) {
                            {
                                uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_7) | ((uint64_t)0x40004040 << 32);
                                uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_7) | ((uint64_t)0x40004040 << 32);

                                tcgen05_mma_mxf4nvf4_bs_cta2((tmem_acc + (acc_base_1)), a_desc + 0, b_desc + 0,
                                    0x10300480U, tmem_tmem_sfa + (sfa_col_1 + 12) + 0, tmem_tmem_sfb + (sfb_col_1 + 24 + sfb_shift_1) + 0, ((((0) ? init_flag_1 : 0)) ? 0 : 1));
                            }
                        }
                        elect_commit_cg2_multicast(tma_empty_addr + (mma_stage) * 8, (uint16_t)(3));
                        mma_stage += 1;
                        if (mma_stage == 6) { mma_stage = 0; _phase_tma_full ^= 1; }
                    }
                    elect_commit_cg2_multicast(acc_full_addr + (acc_stage) * 8, (uint16_t)(3));
                    _phase_acc_empty ^= 1;
                    completed += 1;
                }
                asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
                if (completed > 0) {
                    mbarrier_wait(acc_empty_addr, completed - 1 & 1);
                }
            }
            int dealloc_peer_rank = cta_rank ^ 1;
            if (cta_rank != 0) {
                asm volatile(
                    "{\n\t"
                    ".reg .b32 remAddr32;\n\t"
                    "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                    "}"
                    :: "r"(tmem_dealloc_bar_addr), "r"(dealloc_peer_rank) : "memory");
            }
            mbarrier_wait(tmem_dealloc_bar_addr, 0);
            if (cta_rank == 0) {
                asm volatile(
                    "{\n\t"
                    ".reg .b32 remAddr32;\n\t"
                    "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                    "}"
                    :: "r"(tmem_dealloc_bar_addr), "r"(dealloc_peer_rank) : "memory");
            }
            int _tmem_dealloc_addr = *((volatile int*)tmem_addr_storage);
            asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;" :: "r"(_tmem_dealloc_addr), "r"(512));
        }
    }
    // ---- Role: prefetch ----
    if (warp == 2) {
        // idle — no tasks assigned
    }
    // ---- Role: sched ----
    if (warp == 3) {
        // idle — no tasks assigned
    }
    // ---- Role: epilogue ----
    if (warp >= 4 && warp <= 7) {
        { // epilogue_main
            const int epi_warp = warp - 4;
            int epi_row = epi_warp % 4 * 32 + lane;
            int epi_srow = epi_warp * 32 + lane;
            unsigned int acc_stage_1 = 0;
            unsigned int store_stage = 0;
            asm volatile("griddepcontrol.wait;" ::: "memory");
            int ep_first = num_tiles - sk_tiles;
            unsigned int _phase_acc_full = 0;
            #pragma unroll 1
            for (unsigned int tile_2 = cluster_id; tile_2 < ep_first; tile_2 += num_clusters) {
                int w_idx_2 = tile_2 / (unsigned int)tok_tiles;
                int tok_idx_2 = tile_2 - (unsigned int)(w_idx_2 * tok_tiles);
                int off_tok = tok_idx_2 * 256 + cta_rank * 128;
                int off_w = w_idx_2 * 192;
                int acc_base_2 = (int)acc_stage_1 * 192;
                int lane_addr = taddr + (unsigned int)(epi_warp * 32 << 16) + (unsigned int)acc_base_2;
                int _min_0 = ((off_tok + epi_row) < (M - 1) ? (off_tok + epi_row) : (M - 1));
                int tok_row = _min_0;
                float alpha_row = alpha[tok_row];
                mbarrier_wait(acc_full_addr + (acc_stage_1) * 8, _phase_acc_full);
                asm volatile("tcgen05.fence::after_thread_sync;");
                #pragma unroll
                for (int subtile = 0; subtile < 6; subtile++) {
                    int tmem_addr = lane_addr + subtile * 32;
                    float _tmem_load_0[32];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(_tmem_load_0[0]), "=f"(_tmem_load_0[1]), "=f"(_tmem_load_0[2]), "=f"(_tmem_load_0[3]), "=f"(_tmem_load_0[4]), "=f"(_tmem_load_0[5]), "=f"(_tmem_load_0[6]), "=f"(_tmem_load_0[7]), "=f"(_tmem_load_0[8]), "=f"(_tmem_load_0[9]), "=f"(_tmem_load_0[10]), "=f"(_tmem_load_0[11]), "=f"(_tmem_load_0[12]), "=f"(_tmem_load_0[13]), "=f"(_tmem_load_0[14]), "=f"(_tmem_load_0[15]), "=f"(_tmem_load_0[16]), "=f"(_tmem_load_0[17]), "=f"(_tmem_load_0[18]), "=f"(_tmem_load_0[19]), "=f"(_tmem_load_0[20]), "=f"(_tmem_load_0[21]), "=f"(_tmem_load_0[22]), "=f"(_tmem_load_0[23]), "=f"(_tmem_load_0[24]), "=f"(_tmem_load_0[25]), "=f"(_tmem_load_0[26]), "=f"(_tmem_load_0[27]), "=f"(_tmem_load_0[28]), "=f"(_tmem_load_0[29]), "=f"(_tmem_load_0[30]), "=f"(_tmem_load_0[31])
                        : "r"(tmem_addr));
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    if (subtile == 5) {
                        asm volatile("tcgen05.fence::before_thread_sync;");
                        asm volatile(
                            "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                            :: "r"((acc_empty_addr + (acc_stage_1) * 8) & 0xFEFFFFFF) : "memory");
                    }
                    {
                        float2 _pair_scale_even2_0 = make_float2(alpha_row, alpha_row);
                        float2 _pair_scale_odd2_0 = make_float2(alpha_row, alpha_row);
                        float2* _pair_scale_src2_0 = reinterpret_cast<float2*>(&_tmem_load_0[0]);
                        #if __CUDA_ARCH__ >= 1000
                        asm volatile("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_0[0]) : "l"(*(unsigned long long*)&_pair_scale_even2_0));
                        asm volatile("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_0[1]) : "l"(*(unsigned long long*)&_pair_scale_odd2_0));
                        asm volatile("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_0[2]) : "l"(*(unsigned long long*)&_pair_scale_even2_0));
                        asm volatile("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_0[3]) : "l"(*(unsigned long long*)&_pair_scale_odd2_0));
                        asm volatile("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_0[4]) : "l"(*(unsigned long long*)&_pair_scale_even2_0));
                        asm volatile("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_0[5]) : "l"(*(unsigned long long*)&_pair_scale_odd2_0));
                        asm volatile("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_0[6]) : "l"(*(unsigned long long*)&_pair_scale_even2_0));
                        asm volatile("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_0[7]) : "l"(*(unsigned long long*)&_pair_scale_odd2_0));
                        asm volatile("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_0[8]) : "l"(*(unsigned long long*)&_pair_scale_even2_0));
                        asm volatile("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_0[9]) : "l"(*(unsigned long long*)&_pair_scale_odd2_0));
                        asm volatile("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_0[10]) : "l"(*(unsigned long long*)&_pair_scale_even2_0));
                        asm volatile("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_0[11]) : "l"(*(unsigned long long*)&_pair_scale_odd2_0));
                        asm volatile("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_0[12]) : "l"(*(unsigned long long*)&_pair_scale_even2_0));
                        asm volatile("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_0[13]) : "l"(*(unsigned long long*)&_pair_scale_odd2_0));
                        asm volatile("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_0[14]) : "l"(*(unsigned long long*)&_pair_scale_even2_0));
                        asm volatile("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_0[15]) : "l"(*(unsigned long long*)&_pair_scale_odd2_0));
                        #else
                        _tmem_load_0[0] *= alpha_row;
                        _tmem_load_0[1] *= alpha_row;
                        _tmem_load_0[2] *= alpha_row;
                        _tmem_load_0[3] *= alpha_row;
                        _tmem_load_0[4] *= alpha_row;
                        _tmem_load_0[5] *= alpha_row;
                        _tmem_load_0[6] *= alpha_row;
                        _tmem_load_0[7] *= alpha_row;
                        _tmem_load_0[8] *= alpha_row;
                        _tmem_load_0[9] *= alpha_row;
                        _tmem_load_0[10] *= alpha_row;
                        _tmem_load_0[11] *= alpha_row;
                        _tmem_load_0[12] *= alpha_row;
                        _tmem_load_0[13] *= alpha_row;
                        _tmem_load_0[14] *= alpha_row;
                        _tmem_load_0[15] *= alpha_row;
                        _tmem_load_0[16] *= alpha_row;
                        _tmem_load_0[17] *= alpha_row;
                        _tmem_load_0[18] *= alpha_row;
                        _tmem_load_0[19] *= alpha_row;
                        _tmem_load_0[20] *= alpha_row;
                        _tmem_load_0[21] *= alpha_row;
                        _tmem_load_0[22] *= alpha_row;
                        _tmem_load_0[23] *= alpha_row;
                        _tmem_load_0[24] *= alpha_row;
                        _tmem_load_0[25] *= alpha_row;
                        _tmem_load_0[26] *= alpha_row;
                        _tmem_load_0[27] *= alpha_row;
                        _tmem_load_0[28] *= alpha_row;
                        _tmem_load_0[29] *= alpha_row;
                        _tmem_load_0[30] *= alpha_row;
                        _tmem_load_0[31] *= alpha_row;
                        #endif
                    }
                    uint32_t _tmem_load_0_bf16[16];
                    #pragma unroll
                    for (int _lp = 0; _lp < 16; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 0], _tmem_load_0[_lp*2+1 + 0]));
                        _tmem_load_0_bf16[_lp] = *(uint32_t*)&_bf2;
                    }
                    if (subtile > 0) {
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                        if (warp == 4) {
                            if (elect_sync()) {
                                tma_store_2d((&out), off_w + (subtile - 1) * 32, off_tok, smem_out_addr + store_stage * 8192);
                            }
                        }
                        if (warp == 4) {
                            asm volatile("cp.async.bulk.commit_group;");
                            asm volatile("cp.async.bulk.wait_group.read 1;");
                        }
                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                        store_stage = store_stage + 1;
                        if (store_stage == 2) {
                            store_stage = 0;
                        }
                    }
                    int out_stage_row = store_stage * 128 + (unsigned int)epi_srow;
                    unsigned int out_abs = smem_out_addr + (unsigned int)(out_stage_row * 64);
                    unsigned int out_swz = out_abs / 8 & 48;
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_out_addr + ((unsigned int)(out_stage_row * 64) + (0 ^ out_swz)))), "r"(_tmem_load_0_bf16[0]), "r"(_tmem_load_0_bf16[1]), "r"(_tmem_load_0_bf16[2]), "r"(_tmem_load_0_bf16[3]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_out_addr + ((unsigned int)(out_stage_row * 64) + (16 ^ out_swz)))), "r"(_tmem_load_0_bf16[4]), "r"(_tmem_load_0_bf16[5]), "r"(_tmem_load_0_bf16[6]), "r"(_tmem_load_0_bf16[7]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_out_addr + ((unsigned int)(out_stage_row * 64) + (32 ^ out_swz)))), "r"(_tmem_load_0_bf16[8]), "r"(_tmem_load_0_bf16[9]), "r"(_tmem_load_0_bf16[10]), "r"(_tmem_load_0_bf16[11]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_out_addr + ((unsigned int)(out_stage_row * 64) + (48 ^ out_swz)))), "r"(_tmem_load_0_bf16[12]), "r"(_tmem_load_0_bf16[13]), "r"(_tmem_load_0_bf16[14]), "r"(_tmem_load_0_bf16[15]) : "memory");
                }
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("barrier.sync 1, 128;" ::: "memory");
                if (warp == 4) {
                    if (elect_sync()) {
                        tma_store_2d((&out), off_w + 160, off_tok, smem_out_addr + store_stage * 8192);
                    }
                }
                if (warp == 4) {
                    asm volatile("cp.async.bulk.commit_group;");
                    asm volatile("cp.async.bulk.wait_group.read 1;");
                }
                asm volatile("barrier.sync 1, 128;" ::: "memory");
                store_stage = store_stage + 1;
                if (store_stage == 2) {
                    store_stage = 0;
                }
                _phase_acc_full ^= 1;
            }
            if (cluster_id < (unsigned int)sk_pairs) {
                int ep_t = (int)cluster_id / 6;
                int ep_j = (int)cluster_id - ep_t * 6;
                int w_idx_s = (ep_first + ep_t) / tok_tiles;
                int tok_idx_s = ep_first + ep_t - w_idx_s * tok_tiles;
                int off_tok_s = tok_idx_s * 256 + cta_rank * 128;
                int off_w_s = w_idx_s * 192;
                int acc_base_s = (int)acc_stage_1 * 192;
                int lane_addr_s = taddr + (unsigned int)(epi_warp * 32 << 16) + (unsigned int)acc_base_s;
                int _min_1 = ((off_tok_s + epi_row) < (M - 1) ? (off_tok_s + epi_row) : (M - 1));
                int tok_row_s = _min_1;
                float alpha_row_s = alpha[tok_row_s];
                int rank_s = (int)cta_rank;
                int tile_slots = ep_t * 6 * 49152 + rank_s * 24576;
                int own_slot = tile_slots + ep_j * 49152;
                int flag_base = ep_t * 6 * 2 + rank_s;
                int thread_off = epi_warp * 1024 + lane * 4;
                mbarrier_wait(acc_full_addr + (acc_stage_1) * 8, _phase_acc_full);
                asm volatile("tcgen05.fence::after_thread_sync;");
                #pragma unroll
                for (int subtile_1 = 0; subtile_1 < 6; subtile_1++) {
                    if (subtile_1 % 6 != ep_j) {
                        float _tmem_load_1[32];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                            : "=f"(_tmem_load_1[0]), "=f"(_tmem_load_1[1]), "=f"(_tmem_load_1[2]), "=f"(_tmem_load_1[3]), "=f"(_tmem_load_1[4]), "=f"(_tmem_load_1[5]), "=f"(_tmem_load_1[6]), "=f"(_tmem_load_1[7]), "=f"(_tmem_load_1[8]), "=f"(_tmem_load_1[9]), "=f"(_tmem_load_1[10]), "=f"(_tmem_load_1[11]), "=f"(_tmem_load_1[12]), "=f"(_tmem_load_1[13]), "=f"(_tmem_load_1[14]), "=f"(_tmem_load_1[15]), "=f"(_tmem_load_1[16]), "=f"(_tmem_load_1[17]), "=f"(_tmem_load_1[18]), "=f"(_tmem_load_1[19]), "=f"(_tmem_load_1[20]), "=f"(_tmem_load_1[21]), "=f"(_tmem_load_1[22]), "=f"(_tmem_load_1[23]), "=f"(_tmem_load_1[24]), "=f"(_tmem_load_1[25]), "=f"(_tmem_load_1[26]), "=f"(_tmem_load_1[27]), "=f"(_tmem_load_1[28]), "=f"(_tmem_load_1[29]), "=f"(_tmem_load_1[30]), "=f"(_tmem_load_1[31])
                            : "r"(lane_addr_s + subtile_1 * 32));
                        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                        int ws_off = own_slot + subtile_1 * 4096 + thread_off;
                        {
                            float4 _v4 = make_float4(_tmem_load_1[0 + 0], _tmem_load_1[0 + 1], _tmem_load_1[0 + 2], _tmem_load_1[0 + 3]);
                            *reinterpret_cast<float4*>(red_ws + ws_off) = _v4;
                        }
                        {
                            float4 _v4 = make_float4(_tmem_load_1[4 + 0], _tmem_load_1[4 + 1], _tmem_load_1[4 + 2], _tmem_load_1[4 + 3]);
                            *reinterpret_cast<float4*>(red_ws + ws_off + 128) = _v4;
                        }
                        {
                            float4 _v4 = make_float4(_tmem_load_1[8 + 0], _tmem_load_1[8 + 1], _tmem_load_1[8 + 2], _tmem_load_1[8 + 3]);
                            *reinterpret_cast<float4*>(red_ws + ws_off + 256) = _v4;
                        }
                        {
                            float4 _v4 = make_float4(_tmem_load_1[12 + 0], _tmem_load_1[12 + 1], _tmem_load_1[12 + 2], _tmem_load_1[12 + 3]);
                            *reinterpret_cast<float4*>(red_ws + ws_off + 384) = _v4;
                        }
                        {
                            float4 _v4 = make_float4(_tmem_load_1[16 + 0], _tmem_load_1[16 + 1], _tmem_load_1[16 + 2], _tmem_load_1[16 + 3]);
                            *reinterpret_cast<float4*>(red_ws + ws_off + 512) = _v4;
                        }
                        {
                            float4 _v4 = make_float4(_tmem_load_1[20 + 0], _tmem_load_1[20 + 1], _tmem_load_1[20 + 2], _tmem_load_1[20 + 3]);
                            *reinterpret_cast<float4*>(red_ws + ws_off + 640) = _v4;
                        }
                        {
                            float4 _v4 = make_float4(_tmem_load_1[24 + 0], _tmem_load_1[24 + 1], _tmem_load_1[24 + 2], _tmem_load_1[24 + 3]);
                            *reinterpret_cast<float4*>(red_ws + ws_off + 768) = _v4;
                        }
                        {
                            float4 _v4 = make_float4(_tmem_load_1[28 + 0], _tmem_load_1[28 + 1], _tmem_load_1[28 + 2], _tmem_load_1[28 + 3]);
                            *reinterpret_cast<float4*>(red_ws + ws_off + 896) = _v4;
                        }
                    }
                }
                asm volatile("barrier.sync 1, 128;" ::: "memory");
                unsigned int gen[1];
                if (warp == 4) {
                    if (elect_sync()) {
                        {
                            unsigned int* _gc_p = reinterpret_cast<unsigned int*>(red_flags) + (flag_base + ep_j * 2);
                            unsigned int _gc_old;
                            asm volatile("atom.release.gpu.global.add.u32 %0, [%1], 1;" : "=r"(_gc_old) : "l"(_gc_p) : "memory");
                        }
                        {
                            uint32_t _scalar_bits_1;
                            asm volatile("ld.global.cg.b32 %0, [%1];"
                                : "=r"(_scalar_bits_1) : "l"((const void*)(red_gen + (flag_base + ep_j * 2))) : "memory");
                            gen[0] = (unsigned int)_scalar_bits_1;
                        }
                    }
                }
                if (warp == 4) {
                    if (elect_sync()) {
                        if (0 != ep_j) {
                            {
                                unsigned int* _gca_p = reinterpret_cast<unsigned int*>(red_flags) + (flag_base);
                                while (true) {
                                    unsigned int _gca_v;
                                    asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(_gca_v) : "l"(_gca_p));
                                    if (static_cast<int32_t>(_gca_v - static_cast<uint32_t>(gen[0])) >= 0) break;
                                }
                            }
                        }
                        if (1 != ep_j) {
                            {
                                unsigned int* _gca_p = reinterpret_cast<unsigned int*>(red_flags) + (flag_base + 2);
                                while (true) {
                                    unsigned int _gca_v;
                                    asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(_gca_v) : "l"(_gca_p));
                                    if (static_cast<int32_t>(_gca_v - static_cast<uint32_t>(gen[0])) >= 0) break;
                                }
                            }
                        }
                        if (2 != ep_j) {
                            {
                                unsigned int* _gca_p = reinterpret_cast<unsigned int*>(red_flags) + (flag_base + 4);
                                while (true) {
                                    unsigned int _gca_v;
                                    asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(_gca_v) : "l"(_gca_p));
                                    if (static_cast<int32_t>(_gca_v - static_cast<uint32_t>(gen[0])) >= 0) break;
                                }
                            }
                        }
                        if (3 != ep_j) {
                            {
                                unsigned int* _gca_p = reinterpret_cast<unsigned int*>(red_flags) + (flag_base + 6);
                                while (true) {
                                    unsigned int _gca_v;
                                    asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(_gca_v) : "l"(_gca_p));
                                    if (static_cast<int32_t>(_gca_v - static_cast<uint32_t>(gen[0])) >= 0) break;
                                }
                            }
                        }
                        if (4 != ep_j) {
                            {
                                unsigned int* _gca_p = reinterpret_cast<unsigned int*>(red_flags) + (flag_base + 8);
                                while (true) {
                                    unsigned int _gca_v;
                                    asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(_gca_v) : "l"(_gca_p));
                                    if (static_cast<int32_t>(_gca_v - static_cast<uint32_t>(gen[0])) >= 0) break;
                                }
                            }
                        }
                        if (5 != ep_j) {
                            {
                                unsigned int* _gca_p = reinterpret_cast<unsigned int*>(red_flags) + (flag_base + 10);
                                while (true) {
                                    unsigned int _gca_v;
                                    asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(_gca_v) : "l"(_gca_p));
                                    if (static_cast<int32_t>(_gca_v - static_cast<uint32_t>(gen[0])) >= 0) break;
                                }
                            }
                        }
                    }
                }
                asm volatile("barrier.sync 1, 128;" ::: "memory");
                float red[32];
                float ring0[32];
                float ring1[32];
                float ring2[32];
                #pragma unroll
                for (int subtile_2 = 0; subtile_2 < 6; subtile_2++) {
                    if (subtile_2 % 6 == ep_j) {
                        int red_off = tile_slots + subtile_2 * 4096 + thread_off;
                        if (0 == ep_j) {
                            asm volatile(
                                "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                                : "=f"(red[0]), "=f"(red[1]), "=f"(red[2]), "=f"(red[3]), "=f"(red[4]), "=f"(red[5]), "=f"(red[6]), "=f"(red[7]), "=f"(red[8]), "=f"(red[9]), "=f"(red[10]), "=f"(red[11]), "=f"(red[12]), "=f"(red[13]), "=f"(red[14]), "=f"(red[15]), "=f"(red[16]), "=f"(red[17]), "=f"(red[18]), "=f"(red[19]), "=f"(red[20]), "=f"(red[21]), "=f"(red[22]), "=f"(red[23]), "=f"(red[24]), "=f"(red[25]), "=f"(red[26]), "=f"(red[27]), "=f"(red[28]), "=f"(red[29]), "=f"(red[30]), "=f"(red[31])
                                : "r"(lane_addr_s + subtile_2 * 32));
                            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                        } else {
                            {
                                unsigned _v4_2_0;
                                unsigned _v4_2_1;
                                unsigned _v4_2_2;
                                unsigned _v4_2_3;
                                asm volatile("ld.global.cg.v4.b32 {%0, %1, %2, %3}, [%4];"
                                    : "=r"(_v4_2_0), "=r"(_v4_2_1), "=r"(_v4_2_2), "=r"(_v4_2_3) : "l"((const void*)(red_ws + (red_off))) : "memory");
                                red[0 + 0] = __uint_as_float(_v4_2_0);
                                red[0 + 1] = __uint_as_float(_v4_2_1);
                                red[0 + 2] = __uint_as_float(_v4_2_2);
                                red[0 + 3] = __uint_as_float(_v4_2_3);
                            }
                            {
                                unsigned _v4_3_0;
                                unsigned _v4_3_1;
                                unsigned _v4_3_2;
                                unsigned _v4_3_3;
                                asm volatile("ld.global.cg.v4.b32 {%0, %1, %2, %3}, [%4];"
                                    : "=r"(_v4_3_0), "=r"(_v4_3_1), "=r"(_v4_3_2), "=r"(_v4_3_3) : "l"((const void*)(red_ws + (red_off + 128))) : "memory");
                                red[4 + 0] = __uint_as_float(_v4_3_0);
                                red[4 + 1] = __uint_as_float(_v4_3_1);
                                red[4 + 2] = __uint_as_float(_v4_3_2);
                                red[4 + 3] = __uint_as_float(_v4_3_3);
                            }
                            {
                                unsigned _v4_4_0;
                                unsigned _v4_4_1;
                                unsigned _v4_4_2;
                                unsigned _v4_4_3;
                                asm volatile("ld.global.cg.v4.b32 {%0, %1, %2, %3}, [%4];"
                                    : "=r"(_v4_4_0), "=r"(_v4_4_1), "=r"(_v4_4_2), "=r"(_v4_4_3) : "l"((const void*)(red_ws + (red_off + 256))) : "memory");
                                red[8 + 0] = __uint_as_float(_v4_4_0);
                                red[8 + 1] = __uint_as_float(_v4_4_1);
                                red[8 + 2] = __uint_as_float(_v4_4_2);
                                red[8 + 3] = __uint_as_float(_v4_4_3);
                            }
                            {
                                unsigned _v4_5_0;
                                unsigned _v4_5_1;
                                unsigned _v4_5_2;
                                unsigned _v4_5_3;
                                asm volatile("ld.global.cg.v4.b32 {%0, %1, %2, %3}, [%4];"
                                    : "=r"(_v4_5_0), "=r"(_v4_5_1), "=r"(_v4_5_2), "=r"(_v4_5_3) : "l"((const void*)(red_ws + (red_off + 384))) : "memory");
                                red[12 + 0] = __uint_as_float(_v4_5_0);
                                red[12 + 1] = __uint_as_float(_v4_5_1);
                                red[12 + 2] = __uint_as_float(_v4_5_2);
                                red[12 + 3] = __uint_as_float(_v4_5_3);
                            }
                            {
                                unsigned _v4_6_0;
                                unsigned _v4_6_1;
                                unsigned _v4_6_2;
                                unsigned _v4_6_3;
                                asm volatile("ld.global.cg.v4.b32 {%0, %1, %2, %3}, [%4];"
                                    : "=r"(_v4_6_0), "=r"(_v4_6_1), "=r"(_v4_6_2), "=r"(_v4_6_3) : "l"((const void*)(red_ws + (red_off + 512))) : "memory");
                                red[16 + 0] = __uint_as_float(_v4_6_0);
                                red[16 + 1] = __uint_as_float(_v4_6_1);
                                red[16 + 2] = __uint_as_float(_v4_6_2);
                                red[16 + 3] = __uint_as_float(_v4_6_3);
                            }
                            {
                                unsigned _v4_7_0;
                                unsigned _v4_7_1;
                                unsigned _v4_7_2;
                                unsigned _v4_7_3;
                                asm volatile("ld.global.cg.v4.b32 {%0, %1, %2, %3}, [%4];"
                                    : "=r"(_v4_7_0), "=r"(_v4_7_1), "=r"(_v4_7_2), "=r"(_v4_7_3) : "l"((const void*)(red_ws + (red_off + 640))) : "memory");
                                red[20 + 0] = __uint_as_float(_v4_7_0);
                                red[20 + 1] = __uint_as_float(_v4_7_1);
                                red[20 + 2] = __uint_as_float(_v4_7_2);
                                red[20 + 3] = __uint_as_float(_v4_7_3);
                            }
                            {
                                unsigned _v4_8_0;
                                unsigned _v4_8_1;
                                unsigned _v4_8_2;
                                unsigned _v4_8_3;
                                asm volatile("ld.global.cg.v4.b32 {%0, %1, %2, %3}, [%4];"
                                    : "=r"(_v4_8_0), "=r"(_v4_8_1), "=r"(_v4_8_2), "=r"(_v4_8_3) : "l"((const void*)(red_ws + (red_off + 768))) : "memory");
                                red[24 + 0] = __uint_as_float(_v4_8_0);
                                red[24 + 1] = __uint_as_float(_v4_8_1);
                                red[24 + 2] = __uint_as_float(_v4_8_2);
                                red[24 + 3] = __uint_as_float(_v4_8_3);
                            }
                            {
                                unsigned _v4_9_0;
                                unsigned _v4_9_1;
                                unsigned _v4_9_2;
                                unsigned _v4_9_3;
                                asm volatile("ld.global.cg.v4.b32 {%0, %1, %2, %3}, [%4];"
                                    : "=r"(_v4_9_0), "=r"(_v4_9_1), "=r"(_v4_9_2), "=r"(_v4_9_3) : "l"((const void*)(red_ws + (red_off + 896))) : "memory");
                                red[28 + 0] = __uint_as_float(_v4_9_0);
                                red[28 + 1] = __uint_as_float(_v4_9_1);
                                red[28 + 2] = __uint_as_float(_v4_9_2);
                                red[28 + 3] = __uint_as_float(_v4_9_3);
                            }
                        }
                        if (1 == ep_j) {
                            asm volatile(
                                "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                                : "=f"(ring0[0]), "=f"(ring0[1]), "=f"(ring0[2]), "=f"(ring0[3]), "=f"(ring0[4]), "=f"(ring0[5]), "=f"(ring0[6]), "=f"(ring0[7]), "=f"(ring0[8]), "=f"(ring0[9]), "=f"(ring0[10]), "=f"(ring0[11]), "=f"(ring0[12]), "=f"(ring0[13]), "=f"(ring0[14]), "=f"(ring0[15]), "=f"(ring0[16]), "=f"(ring0[17]), "=f"(ring0[18]), "=f"(ring0[19]), "=f"(ring0[20]), "=f"(ring0[21]), "=f"(ring0[22]), "=f"(ring0[23]), "=f"(ring0[24]), "=f"(ring0[25]), "=f"(ring0[26]), "=f"(ring0[27]), "=f"(ring0[28]), "=f"(ring0[29]), "=f"(ring0[30]), "=f"(ring0[31])
                                : "r"(lane_addr_s + subtile_2 * 32));
                            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                        } else {
                            {
                                unsigned _v4_10_0;
                                unsigned _v4_10_1;
                                unsigned _v4_10_2;
                                unsigned _v4_10_3;
                                asm volatile("ld.global.cg.v4.b32 {%0, %1, %2, %3}, [%4];"
                                    : "=r"(_v4_10_0), "=r"(_v4_10_1), "=r"(_v4_10_2), "=r"(_v4_10_3) : "l"((const void*)(red_ws + (red_off + 49152))) : "memory");
                                ring0[0 + 0] = __uint_as_float(_v4_10_0);
                                ring0[0 + 1] = __uint_as_float(_v4_10_1);
                                ring0[0 + 2] = __uint_as_float(_v4_10_2);
                                ring0[0 + 3] = __uint_as_float(_v4_10_3);
                            }
                            {
                                unsigned _v4_11_0;
                                unsigned _v4_11_1;
                                unsigned _v4_11_2;
                                unsigned _v4_11_3;
                                asm volatile("ld.global.cg.v4.b32 {%0, %1, %2, %3}, [%4];"
                                    : "=r"(_v4_11_0), "=r"(_v4_11_1), "=r"(_v4_11_2), "=r"(_v4_11_3) : "l"((const void*)(red_ws + (red_off + 49152 + 128))) : "memory");
                                ring0[4 + 0] = __uint_as_float(_v4_11_0);
                                ring0[4 + 1] = __uint_as_float(_v4_11_1);
                                ring0[4 + 2] = __uint_as_float(_v4_11_2);
                                ring0[4 + 3] = __uint_as_float(_v4_11_3);
                            }
                            {
                                unsigned _v4_12_0;
                                unsigned _v4_12_1;
                                unsigned _v4_12_2;
                                unsigned _v4_12_3;
                                asm volatile("ld.global.cg.v4.b32 {%0, %1, %2, %3}, [%4];"
                                    : "=r"(_v4_12_0), "=r"(_v4_12_1), "=r"(_v4_12_2), "=r"(_v4_12_3) : "l"((const void*)(red_ws + (red_off + 49152 + 256))) : "memory");
                                ring0[8 + 0] = __uint_as_float(_v4_12_0);
                                ring0[8 + 1] = __uint_as_float(_v4_12_1);
                                ring0[8 + 2] = __uint_as_float(_v4_12_2);
                                ring0[8 + 3] = __uint_as_float(_v4_12_3);
                            }
                            {
                                unsigned _v4_13_0;
                                unsigned _v4_13_1;
                                unsigned _v4_13_2;
                                unsigned _v4_13_3;
                                asm volatile("ld.global.cg.v4.b32 {%0, %1, %2, %3}, [%4];"
                                    : "=r"(_v4_13_0), "=r"(_v4_13_1), "=r"(_v4_13_2), "=r"(_v4_13_3) : "l"((const void*)(red_ws + (red_off + 49152 + 384))) : "memory");
                                ring0[12 + 0] = __uint_as_float(_v4_13_0);
                                ring0[12 + 1] = __uint_as_float(_v4_13_1);
                                ring0[12 + 2] = __uint_as_float(_v4_13_2);
                                ring0[12 + 3] = __uint_as_float(_v4_13_3);
                            }
                            {
                                unsigned _v4_14_0;
                                unsigned _v4_14_1;
                                unsigned _v4_14_2;
                                unsigned _v4_14_3;
                                asm volatile("ld.global.cg.v4.b32 {%0, %1, %2, %3}, [%4];"
                                    : "=r"(_v4_14_0), "=r"(_v4_14_1), "=r"(_v4_14_2), "=r"(_v4_14_3) : "l"((const void*)(red_ws + (red_off + 49152 + 512))) : "memory");
                                ring0[16 + 0] = __uint_as_float(_v4_14_0);
                                ring0[16 + 1] = __uint_as_float(_v4_14_1);
                                ring0[16 + 2] = __uint_as_float(_v4_14_2);
                                ring0[16 + 3] = __uint_as_float(_v4_14_3);
                            }
                            {
                                unsigned _v4_15_0;
                                unsigned _v4_15_1;
                                unsigned _v4_15_2;
                                unsigned _v4_15_3;
                                asm volatile("ld.global.cg.v4.b32 {%0, %1, %2, %3}, [%4];"
                                    : "=r"(_v4_15_0), "=r"(_v4_15_1), "=r"(_v4_15_2), "=r"(_v4_15_3) : "l"((const void*)(red_ws + (red_off + 49152 + 640))) : "memory");
                                ring0[20 + 0] = __uint_as_float(_v4_15_0);
                                ring0[20 + 1] = __uint_as_float(_v4_15_1);
                                ring0[20 + 2] = __uint_as_float(_v4_15_2);
                                ring0[20 + 3] = __uint_as_float(_v4_15_3);
                            }
                            {
                                unsigned _v4_16_0;
                                unsigned _v4_16_1;
                                unsigned _v4_16_2;
                                unsigned _v4_16_3;
                                asm volatile("ld.global.cg.v4.b32 {%0, %1, %2, %3}, [%4];"
                                    : "=r"(_v4_16_0), "=r"(_v4_16_1), "=r"(_v4_16_2), "=r"(_v4_16_3) : "l"((const void*)(red_ws + (red_off + 49152 + 768))) : "memory");
                                ring0[24 + 0] = __uint_as_float(_v4_16_0);
                                ring0[24 + 1] = __uint_as_float(_v4_16_1);
                                ring0[24 + 2] = __uint_as_float(_v4_16_2);
                                ring0[24 + 3] = __uint_as_float(_v4_16_3);
                            }
                            {
                                unsigned _v4_17_0;
                                unsigned _v4_17_1;
                                unsigned _v4_17_2;
                                unsigned _v4_17_3;
                                asm volatile("ld.global.cg.v4.b32 {%0, %1, %2, %3}, [%4];"
                                    : "=r"(_v4_17_0), "=r"(_v4_17_1), "=r"(_v4_17_2), "=r"(_v4_17_3) : "l"((const void*)(red_ws + (red_off + 49152 + 896))) : "memory");
                                ring0[28 + 0] = __uint_as_float(_v4_17_0);
                                ring0[28 + 1] = __uint_as_float(_v4_17_1);
                                ring0[28 + 2] = __uint_as_float(_v4_17_2);
                                ring0[28 + 3] = __uint_as_float(_v4_17_3);
                            }
                        }
                        if (2 == ep_j) {
                            asm volatile(
                                "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                                : "=f"(ring1[0]), "=f"(ring1[1]), "=f"(ring1[2]), "=f"(ring1[3]), "=f"(ring1[4]), "=f"(ring1[5]), "=f"(ring1[6]), "=f"(ring1[7]), "=f"(ring1[8]), "=f"(ring1[9]), "=f"(ring1[10]), "=f"(ring1[11]), "=f"(ring1[12]), "=f"(ring1[13]), "=f"(ring1[14]), "=f"(ring1[15]), "=f"(ring1[16]), "=f"(ring1[17]), "=f"(ring1[18]), "=f"(ring1[19]), "=f"(ring1[20]), "=f"(ring1[21]), "=f"(ring1[22]), "=f"(ring1[23]), "=f"(ring1[24]), "=f"(ring1[25]), "=f"(ring1[26]), "=f"(ring1[27]), "=f"(ring1[28]), "=f"(ring1[29]), "=f"(ring1[30]), "=f"(ring1[31])
                                : "r"(lane_addr_s + subtile_2 * 32));
                            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                        } else {
                            {
                                unsigned _v4_18_0;
                                unsigned _v4_18_1;
                                unsigned _v4_18_2;
                                unsigned _v4_18_3;
                                asm volatile("ld.global.cg.v4.b32 {%0, %1, %2, %3}, [%4];"
                                    : "=r"(_v4_18_0), "=r"(_v4_18_1), "=r"(_v4_18_2), "=r"(_v4_18_3) : "l"((const void*)(red_ws + (red_off + 98304))) : "memory");
                                ring1[0 + 0] = __uint_as_float(_v4_18_0);
                                ring1[0 + 1] = __uint_as_float(_v4_18_1);
                                ring1[0 + 2] = __uint_as_float(_v4_18_2);
                                ring1[0 + 3] = __uint_as_float(_v4_18_3);
                            }
                            {
                                unsigned _v4_19_0;
                                unsigned _v4_19_1;
                                unsigned _v4_19_2;
                                unsigned _v4_19_3;
                                asm volatile("ld.global.cg.v4.b32 {%0, %1, %2, %3}, [%4];"
                                    : "=r"(_v4_19_0), "=r"(_v4_19_1), "=r"(_v4_19_2), "=r"(_v4_19_3) : "l"((const void*)(red_ws + (red_off + 98304 + 128))) : "memory");
                                ring1[4 + 0] = __uint_as_float(_v4_19_0);
                                ring1[4 + 1] = __uint_as_float(_v4_19_1);
                                ring1[4 + 2] = __uint_as_float(_v4_19_2);
                                ring1[4 + 3] = __uint_as_float(_v4_19_3);
                            }
                            {
                                unsigned _v4_20_0;
                                unsigned _v4_20_1;
                                unsigned _v4_20_2;
                                unsigned _v4_20_3;
                                asm volatile("ld.global.cg.v4.b32 {%0, %1, %2, %3}, [%4];"
                                    : "=r"(_v4_20_0), "=r"(_v4_20_1), "=r"(_v4_20_2), "=r"(_v4_20_3) : "l"((const void*)(red_ws + (red_off + 98304 + 256))) : "memory");
                                ring1[8 + 0] = __uint_as_float(_v4_20_0);
                                ring1[8 + 1] = __uint_as_float(_v4_20_1);
                                ring1[8 + 2] = __uint_as_float(_v4_20_2);
                                ring1[8 + 3] = __uint_as_float(_v4_20_3);
                            }
                            {
                                unsigned _v4_21_0;
                                unsigned _v4_21_1;
                                unsigned _v4_21_2;
                                unsigned _v4_21_3;
                                asm volatile("ld.global.cg.v4.b32 {%0, %1, %2, %3}, [%4];"
                                    : "=r"(_v4_21_0), "=r"(_v4_21_1), "=r"(_v4_21_2), "=r"(_v4_21_3) : "l"((const void*)(red_ws + (red_off + 98304 + 384))) : "memory");
                                ring1[12 + 0] = __uint_as_float(_v4_21_0);
                                ring1[12 + 1] = __uint_as_float(_v4_21_1);
                                ring1[12 + 2] = __uint_as_float(_v4_21_2);
                                ring1[12 + 3] = __uint_as_float(_v4_21_3);
                            }
                            {
                                unsigned _v4_22_0;
                                unsigned _v4_22_1;
                                unsigned _v4_22_2;
                                unsigned _v4_22_3;
                                asm volatile("ld.global.cg.v4.b32 {%0, %1, %2, %3}, [%4];"
                                    : "=r"(_v4_22_0), "=r"(_v4_22_1), "=r"(_v4_22_2), "=r"(_v4_22_3) : "l"((const void*)(red_ws + (red_off + 98304 + 512))) : "memory");
                                ring1[16 + 0] = __uint_as_float(_v4_22_0);
                                ring1[16 + 1] = __uint_as_float(_v4_22_1);
                                ring1[16 + 2] = __uint_as_float(_v4_22_2);
                                ring1[16 + 3] = __uint_as_float(_v4_22_3);
                            }
                            {
                                unsigned _v4_23_0;
                                unsigned _v4_23_1;
                                unsigned _v4_23_2;
                                unsigned _v4_23_3;
                                asm volatile("ld.global.cg.v4.b32 {%0, %1, %2, %3}, [%4];"
                                    : "=r"(_v4_23_0), "=r"(_v4_23_1), "=r"(_v4_23_2), "=r"(_v4_23_3) : "l"((const void*)(red_ws + (red_off + 98304 + 640))) : "memory");
                                ring1[20 + 0] = __uint_as_float(_v4_23_0);
                                ring1[20 + 1] = __uint_as_float(_v4_23_1);
                                ring1[20 + 2] = __uint_as_float(_v4_23_2);
                                ring1[20 + 3] = __uint_as_float(_v4_23_3);
                            }
                            {
                                unsigned _v4_24_0;
                                unsigned _v4_24_1;
                                unsigned _v4_24_2;
                                unsigned _v4_24_3;
                                asm volatile("ld.global.cg.v4.b32 {%0, %1, %2, %3}, [%4];"
                                    : "=r"(_v4_24_0), "=r"(_v4_24_1), "=r"(_v4_24_2), "=r"(_v4_24_3) : "l"((const void*)(red_ws + (red_off + 98304 + 768))) : "memory");
                                ring1[24 + 0] = __uint_as_float(_v4_24_0);
                                ring1[24 + 1] = __uint_as_float(_v4_24_1);
                                ring1[24 + 2] = __uint_as_float(_v4_24_2);
                                ring1[24 + 3] = __uint_as_float(_v4_24_3);
                            }
                            {
                                unsigned _v4_25_0;
                                unsigned _v4_25_1;
                                unsigned _v4_25_2;
                                unsigned _v4_25_3;
                                asm volatile("ld.global.cg.v4.b32 {%0, %1, %2, %3}, [%4];"
                                    : "=r"(_v4_25_0), "=r"(_v4_25_1), "=r"(_v4_25_2), "=r"(_v4_25_3) : "l"((const void*)(red_ws + (red_off + 98304 + 896))) : "memory");
                                ring1[28 + 0] = __uint_as_float(_v4_25_0);
                                ring1[28 + 1] = __uint_as_float(_v4_25_1);
                                ring1[28 + 2] = __uint_as_float(_v4_25_2);
                                ring1[28 + 3] = __uint_as_float(_v4_25_3);
                            }
                        }
                        #pragma unroll
                        for (int _la = 0; _la < 32; _la++)
                            red[_la] = red[_la] + ring0[_la];
                        if (3 == ep_j) {
                            asm volatile(
                                "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                                : "=f"(ring2[0]), "=f"(ring2[1]), "=f"(ring2[2]), "=f"(ring2[3]), "=f"(ring2[4]), "=f"(ring2[5]), "=f"(ring2[6]), "=f"(ring2[7]), "=f"(ring2[8]), "=f"(ring2[9]), "=f"(ring2[10]), "=f"(ring2[11]), "=f"(ring2[12]), "=f"(ring2[13]), "=f"(ring2[14]), "=f"(ring2[15]), "=f"(ring2[16]), "=f"(ring2[17]), "=f"(ring2[18]), "=f"(ring2[19]), "=f"(ring2[20]), "=f"(ring2[21]), "=f"(ring2[22]), "=f"(ring2[23]), "=f"(ring2[24]), "=f"(ring2[25]), "=f"(ring2[26]), "=f"(ring2[27]), "=f"(ring2[28]), "=f"(ring2[29]), "=f"(ring2[30]), "=f"(ring2[31])
                                : "r"(lane_addr_s + subtile_2 * 32));
                            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                        } else {
                            {
                                unsigned _v4_26_0;
                                unsigned _v4_26_1;
                                unsigned _v4_26_2;
                                unsigned _v4_26_3;
                                asm volatile("ld.global.cg.v4.b32 {%0, %1, %2, %3}, [%4];"
                                    : "=r"(_v4_26_0), "=r"(_v4_26_1), "=r"(_v4_26_2), "=r"(_v4_26_3) : "l"((const void*)(red_ws + (red_off + 147456))) : "memory");
                                ring2[0 + 0] = __uint_as_float(_v4_26_0);
                                ring2[0 + 1] = __uint_as_float(_v4_26_1);
                                ring2[0 + 2] = __uint_as_float(_v4_26_2);
                                ring2[0 + 3] = __uint_as_float(_v4_26_3);
                            }
                            {
                                unsigned _v4_27_0;
                                unsigned _v4_27_1;
                                unsigned _v4_27_2;
                                unsigned _v4_27_3;
                                asm volatile("ld.global.cg.v4.b32 {%0, %1, %2, %3}, [%4];"
                                    : "=r"(_v4_27_0), "=r"(_v4_27_1), "=r"(_v4_27_2), "=r"(_v4_27_3) : "l"((const void*)(red_ws + (red_off + 147456 + 128))) : "memory");
                                ring2[4 + 0] = __uint_as_float(_v4_27_0);
                                ring2[4 + 1] = __uint_as_float(_v4_27_1);
                                ring2[4 + 2] = __uint_as_float(_v4_27_2);
                                ring2[4 + 3] = __uint_as_float(_v4_27_3);
                            }
                            {
                                unsigned _v4_28_0;
                                unsigned _v4_28_1;
                                unsigned _v4_28_2;
                                unsigned _v4_28_3;
                                asm volatile("ld.global.cg.v4.b32 {%0, %1, %2, %3}, [%4];"
                                    : "=r"(_v4_28_0), "=r"(_v4_28_1), "=r"(_v4_28_2), "=r"(_v4_28_3) : "l"((const void*)(red_ws + (red_off + 147456 + 256))) : "memory");
                                ring2[8 + 0] = __uint_as_float(_v4_28_0);
                                ring2[8 + 1] = __uint_as_float(_v4_28_1);
                                ring2[8 + 2] = __uint_as_float(_v4_28_2);
                                ring2[8 + 3] = __uint_as_float(_v4_28_3);
                            }
                            {
                                unsigned _v4_29_0;
                                unsigned _v4_29_1;
                                unsigned _v4_29_2;
                                unsigned _v4_29_3;
                                asm volatile("ld.global.cg.v4.b32 {%0, %1, %2, %3}, [%4];"
                                    : "=r"(_v4_29_0), "=r"(_v4_29_1), "=r"(_v4_29_2), "=r"(_v4_29_3) : "l"((const void*)(red_ws + (red_off + 147456 + 384))) : "memory");
                                ring2[12 + 0] = __uint_as_float(_v4_29_0);
                                ring2[12 + 1] = __uint_as_float(_v4_29_1);
                                ring2[12 + 2] = __uint_as_float(_v4_29_2);
                                ring2[12 + 3] = __uint_as_float(_v4_29_3);
                            }
                            {
                                unsigned _v4_30_0;
                                unsigned _v4_30_1;
                                unsigned _v4_30_2;
                                unsigned _v4_30_3;
                                asm volatile("ld.global.cg.v4.b32 {%0, %1, %2, %3}, [%4];"
                                    : "=r"(_v4_30_0), "=r"(_v4_30_1), "=r"(_v4_30_2), "=r"(_v4_30_3) : "l"((const void*)(red_ws + (red_off + 147456 + 512))) : "memory");
                                ring2[16 + 0] = __uint_as_float(_v4_30_0);
                                ring2[16 + 1] = __uint_as_float(_v4_30_1);
                                ring2[16 + 2] = __uint_as_float(_v4_30_2);
                                ring2[16 + 3] = __uint_as_float(_v4_30_3);
                            }
                            {
                                unsigned _v4_31_0;
                                unsigned _v4_31_1;
                                unsigned _v4_31_2;
                                unsigned _v4_31_3;
                                asm volatile("ld.global.cg.v4.b32 {%0, %1, %2, %3}, [%4];"
                                    : "=r"(_v4_31_0), "=r"(_v4_31_1), "=r"(_v4_31_2), "=r"(_v4_31_3) : "l"((const void*)(red_ws + (red_off + 147456 + 640))) : "memory");
                                ring2[20 + 0] = __uint_as_float(_v4_31_0);
                                ring2[20 + 1] = __uint_as_float(_v4_31_1);
                                ring2[20 + 2] = __uint_as_float(_v4_31_2);
                                ring2[20 + 3] = __uint_as_float(_v4_31_3);
                            }
                            {
                                unsigned _v4_32_0;
                                unsigned _v4_32_1;
                                unsigned _v4_32_2;
                                unsigned _v4_32_3;
                                asm volatile("ld.global.cg.v4.b32 {%0, %1, %2, %3}, [%4];"
                                    : "=r"(_v4_32_0), "=r"(_v4_32_1), "=r"(_v4_32_2), "=r"(_v4_32_3) : "l"((const void*)(red_ws + (red_off + 147456 + 768))) : "memory");
                                ring2[24 + 0] = __uint_as_float(_v4_32_0);
                                ring2[24 + 1] = __uint_as_float(_v4_32_1);
                                ring2[24 + 2] = __uint_as_float(_v4_32_2);
                                ring2[24 + 3] = __uint_as_float(_v4_32_3);
                            }
                            {
                                unsigned _v4_33_0;
                                unsigned _v4_33_1;
                                unsigned _v4_33_2;
                                unsigned _v4_33_3;
                                asm volatile("ld.global.cg.v4.b32 {%0, %1, %2, %3}, [%4];"
                                    : "=r"(_v4_33_0), "=r"(_v4_33_1), "=r"(_v4_33_2), "=r"(_v4_33_3) : "l"((const void*)(red_ws + (red_off + 147456 + 896))) : "memory");
                                ring2[28 + 0] = __uint_as_float(_v4_33_0);
                                ring2[28 + 1] = __uint_as_float(_v4_33_1);
                                ring2[28 + 2] = __uint_as_float(_v4_33_2);
                                ring2[28 + 3] = __uint_as_float(_v4_33_3);
                            }
                        }
                        #pragma unroll
                        for (int _la = 0; _la < 32; _la++)
                            red[_la] = red[_la] + ring1[_la];
                        if (4 == ep_j) {
                            asm volatile(
                                "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                                : "=f"(ring0[0]), "=f"(ring0[1]), "=f"(ring0[2]), "=f"(ring0[3]), "=f"(ring0[4]), "=f"(ring0[5]), "=f"(ring0[6]), "=f"(ring0[7]), "=f"(ring0[8]), "=f"(ring0[9]), "=f"(ring0[10]), "=f"(ring0[11]), "=f"(ring0[12]), "=f"(ring0[13]), "=f"(ring0[14]), "=f"(ring0[15]), "=f"(ring0[16]), "=f"(ring0[17]), "=f"(ring0[18]), "=f"(ring0[19]), "=f"(ring0[20]), "=f"(ring0[21]), "=f"(ring0[22]), "=f"(ring0[23]), "=f"(ring0[24]), "=f"(ring0[25]), "=f"(ring0[26]), "=f"(ring0[27]), "=f"(ring0[28]), "=f"(ring0[29]), "=f"(ring0[30]), "=f"(ring0[31])
                                : "r"(lane_addr_s + subtile_2 * 32));
                            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                        } else {
                            {
                                unsigned _v4_34_0;
                                unsigned _v4_34_1;
                                unsigned _v4_34_2;
                                unsigned _v4_34_3;
                                asm volatile("ld.global.cg.v4.b32 {%0, %1, %2, %3}, [%4];"
                                    : "=r"(_v4_34_0), "=r"(_v4_34_1), "=r"(_v4_34_2), "=r"(_v4_34_3) : "l"((const void*)(red_ws + (red_off + 196608))) : "memory");
                                ring0[0 + 0] = __uint_as_float(_v4_34_0);
                                ring0[0 + 1] = __uint_as_float(_v4_34_1);
                                ring0[0 + 2] = __uint_as_float(_v4_34_2);
                                ring0[0 + 3] = __uint_as_float(_v4_34_3);
                            }
                            {
                                unsigned _v4_35_0;
                                unsigned _v4_35_1;
                                unsigned _v4_35_2;
                                unsigned _v4_35_3;
                                asm volatile("ld.global.cg.v4.b32 {%0, %1, %2, %3}, [%4];"
                                    : "=r"(_v4_35_0), "=r"(_v4_35_1), "=r"(_v4_35_2), "=r"(_v4_35_3) : "l"((const void*)(red_ws + (red_off + 196608 + 128))) : "memory");
                                ring0[4 + 0] = __uint_as_float(_v4_35_0);
                                ring0[4 + 1] = __uint_as_float(_v4_35_1);
                                ring0[4 + 2] = __uint_as_float(_v4_35_2);
                                ring0[4 + 3] = __uint_as_float(_v4_35_3);
                            }
                            {
                                unsigned _v4_36_0;
                                unsigned _v4_36_1;
                                unsigned _v4_36_2;
                                unsigned _v4_36_3;
                                asm volatile("ld.global.cg.v4.b32 {%0, %1, %2, %3}, [%4];"
                                    : "=r"(_v4_36_0), "=r"(_v4_36_1), "=r"(_v4_36_2), "=r"(_v4_36_3) : "l"((const void*)(red_ws + (red_off + 196608 + 256))) : "memory");
                                ring0[8 + 0] = __uint_as_float(_v4_36_0);
                                ring0[8 + 1] = __uint_as_float(_v4_36_1);
                                ring0[8 + 2] = __uint_as_float(_v4_36_2);
                                ring0[8 + 3] = __uint_as_float(_v4_36_3);
                            }
                            {
                                unsigned _v4_37_0;
                                unsigned _v4_37_1;
                                unsigned _v4_37_2;
                                unsigned _v4_37_3;
                                asm volatile("ld.global.cg.v4.b32 {%0, %1, %2, %3}, [%4];"
                                    : "=r"(_v4_37_0), "=r"(_v4_37_1), "=r"(_v4_37_2), "=r"(_v4_37_3) : "l"((const void*)(red_ws + (red_off + 196608 + 384))) : "memory");
                                ring0[12 + 0] = __uint_as_float(_v4_37_0);
                                ring0[12 + 1] = __uint_as_float(_v4_37_1);
                                ring0[12 + 2] = __uint_as_float(_v4_37_2);
                                ring0[12 + 3] = __uint_as_float(_v4_37_3);
                            }
                            {
                                unsigned _v4_38_0;
                                unsigned _v4_38_1;
                                unsigned _v4_38_2;
                                unsigned _v4_38_3;
                                asm volatile("ld.global.cg.v4.b32 {%0, %1, %2, %3}, [%4];"
                                    : "=r"(_v4_38_0), "=r"(_v4_38_1), "=r"(_v4_38_2), "=r"(_v4_38_3) : "l"((const void*)(red_ws + (red_off + 196608 + 512))) : "memory");
                                ring0[16 + 0] = __uint_as_float(_v4_38_0);
                                ring0[16 + 1] = __uint_as_float(_v4_38_1);
                                ring0[16 + 2] = __uint_as_float(_v4_38_2);
                                ring0[16 + 3] = __uint_as_float(_v4_38_3);
                            }
                            {
                                unsigned _v4_39_0;
                                unsigned _v4_39_1;
                                unsigned _v4_39_2;
                                unsigned _v4_39_3;
                                asm volatile("ld.global.cg.v4.b32 {%0, %1, %2, %3}, [%4];"
                                    : "=r"(_v4_39_0), "=r"(_v4_39_1), "=r"(_v4_39_2), "=r"(_v4_39_3) : "l"((const void*)(red_ws + (red_off + 196608 + 640))) : "memory");
                                ring0[20 + 0] = __uint_as_float(_v4_39_0);
                                ring0[20 + 1] = __uint_as_float(_v4_39_1);
                                ring0[20 + 2] = __uint_as_float(_v4_39_2);
                                ring0[20 + 3] = __uint_as_float(_v4_39_3);
                            }
                            {
                                unsigned _v4_40_0;
                                unsigned _v4_40_1;
                                unsigned _v4_40_2;
                                unsigned _v4_40_3;
                                asm volatile("ld.global.cg.v4.b32 {%0, %1, %2, %3}, [%4];"
                                    : "=r"(_v4_40_0), "=r"(_v4_40_1), "=r"(_v4_40_2), "=r"(_v4_40_3) : "l"((const void*)(red_ws + (red_off + 196608 + 768))) : "memory");
                                ring0[24 + 0] = __uint_as_float(_v4_40_0);
                                ring0[24 + 1] = __uint_as_float(_v4_40_1);
                                ring0[24 + 2] = __uint_as_float(_v4_40_2);
                                ring0[24 + 3] = __uint_as_float(_v4_40_3);
                            }
                            {
                                unsigned _v4_41_0;
                                unsigned _v4_41_1;
                                unsigned _v4_41_2;
                                unsigned _v4_41_3;
                                asm volatile("ld.global.cg.v4.b32 {%0, %1, %2, %3}, [%4];"
                                    : "=r"(_v4_41_0), "=r"(_v4_41_1), "=r"(_v4_41_2), "=r"(_v4_41_3) : "l"((const void*)(red_ws + (red_off + 196608 + 896))) : "memory");
                                ring0[28 + 0] = __uint_as_float(_v4_41_0);
                                ring0[28 + 1] = __uint_as_float(_v4_41_1);
                                ring0[28 + 2] = __uint_as_float(_v4_41_2);
                                ring0[28 + 3] = __uint_as_float(_v4_41_3);
                            }
                        }
                        #pragma unroll
                        for (int _la = 0; _la < 32; _la++)
                            red[_la] = red[_la] + ring2[_la];
                        if (5 == ep_j) {
                            asm volatile(
                                "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                                : "=f"(ring1[0]), "=f"(ring1[1]), "=f"(ring1[2]), "=f"(ring1[3]), "=f"(ring1[4]), "=f"(ring1[5]), "=f"(ring1[6]), "=f"(ring1[7]), "=f"(ring1[8]), "=f"(ring1[9]), "=f"(ring1[10]), "=f"(ring1[11]), "=f"(ring1[12]), "=f"(ring1[13]), "=f"(ring1[14]), "=f"(ring1[15]), "=f"(ring1[16]), "=f"(ring1[17]), "=f"(ring1[18]), "=f"(ring1[19]), "=f"(ring1[20]), "=f"(ring1[21]), "=f"(ring1[22]), "=f"(ring1[23]), "=f"(ring1[24]), "=f"(ring1[25]), "=f"(ring1[26]), "=f"(ring1[27]), "=f"(ring1[28]), "=f"(ring1[29]), "=f"(ring1[30]), "=f"(ring1[31])
                                : "r"(lane_addr_s + subtile_2 * 32));
                            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                        } else {
                            {
                                unsigned _v4_42_0;
                                unsigned _v4_42_1;
                                unsigned _v4_42_2;
                                unsigned _v4_42_3;
                                asm volatile("ld.global.cg.v4.b32 {%0, %1, %2, %3}, [%4];"
                                    : "=r"(_v4_42_0), "=r"(_v4_42_1), "=r"(_v4_42_2), "=r"(_v4_42_3) : "l"((const void*)(red_ws + (red_off + 245760))) : "memory");
                                ring1[0 + 0] = __uint_as_float(_v4_42_0);
                                ring1[0 + 1] = __uint_as_float(_v4_42_1);
                                ring1[0 + 2] = __uint_as_float(_v4_42_2);
                                ring1[0 + 3] = __uint_as_float(_v4_42_3);
                            }
                            {
                                unsigned _v4_43_0;
                                unsigned _v4_43_1;
                                unsigned _v4_43_2;
                                unsigned _v4_43_3;
                                asm volatile("ld.global.cg.v4.b32 {%0, %1, %2, %3}, [%4];"
                                    : "=r"(_v4_43_0), "=r"(_v4_43_1), "=r"(_v4_43_2), "=r"(_v4_43_3) : "l"((const void*)(red_ws + (red_off + 245760 + 128))) : "memory");
                                ring1[4 + 0] = __uint_as_float(_v4_43_0);
                                ring1[4 + 1] = __uint_as_float(_v4_43_1);
                                ring1[4 + 2] = __uint_as_float(_v4_43_2);
                                ring1[4 + 3] = __uint_as_float(_v4_43_3);
                            }
                            {
                                unsigned _v4_44_0;
                                unsigned _v4_44_1;
                                unsigned _v4_44_2;
                                unsigned _v4_44_3;
                                asm volatile("ld.global.cg.v4.b32 {%0, %1, %2, %3}, [%4];"
                                    : "=r"(_v4_44_0), "=r"(_v4_44_1), "=r"(_v4_44_2), "=r"(_v4_44_3) : "l"((const void*)(red_ws + (red_off + 245760 + 256))) : "memory");
                                ring1[8 + 0] = __uint_as_float(_v4_44_0);
                                ring1[8 + 1] = __uint_as_float(_v4_44_1);
                                ring1[8 + 2] = __uint_as_float(_v4_44_2);
                                ring1[8 + 3] = __uint_as_float(_v4_44_3);
                            }
                            {
                                unsigned _v4_45_0;
                                unsigned _v4_45_1;
                                unsigned _v4_45_2;
                                unsigned _v4_45_3;
                                asm volatile("ld.global.cg.v4.b32 {%0, %1, %2, %3}, [%4];"
                                    : "=r"(_v4_45_0), "=r"(_v4_45_1), "=r"(_v4_45_2), "=r"(_v4_45_3) : "l"((const void*)(red_ws + (red_off + 245760 + 384))) : "memory");
                                ring1[12 + 0] = __uint_as_float(_v4_45_0);
                                ring1[12 + 1] = __uint_as_float(_v4_45_1);
                                ring1[12 + 2] = __uint_as_float(_v4_45_2);
                                ring1[12 + 3] = __uint_as_float(_v4_45_3);
                            }
                            {
                                unsigned _v4_46_0;
                                unsigned _v4_46_1;
                                unsigned _v4_46_2;
                                unsigned _v4_46_3;
                                asm volatile("ld.global.cg.v4.b32 {%0, %1, %2, %3}, [%4];"
                                    : "=r"(_v4_46_0), "=r"(_v4_46_1), "=r"(_v4_46_2), "=r"(_v4_46_3) : "l"((const void*)(red_ws + (red_off + 245760 + 512))) : "memory");
                                ring1[16 + 0] = __uint_as_float(_v4_46_0);
                                ring1[16 + 1] = __uint_as_float(_v4_46_1);
                                ring1[16 + 2] = __uint_as_float(_v4_46_2);
                                ring1[16 + 3] = __uint_as_float(_v4_46_3);
                            }
                            {
                                unsigned _v4_47_0;
                                unsigned _v4_47_1;
                                unsigned _v4_47_2;
                                unsigned _v4_47_3;
                                asm volatile("ld.global.cg.v4.b32 {%0, %1, %2, %3}, [%4];"
                                    : "=r"(_v4_47_0), "=r"(_v4_47_1), "=r"(_v4_47_2), "=r"(_v4_47_3) : "l"((const void*)(red_ws + (red_off + 245760 + 640))) : "memory");
                                ring1[20 + 0] = __uint_as_float(_v4_47_0);
                                ring1[20 + 1] = __uint_as_float(_v4_47_1);
                                ring1[20 + 2] = __uint_as_float(_v4_47_2);
                                ring1[20 + 3] = __uint_as_float(_v4_47_3);
                            }
                            {
                                unsigned _v4_48_0;
                                unsigned _v4_48_1;
                                unsigned _v4_48_2;
                                unsigned _v4_48_3;
                                asm volatile("ld.global.cg.v4.b32 {%0, %1, %2, %3}, [%4];"
                                    : "=r"(_v4_48_0), "=r"(_v4_48_1), "=r"(_v4_48_2), "=r"(_v4_48_3) : "l"((const void*)(red_ws + (red_off + 245760 + 768))) : "memory");
                                ring1[24 + 0] = __uint_as_float(_v4_48_0);
                                ring1[24 + 1] = __uint_as_float(_v4_48_1);
                                ring1[24 + 2] = __uint_as_float(_v4_48_2);
                                ring1[24 + 3] = __uint_as_float(_v4_48_3);
                            }
                            {
                                unsigned _v4_49_0;
                                unsigned _v4_49_1;
                                unsigned _v4_49_2;
                                unsigned _v4_49_3;
                                asm volatile("ld.global.cg.v4.b32 {%0, %1, %2, %3}, [%4];"
                                    : "=r"(_v4_49_0), "=r"(_v4_49_1), "=r"(_v4_49_2), "=r"(_v4_49_3) : "l"((const void*)(red_ws + (red_off + 245760 + 896))) : "memory");
                                ring1[28 + 0] = __uint_as_float(_v4_49_0);
                                ring1[28 + 1] = __uint_as_float(_v4_49_1);
                                ring1[28 + 2] = __uint_as_float(_v4_49_2);
                                ring1[28 + 3] = __uint_as_float(_v4_49_3);
                            }
                        }
                        #pragma unroll
                        for (int _la = 0; _la < 32; _la++)
                            red[_la] = red[_la] + ring0[_la];
                        #pragma unroll
                        for (int _la = 0; _la < 32; _la++)
                            red[_la] = red[_la] + ring1[_la];
                        {
                            float2 _pair_scale_even2_50 = make_float2(alpha_row_s, alpha_row_s);
                            float2 _pair_scale_odd2_50 = make_float2(alpha_row_s, alpha_row_s);
                            float2* _pair_scale_src2_50 = reinterpret_cast<float2*>(&red[0]);
                            #if __CUDA_ARCH__ >= 1000
                            asm volatile("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_50[0]) : "l"(*(unsigned long long*)&_pair_scale_even2_50));
                            asm volatile("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_50[1]) : "l"(*(unsigned long long*)&_pair_scale_odd2_50));
                            asm volatile("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_50[2]) : "l"(*(unsigned long long*)&_pair_scale_even2_50));
                            asm volatile("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_50[3]) : "l"(*(unsigned long long*)&_pair_scale_odd2_50));
                            asm volatile("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_50[4]) : "l"(*(unsigned long long*)&_pair_scale_even2_50));
                            asm volatile("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_50[5]) : "l"(*(unsigned long long*)&_pair_scale_odd2_50));
                            asm volatile("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_50[6]) : "l"(*(unsigned long long*)&_pair_scale_even2_50));
                            asm volatile("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_50[7]) : "l"(*(unsigned long long*)&_pair_scale_odd2_50));
                            asm volatile("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_50[8]) : "l"(*(unsigned long long*)&_pair_scale_even2_50));
                            asm volatile("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_50[9]) : "l"(*(unsigned long long*)&_pair_scale_odd2_50));
                            asm volatile("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_50[10]) : "l"(*(unsigned long long*)&_pair_scale_even2_50));
                            asm volatile("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_50[11]) : "l"(*(unsigned long long*)&_pair_scale_odd2_50));
                            asm volatile("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_50[12]) : "l"(*(unsigned long long*)&_pair_scale_even2_50));
                            asm volatile("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_50[13]) : "l"(*(unsigned long long*)&_pair_scale_odd2_50));
                            asm volatile("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_50[14]) : "l"(*(unsigned long long*)&_pair_scale_even2_50));
                            asm volatile("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_50[15]) : "l"(*(unsigned long long*)&_pair_scale_odd2_50));
                            #else
                            red[0] *= alpha_row_s;
                            red[1] *= alpha_row_s;
                            red[2] *= alpha_row_s;
                            red[3] *= alpha_row_s;
                            red[4] *= alpha_row_s;
                            red[5] *= alpha_row_s;
                            red[6] *= alpha_row_s;
                            red[7] *= alpha_row_s;
                            red[8] *= alpha_row_s;
                            red[9] *= alpha_row_s;
                            red[10] *= alpha_row_s;
                            red[11] *= alpha_row_s;
                            red[12] *= alpha_row_s;
                            red[13] *= alpha_row_s;
                            red[14] *= alpha_row_s;
                            red[15] *= alpha_row_s;
                            red[16] *= alpha_row_s;
                            red[17] *= alpha_row_s;
                            red[18] *= alpha_row_s;
                            red[19] *= alpha_row_s;
                            red[20] *= alpha_row_s;
                            red[21] *= alpha_row_s;
                            red[22] *= alpha_row_s;
                            red[23] *= alpha_row_s;
                            red[24] *= alpha_row_s;
                            red[25] *= alpha_row_s;
                            red[26] *= alpha_row_s;
                            red[27] *= alpha_row_s;
                            red[28] *= alpha_row_s;
                            red[29] *= alpha_row_s;
                            red[30] *= alpha_row_s;
                            red[31] *= alpha_row_s;
                            #endif
                        }
                        uint32_t red_bf16[16];
                        #pragma unroll
                        for (int _lp = 0; _lp < 16; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(red[_lp*2 + 0], red[_lp*2+1 + 0]));
                            red_bf16[_lp] = *(uint32_t*)&_bf2;
                        }
                        if (warp == 4) {
                            asm volatile("cp.async.bulk.wait_group.read 1;");
                        }
                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                        int out_stage_row_s = store_stage * 128 + (unsigned int)epi_srow;
                        unsigned int out_abs_s = smem_out_addr + (unsigned int)(out_stage_row_s * 64);
                        unsigned int out_swz_s = out_abs_s / 8 & 48;
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_out_addr + ((unsigned int)(out_stage_row_s * 64) + (0 ^ out_swz_s)))), "r"(red_bf16[0]), "r"(red_bf16[1]), "r"(red_bf16[2]), "r"(red_bf16[3]) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_out_addr + ((unsigned int)(out_stage_row_s * 64) + (16 ^ out_swz_s)))), "r"(red_bf16[4]), "r"(red_bf16[5]), "r"(red_bf16[6]), "r"(red_bf16[7]) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_out_addr + ((unsigned int)(out_stage_row_s * 64) + (32 ^ out_swz_s)))), "r"(red_bf16[8]), "r"(red_bf16[9]), "r"(red_bf16[10]), "r"(red_bf16[11]) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_out_addr + ((unsigned int)(out_stage_row_s * 64) + (48 ^ out_swz_s)))), "r"(red_bf16[12]), "r"(red_bf16[13]), "r"(red_bf16[14]), "r"(red_bf16[15]) : "memory");
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                        if (warp == 4) {
                            if (elect_sync()) {
                                tma_store_2d((&out), off_w_s + subtile_2 * 32, off_tok_s, smem_out_addr + store_stage * 8192);
                            }
                        }
                        if (warp == 4) {
                            asm volatile("cp.async.bulk.commit_group;");
                        }
                        store_stage = store_stage + 1;
                        if (store_stage == 2) {
                            store_stage = 0;
                        }
                    }
                }
                asm volatile("tcgen05.fence::before_thread_sync;");
                asm volatile(
                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                    :: "r"((acc_empty_addr + (acc_stage_1) * 8) & 0xFEFFFFFF) : "memory");
                _phase_acc_full ^= 1;
            }
            if (warp == 4) {
                asm volatile("cp.async.bulk.wait_group 0;");
            }
        }
    }

    // Cleanup
}

} // extern "C"
