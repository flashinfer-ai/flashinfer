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
#define TMEM_NCOLS 304
#define TMEM_ACCUM_OFFSET 0
#define TMEM_TMEM_SFA_OFFSET 256
#define TMEM_TMEM_SFB_OFFSET 272
#define NUM_TMA_PIPE_STAGES 5
#define NUM_MAINLOOP_PIPE_STAGES 1
#define NUM_WORK_PIPE_STAGES 4
#define SMEM_SMEM_A_OFF 1024
#define SMEM_SMEM_A_STAGE_BYTES 16384
#define SMEM_SMEM_A_STRIDE 38912
#define SMEM_SMEM_B_OFF 17408
#define SMEM_SMEM_B_STAGE_BYTES 16384
#define SMEM_SMEM_B_STRIDE 38912
#define SMEM_SMEM_SFA_ALL_OFF 33792
#define SMEM_SMEM_SFA_ALL_STAGE_BYTES 2048
#define SMEM_SMEM_SFA_ALL_STRIDE 38912
#define SMEM_SMEM_SFB_ALL_OFF 35840
#define SMEM_SMEM_SFB_ALL_STAGE_BYTES 4096
#define SMEM_SMEM_SFB_ALL_STRIDE 38912
#define SMEM_SMEM_V4_OFF 33792
#define SMEM_SMEM_V4_STAGE_BYTES 512
#define SMEM_SMEM_V4_STRIDE 38912
#define SMEM_SMEM_V5_OFF 34304
#define SMEM_SMEM_V5_STAGE_BYTES 512
#define SMEM_SMEM_V5_STRIDE 38912
#define SMEM_SMEM_V6_OFF 34816
#define SMEM_SMEM_V6_STAGE_BYTES 512
#define SMEM_SMEM_V6_STRIDE 38912
#define SMEM_SMEM_V7_OFF 35328
#define SMEM_SMEM_V7_STAGE_BYTES 512
#define SMEM_SMEM_V7_STRIDE 38912
#define SMEM_SMEM_V8_OFF 35840
#define SMEM_SMEM_V8_STAGE_BYTES 1024
#define SMEM_SMEM_V8_STRIDE 38912
#define SMEM_SMEM_V9_OFF 36864
#define SMEM_SMEM_V9_STAGE_BYTES 1024
#define SMEM_SMEM_V9_STRIDE 38912
#define SMEM_SMEM_V10_OFF 37888
#define SMEM_SMEM_V10_STAGE_BYTES 1024
#define SMEM_SMEM_V10_STRIDE 38912
#define SMEM_SMEM_V11_OFF 38912
#define SMEM_SMEM_V11_STAGE_BYTES 1024
#define SMEM_SMEM_V11_STRIDE 38912
#define SMEM_EPI_STAGING_OFF 195584
#define SMEM_EPI_STAGING_STAGE_BYTES 16384
#define SMEM_EPI_STAGING_STRIDE 16384
#define SMEM_WORK_RESPONSE_OFF 228352
#define SMEM_WORK_RESPONSE_STAGE_BYTES 16
#define SMEM_WORK_RESPONSE_STRIDE 16
#define SMEM_TOTAL 228480
#define BLOCK_M 128
#define BLOCK_N 256
#define B_HALF_N 128
#define BLOCK_K 256
#define CTA_GROUP 2
#define NUM_STAGES 5
#define GROUP_M 64
#define WORK_STAGES 4
#define WORK_CONSUMERS 546
#define NUM_K_ITERS 21
#define N_TILES 84
#define SF_K_TILES 84
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



__device__ __forceinline__ void tcgen05_mma_mxf4nvf4_bs_cta2_leader(
    int taddr, uint64_t a_desc, uint64_t b_desc, uint32_t i_desc,
    int sfa_taddr, int sfb_taddr, int enable_input_d, uint32_t leader) {
    asm volatile(
        "{\n\t"
        ".reg .pred leader, p;\n\t"
        "setp.ne.b32 leader, %7, 0;\n\t"
        "setp.ne.b32 p, %6, 0;\n\t"
        "@leader tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::4X [%0], %1, %2, %3, [%4], [%5], p;\n\t"
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






// ex2_emulation_f32x2 defined in softmax_frag_exp2_cast helper (or standalone)

__device__ __forceinline__ float2 fma_f32x2_rn_ftz(float2 a, float2 b, float2 c) {
    float2 r;
    asm("fma.rn.ftz.f32x2 %0, %1, %2, %3;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b),
          "l"(*(const unsigned long long*)&c));
    return r;
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



__device__ __forceinline__ void tma_store_4d(
    const void *tmap, int x, int y, int z, int w, unsigned smem_addr) {
    asm volatile(
        "cp.async.bulk.tensor.4d.global.shared::cta.tile.bulk_group"
        " [%0, {%1, %2, %3, %4}], [%5];"
        :: "l"(tmap), "r"(x), "r"(y), "r"(z), "r"(w), "r"(smem_addr) : "memory");
}




__device__ __forceinline__ unsigned int __as_u32(float v) {
    unsigned int u;
    asm("mov.b32 %0, %1;" : "=r"(u) : "f"(v));
    return u;
}
__device__ __forceinline__ unsigned int __as_u32(__nv_bfloat162 v) {
    return *reinterpret_cast<const unsigned int*>(&v);
}
__device__ __forceinline__ unsigned int __as_u32(unsigned int v) { return v; }
__device__ __forceinline__ unsigned int __as_u32(int v) {
    unsigned int u;
    asm("mov.b32 %0, %1;" : "=r"(u) : "r"(v));
    return u;
}

extern "C" {

__global__ __launch_bounds__(320) __cluster_dims__(2,1,1) void
kernel_cake_minimax_h3_nvfp4_pre_attention_945b700f51782c295c7f(const __grid_constant__ CUtensorMap A, const __grid_constant__ CUtensorMap B, const __grid_constant__ CUtensorMap SFA, const __grid_constant__ CUtensorMap SFB, float* __restrict__ alpha, __nv_bfloat16* __restrict__ q_norm_weight, __nv_bfloat16* __restrict__ k_norm_weight, __nv_bfloat16* __restrict__ rope_cos_sin, float* __restrict__ out_global_scale, uint8_t* __restrict__ out_q, uint8_t* __restrict__ out_sf, const __grid_constant__ CUtensorMap OUTQ, unsigned int* __restrict__ qkv_words, unsigned int* __restrict__ debug_q_words, unsigned int* __restrict__ debug_k_words, int write_debug, float eps, int M, int m_tiles, int HEADS_PER_DESTINATION, int ROWS_PER_DESTINATION, int SCALE_STRIDE)
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
    uint8_t* smem_a = reinterpret_cast<uint8_t*>(smem_raw + 1024);
    const int smem_a_addr = smem + 1024;
    uint8_t* smem_b = reinterpret_cast<uint8_t*>(smem_raw + 17408);
    const int smem_b_addr = smem + 17408;
    uint8_t* smem_sfa_all = reinterpret_cast<uint8_t*>(smem_raw + 33792);
    const int smem_sfa_all_addr = smem + 33792;
    uint8_t* smem_sfb_all = reinterpret_cast<uint8_t*>(smem_raw + 35840);
    const int smem_sfb_all_addr = smem + 35840;
    uint8_t* smem_v4 = reinterpret_cast<uint8_t*>(smem_raw + 33792);
    const int smem_v4_addr = smem + 33792;
    uint8_t* smem_v5 = reinterpret_cast<uint8_t*>(smem_raw + 34304);
    const int smem_v5_addr = smem + 34304;
    uint8_t* smem_v6 = reinterpret_cast<uint8_t*>(smem_raw + 34816);
    const int smem_v6_addr = smem + 34816;
    uint8_t* smem_v7 = reinterpret_cast<uint8_t*>(smem_raw + 35328);
    const int smem_v7_addr = smem + 35328;
    uint8_t* smem_v8 = reinterpret_cast<uint8_t*>(smem_raw + 35840);
    const int smem_v8_addr = smem + 35840;
    uint8_t* smem_v9 = reinterpret_cast<uint8_t*>(smem_raw + 36864);
    const int smem_v9_addr = smem + 36864;
    uint8_t* smem_v10 = reinterpret_cast<uint8_t*>(smem_raw + 37888);
    const int smem_v10_addr = smem + 37888;
    uint8_t* smem_v11 = reinterpret_cast<uint8_t*>(smem_raw + 38912);
    const int smem_v11_addr = smem + 38912;
    uint8_t* epi_staging = reinterpret_cast<uint8_t*>(smem_raw + 195584);
    const int epi_staging_addr = smem + 195584;
    unsigned int* work_response = reinterpret_cast<unsigned int*>(smem_raw + 228352);
    const int work_response_addr = smem + 228352;

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

    // TMEM alloc (512 columns, 304 used)
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
    const int tmem_tmem_sfa = taddr + 256;
    const int tmem_tmem_sfb = taddr + 272;

    // ---- Role: load ----
    if (warp == 0) {
        { // load_main
            unsigned int load_stage = 0;
            unsigned int work_stage = 0;
            int weight_row_base = cta_rank * B_HALF_N;
            unsigned int _phase_work_empty = 1;
            unsigned int _phase_mma_done = 1;
            unsigned int _phase_work_full = 0;
            if (elect_sync()) {
                unsigned int this_bid = bid;
                #pragma unroll 1
                for (unsigned int _tile_iter = 0; _tile_iter < m_tiles / CTA_GROUP * N_TILES; _tile_iter++) {
                    if (cta_rank == 0) {
                        mbarrier_wait_cluster_hint(work_empty_addr + (work_stage) * 8, _phase_work_empty, 10000000);
                        asm volatile(
                            "{\n\t"
                            ".reg .b32 remAddr32;\n\t"
                            "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                            "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [remAddr32], %2;\n\t"
                            "}"
                            :: "r"(work_full_addr + work_stage * 8), "r"(0), "r"((uint32_t)(16)) : "memory");
                        asm volatile(
                            "{\n\t"
                            ".reg .b32 remAddr32;\n\t"
                            "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                            "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [remAddr32], %2;\n\t"
                            "}"
                            :: "r"(work_full_addr + work_stage * 8), "r"(1), "r"((uint32_t)(16)) : "memory");
                        asm volatile(
                            "fence.proxy.async.shared::cta;\n\t"
                            "clusterlaunchcontrol.try_cancel.async.shared::cta"
                                ".mbarrier::complete_tx::bytes.multicast::cluster::all.b128"
                                " [%0], [%1];"
                            :: "r"(work_response_addr + work_stage * 16 + 0 * 16), "r"(work_full_addr + work_stage * 8)
                            : "memory");
                    }
                    int group = this_bid / (unsigned int)tiles_per_group;
                    int first_m = group * GROUP_M;
                    int remaining = m_tiles - first_m;
                    int group_size = ((remaining >= GROUP_M) ? GROUP_M : remaining);
                    int local = this_bid % (unsigned int)tiles_per_group;
                    int bid_m = first_m + local % group_size;
                    int bid_n = local / group_size;
                    int off_m = bid_m * BLOCK_M;
                    int off_n = bid_n * BLOCK_N;
                    int sfa_tile_row = bid_m * SF_K_TILES;
                    int sfb_tile_row = bid_n * SF_K_TILES;
                    #pragma unroll 1
                    for (int iter_k = 0; iter_k < NUM_K_ITERS; iter_k++) {
                        mbarrier_wait(mma_done_addr + (load_stage) * 8, _phase_mma_done);
                        tma_3d_gmem2smem_cta2(smem_a_addr + load_stage * 38912, (&A), 0, off_m, iter_k, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                        tma_3d_gmem2smem_cta2(smem_b_addr + load_stage * 38912, (&B), 0, weight_row_base + off_n, iter_k, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                        int k_set_base = iter_k * 4;
                        tma_3d_gmem2smem_cta2(smem_sfa_all_addr + load_stage * 38912, (&SFA), 0, 0, sfa_tile_row + k_set_base, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                        tma_3d_gmem2smem_cta2(smem_sfb_all_addr + load_stage * 38912, (&SFB), 0, 0, sfb_tile_row + k_set_base, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                        asm volatile(
                            "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                            :: "r"((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF), "r"((uint32_t)(38912)) : "memory");
                        load_stage += 1;
                        if (load_stage == 5) { load_stage = 0; _phase_mma_done ^= 1; }
                    }
                    mbarrier_wait(work_full_addr + (work_stage) * 8, _phase_work_full);
                    uint32_t _clc_valid_0 = 0;
                    uint32_t _clc_ctaid_x_0;
                    uint32_t _clc_ctaid_y_0;
                    uint32_t _clc_ctaid_z_0;
                    asm volatile(
                        "{\n\t"
                        ".reg .pred p1;\n\t"
                        ".reg .b128 clc_r;\n\t"
                        "ld.shared.b128 clc_r, [%4];\n\t"
                        "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                        "selp.u32 %3, 1, 0, p1;\n\t"
                        "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid.v4.b32.b128 {%0, %1, %2, _}, clc_r;\n\t"
                        "}\n"
                        : "=r"(_clc_ctaid_x_0), "=r"(_clc_ctaid_y_0), "=r"(_clc_ctaid_z_0), "=r"(_clc_valid_0)
                        : "r"(work_response_addr + work_stage * 16 + 0 * 16)
                        : "memory");
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    asm volatile(
                        "{\n\t"
                        ".reg .b32 remAddr32;\n\t"
                        "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                        "}"
                        :: "r"(work_empty_addr + work_stage * 8), "r"(0) : "memory");
                    work_stage += 1;
                    if (work_stage == 4) { work_stage = 0; _phase_work_empty ^= 1; _phase_work_full ^= 1; }
                    if (_clc_valid_0 == 0) {
                        break;
                    }
                    this_bid = _clc_ctaid_x_0 + (unsigned int)cta_rank;
                }
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
            unsigned int mma_tma_stage = 0;
            unsigned int work_stage_1 = 0;
            unsigned int _phase_epilogue_done = 1;
            unsigned int _phase_tma_full = 0;
            unsigned int _phase_work_full_1 = 0;
            if (cta_rank == 0) {
                #pragma unroll 1
                for (unsigned int _tile_iter_1 = 0; _tile_iter_1 < m_tiles / CTA_GROUP * N_TILES; _tile_iter_1++) {
                    mbarrier_wait(epilogue_done_addr, _phase_epilogue_done);
                    #pragma unroll 1
                    for (int iter_k_1 = 0; iter_k_1 < NUM_K_ITERS; iter_k_1++) {
                        mbarrier_wait(tma_full_addr + (mma_tma_stage) * 8, _phase_tma_full);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int init_flag = ((iter_k_1 == 0) ? 1 : 0);
                        uint32_t _elected_0 = elect_sync();
                        tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfa, make_sf_cp_desc_lo_sbo128((((smem_v4_addr) >> 4) + (mma_tma_stage) * 2432)), _elected_0);
                        tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfb, make_sf_cp_desc_lo_sbo128((((smem_v8_addr) >> 4) + (mma_tma_stage) * 2432)), _elected_0);
                        tcgen05_cp_32x128b_warpx4_cta2_leader((tmem_tmem_sfb + 4), make_sf_cp_desc_lo_sbo128((((smem_v8_addr) >> 4) + (mma_tma_stage) * 2432 + 32)), _elected_0);
                        tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfa + 4, make_sf_cp_desc_lo_sbo128((((smem_v5_addr) >> 4) + (mma_tma_stage) * 2432)), _elected_0);
                        tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfb + 8, make_sf_cp_desc_lo_sbo128((((smem_v9_addr) >> 4) + (mma_tma_stage) * 2432)), _elected_0);
                        tcgen05_cp_32x128b_warpx4_cta2_leader((tmem_tmem_sfb + 8 + 4), make_sf_cp_desc_lo_sbo128((((smem_v9_addr) >> 4) + (mma_tma_stage) * 2432 + 32)), _elected_0);
                        tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfa + 8, make_sf_cp_desc_lo_sbo128((((smem_v6_addr) >> 4) + (mma_tma_stage) * 2432)), _elected_0);
                        tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfb + 16, make_sf_cp_desc_lo_sbo128((((smem_v10_addr) >> 4) + (mma_tma_stage) * 2432)), _elected_0);
                        tcgen05_cp_32x128b_warpx4_cta2_leader((tmem_tmem_sfb + 16 + 4), make_sf_cp_desc_lo_sbo128((((smem_v10_addr) >> 4) + (mma_tma_stage) * 2432 + 32)), _elected_0);
                        tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfa + 12, make_sf_cp_desc_lo_sbo128((((smem_v7_addr) >> 4) + (mma_tma_stage) * 2432)), _elected_0);
                        tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfb + 24, make_sf_cp_desc_lo_sbo128((((smem_v11_addr) >> 4) + (mma_tma_stage) * 2432)), _elected_0);
                        tcgen05_cp_32x128b_warpx4_cta2_leader((tmem_tmem_sfb + 24 + 4), make_sf_cp_desc_lo_sbo128((((smem_v11_addr) >> 4) + (mma_tma_stage) * 2432 + 32)), _elected_0);
                        int _smem_desc_lo_1 = (((smem_a_addr) >> 4) & 0x3FFF) + (mma_tma_stage) * 2432;
                        int _smem_desc_lo_2 = (((smem_b_addr) >> 4) & 0x3FFF) + (mma_tma_stage) * 2432;
                        {
                            uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_1) | ((uint64_t)0x40004040 << 32);
                            uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_2) | ((uint64_t)0x40004040 << 32);

                            tcgen05_mma_mxf4nvf4_bs_cta2_leader(tmem_accum, a_desc + 0, b_desc + 0,
                                0x10400480U, tmem_tmem_sfa + 0, tmem_tmem_sfb + 0, ((init_flag) ? 0 : 1), _elected_0);
                        }
                        int _smem_desc_lo_3 = (((smem_a_addr + 32) >> 4) & 0x3FFF) + (mma_tma_stage) * 2432;
                        int _smem_desc_lo_4 = (((smem_b_addr + 32) >> 4) & 0x3FFF) + (mma_tma_stage) * 2432;
                        {
                            uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_3) | ((uint64_t)0x40004040 << 32);
                            uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_4) | ((uint64_t)0x40004040 << 32);

                            tcgen05_mma_mxf4nvf4_bs_cta2_leader(tmem_accum, a_desc + 0, b_desc + 0,
                                0x10400480U, tmem_tmem_sfa + 4 + 0, tmem_tmem_sfb + 8 + 0, 1, _elected_0);
                        }
                        int _smem_desc_lo_5 = (((smem_a_addr + 64) >> 4) & 0x3FFF) + (mma_tma_stage) * 2432;
                        int _smem_desc_lo_6 = (((smem_b_addr + 64) >> 4) & 0x3FFF) + (mma_tma_stage) * 2432;
                        {
                            uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_5) | ((uint64_t)0x40004040 << 32);
                            uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_6) | ((uint64_t)0x40004040 << 32);

                            tcgen05_mma_mxf4nvf4_bs_cta2_leader(tmem_accum, a_desc + 0, b_desc + 0,
                                0x10400480U, tmem_tmem_sfa + 8 + 0, tmem_tmem_sfb + 16 + 0, 1, _elected_0);
                        }
                        int _smem_desc_lo_7 = (((smem_a_addr + 96) >> 4) & 0x3FFF) + (mma_tma_stage) * 2432;
                        int _smem_desc_lo_8 = (((smem_b_addr + 96) >> 4) & 0x3FFF) + (mma_tma_stage) * 2432;
                        {
                            uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_7) | ((uint64_t)0x40004040 << 32);
                            uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_8) | ((uint64_t)0x40004040 << 32);

                            tcgen05_mma_mxf4nvf4_bs_cta2_leader(tmem_accum, a_desc + 0, b_desc + 0,
                                0x10400480U, tmem_tmem_sfa + 12 + 0, tmem_tmem_sfb + 24 + 0, 1, _elected_0);
                        }
                        elect_commit_cg2_multicast(mma_done_addr + (mma_tma_stage) * 8, (uint16_t)(3));
                        mma_tma_stage += 1;
                        if (mma_tma_stage == 5) { mma_tma_stage = 0; _phase_tma_full ^= 1; }
                    }
                    elect_commit_cg2_multicast(mainloop_done_addr, (uint16_t)(3));
                    _phase_epilogue_done ^= 1;
                    mbarrier_wait(work_full_addr + (work_stage_1) * 8, _phase_work_full_1);
                    uint32_t _clc_valid_1 = 0;
                    uint32_t _clc_ctaid_x_1;
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
                        : "=r"(_clc_ctaid_x_1), "=r"(_clc_ctaid_y_1), "=r"(_clc_ctaid_z_1), "=r"(_clc_valid_1)
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
                        :: "r"(work_empty_addr + work_stage_1 * 8), "r"(0) : "memory");
                    work_stage_1 += 1;
                    if (work_stage_1 == 4) { work_stage_1 = 0; _phase_work_full_1 ^= 1; }
                    if (_clc_valid_1 == 0) {
                        break;
                    }
                }
            }
        }
    }
    // ---- Role: epilogue ----
    if (warp >= 2 && warp <= 9) {
        { // epilogue_main
            unsigned int work_stage_2 = 0;
            const int epi_warp = warp % 4;
            const int head_half = (warp - 2) / 4;
            const int local_row = epi_warp * 32 + lane;
            float alpha_v = alpha[0];
            int rows_per_token = HEADS_PER_DESTINATION * 3;
            unsigned int this_bid_1 = bid;
            unsigned int tile_parity = 0;
            int store_bar = 14 + head_half;
            unsigned int _phase_mainloop_done = 0;
            unsigned int _phase_work_full_2 = 0;
            #pragma unroll 1
            for (unsigned int _tile_iter_2 = 0; _tile_iter_2 < m_tiles / CTA_GROUP * N_TILES; _tile_iter_2++) {
                mbarrier_wait(mainloop_done_addr, _phase_mainloop_done);
                asm volatile("tcgen05.fence::after_thread_sync;");
                int group_1 = this_bid_1 / (unsigned int)tiles_per_group;
                int first_m_1 = group_1 * GROUP_M;
                int remaining_1 = m_tiles - first_m_1;
                int group_size_1 = ((remaining_1 >= GROUP_M) ? GROUP_M : remaining_1);
                int local_1 = this_bid_1 % (unsigned int)tiles_per_group;
                int bid_m_1 = first_m_1 + local_1 % group_size_1;
                int bid_n_1 = local_1 / group_size_1;
                int off_m_1 = bid_m_1 * BLOCK_M;
                int off_n_1 = bid_n_1 * BLOCK_N;
                int token = off_m_1 + local_row;
                int head_kind = bid_n_1 * 2 + head_half;
                int head = head_kind / 3;
                int kind = head_kind % 3;
                int lane_addr = taddr + (unsigned int)(epi_warp * 32 << 16) + (unsigned int)(head_half * B_HALF_N);
                unsigned int words[64];
                #pragma unroll
                for (int c = 0; c < 4; c++) {
                    float _tmem_load_0[32];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(_tmem_load_0[0]), "=f"(_tmem_load_0[1]), "=f"(_tmem_load_0[2]), "=f"(_tmem_load_0[3]), "=f"(_tmem_load_0[4]), "=f"(_tmem_load_0[5]), "=f"(_tmem_load_0[6]), "=f"(_tmem_load_0[7]), "=f"(_tmem_load_0[8]), "=f"(_tmem_load_0[9]), "=f"(_tmem_load_0[10]), "=f"(_tmem_load_0[11]), "=f"(_tmem_load_0[12]), "=f"(_tmem_load_0[13]), "=f"(_tmem_load_0[14]), "=f"(_tmem_load_0[15]), "=f"(_tmem_load_0[16]), "=f"(_tmem_load_0[17]), "=f"(_tmem_load_0[18]), "=f"(_tmem_load_0[19]), "=f"(_tmem_load_0[20]), "=f"(_tmem_load_0[21]), "=f"(_tmem_load_0[22]), "=f"(_tmem_load_0[23]), "=f"(_tmem_load_0[24]), "=f"(_tmem_load_0[25]), "=f"(_tmem_load_0[26]), "=f"(_tmem_load_0[27]), "=f"(_tmem_load_0[28]), "=f"(_tmem_load_0[29]), "=f"(_tmem_load_0[30]), "=f"(_tmem_load_0[31])
                        : "r"(lane_addr + c * 32));
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    #pragma unroll
                    for (int j = 0; j < 16; j++) {
                        words[c * 16 + j] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_0[2 * j] * alpha_v, _tmem_load_0[2 * j + 1] * alpha_v)));
                    }
                }
                if (elect_sync()) {
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    asm volatile(
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                        :: "r"((epilogue_done_addr) & 0xFEFFFFFF) : "memory");
                }
                _phase_mainloop_done ^= 1;
                int destination = head / HEADS_PER_DESTINATION;
                int local_head = head % HEADS_PER_DESTINATION;
                int head_kind_local = local_head * 3 + kind;
                int row_in_destination = token * rows_per_token + head_kind_local;
                unsigned long long output_row = (unsigned long long)destination * (unsigned long long)ROWS_PER_DESTINATION + (unsigned long long)row_in_destination;
                unsigned long long q_row_base = output_row * 64;
                unsigned int sr = (unsigned int)row_in_destination;
                unsigned int scale_swizzle = sr >> 7 << 10 | (sr & 31) << 4 | (sr >> 5 & 3) << 2;
                unsigned long long scale_row_base = (unsigned long long)destination * (unsigned long long)SCALE_STRIDE + (unsigned long long)scale_swizzle;
                asm volatile("tcgen05.fence::before_thread_sync;");
                unsigned int staging_buf = (unsigned int)head_half;
                if (kind < 2) {
                    float partial[8];
                    #pragma unroll
                    for (int b = 0; b < 8; b++) {
                        float sum_lo = 0.0f;
                        #pragma unroll
                        for (int j_1 = 0; j_1 < 4; j_1++) {
                            float v_lo = __uint_as_float(words[b * 8 + j_1] << 16);
                            float v_hi = __uint_as_float(words[b * 8 + j_1] & 4294901760u);
                            sum_lo = __fmaf_rn(v_lo, v_lo, sum_lo);
                            sum_lo = __fmaf_rn(v_hi, v_hi, sum_lo);
                        }
                        float sum_hi = 0.0f;
                        #pragma unroll
                        for (int j_2 = 4; j_2 < 8; j_2++) {
                            float v_lo2 = __uint_as_float(words[b * 8 + j_2] << 16);
                            float v_hi2 = __uint_as_float(words[b * 8 + j_2] & 4294901760u);
                            sum_hi = __fmaf_rn(v_lo2, v_lo2, sum_hi);
                            sum_hi = __fmaf_rn(v_hi2, v_hi2, sum_hi);
                        }
                        partial[b] = sum_lo + sum_hi;
                    }
                    float sum_01 = partial[0] + partial[1];
                    float sum_23 = partial[2] + partial[3];
                    float sum_45 = partial[4] + partial[5];
                    float sum_67 = partial[6] + partial[7];
                    float sum_sq = sum_01 + sum_23 + (sum_45 + sum_67);
                    float mean_sq = __fdiv_rn(sum_sq, 128.0f);
                    float rstd = rsqrtf(mean_sq + eps);
                    float2 _f2_0 = make_float2(rstd, rstd);
                    unsigned int normalized[64];
                    #pragma unroll
                    for (int b_1 = 0; b_1 < 8; b_1++) {
                        unsigned int loaded_w[8];
                        #pragma unroll
                        for (int h = 0; h < 2; h++) {
                            {
                                const uint4* _vptr_0 = reinterpret_cast<const uint4*>(((kind == 0) ? q_norm_weight : k_norm_weight) + b_1 * 16 + h * 8);
                                uint4* _vdst_0 = reinterpret_cast<uint4*>(&loaded_w[4 * h]);
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
                                    _vdst_0[_blk] = _vptr_0[_blk];
                                }
                            }
                        }
                        #pragma unroll
                        for (int j_3 = 0; j_3 < 8; j_3++) {
                            float2 _f32x2_mul_a_1 = make_float2(__uint_as_float(words[b_1 * 8 + j_3] << 16), __uint_as_float(words[b_1 * 8 + j_3] & 4294901760u));
                            float2 scaled;
                            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled) : "l"(*(const unsigned long long*)&_f32x2_mul_a_1), "l"(*(const unsigned long long*)&_f2_0));
                            float2 _f32x2_mul_b_2 = make_float2(__uint_as_float(loaded_w[j_3] << 16), __uint_as_float(loaded_w[j_3] & 4294901760u));
                            float2 weighted;
                            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&weighted) : "l"(*(const unsigned long long*)&scaled), "l"(*(const unsigned long long*)&_f32x2_mul_b_2));
                            normalized[b_1 * 8 + j_3] = __as_u32(__float22bfloat162_rn(make_float2(weighted.x, weighted.y)));
                        }
                    }
                    int rope_token = ((token) < (M - 1) ? (token) : (M - 1));
                    #pragma unroll
                    for (int b_2 = 0; b_2 < 3; b_2++) {
                        unsigned int loaded_cos[8];
                        unsigned int loaded_sin[8];
                        #pragma unroll
                        for (int h_1 = 0; h_1 < 2; h_1++) {
                            {
                                const uint4* _vptr_3 = reinterpret_cast<const uint4*>(rope_cos_sin + rope_token * 96 + b_2 * 16 + h_1 * 8);
                                uint4* _vdst_3 = reinterpret_cast<uint4*>(&loaded_cos[4 * h_1]);
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
                                    _vdst_3[_blk] = _vptr_3[_blk];
                                }
                            }
                            {
                                const uint4* _vptr_4 = reinterpret_cast<const uint4*>(rope_cos_sin + rope_token * 96 + 48 + b_2 * 16 + h_1 * 8);
                                uint4* _vdst_4 = reinterpret_cast<uint4*>(&loaded_sin[4 * h_1]);
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
                                    _vdst_4[_blk] = _vptr_4[_blk];
                                }
                            }
                        }
                        #pragma unroll
                        for (int j_4 = 0; j_4 < 8; j_4++) {
                            unsigned int n_lo_word = normalized[b_2 * 8 + j_4];
                            unsigned int n_hi_word = normalized[(b_2 + 3) * 8 + j_4];
                            float2 _f2_1 = make_float2(__uint_as_float(loaded_cos[j_4] << 16), __uint_as_float(loaded_cos[j_4] & 4294901760u));
                            float2 _f2_2 = make_float2(__uint_as_float(loaded_sin[j_4] << 16 ^ 2147483648u), __uint_as_float(loaded_sin[j_4] & 4294901760u ^ 2147483648u));
                            float2 _f2_3 = make_float2(__uint_as_float(loaded_sin[j_4] << 16), __uint_as_float(loaded_sin[j_4] & 4294901760u));
                            float2 _f32x2_mul_a_5 = make_float2(__uint_as_float(n_hi_word << 16), __uint_as_float(n_hi_word & 4294901760u));
                            float2 term_lo;
                            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&term_lo) : "l"(*(const unsigned long long*)&_f32x2_mul_a_5), "l"(*(const unsigned long long*)&_f2_2));
                            float2 rotated = fma_f32x2_rn_ftz(_f2_1, make_float2(__uint_as_float(n_lo_word << 16), __uint_as_float(n_lo_word & 4294901760u)), term_lo);
                            float2 _f32x2_mul_a_6 = make_float2(__uint_as_float(n_lo_word << 16), __uint_as_float(n_lo_word & 4294901760u));
                            float2 term_hi;
                            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&term_hi) : "l"(*(const unsigned long long*)&_f32x2_mul_a_6), "l"(*(const unsigned long long*)&_f2_3));
                            float2 rotated_hi = fma_f32x2_rn_ftz(_f2_1, make_float2(__uint_as_float(n_hi_word << 16), __uint_as_float(n_hi_word & 4294901760u)), term_hi);
                            normalized[b_2 * 8 + j_4] = __as_u32(__float22bfloat162_rn(make_float2(rotated.x, rotated.y)));
                            normalized[(b_2 + 3) * 8 + j_4] = __as_u32(__float22bfloat162_rn(make_float2(rotated_hi.x, rotated_hi.y)));
                        }
                    }
                    if (write_debug != 0) {
                        if (token < M) {
                            unsigned long long debug_row = ((unsigned long long)token * 56 + (unsigned long long)head) * 64;
                            #pragma unroll
                            for (int c_1 = 0; c_1 < 16; c_1++) {
                                unsigned int debug_chunk[4];
                                #pragma unroll
                                for (int j_5 = 0; j_5 < 4; j_5++) {
                                    debug_chunk[j_5] = normalized[4 * c_1 + j_5];
                                }
                                {
                                    auto _gst_7 = ((kind == 0) ? debug_q_words : debug_k_words) + (debug_row + (unsigned long long)(4 * c_1));
                                    __builtin_assume(__isGlobal(_gst_7));
                                    {
                                        int4 _iv4 = make_int4(debug_chunk[0 + 0], debug_chunk[0 + 1], debug_chunk[0 + 2], debug_chunk[0 + 3]);
                                        *reinterpret_cast<int4*>(_gst_7 + 0) = _iv4;
                                    }
                                }
                            }
                        }
                    }
                    if (epi_warp == 0) {
                        if (elect_sync()) {
                            asm volatile("cp.async.bulk.wait_group.read 0;");
                        }
                    }
                    asm volatile("barrier.sync %0, 128;" :: "r"(store_bar) : "memory");
                    float out_scale_sel = ((kind == 0) ? out_global_scale[0] : ((kind == 1) ? out_global_scale[1] : out_global_scale[2]));
                    float2 _f2_4 = make_float2(out_scale_sel, out_scale_sel);
                    #pragma unroll
                    for (int b_3 = 0; b_3 < 8; b_3++) {
                        float cast_values[16];
                        unsigned int e4m3_words[4];
                        #pragma unroll
                        for (int j_6 = 0; j_6 < 8; j_6++) {
                            float2 _f32x2_mul_a_8 = make_float2(__uint_as_float(normalized[b_3 * 8 + j_6] << 16), __uint_as_float(normalized[b_3 * 8 + j_6] & 4294901760u));
                            float2 scaled_1;
                            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_8), "l"(*(const unsigned long long*)&_f2_4));
                            cast_values[2 * j_6] = scaled_1.x;
                            cast_values[2 * j_6 + 1] = scaled_1.y;
                        }
                        {
                            uint32_t _packed;
                            asm volatile("{\n\t"
                                ".reg .b16 _lo;\n\t"
                                ".reg .b16 _hi;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                "mov.b32 %0, {_lo, _hi};\n\t"
                                "}"
                                : "=r"(_packed) : "f"(cast_values[0]), "f"(cast_values[1]),
                                                   "f"(cast_values[2]), "f"(cast_values[3]));
                            e4m3_words[0] = _packed;
                        }
                        {
                            uint32_t _packed;
                            asm volatile("{\n\t"
                                ".reg .b16 _lo;\n\t"
                                ".reg .b16 _hi;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                "mov.b32 %0, {_lo, _hi};\n\t"
                                "}"
                                : "=r"(_packed) : "f"(cast_values[4]), "f"(cast_values[5]),
                                                   "f"(cast_values[6]), "f"(cast_values[7]));
                            e4m3_words[1] = _packed;
                        }
                        {
                            uint32_t _packed;
                            asm volatile("{\n\t"
                                ".reg .b16 _lo;\n\t"
                                ".reg .b16 _hi;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                "mov.b32 %0, {_lo, _hi};\n\t"
                                "}"
                                : "=r"(_packed) : "f"(cast_values[8]), "f"(cast_values[9]),
                                                   "f"(cast_values[10]), "f"(cast_values[11]));
                            e4m3_words[2] = _packed;
                        }
                        {
                            uint32_t _packed;
                            asm volatile("{\n\t"
                                ".reg .b16 _lo;\n\t"
                                ".reg .b16 _hi;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                "mov.b32 %0, {_lo, _hi};\n\t"
                                "}"
                                : "=r"(_packed) : "f"(cast_values[12]), "f"(cast_values[13]),
                                                   "f"(cast_values[14]), "f"(cast_values[15]));
                            e4m3_words[3] = _packed;
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((epi_staging_addr + staging_buf * 16384 + (unsigned int)(local_row * 128 + b_3 * 16 ^ (local_row * 128 + b_3 * 16 >> 7 & 7) << 4))), "r"(e4m3_words[0]), "r"(e4m3_words[1]), "r"(e4m3_words[2]), "r"(e4m3_words[3]) : "memory");
                    }
                } else {
                    if (epi_warp == 0) {
                        if (elect_sync()) {
                            asm volatile("cp.async.bulk.wait_group.read 0;");
                        }
                    }
                    asm volatile("barrier.sync %0, 128;" :: "r"(store_bar) : "memory");
                    float out_scale_sel_1 = ((kind == 0) ? out_global_scale[0] : ((kind == 1) ? out_global_scale[1] : out_global_scale[2]));
                    float2 _f2_5 = make_float2(out_scale_sel_1, out_scale_sel_1);
                    #pragma unroll
                    for (int b_4 = 0; b_4 < 8; b_4++) {
                        float cast_values_1[16];
                        unsigned int e4m3_words_1[4];
                        #pragma unroll
                        for (int j_7 = 0; j_7 < 8; j_7++) {
                            float2 _f32x2_mul_a_9 = make_float2(__uint_as_float(words[b_4 * 8 + j_7] << 16), __uint_as_float(words[b_4 * 8 + j_7] & 4294901760u));
                            float2 scaled_2;
                            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_2) : "l"(*(const unsigned long long*)&_f32x2_mul_a_9), "l"(*(const unsigned long long*)&_f2_5));
                            cast_values_1[2 * j_7] = scaled_2.x;
                            cast_values_1[2 * j_7 + 1] = scaled_2.y;
                        }
                        {
                            uint32_t _packed;
                            asm volatile("{\n\t"
                                ".reg .b16 _lo;\n\t"
                                ".reg .b16 _hi;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                "mov.b32 %0, {_lo, _hi};\n\t"
                                "}"
                                : "=r"(_packed) : "f"(cast_values_1[0]), "f"(cast_values_1[1]),
                                                   "f"(cast_values_1[2]), "f"(cast_values_1[3]));
                            e4m3_words_1[0] = _packed;
                        }
                        {
                            uint32_t _packed;
                            asm volatile("{\n\t"
                                ".reg .b16 _lo;\n\t"
                                ".reg .b16 _hi;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                "mov.b32 %0, {_lo, _hi};\n\t"
                                "}"
                                : "=r"(_packed) : "f"(cast_values_1[4]), "f"(cast_values_1[5]),
                                                   "f"(cast_values_1[6]), "f"(cast_values_1[7]));
                            e4m3_words_1[1] = _packed;
                        }
                        {
                            uint32_t _packed;
                            asm volatile("{\n\t"
                                ".reg .b16 _lo;\n\t"
                                ".reg .b16 _hi;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                "mov.b32 %0, {_lo, _hi};\n\t"
                                "}"
                                : "=r"(_packed) : "f"(cast_values_1[8]), "f"(cast_values_1[9]),
                                                   "f"(cast_values_1[10]), "f"(cast_values_1[11]));
                            e4m3_words_1[2] = _packed;
                        }
                        {
                            uint32_t _packed;
                            asm volatile("{\n\t"
                                ".reg .b16 _lo;\n\t"
                                ".reg .b16 _hi;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                "mov.b32 %0, {_lo, _hi};\n\t"
                                "}"
                                : "=r"(_packed) : "f"(cast_values_1[12]), "f"(cast_values_1[13]),
                                                   "f"(cast_values_1[14]), "f"(cast_values_1[15]));
                            e4m3_words_1[3] = _packed;
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((epi_staging_addr + staging_buf * 16384 + (unsigned int)(local_row * 128 + b_4 * 16 ^ (local_row * 128 + b_4 * 16 >> 7 & 7) << 4))), "r"(e4m3_words_1[0]), "r"(e4m3_words_1[1]), "r"(e4m3_words_1[2]), "r"(e4m3_words_1[3]) : "memory");
                    }
                }
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("barrier.sync %0, 128;" :: "r"(store_bar) : "memory");
                if (epi_warp == 0) {
                    if (elect_sync()) {
                        tma_store_4d((&OUTQ), 0, head_kind_local, off_m_1, destination, epi_staging_addr + staging_buf * 16384);
                        asm volatile("cp.async.bulk.commit_group;");
                    }
                }
                tile_parity = tile_parity ^ 1;
                mbarrier_wait(work_full_addr + (work_stage_2) * 8, _phase_work_full_2);
                uint32_t _clc_valid_2 = 0;
                uint32_t _clc_ctaid_x_2;
                uint32_t _clc_ctaid_y_2;
                uint32_t _clc_ctaid_z_2;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%4];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "selp.u32 %3, 1, 0, p1;\n\t"
                    "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid.v4.b32.b128 {%0, %1, %2, _}, clc_r;\n\t"
                    "}\n"
                    : "=r"(_clc_ctaid_x_2), "=r"(_clc_ctaid_y_2), "=r"(_clc_ctaid_z_2), "=r"(_clc_valid_2)
                    : "r"(work_response_addr + work_stage_2 * 16 + 0 * 16)
                    : "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile(
                    "{\n\t"
                    ".reg .b32 remAddr32;\n\t"
                    "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                    "}"
                    :: "r"(work_empty_addr + work_stage_2 * 8), "r"(0) : "memory");
                work_stage_2 += 1;
                if (work_stage_2 == 4) { work_stage_2 = 0; _phase_work_full_2 ^= 1; }
                if (_clc_valid_2 == 0) {
                    break;
                }
                this_bid_1 = _clc_ctaid_x_2 + (unsigned int)cta_rank;
            }
            if (epi_warp == 0) {
                if (elect_sync()) {
                    asm volatile("cp.async.bulk.wait_group 0;");
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
