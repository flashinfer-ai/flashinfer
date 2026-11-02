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

#define CAKE_INF CUDART_INF_F
#define TMEM_NCOLS 160
#define TMEM_ACCUM_OFFSET 0
#define TMEM_SFA_OFFSET 16
#define TMEM_SFB_OFFSET 112
#define NUM_K_PIPE_STAGES 6
#define NUM_MMA_PIPE_STAGES 2
#define NUM_WORK_PIPE_STAGES 3
#define NUM_THROTTLE_PIPE_STAGES 3
#define SMEM_SMEM_A_OFF 1024
#define SMEM_SMEM_A_STAGE_BYTES 16384
#define SMEM_SMEM_A_STRIDE 16384
#define SMEM_SMEM_B_OFF 99328
#define SMEM_SMEM_B_STAGE_BYTES 1024
#define SMEM_SMEM_B_STRIDE 1024
#define SMEM_SMEM_SFA_OFF 105472
#define SMEM_SMEM_SFA_STAGE_BYTES 2048
#define SMEM_SMEM_SFA_STRIDE 2048
#define SMEM_SMEM_SFB_OFF 117760
#define SMEM_SMEM_SFB_STAGE_BYTES 128
#define SMEM_SMEM_SFB_STRIDE 128
#define SMEM_EPI_STAGING_OFF 118784
#define SMEM_EPI_STAGING_STAGE_BYTES 2048
#define SMEM_EPI_STAGING_STRIDE 2048
#define SMEM_WORK_RESPONSE_OFF 120832
#define SMEM_WORK_RESPONSE_STAGE_BYTES 16
#define SMEM_WORK_RESPONSE_STRIDE 16
#define SMEM_TOTAL 120960
#define THREADS 512
#define LAUNCH_MIN_BLOCKS 1

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


__device__ __forceinline__ uint32_t mbarrier_try_wait(int mbar_addr, int phase) {
    uint32_t token;
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64"
        " P1, [%1], %2;\n\t"
        "selp.u32 %0, 1, 0, P1;\n\t"
        "}\n"
        : "=r"(token)
        : "r"(mbar_addr), "r"(phase) : "memory");
    return token;
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

__device__ __forceinline__ void mbarrier_wait_hint(
        int mbar_addr, int phase, uint32_t suspend_time_hint) {
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        ".reg .u32 WAIT_ADDR;\n\t"
        "mov.u32 WAIT_ADDR, %0;\n\t"
        "LAB_WAIT_HINT:\n\t"
        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64"
        " P1, [WAIT_ADDR], %1, %2;\n\t"
        "@P1 bra.uni DONE_HINT;\n\t"
        "bra.uni LAB_WAIT_HINT;\n\t"
        "DONE_HINT:\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"(phase), "r"(suspend_time_hint) : "memory");
}

__device__ __forceinline__ void mbarrier_wait_token_hint(
        int mbar_addr, int phase, uint32_t token, uint32_t suspend_time_hint) {
    if (token == 0) {
        mbarrier_wait_hint(mbar_addr, phase, suspend_time_hint);
    }
}


__device__ __forceinline__ void tcgen05_mma_mxf4nvf4_bs(
    int taddr, uint64_t a_desc, uint64_t b_desc, uint32_t i_desc,
    int sfa_taddr, int sfb_taddr, int enable_input_d) {
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, %6, 0;\n\t"
        "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X"
        " [%0], %1, %2, %3, [%4], [%5], p;\n\t"
        "}\n"
        :: "r"(taddr), "l"(a_desc), "l"(b_desc),
           "r"(i_desc), "r"(sfa_taddr), "r"(sfb_taddr),
           "r"(enable_input_d));
}




union MmaSmemDesc {
    uint64_t u64;
    uint32_t u32[2];
};


__device__ __forceinline__ void elect_commit(int mbar_addr) {
    asm volatile(
        "{\n\t"
        ".reg .pred leader;\n\t"
        "elect.sync _|leader, 0xFFFFFFFF;\n\t"
        "@leader tcgen05.commit.cta_group::1.mbarrier::arrive::one"
        ".shared::cluster.b64 [%0];\n\t"
        "}\n"
        :: "r"(mbar_addr));
}


__device__ __forceinline__ void mbarrier_arrive(int mbar_addr) {
    asm volatile(
        "mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];"
        :: "r"(mbar_addr) : "memory");
}


__device__ __forceinline__ void mbarrier_arrive_expect_tx(int mbar_addr, uint32_t bytes) {
    asm volatile(
        "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
        :: "r"(mbar_addr), "r"(bytes) : "memory");
}




// ex2_emulation_f32x2 defined in softmax_frag_exp2_cast helper (or standalone)


__device__ __forceinline__ void elect_commit2(int mbar_addr0, int mbar_addr1) {
    asm volatile(
        "{\n\t"
        ".reg .pred leader;\n\t"
        "elect.sync _|leader, 0xFFFFFFFF;\n\t"
        "@leader tcgen05.commit.cta_group::1.mbarrier::arrive::one"
        ".shared::cluster.b64 [%0];\n\t"
        "@leader tcgen05.commit.cta_group::1.mbarrier::arrive::one"
        ".shared::cluster.b64 [%1];\n\t"
        "}\n"
        :: "r"(mbar_addr0), "r"(mbar_addr1) : "memory");
}






__device__ __forceinline__ void tma_3d_gmem2smem(
    int dst, const void *tmap_ptr, int x, int y, int z, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cta.global"
        ".mbarrier::complete_tx::bytes"
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



__device__ __forceinline__ void tcgen05_commit_leader(
    int mbar_addr, uint32_t leader) {
    asm volatile(
        "{\n\t"
        ".reg .pred leader;\n\t"
        "setp.ne.b32 leader, %1, 0;\n\t"
        "@leader tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"(leader) : "memory");
}


__device__ __forceinline__ uint32_t make_warp_uniform(uint32_t val) {
    uint32_t result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1f, 0xffffffff;"
        : "=r"(result) : "r"(val));
    return result;
}

extern "C" {

__global__ __launch_bounds__(512, LAUNCH_MIN_BLOCKS) void
kernel_cake_warp_decode_c2a25e69aaf3a844baef(const __grid_constant__ CUtensorMap A, const __grid_constant__ CUtensorMap B, const __grid_constant__ CUtensorMap SFA, const __grid_constant__ CUtensorMap SFB, const __grid_constant__ CUtensorMap C, float* __restrict__ scale_c, int* __restrict__ tile_expert, int* __restrict__ tile_mn_limit, int* __restrict__ num_tiles, int* __restrict__ work_counter, int grid_m, int max_tiles, int K_tiles)
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
    #define a_full_addr (mbar_base + 0)
    #define a_empty_addr (mbar_base + 48)
    #define b_full_addr (mbar_base + 96)
    #define b_empty_addr (mbar_base + 144)
    #define sfa_full_addr (mbar_base + 192)
    #define sfa_empty_addr (mbar_base + 240)
    #define sfb_full_addr (mbar_base + 288)
    #define sfb_empty_addr (mbar_base + 336)
    #define tmem_sfa_full_addr (mbar_base + 384)
    #define tmem_sfa_empty_addr (mbar_base + 432)
    #define tmem_sfb_full_addr (mbar_base + 480)
    #define tmem_sfb_empty_addr (mbar_base + 528)
    #define mma_full_addr (mbar_base + 576)
    #define mma_empty_addr (mbar_base + 592)
    #define work_full_addr (mbar_base + 608)
    #define work_empty_addr (mbar_base + 632)
    #define throttle_full_addr (mbar_base + 656)
    #define throttle_empty_addr (mbar_base + 680)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    const unsigned int clusters_x = gridDim.x / 1;
    const unsigned int cluster_id = ((blockIdx.z * gridDim.y + blockIdx.y) * clusters_x) + blockIdx.x / 1;
    const unsigned int num_clusters = clusters_x * gridDim.y * gridDim.z;

    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));

    // Kernel setup ops
    uint8_t* smem_a = reinterpret_cast<uint8_t*>(smem_raw + 1024);
    const int smem_a_addr = smem + 1024;
    uint8_t* smem_b = reinterpret_cast<uint8_t*>(smem_raw + 99328);
    const int smem_b_addr = smem + 99328;
    uint8_t* smem_sfa = reinterpret_cast<uint8_t*>(smem_raw + 105472);
    const int smem_sfa_addr = smem + 105472;
    uint8_t* smem_sfb = reinterpret_cast<uint8_t*>(smem_raw + 117760);
    const int smem_sfb_addr = smem + 117760;
    unsigned long long* epi_staging = reinterpret_cast<unsigned long long*>(smem_raw + 118784);
    const int epi_staging_addr = smem + 118784;
    unsigned int* work_response = reinterpret_cast<unsigned int*>(smem_raw + 120832);
    const int work_response_addr = smem + 120832;
    asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&A))) : "memory");
    asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&B))) : "memory");
    asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&SFA))) : "memory");
    asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&SFB))) : "memory");
    asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&C))) : "memory");
    int vec_load[1];
    {
        uint32_t _scalar_bits_0;
        asm volatile("ld.global.b32 %0, [%1];"
            : "=r"(_scalar_bits_0) : "l"((const void*)(num_tiles + (0))) : "memory");
        vec_load[0] = (int32_t)_scalar_bits_0;
    }
    unsigned int num_live = (unsigned int)vec_load[0];
    unsigned int first = (unsigned int)bid;
    unsigned int first_m = first % (unsigned int)grid_m;
    unsigned int first_n = first / (unsigned int)grid_m;
    int vec_load_0[1];
    {
        uint32_t _scalar_bits_1;
        asm volatile("ld.global.b32 %0, [%1];"
            : "=r"(_scalar_bits_1) : "l"((const void*)(tile_expert + (first_n))) : "memory");
        vec_load_0[0] = (int32_t)_scalar_bits_1;
    }
    int first_expert = vec_load_0[0];

    // Mbarrier init (18 pipeline groups, 0 ordered-sequence groups, 88 barriers)
    // Mbarriers at smem_raw[0..704)

    if (warp == 0) {
        // --- pipeline 'k_pipe' ---
        // a_full: 6 barriers, init_count=1
        // a_empty: 6 barriers, init_count=1
        // b_full: 6 barriers, init_count=1
        // b_empty: 6 barriers, init_count=1
        // sfa_full: 6 barriers, init_count=1
        // sfa_empty: 6 barriers, init_count=1
        // sfb_full: 6 barriers, init_count=1
        // sfb_empty: 6 barriers, init_count=1
        // tmem_sfa_full: 6 barriers, init_count=1
        // tmem_sfa_empty: 6 barriers, init_count=1
        // tmem_sfb_full: 6 barriers, init_count=128
        // tmem_sfb_empty: 6 barriers, init_count=1
        // --- pipeline 'mma_pipe' ---
        // mma_full: 2 barriers, init_count=1
        // mma_empty: 2 barriers, init_count=128
        // --- pipeline 'work_pipe' ---
        // work_full: 3 barriers, init_count=1
        // work_empty: 3 barriers, init_count=480
        // --- pipeline 'throttle_pipe' ---
        // throttle_full: 3 barriers, init_count=32
        // throttle_empty: 3 barriers, init_count=32
        // Warp-cooperative initialization in physical record order.
        mbarrier_init(smem + 0 + lane * 8, 1);
        uint32_t _mbarrier_init_count_0_32 = 128;
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_0_32) : "r"(lane), "n"(28), "r"((uint32_t)(1)));
        mbarrier_init(smem + 256 + lane * 8, _mbarrier_init_count_0_32);
        uint32_t _mbarrier_init_count_0_64 = 32;
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_0_64) : "r"(lane), "n"(18), "r"((uint32_t)(480)));
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_0_64) : "r"(lane), "n"(15), "r"((uint32_t)(1)));
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_0_64) : "r"(lane), "n"(12), "r"((uint32_t)(128)));
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_0_64) : "r"(lane), "n"(10), "r"((uint32_t)(1)));
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_0_64) : "r"(lane), "n"(2), "r"((uint32_t)(128)));
        if (lane < 24) {
            mbarrier_init(smem + 512 + lane * 8, _mbarrier_init_count_0_64);
        }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }

    __syncthreads();

    // TMEM alloc (256 columns, 160 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 704);
    if (warp == 0) {
        int _tmem_hold = smem + 704;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(256) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_accum = taddr;
    const int tmem_sfa = taddr + 16;
    const int tmem_sfb = taddr + 112;
    if (warp == 10) {
        if (first < num_live * (unsigned int)grid_m) {
            if (elect_sync()) {
                #pragma unroll 1
                for (int k = 0; k < K_tiles; k++) {
                    asm volatile("cp.async.bulk.prefetch.tensor.3d.L2.global.tile [%0, {%1, %2, %3}];" :: "l"((uint64_t)((&A))), "r"((int)(k * 256)), "r"((int)(first_m * 128)), "r"((int)(first_expert)) : "memory");
                    asm volatile("cp.async.bulk.prefetch.tensor.4d.L2.global.tile [%0, {%1, %2, %3, %4}];" :: "l"((uint64_t)((&SFA))), "r"((int)(0)), "r"((int)(0)), "r"((int)(k * 4)), "r"((int)((unsigned int)first_expert * (unsigned int)grid_m + first_m)) : "memory");
                }
            }
        }
    }
    if (warp >= 8 && warp <= 15) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 24;");
    }

    // ---- Role: load_a ----
    if (warp == 10) {
        { // load_a_main
            unsigned int live_tiles = num_live * (unsigned int)grid_m;
            unsigned int stage = 0;
            unsigned int work_stage = 0;
            unsigned int throttle_stage = 0;
            unsigned int valid = (unsigned int)(first < live_tiles);
            unsigned int m_tile = first_m;
            unsigned int n_tile = first_n;
            unsigned int _phase_throttle_empty = 1;
            unsigned int _phase_a_empty = 1;
            unsigned int _phase_work_full = 0;
            #pragma unroll 1
            for (unsigned int _tile = 0; _tile < (unsigned int)grid_m * (unsigned int)max_tiles; _tile++) {
                if (valid == 0) {
                    break;
                }
                int vec_load_1[1];
                {
                    uint32_t _scalar_bits_0;
                    asm volatile("ld.global.b32 %0, [%1];"
                        : "=r"(_scalar_bits_0) : "l"((const void*)(tile_expert + (n_tile))) : "memory");
                    vec_load_1[0] = (int32_t)_scalar_bits_0;
                }
                int expert = vec_load_1[0];
                mbarrier_wait_hint(throttle_empty_addr + (throttle_stage) * 8, _phase_throttle_empty, 10000000);
                mbarrier_arrive(throttle_full_addr + (throttle_stage) * 8);
                throttle_stage += 1;
                if (throttle_stage == 3) { throttle_stage = 0; _phase_throttle_empty ^= 1; }
                #pragma unroll 1
                for (int k_1 = 0; k_1 < K_tiles; k_1++) {
                    mbarrier_wait_hint(a_empty_addr + (stage) * 8, _phase_a_empty, 10000000);
                    if (elect_sync()) {
                        asm volatile(
                            "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                            " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                            :: "r"(smem_a_addr + stage * 16384), "l"((&A)), "r"(k_1 * 256), "r"(m_tile * 128), "r"(expert),
                               "r"(a_full_addr + (stage) * 8), "l"(0x12F0000000000000ULL) : "memory");
                        mbarrier_arrive_expect_tx(a_full_addr + (stage) * 8, 16384);
                    }
                    stage += 1;
                    if (stage == 6) { stage = 0; _phase_a_empty ^= 1; }
                }
                mbarrier_wait_hint(work_full_addr + (work_stage) * 8, _phase_work_full, 10000000);
                int index = work_stage * 4;
                unsigned int valid_1 = work_response[index];
                unsigned int m_tile_0 = work_response[index + 1];
                unsigned int n_tile_0 = work_response[index + 2];
                mbarrier_arrive(work_empty_addr + (work_stage) * 8);
                work_stage += 1;
                if (work_stage == 3) { work_stage = 0; _phase_work_full ^= 1; }
                valid = valid_1;
                m_tile = m_tile_0;
                n_tile = n_tile_0;
            }
            #pragma unroll 1
            for (int _ = 0; _ < 6; _++) {
                mbarrier_wait(a_empty_addr + (stage) * 8, _phase_a_empty);
                stage += 1;
                if (stage == 6) { stage = 0; _phase_a_empty ^= 1; }
            }
        }
    }
    // ---- Role: load_b ----
    if (warp == 8) {
        { // load_b_main
            asm volatile("griddepcontrol.wait;" ::: "memory");
            unsigned int live_tiles_1 = num_live * (unsigned int)grid_m;
            unsigned int stage_1 = 0;
            unsigned int work_stage_1 = 0;
            unsigned int valid_2 = (unsigned int)(first < live_tiles_1);
            unsigned int n_tile_1 = first_n;
            unsigned int m_tile_1 = 0;
            unsigned int _phase_b_empty = 1;
            unsigned int _phase_work_full_1 = 0;
            #pragma unroll 1
            for (unsigned int _tile_1 = 0; _tile_1 < (unsigned int)grid_m * (unsigned int)max_tiles; _tile_1++) {
                if (valid_2 == 0) {
                    break;
                }
                #pragma unroll 1
                for (int k_2 = 0; k_2 < K_tiles; k_2++) {
                    mbarrier_wait_hint(b_empty_addr + (stage_1) * 8, _phase_b_empty, 10000000);
                    if (elect_sync()) {
                        tma_3d_gmem2smem(smem_b_addr + stage_1 * 1024, (&B), k_2 * 256, 0, n_tile_1, b_full_addr + (stage_1) * 8);
                        mbarrier_arrive_expect_tx(b_full_addr + (stage_1) * 8, 1024);
                    }
                    stage_1 += 1;
                    if (stage_1 == 6) { stage_1 = 0; _phase_b_empty ^= 1; }
                }
                mbarrier_wait_hint(work_full_addr + (work_stage_1) * 8, _phase_work_full_1, 10000000);
                int index_1 = work_stage_1 * 4;
                unsigned int valid_3 = work_response[index_1];
                unsigned int m_tile_3 = work_response[index_1 + 1];
                unsigned int n_tile_3 = work_response[index_1 + 2];
                mbarrier_arrive(work_empty_addr + (work_stage_1) * 8);
                work_stage_1 += 1;
                if (work_stage_1 == 3) { work_stage_1 = 0; _phase_work_full_1 ^= 1; }
                valid_2 = valid_3;
                m_tile_1 = m_tile_3;
                n_tile_1 = n_tile_3;
            }
            asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
            #pragma unroll 1
            for (int __1 = 0; __1 < 6; __1++) {
                mbarrier_wait(b_empty_addr + (stage_1) * 8, _phase_b_empty);
                stage_1 += 1;
                if (stage_1 == 6) { stage_1 = 0; _phase_b_empty ^= 1; }
            }
        }
    }
    // ---- Role: load_sfa ----
    if (warp == 11) {
        { // load_sfa_main
            unsigned int live_tiles_2 = num_live * (unsigned int)grid_m;
            unsigned int stage_2 = 0;
            unsigned int work_stage_2 = 0;
            unsigned int valid_4 = (unsigned int)(first < live_tiles_2);
            unsigned int m_tile_2 = first_m;
            unsigned int n_tile_2 = first_n;
            unsigned int _phase_sfa_empty = 1;
            unsigned int _phase_work_full_2 = 0;
            #pragma unroll 1
            for (unsigned int _tile_2 = 0; _tile_2 < (unsigned int)grid_m * (unsigned int)max_tiles; _tile_2++) {
                if (valid_4 == 0) {
                    break;
                }
                int vec_load_2[1];
                {
                    uint32_t _scalar_bits_0;
                    asm volatile("ld.global.b32 %0, [%1];"
                        : "=r"(_scalar_bits_0) : "l"((const void*)(tile_expert + (n_tile_2))) : "memory");
                    vec_load_2[0] = (int32_t)_scalar_bits_0;
                }
                unsigned int sf_tile = (unsigned int)vec_load_2[0] * (unsigned int)grid_m + m_tile_2;
                #pragma unroll 1
                for (int k_3 = 0; k_3 < K_tiles; k_3++) {
                    mbarrier_wait_hint(sfa_empty_addr + (stage_2) * 8, _phase_sfa_empty, 10000000);
                    if (elect_sync()) {
                        asm volatile(
                            "cp.async.bulk.tensor.4d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                            " [%0], [%1, {%2, %3, %4, %5}], [%6], %7;"
                            :: "r"(smem_sfa_addr + stage_2 * 2048), "l"((&SFA)), "r"(0), "r"(0), "r"(k_3 * 4), "r"(sf_tile),
                               "r"(sfa_full_addr + (stage_2) * 8), "l"(0x12F0000000000000ULL) : "memory");
                        mbarrier_arrive_expect_tx(sfa_full_addr + (stage_2) * 8, 2048);
                    }
                    stage_2 += 1;
                    if (stage_2 == 6) { stage_2 = 0; _phase_sfa_empty ^= 1; }
                }
                mbarrier_wait_hint(work_full_addr + (work_stage_2) * 8, _phase_work_full_2, 10000000);
                int index_2 = work_stage_2 * 4;
                unsigned int valid_2_1 = work_response[index_2];
                unsigned int m_tile_2_1 = work_response[index_2 + 1];
                unsigned int n_tile_1_1 = work_response[index_2 + 2];
                mbarrier_arrive(work_empty_addr + (work_stage_2) * 8);
                work_stage_2 += 1;
                if (work_stage_2 == 3) { work_stage_2 = 0; _phase_work_full_2 ^= 1; }
                valid_4 = valid_2_1;
                m_tile_2 = m_tile_2_1;
                n_tile_2 = n_tile_1_1;
            }
            #pragma unroll 1
            for (int __2 = 0; __2 < 6; __2++) {
                mbarrier_wait(sfa_empty_addr + (stage_2) * 8, _phase_sfa_empty);
                stage_2 += 1;
                if (stage_2 == 6) { stage_2 = 0; _phase_sfa_empty ^= 1; }
            }
        }
    }
    // ---- Role: load_sfb ----
    if (warp == 9) {
        { // load_sfb_main
            asm volatile("griddepcontrol.wait;" ::: "memory");
            unsigned int live_tiles_3 = num_live * (unsigned int)grid_m;
            unsigned int stage_3 = 0;
            unsigned int work_stage_3 = 0;
            unsigned int valid_5 = (unsigned int)(first < live_tiles_3);
            unsigned int n_tile_4 = first_n;
            unsigned int m_tile_4 = 0;
            unsigned int _phase_sfb_empty = 1;
            unsigned int _phase_work_full_3 = 0;
            #pragma unroll 1
            for (unsigned int _tile_3 = 0; _tile_3 < (unsigned int)grid_m * (unsigned int)max_tiles; _tile_3++) {
                if (valid_5 == 0) {
                    break;
                }
                #pragma unroll 1
                for (int k_4 = 0; k_4 < K_tiles; k_4++) {
                    mbarrier_wait_hint(sfb_empty_addr + (stage_3) * 8, _phase_sfb_empty, 10000000);
                    if (elect_sync()) {
                        tma_3d_gmem2smem(smem_sfb_addr + stage_3 * 128, (&SFB), 0, k_4 * 4, n_tile_4, sfb_full_addr + (stage_3) * 8);
                        mbarrier_arrive_expect_tx(sfb_full_addr + (stage_3) * 8, 128);
                    }
                    stage_3 += 1;
                    if (stage_3 == 6) { stage_3 = 0; _phase_sfb_empty ^= 1; }
                }
                mbarrier_wait_hint(work_full_addr + (work_stage_3) * 8, _phase_work_full_3, 10000000);
                int index_3 = work_stage_3 * 4;
                unsigned int valid_4_1 = work_response[index_3];
                unsigned int m_tile_4_1 = work_response[index_3 + 1];
                unsigned int n_tile_4_1 = work_response[index_3 + 2];
                mbarrier_arrive(work_empty_addr + (work_stage_3) * 8);
                work_stage_3 += 1;
                if (work_stage_3 == 3) { work_stage_3 = 0; _phase_work_full_3 ^= 1; }
                valid_5 = valid_4_1;
                m_tile_4 = m_tile_4_1;
                n_tile_4 = n_tile_4_1;
            }
            asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
            #pragma unroll 1
            for (int __3 = 0; __3 < 6; __3++) {
                mbarrier_wait(sfb_empty_addr + (stage_3) * 8, _phase_sfb_empty);
                stage_3 += 1;
                if (stage_3 == 6) { stage_3 = 0; _phase_sfb_empty ^= 1; }
            }
        }
    }
    // ---- Role: copy_sfa ----
    if (warp == 12) {
        { // copy_sfa_main
            unsigned int live_tiles_4 = num_live * (unsigned int)grid_m;
            unsigned int stage_4 = 0;
            unsigned int work_stage_4 = 0;
            unsigned int valid_6 = (unsigned int)(first < live_tiles_4);
            unsigned int m_tile_5 = 0;
            unsigned int n_tile_5 = 0;
            unsigned int _phase_sfa_full = 0;
            unsigned int _phase_tmem_sfa_empty = 1;
            unsigned int _phase_work_full_4 = 0;
            #pragma unroll 1
            for (unsigned int _tile_4 = 0; _tile_4 < (unsigned int)grid_m * (unsigned int)max_tiles; _tile_4++) {
                if (valid_6 == 0) {
                    break;
                }
                #pragma unroll 1
                for (int _k = 0; _k < K_tiles; _k++) {
                    mbarrier_wait_hint(sfa_full_addr + (stage_4) * 8, _phase_sfa_full, 10000000);
                    mbarrier_wait_hint(tmem_sfa_empty_addr + (stage_4) * 8, _phase_tmem_sfa_empty, 10000000);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    uint32_t _elected_0 = elect_sync();
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                    #endif
                    {
                        uint64_t _tcgen05_cp_desc_1 = ((((uint64_t)(smem_sfa_addr + stage_4 * 2048)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                        asm volatile(
                            "{\n\t.reg .pred leader;\n\tsetp.ne.b32 leader, %2, 0;\n\t"
                            "@leader tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;\n\t}\n"
                            :: "r"((uint32_t)((unsigned int)tmem_sfa + stage_4 * 16)), "l"(_tcgen05_cp_desc_1), "r"(_elected_0)
                            : "memory");
                    }
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                    #endif
                    {
                        uint64_t _tcgen05_cp_desc_2 = ((((uint64_t)(smem_sfa_addr + stage_4 * 2048 + 512)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                        asm volatile(
                            "{\n\t.reg .pred leader;\n\tsetp.ne.b32 leader, %2, 0;\n\t"
                            "@leader tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;\n\t}\n"
                            :: "r"((uint32_t)((unsigned int)tmem_sfa + (stage_4 * 16 + 4))), "l"(_tcgen05_cp_desc_2), "r"(_elected_0)
                            : "memory");
                    }
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                    #endif
                    {
                        uint64_t _tcgen05_cp_desc_3 = ((((uint64_t)(smem_sfa_addr + stage_4 * 2048 + 1024)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                        asm volatile(
                            "{\n\t.reg .pred leader;\n\tsetp.ne.b32 leader, %2, 0;\n\t"
                            "@leader tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;\n\t}\n"
                            :: "r"((uint32_t)((unsigned int)tmem_sfa + (stage_4 * 16 + 8))), "l"(_tcgen05_cp_desc_3), "r"(_elected_0)
                            : "memory");
                    }
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                    #endif
                    {
                        uint64_t _tcgen05_cp_desc_4 = ((((uint64_t)(smem_sfa_addr + stage_4 * 2048 + 1536)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                        asm volatile(
                            "{\n\t.reg .pred leader;\n\tsetp.ne.b32 leader, %2, 0;\n\t"
                            "@leader tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;\n\t}\n"
                            :: "r"((uint32_t)((unsigned int)tmem_sfa + (stage_4 * 16 + 12))), "l"(_tcgen05_cp_desc_4), "r"(_elected_0)
                            : "memory");
                    }
                    elect_commit2(tmem_sfa_full_addr + (stage_4) * 8, sfa_empty_addr + (stage_4) * 8);
                    stage_4 += 1;
                    if (stage_4 == 6) { stage_4 = 0; _phase_sfa_full ^= 1; _phase_tmem_sfa_empty ^= 1; }
                }
                mbarrier_wait_hint(work_full_addr + (work_stage_4) * 8, _phase_work_full_4, 10000000);
                int index_4 = work_stage_4 * 4;
                unsigned int valid_0 = work_response[index_4];
                unsigned int m_tile_1_1 = work_response[index_4 + 1];
                unsigned int n_tile_2_1 = work_response[index_4 + 2];
                asm volatile("tcgen05.fence::before_thread_sync;");
                mbarrier_arrive(work_empty_addr + (work_stage_4) * 8);
                work_stage_4 += 1;
                if (work_stage_4 == 3) { work_stage_4 = 0; _phase_work_full_4 ^= 1; }
                valid_6 = valid_0;
                m_tile_5 = m_tile_1_1;
                n_tile_5 = n_tile_2_1;
            }
            #pragma unroll 1
            for (int __4 = 0; __4 < 6; __4++) {
                mbarrier_wait(tmem_sfa_empty_addr + (stage_4) * 8, _phase_tmem_sfa_empty);
                stage_4 += 1;
                if (stage_4 == 6) { stage_4 = 0; _phase_sfa_full ^= 1; _phase_tmem_sfa_empty ^= 1; }
            }
        }
    }
    // ---- Role: copy_sfb ----
    if (warp >= 4 && warp <= 7) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 72;");
        { // copy_sfb_main
            unsigned int live_tiles_5 = num_live * (unsigned int)grid_m;
            unsigned int stage_5 = 0;
            unsigned int work_stage_5 = 0;
            unsigned int valid_7 = (unsigned int)(first < live_tiles_5);
            unsigned int m_tile_6 = 0;
            unsigned int n_tile_6 = 0;
            const int lane_0 = lane;
            const int tmem_lane_base = warp % 4 * 32;
            unsigned int word[4];
            unsigned int _phase_sfb_full = 0;
            unsigned int _phase_tmem_sfb_empty = 1;
            unsigned int _phase_work_full_5 = 0;
            #pragma unroll 1
            for (unsigned int _tile_5 = 0; _tile_5 < (unsigned int)grid_m * (unsigned int)max_tiles; _tile_5++) {
                if (valid_7 == 0) {
                    break;
                }
                #pragma unroll 1
                for (int _k_1 = 0; _k_1 < K_tiles; _k_1++) {
                    mbarrier_wait_hint(sfb_full_addr + (stage_5) * 8, _phase_sfb_full, 10000000);
                    mbarrier_wait_hint(tmem_sfb_empty_addr + (stage_5) * 8, _phase_tmem_sfb_empty, 10000000);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    word[0] = 0;
                    if (lane_0 < 8) {
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&word[0])) : "r"(smem_sfb_addr + stage_5 * 128 + (unsigned int)(lane_0 * 4)));
                    }
                    word[1] = 0;
                    if (lane_0 < 8) {
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&word[1])) : "r"(smem_sfb_addr + stage_5 * 128 + (unsigned int)(32 + lane_0 * 4)));
                    }
                    word[2] = 0;
                    if (lane_0 < 8) {
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&word[2])) : "r"(smem_sfb_addr + stage_5 * 128 + (unsigned int)(64 + lane_0 * 4)));
                    }
                    word[3] = 0;
                    if (lane_0 < 8) {
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&word[3])) : "r"(smem_sfb_addr + stage_5 * 128 + (unsigned int)(96 + lane_0 * 4)));
                    }
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x1.b32"
                        " [%0], {%1};"
                        :: "r"((unsigned int)tmem_sfb + (unsigned int)(tmem_lane_base << 16) + stage_5 * 8), "r"(word[0]));
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x1.b32"
                        " [%0], {%1};"
                        :: "r"((unsigned int)tmem_sfb + (unsigned int)(tmem_lane_base << 16) + stage_5 * 8 + 2), "r"((word + 1)[0]));
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x1.b32"
                        " [%0], {%1};"
                        :: "r"((unsigned int)tmem_sfb + (unsigned int)(tmem_lane_base << 16) + stage_5 * 8 + 4), "r"((word + 2)[0]));
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x1.b32"
                        " [%0], {%1};"
                        :: "r"((unsigned int)tmem_sfb + (unsigned int)(tmem_lane_base << 16) + stage_5 * 8 + 6), "r"((word + 3)[0]));
                    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    mbarrier_arrive(tmem_sfb_full_addr + (stage_5) * 8);
                    asm volatile("barrier.sync 2, 128;" ::: "memory");
                    if (warp == 4) {
                        if (elect_sync()) {
                            mbarrier_arrive(sfb_empty_addr + (stage_5) * 8);
                        }
                    }
                    stage_5 += 1;
                    if (stage_5 == 6) { stage_5 = 0; _phase_sfb_full ^= 1; _phase_tmem_sfb_empty ^= 1; }
                }
                mbarrier_wait_hint(work_full_addr + (work_stage_5) * 8, _phase_work_full_5, 10000000);
                int index_5 = work_stage_5 * 4;
                unsigned int valid_6_1 = work_response[index_5];
                unsigned int m_tile_5_1 = work_response[index_5 + 1];
                unsigned int n_tile_5_1 = work_response[index_5 + 2];
                mbarrier_arrive(work_empty_addr + (work_stage_5) * 8);
                work_stage_5 += 1;
                if (work_stage_5 == 3) { work_stage_5 = 0; _phase_work_full_5 ^= 1; }
                valid_7 = valid_6_1;
                m_tile_6 = m_tile_5_1;
                n_tile_6 = n_tile_5_1;
            }
            #pragma unroll 1
            for (int __5 = 0; __5 < 6; __5++) {
                mbarrier_wait(tmem_sfb_empty_addr + (stage_5) * 8, _phase_tmem_sfb_empty);
                stage_5 += 1;
                if (stage_5 == 6) { stage_5 = 0; _phase_sfb_full ^= 1; _phase_tmem_sfb_empty ^= 1; }
            }
        }
    }
    // ---- Role: mma ----
    if (warp == 13) {
        { // mma_main
            unsigned int live_tiles_6 = num_live * (unsigned int)grid_m;
            unsigned int k_stage = 0;
            unsigned int acc_stage = 0;
            unsigned int work_stage_6 = 0;
            unsigned int a_token = 0;
            unsigned int b_token = 0;
            unsigned int sfa_token = 0;
            unsigned int sfb_token = 0;
            unsigned int valid_8 = (unsigned int)(first < live_tiles_6);
            unsigned int m_tile_7 = 0;
            unsigned int n_tile_7 = 0;
            unsigned int _phase_mma_empty = 1;
            unsigned int _phase_a_full = 0;
            unsigned int _phase_work_full_6 = 0;
            #pragma unroll 1
            for (unsigned int _tile_6 = 0; _tile_6 < (unsigned int)grid_m * (unsigned int)max_tiles; _tile_6++) {
                if (valid_8 == 0) {
                    break;
                }
                mbarrier_wait_hint(mma_empty_addr + (acc_stage) * 8, _phase_mma_empty, 10000000);
                #pragma unroll 1
                for (int k_5 = 0; k_5 < K_tiles; k_5++) {
                    uint32_t _mbar_token_0 = mbarrier_try_wait(a_full_addr + (k_stage) * 8, _phase_a_full);
                    a_token = _mbar_token_0;
                    uint32_t _mbar_token_1 = mbarrier_try_wait(b_full_addr + (k_stage) * 8, _phase_a_full);
                    b_token = _mbar_token_1;
                    uint32_t _mbar_token_2 = mbarrier_try_wait(tmem_sfa_full_addr + (k_stage) * 8, _phase_a_full);
                    sfa_token = _mbar_token_2;
                    uint32_t _mbar_token_3 = mbarrier_try_wait(tmem_sfb_full_addr + (k_stage) * 8, _phase_a_full);
                    sfb_token = _mbar_token_3;
                    mbarrier_wait_token_hint(a_full_addr + (k_stage) * 8, _phase_a_full, a_token, 10000000);
                    mbarrier_wait_token_hint(b_full_addr + (k_stage) * 8, _phase_a_full, b_token, 10000000);
                    mbarrier_wait_token_hint(tmem_sfa_full_addr + (k_stage) * 8, _phase_a_full, sfa_token, 10000000);
                    mbarrier_wait_token_hint(tmem_sfb_full_addr + (k_stage) * 8, _phase_a_full, sfb_token, 10000000);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int _mma_a_lo_0 = make_warp_uniform((((smem_a_addr) >> 4) & 0x3FFF) + (k_stage) * 1024);
                    int _mma_b_lo_0 = make_warp_uniform((((smem_b_addr) >> 4) & 0x3FFF) + (k_stage) * 64);
                    if (elect_sync()) {
                        {
                            uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_0) | ((uint64_t)0x40004040 << 32);
                            uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_0) | ((uint64_t)0x40004040 << 32);

                            tcgen05_mma_mxf4nvf4_bs((tmem_accum + (acc_stage * 8)), a_desc + 0, b_desc + 0,
                                0x8020480U, (unsigned int)tmem_sfa + k_stage * 16 + 0, (unsigned int)tmem_sfb + k_stage * 8 + 0, ((((k_5 == 0) ? 1 : 0)) ? 0 : 1));
                        }
                    }
                    int _mma_a_lo_1 = make_warp_uniform((((smem_a_addr + 32) >> 4) & 0x3FFF) + (k_stage) * 1024);
                    int _mma_b_lo_1 = make_warp_uniform((((smem_b_addr + 32) >> 4) & 0x3FFF) + (k_stage) * 64);
                    if (elect_sync()) {
                        {
                            uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_1) | ((uint64_t)0x40004040 << 32);
                            uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_1) | ((uint64_t)0x40004040 << 32);

                            tcgen05_mma_mxf4nvf4_bs((tmem_accum + (acc_stage * 8)), a_desc + 0, b_desc + 0,
                                0x8020480U, (unsigned int)tmem_sfa + (k_stage * 16 + 4) + 0, (unsigned int)tmem_sfb + (k_stage * 8 + 2) + 0, 1);
                        }
                    }
                    int _mma_a_lo_2 = make_warp_uniform((((smem_a_addr + 64) >> 4) & 0x3FFF) + (k_stage) * 1024);
                    int _mma_b_lo_2 = make_warp_uniform((((smem_b_addr + 64) >> 4) & 0x3FFF) + (k_stage) * 64);
                    if (elect_sync()) {
                        {
                            uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_2) | ((uint64_t)0x40004040 << 32);
                            uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_2) | ((uint64_t)0x40004040 << 32);

                            tcgen05_mma_mxf4nvf4_bs((tmem_accum + (acc_stage * 8)), a_desc + 0, b_desc + 0,
                                0x8020480U, (unsigned int)tmem_sfa + (k_stage * 16 + 8) + 0, (unsigned int)tmem_sfb + (k_stage * 8 + 4) + 0, 1);
                        }
                    }
                    int _mma_a_lo_3 = make_warp_uniform((((smem_a_addr + 96) >> 4) & 0x3FFF) + (k_stage) * 1024);
                    int _mma_b_lo_3 = make_warp_uniform((((smem_b_addr + 96) >> 4) & 0x3FFF) + (k_stage) * 64);
                    if (elect_sync()) {
                        {
                            uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_3) | ((uint64_t)0x40004040 << 32);
                            uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_3) | ((uint64_t)0x40004040 << 32);

                            tcgen05_mma_mxf4nvf4_bs((tmem_accum + (acc_stage * 8)), a_desc + 0, b_desc + 0,
                                0x8020480U, (unsigned int)tmem_sfa + (k_stage * 16 + 12) + 0, (unsigned int)tmem_sfb + (k_stage * 8 + 6) + 0, 1);
                        }
                    }
                    uint32_t _elected_0 = elect_sync();
                    tcgen05_commit_leader(a_empty_addr + (k_stage) * 8, _elected_0);
                    tcgen05_commit_leader(b_empty_addr + (k_stage) * 8, _elected_0);
                    tcgen05_commit_leader(tmem_sfa_empty_addr + (k_stage) * 8, _elected_0);
                    tcgen05_commit_leader(tmem_sfb_empty_addr + (k_stage) * 8, _elected_0);
                    k_stage += 1;
                    if (k_stage == 6) { k_stage = 0; _phase_a_full ^= 1; }
                }
                elect_commit(mma_full_addr + (acc_stage) * 8);
                acc_stage += 1;
                if (acc_stage == 2) { acc_stage = 0; _phase_mma_empty ^= 1; }
                mbarrier_wait_hint(work_full_addr + (work_stage_6) * 8, _phase_work_full_6, 10000000);
                int index_6 = work_stage_6 * 4;
                unsigned int valid_0_1 = work_response[index_6];
                unsigned int m_tile_1_2 = work_response[index_6 + 1];
                unsigned int n_tile_2_2 = work_response[index_6 + 2];
                asm volatile("tcgen05.fence::before_thread_sync;");
                mbarrier_arrive(work_empty_addr + (work_stage_6) * 8);
                work_stage_6 += 1;
                if (work_stage_6 == 3) { work_stage_6 = 0; _phase_work_full_6 ^= 1; }
                valid_8 = valid_0_1;
                m_tile_7 = m_tile_1_2;
                n_tile_7 = n_tile_2_2;
            }
        }
    }
    // ---- Role: epilogue ----
    if (warp <= 3) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 160;");
        { // epilogue_main
            asm volatile("griddepcontrol.wait;" ::: "memory");
            unsigned int live_tiles_7 = num_live * (unsigned int)grid_m;
            unsigned int acc_stage_1 = 0;
            unsigned int work_stage_7 = 0;
            unsigned int valid_9 = (unsigned int)(first < live_tiles_7);
            unsigned int m_tile_8 = first_m;
            unsigned int n_tile_8 = first_n;
            int row = warp * 32;
            int feature = row + lane / 4 * 4;
            int token = lane % 4 * 2;
            float values[4];
            unsigned int packed[2];
            unsigned long long word_1 = 0;
            unsigned int _phase_mma_full = 0;
            unsigned int _phase_work_full_7 = 0;
            #pragma unroll 1
            for (unsigned int _tile_7 = 0; _tile_7 < (unsigned int)grid_m * (unsigned int)max_tiles; _tile_7++) {
                if (valid_9 == 0) {
                    break;
                }
                int vec_load_3[1];
                {
                    uint32_t _scalar_bits_0;
                    asm volatile("ld.global.b32 %0, [%1];"
                        : "=r"(_scalar_bits_0) : "l"((const void*)(tile_expert + (n_tile_8))) : "memory");
                    vec_load_3[0] = (int32_t)_scalar_bits_0;
                }
                int expert_1 = vec_load_3[0];
                int vec_load_4[1];
                {
                    uint32_t _scalar_bits_1;
                    asm volatile("ld.global.b32 %0, [%1];"
                        : "=r"(_scalar_bits_1) : "l"((const void*)(tile_mn_limit + (n_tile_8))) : "memory");
                    vec_load_4[0] = (int32_t)_scalar_bits_1;
                }
                int mn_limit = vec_load_4[0];
                float sc = scale_c[expert_1];
                mbarrier_wait_hint(mma_full_addr + (acc_stage_1) * 8, _phase_mma_full, 10000000);
                asm volatile("tcgen05.fence::after_thread_sync;");
                float _tmem_load_0[4];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x1.b32"
                    " {%0, %1, %2, %3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[3]))
                    : "r"((unsigned int)tmem_accum + (unsigned int)(row << 16) + acc_stage_1 * 8));
                float _tmem_load_1[4];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x1.b32"
                    " {%0, %1, %2, %3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[3]))
                    : "r"((unsigned int)tmem_accum + (unsigned int)(row + 16 << 16) + acc_stage_1 * 8));
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                asm volatile("cp.async.bulk.wait_group.read 0;");
                asm volatile("tcgen05.fence::before_thread_sync;");
                asm volatile("barrier.sync 1, 128;" ::: "memory");
                float2 sc2 = make_float2(sc, sc);
                float2 _f32x2_mul_a_2 = make_float2(_tmem_load_0[0], _tmem_load_0[2]);
                float2 p0_1_0;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&p0_1_0) : "l"(*(const unsigned long long*)&_f32x2_mul_a_2), "l"(*(const unsigned long long*)&sc2));
                float2 _f32x2_mul_a_3 = make_float2(_tmem_load_1[0], _tmem_load_1[2]);
                float2 p1_2_0;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&p1_2_0) : "l"(*(const unsigned long long*)&_f32x2_mul_a_3), "l"(*(const unsigned long long*)&sc2));
                values[0] = p0_1_0.x;
                values[1] = p0_1_0.y;
                values[2] = p1_2_0.x;
                values[3] = p1_2_0.y;
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(values[_lp*2 + 0], values[_lp*2+1 + 0]));
                    packed[_lp] = *(uint32_t*)&_bf2;
                }
                word_1 = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                int tok_1_0 = token;
                int half_1_0 = feature / 64;
                int chunk_1_0 = feature % 64 / 8 ^ tok_1_0 % 8;
                epi_staging[half_1_0 * 128 + tok_1_0 * 16 + chunk_1_0 * 2 + feature % 8 / 4] = word_1;
                float2 _f32x2_mul_a_4 = make_float2(_tmem_load_0[1], _tmem_load_0[3]);
                float2 p0_1_1;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&p0_1_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_4), "l"(*(const unsigned long long*)&sc2));
                float2 _f32x2_mul_a_5 = make_float2(_tmem_load_1[1], _tmem_load_1[3]);
                float2 p1_2_1;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&p1_2_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_5), "l"(*(const unsigned long long*)&sc2));
                values[0] = p0_1_1.x;
                values[1] = p0_1_1.y;
                values[2] = p1_2_1.x;
                values[3] = p1_2_1.y;
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(values[_lp*2 + 0], values[_lp*2+1 + 0]));
                    packed[_lp] = *(uint32_t*)&_bf2;
                }
                word_1 = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                int tok_1_1 = token + 1;
                int half_1_1 = feature / 64;
                int chunk_1_1 = feature % 64 / 8 ^ tok_1_1 % 8;
                epi_staging[half_1_1 * 128 + tok_1_1 * 16 + chunk_1_1 * 2 + feature % 8 / 4] = word_1;
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("barrier.sync 1, 128;" ::: "memory");
                if (warp == 0) {
                    if (elect_sync()) {
                        int padding_rows = (8 - mn_limit % 8) % 8;
                        int token_base = n_tile_8 * 8;
                        int off_m = m_tile_8 * 128;
                        tma_store_4d((&C), off_m, padding_rows, 1073741824, token_base - padding_rows + 1073741824, epi_staging_addr);
                        tma_store_4d((&C), off_m + 64, padding_rows, 1073741824, token_base - padding_rows + 1073741824, epi_staging_addr + 1024);
                        asm volatile("cp.async.bulk.commit_group;");
                    }
                }
                asm volatile("barrier.sync 1, 128;" ::: "memory");
                asm volatile("tcgen05.fence::before_thread_sync;");
                mbarrier_arrive(mma_empty_addr + (acc_stage_1) * 8);
                acc_stage_1 += 1;
                if (acc_stage_1 == 2) { acc_stage_1 = 0; _phase_mma_full ^= 1; }
                mbarrier_wait_hint(work_full_addr + (work_stage_7) * 8, _phase_work_full_7, 10000000);
                int index_7 = work_stage_7 * 4;
                unsigned int valid_7_1 = work_response[index_7];
                unsigned int m_tile_7_1 = work_response[index_7 + 1];
                unsigned int n_tile_6_1 = work_response[index_7 + 2];
                mbarrier_arrive(work_empty_addr + (work_stage_7) * 8);
                work_stage_7 += 1;
                if (work_stage_7 == 3) { work_stage_7 = 0; _phase_work_full_7 ^= 1; }
                valid_9 = valid_7_1;
                m_tile_8 = m_tile_7_1;
                n_tile_8 = n_tile_6_1;
            }
            asm volatile("cp.async.bulk.wait_group.read 0;");
        }
    }
    // ---- Role: work_id ----
    if (warp == 14) {
        { // work_id_main
            unsigned int live_tiles_8 = num_live * (unsigned int)grid_m;
            unsigned int work_stage_8 = 0;
            unsigned int throttle_stage_1 = 0;
            unsigned int valid_10 = (unsigned int)(first < live_tiles_8);
            unsigned int m_tile_9 = 0;
            unsigned int n_tile_9 = 0;
            unsigned int static_next = first;
            unsigned int _phase_throttle_full = 0;
            unsigned int _phase_work_empty = 1;
            unsigned int _phase_work_full_8 = 0;
            #pragma unroll 1
            for (unsigned int _tile_8 = 0; _tile_8 < (unsigned int)grid_m * (unsigned int)max_tiles; _tile_8++) {
                if (valid_10 == 0) {
                    break;
                }
                mbarrier_wait_hint(throttle_full_addr + (throttle_stage_1) * 8, _phase_throttle_full, 10000000);
                mbarrier_arrive(throttle_empty_addr + (throttle_stage_1) * 8);
                throttle_stage_1 += 1;
                if (throttle_stage_1 == 3) { throttle_stage_1 = 0; _phase_throttle_full ^= 1; }
                mbarrier_wait_hint(work_empty_addr + (work_stage_8) * 8, _phase_work_empty, 10000000);
                static_next = static_next + (unsigned int)num_bids;
                if (elect_sync()) {
                    unsigned int claim = static_next;
                    if (static_next >= (unsigned int)num_bids * 3) {
                        int _atomic_fetch_add_0 = atomicAdd(work_counter, 1);
                        claim = (unsigned int)_atomic_fetch_add_0 + (unsigned int)num_bids * 3;
                    }
                    int index_8 = work_stage_8 * 4;
                    work_response[index_8] = (unsigned int)(claim < live_tiles_8);
                    work_response[index_8 + 1] = claim % (unsigned int)grid_m;
                    work_response[index_8 + 2] = claim / (unsigned int)grid_m;
                    work_response[index_8 + 3] = 0;
                    mbarrier_arrive(work_full_addr + (work_stage_8) * 8);
                }
                __syncwarp();
                mbarrier_wait_hint(work_full_addr + (work_stage_8) * 8, _phase_work_full_8, 10000000);
                int index_9 = work_stage_8 * 4;
                unsigned int valid_8_1 = work_response[index_9];
                unsigned int m_tile_8_1 = work_response[index_9 + 1];
                unsigned int n_tile_8_1 = work_response[index_9 + 2];
                mbarrier_arrive(work_empty_addr + (work_stage_8) * 8);
                work_stage_8 += 1;
                if (work_stage_8 == 3) { work_stage_8 = 0; _phase_work_empty ^= 1; _phase_work_full_8 ^= 1; }
                valid_10 = valid_8_1;
                m_tile_9 = m_tile_8_1;
                n_tile_9 = n_tile_8_1;
            }
            if (elect_sync()) {
                __threadfence();
                int retired = atomicAdd(work_counter + 1, 1);
                if (retired == num_bids - 1) {
                    __threadfence();
                    work_counter[0] = 0;
                    work_counter[1] = 0;
                }
            }
        }
    }

    // Cleanup
    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
    __syncthreads(); // barrier before TMEM dealloc

    if (warp == 0) {
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(256));
    }
}

} // extern "C"
