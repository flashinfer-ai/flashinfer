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
#define TMEM_NCOLS 80
#define TMEM_ACCUM_OFFSET 0
#define TMEM_TMEM_SFW_OFFSET 64
#define TMEM_TMEM_SFX_OFFSET 72
#define NUM_TMA_PIPE_STAGES 2
#define NUM_MAINLOOP_PIPE_STAGES 1
#define NUM_XB_PIPE_STAGES 2
#define NUM_RED_PIPE_STAGES 1
#define SMEM_SMEM_W_OFF 1024
#define SMEM_SMEM_W_STAGE_BYTES 32768
#define SMEM_SMEM_W_STRIDE 51200
#define SMEM_SMEM_SFW0_OFF 50176
#define SMEM_SMEM_SFW0_STAGE_BYTES 512
#define SMEM_SMEM_SFW0_STRIDE 51200
#define SMEM_SMEM_SFW1_OFF 50688
#define SMEM_SMEM_SFW1_STAGE_BYTES 512
#define SMEM_SMEM_SFW1_STRIDE 51200
#define SMEM_SMEM_X_OFF 33792
#define SMEM_SMEM_X_STAGE_BYTES 16384
#define SMEM_SMEM_X_STRIDE 51200
#define SMEM_SMEM_SFX_ALL_OFF 51200
#define SMEM_SMEM_SFX_ALL_STAGE_BYTES 1024
#define SMEM_SMEM_SFX_ALL_STRIDE 51200
#define SMEM_SMEM_SFX0_OFF 51200
#define SMEM_SMEM_SFX0_STAGE_BYTES 512
#define SMEM_SMEM_SFX0_STRIDE 51200
#define SMEM_SMEM_SFX1_OFF 51712
#define SMEM_SMEM_SFX1_STAGE_BYTES 512
#define SMEM_SMEM_SFX1_STRIDE 51200
#define SMEM_SMEM_XB_OFF 103424
#define SMEM_SMEM_XB_STAGE_BYTES 32768
#define SMEM_SMEM_XB_STRIDE 32768
#define SMEM_SMEM_XB0_OFF 103424
#define SMEM_SMEM_XB0_STAGE_BYTES 16384
#define SMEM_SMEM_XB0_STRIDE 32768
#define SMEM_SMEM_XB1_OFF 119808
#define SMEM_SMEM_XB1_STAGE_BYTES 16384
#define SMEM_SMEM_XB1_STRIDE 32768
#define SMEM_SMEM_EPI_OFF 168960
#define SMEM_SMEM_EPI_STAGE_BYTES 16384
#define SMEM_SMEM_EPI_STRIDE 16384
#define SMEM_SMEM_RED_OFF 168960
#define SMEM_SMEM_RED_STAGE_BYTES 34816
#define SMEM_SMEM_RED_STRIDE 34816
#define SMEM_SMEM_STG_OFF 203776
#define SMEM_SMEM_STG_STAGE_BYTES 26112
#define SMEM_SMEM_STG_STRIDE 26112
#define SMEM_TOTAL 229888
#define THREADS 736

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



__device__ __forceinline__ void tcgen05_mma_mxf8_bs_leader(
    int taddr, uint64_t a_desc, uint64_t b_desc, uint32_t i_desc,
    int sfa_taddr, int sfb_taddr, int enable_input_d, uint32_t leader) {
    asm volatile(
        "{\n\t"
        ".reg .pred leader, p;\n\t"
        "setp.ne.b32 leader, %7, 0;\n\t"
        "setp.ne.b32 p, %6, 0;\n\t"
        "@leader tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale [%0], %1, %2, %3, [%4], [%5], p;\n\t"
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



__device__ __forceinline__ float max_noftz(float a, float b) {
    float c;
    asm("max.f32 %0, %1, %2;" : "=f"(c) : "f"(a), "f"(b));
    return c;
}






__device__ __forceinline__ void mul_f32x2_noftz_inplace(float2* a, float2 b) {
    asm("mul.rn.f32x2 %0, %0, %1;"
        : "+l"(*(unsigned long long*)a) : "l"(*(unsigned long long*)&b));
}


// ex2_emulation_f32x2 defined in softmax_frag_exp2_cast helper (or standalone)




__device__ __forceinline__ uint64_t make_sf_cp_desc_lo_sbo128(int lo) {
    const int SBO = 128;
    return (uint64_t)(uint32_t)lo
         | (desc_encode(SBO) << 32ULL)
         | (1ULL << 46ULL);
}



__device__ __forceinline__ void tcgen05_cp_32x128b_warpx4_leader(
    int taddr, uint64_t s_desc, uint32_t leader) {
    asm volatile(
        "{\n\t"
        ".reg .pred leader;\n\t"
        "setp.ne.b32 leader, %2, 0;\n\t"
        "@leader tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;\n\t"
        "}\n"
        :: "r"(taddr), "l"(s_desc), "r"(leader) : "memory");
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

__global__ __launch_bounds__(736) __cluster_dims__(4,1,1) void
kernel_cake_kimi_k3_fp8_projection_de86c87283901f3d90f1(const __grid_constant__ CUtensorMap W, const __grid_constant__ CUtensorMap X, const __grid_constant__ CUtensorMap SFW, const __grid_constant__ CUtensorMap SFX, __nv_bfloat16* __restrict__ out, float* __restrict__ partials, unsigned int* __restrict__ counters, int M, int n_tiles, int n_valid, int ldo, int num_k_iters, int sf_k_tiles, int split, int tok_per_cta, int total_work, int store_vec, __nv_bfloat16* __restrict__ x, int K, const __grid_constant__ CUtensorMap XB)
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
    #define mma_done_addr (mbar_base + 16)
    #define xb_full_addr (mbar_base + 32)
    #define xb_empty_addr (mbar_base + 48)
    #define xb_full1_addr (mbar_base + 64)
    #define xb_empty1_addr (mbar_base + 80)
    #define mainloop_done_addr (mbar_base + 96)
    #define epilogue_done_addr (mbar_base + 104)
    #define red_full_addr (mbar_base + 112)
    #define peers_free_addr (mbar_base + 120)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    const unsigned int clusters_x = gridDim.x / 4;
    const unsigned int cluster_id = ((blockIdx.z * gridDim.y + blockIdx.y) * clusters_x) + blockIdx.x / 4;
    const unsigned int num_clusters = clusters_x * gridDim.y * gridDim.z;

    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));

    // Kernel setup ops
    uint8_t* smem_w = reinterpret_cast<uint8_t*>(smem_raw + 1024);
    const int smem_w_addr = smem + 1024;
    uint8_t* smem_sfw0 = reinterpret_cast<uint8_t*>(smem_raw + 50176);
    const int smem_sfw0_addr = smem + 50176;
    uint8_t* smem_sfw1 = reinterpret_cast<uint8_t*>(smem_raw + 50688);
    const int smem_sfw1_addr = smem + 50688;
    uint8_t* smem_x = reinterpret_cast<uint8_t*>(smem_raw + 33792);
    const int smem_x_addr = smem + 33792;
    uint8_t* smem_sfx_all = reinterpret_cast<uint8_t*>(smem_raw + 51200);
    const int smem_sfx_all_addr = smem + 51200;
    uint8_t* smem_sfx0 = reinterpret_cast<uint8_t*>(smem_raw + 51200);
    const int smem_sfx0_addr = smem + 51200;
    uint8_t* smem_sfx1 = reinterpret_cast<uint8_t*>(smem_raw + 51712);
    const int smem_sfx1_addr = smem + 51712;
    uint8_t* smem_xb = reinterpret_cast<uint8_t*>(smem_raw + 103424);
    const int smem_xb_addr = smem + 103424;
    uint8_t* smem_xb0 = reinterpret_cast<uint8_t*>(smem_raw + 103424);
    const int smem_xb0_addr = smem + 103424;
    uint8_t* smem_xb1 = reinterpret_cast<uint8_t*>(smem_raw + 119808);
    const int smem_xb1_addr = smem + 119808;
    uint8_t* smem_epi = reinterpret_cast<uint8_t*>(smem_raw + 168960);
    const int smem_epi_addr = smem + 168960;
    float* smem_red = reinterpret_cast<float*>(smem_raw + 168960);
    const int smem_red_addr = smem + 168960;
    float* smem_stg = reinterpret_cast<float*>(smem_raw + 203776);
    const int smem_stg_addr = smem + 203776;

    // Mbarrier init (10 pipeline groups, 0 ordered-sequence groups, 16 barriers)
    // Mbarriers at smem_raw[0..128)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'tma_pipe' ---
            // tma_full: 2 barriers, init_count=17
            mbarrier_init(smem + 0, 17);
            mbarrier_init(smem + 8, 17);
            // mma_done: 2 barriers, init_count=1
            mbarrier_init(smem + 16, 1);
            mbarrier_init(smem + 24, 1);
            // --- pipeline 'xb_pipe' ---
            // xb_full: 2 barriers, init_count=1
            mbarrier_init(smem + 32, 1);
            mbarrier_init(smem + 40, 1);
            // xb_empty: 2 barriers, init_count=16
            mbarrier_init(smem + 48, 16);
            mbarrier_init(smem + 56, 16);
            // xb_full1: 2 barriers, init_count=1
            mbarrier_init(smem + 64, 1);
            mbarrier_init(smem + 72, 1);
            // xb_empty1: 2 barriers, init_count=16
            mbarrier_init(smem + 80, 16);
            mbarrier_init(smem + 88, 16);
            // --- pipeline 'mainloop_pipe' ---
            // mainloop_done: 1 barriers, init_count=1
            mbarrier_init(smem + 96, 1);
            // epilogue_done: 1 barriers, init_count=4
            mbarrier_init(smem + 104, 4);
            // --- pipeline 'red_pipe' ---
            // red_full: 1 barriers, init_count=1
            mbarrier_init(smem + 112, 1);
            // peers_free: 1 barriers, init_count=3
            mbarrier_init(smem + 120, 3);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    // TMEM alloc (128 columns, 80 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 128);
    if (warp == 0) {
        int _tmem_hold = smem + 128;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(128) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_accum = taddr;
    const int tmem_tmem_sfw = taddr + 64;
    const int tmem_tmem_sfx = taddr + 72;

    // ---- Role: load ----
    if (warp == 0) {
        { // load_main
            unsigned int _phase_mma_done = 1;
            if (elect_sync()) {
                int w0 = bid;
                int ws = num_bids;
                int n_items = (total_work - bid + num_bids - 1) / num_bids;
                unsigned int load_stage = 0;
                #pragma unroll 1
                for (int it = 0; it < n_items; it++) {
                    int work = w0 + it * ws;
                    int tile = work / split;
                    int rank = work - tile * split;
                    int k_begin = num_k_iters * rank / split;
                    int k_end = num_k_iters * (rank + 1) / split;
                    int k_count = k_end - k_begin;
                    int n_tile = tile % n_tiles;
                    int m_tile = tile / n_tiles;
                    int w_tile0 = n_tile * (2 * num_k_iters);
                    int x_row = m_tile * 64;
                    int sfw_unit0 = n_tile / 2 * sf_k_tiles * 2 + n_tile % 2;
                    int sfx_unit0 = m_tile * sf_k_tiles;
                    #pragma unroll 1
                    for (int i = 0; i < k_count; i++) {
                        int iter_k = k_begin + i;
                        mbarrier_wait(mma_done_addr + (load_stage) * 8, _phase_mma_done);
                        int k_group = iter_k * 2;
                        tma_3d_gmem2smem(smem_w_addr + load_stage * 51200, (&W), 0, 0, w_tile0 + k_group, tma_full_addr + (load_stage) * 8);
                        tma_3d_gmem2smem(smem_sfw0_addr + load_stage * 51200, (&SFW), 0, 0, sfw_unit0 + k_group * 2, tma_full_addr + (load_stage) * 8);
                        tma_3d_gmem2smem(smem_sfw1_addr + load_stage * 51200, (&SFW), 0, 0, sfw_unit0 + k_group * 2 + 2, tma_full_addr + (load_stage) * 8);
                        mbarrier_arrive_expect_tx(tma_full_addr + (load_stage) * 8, 33792);
                        load_stage += 1;
                        if (load_stage == 2) { load_stage = 0; _phase_mma_done ^= 1; }
                    }
                }
                asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
            }
        }
    }
    // ---- Role: mma ----
    if (warp == 1) {
        { // mma_main
            int w0_m = bid;
            int ws_m = num_bids;
            int n_items_m = (total_work - bid + num_bids - 1) / num_bids;
            unsigned int mma_stage = 0;
            unsigned int xq_m = 0;
            unsigned int res_stage_m = 0;
            int m_prev_m = -1;
            unsigned int _phase_epilogue_done = 1;
            unsigned int _phase_tma_full = 0;
            #pragma unroll 1
            for (int it_m = 0; it_m < n_items_m; it_m++) {
                int work_m = w0_m + it_m * ws_m;
                int tile_1 = work_m / split;
                int rank_1 = work_m - tile_1 * split;
                int k_begin_1 = num_k_iters * rank_1 / split;
                int k_end_1 = num_k_iters * (rank_1 + 1) / split;
                int k_count_1 = k_end_1 - k_begin_1;
                mbarrier_wait(epilogue_done_addr, _phase_epilogue_done);
                #pragma unroll 1
                for (int i_1 = 0; i_1 < k_count_1; i_1++) {
                    mbarrier_wait(tma_full_addr + (mma_stage) * 8, _phase_tma_full);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int init_flag = ((i_1 == 0) ? 1 : 0);
                    uint32_t _elected_0 = elect_sync();
                    tcgen05_cp_32x128b_warpx4_leader(tmem_tmem_sfw, make_sf_cp_desc_lo_sbo128((((smem_sfw0_addr) >> 4) + (mma_stage) * 3200)), _elected_0);
                    tcgen05_cp_32x128b_warpx4_leader(tmem_tmem_sfx, make_sf_cp_desc_lo_sbo128((((smem_sfx0_addr) >> 4) + (mma_stage) * 3200)), _elected_0);
                    tcgen05_cp_32x128b_warpx4_leader(tmem_tmem_sfw + 4, make_sf_cp_desc_lo_sbo128((((smem_sfw1_addr) >> 4) + (mma_stage) * 3200)), _elected_0);
                    tcgen05_cp_32x128b_warpx4_leader(tmem_tmem_sfx + 4, make_sf_cp_desc_lo_sbo128((((smem_sfx1_addr) >> 4) + (mma_stage) * 3200)), _elected_0);
                    int _smem_desc_lo_1 = (((smem_w_addr) >> 4) & 0x3FFF) + (mma_stage) * 3200;
                    int _smem_desc_lo_2 = (((smem_x_addr) >> 4) & 0x3FFF) + (mma_stage) * 3200;
                    {
                        uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_1) | ((uint64_t)0x40004040 << 32);
                        uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_2) | ((uint64_t)0x40004040 << 32);

                        tcgen05_mma_mxf8_bs_leader((tmem_accum + (0 * 64)), a_desc + 0, b_desc + 0,
                            0x8900000U, tmem_tmem_sfw, tmem_tmem_sfx, ((init_flag) ? 0 : 1), _elected_0);
                        tcgen05_mma_mxf8_bs_leader((tmem_accum + (0 * 64)), a_desc + 2, b_desc + 2,
                            0x28900010U, tmem_tmem_sfw, tmem_tmem_sfx, 1, _elected_0);
                        tcgen05_mma_mxf8_bs_leader((tmem_accum + (0 * 64)), a_desc + 4, b_desc + 4,
                            0x48900020U, tmem_tmem_sfw, tmem_tmem_sfx, 1, _elected_0);
                        tcgen05_mma_mxf8_bs_leader((tmem_accum + (0 * 64)), a_desc + 6, b_desc + 6,
                            0x68900030U, tmem_tmem_sfw, tmem_tmem_sfx, 1, _elected_0);
                    }
                    int init2 = 0;
                    int _smem_desc_lo_3 = (((smem_w_addr + 16384) >> 4) & 0x3FFF) + (mma_stage) * 3200;
                    int _smem_desc_lo_4 = (((smem_x_addr + 8192) >> 4) & 0x3FFF) + (mma_stage) * 3200;
                    {
                        uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_3) | ((uint64_t)0x40004040 << 32);
                        uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_4) | ((uint64_t)0x40004040 << 32);

                        tcgen05_mma_mxf8_bs_leader((tmem_accum + (0 * 64)), a_desc + 0, b_desc + 0,
                            0x8900000U, tmem_tmem_sfw + 4, tmem_tmem_sfx + 4, ((init2) ? 0 : 1), _elected_0);
                        tcgen05_mma_mxf8_bs_leader((tmem_accum + (0 * 64)), a_desc + 2, b_desc + 2,
                            0x28900010U, tmem_tmem_sfw + 4, tmem_tmem_sfx + 4, 1, _elected_0);
                        tcgen05_mma_mxf8_bs_leader((tmem_accum + (0 * 64)), a_desc + 4, b_desc + 4,
                            0x48900020U, tmem_tmem_sfw + 4, tmem_tmem_sfx + 4, 1, _elected_0);
                        tcgen05_mma_mxf8_bs_leader((tmem_accum + (0 * 64)), a_desc + 6, b_desc + 6,
                            0x68900030U, tmem_tmem_sfw + 4, tmem_tmem_sfx + 4, 1, _elected_0);
                    }
                    elect_commit(mma_done_addr + (mma_stage) * 8);
                    mma_stage += 1;
                    if (mma_stage == 2) { mma_stage = 0; _phase_tma_full ^= 1; }
                }
                elect_commit(mainloop_done_addr);
                _phase_epilogue_done ^= 1;
            }
            mbarrier_wait(epilogue_done_addr, _phase_epilogue_done);
            __syncwarp();
            asm volatile("tcgen05.fence::after_thread_sync;");
            asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(128));
        }
    }
    // ---- Role: epilogue ----
    if (warp >= 2 && warp <= 5) {
        { // epilogue_main
            const int epi_warp = warp % 4;
            const int epi_group = (warp - 2) / 4;
            int tok_g0 = epi_group * 64;
            const int lane_row = epi_warp * 32 + lane;
            const int epi_tid = epi_warp * 32 + lane;
            int w0_e = bid;
            int ws_e = num_bids;
            int n_items_e = (total_work - bid + num_bids - 1) / num_bids;
            unsigned int _phase_mainloop_done = 0;
            unsigned int _phase_peers_free = 1;
            unsigned int _phase_red_full = 0;
            #pragma unroll 1
            for (int it_e = 0; it_e < n_items_e; it_e++) {
                int work_e = w0_e + it_e * ws_e;
                int tile_2 = work_e / split;
                int rank_2 = work_e - tile_2 * split;
                int k_begin_2 = num_k_iters * rank_2 / split;
                int k_end_2 = num_k_iters * (rank_2 + 1) / split;
                int k_count_2 = k_end_2 - k_begin_2;
                int n_tile_e = tile_2 % n_tiles;
                int m_tile_e = tile_2 / n_tiles;
                int feature = n_tile_e * 128 + lane_row;
                int tok0 = m_tile_e * 64;
                unsigned long long col = (unsigned long long)feature;
                mbarrier_wait(mainloop_done_addr, _phase_mainloop_done);
                asm volatile("tcgen05.fence::after_thread_sync;");
                int acc_col_e = 0;
                float _tmem_load_0[64];
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(_tmem_load_0[0]), "=f"(_tmem_load_0[1]), "=f"(_tmem_load_0[2]), "=f"(_tmem_load_0[3]), "=f"(_tmem_load_0[4]), "=f"(_tmem_load_0[5]), "=f"(_tmem_load_0[6]), "=f"(_tmem_load_0[7]), "=f"(_tmem_load_0[8]), "=f"(_tmem_load_0[9]), "=f"(_tmem_load_0[10]), "=f"(_tmem_load_0[11]), "=f"(_tmem_load_0[12]), "=f"(_tmem_load_0[13]), "=f"(_tmem_load_0[14]), "=f"(_tmem_load_0[15]), "=f"(_tmem_load_0[16]), "=f"(_tmem_load_0[17]), "=f"(_tmem_load_0[18]), "=f"(_tmem_load_0[19]), "=f"(_tmem_load_0[20]), "=f"(_tmem_load_0[21]), "=f"(_tmem_load_0[22]), "=f"(_tmem_load_0[23]), "=f"(_tmem_load_0[24]), "=f"(_tmem_load_0[25]), "=f"(_tmem_load_0[26]), "=f"(_tmem_load_0[27]), "=f"(_tmem_load_0[28]), "=f"(_tmem_load_0[29]), "=f"(_tmem_load_0[30]), "=f"(_tmem_load_0[31])
                    : "r"(taddr + (unsigned int)(epi_warp * 32 << 16) + (unsigned int)acc_col_e + (unsigned int)tok_g0));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(_tmem_load_0[32]), "=f"(_tmem_load_0[33]), "=f"(_tmem_load_0[34]), "=f"(_tmem_load_0[35]), "=f"(_tmem_load_0[36]), "=f"(_tmem_load_0[37]), "=f"(_tmem_load_0[38]), "=f"(_tmem_load_0[39]), "=f"(_tmem_load_0[40]), "=f"(_tmem_load_0[41]), "=f"(_tmem_load_0[42]), "=f"(_tmem_load_0[43]), "=f"(_tmem_load_0[44]), "=f"(_tmem_load_0[45]), "=f"(_tmem_load_0[46]), "=f"(_tmem_load_0[47]), "=f"(_tmem_load_0[48]), "=f"(_tmem_load_0[49]), "=f"(_tmem_load_0[50]), "=f"(_tmem_load_0[51]), "=f"(_tmem_load_0[52]), "=f"(_tmem_load_0[53]), "=f"(_tmem_load_0[54]), "=f"(_tmem_load_0[55]), "=f"(_tmem_load_0[56]), "=f"(_tmem_load_0[57]), "=f"(_tmem_load_0[58]), "=f"(_tmem_load_0[59]), "=f"(_tmem_load_0[60]), "=f"(_tmem_load_0[61]), "=f"(_tmem_load_0[62]), "=f"(_tmem_load_0[63])
                    : "r"(taddr + (unsigned int)(epi_warp * 32 << 16) + (unsigned int)acc_col_e + (unsigned int)tok_g0 + 32));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                if (elect_sync()) {
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    mbarrier_arrive(epilogue_done_addr);
                }
                _phase_mainloop_done ^= 1;
                unsigned int epi_base = smem_epi_addr + (unsigned int)(epi_group * 16384);
                int my_rank = cta_rank;
                int _min_0 = ((64) < ((my_rank + 1) * 16) ? (64) : ((my_rank + 1) * 16));
                int rows_mine = _min_0 - my_rank * 16;
                unsigned int line_b = (unsigned int)(lane_row * 68);
                unsigned int slot_b = (unsigned int)(my_rank * 8704);
                mbarrier_wait_cluster_hint(peers_free_addr, _phase_peers_free, 10000000);
                if (warp == 2) {
                    asm volatile("cp.async.bulk.wait_group.read 0;");
                }
                asm volatile("tcgen05.fence::before_thread_sync;");
                asm volatile("barrier.sync 1, 128;" ::: "memory");
                if (epi_group == 0) {
                    if (0 == my_rank) {
                        {
                            uint32_t _addr_1 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_1), "f"(_tmem_load_0[0]) : "memory");
                        }
                        {
                            uint32_t _addr_2 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 4));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_2), "f"(_tmem_load_0[1]) : "memory");
                        }
                        {
                            uint32_t _addr_3 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 8));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_3), "f"(_tmem_load_0[2]) : "memory");
                        }
                        {
                            uint32_t _addr_4 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 12));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_4), "f"(_tmem_load_0[3]) : "memory");
                        }
                        {
                            uint32_t _addr_5 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 16));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_5), "f"(_tmem_load_0[4]) : "memory");
                        }
                        {
                            uint32_t _addr_6 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 20));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_6), "f"(_tmem_load_0[5]) : "memory");
                        }
                        {
                            uint32_t _addr_7 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 24));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_7), "f"(_tmem_load_0[6]) : "memory");
                        }
                        {
                            uint32_t _addr_8 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 28));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_8), "f"(_tmem_load_0[7]) : "memory");
                        }
                        {
                            uint32_t _addr_9 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 32));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_9), "f"(_tmem_load_0[8]) : "memory");
                        }
                        {
                            uint32_t _addr_10 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 36));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_10), "f"(_tmem_load_0[9]) : "memory");
                        }
                        {
                            uint32_t _addr_11 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 40));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_11), "f"(_tmem_load_0[10]) : "memory");
                        }
                        {
                            uint32_t _addr_12 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 44));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_12), "f"(_tmem_load_0[11]) : "memory");
                        }
                        {
                            uint32_t _addr_13 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 48));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_13), "f"(_tmem_load_0[12]) : "memory");
                        }
                        {
                            uint32_t _addr_14 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 52));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_14), "f"(_tmem_load_0[13]) : "memory");
                        }
                        {
                            uint32_t _addr_15 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 56));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_15), "f"(_tmem_load_0[14]) : "memory");
                        }
                        {
                            uint32_t _addr_16 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 60));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_16), "f"(_tmem_load_0[15]) : "memory");
                        }
                    } else {
                        int _max_18 = ((0) > (-my_rank) ? (0) : (-my_rank));
                        int _min_17 = ((1) < (_max_18) ? (1) : (_max_18));
                        int o_idx = -_min_17;
                        unsigned int stg_b = (unsigned int)(o_idx * 8704) + line_b;
                        {
                            uint32_t _addr_19 = static_cast<uint32_t>(smem_stg_addr + stg_b);
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_19), "f"(_tmem_load_0[0]) : "memory");
                        }
                        {
                            uint32_t _addr_20 = static_cast<uint32_t>(smem_stg_addr + (stg_b + 4));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_20), "f"(_tmem_load_0[1]) : "memory");
                        }
                        {
                            uint32_t _addr_21 = static_cast<uint32_t>(smem_stg_addr + (stg_b + 8));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_21), "f"(_tmem_load_0[2]) : "memory");
                        }
                        {
                            uint32_t _addr_22 = static_cast<uint32_t>(smem_stg_addr + (stg_b + 12));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_22), "f"(_tmem_load_0[3]) : "memory");
                        }
                        {
                            uint32_t _addr_23 = static_cast<uint32_t>(smem_stg_addr + (stg_b + 16));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_23), "f"(_tmem_load_0[4]) : "memory");
                        }
                        {
                            uint32_t _addr_24 = static_cast<uint32_t>(smem_stg_addr + (stg_b + 20));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_24), "f"(_tmem_load_0[5]) : "memory");
                        }
                        {
                            uint32_t _addr_25 = static_cast<uint32_t>(smem_stg_addr + (stg_b + 24));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_25), "f"(_tmem_load_0[6]) : "memory");
                        }
                        {
                            uint32_t _addr_26 = static_cast<uint32_t>(smem_stg_addr + (stg_b + 28));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_26), "f"(_tmem_load_0[7]) : "memory");
                        }
                        {
                            uint32_t _addr_27 = static_cast<uint32_t>(smem_stg_addr + (stg_b + 32));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_27), "f"(_tmem_load_0[8]) : "memory");
                        }
                        {
                            uint32_t _addr_28 = static_cast<uint32_t>(smem_stg_addr + (stg_b + 36));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_28), "f"(_tmem_load_0[9]) : "memory");
                        }
                        {
                            uint32_t _addr_29 = static_cast<uint32_t>(smem_stg_addr + (stg_b + 40));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_29), "f"(_tmem_load_0[10]) : "memory");
                        }
                        {
                            uint32_t _addr_30 = static_cast<uint32_t>(smem_stg_addr + (stg_b + 44));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_30), "f"(_tmem_load_0[11]) : "memory");
                        }
                        {
                            uint32_t _addr_31 = static_cast<uint32_t>(smem_stg_addr + (stg_b + 48));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_31), "f"(_tmem_load_0[12]) : "memory");
                        }
                        {
                            uint32_t _addr_32 = static_cast<uint32_t>(smem_stg_addr + (stg_b + 52));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_32), "f"(_tmem_load_0[13]) : "memory");
                        }
                        {
                            uint32_t _addr_33 = static_cast<uint32_t>(smem_stg_addr + (stg_b + 56));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_33), "f"(_tmem_load_0[14]) : "memory");
                        }
                        {
                            uint32_t _addr_34 = static_cast<uint32_t>(smem_stg_addr + (stg_b + 60));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_34), "f"(_tmem_load_0[15]) : "memory");
                        }
                    }
                    if (1 == my_rank) {
                        {
                            uint32_t _addr_35 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_35), "f"(_tmem_load_0[16]) : "memory");
                        }
                        {
                            uint32_t _addr_36 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 4));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_36), "f"(_tmem_load_0[17]) : "memory");
                        }
                        {
                            uint32_t _addr_37 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 8));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_37), "f"(_tmem_load_0[18]) : "memory");
                        }
                        {
                            uint32_t _addr_38 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 12));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_38), "f"(_tmem_load_0[19]) : "memory");
                        }
                        {
                            uint32_t _addr_39 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 16));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_39), "f"(_tmem_load_0[20]) : "memory");
                        }
                        {
                            uint32_t _addr_40 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 20));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_40), "f"(_tmem_load_0[21]) : "memory");
                        }
                        {
                            uint32_t _addr_41 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 24));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_41), "f"(_tmem_load_0[22]) : "memory");
                        }
                        {
                            uint32_t _addr_42 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 28));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_42), "f"(_tmem_load_0[23]) : "memory");
                        }
                        {
                            uint32_t _addr_43 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 32));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_43), "f"(_tmem_load_0[24]) : "memory");
                        }
                        {
                            uint32_t _addr_44 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 36));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_44), "f"(_tmem_load_0[25]) : "memory");
                        }
                        {
                            uint32_t _addr_45 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 40));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_45), "f"(_tmem_load_0[26]) : "memory");
                        }
                        {
                            uint32_t _addr_46 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 44));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_46), "f"(_tmem_load_0[27]) : "memory");
                        }
                        {
                            uint32_t _addr_47 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 48));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_47), "f"(_tmem_load_0[28]) : "memory");
                        }
                        {
                            uint32_t _addr_48 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 52));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_48), "f"(_tmem_load_0[29]) : "memory");
                        }
                        {
                            uint32_t _addr_49 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 56));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_49), "f"(_tmem_load_0[30]) : "memory");
                        }
                        {
                            uint32_t _addr_50 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 60));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_50), "f"(_tmem_load_0[31]) : "memory");
                        }
                    } else {
                        int _max_52 = ((0) > (1 - my_rank) ? (0) : (1 - my_rank));
                        int _min_51 = ((1) < (_max_52) ? (1) : (_max_52));
                        int o_idx_1 = 1 - _min_51;
                        unsigned int stg_b_1 = (unsigned int)(o_idx_1 * 8704) + line_b;
                        {
                            uint32_t _addr_53 = static_cast<uint32_t>(smem_stg_addr + stg_b_1);
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_53), "f"(_tmem_load_0[16]) : "memory");
                        }
                        {
                            uint32_t _addr_54 = static_cast<uint32_t>(smem_stg_addr + (stg_b_1 + 4));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_54), "f"(_tmem_load_0[17]) : "memory");
                        }
                        {
                            uint32_t _addr_55 = static_cast<uint32_t>(smem_stg_addr + (stg_b_1 + 8));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_55), "f"(_tmem_load_0[18]) : "memory");
                        }
                        {
                            uint32_t _addr_56 = static_cast<uint32_t>(smem_stg_addr + (stg_b_1 + 12));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_56), "f"(_tmem_load_0[19]) : "memory");
                        }
                        {
                            uint32_t _addr_57 = static_cast<uint32_t>(smem_stg_addr + (stg_b_1 + 16));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_57), "f"(_tmem_load_0[20]) : "memory");
                        }
                        {
                            uint32_t _addr_58 = static_cast<uint32_t>(smem_stg_addr + (stg_b_1 + 20));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_58), "f"(_tmem_load_0[21]) : "memory");
                        }
                        {
                            uint32_t _addr_59 = static_cast<uint32_t>(smem_stg_addr + (stg_b_1 + 24));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_59), "f"(_tmem_load_0[22]) : "memory");
                        }
                        {
                            uint32_t _addr_60 = static_cast<uint32_t>(smem_stg_addr + (stg_b_1 + 28));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_60), "f"(_tmem_load_0[23]) : "memory");
                        }
                        {
                            uint32_t _addr_61 = static_cast<uint32_t>(smem_stg_addr + (stg_b_1 + 32));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_61), "f"(_tmem_load_0[24]) : "memory");
                        }
                        {
                            uint32_t _addr_62 = static_cast<uint32_t>(smem_stg_addr + (stg_b_1 + 36));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_62), "f"(_tmem_load_0[25]) : "memory");
                        }
                        {
                            uint32_t _addr_63 = static_cast<uint32_t>(smem_stg_addr + (stg_b_1 + 40));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_63), "f"(_tmem_load_0[26]) : "memory");
                        }
                        {
                            uint32_t _addr_64 = static_cast<uint32_t>(smem_stg_addr + (stg_b_1 + 44));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_64), "f"(_tmem_load_0[27]) : "memory");
                        }
                        {
                            uint32_t _addr_65 = static_cast<uint32_t>(smem_stg_addr + (stg_b_1 + 48));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_65), "f"(_tmem_load_0[28]) : "memory");
                        }
                        {
                            uint32_t _addr_66 = static_cast<uint32_t>(smem_stg_addr + (stg_b_1 + 52));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_66), "f"(_tmem_load_0[29]) : "memory");
                        }
                        {
                            uint32_t _addr_67 = static_cast<uint32_t>(smem_stg_addr + (stg_b_1 + 56));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_67), "f"(_tmem_load_0[30]) : "memory");
                        }
                        {
                            uint32_t _addr_68 = static_cast<uint32_t>(smem_stg_addr + (stg_b_1 + 60));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_68), "f"(_tmem_load_0[31]) : "memory");
                        }
                    }
                    if (2 == my_rank) {
                        {
                            uint32_t _addr_69 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_69), "f"(_tmem_load_0[32]) : "memory");
                        }
                        {
                            uint32_t _addr_70 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 4));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_70), "f"(_tmem_load_0[33]) : "memory");
                        }
                        {
                            uint32_t _addr_71 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 8));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_71), "f"(_tmem_load_0[34]) : "memory");
                        }
                        {
                            uint32_t _addr_72 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 12));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_72), "f"(_tmem_load_0[35]) : "memory");
                        }
                        {
                            uint32_t _addr_73 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 16));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_73), "f"(_tmem_load_0[36]) : "memory");
                        }
                        {
                            uint32_t _addr_74 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 20));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_74), "f"(_tmem_load_0[37]) : "memory");
                        }
                        {
                            uint32_t _addr_75 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 24));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_75), "f"(_tmem_load_0[38]) : "memory");
                        }
                        {
                            uint32_t _addr_76 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 28));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_76), "f"(_tmem_load_0[39]) : "memory");
                        }
                        {
                            uint32_t _addr_77 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 32));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_77), "f"(_tmem_load_0[40]) : "memory");
                        }
                        {
                            uint32_t _addr_78 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 36));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_78), "f"(_tmem_load_0[41]) : "memory");
                        }
                        {
                            uint32_t _addr_79 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 40));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_79), "f"(_tmem_load_0[42]) : "memory");
                        }
                        {
                            uint32_t _addr_80 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 44));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_80), "f"(_tmem_load_0[43]) : "memory");
                        }
                        {
                            uint32_t _addr_81 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 48));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_81), "f"(_tmem_load_0[44]) : "memory");
                        }
                        {
                            uint32_t _addr_82 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 52));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_82), "f"(_tmem_load_0[45]) : "memory");
                        }
                        {
                            uint32_t _addr_83 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 56));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_83), "f"(_tmem_load_0[46]) : "memory");
                        }
                        {
                            uint32_t _addr_84 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 60));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_84), "f"(_tmem_load_0[47]) : "memory");
                        }
                    } else {
                        int _max_86 = ((0) > (2 - my_rank) ? (0) : (2 - my_rank));
                        int _min_85 = ((1) < (_max_86) ? (1) : (_max_86));
                        int o_idx_2 = 2 - _min_85;
                        unsigned int stg_b_2 = (unsigned int)(o_idx_2 * 8704) + line_b;
                        {
                            uint32_t _addr_87 = static_cast<uint32_t>(smem_stg_addr + stg_b_2);
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_87), "f"(_tmem_load_0[32]) : "memory");
                        }
                        {
                            uint32_t _addr_88 = static_cast<uint32_t>(smem_stg_addr + (stg_b_2 + 4));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_88), "f"(_tmem_load_0[33]) : "memory");
                        }
                        {
                            uint32_t _addr_89 = static_cast<uint32_t>(smem_stg_addr + (stg_b_2 + 8));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_89), "f"(_tmem_load_0[34]) : "memory");
                        }
                        {
                            uint32_t _addr_90 = static_cast<uint32_t>(smem_stg_addr + (stg_b_2 + 12));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_90), "f"(_tmem_load_0[35]) : "memory");
                        }
                        {
                            uint32_t _addr_91 = static_cast<uint32_t>(smem_stg_addr + (stg_b_2 + 16));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_91), "f"(_tmem_load_0[36]) : "memory");
                        }
                        {
                            uint32_t _addr_92 = static_cast<uint32_t>(smem_stg_addr + (stg_b_2 + 20));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_92), "f"(_tmem_load_0[37]) : "memory");
                        }
                        {
                            uint32_t _addr_93 = static_cast<uint32_t>(smem_stg_addr + (stg_b_2 + 24));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_93), "f"(_tmem_load_0[38]) : "memory");
                        }
                        {
                            uint32_t _addr_94 = static_cast<uint32_t>(smem_stg_addr + (stg_b_2 + 28));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_94), "f"(_tmem_load_0[39]) : "memory");
                        }
                        {
                            uint32_t _addr_95 = static_cast<uint32_t>(smem_stg_addr + (stg_b_2 + 32));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_95), "f"(_tmem_load_0[40]) : "memory");
                        }
                        {
                            uint32_t _addr_96 = static_cast<uint32_t>(smem_stg_addr + (stg_b_2 + 36));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_96), "f"(_tmem_load_0[41]) : "memory");
                        }
                        {
                            uint32_t _addr_97 = static_cast<uint32_t>(smem_stg_addr + (stg_b_2 + 40));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_97), "f"(_tmem_load_0[42]) : "memory");
                        }
                        {
                            uint32_t _addr_98 = static_cast<uint32_t>(smem_stg_addr + (stg_b_2 + 44));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_98), "f"(_tmem_load_0[43]) : "memory");
                        }
                        {
                            uint32_t _addr_99 = static_cast<uint32_t>(smem_stg_addr + (stg_b_2 + 48));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_99), "f"(_tmem_load_0[44]) : "memory");
                        }
                        {
                            uint32_t _addr_100 = static_cast<uint32_t>(smem_stg_addr + (stg_b_2 + 52));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_100), "f"(_tmem_load_0[45]) : "memory");
                        }
                        {
                            uint32_t _addr_101 = static_cast<uint32_t>(smem_stg_addr + (stg_b_2 + 56));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_101), "f"(_tmem_load_0[46]) : "memory");
                        }
                        {
                            uint32_t _addr_102 = static_cast<uint32_t>(smem_stg_addr + (stg_b_2 + 60));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_102), "f"(_tmem_load_0[47]) : "memory");
                        }
                    }
                    if (3 == my_rank) {
                        {
                            uint32_t _addr_103 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_103), "f"(_tmem_load_0[48]) : "memory");
                        }
                        {
                            uint32_t _addr_104 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 4));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_104), "f"(_tmem_load_0[49]) : "memory");
                        }
                        {
                            uint32_t _addr_105 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 8));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_105), "f"(_tmem_load_0[50]) : "memory");
                        }
                        {
                            uint32_t _addr_106 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 12));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_106), "f"(_tmem_load_0[51]) : "memory");
                        }
                        {
                            uint32_t _addr_107 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 16));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_107), "f"(_tmem_load_0[52]) : "memory");
                        }
                        {
                            uint32_t _addr_108 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 20));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_108), "f"(_tmem_load_0[53]) : "memory");
                        }
                        {
                            uint32_t _addr_109 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 24));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_109), "f"(_tmem_load_0[54]) : "memory");
                        }
                        {
                            uint32_t _addr_110 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 28));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_110), "f"(_tmem_load_0[55]) : "memory");
                        }
                        {
                            uint32_t _addr_111 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 32));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_111), "f"(_tmem_load_0[56]) : "memory");
                        }
                        {
                            uint32_t _addr_112 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 36));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_112), "f"(_tmem_load_0[57]) : "memory");
                        }
                        {
                            uint32_t _addr_113 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 40));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_113), "f"(_tmem_load_0[58]) : "memory");
                        }
                        {
                            uint32_t _addr_114 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 44));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_114), "f"(_tmem_load_0[59]) : "memory");
                        }
                        {
                            uint32_t _addr_115 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 48));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_115), "f"(_tmem_load_0[60]) : "memory");
                        }
                        {
                            uint32_t _addr_116 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 52));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_116), "f"(_tmem_load_0[61]) : "memory");
                        }
                        {
                            uint32_t _addr_117 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 56));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_117), "f"(_tmem_load_0[62]) : "memory");
                        }
                        {
                            uint32_t _addr_118 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 60));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_118), "f"(_tmem_load_0[63]) : "memory");
                        }
                    } else {
                        int _max_120 = ((0) > (3 - my_rank) ? (0) : (3 - my_rank));
                        int _min_119 = ((1) < (_max_120) ? (1) : (_max_120));
                        int o_idx_3 = 3 - _min_119;
                        unsigned int stg_b_3 = (unsigned int)(o_idx_3 * 8704) + line_b;
                        {
                            uint32_t _addr_121 = static_cast<uint32_t>(smem_stg_addr + stg_b_3);
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_121), "f"(_tmem_load_0[48]) : "memory");
                        }
                        {
                            uint32_t _addr_122 = static_cast<uint32_t>(smem_stg_addr + (stg_b_3 + 4));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_122), "f"(_tmem_load_0[49]) : "memory");
                        }
                        {
                            uint32_t _addr_123 = static_cast<uint32_t>(smem_stg_addr + (stg_b_3 + 8));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_123), "f"(_tmem_load_0[50]) : "memory");
                        }
                        {
                            uint32_t _addr_124 = static_cast<uint32_t>(smem_stg_addr + (stg_b_3 + 12));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_124), "f"(_tmem_load_0[51]) : "memory");
                        }
                        {
                            uint32_t _addr_125 = static_cast<uint32_t>(smem_stg_addr + (stg_b_3 + 16));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_125), "f"(_tmem_load_0[52]) : "memory");
                        }
                        {
                            uint32_t _addr_126 = static_cast<uint32_t>(smem_stg_addr + (stg_b_3 + 20));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_126), "f"(_tmem_load_0[53]) : "memory");
                        }
                        {
                            uint32_t _addr_127 = static_cast<uint32_t>(smem_stg_addr + (stg_b_3 + 24));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_127), "f"(_tmem_load_0[54]) : "memory");
                        }
                        {
                            uint32_t _addr_128 = static_cast<uint32_t>(smem_stg_addr + (stg_b_3 + 28));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_128), "f"(_tmem_load_0[55]) : "memory");
                        }
                        {
                            uint32_t _addr_129 = static_cast<uint32_t>(smem_stg_addr + (stg_b_3 + 32));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_129), "f"(_tmem_load_0[56]) : "memory");
                        }
                        {
                            uint32_t _addr_130 = static_cast<uint32_t>(smem_stg_addr + (stg_b_3 + 36));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_130), "f"(_tmem_load_0[57]) : "memory");
                        }
                        {
                            uint32_t _addr_131 = static_cast<uint32_t>(smem_stg_addr + (stg_b_3 + 40));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_131), "f"(_tmem_load_0[58]) : "memory");
                        }
                        {
                            uint32_t _addr_132 = static_cast<uint32_t>(smem_stg_addr + (stg_b_3 + 44));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_132), "f"(_tmem_load_0[59]) : "memory");
                        }
                        {
                            uint32_t _addr_133 = static_cast<uint32_t>(smem_stg_addr + (stg_b_3 + 48));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_133), "f"(_tmem_load_0[60]) : "memory");
                        }
                        {
                            uint32_t _addr_134 = static_cast<uint32_t>(smem_stg_addr + (stg_b_3 + 52));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_134), "f"(_tmem_load_0[61]) : "memory");
                        }
                        {
                            uint32_t _addr_135 = static_cast<uint32_t>(smem_stg_addr + (stg_b_3 + 56));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_135), "f"(_tmem_load_0[62]) : "memory");
                        }
                        {
                            uint32_t _addr_136 = static_cast<uint32_t>(smem_stg_addr + (stg_b_3 + 60));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_136), "f"(_tmem_load_0[63]) : "memory");
                        }
                    }
                }
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("barrier.sync 1, 128;" ::: "memory");
                int _min_137 = ((16) < (rows_mine) ? (16) : (rows_mine));
                int rows_round = ((0) > (_min_137) ? (0) : (_min_137));
                if (warp == 2) {
                    int owner_l = lane;
                    int _min_139 = ((64) < ((owner_l + 1) * 16) ? (64) : ((owner_l + 1) * 16));
                    int _min_138 = ((16) < (_min_139 - owner_l * 16) ? (16) : (_min_139 - owner_l * 16));
                    int n_o_l = ((0) > (_min_138) ? (0) : (_min_138));
                    if (owner_l < 4 && owner_l != my_rank && n_o_l > 0) {
                        int _max_141 = ((0) > (owner_l - my_rank) ? (0) : (owner_l - my_rank));
                        int _min_140 = ((1) < (_max_141) ? (1) : (_max_141));
                        int o_idx_l = owner_l - _min_140;
                        uint32_t _mbar_peer_142;
                        asm(
                            "mapa.shared::cluster.u32 %0, %1, %2;"
                            : "=r"(_mbar_peer_142) : "r"(red_full_addr), "r"(owner_l));
                        uint32_t _cluster_mapa_143;
                        asm volatile(
                            "mapa.shared::cluster.u32 %0, %1, %2;"
                            : "=r"(_cluster_mapa_143) : "r"(smem_red_addr + slot_b), "r"(owner_l));
                        asm volatile(
                            "cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes"
                            " [%0], [%1], %2, [%3];"
                            :: "r"(_cluster_mapa_143), "r"(smem_stg_addr + (unsigned int)(o_idx_l * 8704)), "r"((uint32_t)(8704)), "r"(_mbar_peer_142)
                            : "memory");
                        asm volatile("cp.async.bulk.commit_group;");
                    }
                    if (lane == 0) {
                        int _min_144 = ((1) < (rows_round) ? (1) : (rows_round));
                        mbarrier_arrive_expect_tx(red_full_addr, _min_144 * 26112);
                    }
                }
                mbarrier_wait_cluster_hint(red_full_addr, _phase_red_full, 10000000);
                if (epi_group == 0) {
                    if (0 == my_rank) {
                        float total_1_0 = 0.0f;
                        #pragma unroll
                        for (int src_1_0 = 0; src_1_0 < 4; src_1_0++) {
                            total_1_0 = total_1_0 + smem_red[src_1_0 * 2176 + lane_row * 17];
                        }
                        int tok_c_2_0 = tok0;
                        if (feature < n_valid && tok_c_2_0 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_2_0 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_1_0;
                        }
                        float total_3_0 = 0.0f;
                        #pragma unroll
                        for (int src_1_1 = 0; src_1_1 < 4; src_1_1++) {
                            total_3_0 = total_3_0 + smem_red[src_1_1 * 2176 + lane_row * 17 + 1];
                        }
                        int tok_c_4_0 = tok0 + 1;
                        if (feature < n_valid && tok_c_4_0 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_4_0 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_3_0;
                        }
                        float total_5_0 = 0.0f;
                        #pragma unroll
                        for (int src_1_2 = 0; src_1_2 < 4; src_1_2++) {
                            total_5_0 = total_5_0 + smem_red[src_1_2 * 2176 + lane_row * 17 + 2];
                        }
                        int tok_c_6_0 = tok0 + 2;
                        if (feature < n_valid && tok_c_6_0 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_6_0 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_5_0;
                        }
                        float total_7_0 = 0.0f;
                        #pragma unroll
                        for (int src_1_3 = 0; src_1_3 < 4; src_1_3++) {
                            total_7_0 = total_7_0 + smem_red[src_1_3 * 2176 + lane_row * 17 + 3];
                        }
                        int tok_c_8_0 = tok0 + 3;
                        if (feature < n_valid && tok_c_8_0 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_8_0 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_7_0;
                        }
                        float total_9_0 = 0.0f;
                        #pragma unroll
                        for (int src_1_4 = 0; src_1_4 < 4; src_1_4++) {
                            total_9_0 = total_9_0 + smem_red[src_1_4 * 2176 + lane_row * 17 + 4];
                        }
                        int tok_c_10_0 = tok0 + 4;
                        if (feature < n_valid && tok_c_10_0 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_10_0 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_9_0;
                        }
                        float total_11_0 = 0.0f;
                        #pragma unroll
                        for (int src_1_5 = 0; src_1_5 < 4; src_1_5++) {
                            total_11_0 = total_11_0 + smem_red[src_1_5 * 2176 + lane_row * 17 + 5];
                        }
                        int tok_c_12_0 = tok0 + 5;
                        if (feature < n_valid && tok_c_12_0 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_12_0 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_11_0;
                        }
                        float total_13_0 = 0.0f;
                        #pragma unroll
                        for (int src_1_6 = 0; src_1_6 < 4; src_1_6++) {
                            total_13_0 = total_13_0 + smem_red[src_1_6 * 2176 + lane_row * 17 + 6];
                        }
                        int tok_c_14_0 = tok0 + 6;
                        if (feature < n_valid && tok_c_14_0 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_14_0 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_13_0;
                        }
                        float total_15_0 = 0.0f;
                        #pragma unroll
                        for (int src_1_7 = 0; src_1_7 < 4; src_1_7++) {
                            total_15_0 = total_15_0 + smem_red[src_1_7 * 2176 + lane_row * 17 + 7];
                        }
                        int tok_c_16_0 = tok0 + 7;
                        if (feature < n_valid && tok_c_16_0 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_16_0 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_15_0;
                        }
                        float total_17_0 = 0.0f;
                        #pragma unroll
                        for (int src_1_8 = 0; src_1_8 < 4; src_1_8++) {
                            total_17_0 = total_17_0 + smem_red[src_1_8 * 2176 + lane_row * 17 + 8];
                        }
                        int tok_c_18_0 = tok0 + 8;
                        if (feature < n_valid && tok_c_18_0 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_18_0 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_17_0;
                        }
                        float total_19_0 = 0.0f;
                        #pragma unroll
                        for (int src_1_9 = 0; src_1_9 < 4; src_1_9++) {
                            total_19_0 = total_19_0 + smem_red[src_1_9 * 2176 + lane_row * 17 + 9];
                        }
                        int tok_c_20_0 = tok0 + 9;
                        if (feature < n_valid && tok_c_20_0 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_20_0 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_19_0;
                        }
                        float total_21_0 = 0.0f;
                        #pragma unroll
                        for (int src_1_10 = 0; src_1_10 < 4; src_1_10++) {
                            total_21_0 = total_21_0 + smem_red[src_1_10 * 2176 + lane_row * 17 + 10];
                        }
                        int tok_c_22_0 = tok0 + 10;
                        if (feature < n_valid && tok_c_22_0 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_22_0 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_21_0;
                        }
                        float total_23_0 = 0.0f;
                        #pragma unroll
                        for (int src_1_11 = 0; src_1_11 < 4; src_1_11++) {
                            total_23_0 = total_23_0 + smem_red[src_1_11 * 2176 + lane_row * 17 + 11];
                        }
                        int tok_c_24_0 = tok0 + 11;
                        if (feature < n_valid && tok_c_24_0 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_24_0 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_23_0;
                        }
                        float total_25_0 = 0.0f;
                        #pragma unroll
                        for (int src_1_12 = 0; src_1_12 < 4; src_1_12++) {
                            total_25_0 = total_25_0 + smem_red[src_1_12 * 2176 + lane_row * 17 + 12];
                        }
                        int tok_c_26_0 = tok0 + 12;
                        if (feature < n_valid && tok_c_26_0 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_26_0 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_25_0;
                        }
                        float total_27_0 = 0.0f;
                        #pragma unroll
                        for (int src_1_13 = 0; src_1_13 < 4; src_1_13++) {
                            total_27_0 = total_27_0 + smem_red[src_1_13 * 2176 + lane_row * 17 + 13];
                        }
                        int tok_c_28_0 = tok0 + 13;
                        if (feature < n_valid && tok_c_28_0 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_28_0 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_27_0;
                        }
                        float total_29_0 = 0.0f;
                        #pragma unroll
                        for (int src_1_14 = 0; src_1_14 < 4; src_1_14++) {
                            total_29_0 = total_29_0 + smem_red[src_1_14 * 2176 + lane_row * 17 + 14];
                        }
                        int tok_c_30_0 = tok0 + 14;
                        if (feature < n_valid && tok_c_30_0 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_30_0 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_29_0;
                        }
                        float total_30_0 = 0.0f;
                        #pragma unroll
                        for (int src_1_15 = 0; src_1_15 < 4; src_1_15++) {
                            total_30_0 = total_30_0 + smem_red[src_1_15 * 2176 + lane_row * 17 + 15];
                        }
                        int tok_c_31_0 = tok0 + 15;
                        if (feature < n_valid && tok_c_31_0 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_31_0 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_30_0;
                        }
                    }
                    if (1 == my_rank) {
                        float total_1_1 = 0.0f;
                        #pragma unroll
                        for (int src_1_16 = 0; src_1_16 < 4; src_1_16++) {
                            total_1_1 = total_1_1 + smem_red[src_1_16 * 2176 + lane_row * 17];
                        }
                        int tok_c_2_1 = tok0 + 16;
                        if (feature < n_valid && tok_c_2_1 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_2_1 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_1_1;
                        }
                        float total_3_1 = 0.0f;
                        #pragma unroll
                        for (int src_1_17 = 0; src_1_17 < 4; src_1_17++) {
                            total_3_1 = total_3_1 + smem_red[src_1_17 * 2176 + lane_row * 17 + 1];
                        }
                        int tok_c_4_1 = tok0 + 17;
                        if (feature < n_valid && tok_c_4_1 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_4_1 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_3_1;
                        }
                        float total_5_1 = 0.0f;
                        #pragma unroll
                        for (int src_1_18 = 0; src_1_18 < 4; src_1_18++) {
                            total_5_1 = total_5_1 + smem_red[src_1_18 * 2176 + lane_row * 17 + 2];
                        }
                        int tok_c_6_1 = tok0 + 18;
                        if (feature < n_valid && tok_c_6_1 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_6_1 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_5_1;
                        }
                        float total_7_1 = 0.0f;
                        #pragma unroll
                        for (int src_1_19 = 0; src_1_19 < 4; src_1_19++) {
                            total_7_1 = total_7_1 + smem_red[src_1_19 * 2176 + lane_row * 17 + 3];
                        }
                        int tok_c_8_1 = tok0 + 19;
                        if (feature < n_valid && tok_c_8_1 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_8_1 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_7_1;
                        }
                        float total_9_1 = 0.0f;
                        #pragma unroll
                        for (int src_1_20 = 0; src_1_20 < 4; src_1_20++) {
                            total_9_1 = total_9_1 + smem_red[src_1_20 * 2176 + lane_row * 17 + 4];
                        }
                        int tok_c_10_1 = tok0 + 20;
                        if (feature < n_valid && tok_c_10_1 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_10_1 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_9_1;
                        }
                        float total_11_1 = 0.0f;
                        #pragma unroll
                        for (int src_1_21 = 0; src_1_21 < 4; src_1_21++) {
                            total_11_1 = total_11_1 + smem_red[src_1_21 * 2176 + lane_row * 17 + 5];
                        }
                        int tok_c_12_1 = tok0 + 21;
                        if (feature < n_valid && tok_c_12_1 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_12_1 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_11_1;
                        }
                        float total_13_1 = 0.0f;
                        #pragma unroll
                        for (int src_1_22 = 0; src_1_22 < 4; src_1_22++) {
                            total_13_1 = total_13_1 + smem_red[src_1_22 * 2176 + lane_row * 17 + 6];
                        }
                        int tok_c_14_1 = tok0 + 22;
                        if (feature < n_valid && tok_c_14_1 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_14_1 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_13_1;
                        }
                        float total_15_1 = 0.0f;
                        #pragma unroll
                        for (int src_1_23 = 0; src_1_23 < 4; src_1_23++) {
                            total_15_1 = total_15_1 + smem_red[src_1_23 * 2176 + lane_row * 17 + 7];
                        }
                        int tok_c_16_1 = tok0 + 23;
                        if (feature < n_valid && tok_c_16_1 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_16_1 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_15_1;
                        }
                        float total_17_1 = 0.0f;
                        #pragma unroll
                        for (int src_1_24 = 0; src_1_24 < 4; src_1_24++) {
                            total_17_1 = total_17_1 + smem_red[src_1_24 * 2176 + lane_row * 17 + 8];
                        }
                        int tok_c_18_1 = tok0 + 24;
                        if (feature < n_valid && tok_c_18_1 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_18_1 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_17_1;
                        }
                        float total_19_1 = 0.0f;
                        #pragma unroll
                        for (int src_1_25 = 0; src_1_25 < 4; src_1_25++) {
                            total_19_1 = total_19_1 + smem_red[src_1_25 * 2176 + lane_row * 17 + 9];
                        }
                        int tok_c_20_1 = tok0 + 25;
                        if (feature < n_valid && tok_c_20_1 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_20_1 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_19_1;
                        }
                        float total_21_1 = 0.0f;
                        #pragma unroll
                        for (int src_1_26 = 0; src_1_26 < 4; src_1_26++) {
                            total_21_1 = total_21_1 + smem_red[src_1_26 * 2176 + lane_row * 17 + 10];
                        }
                        int tok_c_22_1 = tok0 + 26;
                        if (feature < n_valid && tok_c_22_1 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_22_1 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_21_1;
                        }
                        float total_23_1 = 0.0f;
                        #pragma unroll
                        for (int src_1_27 = 0; src_1_27 < 4; src_1_27++) {
                            total_23_1 = total_23_1 + smem_red[src_1_27 * 2176 + lane_row * 17 + 11];
                        }
                        int tok_c_24_1 = tok0 + 27;
                        if (feature < n_valid && tok_c_24_1 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_24_1 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_23_1;
                        }
                        float total_25_1 = 0.0f;
                        #pragma unroll
                        for (int src_1_28 = 0; src_1_28 < 4; src_1_28++) {
                            total_25_1 = total_25_1 + smem_red[src_1_28 * 2176 + lane_row * 17 + 12];
                        }
                        int tok_c_26_1 = tok0 + 28;
                        if (feature < n_valid && tok_c_26_1 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_26_1 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_25_1;
                        }
                        float total_27_1 = 0.0f;
                        #pragma unroll
                        for (int src_1_29 = 0; src_1_29 < 4; src_1_29++) {
                            total_27_1 = total_27_1 + smem_red[src_1_29 * 2176 + lane_row * 17 + 13];
                        }
                        int tok_c_28_1 = tok0 + 29;
                        if (feature < n_valid && tok_c_28_1 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_28_1 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_27_1;
                        }
                        float total_29_1 = 0.0f;
                        #pragma unroll
                        for (int src_1_30 = 0; src_1_30 < 4; src_1_30++) {
                            total_29_1 = total_29_1 + smem_red[src_1_30 * 2176 + lane_row * 17 + 14];
                        }
                        int tok_c_30_1 = tok0 + 30;
                        if (feature < n_valid && tok_c_30_1 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_30_1 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_29_1;
                        }
                        float total_30_1 = 0.0f;
                        #pragma unroll
                        for (int src_1_31 = 0; src_1_31 < 4; src_1_31++) {
                            total_30_1 = total_30_1 + smem_red[src_1_31 * 2176 + lane_row * 17 + 15];
                        }
                        int tok_c_31_1 = tok0 + 31;
                        if (feature < n_valid && tok_c_31_1 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_31_1 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_30_1;
                        }
                    }
                    if (2 == my_rank) {
                        float total_1_2 = 0.0f;
                        #pragma unroll
                        for (int src_1_32 = 0; src_1_32 < 4; src_1_32++) {
                            total_1_2 = total_1_2 + smem_red[src_1_32 * 2176 + lane_row * 17];
                        }
                        int tok_c_2_2 = tok0 + 32;
                        if (feature < n_valid && tok_c_2_2 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_2_2 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_1_2;
                        }
                        float total_3_2 = 0.0f;
                        #pragma unroll
                        for (int src_1_33 = 0; src_1_33 < 4; src_1_33++) {
                            total_3_2 = total_3_2 + smem_red[src_1_33 * 2176 + lane_row * 17 + 1];
                        }
                        int tok_c_4_2 = tok0 + 33;
                        if (feature < n_valid && tok_c_4_2 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_4_2 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_3_2;
                        }
                        float total_5_2 = 0.0f;
                        #pragma unroll
                        for (int src_1_34 = 0; src_1_34 < 4; src_1_34++) {
                            total_5_2 = total_5_2 + smem_red[src_1_34 * 2176 + lane_row * 17 + 2];
                        }
                        int tok_c_6_2 = tok0 + 34;
                        if (feature < n_valid && tok_c_6_2 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_6_2 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_5_2;
                        }
                        float total_7_2 = 0.0f;
                        #pragma unroll
                        for (int src_1_35 = 0; src_1_35 < 4; src_1_35++) {
                            total_7_2 = total_7_2 + smem_red[src_1_35 * 2176 + lane_row * 17 + 3];
                        }
                        int tok_c_8_2 = tok0 + 35;
                        if (feature < n_valid && tok_c_8_2 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_8_2 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_7_2;
                        }
                        float total_9_2 = 0.0f;
                        #pragma unroll
                        for (int src_1_36 = 0; src_1_36 < 4; src_1_36++) {
                            total_9_2 = total_9_2 + smem_red[src_1_36 * 2176 + lane_row * 17 + 4];
                        }
                        int tok_c_10_2 = tok0 + 36;
                        if (feature < n_valid && tok_c_10_2 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_10_2 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_9_2;
                        }
                        float total_11_2 = 0.0f;
                        #pragma unroll
                        for (int src_1_37 = 0; src_1_37 < 4; src_1_37++) {
                            total_11_2 = total_11_2 + smem_red[src_1_37 * 2176 + lane_row * 17 + 5];
                        }
                        int tok_c_12_2 = tok0 + 37;
                        if (feature < n_valid && tok_c_12_2 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_12_2 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_11_2;
                        }
                        float total_13_2 = 0.0f;
                        #pragma unroll
                        for (int src_1_38 = 0; src_1_38 < 4; src_1_38++) {
                            total_13_2 = total_13_2 + smem_red[src_1_38 * 2176 + lane_row * 17 + 6];
                        }
                        int tok_c_14_2 = tok0 + 38;
                        if (feature < n_valid && tok_c_14_2 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_14_2 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_13_2;
                        }
                        float total_15_2 = 0.0f;
                        #pragma unroll
                        for (int src_1_39 = 0; src_1_39 < 4; src_1_39++) {
                            total_15_2 = total_15_2 + smem_red[src_1_39 * 2176 + lane_row * 17 + 7];
                        }
                        int tok_c_16_2 = tok0 + 39;
                        if (feature < n_valid && tok_c_16_2 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_16_2 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_15_2;
                        }
                        float total_17_2 = 0.0f;
                        #pragma unroll
                        for (int src_1_40 = 0; src_1_40 < 4; src_1_40++) {
                            total_17_2 = total_17_2 + smem_red[src_1_40 * 2176 + lane_row * 17 + 8];
                        }
                        int tok_c_18_2 = tok0 + 40;
                        if (feature < n_valid && tok_c_18_2 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_18_2 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_17_2;
                        }
                        float total_19_2 = 0.0f;
                        #pragma unroll
                        for (int src_1_41 = 0; src_1_41 < 4; src_1_41++) {
                            total_19_2 = total_19_2 + smem_red[src_1_41 * 2176 + lane_row * 17 + 9];
                        }
                        int tok_c_20_2 = tok0 + 41;
                        if (feature < n_valid && tok_c_20_2 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_20_2 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_19_2;
                        }
                        float total_21_2 = 0.0f;
                        #pragma unroll
                        for (int src_1_42 = 0; src_1_42 < 4; src_1_42++) {
                            total_21_2 = total_21_2 + smem_red[src_1_42 * 2176 + lane_row * 17 + 10];
                        }
                        int tok_c_22_2 = tok0 + 42;
                        if (feature < n_valid && tok_c_22_2 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_22_2 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_21_2;
                        }
                        float total_23_2 = 0.0f;
                        #pragma unroll
                        for (int src_1_43 = 0; src_1_43 < 4; src_1_43++) {
                            total_23_2 = total_23_2 + smem_red[src_1_43 * 2176 + lane_row * 17 + 11];
                        }
                        int tok_c_24_2 = tok0 + 43;
                        if (feature < n_valid && tok_c_24_2 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_24_2 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_23_2;
                        }
                        float total_25_2 = 0.0f;
                        #pragma unroll
                        for (int src_1_44 = 0; src_1_44 < 4; src_1_44++) {
                            total_25_2 = total_25_2 + smem_red[src_1_44 * 2176 + lane_row * 17 + 12];
                        }
                        int tok_c_26_2 = tok0 + 44;
                        if (feature < n_valid && tok_c_26_2 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_26_2 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_25_2;
                        }
                        float total_27_2 = 0.0f;
                        #pragma unroll
                        for (int src_1_45 = 0; src_1_45 < 4; src_1_45++) {
                            total_27_2 = total_27_2 + smem_red[src_1_45 * 2176 + lane_row * 17 + 13];
                        }
                        int tok_c_28_2 = tok0 + 45;
                        if (feature < n_valid && tok_c_28_2 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_28_2 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_27_2;
                        }
                        float total_29_2 = 0.0f;
                        #pragma unroll
                        for (int src_1_46 = 0; src_1_46 < 4; src_1_46++) {
                            total_29_2 = total_29_2 + smem_red[src_1_46 * 2176 + lane_row * 17 + 14];
                        }
                        int tok_c_30_2 = tok0 + 46;
                        if (feature < n_valid && tok_c_30_2 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_30_2 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_29_2;
                        }
                        float total_30_2 = 0.0f;
                        #pragma unroll
                        for (int src_1_47 = 0; src_1_47 < 4; src_1_47++) {
                            total_30_2 = total_30_2 + smem_red[src_1_47 * 2176 + lane_row * 17 + 15];
                        }
                        int tok_c_31_2 = tok0 + 47;
                        if (feature < n_valid && tok_c_31_2 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_31_2 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_30_2;
                        }
                    }
                    if (3 == my_rank) {
                        float total_1_3 = 0.0f;
                        #pragma unroll
                        for (int src_1_48 = 0; src_1_48 < 4; src_1_48++) {
                            total_1_3 = total_1_3 + smem_red[src_1_48 * 2176 + lane_row * 17];
                        }
                        int tok_c_2_3 = tok0 + 48;
                        if (feature < n_valid && tok_c_2_3 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_2_3 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_1_3;
                        }
                        float total_3_3 = 0.0f;
                        #pragma unroll
                        for (int src_1_49 = 0; src_1_49 < 4; src_1_49++) {
                            total_3_3 = total_3_3 + smem_red[src_1_49 * 2176 + lane_row * 17 + 1];
                        }
                        int tok_c_4_3 = tok0 + 49;
                        if (feature < n_valid && tok_c_4_3 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_4_3 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_3_3;
                        }
                        float total_5_3 = 0.0f;
                        #pragma unroll
                        for (int src_1_50 = 0; src_1_50 < 4; src_1_50++) {
                            total_5_3 = total_5_3 + smem_red[src_1_50 * 2176 + lane_row * 17 + 2];
                        }
                        int tok_c_6_3 = tok0 + 50;
                        if (feature < n_valid && tok_c_6_3 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_6_3 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_5_3;
                        }
                        float total_7_3 = 0.0f;
                        #pragma unroll
                        for (int src_1_51 = 0; src_1_51 < 4; src_1_51++) {
                            total_7_3 = total_7_3 + smem_red[src_1_51 * 2176 + lane_row * 17 + 3];
                        }
                        int tok_c_8_3 = tok0 + 51;
                        if (feature < n_valid && tok_c_8_3 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_8_3 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_7_3;
                        }
                        float total_9_3 = 0.0f;
                        #pragma unroll
                        for (int src_1_52 = 0; src_1_52 < 4; src_1_52++) {
                            total_9_3 = total_9_3 + smem_red[src_1_52 * 2176 + lane_row * 17 + 4];
                        }
                        int tok_c_10_3 = tok0 + 52;
                        if (feature < n_valid && tok_c_10_3 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_10_3 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_9_3;
                        }
                        float total_11_3 = 0.0f;
                        #pragma unroll
                        for (int src_1_53 = 0; src_1_53 < 4; src_1_53++) {
                            total_11_3 = total_11_3 + smem_red[src_1_53 * 2176 + lane_row * 17 + 5];
                        }
                        int tok_c_12_3 = tok0 + 53;
                        if (feature < n_valid && tok_c_12_3 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_12_3 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_11_3;
                        }
                        float total_13_3 = 0.0f;
                        #pragma unroll
                        for (int src_1_54 = 0; src_1_54 < 4; src_1_54++) {
                            total_13_3 = total_13_3 + smem_red[src_1_54 * 2176 + lane_row * 17 + 6];
                        }
                        int tok_c_14_3 = tok0 + 54;
                        if (feature < n_valid && tok_c_14_3 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_14_3 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_13_3;
                        }
                        float total_15_3 = 0.0f;
                        #pragma unroll
                        for (int src_1_55 = 0; src_1_55 < 4; src_1_55++) {
                            total_15_3 = total_15_3 + smem_red[src_1_55 * 2176 + lane_row * 17 + 7];
                        }
                        int tok_c_16_3 = tok0 + 55;
                        if (feature < n_valid && tok_c_16_3 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_16_3 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_15_3;
                        }
                        float total_17_3 = 0.0f;
                        #pragma unroll
                        for (int src_1_56 = 0; src_1_56 < 4; src_1_56++) {
                            total_17_3 = total_17_3 + smem_red[src_1_56 * 2176 + lane_row * 17 + 8];
                        }
                        int tok_c_18_3 = tok0 + 56;
                        if (feature < n_valid && tok_c_18_3 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_18_3 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_17_3;
                        }
                        float total_19_3 = 0.0f;
                        #pragma unroll
                        for (int src_1_57 = 0; src_1_57 < 4; src_1_57++) {
                            total_19_3 = total_19_3 + smem_red[src_1_57 * 2176 + lane_row * 17 + 9];
                        }
                        int tok_c_20_3 = tok0 + 57;
                        if (feature < n_valid && tok_c_20_3 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_20_3 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_19_3;
                        }
                        float total_21_3 = 0.0f;
                        #pragma unroll
                        for (int src_1_58 = 0; src_1_58 < 4; src_1_58++) {
                            total_21_3 = total_21_3 + smem_red[src_1_58 * 2176 + lane_row * 17 + 10];
                        }
                        int tok_c_22_3 = tok0 + 58;
                        if (feature < n_valid && tok_c_22_3 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_22_3 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_21_3;
                        }
                        float total_23_3 = 0.0f;
                        #pragma unroll
                        for (int src_1_59 = 0; src_1_59 < 4; src_1_59++) {
                            total_23_3 = total_23_3 + smem_red[src_1_59 * 2176 + lane_row * 17 + 11];
                        }
                        int tok_c_24_3 = tok0 + 59;
                        if (feature < n_valid && tok_c_24_3 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_24_3 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_23_3;
                        }
                        float total_25_3 = 0.0f;
                        #pragma unroll
                        for (int src_1_60 = 0; src_1_60 < 4; src_1_60++) {
                            total_25_3 = total_25_3 + smem_red[src_1_60 * 2176 + lane_row * 17 + 12];
                        }
                        int tok_c_26_3 = tok0 + 60;
                        if (feature < n_valid && tok_c_26_3 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_26_3 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_25_3;
                        }
                        float total_27_3 = 0.0f;
                        #pragma unroll
                        for (int src_1_61 = 0; src_1_61 < 4; src_1_61++) {
                            total_27_3 = total_27_3 + smem_red[src_1_61 * 2176 + lane_row * 17 + 13];
                        }
                        int tok_c_28_3 = tok0 + 61;
                        if (feature < n_valid && tok_c_28_3 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_28_3 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_27_3;
                        }
                        float total_29_3 = 0.0f;
                        #pragma unroll
                        for (int src_1_62 = 0; src_1_62 < 4; src_1_62++) {
                            total_29_3 = total_29_3 + smem_red[src_1_62 * 2176 + lane_row * 17 + 14];
                        }
                        int tok_c_30_3 = tok0 + 62;
                        if (feature < n_valid && tok_c_30_3 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_30_3 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_29_3;
                        }
                        float total_30_3 = 0.0f;
                        #pragma unroll
                        for (int src_1_63 = 0; src_1_63 < 4; src_1_63++) {
                            total_30_3 = total_30_3 + smem_red[src_1_63 * 2176 + lane_row * 17 + 15];
                        }
                        int tok_c_31_3 = tok0 + 63;
                        if (feature < n_valid && tok_c_31_3 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_31_3 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_30_3;
                        }
                    }
                }
                asm volatile("barrier.sync 1, 128;" ::: "memory");
                if (n_items_e > it_e + 1) {
                    if (tid == 64) {
                        #pragma unroll
                        for (int p = 0; p < 4; p++) {
                            if (p != my_rank) {
                                asm volatile(
                                    "{\n\t"
                                    ".reg .b32 remAddr32;\n\t"
                                    "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                                    "mbarrier.arrive.release.cluster.shared::cluster.b64 _, [remAddr32];\n\t"
                                    "}"
                                    :: "r"(peers_free_addr), "r"(p) : "memory");
                            }
                        }
                    }
                }
                _phase_peers_free ^= 1;
                _phase_red_full ^= 1;
            }
        }
    }
    // ---- Role: quant ----
    if (warp >= 6 && warp <= 21) {
        { // quant_main
            int w0_q = bid;
            int ws_q = num_bids;
            int n_items_q = (total_work - bid + num_bids - 1) / num_bids;
            const int qwarp = warp - 6;
            const int half = lane >> 4;
            const int half_lane = lane & 15;
            const int lane_q = lane;
            unsigned int q_stage = 0;
            unsigned int xb_q = 0;
            unsigned int xq_q = 0;
            unsigned int _phase_mma_done_1 = 1;
            unsigned int _phase_xb_full = 0;
            #pragma unroll 1
            for (int it_q = 0; it_q < n_items_q; it_q++) {
                int work_q = w0_q + it_q * ws_q;
                int tile_3 = work_q / split;
                int rank_3 = work_q - tile_3 * split;
                int k_begin_3 = num_k_iters * rank_3 / split;
                int k_end_3 = num_k_iters * (rank_3 + 1) / split;
                int k_count_3 = k_end_3 - k_begin_3;
                int m_tile_q = tile_3 / n_tiles;
                int tok0_q = m_tile_q * 64;
                #pragma unroll 1
                for (int i_2 = 0; i_2 < k_count_3; i_2++) {
                    mbarrier_wait(mma_done_addr + (q_stage) * 8, _phase_mma_done_1);
                    unsigned int x_stage = smem_x_addr + q_stage * 51200;
                    unsigned int sf_stage_off = q_stage * 51200;
                    const int grp = lane_q / 8;
                    const int sub = lane_q - grp * 8;
                    const int stag = (grp & 1) * 8;
                    if (qwarp >= 0 && qwarp < 16) {
                        mbarrier_wait(xb_full_addr + (xb_q) * 8, _phase_xb_full);
                        unsigned int xb_half = smem_xb0_addr + xb_q * 32768;
                        unsigned int panel_h = x_stage;
                        #pragma unroll
                        for (int itn = 0; itn < 1; itn++) {
                            int tok_h = (itn * 16 + qwarp) * 4 + grp;
                            int tok_gh = tok0_q + tok_h;
                            unsigned int row_h = xb_half + (unsigned int)(tok_h * 256);
                            unsigned int wordsh[8];
                            #pragma unroll
                            for (int ci = 0; ci < 2; ci++) {
                                int chunk_h = sub + 8 * ci + stag & 15;
                                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&wordsh[4 * ci])), "=r"(*reinterpret_cast<uint32_t*>(&wordsh[(4 * ci) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&wordsh[(4 * ci) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&wordsh[(4 * ci) + 3]))
                                    : "r"(row_h + (unsigned int)(chunk_h * 16)));
                            }
                            float amax_h = 0.0f;
                            #pragma unroll
                            for (int ci_1 = 0; ci_1 < 2; ci_1++) {
                                float mags_h[8];
                                #pragma unroll
                                for (int j = 0; j < 4; j++) {
                                    unsigned int lo_bh = wordsh[4 * ci_1 + j] << 16;
                                    unsigned int hi_bh = wordsh[4 * ci_1 + j] & 4294901760u;
                                    float lo_fh = 0.0f;
                                    float hi_fh = 0.0f;
                                    lo_fh = reinterpret_cast<float*>(&lo_bh)[0];
                                    hi_fh = reinterpret_cast<float*>(&hi_bh)[0];
                                    mags_h[2 * j] = lo_fh;
                                    mags_h[2 * j + 1] = hi_fh;
                                }
                                mags_h[0] = fabsf(mags_h[0]);
                                mags_h[1] = fabsf(mags_h[1]);
                                mags_h[2] = fabsf(mags_h[2]);
                                mags_h[3] = fabsf(mags_h[3]);
                                mags_h[4] = fabsf(mags_h[4]);
                                mags_h[5] = fabsf(mags_h[5]);
                                mags_h[6] = fabsf(mags_h[6]);
                                mags_h[7] = fabsf(mags_h[7]);
                                float _reg_array_reduce_0 = mags_h[0];
                                #pragma unroll
                                for (int _lr = 1; _lr < 8; _lr++) {
                                    _reg_array_reduce_0 = max_noftz(_reg_array_reduce_0, mags_h[_lr]);
                                }
                                amax_h = max_noftz(amax_h, _reg_array_reduce_0);
                            }
                            float o_h = __shfl_xor_sync(0xFFFFFFFF, amax_h, 1);
                            amax_h = max_noftz(amax_h, o_h);
                            float o_h_0 = __shfl_xor_sync(0xFFFFFFFF, amax_h, 2);
                            amax_h = max_noftz(amax_h, o_h_0);
                            float o_h_1 = __shfl_xor_sync(0xFFFFFFFF, amax_h, 4);
                            amax_h = max_noftz(amax_h, o_h_1);
                            amax_h = max_noftz(amax_h, 1e-04f);
                            unsigned int amax_bits_h = __as_u32(amax_h);
                            unsigned int sb_raw_h = (amax_bits_h + 2097151 >> 23) - 8;
                            unsigned int _max_1 = ((sb_raw_h) > (1) ? (sb_raw_h) : (1));
                            unsigned int scale_byte_h = ((_max_1) < (254) ? (_max_1) : (254));
                            unsigned int inverse_bits_h = 254 - scale_byte_h << 23;
                            float inverse_h = 0.0f;
                            inverse_h = reinterpret_cast<float*>(&inverse_bits_h)[0];
                            unsigned int sf4_h = scale_byte_h * 16843009;
                            float sf4_bits_h = 0.0f;
                            sf4_bits_h = reinterpret_cast<float*>(&sf4_h)[0];
                            unsigned int sf_off_h = sf_stage_off + (unsigned int)((tok_h & 31) * 16 + (tok_h >> 5) * 4);
                            #pragma unroll
                            for (int ci_2 = 0; ci_2 < 2; ci_2++) {
                                int chunk_sh = sub + 8 * ci_2 + stag & 15;
                                float vals_h[8];
                                #pragma unroll
                                for (int j_1 = 0; j_1 < 4; j_1++) {
                                    unsigned int lo_ch = wordsh[4 * ci_2 + j_1] << 16;
                                    unsigned int hi_ch = wordsh[4 * ci_2 + j_1] & 4294901760u;
                                    float lo_vh = 0.0f;
                                    float hi_vh = 0.0f;
                                    lo_vh = reinterpret_cast<float*>(&lo_ch)[0];
                                    hi_vh = reinterpret_cast<float*>(&hi_ch)[0];
                                    vals_h[2 * j_1] = lo_vh;
                                    vals_h[2 * j_1 + 1] = hi_vh;
                                }
                                const float2 _scale2_2 = {inverse_h, inverse_h};
                                #pragma unroll
                                for (int _ls = 0; _ls < 4; _ls++)
                                    mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(vals_h)[_ls], _scale2_2);
                                uint16_t eh0;
                                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(eh0) : "f"(vals_h[1]), "f"(vals_h[0]));
                                uint16_t eh1;
                                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(eh1) : "f"(vals_h[3]), "f"(vals_h[2]));
                                uint16_t eh2;
                                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(eh2) : "f"(vals_h[5]), "f"(vals_h[4]));
                                uint16_t eh3;
                                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(eh3) : "f"(vals_h[7]), "f"(vals_h[6]));
                                unsigned int wh0 = (unsigned int)eh0 | (unsigned int)eh1 << 16;
                                unsigned int wh1 = (unsigned int)eh2 | (unsigned int)eh3 << 16;
                                if (tok_gh < M) {
                                    asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"((panel_h + (unsigned int)(tok_h * 128 + chunk_sh * 8 ^ (tok_h * 128 + chunk_sh * 8 >> 7 & 7) << 4))), "r"(wh0), "r"(wh1) : "memory");
                                }
                            }
                            if (tok_gh < M) {
                                if (sub == 0) {
                                    {
                                        uint32_t _addr_3 = static_cast<uint32_t>(smem_sfx0_addr + sf_off_h);
                                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_3), "f"(sf4_bits_h) : "memory");
                                    }
                                }
                            }
                        }
                        __syncwarp();
                        if (elect_sync()) {
                            mbarrier_arrive(xb_empty_addr + (xb_q) * 8);
                        }
                    }
                    if (qwarp >= 0 && qwarp < 16) {
                        mbarrier_wait(xb_full1_addr + (xb_q) * 8, _phase_xb_full);
                        unsigned int xb_half_1 = smem_xb1_addr + xb_q * 32768;
                        unsigned int panel_h_1 = x_stage + 8192;
                        #pragma unroll
                        for (int itn_1 = 0; itn_1 < 1; itn_1++) {
                            int tok_h_1 = (itn_1 * 16 + qwarp) * 4 + grp;
                            int tok_gh_1 = tok0_q + tok_h_1;
                            unsigned int row_h_1 = xb_half_1 + (unsigned int)(tok_h_1 * 256);
                            unsigned int wordsh_1[8];
                            #pragma unroll
                            for (int ci_3 = 0; ci_3 < 2; ci_3++) {
                                int chunk_h_1 = sub + 8 * ci_3 + stag & 15;
                                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&wordsh_1[4 * ci_3])), "=r"(*reinterpret_cast<uint32_t*>(&wordsh_1[(4 * ci_3) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&wordsh_1[(4 * ci_3) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&wordsh_1[(4 * ci_3) + 3]))
                                    : "r"(row_h_1 + (unsigned int)(chunk_h_1 * 16)));
                            }
                            float amax_h_1 = 0.0f;
                            #pragma unroll
                            for (int ci_4 = 0; ci_4 < 2; ci_4++) {
                                float mags_h_1[8];
                                #pragma unroll
                                for (int j_2 = 0; j_2 < 4; j_2++) {
                                    unsigned int lo_bh_1 = wordsh_1[4 * ci_4 + j_2] << 16;
                                    unsigned int hi_bh_1 = wordsh_1[4 * ci_4 + j_2] & 4294901760u;
                                    float lo_fh_1 = 0.0f;
                                    float hi_fh_1 = 0.0f;
                                    lo_fh_1 = reinterpret_cast<float*>(&lo_bh_1)[0];
                                    hi_fh_1 = reinterpret_cast<float*>(&hi_bh_1)[0];
                                    mags_h_1[2 * j_2] = lo_fh_1;
                                    mags_h_1[2 * j_2 + 1] = hi_fh_1;
                                }
                                mags_h_1[0] = fabsf(mags_h_1[0]);
                                mags_h_1[1] = fabsf(mags_h_1[1]);
                                mags_h_1[2] = fabsf(mags_h_1[2]);
                                mags_h_1[3] = fabsf(mags_h_1[3]);
                                mags_h_1[4] = fabsf(mags_h_1[4]);
                                mags_h_1[5] = fabsf(mags_h_1[5]);
                                mags_h_1[6] = fabsf(mags_h_1[6]);
                                mags_h_1[7] = fabsf(mags_h_1[7]);
                                float _reg_array_reduce_4 = mags_h_1[0];
                                #pragma unroll
                                for (int _lr = 1; _lr < 8; _lr++) {
                                    _reg_array_reduce_4 = max_noftz(_reg_array_reduce_4, mags_h_1[_lr]);
                                }
                                amax_h_1 = max_noftz(amax_h_1, _reg_array_reduce_4);
                            }
                            float o_h_2 = __shfl_xor_sync(0xFFFFFFFF, amax_h_1, 1);
                            amax_h_1 = max_noftz(amax_h_1, o_h_2);
                            float o_h_0_1 = __shfl_xor_sync(0xFFFFFFFF, amax_h_1, 2);
                            amax_h_1 = max_noftz(amax_h_1, o_h_0_1);
                            float o_h_1_1 = __shfl_xor_sync(0xFFFFFFFF, amax_h_1, 4);
                            amax_h_1 = max_noftz(amax_h_1, o_h_1_1);
                            amax_h_1 = max_noftz(amax_h_1, 1e-04f);
                            unsigned int amax_bits_h_1 = __as_u32(amax_h_1);
                            unsigned int sb_raw_h_1 = (amax_bits_h_1 + 2097151 >> 23) - 8;
                            unsigned int _max_5 = ((sb_raw_h_1) > (1) ? (sb_raw_h_1) : (1));
                            unsigned int scale_byte_h_1 = ((_max_5) < (254) ? (_max_5) : (254));
                            unsigned int inverse_bits_h_1 = 254 - scale_byte_h_1 << 23;
                            float inverse_h_1 = 0.0f;
                            inverse_h_1 = reinterpret_cast<float*>(&inverse_bits_h_1)[0];
                            unsigned int sf4_h_1 = scale_byte_h_1 * 16843009;
                            float sf4_bits_h_1 = 0.0f;
                            sf4_bits_h_1 = reinterpret_cast<float*>(&sf4_h_1)[0];
                            unsigned int sf_off_h_1 = sf_stage_off + (unsigned int)((tok_h_1 & 31) * 16 + (tok_h_1 >> 5) * 4);
                            #pragma unroll
                            for (int ci_5 = 0; ci_5 < 2; ci_5++) {
                                int chunk_sh_1 = sub + 8 * ci_5 + stag & 15;
                                float vals_h_1[8];
                                #pragma unroll
                                for (int j_3 = 0; j_3 < 4; j_3++) {
                                    unsigned int lo_ch_1 = wordsh_1[4 * ci_5 + j_3] << 16;
                                    unsigned int hi_ch_1 = wordsh_1[4 * ci_5 + j_3] & 4294901760u;
                                    float lo_vh_1 = 0.0f;
                                    float hi_vh_1 = 0.0f;
                                    lo_vh_1 = reinterpret_cast<float*>(&lo_ch_1)[0];
                                    hi_vh_1 = reinterpret_cast<float*>(&hi_ch_1)[0];
                                    vals_h_1[2 * j_3] = lo_vh_1;
                                    vals_h_1[2 * j_3 + 1] = hi_vh_1;
                                }
                                const float2 _scale2_6 = {inverse_h_1, inverse_h_1};
                                #pragma unroll
                                for (int _ls = 0; _ls < 4; _ls++)
                                    mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(vals_h_1)[_ls], _scale2_6);
                                uint16_t eh0_1;
                                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(eh0_1) : "f"(vals_h_1[1]), "f"(vals_h_1[0]));
                                uint16_t eh1_1;
                                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(eh1_1) : "f"(vals_h_1[3]), "f"(vals_h_1[2]));
                                uint16_t eh2_1;
                                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(eh2_1) : "f"(vals_h_1[5]), "f"(vals_h_1[4]));
                                uint16_t eh3_1;
                                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(eh3_1) : "f"(vals_h_1[7]), "f"(vals_h_1[6]));
                                unsigned int wh0_1 = (unsigned int)eh0_1 | (unsigned int)eh1_1 << 16;
                                unsigned int wh1_1 = (unsigned int)eh2_1 | (unsigned int)eh3_1 << 16;
                                if (tok_gh_1 < M) {
                                    asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"((panel_h_1 + (unsigned int)(tok_h_1 * 128 + chunk_sh_1 * 8 ^ (tok_h_1 * 128 + chunk_sh_1 * 8 >> 7 & 7) << 4))), "r"(wh0_1), "r"(wh1_1) : "memory");
                                }
                            }
                            if (tok_gh_1 < M) {
                                if (sub == 0) {
                                    {
                                        uint32_t _addr_7 = static_cast<uint32_t>(smem_sfx1_addr + sf_off_h_1);
                                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_7), "f"(sf4_bits_h_1) : "memory");
                                    }
                                }
                            }
                        }
                        __syncwarp();
                        if (elect_sync()) {
                            mbarrier_arrive(xb_empty1_addr + (xb_q) * 8);
                        }
                    }
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    __syncwarp();
                    if (elect_sync()) {
                        mbarrier_arrive(tma_full_addr + (q_stage) * 8);
                    }
                    q_stage += 1;
                    if (q_stage == 2) { q_stage = 0; _phase_mma_done_1 ^= 1; }
                    xb_q += 1;
                    if (xb_q == 2) { xb_q = 0; _phase_xb_full ^= 1; }
                }
            }
        }
    }
    // ---- Role: xload ----
    if (warp == 22) {
        { // xload_main
            unsigned int _phase_xb_empty = 1;
            if (elect_sync()) {
                int w0_x = bid;
                int ws_x = num_bids;
                int n_items_x = (total_work - bid + num_bids - 1) / num_bids;
                unsigned int xb_stage_l = 0;
                #pragma unroll 1
                for (int it_x = 0; it_x < n_items_x; it_x++) {
                    int work_x = w0_x + it_x * ws_x;
                    int tile_4 = work_x / split;
                    int rank_4 = work_x - tile_4 * split;
                    int k_begin_4 = num_k_iters * rank_4 / split;
                    int k_end_4 = num_k_iters * (rank_4 + 1) / split;
                    int k_count_4 = k_end_4 - k_begin_4;
                    int m_tile_x = tile_4 / n_tiles;
                    int x_row_x = m_tile_x * 64;
                    #pragma unroll 1
                    for (int i_3 = 0; i_3 < k_count_4; i_3++) {
                        int k_group_x = (k_begin_4 + i_3) * 2;
                        mbarrier_wait(xb_empty_addr + (xb_stage_l) * 8, _phase_xb_empty);
                        if (it_x == 0 && i_3 == 0) {
                            asm volatile("griddepcontrol.wait;" ::: "memory");
                        }
                        asm volatile(
                            "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                            " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                            :: "r"(smem_xb0_addr + xb_stage_l * 32768), "l"((&XB)), "r"(0), "r"(x_row_x), "r"(k_group_x),
                               "r"(xb_full_addr + (xb_stage_l) * 8), "l"(0x12F0000000000000ULL) : "memory");
                        mbarrier_arrive_expect_tx(xb_full_addr + (xb_stage_l) * 8, 16384);
                        mbarrier_wait(xb_empty1_addr + (xb_stage_l) * 8, _phase_xb_empty);
                        asm volatile(
                            "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                            " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                            :: "r"(smem_xb1_addr + xb_stage_l * 32768), "l"((&XB)), "r"(0), "r"(x_row_x), "r"(k_group_x + 1),
                               "r"(xb_full1_addr + (xb_stage_l) * 8), "l"(0x12F0000000000000ULL) : "memory");
                        mbarrier_arrive_expect_tx(xb_full1_addr + (xb_stage_l) * 8, 16384);
                        xb_stage_l += 1;
                        if (xb_stage_l == 2) { xb_stage_l = 0; _phase_xb_empty ^= 1; }
                    }
                }
            }
        }
    }

    // Kernel teardown ops
    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");

    // Cleanup
}

} // extern "C"
