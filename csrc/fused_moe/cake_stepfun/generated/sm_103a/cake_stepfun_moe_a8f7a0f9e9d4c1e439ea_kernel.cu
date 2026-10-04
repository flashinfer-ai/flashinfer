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

#define CAKE_INF CUDART_INF_F
#define TMEM_NCOLS 8
#define TMEM_ACCUM_OFFSET 0
#define NUM_K_PIPE_STAGES 3
#define NUM_OUT_PIPE_STAGES 1
#define SMEM_SMEM_A_OFF 1024
#define SMEM_SMEM_A_STAGE_BYTES 32768
#define SMEM_SMEM_A_STRIDE 32768
#define SMEM_SMEM_B_OFF 99328
#define SMEM_SMEM_B_STAGE_BYTES 2048
#define SMEM_SMEM_B_STRIDE 2048
#define SMEM_EPI_STAGING_OFF 105472
#define SMEM_EPI_STAGING_STAGE_BYTES 512
#define SMEM_EPI_STAGING_STRIDE 512
#define SMEM_TOTAL 105984
#define THREADS 256
#define BLOCK_M 128
#define BLOCK_N 8
#define BLOCK_K 256
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


__device__ __forceinline__ void mma_ss_step(
    int a_lo, int b_lo, int taddr, uint32_t i_desc, int enable_d,
    uint32_t a_dhi, uint32_t b_dhi) {
    asm volatile(
        "{\n\t"
        ".reg .pred leader, p;\n\t"
        ".reg .b32 adhi, bdhi;\n\t"
        ".reg .b64 da, db;\n\t"
        "elect.sync _|leader, 0xFFFFFFFF;\n\t"
        "setp.ne.b32 p, %4, 0;\n\t"
        "mov.b32 adhi, %5;\n\t"
        "mov.b32 bdhi, %6;\n\t"
        "mov.b64 da, {%0, adhi};\n\t"
        "mov.b64 db, {%1, bdhi};\n\t"
        "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, %3, p;\n\t"
        "}\n"
        :: "r"(a_lo), "r"(b_lo), "r"(taddr), "r"(i_desc), "r"(enable_d), "r"(a_dhi), "r"(b_dhi));
}


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



__device__ __forceinline__ void mbarrier_arrive_expect_tx(int mbar_addr, uint32_t bytes) {
    asm volatile(
        "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
        :: "r"(mbar_addr), "r"(bytes) : "memory");
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


__device__ __forceinline__ float max_noftz(float a, float b) {
    float c;
    asm("max.f32 %0, %1, %2;" : "=f"(c) : "f"(a), "f"(b));
    return c;
}


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






__device__ __forceinline__ void tma_4d_gmem2smem(
    int dst, const void *tmap_ptr, int x, int y, int z, int w, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.4d.shared::cta.global"
        ".mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3, %4, %5}], [%6];"
        :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y), "r"(z), "r"(w),
           "r"(mbar_addr) : "memory");
}


__device__ __forceinline__ void tma_gather4_gmem2smem(
    int dst, const void *tmap_ptr,
    int col_idx, int row0, int row1, int row2, int row3,
    int mbar_addr) {
    // Canonical .shared::cta form for non-multicast gather4, matching
    // trtllm-gen / cuda_ptx and the PTX ISA qualifier order
    // (dim.dst.src.load_mode.completion_mechanism). Per the PTX grammar,
    // .shared::cluster is reserved for the multicast variant (ctaMask).
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
        ".mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3, %4, %5, %6}], [%7];"
        :: "r"(dst), "l"(tmap_ptr), "r"(col_idx),
           "r"(row0), "r"(row1), "r"(row2), "r"(row3),
           "r"(mbar_addr) : "memory");
}


__device__ __forceinline__ void tma_store_4d(
    const void *tmap, int x, int y, int z, int w, unsigned smem_addr) {
    asm volatile(
        "cp.async.bulk.tensor.4d.global.shared::cta.tile.bulk_group"
        " [%0, {%1, %2, %3, %4}], [%5];"
        :: "l"(tmap), "r"(x), "r"(y), "r"(z), "r"(w), "r"(smem_addr) : "memory");
}





__device__ __forceinline__ uint32_t make_warp_uniform(uint32_t val) {
    uint32_t result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1f, 0xffffffff;"
        : "=r"(result) : "r"(val));
    return result;
}

extern "C" {

__global__ __launch_bounds__(256, LAUNCH_MIN_BLOCKS) void
kernel_cake_stepfun_moe_a8f7a0f9e9d4c1e439ea(const __grid_constant__ CUtensorMap A, const __grid_constant__ CUtensorMap B, const __grid_constant__ CUtensorMap C, int* __restrict__ route_map, int* __restrict__ tile_expert, int* __restrict__ tile_mn_limit, int* __restrict__ num_non_exiting_ctas, float* __restrict__ scale_c, float* __restrict__ scale_gate, float* __restrict__ clamp_limit, float* __restrict__ act_alpha, float* __restrict__ act_beta, int M_out, int K, int grid_m, int grid_n, int K_tiles)
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
    #define b_full_addr (mbar_base + 24)
    #define k_done_addr (mbar_base + 48)
    #define mma_full_addr (mbar_base + 72)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    uint8_t* smem_a = reinterpret_cast<uint8_t*>(smem_raw + 1024);
    const int smem_a_addr = smem + 1024;
    uint8_t* smem_b = reinterpret_cast<uint8_t*>(smem_raw + 99328);
    const int smem_b_addr = smem + 99328;
    uint16_t* epi_staging = reinterpret_cast<uint16_t*>(smem_raw + 105472);
    const int epi_staging_addr = smem + 105472;
    asm volatile("griddepcontrol.wait;" ::: "memory");
    if ((int)blockIdx.y >= num_non_exiting_ctas[0]) return;

    // Mbarrier init (4 pipeline groups, 0 ordered-sequence groups, 10 barriers)
    // Mbarriers at smem_raw[0..80)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'k_pipe' ---
            // a_full: 3 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            mbarrier_init(smem + 8, 1);
            mbarrier_init(smem + 16, 1);
            // b_full: 3 barriers, init_count=1
            mbarrier_init(smem + 24, 1);
            mbarrier_init(smem + 32, 1);
            mbarrier_init(smem + 40, 1);
            // k_done: 3 barriers, init_count=1
            mbarrier_init(smem + 48, 1);
            mbarrier_init(smem + 56, 1);
            mbarrier_init(smem + 64, 1);
            // --- pipeline 'out_pipe' ---
            // mma_full: 1 barriers, init_count=1
            mbarrier_init(smem + 72, 1);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    // TMEM alloc (32 columns, 8 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 80);
    if (warp == 0) {
        int _tmem_hold = smem + 80;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(32) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_accum = taddr;

    // ---- Role: epilogue ----
    if (warp <= 3) {
        { // epilogue_main
            asm volatile("griddepcontrol.wait;" ::: "memory");
            const int warp_0 = warp;
            const int lane_1 = lane;
            int n_tile = blockIdx.y;
            int m_tile = blockIdx.x;
            int expert = tile_expert[n_tile];
            int valid_rows = tile_mn_limit[n_tile] - n_tile * BLOCK_N;
            float sc = scale_c[expert];
            float sg = scale_gate[expert];
            float cl = clamp_limit[expert];
            float neg_cl = -cl;
            float quad[4] = {0};
            float fused = 1.4426950216293335f * sg;
            unsigned int _phase_mma_full_0 = 0;
            mbarrier_wait(mma_full_addr, _phase_mma_full_0);
            _phase_mma_full_0 ^= 1;
            asm volatile("tcgen05.fence::after_thread_sync;");
            float _tmem_load_0[4];
            asm volatile(
                "tcgen05.ld.sync.aligned.16x256b.x1.b32"
                " {%0, %1, %2, %3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[3]))
                : "r"(taddr));
            asm volatile("tcgen05.wait::ld.sync.aligned;");
            float _tmem_load_1[4];
            asm volatile(
                "tcgen05.ld.sync.aligned.16x256b.x1.b32"
                " {%0, %1, %2, %3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[3]))
                : "r"(taddr + 1048576));
            asm volatile("tcgen05.wait::ld.sync.aligned;");
            int base_row = warp_0 * 16 + lane_1 / 4 * 2;
            asm volatile("cp.async.bulk.wait_group.read 0;");
            asm volatile("barrier.sync 7, 128;" ::: "memory");
            for (int token_group = 0; token_group < 1; token_group++) {
                int token0 = lane_1 % 4 * 2 + token_group * 8;
                int token1 = token0 + 1;
                float x0_00 = _tmem_load_0[token_group * 4];
                float x0_01 = _tmem_load_0[token_group * 4 + 1];
                float x1_00 = _tmem_load_0[token_group * 4 + 2];
                float x1_01 = _tmem_load_0[token_group * 4 + 3];
                float x0_10 = _tmem_load_1[token_group * 4];
                float x0_11 = _tmem_load_1[token_group * 4 + 1];
                float x1_10 = _tmem_load_1[token_group * 4 + 2];
                float x1_11 = _tmem_load_1[token_group * 4 + 3];
                float _max_0 = max_noftz(x0_00, neg_cl);
                float _min_0 = fminf(_max_0, cl);
                float x0c_00 = _min_0;
                float _max_1 = max_noftz(x0_01, neg_cl);
                float _min_1 = fminf(_max_1, cl);
                float x0c_01 = _min_1;
                float _max_2 = max_noftz(x0_10, neg_cl);
                float _min_2 = fminf(_max_2, cl);
                float x0c_10 = _min_2;
                float _max_3 = max_noftz(x0_11, neg_cl);
                float _min_3 = fminf(_max_3, cl);
                float x0c_11 = _min_3;
                float x0s_00 = x0c_00 * sc;
                float x0s_01 = x0c_01 * sc;
                float x0s_10 = x0c_10 * sc;
                float x0s_11 = x0c_11 * sc;
                float lin_00 = x0s_00 * sg;
                float lin_01 = x0s_01 * sg;
                float lin_10 = x0s_10 * sg;
                float lin_11 = x0s_11 * sg;
                float _exp2_0 = approx_exp2(-(x1_00 * fused));
                float _rcp_0 = approx_rcp(1.0f + _exp2_0);
                float sig_00 = _rcp_0;
                float _exp2_1 = approx_exp2(-(x1_01 * fused));
                float _rcp_1 = approx_rcp(1.0f + _exp2_1);
                float sig_01 = _rcp_1;
                float _exp2_2 = approx_exp2(-(x1_10 * fused));
                float _rcp_2 = approx_rcp(1.0f + _exp2_2);
                float sig_10 = _rcp_2;
                float _exp2_3 = approx_exp2(-(x1_11 * fused));
                float _rcp_3 = approx_rcp(1.0f + _exp2_3);
                float sig_11 = _rcp_3;
                float act_00 = x1_00 * sig_00;
                float act_01 = x1_01 * sig_01;
                float act_10 = x1_10 * sig_10;
                float act_11 = x1_11 * sig_11;
                float _min_4 = fminf(act_00, cl);
                act_00 = _min_4;
                float _min_5 = fminf(act_01, cl);
                act_01 = _min_5;
                float _min_6 = fminf(act_10, cl);
                act_10 = _min_6;
                float _min_7 = fminf(act_11, cl);
                act_11 = _min_7;
                quad[0] = lin_00 * act_00;
                quad[1] = lin_10 * act_10;
                quad[2] = lin_01 * act_01;
                quad[3] = lin_11 * act_11;
                uint32_t _fp8_0[1];
                {
                    uint32_t _packed;
                    asm volatile("{\n\t"
                        ".reg .b16 _lo;\n\t"
                        ".reg .b16 _hi;\n\t"
                        "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                        "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                        "mov.b32 %0, {_lo, _hi};\n\t"
                        "}"
                        : "=r"(_packed) : "f"(quad[0]), "f"(quad[1]),
                                           "f"(quad[2]), "f"(quad[3]));
                    _fp8_0[0] = _packed;
                }
                int off0 = token0 * 64 + base_row;
                int off1 = token1 * 64 + base_row;
                int swz0 = off0 ^ (off0 >> 7 & 3) << 4;
                int swz1 = off1 ^ (off1 >> 7 & 3) << 4;
                epi_staging[swz0 >> 1] = _fp8_0[0] & 65535;
                epi_staging[swz1 >> 1] = _fp8_0[0] >> 16;
            }
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            asm volatile("barrier.sync 7, 128;" ::: "memory");
            if (warp == 0) {
                if (elect_sync()) {
                    int padding_rows = (8 - valid_rows % 8) % 8;
                    tma_store_4d((&C), m_tile * 64, padding_rows, 1073741824, n_tile * 8 - padding_rows + 1073741824, epi_staging_addr);
                }
            }
            asm volatile("cp.async.bulk.commit_group;");
            asm volatile("barrier.sync 7, 128;" ::: "memory");
        }
    }
    // ---- Role: load_b ----
    if (warp == 4) {
        { // load_b_main
            asm volatile("griddepcontrol.wait;" ::: "memory");
            unsigned int stage = 0;
            int n_tile_1 = blockIdx.y;
            int routed[8] = {0};
            for (int slot = 0; slot < 8; slot++) {
                routed[slot] = route_map[n_tile_1 * 8 + slot];
            }
            unsigned int _phase_k_done = 1;
            #pragma unroll 1
            for (int iter_k = 0; iter_k < K_tiles; iter_k++) {
                mbarrier_wait(k_done_addr + (stage) * 8, _phase_k_done);
                int dst_base = smem_b_addr + stage * 2048;
                if (elect_sync()) {
                    tma_gather4_gmem2smem(dst_base, (&B), iter_k * 256, routed[0], routed[1], routed[2], routed[3], b_full_addr + (stage) * 8);
                    tma_gather4_gmem2smem(dst_base + 512, (&B), iter_k * 256, routed[4], routed[5], routed[6], routed[7], b_full_addr + (stage) * 8);
                    tma_gather4_gmem2smem(dst_base + 1024, (&B), iter_k * 256 + 128, routed[0], routed[1], routed[2], routed[3], b_full_addr + (stage) * 8);
                    tma_gather4_gmem2smem(dst_base + 1024 + 512, (&B), iter_k * 256 + 128, routed[4], routed[5], routed[6], routed[7], b_full_addr + (stage) * 8);
                    mbarrier_arrive_expect_tx(b_full_addr + (stage) * 8, 2048);
                }
                stage += 1;
                if (stage == 3) { stage = 0; _phase_k_done ^= 1; }
            }
            asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
        }
    }
    // ---- Role: load_a ----
    if (warp == 5) {
        { // load_a_main
            asm volatile("griddepcontrol.wait;" ::: "memory");
            unsigned int stage_1 = 0;
            int m_tile_1 = blockIdx.x;
            int n_tile_2 = blockIdx.y;
            int expert_1 = tile_expert[n_tile_2];
            int sole_reader = 1;
            int bound = num_non_exiting_ctas[0];
            if (bound > n_tile_2 + 1) {
                if (tile_expert[n_tile_2 + 1] == expert_1) {
                    sole_reader = 0;
                }
            }
            if (n_tile_2 > 0) {
                if (tile_expert[n_tile_2 - 1] == expert_1) {
                    sole_reader = 0;
                }
            }
            unsigned int _phase_k_done_1 = 1;
            #pragma unroll 1
            for (int iter_k_1 = 0; iter_k_1 < K_tiles; iter_k_1++) {
                mbarrier_wait(k_done_addr + (stage_1) * 8, _phase_k_done_1);
                if (elect_sync()) {
                    if (sole_reader == 1) {
                        asm volatile(
                            "cp.async.bulk.tensor.4d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                            " [%0], [%1, {%2, %3, %4, %5}], [%6], %7;"
                            :: "r"(smem_a_addr + stage_1 * 32768), "l"((&A)), "r"(0), "r"(m_tile_1 * 128), "r"(iter_k_1 * 2), "r"(expert_1),
                               "r"(a_full_addr + (stage_1) * 8), "l"(0x12F0000000000000ULL) : "memory");
                    } else {
                        tma_4d_gmem2smem(smem_a_addr + stage_1 * 32768, (&A), 0, m_tile_1 * 128, iter_k_1 * 2, expert_1, a_full_addr + (stage_1) * 8);
                    }
                    if (sole_reader == 1) {
                        asm volatile(
                            "cp.async.bulk.tensor.4d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                            " [%0], [%1, {%2, %3, %4, %5}], [%6], %7;"
                            :: "r"(smem_a_addr + stage_1 * 32768 + 16384), "l"((&A)), "r"(0), "r"(m_tile_1 * 128), "r"(iter_k_1 * 2 + 1), "r"(expert_1),
                               "r"(a_full_addr + (stage_1) * 8), "l"(0x12F0000000000000ULL) : "memory");
                    } else {
                        tma_4d_gmem2smem(smem_a_addr + stage_1 * 32768 + 16384, (&A), 0, m_tile_1 * 128, iter_k_1 * 2 + 1, expert_1, a_full_addr + (stage_1) * 8);
                    }
                    mbarrier_arrive_expect_tx(a_full_addr + (stage_1) * 8, 32768);
                }
                stage_1 += 1;
                if (stage_1 == 3) { stage_1 = 0; _phase_k_done_1 ^= 1; }
            }
        }
    }
    // ---- Role: mma ----
    if (warp == 6) {
        { // mma_main
            unsigned int stage_2 = 0;
            unsigned int _phase_a_full = 0;
            unsigned int _phase_b_full = 0;
            #pragma unroll 1
            for (int iter_k_2 = 0; iter_k_2 < K_tiles; iter_k_2++) {
                mbarrier_wait(a_full_addr + (stage_2) * 8, _phase_a_full);
                mbarrier_wait(b_full_addr + (stage_2) * 8, _phase_b_full);
                asm volatile("tcgen05.fence::after_thread_sync;");
                int _mma_a_lo_0 = make_warp_uniform((((smem_a_addr) >> 4) & 0x3FFF) + (stage_2) * 2048);
                int _mma_b_lo_0 = make_warp_uniform((((smem_b_addr) >> 4) & 0x3FFF) + (stage_2) * 128);
                mma_ss_step(_mma_a_lo_0, _mma_b_lo_0, tmem_accum, 134348816, ((((1) ? ((iter_k_2 == 0) ? 1 : 0) : 0)) ? 0 : 1), 0x40004040U, 0x40004040U);
                int _mma_a_lo_1 = make_warp_uniform((((smem_a_addr + 32) >> 4) & 0x3FFF) + (stage_2) * 2048);
                int _mma_b_lo_1 = make_warp_uniform((((smem_b_addr + 32) >> 4) & 0x3FFF) + (stage_2) * 128);
                mma_ss_step(_mma_a_lo_1, _mma_b_lo_1, tmem_accum, 134348816, ((((0) ? ((iter_k_2 == 0) ? 1 : 0) : 0)) ? 0 : 1), 0x40004040U, 0x40004040U);
                int _mma_a_lo_2 = make_warp_uniform((((smem_a_addr + 64) >> 4) & 0x3FFF) + (stage_2) * 2048);
                int _mma_b_lo_2 = make_warp_uniform((((smem_b_addr + 64) >> 4) & 0x3FFF) + (stage_2) * 128);
                mma_ss_step(_mma_a_lo_2, _mma_b_lo_2, tmem_accum, 134348816, ((((0) ? ((iter_k_2 == 0) ? 1 : 0) : 0)) ? 0 : 1), 0x40004040U, 0x40004040U);
                int _mma_a_lo_3 = make_warp_uniform((((smem_a_addr + 96) >> 4) & 0x3FFF) + (stage_2) * 2048);
                int _mma_b_lo_3 = make_warp_uniform((((smem_b_addr + 96) >> 4) & 0x3FFF) + (stage_2) * 128);
                mma_ss_step(_mma_a_lo_3, _mma_b_lo_3, tmem_accum, 134348816, ((((0) ? ((iter_k_2 == 0) ? 1 : 0) : 0)) ? 0 : 1), 0x40004040U, 0x40004040U);
                int _mma_a_lo_4 = make_warp_uniform((((smem_a_addr + 16384) >> 4) & 0x3FFF) + (stage_2) * 2048);
                int _mma_b_lo_4 = make_warp_uniform((((smem_b_addr + 1024) >> 4) & 0x3FFF) + (stage_2) * 128);
                mma_ss_step(_mma_a_lo_4, _mma_b_lo_4, tmem_accum, 134348816, ((((0) ? ((iter_k_2 == 0) ? 1 : 0) : 0)) ? 0 : 1), 0x40004040U, 0x40004040U);
                int _mma_a_lo_5 = make_warp_uniform((((smem_a_addr + 16416) >> 4) & 0x3FFF) + (stage_2) * 2048);
                int _mma_b_lo_5 = make_warp_uniform((((smem_b_addr + 1056) >> 4) & 0x3FFF) + (stage_2) * 128);
                mma_ss_step(_mma_a_lo_5, _mma_b_lo_5, tmem_accum, 134348816, ((((0) ? ((iter_k_2 == 0) ? 1 : 0) : 0)) ? 0 : 1), 0x40004040U, 0x40004040U);
                int _mma_a_lo_6 = make_warp_uniform((((smem_a_addr + 16448) >> 4) & 0x3FFF) + (stage_2) * 2048);
                int _mma_b_lo_6 = make_warp_uniform((((smem_b_addr + 1088) >> 4) & 0x3FFF) + (stage_2) * 128);
                mma_ss_step(_mma_a_lo_6, _mma_b_lo_6, tmem_accum, 134348816, ((((0) ? ((iter_k_2 == 0) ? 1 : 0) : 0)) ? 0 : 1), 0x40004040U, 0x40004040U);
                int _mma_a_lo_7 = make_warp_uniform((((smem_a_addr + 16480) >> 4) & 0x3FFF) + (stage_2) * 2048);
                int _mma_b_lo_7 = make_warp_uniform((((smem_b_addr + 1120) >> 4) & 0x3FFF) + (stage_2) * 128);
                mma_ss_step(_mma_a_lo_7, _mma_b_lo_7, tmem_accum, 134348816, ((((0) ? ((iter_k_2 == 0) ? 1 : 0) : 0)) ? 0 : 1), 0x40004040U, 0x40004040U);
                if (iter_k_2 + 1 == K_tiles) {
                    elect_commit2(k_done_addr + (stage_2) * 8, mma_full_addr);
                } else {
                    elect_commit(k_done_addr + (stage_2) * 8);
                }
                stage_2 += 1;
                if (stage_2 == 3) { stage_2 = 0; _phase_a_full ^= 1; _phase_b_full ^= 1; }
            }
        }
    }
    // ---- Role: padding ----
    if (warp == 7) {
        // idle — no tasks assigned
    }

    // Cleanup
    __syncthreads(); // barrier before TMEM dealloc

    if (warp == 0) {
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(32));
    }
}

} // extern "C"
