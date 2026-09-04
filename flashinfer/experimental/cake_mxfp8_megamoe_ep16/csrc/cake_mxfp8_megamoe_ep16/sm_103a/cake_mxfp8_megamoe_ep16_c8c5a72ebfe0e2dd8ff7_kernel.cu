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
struct __align__(128) CakeTensorMap { uint64_t opaque[16]; };
template <int N>
struct __align__(128) CakeTensorMapPack { CakeTensorMap maps[N]; };

#if defined(__CUDACC_RTC__)
typedef struct __align__(128) { uint64_t opaque[16]; } CUtensorMap;
#else
#include <cuda.h>
#endif

static_assert(sizeof(CUtensorMap) == 128, "CUtensorMap CUDA ABI must be 128 bytes");
static_assert(alignof(CUtensorMap) == 128, "CUtensorMap CUDA ABI must be 128-byte aligned");
#include <cuda_bf16.h>
#include <cuda_fp8.h>

__device__ __forceinline__ int make_warp_uniform(int x) {
    int result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1F, 0xFFFFFFFF;"
                 : "=r"(result) : "r"(x));
    return result;
}

#define CAKE_INF CUDART_INF_F
#define TMEM_NCOLS 76
#define TMEM_FC1__TMEM_ACC_OFFSET 0
#define TMEM_FC1__TMEM_SFA_OFFSET 64
#define TMEM_FC1__TMEM_SFB_OFFSET 72
#define TMEM_FC2__TMEM_ACC_OFFSET 0
#define TMEM_FC2__TMEM_SFA_OFFSET 32
#define TMEM_FC2__TMEM_SFB_OFFSET 36
#define NUM_FC1__TMA_PIPE_STAGES 4
#define NUM_FC1__MAINLOOP_PIPE_STAGES 1
#define NUM_FC2__TMA_PIPE_STAGES 8
#define NUM_FC2__MAINLOOP_PIPE_STAGES 1
#define SMEM_DISPATCH_ROW_OFF 1024
#define SMEM_DISPATCH_ROW_STAGE_BYTES 3072
#define SMEM_DISPATCH_ROW_STRIDE 3072
#define SMEM_DISPATCH_SCALE_ROW_OFF 4096
#define SMEM_DISPATCH_SCALE_ROW_STAGE_BYTES 96
#define SMEM_DISPATCH_SCALE_ROW_STRIDE 96
#define SMEM_DISPATCH_WEIGHT_ROW_OFF 4192
#define SMEM_DISPATCH_WEIGHT_ROW_STAGE_BYTES 32
#define SMEM_DISPATCH_WEIGHT_ROW_STRIDE 32
#define SMEM_FC1__SMEM_A_OFF 1024
#define SMEM_FC1__SMEM_A_STAGE_BYTES 32768
#define SMEM_FC1__SMEM_A_STRIDE 38400
#define SMEM_FC1__SMEM_B_OFF 33792
#define SMEM_FC1__SMEM_B_STAGE_BYTES 4096
#define SMEM_FC1__SMEM_B_STRIDE 38400
#define SMEM_FC1__SMEM_SFA_CP_OFF 37888
#define SMEM_FC1__SMEM_SFA_CP_STAGE_BYTES 1024
#define SMEM_FC1__SMEM_SFA_CP_STRIDE 38400
#define SMEM_FC1__SMEM_SFB_CP_OFF 38912
#define SMEM_FC1__SMEM_SFB_CP_STAGE_BYTES 512
#define SMEM_FC1__SMEM_SFB_CP_STRIDE 38400
#define SMEM_FC1__EPI_STAGING_OFF 154624
#define SMEM_FC1__EPI_STAGING_STAGE_BYTES 16384
#define SMEM_FC1__EPI_STAGING_STRIDE 16384
#define SMEM_FC2__SMEM_A_OFF 1024
#define SMEM_FC2__SMEM_A_STAGE_BYTES 16384
#define SMEM_FC2__SMEM_A_STRIDE 21504
#define SMEM_FC2__SMEM_B_OFF 17408
#define SMEM_FC2__SMEM_B_STAGE_BYTES 4096
#define SMEM_FC2__SMEM_B_STRIDE 21504
#define SMEM_FC2__SMEM_SFA_CP_OFF 21504
#define SMEM_FC2__SMEM_SFA_CP_STAGE_BYTES 512
#define SMEM_FC2__SMEM_SFA_CP_STRIDE 21504
#define SMEM_FC2__SMEM_SFB_CP_OFF 22016
#define SMEM_FC2__SMEM_SFB_CP_STAGE_BYTES 512
#define SMEM_FC2__SMEM_SFB_CP_STRIDE 21504
#define SMEM_FC2__EPI_STAGING_OFF 173056
#define SMEM_FC2__EPI_STAGING_STAGE_BYTES 8192
#define SMEM_FC2__EPI_STAGING_STRIDE 8192
#define SMEM_TOTAL 181248
#define THREADS 256
#define fc1_num_tiles fc1_grid_m
#define fc2_num_tiles fc2_grid_m

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

__device__ __forceinline__ uint32_t mbarrier_try_wait_cluster(int mbar_addr, int phase) {
    uint32_t token;
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64"
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

__device__ __forceinline__ void mbarrier_wait_cluster(int mbar_addr, int phase) {
    uint32_t ticks = 0x989680;
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "LAB_WAIT_CLUSTER:\n\t"
        "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64"
        " P1, [%0], %1, %2;\n\t"
        "@P1 bra.uni DONE_CLUSTER;\n\t"
        "bra.uni LAB_WAIT_CLUSTER;\n\t"
        "DONE_CLUSTER:\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"(phase), "r"(ticks) : "memory");
}

__device__ __forceinline__ void mbarrier_wait_token(int mbar_addr, int phase, uint32_t token) {
    if (token == 0) {
        mbarrier_wait(mbar_addr, phase);
    }
}

__device__ __forceinline__ void mbarrier_wait_token_cluster(int mbar_addr, int phase, uint32_t token) {
    if (token == 0) {
        mbarrier_wait_cluster(mbar_addr, phase);
    }
}


__device__ __forceinline__ void tcgen05_mma_mxf8_bs(
    int taddr, uint64_t a_desc, uint64_t b_desc, uint32_t i_desc,
    int sfa_taddr, int sfb_taddr, int enable_input_d) {
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, %6, 0;\n\t"
        "tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale"
        " [%0], %1, %2, %3, [%4], [%5], p;\n\t"
        "}\n"
        :: "r"(taddr), "l"(a_desc), "l"(b_desc),
           "r"(i_desc), "r"(sfa_taddr), "r"(sfb_taddr),
           "r"(enable_input_d));
}


__device__ __forceinline__ uint64_t desc_encode(uint64_t x) {
    return (x & 0x3FFFFULL) >> 4ULL;
}


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
        "@leader tcgen05.mma.cta_group::1.kind::mxf8f6f4 [%2], da, db, %3, p;\n\t"
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


__device__ __forceinline__ void fence_async_shared() {
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}


__device__ __forceinline__ uint64_t make_smem_desc(int addr) {
    const int SBO = 1024;
    return desc_encode(addr)
         | (desc_encode(SBO) << 32ULL)
         | (1ULL << 46ULL)
         | (2ULL << 61ULL);
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


__device__ __forceinline__ void tcgen05_commit(int mbar_addr) {
    asm volatile(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one"
        ".shared::cluster.b64 [%0];"
        :: "r"(mbar_addr) : "memory");
}


__device__ __forceinline__ void tmem_ld_x8(float* dst, int tmem_addr) {
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x8.b32"
        " {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
        : "=f"(dst[0]), "=f"(dst[1]), "=f"(dst[2]), "=f"(dst[3]),
          "=f"(dst[4]), "=f"(dst[5]), "=f"(dst[6]), "=f"(dst[7])
        : "r"(tmem_addr));
}


__device__ __forceinline__ void tmem_ld_x8_wait(float* dst, int addr) {
    tmem_ld_x8(dst, addr);
    asm volatile("tcgen05.wait::ld.sync.aligned;");
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

__global__ __launch_bounds__(256) void
kernel_cake_mxfp8_megamoe_ep16_c8c5a72ebfe0e2dd8ff7(CakeTensorMap const* A1, CakeTensorMap const* B1, CakeTensorMap const* SFA1, CakeTensorMap const* SFB1, uint8_t* __restrict__ C_fp8, uint8_t* __restrict__ SF_out, uint8_t* __restrict__ B1_raw, uint8_t* __restrict__ SFB1_raw, int* __restrict__ route_map, float* __restrict__ route_weights, int* __restrict__ dispatch_work_source_ranks, int* __restrict__ dispatch_work_source_tokens, int* __restrict__ dispatch_work_local_groups, int* __restrict__ dispatch_work_local_rows, int* __restrict__ fc1_tile_expert, int* __restrict__ fc1_tile_m_local, int* __restrict__ expert_row_offsets, float* __restrict__ scale_gate, float* __restrict__ clamp_limit, float* __restrict__ act_alpha, float* __restrict__ act_beta, CakeTensorMap const* A2, CakeTensorMap const* B2, CakeTensorMap const* SFA2, CakeTensorMap const* SFB2, __nv_bfloat16* __restrict__ C, int* __restrict__ fc2_tile_expert, int* __restrict__ fc2_tile_m_local, unsigned int* __restrict__ fc1_done, int epoch, int routes_per_rank, int dispatch_work_count, int tokens_per_rank, int N1, int K1_tiles, int fc1_grid_m, int fc1_grid_n, int rows_per_expert, int N2, int K2_tiles, int fc2_grid_m, int32_t dispatch_pg_world, int32_t dispatch_pg_rank, unsigned* const* __restrict__ dispatch_pg_flags, uint8_t* __restrict__ dispatch_hidden_states_mxfp8, uint8_t* const* __restrict__ dispatch_hidden_states_mxfp8_peers, uint8_t* __restrict__ dispatch_hidden_scales_mxfp8, uint8_t* const* __restrict__ dispatch_hidden_scales_mxfp8_peers, float* __restrict__ dispatch_topk_weights, float* const* __restrict__ dispatch_topk_weights_peers, unsigned int* __restrict__ dispatch_input_ready, unsigned int* const* __restrict__ dispatch_input_ready_peers, int32_t remote_ready_pg_world, int32_t remote_ready_pg_rank, unsigned* const* __restrict__ remote_ready_pg_flags, unsigned int* __restrict__ remote_ready, unsigned int* const* __restrict__ remote_ready_peers, int32_t direct_output_pg_world, int32_t direct_output_pg_rank, unsigned* const* __restrict__ direct_output_pg_flags, __nv_bfloat16* __restrict__ direct_output, __nv_bfloat16* const* __restrict__ direct_output_peers)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int mbar_base = smem;
    #define dispatch_full_addr (mbar_base + 0)
    #define fc1__tma_full_addr (mbar_base + 8)
    #define fc1__mma_done_addr (mbar_base + 40)
    #define fc1__mainloop_done_addr (mbar_base + 72)
    #define fc1__epilogue_done_addr (mbar_base + 80)
    #define fc2__tma_full_addr (mbar_base + 88)
    #define fc2__mma_done_addr (mbar_base + 152)
    #define fc2__mainloop_done_addr (mbar_base + 216)
    #define fc2__epilogue_done_addr (mbar_base + 224)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    if (tid == 0) {
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(A1)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(B1)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(SFA1)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(SFB1)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(A2)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(B2)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(SFA2)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(SFB2)) : "memory");
    }
    __syncthreads();


    // Kernel setup ops
    uint8_t* dispatch_row = reinterpret_cast<uint8_t*>(smem_raw + 1024);
    const int dispatch_row_addr = smem + 1024;
    uint8_t* dispatch_scale_row = reinterpret_cast<uint8_t*>(smem_raw + 4096);
    const int dispatch_scale_row_addr = smem + 4096;
    float* dispatch_weight_row = reinterpret_cast<float*>(smem_raw + 4192);
    const int dispatch_weight_row_addr = smem + 4192;
    uint8_t* fc1__smem_a = reinterpret_cast<uint8_t*>(smem_raw + 1024);
    const int fc1__smem_a_addr = smem + 1024;
    uint8_t* fc1__smem_b = reinterpret_cast<uint8_t*>(smem_raw + 33792);
    const int fc1__smem_b_addr = smem + 33792;
    uint8_t* fc1__smem_sfa_cp = reinterpret_cast<uint8_t*>(smem_raw + 37888);
    const int fc1__smem_sfa_cp_addr = smem + 37888;
    uint8_t* fc1__smem_sfb_cp = reinterpret_cast<uint8_t*>(smem_raw + 38912);
    const int fc1__smem_sfb_cp_addr = smem + 38912;
    float* fc1__epi_staging = reinterpret_cast<float*>(smem_raw + 154624);
    const int fc1__epi_staging_addr = smem + 154624;
    uint8_t* fc2__smem_a = reinterpret_cast<uint8_t*>(smem_raw + 1024);
    const int fc2__smem_a_addr = smem + 1024;
    uint8_t* fc2__smem_b = reinterpret_cast<uint8_t*>(smem_raw + 17408);
    const int fc2__smem_b_addr = smem + 17408;
    uint8_t* fc2__smem_sfa_cp = reinterpret_cast<uint8_t*>(smem_raw + 21504);
    const int fc2__smem_sfa_cp_addr = smem + 21504;
    uint8_t* fc2__smem_sfb_cp = reinterpret_cast<uint8_t*>(smem_raw + 22016);
    const int fc2__smem_sfb_cp_addr = smem + 22016;
    __nv_bfloat16* fc2__epi_staging = reinterpret_cast<__nv_bfloat16*>(smem_raw + 173056);
    const int fc2__epi_staging_addr = smem + 173056;

    // Mbarrier init (9 groups, 29 barriers)
    // Mbarriers at smem_raw[0..232)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // dispatch_full: 1 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            // --- pipeline 'fc1__tma_pipe' ---
            // fc1__tma_full: 4 barriers, init_count=1
            mbarrier_init(smem + 8, 1);
            mbarrier_init(smem + 16, 1);
            mbarrier_init(smem + 24, 1);
            mbarrier_init(smem + 32, 1);
            // fc1__mma_done: 4 barriers, init_count=1
            mbarrier_init(smem + 40, 1);
            mbarrier_init(smem + 48, 1);
            mbarrier_init(smem + 56, 1);
            mbarrier_init(smem + 64, 1);
            // --- pipeline 'fc1__mainloop_pipe' ---
            // fc1__mainloop_done: 1 barriers, init_count=1
            mbarrier_init(smem + 72, 1);
            // fc1__epilogue_done: 1 barriers, init_count=4
            mbarrier_init(smem + 80, 4);
            // --- pipeline 'fc2__tma_pipe' ---
            // fc2__tma_full: 8 barriers, init_count=1
            mbarrier_init(smem + 88, 1);
            mbarrier_init(smem + 96, 1);
            mbarrier_init(smem + 104, 1);
            mbarrier_init(smem + 112, 1);
            mbarrier_init(smem + 120, 1);
            mbarrier_init(smem + 128, 1);
            mbarrier_init(smem + 136, 1);
            mbarrier_init(smem + 144, 1);
            // fc2__mma_done: 8 barriers, init_count=1
            mbarrier_init(smem + 152, 1);
            mbarrier_init(smem + 160, 1);
            mbarrier_init(smem + 168, 1);
            mbarrier_init(smem + 176, 1);
            mbarrier_init(smem + 184, 1);
            mbarrier_init(smem + 192, 1);
            mbarrier_init(smem + 200, 1);
            mbarrier_init(smem + 208, 1);
            // --- pipeline 'fc2__mainloop_pipe' ---
            // fc2__mainloop_done: 1 barriers, init_count=1
            mbarrier_init(smem + 216, 1);
            // fc2__epilogue_done: 1 barriers, init_count=4
            mbarrier_init(smem + 224, 4);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    // TMEM alloc (128 columns, 76 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 232);
    if (warp == 0) {
        int _tmem_hold = smem + 232;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(128) : "memory");
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_fc1__tmem_acc = taddr;
    const int tmem_fc1__tmem_sfa = taddr + 64;
    const int tmem_fc1__tmem_sfb = taddr + 72;
    const int tmem_fc2__tmem_acc = taddr;
    const int tmem_fc2__tmem_sfa = taddr + 32;
    const int tmem_fc2__tmem_sfb = taddr + 36;

    // ---- Role: epilogue ----
    if (warp <= 3) {
        { // swap_fc12_00_epilogue_dispatch
            #pragma unroll 1
            for (int reset_index = bid * 128 + tid; reset_index < 4096; reset_index += num_bids * 128) {
                route_map[reset_index] = -1;
            }
            int work = bid;
            int valid = ((work < dispatch_work_count) ? 1 : 0);
            int safe_work = ((work < 64) ? work : 0);
            int source_rank = dispatch_work_source_ranks[safe_work];
            int source_token = dispatch_work_source_tokens[safe_work];
            int local_group = dispatch_work_local_groups[safe_work];
            int local_row = dispatch_work_local_rows[safe_work];
            unsigned long long hidden_offset = (unsigned long long)source_token * 3072;
            unsigned long long scales_offset = (unsigned long long)source_token * 96;
            unsigned long long weights_offset = (unsigned long long)source_token * 8 * 4;
            unsigned long long ready_index = (unsigned long long)source_token * 16 + (unsigned long long)dispatch_pg_rank;
            unsigned int* remote_input_ready = reinterpret_cast<unsigned int*>(dispatch_input_ready_peers[source_rank]);
            if (valid != 0) {
                if (warp == 0) {
                    if (elect_sync()) {
                        {
                            unsigned int* _sre_ptr_0 = (reinterpret_cast<unsigned int*>(remote_input_ready) + (ready_index));
                            const unsigned int _sre_expected_0 = static_cast<unsigned int>(1);
                            const unsigned long long _sre_start_0 = clock64();
                            bool _sre_matched_0 = false;
                            do {
                                unsigned int _sre_value_0;
                                asm volatile("ld.relaxed.sys.u32 %0, [%1];" : "=r"(_sre_value_0) : "l"(_sre_ptr_0) : "memory");
                                _sre_matched_0 = (_sre_value_0 == _sre_expected_0);
                            } while (!_sre_matched_0 && ((clock64() - _sre_start_0) <= static_cast<unsigned long long>(4000000000)));
                            if (__builtin_expect(!_sre_matched_0, 0)) {
                                asm volatile("trap;");
                                return;
                            }
                        }
                        asm volatile("fence.acquire.sys;" ::: "memory");
                    }
                }
            }
            asm volatile("barrier.sync 15, 128;" ::: "memory");
            if (warp == 0) {
                if (elect_sync()) {
                    if (valid != 0) {
                        mbarrier_arrive_expect_tx(dispatch_full_addr, 3200);
                        // nvlink_pull: smem(dispatch_row_addr) <- peers[source_rank] + hidden_offset, 3072B
                        {
                            const void* __remote = (const void*)((const char*)((dispatch_hidden_states_mxfp8_peers)[source_rank]) + (uint64_t)(hidden_offset));
                            asm volatile(
                                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes"
                                " [%0], [%1], %2, [%3];"
                                :: "r"(dispatch_row_addr), "l"(__remote), "r"((uint32_t)(3072)), "r"(dispatch_full_addr)
                                : "memory");
                        }
                        // nvlink_pull: smem(dispatch_scale_row_addr) <- peers[source_rank] + scales_offset, 96B
                        {
                            const void* __remote = (const void*)((const char*)((dispatch_hidden_scales_mxfp8_peers)[source_rank]) + (uint64_t)(scales_offset));
                            asm volatile(
                                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes"
                                " [%0], [%1], %2, [%3];"
                                :: "r"(dispatch_scale_row_addr), "l"(__remote), "r"((uint32_t)(96)), "r"(dispatch_full_addr)
                                : "memory");
                        }
                        // nvlink_pull: smem(dispatch_weight_row_addr) <- peers[source_rank] + weights_offset, 32B
                        {
                            const void* __remote = (const void*)((const char*)((dispatch_topk_weights_peers)[source_rank]) + (uint64_t)(weights_offset));
                            asm volatile(
                                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes"
                                " [%0], [%1], %2, [%3];"
                                :: "r"(dispatch_weight_row_addr), "l"(__remote), "r"((uint32_t)(32)), "r"(dispatch_full_addr)
                                : "memory");
                        }
                    } else {
                        mbarrier_arrive(dispatch_full_addr);
                    }
                }
            }
            unsigned int _phase_dispatch_full_0 = 0;
            mbarrier_wait(dispatch_full_addr, _phase_dispatch_full_0);
            _phase_dispatch_full_0 ^= 1;
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            if (valid != 0) {
                if (warp == 0) {
                    if (elect_sync()) {
                        asm volatile("st.relaxed.sys.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(remote_input_ready) + (ready_index))), "r"(static_cast<unsigned int>(0)) : "memory");
                    }
                }
            }
            asm volatile("barrier.sync 13, 256;" ::: "memory");
            int local_expert = local_group * 8 + warp;
            int recv_token = source_rank * tokens_per_rank + source_token;
            int route = recv_token * 8 + warp;
            int packed_row = local_expert * 128 + local_row;
            if (valid != 0 && lane == 0) {
                route_weights[route] = dispatch_weight_row[warp];
                route_map[packed_row] = route;
            }
            int shared_packed_row = local_group * 128 + local_row;
            if (valid != 0) {
                #pragma unroll 1
                for (int block_column = warp; block_column < 96; block_column += 8) {
                    int column = block_column * 32 + lane;
                    B1_raw[(unsigned long long)shared_packed_row * 3072 + (unsigned long long)column] = dispatch_row[column];
                    if (lane == 0) {
                        int a = local_row / 32;
                        int row32 = local_row - a * 32;
                        int tile_k = block_column / 4;
                        int u = block_column - tile_k * 4;
                        int cp_in_tile = row32 * 16 + a * 4 + u;
                        unsigned long long cp_index = (unsigned long long)local_group * 24 * 512 + (unsigned long long)tile_k * 512 + (unsigned long long)cp_in_tile;
                        SFB1_raw[cp_index] = dispatch_scale_row[block_column];
                    }
                }
            }
            asm volatile("barrier.sync 12, 256;" ::: "memory");
            if (warp == 0) {
                if (elect_sync()) {
                    {
                        unsigned int* _gc_p = reinterpret_cast<unsigned int*>(fc1_done) + (0);
                        unsigned int _gc_old;
                        asm volatile("atom.release.gpu.global.add.u32 %0, [%1], 1;" : "=r"(_gc_old) : "l"(_gc_p) : "memory");
                    }
                    {
                        unsigned int* _gca_p = reinterpret_cast<unsigned int*>(fc1_done) + (0);
                        while (true) {
                            unsigned int _gca_v;
                            asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(_gca_v) : "l"(_gca_p));
                            if (_gca_v >= (unsigned int)(num_bids)) break;
                        }
                    }
                }
            }
            asm volatile("barrier.sync 12, 256;" ::: "memory");
        }
        { // swap_fc12_06_epilogue_main
            unsigned int fc1__epi_stage = 0;
            const int fc1__epi_warp = warp % 4;
            const int fc1__hidden_lane = fc1__epi_warp * 32 + lane;
            int fc1__n_out = N1 / 2;
            unsigned int _phase_fc1__mainloop_done = 0;
            #pragma unroll 1
            for (unsigned int tile_idx = bid; tile_idx < fc1_num_tiles; tile_idx += num_bids) {
                int fc1__batch_idx = fc1_tile_expert[tile_idx];
                int fc1__off_n_out = fc1_tile_m_local[tile_idx];
                int fc1__expert_row_base = expert_row_offsets[fc1__batch_idx];
                mbarrier_wait(fc1__mainloop_done_addr + (fc1__epi_stage) * 8, _phase_fc1__mainloop_done);
                asm volatile("tcgen05.fence::after_thread_sync;");
                float fc1__sg = scale_gate[fc1__batch_idx];
                float fc1__cl = clamp_limit[fc1__batch_idx];
                float fc1__al = act_alpha[fc1__batch_idx];
                float fc1__be = act_beta[fc1__batch_idx];
                float fc1__neg_cl = -fc1__cl;
                float fc1__beta_sg = fc1__be * fc1__sg;
                float fc1__alpha_sg = fc1__al * fc1__sg;
                #pragma unroll 1
                for (int n_chunk = 0; n_chunk < 4; n_chunk++) {
                    int fc1__route_col = n_chunk * 8;
                    float fc1___tmem_load_0[8];
                    tmem_ld_x8(&fc1___tmem_load_0[0], taddr + (unsigned int)(fc1__epi_warp * 32 + lane << 16) + (unsigned int)fc1__route_col);
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    float fc1___tmem_load_1[8];
                    tmem_ld_x8(&fc1___tmem_load_1[0], taddr + (unsigned int)(fc1__epi_warp * 32 + lane << 16) + 32 + (unsigned int)fc1__route_col);
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    #pragma unroll
                    for (int j = 0; j < 8; j++) {
                        int fc1__expert_row = n_chunk * 8 + j;
                        float fc1__g = fc1___tmem_load_0[j];
                        float fc1__u = fc1___tmem_load_1[j];
                        float fc1__result = 0.0f;
                        float fc1__sig = 0.0f;
                        float fc1___exp2_0 = approx_exp2((-fc1__u) * 1.4426950408889634f);
                        float fc1___rcp_0 = approx_rcp(1.0f + fc1___exp2_0);
                        fc1__sig = fc1___rcp_0;
                        fc1__result = fc1__sig * fc1__u * fc1__g;
                        fc1__epi_staging[fc1__expert_row * 128 + fc1__hidden_lane] = fc1__result;
                    }
                }
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("barrier.sync 15, 128;" ::: "memory");
                #pragma unroll
                for (int task_iter = 0; task_iter < 1; task_iter++) {
                    int fc1__task = fc1__hidden_lane + task_iter * 128;
                    int fc1__expert_row_1 = fc1__task / 4;
                    int fc1__hidden_group = fc1__task - fc1__expert_row_1 * 4;
                    int fc1__block_addr = fc1__epi_staging_addr + (unsigned int)((fc1__expert_row_1 * 128 + fc1__hidden_group * 32) * 4);
                    float fc1__staged_results[32];
                    #pragma unroll
                    for (int load_iter = 0; load_iter < 8; load_iter++) {
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&fc1__staged_results[load_iter * 4])), "=r"(*reinterpret_cast<uint32_t*>(&fc1__staged_results[(load_iter * 4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&fc1__staged_results[(load_iter * 4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&fc1__staged_results[(load_iter * 4) + 3]))
                            : "r"(fc1__block_addr + load_iter * 16));
                    }
                    int fc1__packed_row = fc1__expert_row_base + fc1__expert_row_1;
                    int fc1__route = route_map[fc1__packed_row];
                    int fc1__active = ((fc1__route >= 0) ? 1 : 0);
                    int fc1__safe_route = ((fc1__active != 0) ? fc1__route : 0);
                    float _cvt_f32_0 = __bfloat162float(fc1__active);
                    float fc1__rw = route_weights[fc1__safe_route] * _cvt_f32_0;
                    float fc1__amax = 1e-07f;
                    #pragma unroll
                    for (int c = 0; c < 4; c++) {
                        #pragma unroll
                        for (int j_1 = 0; j_1 < 8; j_1++) {
                            float fc1__weighted = fc1__staged_results[c * 8 + j_1] * fc1__rw;
                            float fc1___fabs_0 = fabsf(fc1__weighted);
                            float fc1___max_0 = max_noftz(fc1__amax, fc1___fabs_0);
                            fc1__amax = fc1___max_0;
                        }
                    }
                    float fc1__xsf = fc1__amax * 0.002232142857142857f;
                    unsigned int fc1__xbits = __as_u32(fc1__xsf);
                    unsigned int fc1__scale_code = (fc1__xbits >> 23 & 255) + ((fc1__xbits & 8388607) + 8388607 >> 23);
                    unsigned int fc1___max_1 = ((fc1__scale_code) > (1) ? (fc1__scale_code) : (1));
                    fc1__scale_code = fc1___max_1;
                    unsigned int fc1___min_0 = ((fc1__scale_code) < (254) ? (fc1__scale_code) : (254));
                    fc1__scale_code = fc1___min_0;
                    int fc1__scale_i = fc1__scale_code;
                    float _cvt_f32_1 = __bfloat162float(fc1__scale_i);
                    float fc1___exp2_1 = approx_exp2(127.0f - _cvt_f32_1);
                    float fc1__sf_inv = fc1___exp2_1;
                    int fc1__block_column = fc1__off_n_out / 32 + fc1__hidden_group;
                    int fc1__tile_m_out = fc1__expert_row_1 / 128;
                    int fc1__row_in_tile = fc1__expert_row_1 - fc1__tile_m_out * 128;
                    int fc1__a = fc1__row_in_tile / 32;
                    int fc1__row32 = fc1__row_in_tile - fc1__a * 32;
                    int fc1__tile_k_out = fc1__block_column / 4;
                    int fc1__uu = fc1__block_column - fc1__tile_k_out * 4;
                    int fc1__cp_in_tile = fc1__row32 * 16 + fc1__a * 4 + fc1__uu;
                    int fc1__tiles_m_out = rows_per_expert / 128;
                    int fc1__tiles_k_out = fc1__n_out / 128;
                    unsigned long long fc1__cp_index = (unsigned long long)fc1__batch_idx * (unsigned long long)fc1__tiles_m_out * (unsigned long long)fc1__tiles_k_out * 512 + ((unsigned long long)fc1__tile_m_out * (unsigned long long)fc1__tiles_k_out + (unsigned long long)fc1__tile_k_out) * 512 + (unsigned long long)fc1__cp_in_tile;
                    SF_out[fc1__cp_index] = fc1__scale_i;
                    #pragma unroll
                    for (int c_1 = 0; c_1 < 4; c_1++) {
                        float fc1__normalized[8];
                        #pragma unroll
                        for (int j_2 = 0; j_2 < 8; j_2++) {
                            fc1__normalized[j_2] = fc1__staged_results[c_1 * 8 + j_2] * fc1__rw * fc1__sf_inv;
                        }
                        {
                            unsigned int _fp8_pk[2];
                            asm("{\n\t"
                                ".reg .b16 _lo, _hi;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                "mov.b32 %0, {_lo, _hi};\n\t"
                                "}\n"
                                : "=r"(_fp8_pk[0]) : "f"(fc1__normalized[0 + 0]), "f"(fc1__normalized[0 + 1]), "f"(fc1__normalized[0 + 2]), "f"(fc1__normalized[0 + 3]));
                            asm("{\n\t"
                                ".reg .b16 _lo, _hi;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                "mov.b32 %0, {_lo, _hi};\n\t"
                                "}\n"
                                : "=r"(_fp8_pk[1]) : "f"(fc1__normalized[0 + 4]), "f"(fc1__normalized[0 + 5]), "f"(fc1__normalized[0 + 6]), "f"(fc1__normalized[0 + 7]));
                            *reinterpret_cast<uint2*>(reinterpret_cast<unsigned char*>(C_fp8 + ((unsigned long long)fc1__packed_row * (unsigned long long)fc1__n_out + (unsigned long long)fc1__off_n_out + (unsigned long long)(fc1__hidden_group * 32) + (unsigned long long)(c_1 * 8))) + (0)) = *reinterpret_cast<uint2*>(_fp8_pk);
                        }
                    }
                }
                asm volatile("barrier.sync 15, 128;" ::: "memory");
                if (elect_sync()) {
                    asm volatile(
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                        :: "r"((fc1__epilogue_done_addr + (fc1__epi_stage) * 8) & 0xFEFFFFFF) : "memory");
                }
                _phase_fc1__mainloop_done ^= 1;
            }
        }
        { // swap_fc12_07_epilogue_fc1_to_fc2
            asm volatile("barrier.sync 10, 192;" ::: "memory");
            asm volatile("barrier.sync 10, 192;" ::: "memory");
        }
        { // swap_fc12_12_epilogue_main
            unsigned int fc2__epi_stage = 0;
            const int fc2__epi_warp = warp % 4;
            const int fc2__epi_tid = fc2__epi_warp * 32 + lane;
            unsigned int _phase_fc2__mainloop_done = 0;
            #pragma unroll 1
            for (unsigned int tile_idx_1 = bid; tile_idx_1 < fc2_num_tiles; tile_idx_1 += num_bids) {
                int fc2__batch_idx = fc2_tile_expert[tile_idx_1];
                int fc2__off_m = fc2_tile_m_local[tile_idx_1];
                int fc2__expert_row_base = expert_row_offsets[fc2__batch_idx];
                mbarrier_wait(fc2__mainloop_done_addr + (fc2__epi_stage) * 8, _phase_fc2__mainloop_done);
                asm volatile("tcgen05.fence::after_thread_sync;");
                int fc2__ready_route = -1;
                int fc2__ready_tile = 0;
                unsigned int* fc2__ready_base = reinterpret_cast<unsigned int*>(C);
                #pragma unroll 1
                for (int n_chunk_1 = 0; n_chunk_1 < 4; n_chunk_1++) {
                    int fc2__row = fc2__epi_warp * 32;
                    int fc2__col = n_chunk_1 * 8;
                    float fc2___tmem_load_2[8];
                    tmem_ld_x8(&fc2___tmem_load_2[0], taddr + (unsigned int)(fc2__row << 16) + (unsigned int)fc2__col);
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    #pragma unroll
                    for (int j_3 = 0; j_3 < 8; j_3++) {
                        int fc2__expert_row = n_chunk_1 * 8 + j_3;
                        __nv_bfloat16 fc2___cvt_bf16_0 = __float2bfloat16(fc2___tmem_load_2[j_3]);
                        fc2__epi_staging[fc2__expert_row * 128 + fc2__epi_tid] = fc2___cvt_bf16_0;
                    }
                }
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("barrier.sync 15, 128;" ::: "memory");
                #pragma unroll
                for (int copy_iter = 0; copy_iter < 4; copy_iter++) {
                    int fc2__copy_linear = fc2__epi_tid + copy_iter * 128;
                    int fc2__expert_row_1 = fc2__copy_linear / 16;
                    int fc2__hidden_chunk = fc2__copy_linear % 16;
                    unsigned int fc2__packed[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&fc2__packed[0])), "=r"(*reinterpret_cast<uint32_t*>(&fc2__packed[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&fc2__packed[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&fc2__packed[(0) + 3]))
                        : "r"(fc2__epi_staging_addr + (unsigned int)((fc2__expert_row_1 * 128 + fc2__hidden_chunk * 8) * 2)));
                    int fc2__route = route_map[fc2__expert_row_base + fc2__expert_row_1];
                    if (fc2__route >= 0) {
                        __nv_bfloat16* fc2__output_base = C;
                        reinterpret_cast<int4*>(fc2__output_base + ((unsigned long long)fc2__route * (unsigned long long)N2 + (unsigned long long)fc2__off_m + (unsigned long long)(fc2__hidden_chunk * 8)))[0] = reinterpret_cast<int4*>(fc2__packed)[0];
                    }
                }
                asm volatile("barrier.sync 15, 128;" ::: "memory");
                if (elect_sync()) {
                    asm volatile(
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                        :: "r"((fc2__epilogue_done_addr + (fc2__epi_stage) * 8) & 0xFEFFFFFF) : "memory");
                }
                _phase_fc2__mainloop_done ^= 1;
            }
        }
        { // swap_fc12_13_epilogue_fc2_exit
            asm volatile("barrier.sync 11, 256;" ::: "memory");
            asm volatile("barrier.sync 11, 256;" ::: "memory");
            int work_1 = bid;
            int valid_1 = ((work_1 < dispatch_work_count) ? 1 : 0);
            int safe_work_1 = ((work_1 < 64) ? work_1 : 0);
            int source_rank_1 = dispatch_work_source_ranks[safe_work_1];
            int source_token_1 = dispatch_work_source_tokens[safe_work_1];
            int recv_token_1 = source_rank_1 * tokens_per_rank + source_token_1;
            int first_route = recv_token_1 * 8;
            float accum[16];
            accum[0] = 0.0f;
            accum[1] = 0.0f;
            accum[2] = 0.0f;
            accum[3] = 0.0f;
            accum[4] = 0.0f;
            accum[5] = 0.0f;
            accum[6] = 0.0f;
            accum[7] = 0.0f;
            accum[8] = 0.0f;
            accum[9] = 0.0f;
            accum[10] = 0.0f;
            accum[11] = 0.0f;
            accum[12] = 0.0f;
            accum[13] = 0.0f;
            accum[14] = 0.0f;
            accum[15] = 0.0f;
            if (valid_1 != 0 && tid < 192) {
                int column_1 = tid * 16;
                #pragma unroll
                for (int route_slot = 0; route_slot < 8; route_slot++) {
                    unsigned long long slot_base = ((unsigned long long)first_route + (unsigned long long)route_slot) * 3072;
                    float _vec_load_0[16];
                    {
                        const uint4* _vptr_0 = reinterpret_cast<const uint4*>(C + (slot_base + (unsigned long long)column_1) + 0);
                        uint4 _vld_0[2];
                        #pragma unroll
                        for (int _blk = 0; _blk < 2; _blk++) {
                            _vld_0[_blk] = _vptr_0[_blk];
                            uint32_t* _vpairs_0 = reinterpret_cast<uint32_t*>(&_vld_0[_blk]);
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
                                asm volatile(
                                    "{\n\t"
                                    "shl.b32 %0, %2, 16;\n\t"
                                    "and.b32 %1, %2, 0xffff0000;\n\t"
                                    "}\n"
                                    : "=f"((&_vec_load_0[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_0[0 + _blk * 8 + _pair * 2])[1])
                                    : "r"(_vpairs_0[_pair]));
                            }
                        }
                    }
                    #pragma unroll
                    for (int element = 0; element < 16; element++) {
                        accum[element] = accum[element] + _vec_load_0[element];
                    }
                }
                unsigned long long destination_base = (unsigned long long)source_token_1 * 3072;
                {
                    __nv_bfloat162 _pk[8];
                    _pk[0] = __floats2bfloat162_rn(accum[0 + 0], accum[0 + 1]);
                    _pk[1] = __floats2bfloat162_rn(accum[0 + 2], accum[0 + 3]);
                    _pk[2] = __floats2bfloat162_rn(accum[0 + 4], accum[0 + 5]);
                    _pk[3] = __floats2bfloat162_rn(accum[0 + 6], accum[0 + 7]);
                    _pk[4] = __floats2bfloat162_rn(accum[0 + 8], accum[0 + 9]);
                    _pk[5] = __floats2bfloat162_rn(accum[0 + 10], accum[0 + 11]);
                    _pk[6] = __floats2bfloat162_rn(accum[0 + 12], accum[0 + 13]);
                    _pk[7] = __floats2bfloat162_rn(accum[0 + 14], accum[0 + 15]);
                    *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(reinterpret_cast<__nv_bfloat16*>(direct_output_peers[source_rank_1]) + (destination_base + (unsigned long long)column_1)))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                    *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(reinterpret_cast<__nv_bfloat16*>(direct_output_peers[source_rank_1]) + (destination_base + (unsigned long long)column_1)))[8]) = *reinterpret_cast<uint4*>(&_pk[4]);
                }
            }
            asm volatile("barrier.sync 11, 256;" ::: "memory");
        }
    }
    // ---- Role: mma ----
    if (warp == 4) {
        { // swap_fc12_01_mma_dispatch_join
            int work_2 = bid;
            int valid_2 = ((work_2 < dispatch_work_count) ? 1 : 0);
            int safe_work_2 = ((work_2 < 64) ? work_2 : 0);
            int source_rank_2 = dispatch_work_source_ranks[safe_work_2];
            int source_token_2 = dispatch_work_source_tokens[safe_work_2];
            int local_group_1 = dispatch_work_local_groups[safe_work_2];
            int local_row_1 = dispatch_work_local_rows[safe_work_2];
            asm volatile("barrier.sync 13, 256;" ::: "memory");
            int local_expert_1 = local_group_1 * 8 + warp;
            int recv_token_2 = source_rank_2 * tokens_per_rank + source_token_2;
            int route_1 = recv_token_2 * 8 + warp;
            int packed_row_1 = local_expert_1 * 128 + local_row_1;
            if (valid_2 != 0 && lane == 0) {
                route_weights[route_1] = dispatch_weight_row[warp];
                route_map[packed_row_1] = route_1;
            }
            int shared_packed_row_1 = local_group_1 * 128 + local_row_1;
            if (valid_2 != 0) {
                #pragma unroll 1
                for (int block_column_1 = warp; block_column_1 < 96; block_column_1 += 8) {
                    int column_2 = block_column_1 * 32 + lane;
                    B1_raw[(unsigned long long)shared_packed_row_1 * 3072 + (unsigned long long)column_2] = dispatch_row[column_2];
                    if (lane == 0) {
                        int a_1 = local_row_1 / 32;
                        int row32_1 = local_row_1 - a_1 * 32;
                        int tile_k_1 = block_column_1 / 4;
                        int u_1 = block_column_1 - tile_k_1 * 4;
                        int cp_in_tile_1 = row32_1 * 16 + a_1 * 4 + u_1;
                        unsigned long long cp_index_1 = (unsigned long long)local_group_1 * 24 * 512 + (unsigned long long)tile_k_1 * 512 + (unsigned long long)cp_in_tile_1;
                        SFB1_raw[cp_index_1] = dispatch_scale_row[block_column_1];
                    }
                }
            }
            asm volatile("barrier.sync 12, 256;" ::: "memory");
            asm volatile("barrier.sync 12, 256;" ::: "memory");
        }
        { // swap_fc12_05_mma_main
            unsigned int fc1__mma_tma_stage = 0;
            unsigned int fc1__mma_epi_stage = 0;
            unsigned int _phase_fc1__epilogue_done = 1;
            unsigned int _phase_fc1__tma_full = 0;
            #pragma unroll 1
            for (unsigned int tile_idx_2 = bid; tile_idx_2 < fc1_num_tiles; tile_idx_2 += num_bids) {
                mbarrier_wait(fc1__epilogue_done_addr + (fc1__mma_epi_stage) * 8, _phase_fc1__epilogue_done);
                #pragma unroll 1
                for (int iter_k = 0; iter_k < K1_tiles; iter_k++) {
                    mbarrier_wait(fc1__tma_full_addr + (fc1__mma_tma_stage) * 8, _phase_fc1__tma_full);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int fc1__init_flag = ((iter_k == 0) ? 1 : 0);
                    if (elect_sync()) {
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                        #endif
                        {
                            uint64_t _tcgen05_cp_desc_0 = ((((uint64_t)(fc1__smem_sfa_cp_addr + fc1__mma_tma_stage * 38400)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL);
                            asm volatile(
                                "tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;"
                                :: "r"((uint32_t)(tmem_fc1__tmem_sfa)), "l"(_tcgen05_cp_desc_0)
                                : "memory");
                        }
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                        #endif
                        {
                            uint64_t _tcgen05_cp_desc_1 = ((((uint64_t)(fc1__smem_sfa_cp_addr + fc1__mma_tma_stage * 38400 + 512)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL);
                            asm volatile(
                                "tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;"
                                :: "r"((uint32_t)(tmem_fc1__tmem_sfa + 4)), "l"(_tcgen05_cp_desc_1)
                                : "memory");
                        }
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                        #endif
                        {
                            uint64_t _tcgen05_cp_desc_2 = ((((uint64_t)(fc1__smem_sfb_cp_addr + fc1__mma_tma_stage * 38400)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL);
                            asm volatile(
                                "tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;"
                                :: "r"((uint32_t)(tmem_fc1__tmem_sfb)), "l"(_tcgen05_cp_desc_2)
                                : "memory");
                        }
                        int _mma_a_lo_0 = (((fc1__smem_a_addr) >> 4) & 0x3FFF) + (fc1__mma_tma_stage) * 2400;
                        int _mma_b_lo_0 = (((fc1__smem_b_addr) >> 4) & 0x3FFF) + (fc1__mma_tma_stage) * 2400;
                        {
                            uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_0) | ((uint64_t)0x40004040 << 32);
                            uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_0) | ((uint64_t)0x40004040 << 32);

                            tcgen05_mma_mxf8_bs(tmem_fc1__tmem_acc, a_desc + 0, b_desc + 0,
                                0x8880000U, tmem_fc1__tmem_sfa, tmem_fc1__tmem_sfb, ((fc1__init_flag) ? 0 : 1));
                            tcgen05_mma_mxf8_bs(tmem_fc1__tmem_acc, a_desc + 2, b_desc + 2,
                                0x28880010U, tmem_fc1__tmem_sfa, tmem_fc1__tmem_sfb, 1);
                            tcgen05_mma_mxf8_bs(tmem_fc1__tmem_acc, a_desc + 4, b_desc + 4,
                                0x48880020U, tmem_fc1__tmem_sfa, tmem_fc1__tmem_sfb, 1);
                            tcgen05_mma_mxf8_bs(tmem_fc1__tmem_acc, a_desc + 6, b_desc + 6,
                                0x68880030U, tmem_fc1__tmem_sfa, tmem_fc1__tmem_sfb, 1);
                        }
                        int _mma_a_lo_1 = (((fc1__smem_a_addr + 16384) >> 4) & 0x3FFF) + (fc1__mma_tma_stage) * 2400;
                        int _mma_b_lo_1 = (((fc1__smem_b_addr) >> 4) & 0x3FFF) + (fc1__mma_tma_stage) * 2400;
                        {
                            uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_1) | ((uint64_t)0x40004040 << 32);
                            uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_1) | ((uint64_t)0x40004040 << 32);

                            tcgen05_mma_mxf8_bs((tmem_fc1__tmem_acc + (32)), a_desc + 0, b_desc + 0,
                                0x8880000U, tmem_fc1__tmem_sfa + 4, tmem_fc1__tmem_sfb, ((fc1__init_flag) ? 0 : 1));
                            tcgen05_mma_mxf8_bs((tmem_fc1__tmem_acc + (32)), a_desc + 2, b_desc + 2,
                                0x28880010U, tmem_fc1__tmem_sfa + 4, tmem_fc1__tmem_sfb, 1);
                            tcgen05_mma_mxf8_bs((tmem_fc1__tmem_acc + (32)), a_desc + 4, b_desc + 4,
                                0x48880020U, tmem_fc1__tmem_sfa + 4, tmem_fc1__tmem_sfb, 1);
                            tcgen05_mma_mxf8_bs((tmem_fc1__tmem_acc + (32)), a_desc + 6, b_desc + 6,
                                0x68880030U, tmem_fc1__tmem_sfa + 4, tmem_fc1__tmem_sfb, 1);
                        }
                    }
                    elect_commit(fc1__mma_done_addr + (fc1__mma_tma_stage) * 8);
                    fc1__mma_tma_stage += 1;
                    if (fc1__mma_tma_stage == 4) { fc1__mma_tma_stage = 0; _phase_fc1__tma_full ^= 1; }
                }
                elect_commit(fc1__mainloop_done_addr + (fc1__mma_epi_stage) * 8);
                _phase_fc1__epilogue_done ^= 1;
            }
        }
        { // swap_fc12_08_mma_fc1_to_fc2
            asm volatile("barrier.sync 10, 192;" ::: "memory");
            {
                if (elect_sync()) {
                    {
                        unsigned int* _gc_p = reinterpret_cast<unsigned int*>(fc1_done) + (0);
                        unsigned int _gc_old;
                        asm volatile("atom.release.gpu.global.add.u32 %0, [%1], 1;" : "=r"(_gc_old) : "l"(_gc_p) : "memory");
                    }
                    {
                        unsigned int* _gca_p = reinterpret_cast<unsigned int*>(fc1_done) + (0);
                        while (true) {
                            unsigned int _gca_v;
                            asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(_gca_v) : "l"(_gca_p));
                            if (_gca_v >= (unsigned int)(2 * num_bids)) break;
                        }
                    }
                }
            }
            asm volatile("barrier.sync 10, 192;" ::: "memory");
        }
        { // swap_fc12_11_mma_main
            unsigned int fc2__mma_tma_stage = 0;
            unsigned int fc2__mma_epi_stage = 0;
            unsigned int _phase_fc2__epilogue_done = 1;
            unsigned int _phase_fc2__tma_full = 0;
            #pragma unroll 1
            for (unsigned int tile_idx_3 = bid; tile_idx_3 < fc2_num_tiles; tile_idx_3 += num_bids) {
                mbarrier_wait(fc2__epilogue_done_addr + (fc2__mma_epi_stage) * 8, _phase_fc2__epilogue_done);
                #pragma unroll 1
                for (int iter_k_1 = 0; iter_k_1 < K2_tiles; iter_k_1++) {
                    mbarrier_wait(fc2__tma_full_addr + (fc2__mma_tma_stage) * 8, _phase_fc2__tma_full);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int fc2__init_flag = ((iter_k_1 == 0) ? 1 : 0);
                    if (elect_sync()) {
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                        #endif
                        {
                            uint64_t _tcgen05_cp_desc_0 = ((((uint64_t)(fc2__smem_sfa_cp_addr + fc2__mma_tma_stage * 21504)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL);
                            asm volatile(
                                "tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;"
                                :: "r"((uint32_t)(tmem_fc2__tmem_sfa)), "l"(_tcgen05_cp_desc_0)
                                : "memory");
                        }
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                        #endif
                        {
                            uint64_t _tcgen05_cp_desc_1 = ((((uint64_t)(fc2__smem_sfb_cp_addr + fc2__mma_tma_stage * 21504)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL);
                            asm volatile(
                                "tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;"
                                :: "r"((uint32_t)(tmem_fc2__tmem_sfb)), "l"(_tcgen05_cp_desc_1)
                                : "memory");
                        }
                        int _mma_a_lo_2 = (((fc2__smem_a_addr) >> 4) & 0x3FFF) + (fc2__mma_tma_stage) * 1344;
                        int _mma_b_lo_2 = (((fc2__smem_b_addr) >> 4) & 0x3FFF) + (fc2__mma_tma_stage) * 1344;
                        {
                            uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_2) | ((uint64_t)0x40004040 << 32);
                            uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_2) | ((uint64_t)0x40004040 << 32);

                            tcgen05_mma_mxf8_bs(tmem_fc2__tmem_acc, a_desc + 0, b_desc + 0,
                                0x8880000U, tmem_fc2__tmem_sfa, tmem_fc2__tmem_sfb, ((fc2__init_flag) ? 0 : 1));
                            tcgen05_mma_mxf8_bs(tmem_fc2__tmem_acc, a_desc + 2, b_desc + 2,
                                0x28880010U, tmem_fc2__tmem_sfa, tmem_fc2__tmem_sfb, 1);
                            tcgen05_mma_mxf8_bs(tmem_fc2__tmem_acc, a_desc + 4, b_desc + 4,
                                0x48880020U, tmem_fc2__tmem_sfa, tmem_fc2__tmem_sfb, 1);
                            tcgen05_mma_mxf8_bs(tmem_fc2__tmem_acc, a_desc + 6, b_desc + 6,
                                0x68880030U, tmem_fc2__tmem_sfa, tmem_fc2__tmem_sfb, 1);
                        }
                    }
                    elect_commit(fc2__mma_done_addr + (fc2__mma_tma_stage) * 8);
                    fc2__mma_tma_stage += 1;
                    if (fc2__mma_tma_stage == 8) { fc2__mma_tma_stage = 0; _phase_fc2__tma_full ^= 1; }
                }
                elect_commit(fc2__mainloop_done_addr + (fc2__mma_epi_stage) * 8);
                _phase_fc2__epilogue_done ^= 1;
            }
        }
        { // swap_fc12_14_mma_fc2_exit
            asm volatile("barrier.sync 11, 256;" ::: "memory");
            {
                if (elect_sync()) {
                    {
                        unsigned int* _gc_p = reinterpret_cast<unsigned int*>(fc1_done) + (0);
                        unsigned int _gc_old;
                        asm volatile("atom.release.gpu.global.add.u32 %0, [%1], 1;" : "=r"(_gc_old) : "l"(_gc_p) : "memory");
                    }
                    {
                        unsigned int* _gca_p = reinterpret_cast<unsigned int*>(fc1_done) + (0);
                        while (true) {
                            unsigned int _gca_v;
                            asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(_gca_v) : "l"(_gca_p));
                            if (_gca_v >= (unsigned int)(3 * num_bids)) break;
                        }
                    }
                }
            }
            asm volatile("barrier.sync 11, 256;" ::: "memory");
            int work_3 = bid;
            int valid_3 = ((work_3 < dispatch_work_count) ? 1 : 0);
            int safe_work_3 = ((work_3 < 64) ? work_3 : 0);
            int source_rank_3 = dispatch_work_source_ranks[safe_work_3];
            int source_token_3 = dispatch_work_source_tokens[safe_work_3];
            int recv_token_3 = source_rank_3 * tokens_per_rank + source_token_3;
            int first_route_1 = recv_token_3 * 8;
            float accum_1[16];
            accum_1[0] = 0.0f;
            accum_1[1] = 0.0f;
            accum_1[2] = 0.0f;
            accum_1[3] = 0.0f;
            accum_1[4] = 0.0f;
            accum_1[5] = 0.0f;
            accum_1[6] = 0.0f;
            accum_1[7] = 0.0f;
            accum_1[8] = 0.0f;
            accum_1[9] = 0.0f;
            accum_1[10] = 0.0f;
            accum_1[11] = 0.0f;
            accum_1[12] = 0.0f;
            accum_1[13] = 0.0f;
            accum_1[14] = 0.0f;
            accum_1[15] = 0.0f;
            if (valid_3 != 0 && tid < 192) {
                int column_3 = tid * 16;
                #pragma unroll
                for (int route_slot_1 = 0; route_slot_1 < 8; route_slot_1++) {
                    unsigned long long slot_base_1 = ((unsigned long long)first_route_1 + (unsigned long long)route_slot_1) * 3072;
                    float _vec_load_1[16];
                    {
                        const uint4* _vptr_0 = reinterpret_cast<const uint4*>(C + (slot_base_1 + (unsigned long long)column_3) + 0);
                        uint4 _vld_0[2];
                        #pragma unroll
                        for (int _blk = 0; _blk < 2; _blk++) {
                            _vld_0[_blk] = _vptr_0[_blk];
                            uint32_t* _vpairs_0 = reinterpret_cast<uint32_t*>(&_vld_0[_blk]);
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
                                asm volatile(
                                    "{\n\t"
                                    "shl.b32 %0, %2, 16;\n\t"
                                    "and.b32 %1, %2, 0xffff0000;\n\t"
                                    "}\n"
                                    : "=f"((&_vec_load_1[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_1[0 + _blk * 8 + _pair * 2])[1])
                                    : "r"(_vpairs_0[_pair]));
                            }
                        }
                    }
                    #pragma unroll
                    for (int element_1 = 0; element_1 < 16; element_1++) {
                        accum_1[element_1] = accum_1[element_1] + _vec_load_1[element_1];
                    }
                }
                unsigned long long destination_base_1 = (unsigned long long)source_token_3 * 3072;
                {
                    __nv_bfloat162 _pk[8];
                    _pk[0] = __floats2bfloat162_rn(accum_1[0 + 0], accum_1[0 + 1]);
                    _pk[1] = __floats2bfloat162_rn(accum_1[0 + 2], accum_1[0 + 3]);
                    _pk[2] = __floats2bfloat162_rn(accum_1[0 + 4], accum_1[0 + 5]);
                    _pk[3] = __floats2bfloat162_rn(accum_1[0 + 6], accum_1[0 + 7]);
                    _pk[4] = __floats2bfloat162_rn(accum_1[0 + 8], accum_1[0 + 9]);
                    _pk[5] = __floats2bfloat162_rn(accum_1[0 + 10], accum_1[0 + 11]);
                    _pk[6] = __floats2bfloat162_rn(accum_1[0 + 12], accum_1[0 + 13]);
                    _pk[7] = __floats2bfloat162_rn(accum_1[0 + 14], accum_1[0 + 15]);
                    *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(reinterpret_cast<__nv_bfloat16*>(direct_output_peers[source_rank_3]) + (destination_base_1 + (unsigned long long)column_3)))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                    *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(reinterpret_cast<__nv_bfloat16*>(direct_output_peers[source_rank_3]) + (destination_base_1 + (unsigned long long)column_3)))[8]) = *reinterpret_cast<uint4*>(&_pk[4]);
                }
            }
            asm volatile("barrier.sync 11, 256;" ::: "memory");
            {
                if (elect_sync()) {
                    unsigned int _atomic_old_2;
                    asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], %2;"
                        : "=r"(_atomic_old_2) : "l"(fc1_done), "r"(static_cast<uint32_t>(1)) : "memory");
                    if (_atomic_old_2 + 1 == (unsigned int)(4 * num_bids)) {
                        unsigned int _atomic_old_3;
                        asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], %2;"
                            : "=r"(_atomic_old_3) : "l"(fc1_done), "r"(static_cast<uint32_t>(-(4 * num_bids))) : "memory");
                        asm volatile("fence.release.sys;" ::: "memory");
                        #pragma unroll
                        for (int destination_rank = 0; destination_rank < 16; destination_rank++) {
                            asm volatile("st.relaxed.sys.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(remote_ready_peers[destination_rank])) + (remote_ready_pg_rank))), "r"(static_cast<unsigned int>(1)) : "memory");
                        }
                    }
                }
            }
        }
    }
    // ---- Role: load ----
    if (warp == 5) {
        { // swap_fc12_02_load_dispatch_join
            int work_4 = bid;
            int valid_4 = ((work_4 < dispatch_work_count) ? 1 : 0);
            int safe_work_4 = ((work_4 < 64) ? work_4 : 0);
            int source_rank_4 = dispatch_work_source_ranks[safe_work_4];
            int source_token_4 = dispatch_work_source_tokens[safe_work_4];
            int local_group_2 = dispatch_work_local_groups[safe_work_4];
            int local_row_2 = dispatch_work_local_rows[safe_work_4];
            asm volatile("barrier.sync 13, 256;" ::: "memory");
            int local_expert_2 = local_group_2 * 8 + warp;
            int recv_token_4 = source_rank_4 * tokens_per_rank + source_token_4;
            int route_2 = recv_token_4 * 8 + warp;
            int packed_row_2 = local_expert_2 * 128 + local_row_2;
            if (valid_4 != 0 && lane == 0) {
                route_weights[route_2] = dispatch_weight_row[warp];
                route_map[packed_row_2] = route_2;
            }
            int shared_packed_row_2 = local_group_2 * 128 + local_row_2;
            if (valid_4 != 0) {
                #pragma unroll 1
                for (int block_column_2 = warp; block_column_2 < 96; block_column_2 += 8) {
                    int column_4 = block_column_2 * 32 + lane;
                    B1_raw[(unsigned long long)shared_packed_row_2 * 3072 + (unsigned long long)column_4] = dispatch_row[column_4];
                    if (lane == 0) {
                        int a_2 = local_row_2 / 32;
                        int row32_2 = local_row_2 - a_2 * 32;
                        int tile_k_2 = block_column_2 / 4;
                        int u_2 = block_column_2 - tile_k_2 * 4;
                        int cp_in_tile_2 = row32_2 * 16 + a_2 * 4 + u_2;
                        unsigned long long cp_index_2 = (unsigned long long)local_group_2 * 24 * 512 + (unsigned long long)tile_k_2 * 512 + (unsigned long long)cp_in_tile_2;
                        SFB1_raw[cp_index_2] = dispatch_scale_row[block_column_2];
                    }
                }
            }
            asm volatile("barrier.sync 12, 256;" ::: "memory");
            asm volatile("barrier.sync 12, 256;" ::: "memory");
        }
        { // swap_fc12_04_load_main
            unsigned int fc1__load_stage = 0;
            unsigned int _phase_fc1__mma_done = 1;
            if (elect_sync()) {
                #pragma unroll 1
                for (unsigned int tile_idx_4 = bid; tile_idx_4 < fc1_num_tiles; tile_idx_4 += num_bids) {
                    int fc1__batch_idx_1 = fc1_tile_expert[tile_idx_4];
                    int fc1__input_batch_idx = ((1) ? fc1__batch_idx_1 / 8 : fc1__batch_idx_1);
                    int fc1__off_n_out_1 = fc1_tile_m_local[tile_idx_4];
                    int fc1__packed_off = fc1__off_n_out_1 * 2;
                    #pragma unroll 1
                    for (int iter_k_2 = 0; iter_k_2 < K1_tiles; iter_k_2++) {
                        mbarrier_wait(fc1__mma_done_addr + (fc1__load_stage) * 8, _phase_fc1__mma_done);
                        tma_4d_gmem2smem(fc1__smem_a_addr + fc1__load_stage * 38400, A1, 0, fc1__packed_off, iter_k_2, fc1__batch_idx_1, fc1__tma_full_addr + (fc1__load_stage) * 8);
                        tma_4d_gmem2smem(fc1__smem_b_addr + fc1__load_stage * 38400, B1, 0, 0, iter_k_2, fc1__input_batch_idx, fc1__tma_full_addr + (fc1__load_stage) * 8);
                        int fc1__sfa_tile = fc1__packed_off / 256 * K1_tiles + iter_k_2;
                        tma_4d_gmem2smem(fc1__smem_sfa_cp_addr + fc1__load_stage * 38400, SFA1, 0, 0, fc1__sfa_tile, fc1__batch_idx_1, fc1__tma_full_addr + (fc1__load_stage) * 8);
                        tma_4d_gmem2smem(fc1__smem_sfb_cp_addr + fc1__load_stage * 38400, SFB1, 0, 0, iter_k_2, fc1__input_batch_idx, fc1__tma_full_addr + (fc1__load_stage) * 8);
                        mbarrier_arrive_expect_tx(fc1__tma_full_addr + (fc1__load_stage) * 8, 38400);
                        fc1__load_stage += 1;
                        if (fc1__load_stage == 4) { fc1__load_stage = 0; _phase_fc1__mma_done ^= 1; }
                    }
                }
            }
        }
        { // swap_fc12_09_load_fc1_to_fc2
            asm volatile("barrier.sync 10, 192;" ::: "memory");
            asm volatile("barrier.sync 10, 192;" ::: "memory");
        }
        { // swap_fc12_10_load_main
            unsigned int fc2__load_stage = 0;
            unsigned int _phase_fc2__mma_done = 1;
            if (elect_sync()) {
                #pragma unroll 1
                for (unsigned int tile_idx_5 = bid; tile_idx_5 < fc2_num_tiles; tile_idx_5 += num_bids) {
                    int fc2__batch_idx_1 = fc2_tile_expert[tile_idx_5];
                    int fc2__off_m_1 = fc2_tile_m_local[tile_idx_5];
                    #pragma unroll 1
                    for (int iter_k_3 = 0; iter_k_3 < K2_tiles; iter_k_3++) {
                        mbarrier_wait(fc2__mma_done_addr + (fc2__load_stage) * 8, _phase_fc2__mma_done);
                        tma_4d_gmem2smem(fc2__smem_a_addr + fc2__load_stage * 21504, A2, 0, fc2__off_m_1, iter_k_3, fc2__batch_idx_1, fc2__tma_full_addr + (fc2__load_stage) * 8);
                        tma_4d_gmem2smem(fc2__smem_b_addr + fc2__load_stage * 21504, B2, 0, 0, iter_k_3, fc2__batch_idx_1, fc2__tma_full_addr + (fc2__load_stage) * 8);
                        int fc2__sfa_tile = fc2__off_m_1 / 128 * K2_tiles + iter_k_3;
                        tma_4d_gmem2smem(fc2__smem_sfa_cp_addr + fc2__load_stage * 21504, SFA2, 0, 0, fc2__sfa_tile, fc2__batch_idx_1, fc2__tma_full_addr + (fc2__load_stage) * 8);
                        tma_4d_gmem2smem(fc2__smem_sfb_cp_addr + fc2__load_stage * 21504, SFB2, 0, 0, iter_k_3, fc2__batch_idx_1, fc2__tma_full_addr + (fc2__load_stage) * 8);
                        mbarrier_arrive_expect_tx(fc2__tma_full_addr + (fc2__load_stage) * 8, 21504);
                        fc2__load_stage += 1;
                        if (fc2__load_stage == 8) { fc2__load_stage = 0; _phase_fc2__mma_done ^= 1; }
                    }
                }
            }
        }
        { // swap_fc12_15_load_fc2_exit
            asm volatile("barrier.sync 11, 256;" ::: "memory");
            asm volatile("barrier.sync 11, 256;" ::: "memory");
            int work_5 = bid;
            int valid_5 = ((work_5 < dispatch_work_count) ? 1 : 0);
            int safe_work_5 = ((work_5 < 64) ? work_5 : 0);
            int source_rank_5 = dispatch_work_source_ranks[safe_work_5];
            int source_token_5 = dispatch_work_source_tokens[safe_work_5];
            int recv_token_5 = source_rank_5 * tokens_per_rank + source_token_5;
            int first_route_2 = recv_token_5 * 8;
            float accum_2[16];
            accum_2[0] = 0.0f;
            accum_2[1] = 0.0f;
            accum_2[2] = 0.0f;
            accum_2[3] = 0.0f;
            accum_2[4] = 0.0f;
            accum_2[5] = 0.0f;
            accum_2[6] = 0.0f;
            accum_2[7] = 0.0f;
            accum_2[8] = 0.0f;
            accum_2[9] = 0.0f;
            accum_2[10] = 0.0f;
            accum_2[11] = 0.0f;
            accum_2[12] = 0.0f;
            accum_2[13] = 0.0f;
            accum_2[14] = 0.0f;
            accum_2[15] = 0.0f;
            if (valid_5 != 0 && tid < 192) {
                int column_5 = tid * 16;
                #pragma unroll
                for (int route_slot_2 = 0; route_slot_2 < 8; route_slot_2++) {
                    unsigned long long slot_base_2 = ((unsigned long long)first_route_2 + (unsigned long long)route_slot_2) * 3072;
                    float _vec_load_2[16];
                    {
                        const uint4* _vptr_0 = reinterpret_cast<const uint4*>(C + (slot_base_2 + (unsigned long long)column_5) + 0);
                        uint4 _vld_0[2];
                        #pragma unroll
                        for (int _blk = 0; _blk < 2; _blk++) {
                            _vld_0[_blk] = _vptr_0[_blk];
                            uint32_t* _vpairs_0 = reinterpret_cast<uint32_t*>(&_vld_0[_blk]);
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
                                asm volatile(
                                    "{\n\t"
                                    "shl.b32 %0, %2, 16;\n\t"
                                    "and.b32 %1, %2, 0xffff0000;\n\t"
                                    "}\n"
                                    : "=f"((&_vec_load_2[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_2[0 + _blk * 8 + _pair * 2])[1])
                                    : "r"(_vpairs_0[_pair]));
                            }
                        }
                    }
                    #pragma unroll
                    for (int element_2 = 0; element_2 < 16; element_2++) {
                        accum_2[element_2] = accum_2[element_2] + _vec_load_2[element_2];
                    }
                }
                unsigned long long destination_base_2 = (unsigned long long)source_token_5 * 3072;
                {
                    __nv_bfloat162 _pk[8];
                    _pk[0] = __floats2bfloat162_rn(accum_2[0 + 0], accum_2[0 + 1]);
                    _pk[1] = __floats2bfloat162_rn(accum_2[0 + 2], accum_2[0 + 3]);
                    _pk[2] = __floats2bfloat162_rn(accum_2[0 + 4], accum_2[0 + 5]);
                    _pk[3] = __floats2bfloat162_rn(accum_2[0 + 6], accum_2[0 + 7]);
                    _pk[4] = __floats2bfloat162_rn(accum_2[0 + 8], accum_2[0 + 9]);
                    _pk[5] = __floats2bfloat162_rn(accum_2[0 + 10], accum_2[0 + 11]);
                    _pk[6] = __floats2bfloat162_rn(accum_2[0 + 12], accum_2[0 + 13]);
                    _pk[7] = __floats2bfloat162_rn(accum_2[0 + 14], accum_2[0 + 15]);
                    *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(reinterpret_cast<__nv_bfloat16*>(direct_output_peers[source_rank_5]) + (destination_base_2 + (unsigned long long)column_5)))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                    *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(reinterpret_cast<__nv_bfloat16*>(direct_output_peers[source_rank_5]) + (destination_base_2 + (unsigned long long)column_5)))[8]) = *reinterpret_cast<uint4*>(&_pk[4]);
                }
            }
            asm volatile("barrier.sync 11, 256;" ::: "memory");
        }
    }
    // ---- Role: dispatch_helper ----
    if (warp >= 6 && warp <= 7) {
        { // swap_fc12_03_dispatch_helper_dispatch_join
            int work_6 = bid;
            int valid_6 = ((work_6 < dispatch_work_count) ? 1 : 0);
            int safe_work_6 = ((work_6 < 64) ? work_6 : 0);
            int source_rank_6 = dispatch_work_source_ranks[safe_work_6];
            int source_token_6 = dispatch_work_source_tokens[safe_work_6];
            int local_group_3 = dispatch_work_local_groups[safe_work_6];
            int local_row_3 = dispatch_work_local_rows[safe_work_6];
            asm volatile("barrier.sync 13, 256;" ::: "memory");
            int local_expert_3 = local_group_3 * 8 + warp;
            int recv_token_6 = source_rank_6 * tokens_per_rank + source_token_6;
            int route_3 = recv_token_6 * 8 + warp;
            int packed_row_3 = local_expert_3 * 128 + local_row_3;
            if (valid_6 != 0 && lane == 0) {
                route_weights[route_3] = dispatch_weight_row[warp];
                route_map[packed_row_3] = route_3;
            }
            int shared_packed_row_3 = local_group_3 * 128 + local_row_3;
            if (valid_6 != 0) {
                #pragma unroll 1
                for (int block_column_3 = warp; block_column_3 < 96; block_column_3 += 8) {
                    int column_6 = block_column_3 * 32 + lane;
                    B1_raw[(unsigned long long)shared_packed_row_3 * 3072 + (unsigned long long)column_6] = dispatch_row[column_6];
                    if (lane == 0) {
                        int a_3 = local_row_3 / 32;
                        int row32_3 = local_row_3 - a_3 * 32;
                        int tile_k_3 = block_column_3 / 4;
                        int u_3 = block_column_3 - tile_k_3 * 4;
                        int cp_in_tile_3 = row32_3 * 16 + a_3 * 4 + u_3;
                        unsigned long long cp_index_3 = (unsigned long long)local_group_3 * 24 * 512 + (unsigned long long)tile_k_3 * 512 + (unsigned long long)cp_in_tile_3;
                        SFB1_raw[cp_index_3] = dispatch_scale_row[block_column_3];
                    }
                }
            }
            asm volatile("barrier.sync 12, 256;" ::: "memory");
            asm volatile("barrier.sync 12, 256;" ::: "memory");
        }
        { // swap_fc12_16_dispatch_helper_fc2_exit
            asm volatile("barrier.sync 11, 256;" ::: "memory");
            asm volatile("barrier.sync 11, 256;" ::: "memory");
            int work_7 = bid;
            int valid_7 = ((work_7 < dispatch_work_count) ? 1 : 0);
            int safe_work_7 = ((work_7 < 64) ? work_7 : 0);
            int source_rank_7 = dispatch_work_source_ranks[safe_work_7];
            int source_token_7 = dispatch_work_source_tokens[safe_work_7];
            int recv_token_7 = source_rank_7 * tokens_per_rank + source_token_7;
            int first_route_3 = recv_token_7 * 8;
            float accum_3[16];
            accum_3[0] = 0.0f;
            accum_3[1] = 0.0f;
            accum_3[2] = 0.0f;
            accum_3[3] = 0.0f;
            accum_3[4] = 0.0f;
            accum_3[5] = 0.0f;
            accum_3[6] = 0.0f;
            accum_3[7] = 0.0f;
            accum_3[8] = 0.0f;
            accum_3[9] = 0.0f;
            accum_3[10] = 0.0f;
            accum_3[11] = 0.0f;
            accum_3[12] = 0.0f;
            accum_3[13] = 0.0f;
            accum_3[14] = 0.0f;
            accum_3[15] = 0.0f;
            if (valid_7 != 0 && tid < 192) {
                int column_7 = tid * 16;
                #pragma unroll
                for (int route_slot_3 = 0; route_slot_3 < 8; route_slot_3++) {
                    unsigned long long slot_base_3 = ((unsigned long long)first_route_3 + (unsigned long long)route_slot_3) * 3072;
                    float _vec_load_3[16];
                    {
                        const uint4* _vptr_0 = reinterpret_cast<const uint4*>(C + (slot_base_3 + (unsigned long long)column_7) + 0);
                        uint4 _vld_0[2];
                        #pragma unroll
                        for (int _blk = 0; _blk < 2; _blk++) {
                            _vld_0[_blk] = _vptr_0[_blk];
                            uint32_t* _vpairs_0 = reinterpret_cast<uint32_t*>(&_vld_0[_blk]);
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
                                asm volatile(
                                    "{\n\t"
                                    "shl.b32 %0, %2, 16;\n\t"
                                    "and.b32 %1, %2, 0xffff0000;\n\t"
                                    "}\n"
                                    : "=f"((&_vec_load_3[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_3[0 + _blk * 8 + _pair * 2])[1])
                                    : "r"(_vpairs_0[_pair]));
                            }
                        }
                    }
                    #pragma unroll
                    for (int element_3 = 0; element_3 < 16; element_3++) {
                        accum_3[element_3] = accum_3[element_3] + _vec_load_3[element_3];
                    }
                }
                unsigned long long destination_base_3 = (unsigned long long)source_token_7 * 3072;
                {
                    __nv_bfloat162 _pk[8];
                    _pk[0] = __floats2bfloat162_rn(accum_3[0 + 0], accum_3[0 + 1]);
                    _pk[1] = __floats2bfloat162_rn(accum_3[0 + 2], accum_3[0 + 3]);
                    _pk[2] = __floats2bfloat162_rn(accum_3[0 + 4], accum_3[0 + 5]);
                    _pk[3] = __floats2bfloat162_rn(accum_3[0 + 6], accum_3[0 + 7]);
                    _pk[4] = __floats2bfloat162_rn(accum_3[0 + 8], accum_3[0 + 9]);
                    _pk[5] = __floats2bfloat162_rn(accum_3[0 + 10], accum_3[0 + 11]);
                    _pk[6] = __floats2bfloat162_rn(accum_3[0 + 12], accum_3[0 + 13]);
                    _pk[7] = __floats2bfloat162_rn(accum_3[0 + 14], accum_3[0 + 15]);
                    *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(reinterpret_cast<__nv_bfloat16*>(direct_output_peers[source_rank_7]) + (destination_base_3 + (unsigned long long)column_7)))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                    *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(reinterpret_cast<__nv_bfloat16*>(direct_output_peers[source_rank_7]) + (destination_base_3 + (unsigned long long)column_7)))[8]) = *reinterpret_cast<uint4*>(&_pk[4]);
                }
            }
            asm volatile("barrier.sync 11, 256;" ::: "memory");
        }
    }

    // Cleanup
    __syncthreads(); // barrier before TMEM dealloc

    if (warp == 0) {
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(128));
    }
}

} // extern "C"
