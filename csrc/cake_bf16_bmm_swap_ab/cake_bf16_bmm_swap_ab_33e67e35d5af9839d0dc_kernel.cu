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
#define TMEM_NCOLS 32
#define TMEM_ACCUM_OFFSET 0
#define NUM_TMA_PIPE_STAGES 8
#define NUM_EPI_PIPE_STAGES 1
#define SMEM_SMEM_W_OFF 1024
#define SMEM_SMEM_W_STAGE_BYTES 8192
#define SMEM_SMEM_W_STRIDE 8192
#define SMEM_SMEM_A_OFF 66560
#define SMEM_SMEM_A_STAGE_BYTES 2048
#define SMEM_SMEM_A_STRIDE 2048
#define SMEM_RED_BUF_OFF 82944
#define SMEM_RED_BUF_STAGE_BYTES 4096
#define SMEM_RED_BUF_STRIDE 4096
#define SMEM_STAGING_OFF 87040
#define SMEM_STAGING_STAGE_BYTES 2048
#define SMEM_STAGING_STRIDE 2048
#define SMEM_TOTAL 89088
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


__device__ __forceinline__ void tcgen05_mma_f16(
    int taddr, uint64_t a_desc, uint64_t b_desc,
    uint32_t i_desc, int enable_input_d) {
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, %4, 0;\n\t"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n\t"
        "}\n"
        :: "r"(taddr), "l"(a_desc), "l"(b_desc),
           "r"(i_desc), "r"(enable_input_d)
         : "memory");
}



union MmaSmemDesc {
    uint64_t u64;
    uint32_t u32[2];
};

__device__ __forceinline__ void incr_smem_desc_lo(uint64_t& smem_desc, uint32_t offset) {
    MmaSmemDesc tmp;
    tmp.u64 = smem_desc;
    tmp.u32[0] += offset;
    smem_desc = tmp.u64;
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




__device__ __forceinline__ void tma_4d_gmem2smem(
    int dst, const void *tmap_ptr, int x, int y, int z, int w, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.4d.shared::cta.global"
        ".mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3, %4, %5}], [%6];"
        :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y), "r"(z), "r"(w),
           "r"(mbar_addr) : "memory");
}





__device__ __forceinline__ uint32_t make_warp_uniform(uint32_t val) {
    uint32_t result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1f, 0xffffffff;"
        : "=r"(result) : "r"(val));
    return result;
}

extern "C" {

__global__ __launch_bounds__(256) __cluster_dims__(2,1,1) void
kernel_cake_bf16_bmm_swap_ab_33e67e35d5af9839d0dc(const __grid_constant__ CUtensorMap W, const __grid_constant__ CUtensorMap A, uint8_t* __restrict__ out, int M, int N, int K, int B)
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
    #define tma_empty_addr (mbar_base + 64)
    #define mma_epi_full_addr (mbar_base + 128)
    #define red_bar_addr (mbar_base + 136)
    #define tmem_free_addr (mbar_base + 144)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    const unsigned int clusters_x = gridDim.x / 2;
    const unsigned int cluster_id = ((blockIdx.z * gridDim.y + blockIdx.y) * clusters_x) + blockIdx.x / 2;
    const unsigned int num_clusters = clusters_x * gridDim.y * gridDim.z;

    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));

    // Kernel setup ops
    __nv_bfloat16* smem_w = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int smem_w_addr = smem + 1024;
    __nv_bfloat16* smem_a = reinterpret_cast<__nv_bfloat16*>(smem_raw + 66560);
    const int smem_a_addr = smem + 66560;
    float* red_buf = reinterpret_cast<float*>(smem_raw + 82944);
    const int red_buf_addr = smem + 82944;
    unsigned int* staging = reinterpret_cast<unsigned int*>(smem_raw + 87040);
    const int staging_addr = smem + 87040;
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    if (warp == 0) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&W))) : "memory"); }
    if (warp == 1) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&A))) : "memory"); }
    if (threadIdx.x == 96) {
        const int pf_tiles = K / 64 / 2;
        const int pf_begin = blockIdx.x % 2 * pf_tiles;
        const int pf_w = blockIdx.x / 2 * 64;
        const int pf_t = blockIdx.y * 16;
        const int pf_b = blockIdx.z;
        for (int k_tile = pf_begin; k_tile < pf_begin + pf_tiles; k_tile++) {
            asm volatile("cp.async.bulk.prefetch.tensor.4d.L2.global.tile [%0, {%1, %2, %3, %4}];" :: "l"((uint64_t)((&W))), "r"((int)(0)), "r"((int)(pf_w)), "r"((int)(k_tile)), "r"((int)(pf_b)) : "memory");
            asm volatile("cp.async.bulk.prefetch.tensor.4d.L2.global.tile [%0, {%1, %2, %3, %4}];" :: "l"((uint64_t)((&A))), "r"((int)(0)), "r"((int)(pf_t)), "r"((int)(k_tile)), "r"((int)(pf_b)) : "memory");
        }
    }

    // Mbarrier init (5 pipeline groups, 0 ordered-sequence groups, 19 barriers)
    // Mbarriers at smem_raw[0..152)

    if (warp == 0) {
        // --- pipeline 'tma_pipe' ---
        // tma_full: 8 barriers, init_count=2
        // tma_empty: 8 barriers, init_count=1
        // --- pipeline 'epi_pipe' ---
        // mma_epi_full: 1 barriers, init_count=1
        // red_bar: 1 barriers, init_count=1
        // tmem_free: 1 barriers, init_count=4
        // Warp-cooperative initialization in physical record order.
        uint32_t _mbarrier_init_count_0_0 = 4;
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_0_0) : "r"(lane), "n"(18), "r"((uint32_t)(1)));
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_0_0) : "r"(lane), "n"(8), "r"((uint32_t)(2)));
        if (lane < 19) {
            mbarrier_init(smem + 0 + lane * 8, _mbarrier_init_count_0_0);
        }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }

    __syncthreads();
    asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");

    // TMEM alloc (32 columns, 32 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 152);
    if (warp == 2) {
        int _tmem_hold = smem + 152;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(32) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    if (warp == 2 || warp == 4 || warp == 5 || warp == 6 || warp == 7) {
        asm volatile("barrier.sync.aligned %0, %1;" :: "r"(1), "r"(160) : "memory");
        asm volatile("tcgen05.fence::after_thread_sync;");
    }
    const int taddr = (warp == 2 || warp == 4 || warp == 5 || warp == 6 || warp == 7) ? tmem_addr_storage[0] : 0;

    // Kernel post-init ops
    const int tmem_accum = taddr;

    // ---- Role: dma_w ----
    if (warp == 0) {
        { // dma_w_main
            unsigned int w_stage = 0;
            int w_idx = blockIdx.x / 2 * 64;
            int b_idx = blockIdx.z;
            int k_tiles = K / 64 / 2;
            int k_begin = blockIdx.x % 2 * k_tiles;
            unsigned int _phase_tma_empty = 1;
            if (elect_sync()) {
                asm volatile("griddepcontrol.wait;" ::: "memory");
                #pragma unroll 4
                for (int k_tile_1 = k_begin; k_tile_1 < k_begin + k_tiles; k_tile_1++) {
                    mbarrier_wait(tma_empty_addr + (w_stage) * 8, _phase_tma_empty);
                    tma_4d_gmem2smem(smem_w_addr + w_stage * 8192, (&W), 0, w_idx, k_tile_1, b_idx, tma_full_addr + (w_stage) * 8);
                    mbarrier_arrive_expect_tx(tma_full_addr + (w_stage) * 8, 8192);
                    w_stage += 1;
                    if (w_stage == 8) { w_stage = 0; _phase_tma_empty ^= 1; }
                }
            }
        }
    }
    // ---- Role: dma_a ----
    if (warp == 1) {
        { // dma_a_main
            unsigned int a_stage = 0;
            int t_idx = blockIdx.y * 16;
            int b_idx_a = blockIdx.z;
            int k_tiles_a = K / 64 / 2;
            int k_begin_a = blockIdx.x % 2 * k_tiles_a;
            unsigned int _phase_tma_empty_1 = 1;
            if (elect_sync()) {
                asm volatile("griddepcontrol.wait;" ::: "memory");
                #pragma unroll 4
                for (int k_tile_2 = k_begin_a; k_tile_2 < k_begin_a + k_tiles_a; k_tile_2++) {
                    mbarrier_wait(tma_empty_addr + (a_stage) * 8, _phase_tma_empty_1);
                    tma_4d_gmem2smem(smem_a_addr + a_stage * 2048, (&A), 0, t_idx, k_tile_2, b_idx_a, tma_full_addr + (a_stage) * 8);
                    mbarrier_arrive_expect_tx(tma_full_addr + (a_stage) * 8, 2048);
                    a_stage += 1;
                    if (a_stage == 8) { a_stage = 0; _phase_tma_empty_1 ^= 1; }
                }
            }
        }
    }
    // ---- Role: mma ----
    if (warp == 2) {
        { // mma_main
            unsigned int mma_stage = 0;
            int k_tiles_m = K / 64 / 2;
            unsigned int _phase_tma_full = 0;
            #pragma unroll 2
            for (int k_tile_3 = 0; k_tile_3 < k_tiles_m; k_tile_3++) {
                mbarrier_wait(tma_full_addr + (mma_stage) * 8, _phase_tma_full);
                asm volatile("tcgen05.fence::after_thread_sync;");
                int init_flag = ((k_tile_3 == 0) ? 1 : 0);
                int _mma_a_lo_0 = make_warp_uniform((((smem_w_addr) >> 4) & 0x3FFF) + (mma_stage) * 512);
                int _mma_b_lo_0 = make_warp_uniform((((smem_a_addr) >> 4) & 0x3FFF) + (mma_stage) * 128);
                {
                    uint64_t _mma_ss_a_desc_0 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_0);
                    uint64_t _mma_ss_b_desc_0 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_0);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_accum, _mma_ss_a_desc_0, _mma_ss_b_desc_0, 67372176, ((init_flag) ? 0 : 1));
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_accum, _mma_ss_a_desc_0, _mma_ss_b_desc_0, 67372176, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_accum, _mma_ss_a_desc_0, _mma_ss_b_desc_0, 67372176, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_accum, _mma_ss_a_desc_0, _mma_ss_b_desc_0, 67372176, 1);
                    }
                }
                elect_commit(tma_empty_addr + (mma_stage) * 8);
                mma_stage += 1;
                if (mma_stage == 8) { mma_stage = 0; _phase_tma_full ^= 1; }
            }
            elect_commit(mma_epi_full_addr);
            mbarrier_wait(tmem_free_addr, 0);
            asm volatile("tcgen05.fence::after_thread_sync;");
            asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(32));
        }
    }
    // ---- Role: epilogue ----
    if (warp >= 4 && warp <= 7) {
        { // epilogue_main
            asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
            asm volatile("tcgen05.fence::after_thread_sync;");
            const int epi_warp = warp % 4;
            const int epi_tid = epi_warp * 32 + lane;
            int row_local = epi_warp * 16 + lane % 16;
            int col_half = lane / 16;
            int n_row = blockIdx.x / 2 * 64 + row_local;
            int m_first = blockIdx.y * 16 + col_half * 8;
            int b_idx_e = blockIdx.z;
            mbarrier_wait(mma_epi_full_addr, 0);
            asm volatile("tcgen05.fence::after_thread_sync;");
            unsigned int tmem_addr = taddr + (unsigned int)(epi_warp * 32 << 16);
            float _tmem_load_0[8];
            asm volatile(
                "tcgen05.ld.sync.aligned.16x32bx2.x8.b32"
                " {%0, %1, %2, %3, %4, %5, %6, %7}, [%8], 8;"
                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[7]))
                : "r"(tmem_addr));
            asm volatile("tcgen05.wait::ld.sync.aligned;");
            asm volatile("tcgen05.fence::before_thread_sync;");
            __syncwarp();
            if (elect_sync()) {
                mbarrier_arrive(tmem_free_addr);
            }
            if (cta_rank == 0 && warp == 4) {
                if (elect_sync()) {
                    mbarrier_arrive_expect_tx(red_bar_addr, 4096);
                }
            }
            int slot_off = ((cta_rank - 1) * 1024 + epi_tid * 4) * 4;
            uint32_t _mapa_0;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_0) : "r"(red_buf_addr + (unsigned int)slot_off), "r"(0));
            uint32_t _mapa_1;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_1) : "r"(red_bar_addr), "r"(0));
            if (cta_rank != 0) {
                asm volatile(
                    "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                    :: "r"(_mapa_0), "r"(__float_as_uint(_tmem_load_0[0])), "r"(__float_as_uint(_tmem_load_0[1])), "r"(__float_as_uint(_tmem_load_0[2])), "r"(__float_as_uint(_tmem_load_0[3])), "r"(_mapa_1) : "memory");
                asm volatile(
                    "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                    :: "r"(_mapa_0 + 2048), "r"(__float_as_uint(_tmem_load_0[4])), "r"(__float_as_uint(_tmem_load_0[5])), "r"(__float_as_uint(_tmem_load_0[6])), "r"(__float_as_uint(_tmem_load_0[7])), "r"(_mapa_1) : "memory");
            } else {
                mbarrier_wait_cluster_hint(red_bar_addr, 0, 10000000);
                float part[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&part[0])), "=r"(*reinterpret_cast<uint32_t*>(&part[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&part[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&part[(0) + 3]))
                    : "r"(red_buf_addr + (unsigned int)(epi_tid * 4 * 4)));
                float part_0[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&part_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&part_0[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&part_0[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&part_0[(0) + 3]))
                    : "r"(red_buf_addr + (unsigned int)((128 + epi_tid) * 4 * 4)));
                {
                    __nv_bfloat16 _bval_0 = __float2bfloat16_rn(_tmem_load_0[0] + part[0]);
                    uint16_t _bits_0 = *(uint16_t*)&_bval_0;
                    uint32_t _addr_0 = static_cast<uint32_t>(staging_addr + (unsigned int)((col_half * 8 * 64 + row_local) * 2));
                    asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_0), "h"(_bits_0));
                }
                {
                    __nv_bfloat16 _bval_1 = __float2bfloat16_rn(_tmem_load_0[1] + part[1]);
                    uint16_t _bits_1 = *(uint16_t*)&_bval_1;
                    uint32_t _addr_1 = static_cast<uint32_t>(staging_addr + (unsigned int)(((col_half * 8 + 1) * 64 + row_local) * 2));
                    asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_1), "h"(_bits_1));
                }
                {
                    __nv_bfloat16 _bval_2 = __float2bfloat16_rn(_tmem_load_0[2] + part[2]);
                    uint16_t _bits_2 = *(uint16_t*)&_bval_2;
                    uint32_t _addr_2 = static_cast<uint32_t>(staging_addr + (unsigned int)(((col_half * 8 + 2) * 64 + row_local) * 2));
                    asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_2), "h"(_bits_2));
                }
                {
                    __nv_bfloat16 _bval_3 = __float2bfloat16_rn(_tmem_load_0[3] + part[3]);
                    uint16_t _bits_3 = *(uint16_t*)&_bval_3;
                    uint32_t _addr_3 = static_cast<uint32_t>(staging_addr + (unsigned int)(((col_half * 8 + 3) * 64 + row_local) * 2));
                    asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_3), "h"(_bits_3));
                }
                {
                    __nv_bfloat16 _bval_4 = __float2bfloat16_rn(_tmem_load_0[4] + part_0[0]);
                    uint16_t _bits_4 = *(uint16_t*)&_bval_4;
                    uint32_t _addr_4 = static_cast<uint32_t>(staging_addr + (unsigned int)(((col_half * 8 + 4) * 64 + row_local) * 2));
                    asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_4), "h"(_bits_4));
                }
                {
                    __nv_bfloat16 _bval_5 = __float2bfloat16_rn(_tmem_load_0[5] + part_0[1]);
                    uint16_t _bits_5 = *(uint16_t*)&_bval_5;
                    uint32_t _addr_5 = static_cast<uint32_t>(staging_addr + (unsigned int)(((col_half * 8 + 5) * 64 + row_local) * 2));
                    asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_5), "h"(_bits_5));
                }
                {
                    __nv_bfloat16 _bval_6 = __float2bfloat16_rn(_tmem_load_0[6] + part_0[2]);
                    uint16_t _bits_6 = *(uint16_t*)&_bval_6;
                    uint32_t _addr_6 = static_cast<uint32_t>(staging_addr + (unsigned int)(((col_half * 8 + 6) * 64 + row_local) * 2));
                    asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_6), "h"(_bits_6));
                }
                {
                    __nv_bfloat16 _bval_7 = __float2bfloat16_rn(_tmem_load_0[7] + part_0[3]);
                    uint16_t _bits_7 = *(uint16_t*)&_bval_7;
                    uint32_t _addr_7 = static_cast<uint32_t>(staging_addr + (unsigned int)(((col_half * 8 + 7) * 64 + row_local) * 2));
                    asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_7), "h"(_bits_7));
                }
                asm volatile("bar.sync 2, 128;" ::: "memory");
                int c = epi_tid;
                int t_row = c / 8;
                int m_row = blockIdx.y * 16 + t_row;
                int n_col = blockIdx.x / 2 * 64 + c % 8 * 8;
                if (m_row < M && n_col < N) {
                    unsigned int words[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&words[0])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 3]))
                        : "r"(staging_addr + (unsigned int)(c * 16)));
                    reinterpret_cast<int4*>(reinterpret_cast<unsigned int*>(out + (((long long)(b_idx_e * M + m_row) * (long long)N + (long long)n_col) * 2)))[0] = reinterpret_cast<int4*>(words)[0];
                }
            }
        }
    }

    // Cleanup
}

} // extern "C"
