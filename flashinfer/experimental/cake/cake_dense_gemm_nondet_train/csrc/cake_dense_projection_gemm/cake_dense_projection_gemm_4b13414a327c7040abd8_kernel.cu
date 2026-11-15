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





__device__ __forceinline__ void tma_3d_gmem2smem(
    int dst, const void *tmap_ptr, int x, int y, int z, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cta.global"
        ".mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3, %4}], [%5];"
        :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y), "r"(z),
           "r"(mbar_addr) : "memory");
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

extern "C" {

__global__ __launch_bounds__(192, LAUNCH_MIN_BLOCKS) __cluster_dims__(3,1,1) void
kernel_cake_dense_projection_gemm_4b13414a327c7040abd8(const __grid_constant__ CUtensorMap A, const __grid_constant__ CUtensorMap B, float* __restrict__ out, int M, int N, int K, int num_l, int ldo, int out_l)
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
    #define mma_done_addr (mbar_base + 80)
    #define acc_full_addr (mbar_base + 160)
    #define tmem_free_addr (mbar_base + 168)
    #define red_bar_addr (mbar_base + 176)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    const unsigned int clusters_x = gridDim.x / 3;
    const unsigned int cluster_id = ((blockIdx.z * gridDim.y + blockIdx.y) * clusters_x) + blockIdx.x / 3;
    const unsigned int num_clusters = clusters_x * gridDim.y * gridDim.z;

    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));

    // Prologue
    __nv_bfloat16* smem_a = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int smem_a_addr = smem + 1024;
    __nv_bfloat16* smem_b = reinterpret_cast<__nv_bfloat16*>(smem_raw + 17408);
    const int smem_b_addr = smem + 17408;
    float* red_buf = reinterpret_cast<float*>(smem_raw + 205824);
    const int red_buf_addr = smem + 205824;

    // Kernel setup ops
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");

    // Mbarrier init (23 barriers at smem_raw[0..184))

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'tma_pipe' ---
            // tma_full: 10 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            mbarrier_init(smem + 8, 1);
            mbarrier_init(smem + 16, 1);
            mbarrier_init(smem + 24, 1);
            mbarrier_init(smem + 32, 1);
            mbarrier_init(smem + 40, 1);
            mbarrier_init(smem + 48, 1);
            mbarrier_init(smem + 56, 1);
            mbarrier_init(smem + 64, 1);
            mbarrier_init(smem + 72, 1);
            // mma_done: 10 barriers, init_count=1
            mbarrier_init(smem + 80, 1);
            mbarrier_init(smem + 88, 1);
            mbarrier_init(smem + 96, 1);
            mbarrier_init(smem + 104, 1);
            mbarrier_init(smem + 112, 1);
            mbarrier_init(smem + 120, 1);
            mbarrier_init(smem + 128, 1);
            mbarrier_init(smem + 136, 1);
            mbarrier_init(smem + 144, 1);
            mbarrier_init(smem + 152, 1);
            // acc_full: 1 barriers, init_count=1
            mbarrier_init(smem + 160, 1);
            // tmem_free: 1 barriers, init_count=4
            mbarrier_init(smem + 168, 4);
            // red_bar: 1 barriers, init_count=1
            mbarrier_init(smem + 176, 1);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncthreads();
    asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");

    // TMEM alloc (32 columns, 32 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 184);
    if (warp == 1) {
        int _tmem_hold = smem + 184;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(32) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    if (warp != 0) {
        asm volatile("barrier.sync.aligned %0, %1;" :: "r"(1), "r"(160) : "memory");
        asm volatile("tcgen05.fence::after_thread_sync;");
    }
    const unsigned int taddr = (warp >= 1 && warp <= 5) ? tmem_addr_storage[0] : 0;
    const unsigned int tmem_accum = taddr;

    // ---- Role: load ----
    if (warp == 0) {
        { // load_main
            unsigned int load_stage = 0;
            int m_tiles = (M + 128 - 1) / 128;
            int n_tiles = (N + 32 - 1) / 32;
            int k_iters = (K + 64 - 1) / 64;
            int tile = (int)blockIdx.x / 3;
            int tiles_l = m_tiles * n_tiles;
            int l_idx = tile / tiles_l;
            int in_l = tile - l_idx * tiles_l;
            int n_tile = in_l / m_tiles;
            int m_tile = in_l - n_tile * m_tiles;
            int split = (int)cta_rank;
            int k_begin = split * k_iters / 3;
            int k_end = (split + 1) * k_iters / 3;
            int a_row = m_tile * 128;
            int b_col = n_tile * 32;
            unsigned int _phase_mma_done = 1;
            if (elect_sync()) {
                asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&A))) : "memory");
                asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&B))) : "memory");
                asm volatile("griddepcontrol.wait;" ::: "memory");
                #pragma unroll 1
                for (int k_block = k_begin; k_block < k_end; k_block++) {
                    mbarrier_wait(mma_done_addr + (load_stage) * 8, _phase_mma_done);
                    int k0 = k_block * 64;
                    asm volatile(
                        "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                        " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                        :: "r"(smem_a_addr + load_stage * 20480), "l"((&A)), "r"(a_row), "r"(k0), "r"(l_idx),
                           "r"(tma_full_addr + (load_stage) * 8), "l"(0x12F0000000000000ULL) : "memory");
                    asm volatile(
                        "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                        " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                        :: "r"(smem_a_addr + load_stage * 20480 + 8192), "l"((&A)), "r"(a_row + 64), "r"(k0), "r"(l_idx),
                           "r"(tma_full_addr + (load_stage) * 8), "l"(0x12F0000000000000ULL) : "memory");
                    tma_3d_gmem2smem(smem_b_addr + load_stage * 20480, (&B), b_col, k0, l_idx, tma_full_addr + (load_stage) * 8);
                    mbarrier_arrive_expect_tx(tma_full_addr + (load_stage) * 8, 20480);
                    load_stage += 1;
                    if (load_stage == 10) { load_stage = 0; _phase_mma_done ^= 1; }
                }
                #pragma unroll
                for (int _drain = 0; _drain < 10; _drain++) {
                    mbarrier_wait(mma_done_addr + (load_stage) * 8, _phase_mma_done);
                    load_stage += 1;
                    if (load_stage == 10) { load_stage = 0; _phase_mma_done ^= 1; }
                }
            }
            asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
            asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
            asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
        }
    }
    // ---- Role: mma ----
    if (warp == 1) {
        { // mma_main
            unsigned int mma_stage = 0;
            int k_iters_m = (K + 64 - 1) / 64;
            int split_1 = (int)cta_rank;
            int k_begin_1 = split_1 * k_iters_m / 3;
            int k_end_1 = (split_1 + 1) * k_iters_m / 3;
            unsigned int _phase_tma_full = 0;
            #pragma unroll 1
            for (int k_block_1 = k_begin_1; k_block_1 < k_end_1; k_block_1++) {
                mbarrier_wait(tma_full_addr + (mma_stage) * 8, _phase_tma_full);
                asm volatile("tcgen05.fence::after_thread_sync;");
                int init_flag = ((k_block_1 == k_begin_1) ? 1 : 0);
                int a_lo = ((((smem_a_addr) >> 4) & 0x3FFF) | 0x2000000) + (mma_stage) * 1280;
                int b_lo = ((((smem_b_addr) >> 4) & 0x3FFF) | 0x1000000) + (mma_stage) * 1280;
                uint32_t _elected_0 = elect_sync();
                asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "setp.ne.b32 leader, %4, 0;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x80004020;\n\t"
                    "mov.b32 id, 134841488;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 64;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 64;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 64;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(a_lo), "r"(b_lo), "r"(tmem_accum), "r"(((init_flag) ? 0 : 1)), "r"(_elected_0) : "memory");
                tcgen05_commit_leader(mma_done_addr + (mma_stage) * 8, _elected_0);
                mma_stage += 1;
                if (mma_stage == 10) { mma_stage = 0; _phase_tma_full ^= 1; }
            }
            elect_commit(acc_full_addr);
            mbarrier_wait(tmem_free_addr, 0);
            asm volatile("tcgen05.fence::after_thread_sync;");
            asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(32));
            asm volatile("tcgen05.fence::before_thread_sync;");
            asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
            asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
            asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
        }
    }
    // ---- Role: epilogue ----
    if (warp >= 2 && warp <= 5) {
        { // epilogue_main
            asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
            const int quarter = warp % 4;
            const int tid_1 = quarter * 32 + lane;
            int m_tiles_e = (M + 128 - 1) / 128;
            int n_tiles_e = (N + 32 - 1) / 32;
            int tile_1 = (int)blockIdx.x / 3;
            int tiles_l_1 = m_tiles_e * n_tiles_e;
            int l_idx_1 = tile_1 / tiles_l_1;
            int in_l_1 = tile_1 - l_idx_1 * tiles_l_1;
            int n_tile_1 = in_l_1 / m_tiles_e;
            int m_tile_1 = in_l_1 - n_tile_1 * m_tiles_e;
            int row = m_tile_1 * 128 + tid_1;
            int col0 = n_tile_1 * 32;
            long long base_t = (long long)l_idx_1 * (long long)out_l + (long long)row;
            float vals[32];
            if (warp == 2) {
                if (elect_sync()) {
                    if (cta_rank == 0) {
                        mbarrier_arrive_expect_tx(red_bar_addr, 8192);
                    }
                    if (cta_rank == 1) {
                        mbarrier_arrive_expect_tx(red_bar_addr, 12288);
                    }
                    if (cta_rank == 2) {
                        mbarrier_arrive_expect_tx(red_bar_addr, 12288);
                    }
                }
            }
            asm volatile("griddepcontrol.wait;" ::: "memory");
            mbarrier_wait(acc_full_addr, 0);
            asm volatile("tcgen05.fence::after_thread_sync;");
            unsigned int lane_addr = taddr + (unsigned int)(quarter * 32 << 16);
            asm volatile(
                "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                : "=f"(vals[0]), "=f"(vals[1]), "=f"(vals[2]), "=f"(vals[3]), "=f"(vals[4]), "=f"(vals[5]), "=f"(vals[6]), "=f"(vals[7]), "=f"(vals[8]), "=f"(vals[9]), "=f"(vals[10]), "=f"(vals[11]), "=f"(vals[12]), "=f"(vals[13]), "=f"(vals[14]), "=f"(vals[15]), "=f"(vals[16]), "=f"(vals[17]), "=f"(vals[18]), "=f"(vals[19]), "=f"(vals[20]), "=f"(vals[21]), "=f"(vals[22]), "=f"(vals[23]), "=f"(vals[24]), "=f"(vals[25]), "=f"(vals[26]), "=f"(vals[27]), "=f"(vals[28]), "=f"(vals[29]), "=f"(vals[30]), "=f"(vals[31])
                : "r"(lane_addr));
            asm volatile("tcgen05.wait::ld.sync.aligned;");
            asm volatile("tcgen05.fence::before_thread_sync;");
            __syncwarp();
            if (elect_sync()) {
                mbarrier_arrive(tmem_free_addr);
            }
            if (cta_rank != 0) {
                int slot = (int)cta_rank - (((int)cta_rank > 0) ? 1 : 0);
                uint32_t dst;
                asm volatile(
                    "mapa.shared::cluster.u32 %0, %1, %2;"
                    : "=r"(dst) : "r"(red_buf_addr + (unsigned int)((slot * 3 * 32 * 4 + tid_1) * 16)), "r"(0));
                uint32_t _mbar_peer_0;
                asm(
                    "mapa.shared::cluster.u32 %0, %1, %2;"
                    : "=r"(_mbar_peer_0) : "r"(red_bar_addr), "r"(0));
                asm volatile(
                    "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                    :: "r"(dst), "r"(__float_as_uint(vals[0])), "r"(__float_as_uint(vals[1])), "r"(__float_as_uint(vals[2])), "r"(__float_as_uint(vals[3])), "r"(_mbar_peer_0) : "memory");
                uint32_t _mbar_peer_1;
                asm(
                    "mapa.shared::cluster.u32 %0, %1, %2;"
                    : "=r"(_mbar_peer_1) : "r"(red_bar_addr), "r"(0));
                asm volatile(
                    "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                    :: "r"(dst + 2048), "r"(__float_as_uint(vals[4])), "r"(__float_as_uint(vals[5])), "r"(__float_as_uint(vals[6])), "r"(__float_as_uint(vals[7])), "r"(_mbar_peer_1) : "memory");
            }
            if (cta_rank != 1) {
                int slot_1 = (int)cta_rank - (((int)cta_rank > 1) ? 1 : 0);
                uint32_t dst_1;
                asm volatile(
                    "mapa.shared::cluster.u32 %0, %1, %2;"
                    : "=r"(dst_1) : "r"(red_buf_addr + (unsigned int)((slot_1 * 3 * 32 * 4 + tid_1) * 16)), "r"(1));
                uint32_t _mbar_peer_2;
                asm(
                    "mapa.shared::cluster.u32 %0, %1, %2;"
                    : "=r"(_mbar_peer_2) : "r"(red_bar_addr), "r"(1));
                asm volatile(
                    "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                    :: "r"(dst_1), "r"(__float_as_uint(vals[8])), "r"(__float_as_uint(vals[9])), "r"(__float_as_uint(vals[10])), "r"(__float_as_uint(vals[11])), "r"(_mbar_peer_2) : "memory");
                uint32_t _mbar_peer_3;
                asm(
                    "mapa.shared::cluster.u32 %0, %1, %2;"
                    : "=r"(_mbar_peer_3) : "r"(red_bar_addr), "r"(1));
                asm volatile(
                    "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                    :: "r"(dst_1 + 2048), "r"(__float_as_uint(vals[12])), "r"(__float_as_uint(vals[13])), "r"(__float_as_uint(vals[14])), "r"(__float_as_uint(vals[15])), "r"(_mbar_peer_3) : "memory");
                uint32_t _mbar_peer_4;
                asm(
                    "mapa.shared::cluster.u32 %0, %1, %2;"
                    : "=r"(_mbar_peer_4) : "r"(red_bar_addr), "r"(1));
                asm volatile(
                    "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                    :: "r"(dst_1 + 4096), "r"(__float_as_uint(vals[16])), "r"(__float_as_uint(vals[17])), "r"(__float_as_uint(vals[18])), "r"(__float_as_uint(vals[19])), "r"(_mbar_peer_4) : "memory");
            }
            if (cta_rank != 2) {
                int slot_2 = (int)cta_rank - (((int)cta_rank > 2) ? 1 : 0);
                uint32_t dst_2;
                asm volatile(
                    "mapa.shared::cluster.u32 %0, %1, %2;"
                    : "=r"(dst_2) : "r"(red_buf_addr + (unsigned int)((slot_2 * 3 * 32 * 4 + tid_1) * 16)), "r"(2));
                uint32_t _mbar_peer_5;
                asm(
                    "mapa.shared::cluster.u32 %0, %1, %2;"
                    : "=r"(_mbar_peer_5) : "r"(red_bar_addr), "r"(2));
                asm volatile(
                    "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                    :: "r"(dst_2), "r"(__float_as_uint(vals[20])), "r"(__float_as_uint(vals[21])), "r"(__float_as_uint(vals[22])), "r"(__float_as_uint(vals[23])), "r"(_mbar_peer_5) : "memory");
                uint32_t _mbar_peer_6;
                asm(
                    "mapa.shared::cluster.u32 %0, %1, %2;"
                    : "=r"(_mbar_peer_6) : "r"(red_bar_addr), "r"(2));
                asm volatile(
                    "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                    :: "r"(dst_2 + 2048), "r"(__float_as_uint(vals[24])), "r"(__float_as_uint(vals[25])), "r"(__float_as_uint(vals[26])), "r"(__float_as_uint(vals[27])), "r"(_mbar_peer_6) : "memory");
                uint32_t _mbar_peer_7;
                asm(
                    "mapa.shared::cluster.u32 %0, %1, %2;"
                    : "=r"(_mbar_peer_7) : "r"(red_bar_addr), "r"(2));
                asm volatile(
                    "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                    :: "r"(dst_2 + 4096), "r"(__float_as_uint(vals[28])), "r"(__float_as_uint(vals[29])), "r"(__float_as_uint(vals[30])), "r"(__float_as_uint(vals[31])), "r"(_mbar_peer_7) : "memory");
            }
            mbarrier_wait_cluster_hint(red_bar_addr, 0, 10000000);
            asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
            if (cta_rank == 0) {
                float total[12];
                float part[4];
                total[0] = vals[0];
                total[1] = vals[1];
                total[2] = vals[2];
                total[3] = vals[3];
                unsigned int addr_0 = red_buf_addr + (unsigned int)(tid_1 * 16);
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&part[0])), "=r"(*reinterpret_cast<uint32_t*>(&part[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&part[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&part[(0) + 3]))
                    : "r"(addr_0));
                total[0] = total[0] + part[0];
                total[1] = total[1] + part[1];
                total[2] = total[2] + part[2];
                total[3] = total[3] + part[3];
                unsigned int addr_1 = red_buf_addr + (unsigned int)((384 + tid_1) * 16);
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&part[0])), "=r"(*reinterpret_cast<uint32_t*>(&part[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&part[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&part[(0) + 3]))
                    : "r"(addr_1));
                total[0] = total[0] + part[0];
                total[1] = total[1] + part[1];
                total[2] = total[2] + part[2];
                total[3] = total[3] + part[3];
                total[4] = vals[4];
                total[5] = vals[5];
                total[6] = vals[6];
                total[7] = vals[7];
                unsigned int addr_2 = red_buf_addr + (unsigned int)((128 + tid_1) * 16);
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&part[0])), "=r"(*reinterpret_cast<uint32_t*>(&part[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&part[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&part[(0) + 3]))
                    : "r"(addr_2));
                total[4] = total[4] + part[0];
                total[5] = total[5] + part[1];
                total[6] = total[6] + part[2];
                total[7] = total[7] + part[3];
                unsigned int addr_3 = red_buf_addr + (unsigned int)((512 + tid_1) * 16);
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&part[0])), "=r"(*reinterpret_cast<uint32_t*>(&part[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&part[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&part[(0) + 3]))
                    : "r"(addr_3));
                total[4] = total[4] + part[0];
                total[5] = total[5] + part[1];
                total[6] = total[6] + part[2];
                total[7] = total[7] + part[3];
                if (row < M) {
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t + (long long)col0 * (long long)ldo))), "f"(total[(0) + 0]), "r"((unsigned int)(col0 < N)) : "memory");
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t + (long long)(col0 + 1) * (long long)ldo))), "f"(total[(1) + 0]), "r"((unsigned int)(col0 + 1 < N)) : "memory");
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t + (long long)(col0 + 2) * (long long)ldo))), "f"(total[(2) + 0]), "r"((unsigned int)(col0 + 2 < N)) : "memory");
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t + (long long)(col0 + 3) * (long long)ldo))), "f"(total[(3) + 0]), "r"((unsigned int)(col0 + 3 < N)) : "memory");
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t + (long long)(col0 + 4) * (long long)ldo))), "f"(total[(4) + 0]), "r"((unsigned int)(col0 + 4 < N)) : "memory");
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t + (long long)(col0 + 4 + 1) * (long long)ldo))), "f"(total[(5) + 0]), "r"((unsigned int)(col0 + 4 + 1 < N)) : "memory");
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t + (long long)(col0 + 4 + 2) * (long long)ldo))), "f"(total[(6) + 0]), "r"((unsigned int)(col0 + 4 + 2 < N)) : "memory");
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t + (long long)(col0 + 4 + 3) * (long long)ldo))), "f"(total[(7) + 0]), "r"((unsigned int)(col0 + 4 + 3 < N)) : "memory");
                }
            }
            if (cta_rank == 1) {
                float total_1[12];
                float part_1[4];
                unsigned int addr_4 = red_buf_addr + (unsigned int)(tid_1 * 16);
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&total_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&total_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&total_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&total_1[(0) + 3]))
                    : "r"(addr_4));
                total_1[0] = total_1[0] + vals[8];
                total_1[1] = total_1[1] + vals[9];
                total_1[2] = total_1[2] + vals[10];
                total_1[3] = total_1[3] + vals[11];
                unsigned int addr_5 = red_buf_addr + (unsigned int)((384 + tid_1) * 16);
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&part_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&part_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&part_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&part_1[(0) + 3]))
                    : "r"(addr_5));
                total_1[0] = total_1[0] + part_1[0];
                total_1[1] = total_1[1] + part_1[1];
                total_1[2] = total_1[2] + part_1[2];
                total_1[3] = total_1[3] + part_1[3];
                unsigned int addr_6 = red_buf_addr + (unsigned int)((128 + tid_1) * 16);
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&total_1[4])), "=r"(*reinterpret_cast<uint32_t*>(&total_1[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&total_1[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&total_1[(4) + 3]))
                    : "r"(addr_6));
                total_1[4] = total_1[4] + vals[12];
                total_1[5] = total_1[5] + vals[13];
                total_1[6] = total_1[6] + vals[14];
                total_1[7] = total_1[7] + vals[15];
                unsigned int addr_7 = red_buf_addr + (unsigned int)((512 + tid_1) * 16);
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&part_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&part_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&part_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&part_1[(0) + 3]))
                    : "r"(addr_7));
                total_1[4] = total_1[4] + part_1[0];
                total_1[5] = total_1[5] + part_1[1];
                total_1[6] = total_1[6] + part_1[2];
                total_1[7] = total_1[7] + part_1[3];
                unsigned int addr_8 = red_buf_addr + (unsigned int)((256 + tid_1) * 16);
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&total_1[8])), "=r"(*reinterpret_cast<uint32_t*>(&total_1[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&total_1[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&total_1[(8) + 3]))
                    : "r"(addr_8));
                total_1[8] = total_1[8] + vals[16];
                total_1[9] = total_1[9] + vals[17];
                total_1[10] = total_1[10] + vals[18];
                total_1[11] = total_1[11] + vals[19];
                unsigned int addr_9 = red_buf_addr + (unsigned int)((640 + tid_1) * 16);
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&part_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&part_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&part_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&part_1[(0) + 3]))
                    : "r"(addr_9));
                total_1[8] = total_1[8] + part_1[0];
                total_1[9] = total_1[9] + part_1[1];
                total_1[10] = total_1[10] + part_1[2];
                total_1[11] = total_1[11] + part_1[3];
                if (row < M) {
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t + (long long)(col0 + 8) * (long long)ldo))), "f"(total_1[(0) + 0]), "r"((unsigned int)(col0 + 8 < N)) : "memory");
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t + (long long)(col0 + 8 + 1) * (long long)ldo))), "f"(total_1[(1) + 0]), "r"((unsigned int)(col0 + 8 + 1 < N)) : "memory");
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t + (long long)(col0 + 8 + 2) * (long long)ldo))), "f"(total_1[(2) + 0]), "r"((unsigned int)(col0 + 8 + 2 < N)) : "memory");
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t + (long long)(col0 + 8 + 3) * (long long)ldo))), "f"(total_1[(3) + 0]), "r"((unsigned int)(col0 + 8 + 3 < N)) : "memory");
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t + (long long)(col0 + 12) * (long long)ldo))), "f"(total_1[(4) + 0]), "r"((unsigned int)(col0 + 12 < N)) : "memory");
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t + (long long)(col0 + 12 + 1) * (long long)ldo))), "f"(total_1[(5) + 0]), "r"((unsigned int)(col0 + 12 + 1 < N)) : "memory");
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t + (long long)(col0 + 12 + 2) * (long long)ldo))), "f"(total_1[(6) + 0]), "r"((unsigned int)(col0 + 12 + 2 < N)) : "memory");
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t + (long long)(col0 + 12 + 3) * (long long)ldo))), "f"(total_1[(7) + 0]), "r"((unsigned int)(col0 + 12 + 3 < N)) : "memory");
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t + (long long)(col0 + 16) * (long long)ldo))), "f"(total_1[(8) + 0]), "r"((unsigned int)(col0 + 16 < N)) : "memory");
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t + (long long)(col0 + 16 + 1) * (long long)ldo))), "f"(total_1[(9) + 0]), "r"((unsigned int)(col0 + 16 + 1 < N)) : "memory");
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t + (long long)(col0 + 16 + 2) * (long long)ldo))), "f"(total_1[(10) + 0]), "r"((unsigned int)(col0 + 16 + 2 < N)) : "memory");
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t + (long long)(col0 + 16 + 3) * (long long)ldo))), "f"(total_1[(11) + 0]), "r"((unsigned int)(col0 + 16 + 3 < N)) : "memory");
                }
            }
            if (cta_rank == 2) {
                float total_2[12];
                float part_2[4];
                unsigned int addr_10 = red_buf_addr + (unsigned int)(tid_1 * 16);
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&total_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&total_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&total_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&total_2[(0) + 3]))
                    : "r"(addr_10));
                unsigned int addr_11 = red_buf_addr + (unsigned int)((384 + tid_1) * 16);
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&part_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&part_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&part_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&part_2[(0) + 3]))
                    : "r"(addr_11));
                total_2[0] = total_2[0] + part_2[0];
                total_2[1] = total_2[1] + part_2[1];
                total_2[2] = total_2[2] + part_2[2];
                total_2[3] = total_2[3] + part_2[3];
                total_2[0] = total_2[0] + vals[20];
                total_2[1] = total_2[1] + vals[21];
                total_2[2] = total_2[2] + vals[22];
                total_2[3] = total_2[3] + vals[23];
                unsigned int addr_12 = red_buf_addr + (unsigned int)((128 + tid_1) * 16);
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&total_2[4])), "=r"(*reinterpret_cast<uint32_t*>(&total_2[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&total_2[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&total_2[(4) + 3]))
                    : "r"(addr_12));
                unsigned int addr_13 = red_buf_addr + (unsigned int)((512 + tid_1) * 16);
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&part_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&part_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&part_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&part_2[(0) + 3]))
                    : "r"(addr_13));
                total_2[4] = total_2[4] + part_2[0];
                total_2[5] = total_2[5] + part_2[1];
                total_2[6] = total_2[6] + part_2[2];
                total_2[7] = total_2[7] + part_2[3];
                total_2[4] = total_2[4] + vals[24];
                total_2[5] = total_2[5] + vals[25];
                total_2[6] = total_2[6] + vals[26];
                total_2[7] = total_2[7] + vals[27];
                unsigned int addr_14 = red_buf_addr + (unsigned int)((256 + tid_1) * 16);
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&total_2[8])), "=r"(*reinterpret_cast<uint32_t*>(&total_2[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&total_2[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&total_2[(8) + 3]))
                    : "r"(addr_14));
                unsigned int addr_15 = red_buf_addr + (unsigned int)((640 + tid_1) * 16);
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&part_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&part_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&part_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&part_2[(0) + 3]))
                    : "r"(addr_15));
                total_2[8] = total_2[8] + part_2[0];
                total_2[9] = total_2[9] + part_2[1];
                total_2[10] = total_2[10] + part_2[2];
                total_2[11] = total_2[11] + part_2[3];
                total_2[8] = total_2[8] + vals[28];
                total_2[9] = total_2[9] + vals[29];
                total_2[10] = total_2[10] + vals[30];
                total_2[11] = total_2[11] + vals[31];
                if (row < M) {
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t + (long long)(col0 + 20) * (long long)ldo))), "f"(total_2[(0) + 0]), "r"((unsigned int)(col0 + 20 < N)) : "memory");
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t + (long long)(col0 + 20 + 1) * (long long)ldo))), "f"(total_2[(1) + 0]), "r"((unsigned int)(col0 + 20 + 1 < N)) : "memory");
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t + (long long)(col0 + 20 + 2) * (long long)ldo))), "f"(total_2[(2) + 0]), "r"((unsigned int)(col0 + 20 + 2 < N)) : "memory");
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t + (long long)(col0 + 20 + 3) * (long long)ldo))), "f"(total_2[(3) + 0]), "r"((unsigned int)(col0 + 20 + 3 < N)) : "memory");
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t + (long long)(col0 + 24) * (long long)ldo))), "f"(total_2[(4) + 0]), "r"((unsigned int)(col0 + 24 < N)) : "memory");
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t + (long long)(col0 + 24 + 1) * (long long)ldo))), "f"(total_2[(5) + 0]), "r"((unsigned int)(col0 + 24 + 1 < N)) : "memory");
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t + (long long)(col0 + 24 + 2) * (long long)ldo))), "f"(total_2[(6) + 0]), "r"((unsigned int)(col0 + 24 + 2 < N)) : "memory");
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t + (long long)(col0 + 24 + 3) * (long long)ldo))), "f"(total_2[(7) + 0]), "r"((unsigned int)(col0 + 24 + 3 < N)) : "memory");
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t + (long long)(col0 + 28) * (long long)ldo))), "f"(total_2[(8) + 0]), "r"((unsigned int)(col0 + 28 < N)) : "memory");
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t + (long long)(col0 + 28 + 1) * (long long)ldo))), "f"(total_2[(9) + 0]), "r"((unsigned int)(col0 + 28 + 1 < N)) : "memory");
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t + (long long)(col0 + 28 + 2) * (long long)ldo))), "f"(total_2[(10) + 0]), "r"((unsigned int)(col0 + 28 + 2 < N)) : "memory");
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t + (long long)(col0 + 28 + 3) * (long long)ldo))), "f"(total_2[(11) + 0]), "r"((unsigned int)(col0 + 28 + 3 < N)) : "memory");
                }
            }
            asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
        }
    }

    // Cleanup
}

} // extern "C"
