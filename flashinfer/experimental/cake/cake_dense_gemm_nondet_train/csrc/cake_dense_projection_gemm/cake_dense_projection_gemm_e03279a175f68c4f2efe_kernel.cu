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

__global__ __launch_bounds__(192, LAUNCH_MIN_BLOCKS) __cluster_dims__(2,1,1) void
kernel_cake_dense_projection_gemm_e03279a175f68c4f2efe(const __grid_constant__ CUtensorMap A, const __grid_constant__ CUtensorMap B, float* __restrict__ out, int M, int N, int K, int num_l, int ldo, int out_l)
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
    #define mma_done_addr (mbar_base + 48)
    #define acc_full_addr (mbar_base + 96)
    #define tmem_free_addr (mbar_base + 104)
    #define red_bar_addr (mbar_base + 112)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    const unsigned int clusters_x = gridDim.x / 2;
    const unsigned int cluster_id = ((blockIdx.z * gridDim.y + blockIdx.y) * clusters_x) + blockIdx.x / 2;
    const unsigned int num_clusters = clusters_x * gridDim.y * gridDim.z;

    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));

    // Prologue
    __nv_bfloat16* smem_a = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int smem_a_addr = smem + 1024;
    __nv_bfloat16* smem_b = reinterpret_cast<__nv_bfloat16*>(smem_raw + 17408);
    const int smem_b_addr = smem + 17408;
    float* red_buf = reinterpret_cast<float*>(smem_raw + 197632);
    const int red_buf_addr = smem + 197632;

    // Kernel setup ops
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");

    // Mbarrier init (15 barriers at smem_raw[0..120))

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
            // mma_done: 6 barriers, init_count=1
            mbarrier_init(smem + 48, 1);
            mbarrier_init(smem + 56, 1);
            mbarrier_init(smem + 64, 1);
            mbarrier_init(smem + 72, 1);
            mbarrier_init(smem + 80, 1);
            mbarrier_init(smem + 88, 1);
            // acc_full: 1 barriers, init_count=1
            mbarrier_init(smem + 96, 1);
            // tmem_free: 1 barriers, init_count=4
            mbarrier_init(smem + 104, 4);
            // red_bar: 1 barriers, init_count=1
            mbarrier_init(smem + 112, 1);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncthreads();
    asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");

    // TMEM alloc (128 columns, 128 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 120);
    if (warp == 1) {
        int _tmem_hold = smem + 120;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(128) : "memory");
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
            int n_tiles = (N + 128 - 1) / 128;
            int k_iters = (K + 64 - 1) / 64;
            int tile = (int)blockIdx.x / 2;
            int tiles_l = m_tiles * n_tiles;
            int l_idx = tile / tiles_l;
            int in_l = tile - l_idx * tiles_l;
            int n_tile = in_l / m_tiles;
            int m_tile = in_l - n_tile * m_tiles;
            int split = (int)cta_rank;
            int k_begin = split * k_iters / 2;
            int k_end = (split + 1) * k_iters / 2;
            int a_row = m_tile * 128;
            int b_col = n_tile * 128;
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
                        :: "r"(smem_a_addr + load_stage * 32768), "l"((&A)), "r"(a_row), "r"(k0), "r"(l_idx),
                           "r"(tma_full_addr + (load_stage) * 8), "l"(0x12F0000000000000ULL) : "memory");
                    asm volatile(
                        "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                        " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                        :: "r"(smem_a_addr + load_stage * 32768 + 8192), "l"((&A)), "r"(a_row + 64), "r"(k0), "r"(l_idx),
                           "r"(tma_full_addr + (load_stage) * 8), "l"(0x12F0000000000000ULL) : "memory");
                    tma_3d_gmem2smem(smem_b_addr + load_stage * 32768, (&B), b_col, k0, l_idx, tma_full_addr + (load_stage) * 8);
                    tma_3d_gmem2smem(smem_b_addr + load_stage * 32768 + 8192, (&B), b_col + 64, k0, l_idx, tma_full_addr + (load_stage) * 8);
                    mbarrier_arrive_expect_tx(tma_full_addr + (load_stage) * 8, 32768);
                    load_stage += 1;
                    if (load_stage == 6) { load_stage = 0; _phase_mma_done ^= 1; }
                }
                #pragma unroll
                for (int _drain = 0; _drain < 6; _drain++) {
                    mbarrier_wait(mma_done_addr + (load_stage) * 8, _phase_mma_done);
                    load_stage += 1;
                    if (load_stage == 6) { load_stage = 0; _phase_mma_done ^= 1; }
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
            int k_begin_1 = split_1 * k_iters_m / 2;
            int k_end_1 = (split_1 + 1) * k_iters_m / 2;
            unsigned int _phase_tma_full = 0;
            #pragma unroll 1
            for (int k_block_1 = k_begin_1; k_block_1 < k_end_1; k_block_1++) {
                mbarrier_wait(tma_full_addr + (mma_stage) * 8, _phase_tma_full);
                asm volatile("tcgen05.fence::after_thread_sync;");
                int init_flag = ((k_block_1 == k_begin_1) ? 1 : 0);
                int a_lo = ((((smem_a_addr) >> 4) & 0x3FFF) | 0x2000000) + (mma_stage) * 2048;
                int b_lo = ((((smem_b_addr) >> 4) & 0x3FFF) | 0x2000000) + (mma_stage) * 2048;
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
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 136414352;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(a_lo), "r"(b_lo), "r"(tmem_accum), "r"(((init_flag) ? 0 : 1)), "r"(_elected_0) : "memory");
                tcgen05_commit_leader(mma_done_addr + (mma_stage) * 8, _elected_0);
                mma_stage += 1;
                if (mma_stage == 6) { mma_stage = 0; _phase_tma_full ^= 1; }
            }
            elect_commit(acc_full_addr);
            mbarrier_wait(tmem_free_addr, 0);
            asm volatile("tcgen05.fence::after_thread_sync;");
            asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(128));
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
            int n_tiles_e = (N + 128 - 1) / 128;
            int tile_1 = (int)blockIdx.x / 2;
            int tiles_l_1 = m_tiles_e * n_tiles_e;
            int l_idx_1 = tile_1 / tiles_l_1;
            int in_l_1 = tile_1 - l_idx_1 * tiles_l_1;
            int n_tile_1 = in_l_1 / m_tiles_e;
            int m_tile_1 = in_l_1 - n_tile_1 * m_tiles_e;
            int row = m_tile_1 * 128 + tid_1;
            int col0 = n_tile_1 * 128;
            long long base_t = (long long)l_idx_1 * (long long)out_l + (long long)row;
            float vals[128];
            if (warp == 2) {
                if (elect_sync()) {
                    if (cta_rank == 0) {
                        mbarrier_arrive_expect_tx(red_bar_addr, 32768);
                    }
                    if (cta_rank == 1) {
                        mbarrier_arrive_expect_tx(red_bar_addr, 32768);
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
            asm volatile(
                "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                : "=f"(vals[32]), "=f"(vals[33]), "=f"(vals[34]), "=f"(vals[35]), "=f"(vals[36]), "=f"(vals[37]), "=f"(vals[38]), "=f"(vals[39]), "=f"(vals[40]), "=f"(vals[41]), "=f"(vals[42]), "=f"(vals[43]), "=f"(vals[44]), "=f"(vals[45]), "=f"(vals[46]), "=f"(vals[47]), "=f"(vals[48]), "=f"(vals[49]), "=f"(vals[50]), "=f"(vals[51]), "=f"(vals[52]), "=f"(vals[53]), "=f"(vals[54]), "=f"(vals[55]), "=f"(vals[56]), "=f"(vals[57]), "=f"(vals[58]), "=f"(vals[59]), "=f"(vals[60]), "=f"(vals[61]), "=f"(vals[62]), "=f"(vals[63])
                : "r"(lane_addr + 32));
            asm volatile("tcgen05.wait::ld.sync.aligned;");
            asm volatile(
                "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                : "=f"(vals[64]), "=f"(vals[65]), "=f"(vals[66]), "=f"(vals[67]), "=f"(vals[68]), "=f"(vals[69]), "=f"(vals[70]), "=f"(vals[71]), "=f"(vals[72]), "=f"(vals[73]), "=f"(vals[74]), "=f"(vals[75]), "=f"(vals[76]), "=f"(vals[77]), "=f"(vals[78]), "=f"(vals[79]), "=f"(vals[80]), "=f"(vals[81]), "=f"(vals[82]), "=f"(vals[83]), "=f"(vals[84]), "=f"(vals[85]), "=f"(vals[86]), "=f"(vals[87]), "=f"(vals[88]), "=f"(vals[89]), "=f"(vals[90]), "=f"(vals[91]), "=f"(vals[92]), "=f"(vals[93]), "=f"(vals[94]), "=f"(vals[95])
                : "r"(lane_addr + 64));
            asm volatile("tcgen05.wait::ld.sync.aligned;");
            asm volatile(
                "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                : "=f"(vals[96]), "=f"(vals[97]), "=f"(vals[98]), "=f"(vals[99]), "=f"(vals[100]), "=f"(vals[101]), "=f"(vals[102]), "=f"(vals[103]), "=f"(vals[104]), "=f"(vals[105]), "=f"(vals[106]), "=f"(vals[107]), "=f"(vals[108]), "=f"(vals[109]), "=f"(vals[110]), "=f"(vals[111]), "=f"(vals[112]), "=f"(vals[113]), "=f"(vals[114]), "=f"(vals[115]), "=f"(vals[116]), "=f"(vals[117]), "=f"(vals[118]), "=f"(vals[119]), "=f"(vals[120]), "=f"(vals[121]), "=f"(vals[122]), "=f"(vals[123]), "=f"(vals[124]), "=f"(vals[125]), "=f"(vals[126]), "=f"(vals[127])
                : "r"(lane_addr + 96));
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
                    : "=r"(dst) : "r"(red_buf_addr + (unsigned int)((slot * 16 * 32 * 4 + tid_1) * 16)), "r"(0));
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
                uint32_t _mbar_peer_2;
                asm(
                    "mapa.shared::cluster.u32 %0, %1, %2;"
                    : "=r"(_mbar_peer_2) : "r"(red_bar_addr), "r"(0));
                asm volatile(
                    "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                    :: "r"(dst + 4096), "r"(__float_as_uint(vals[8])), "r"(__float_as_uint(vals[9])), "r"(__float_as_uint(vals[10])), "r"(__float_as_uint(vals[11])), "r"(_mbar_peer_2) : "memory");
                uint32_t _mbar_peer_3;
                asm(
                    "mapa.shared::cluster.u32 %0, %1, %2;"
                    : "=r"(_mbar_peer_3) : "r"(red_bar_addr), "r"(0));
                asm volatile(
                    "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                    :: "r"(dst + 6144), "r"(__float_as_uint(vals[12])), "r"(__float_as_uint(vals[13])), "r"(__float_as_uint(vals[14])), "r"(__float_as_uint(vals[15])), "r"(_mbar_peer_3) : "memory");
                uint32_t _mbar_peer_4;
                asm(
                    "mapa.shared::cluster.u32 %0, %1, %2;"
                    : "=r"(_mbar_peer_4) : "r"(red_bar_addr), "r"(0));
                asm volatile(
                    "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                    :: "r"(dst + 8192), "r"(__float_as_uint(vals[16])), "r"(__float_as_uint(vals[17])), "r"(__float_as_uint(vals[18])), "r"(__float_as_uint(vals[19])), "r"(_mbar_peer_4) : "memory");
                uint32_t _mbar_peer_5;
                asm(
                    "mapa.shared::cluster.u32 %0, %1, %2;"
                    : "=r"(_mbar_peer_5) : "r"(red_bar_addr), "r"(0));
                asm volatile(
                    "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                    :: "r"(dst + 10240), "r"(__float_as_uint(vals[20])), "r"(__float_as_uint(vals[21])), "r"(__float_as_uint(vals[22])), "r"(__float_as_uint(vals[23])), "r"(_mbar_peer_5) : "memory");
                uint32_t _mbar_peer_6;
                asm(
                    "mapa.shared::cluster.u32 %0, %1, %2;"
                    : "=r"(_mbar_peer_6) : "r"(red_bar_addr), "r"(0));
                asm volatile(
                    "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                    :: "r"(dst + 12288), "r"(__float_as_uint(vals[24])), "r"(__float_as_uint(vals[25])), "r"(__float_as_uint(vals[26])), "r"(__float_as_uint(vals[27])), "r"(_mbar_peer_6) : "memory");
                uint32_t _mbar_peer_7;
                asm(
                    "mapa.shared::cluster.u32 %0, %1, %2;"
                    : "=r"(_mbar_peer_7) : "r"(red_bar_addr), "r"(0));
                asm volatile(
                    "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                    :: "r"(dst + 14336), "r"(__float_as_uint(vals[28])), "r"(__float_as_uint(vals[29])), "r"(__float_as_uint(vals[30])), "r"(__float_as_uint(vals[31])), "r"(_mbar_peer_7) : "memory");
                uint32_t _mbar_peer_8;
                asm(
                    "mapa.shared::cluster.u32 %0, %1, %2;"
                    : "=r"(_mbar_peer_8) : "r"(red_bar_addr), "r"(0));
                asm volatile(
                    "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                    :: "r"(dst + 16384), "r"(__float_as_uint(vals[32])), "r"(__float_as_uint(vals[33])), "r"(__float_as_uint(vals[34])), "r"(__float_as_uint(vals[35])), "r"(_mbar_peer_8) : "memory");
                uint32_t _mbar_peer_9;
                asm(
                    "mapa.shared::cluster.u32 %0, %1, %2;"
                    : "=r"(_mbar_peer_9) : "r"(red_bar_addr), "r"(0));
                asm volatile(
                    "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                    :: "r"(dst + 18432), "r"(__float_as_uint(vals[36])), "r"(__float_as_uint(vals[37])), "r"(__float_as_uint(vals[38])), "r"(__float_as_uint(vals[39])), "r"(_mbar_peer_9) : "memory");
                uint32_t _mbar_peer_10;
                asm(
                    "mapa.shared::cluster.u32 %0, %1, %2;"
                    : "=r"(_mbar_peer_10) : "r"(red_bar_addr), "r"(0));
                asm volatile(
                    "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                    :: "r"(dst + 20480), "r"(__float_as_uint(vals[40])), "r"(__float_as_uint(vals[41])), "r"(__float_as_uint(vals[42])), "r"(__float_as_uint(vals[43])), "r"(_mbar_peer_10) : "memory");
                uint32_t _mbar_peer_11;
                asm(
                    "mapa.shared::cluster.u32 %0, %1, %2;"
                    : "=r"(_mbar_peer_11) : "r"(red_bar_addr), "r"(0));
                asm volatile(
                    "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                    :: "r"(dst + 22528), "r"(__float_as_uint(vals[44])), "r"(__float_as_uint(vals[45])), "r"(__float_as_uint(vals[46])), "r"(__float_as_uint(vals[47])), "r"(_mbar_peer_11) : "memory");
                uint32_t _mbar_peer_12;
                asm(
                    "mapa.shared::cluster.u32 %0, %1, %2;"
                    : "=r"(_mbar_peer_12) : "r"(red_bar_addr), "r"(0));
                asm volatile(
                    "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                    :: "r"(dst + 24576), "r"(__float_as_uint(vals[48])), "r"(__float_as_uint(vals[49])), "r"(__float_as_uint(vals[50])), "r"(__float_as_uint(vals[51])), "r"(_mbar_peer_12) : "memory");
                uint32_t _mbar_peer_13;
                asm(
                    "mapa.shared::cluster.u32 %0, %1, %2;"
                    : "=r"(_mbar_peer_13) : "r"(red_bar_addr), "r"(0));
                asm volatile(
                    "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                    :: "r"(dst + 26624), "r"(__float_as_uint(vals[52])), "r"(__float_as_uint(vals[53])), "r"(__float_as_uint(vals[54])), "r"(__float_as_uint(vals[55])), "r"(_mbar_peer_13) : "memory");
                uint32_t _mbar_peer_14;
                asm(
                    "mapa.shared::cluster.u32 %0, %1, %2;"
                    : "=r"(_mbar_peer_14) : "r"(red_bar_addr), "r"(0));
                asm volatile(
                    "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                    :: "r"(dst + 28672), "r"(__float_as_uint(vals[56])), "r"(__float_as_uint(vals[57])), "r"(__float_as_uint(vals[58])), "r"(__float_as_uint(vals[59])), "r"(_mbar_peer_14) : "memory");
                uint32_t _mbar_peer_15;
                asm(
                    "mapa.shared::cluster.u32 %0, %1, %2;"
                    : "=r"(_mbar_peer_15) : "r"(red_bar_addr), "r"(0));
                asm volatile(
                    "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                    :: "r"(dst + 30720), "r"(__float_as_uint(vals[60])), "r"(__float_as_uint(vals[61])), "r"(__float_as_uint(vals[62])), "r"(__float_as_uint(vals[63])), "r"(_mbar_peer_15) : "memory");
            }
            if (cta_rank != 1) {
                int slot_1 = (int)cta_rank - (((int)cta_rank > 1) ? 1 : 0);
                uint32_t dst_1;
                asm volatile(
                    "mapa.shared::cluster.u32 %0, %1, %2;"
                    : "=r"(dst_1) : "r"(red_buf_addr + (unsigned int)((slot_1 * 16 * 32 * 4 + tid_1) * 16)), "r"(1));
                uint32_t _mbar_peer_16;
                asm(
                    "mapa.shared::cluster.u32 %0, %1, %2;"
                    : "=r"(_mbar_peer_16) : "r"(red_bar_addr), "r"(1));
                asm volatile(
                    "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                    :: "r"(dst_1), "r"(__float_as_uint(vals[64])), "r"(__float_as_uint(vals[65])), "r"(__float_as_uint(vals[66])), "r"(__float_as_uint(vals[67])), "r"(_mbar_peer_16) : "memory");
                uint32_t _mbar_peer_17;
                asm(
                    "mapa.shared::cluster.u32 %0, %1, %2;"
                    : "=r"(_mbar_peer_17) : "r"(red_bar_addr), "r"(1));
                asm volatile(
                    "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                    :: "r"(dst_1 + 2048), "r"(__float_as_uint(vals[68])), "r"(__float_as_uint(vals[69])), "r"(__float_as_uint(vals[70])), "r"(__float_as_uint(vals[71])), "r"(_mbar_peer_17) : "memory");
                uint32_t _mbar_peer_18;
                asm(
                    "mapa.shared::cluster.u32 %0, %1, %2;"
                    : "=r"(_mbar_peer_18) : "r"(red_bar_addr), "r"(1));
                asm volatile(
                    "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                    :: "r"(dst_1 + 4096), "r"(__float_as_uint(vals[72])), "r"(__float_as_uint(vals[73])), "r"(__float_as_uint(vals[74])), "r"(__float_as_uint(vals[75])), "r"(_mbar_peer_18) : "memory");
                uint32_t _mbar_peer_19;
                asm(
                    "mapa.shared::cluster.u32 %0, %1, %2;"
                    : "=r"(_mbar_peer_19) : "r"(red_bar_addr), "r"(1));
                asm volatile(
                    "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                    :: "r"(dst_1 + 6144), "r"(__float_as_uint(vals[76])), "r"(__float_as_uint(vals[77])), "r"(__float_as_uint(vals[78])), "r"(__float_as_uint(vals[79])), "r"(_mbar_peer_19) : "memory");
                uint32_t _mbar_peer_20;
                asm(
                    "mapa.shared::cluster.u32 %0, %1, %2;"
                    : "=r"(_mbar_peer_20) : "r"(red_bar_addr), "r"(1));
                asm volatile(
                    "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                    :: "r"(dst_1 + 8192), "r"(__float_as_uint(vals[80])), "r"(__float_as_uint(vals[81])), "r"(__float_as_uint(vals[82])), "r"(__float_as_uint(vals[83])), "r"(_mbar_peer_20) : "memory");
                uint32_t _mbar_peer_21;
                asm(
                    "mapa.shared::cluster.u32 %0, %1, %2;"
                    : "=r"(_mbar_peer_21) : "r"(red_bar_addr), "r"(1));
                asm volatile(
                    "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                    :: "r"(dst_1 + 10240), "r"(__float_as_uint(vals[84])), "r"(__float_as_uint(vals[85])), "r"(__float_as_uint(vals[86])), "r"(__float_as_uint(vals[87])), "r"(_mbar_peer_21) : "memory");
                uint32_t _mbar_peer_22;
                asm(
                    "mapa.shared::cluster.u32 %0, %1, %2;"
                    : "=r"(_mbar_peer_22) : "r"(red_bar_addr), "r"(1));
                asm volatile(
                    "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                    :: "r"(dst_1 + 12288), "r"(__float_as_uint(vals[88])), "r"(__float_as_uint(vals[89])), "r"(__float_as_uint(vals[90])), "r"(__float_as_uint(vals[91])), "r"(_mbar_peer_22) : "memory");
                uint32_t _mbar_peer_23;
                asm(
                    "mapa.shared::cluster.u32 %0, %1, %2;"
                    : "=r"(_mbar_peer_23) : "r"(red_bar_addr), "r"(1));
                asm volatile(
                    "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                    :: "r"(dst_1 + 14336), "r"(__float_as_uint(vals[92])), "r"(__float_as_uint(vals[93])), "r"(__float_as_uint(vals[94])), "r"(__float_as_uint(vals[95])), "r"(_mbar_peer_23) : "memory");
                uint32_t _mbar_peer_24;
                asm(
                    "mapa.shared::cluster.u32 %0, %1, %2;"
                    : "=r"(_mbar_peer_24) : "r"(red_bar_addr), "r"(1));
                asm volatile(
                    "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                    :: "r"(dst_1 + 16384), "r"(__float_as_uint(vals[96])), "r"(__float_as_uint(vals[97])), "r"(__float_as_uint(vals[98])), "r"(__float_as_uint(vals[99])), "r"(_mbar_peer_24) : "memory");
                uint32_t _mbar_peer_25;
                asm(
                    "mapa.shared::cluster.u32 %0, %1, %2;"
                    : "=r"(_mbar_peer_25) : "r"(red_bar_addr), "r"(1));
                asm volatile(
                    "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                    :: "r"(dst_1 + 18432), "r"(__float_as_uint(vals[100])), "r"(__float_as_uint(vals[101])), "r"(__float_as_uint(vals[102])), "r"(__float_as_uint(vals[103])), "r"(_mbar_peer_25) : "memory");
                uint32_t _mbar_peer_26;
                asm(
                    "mapa.shared::cluster.u32 %0, %1, %2;"
                    : "=r"(_mbar_peer_26) : "r"(red_bar_addr), "r"(1));
                asm volatile(
                    "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                    :: "r"(dst_1 + 20480), "r"(__float_as_uint(vals[104])), "r"(__float_as_uint(vals[105])), "r"(__float_as_uint(vals[106])), "r"(__float_as_uint(vals[107])), "r"(_mbar_peer_26) : "memory");
                uint32_t _mbar_peer_27;
                asm(
                    "mapa.shared::cluster.u32 %0, %1, %2;"
                    : "=r"(_mbar_peer_27) : "r"(red_bar_addr), "r"(1));
                asm volatile(
                    "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                    :: "r"(dst_1 + 22528), "r"(__float_as_uint(vals[108])), "r"(__float_as_uint(vals[109])), "r"(__float_as_uint(vals[110])), "r"(__float_as_uint(vals[111])), "r"(_mbar_peer_27) : "memory");
                uint32_t _mbar_peer_28;
                asm(
                    "mapa.shared::cluster.u32 %0, %1, %2;"
                    : "=r"(_mbar_peer_28) : "r"(red_bar_addr), "r"(1));
                asm volatile(
                    "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                    :: "r"(dst_1 + 24576), "r"(__float_as_uint(vals[112])), "r"(__float_as_uint(vals[113])), "r"(__float_as_uint(vals[114])), "r"(__float_as_uint(vals[115])), "r"(_mbar_peer_28) : "memory");
                uint32_t _mbar_peer_29;
                asm(
                    "mapa.shared::cluster.u32 %0, %1, %2;"
                    : "=r"(_mbar_peer_29) : "r"(red_bar_addr), "r"(1));
                asm volatile(
                    "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                    :: "r"(dst_1 + 26624), "r"(__float_as_uint(vals[116])), "r"(__float_as_uint(vals[117])), "r"(__float_as_uint(vals[118])), "r"(__float_as_uint(vals[119])), "r"(_mbar_peer_29) : "memory");
                uint32_t _mbar_peer_30;
                asm(
                    "mapa.shared::cluster.u32 %0, %1, %2;"
                    : "=r"(_mbar_peer_30) : "r"(red_bar_addr), "r"(1));
                asm volatile(
                    "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                    :: "r"(dst_1 + 28672), "r"(__float_as_uint(vals[120])), "r"(__float_as_uint(vals[121])), "r"(__float_as_uint(vals[122])), "r"(__float_as_uint(vals[123])), "r"(_mbar_peer_30) : "memory");
                uint32_t _mbar_peer_31;
                asm(
                    "mapa.shared::cluster.u32 %0, %1, %2;"
                    : "=r"(_mbar_peer_31) : "r"(red_bar_addr), "r"(1));
                asm volatile(
                    "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                    :: "r"(dst_1 + 30720), "r"(__float_as_uint(vals[124])), "r"(__float_as_uint(vals[125])), "r"(__float_as_uint(vals[126])), "r"(__float_as_uint(vals[127])), "r"(_mbar_peer_31) : "memory");
            }
            mbarrier_wait_cluster_hint(red_bar_addr, 0, 10000000);
            asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
            if (cta_rank == 0) {
                float total[64];
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
                total[4] = vals[4];
                total[5] = vals[5];
                total[6] = vals[6];
                total[7] = vals[7];
                unsigned int addr_1 = red_buf_addr + (unsigned int)((128 + tid_1) * 16);
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&part[0])), "=r"(*reinterpret_cast<uint32_t*>(&part[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&part[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&part[(0) + 3]))
                    : "r"(addr_1));
                total[4] = total[4] + part[0];
                total[5] = total[5] + part[1];
                total[6] = total[6] + part[2];
                total[7] = total[7] + part[3];
                total[8] = vals[8];
                total[9] = vals[9];
                total[10] = vals[10];
                total[11] = vals[11];
                unsigned int addr_2 = red_buf_addr + (unsigned int)((256 + tid_1) * 16);
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&part[0])), "=r"(*reinterpret_cast<uint32_t*>(&part[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&part[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&part[(0) + 3]))
                    : "r"(addr_2));
                total[8] = total[8] + part[0];
                total[9] = total[9] + part[1];
                total[10] = total[10] + part[2];
                total[11] = total[11] + part[3];
                total[12] = vals[12];
                total[13] = vals[13];
                total[14] = vals[14];
                total[15] = vals[15];
                unsigned int addr_3 = red_buf_addr + (unsigned int)((384 + tid_1) * 16);
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&part[0])), "=r"(*reinterpret_cast<uint32_t*>(&part[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&part[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&part[(0) + 3]))
                    : "r"(addr_3));
                total[12] = total[12] + part[0];
                total[13] = total[13] + part[1];
                total[14] = total[14] + part[2];
                total[15] = total[15] + part[3];
                total[16] = vals[16];
                total[17] = vals[17];
                total[18] = vals[18];
                total[19] = vals[19];
                unsigned int addr_4 = red_buf_addr + (unsigned int)((512 + tid_1) * 16);
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&part[0])), "=r"(*reinterpret_cast<uint32_t*>(&part[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&part[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&part[(0) + 3]))
                    : "r"(addr_4));
                total[16] = total[16] + part[0];
                total[17] = total[17] + part[1];
                total[18] = total[18] + part[2];
                total[19] = total[19] + part[3];
                total[20] = vals[20];
                total[21] = vals[21];
                total[22] = vals[22];
                total[23] = vals[23];
                unsigned int addr_5 = red_buf_addr + (unsigned int)((640 + tid_1) * 16);
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&part[0])), "=r"(*reinterpret_cast<uint32_t*>(&part[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&part[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&part[(0) + 3]))
                    : "r"(addr_5));
                total[20] = total[20] + part[0];
                total[21] = total[21] + part[1];
                total[22] = total[22] + part[2];
                total[23] = total[23] + part[3];
                total[24] = vals[24];
                total[25] = vals[25];
                total[26] = vals[26];
                total[27] = vals[27];
                unsigned int addr_6 = red_buf_addr + (unsigned int)((768 + tid_1) * 16);
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&part[0])), "=r"(*reinterpret_cast<uint32_t*>(&part[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&part[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&part[(0) + 3]))
                    : "r"(addr_6));
                total[24] = total[24] + part[0];
                total[25] = total[25] + part[1];
                total[26] = total[26] + part[2];
                total[27] = total[27] + part[3];
                total[28] = vals[28];
                total[29] = vals[29];
                total[30] = vals[30];
                total[31] = vals[31];
                unsigned int addr_7 = red_buf_addr + (unsigned int)((896 + tid_1) * 16);
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&part[0])), "=r"(*reinterpret_cast<uint32_t*>(&part[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&part[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&part[(0) + 3]))
                    : "r"(addr_7));
                total[28] = total[28] + part[0];
                total[29] = total[29] + part[1];
                total[30] = total[30] + part[2];
                total[31] = total[31] + part[3];
                total[32] = vals[32];
                total[33] = vals[33];
                total[34] = vals[34];
                total[35] = vals[35];
                unsigned int addr_8 = red_buf_addr + (unsigned int)((1024 + tid_1) * 16);
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&part[0])), "=r"(*reinterpret_cast<uint32_t*>(&part[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&part[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&part[(0) + 3]))
                    : "r"(addr_8));
                total[32] = total[32] + part[0];
                total[33] = total[33] + part[1];
                total[34] = total[34] + part[2];
                total[35] = total[35] + part[3];
                total[36] = vals[36];
                total[37] = vals[37];
                total[38] = vals[38];
                total[39] = vals[39];
                unsigned int addr_9 = red_buf_addr + (unsigned int)((1152 + tid_1) * 16);
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&part[0])), "=r"(*reinterpret_cast<uint32_t*>(&part[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&part[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&part[(0) + 3]))
                    : "r"(addr_9));
                total[36] = total[36] + part[0];
                total[37] = total[37] + part[1];
                total[38] = total[38] + part[2];
                total[39] = total[39] + part[3];
                total[40] = vals[40];
                total[41] = vals[41];
                total[42] = vals[42];
                total[43] = vals[43];
                unsigned int addr_10 = red_buf_addr + (unsigned int)((1280 + tid_1) * 16);
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&part[0])), "=r"(*reinterpret_cast<uint32_t*>(&part[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&part[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&part[(0) + 3]))
                    : "r"(addr_10));
                total[40] = total[40] + part[0];
                total[41] = total[41] + part[1];
                total[42] = total[42] + part[2];
                total[43] = total[43] + part[3];
                total[44] = vals[44];
                total[45] = vals[45];
                total[46] = vals[46];
                total[47] = vals[47];
                unsigned int addr_11 = red_buf_addr + (unsigned int)((1408 + tid_1) * 16);
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&part[0])), "=r"(*reinterpret_cast<uint32_t*>(&part[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&part[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&part[(0) + 3]))
                    : "r"(addr_11));
                total[44] = total[44] + part[0];
                total[45] = total[45] + part[1];
                total[46] = total[46] + part[2];
                total[47] = total[47] + part[3];
                total[48] = vals[48];
                total[49] = vals[49];
                total[50] = vals[50];
                total[51] = vals[51];
                unsigned int addr_12 = red_buf_addr + (unsigned int)((1536 + tid_1) * 16);
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&part[0])), "=r"(*reinterpret_cast<uint32_t*>(&part[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&part[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&part[(0) + 3]))
                    : "r"(addr_12));
                total[48] = total[48] + part[0];
                total[49] = total[49] + part[1];
                total[50] = total[50] + part[2];
                total[51] = total[51] + part[3];
                total[52] = vals[52];
                total[53] = vals[53];
                total[54] = vals[54];
                total[55] = vals[55];
                unsigned int addr_13 = red_buf_addr + (unsigned int)((1664 + tid_1) * 16);
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&part[0])), "=r"(*reinterpret_cast<uint32_t*>(&part[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&part[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&part[(0) + 3]))
                    : "r"(addr_13));
                total[52] = total[52] + part[0];
                total[53] = total[53] + part[1];
                total[54] = total[54] + part[2];
                total[55] = total[55] + part[3];
                total[56] = vals[56];
                total[57] = vals[57];
                total[58] = vals[58];
                total[59] = vals[59];
                unsigned int addr_14 = red_buf_addr + (unsigned int)((1792 + tid_1) * 16);
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&part[0])), "=r"(*reinterpret_cast<uint32_t*>(&part[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&part[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&part[(0) + 3]))
                    : "r"(addr_14));
                total[56] = total[56] + part[0];
                total[57] = total[57] + part[1];
                total[58] = total[58] + part[2];
                total[59] = total[59] + part[3];
                total[60] = vals[60];
                total[61] = vals[61];
                total[62] = vals[62];
                total[63] = vals[63];
                unsigned int addr_15 = red_buf_addr + (unsigned int)((1920 + tid_1) * 16);
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&part[0])), "=r"(*reinterpret_cast<uint32_t*>(&part[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&part[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&part[(0) + 3]))
                    : "r"(addr_15));
                total[60] = total[60] + part[0];
                total[61] = total[61] + part[1];
                total[62] = total[62] + part[2];
                total[63] = total[63] + part[3];
                if (row < M) {
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t + (long long)col0 * (long long)ldo))), "f"(total[(0) + 0]), "r"((unsigned int)(col0 < N)) : "memory");
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t + (long long)(col0 + 1) * (long long)ldo))), "f"(total[(1) + 0]), "r"((unsigned int)(col0 + 1 < N)) : "memory");
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t + (long long)(col0 + 2) * (long long)ldo))), "f"(total[(2) + 0]), "r"((unsigned int)(col0 + 2 < N)) : "memory");
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t + (long long)(col0 + 3) * (long long)ldo))), "f"(total[(3) + 0]), "r"((unsigned int)(col0 + 3 < N)) : "memory");
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t + (long long)(col0 + 4) * (long long)ldo))), "f"(total[(4) + 0]), "r"((unsigned int)(col0 + 4 < N)) : "memory");
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t + (long long)(col0 + 4 + 1) * (long long)ldo))), "f"(total[(5) + 0]), "r"((unsigned int)(col0 + 4 + 1 < N)) : "memory");
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t + (long long)(col0 + 4 + 2) * (long long)ldo))), "f"(total[(6) + 0]), "r"((unsigned int)(col0 + 4 + 2 < N)) : "memory");
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t + (long long)(col0 + 4 + 3) * (long long)ldo))), "f"(total[(7) + 0]), "r"((unsigned int)(col0 + 4 + 3 < N)) : "memory");
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t + (long long)(col0 + 8) * (long long)ldo))), "f"(total[(8) + 0]), "r"((unsigned int)(col0 + 8 < N)) : "memory");
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t + (long long)(col0 + 8 + 1) * (long long)ldo))), "f"(total[(9) + 0]), "r"((unsigned int)(col0 + 8 + 1 < N)) : "memory");
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t + (long long)(col0 + 8 + 2) * (long long)ldo))), "f"(total[(10) + 0]), "r"((unsigned int)(col0 + 8 + 2 < N)) : "memory");
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t + (long long)(col0 + 8 + 3) * (long long)ldo))), "f"(total[(11) + 0]), "r"((unsigned int)(col0 + 8 + 3 < N)) : "memory");
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t + (long long)(col0 + 12) * (long long)ldo))), "f"(total[(12) + 0]), "r"((unsigned int)(col0 + 12 < N)) : "memory");
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t + (long long)(col0 + 12 + 1) * (long long)ldo))), "f"(total[(13) + 0]), "r"((unsigned int)(col0 + 12 + 1 < N)) : "memory");
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t + (long long)(col0 + 12 + 2) * (long long)ldo))), "f"(total[(14) + 0]), "r"((unsigned int)(col0 + 12 + 2 < N)) : "memory");
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t + (long long)(col0 + 12 + 3) * (long long)ldo))), "f"(total[(15) + 0]), "r"((unsigned int)(col0 + 12 + 3 < N)) : "memory");
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t + (long long)(col0 + 16) * (long long)ldo))), "f"(total[(16) + 0]), "r"((unsigned int)(col0 + 16 < N)) : "memory");
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t + (long long)(col0 + 16 + 1) * (long long)ldo))), "f"(total[(17) + 0]), "r"((unsigned int)(col0 + 16 + 1 < N)) : "memory");
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t + (long long)(col0 + 16 + 2) * (long long)ldo))), "f"(total[(18) + 0]), "r"((unsigned int)(col0 + 16 + 2 < N)) : "memory");
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t + (long long)(col0 + 16 + 3) * (long long)ldo))), "f"(total[(19) + 0]), "r"((unsigned int)(col0 + 16 + 3 < N)) : "memory");
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t + (long long)(col0 + 20) * (long long)ldo))), "f"(total[(20) + 0]), "r"((unsigned int)(col0 + 20 < N)) : "memory");
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t + (long long)(col0 + 20 + 1) * (long long)ldo))), "f"(total[(21) + 0]), "r"((unsigned int)(col0 + 20 + 1 < N)) : "memory");
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t + (long long)(col0 + 20 + 2) * (long long)ldo))), "f"(total[(22) + 0]), "r"((unsigned int)(col0 + 20 + 2 < N)) : "memory");
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t + (long long)(col0 + 20 + 3) * (long long)ldo))), "f"(total[(23) + 0]), "r"((unsigned int)(col0 + 20 + 3 < N)) : "memory");
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t + (long long)(col0 + 24) * (long long)ldo))), "f"(total[(24) + 0]), "r"((unsigned int)(col0 + 24 < N)) : "memory");
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t + (long long)(col0 + 24 + 1) * (long long)ldo))), "f"(total[(25) + 0]), "r"((unsigned int)(col0 + 24 + 1 < N)) : "memory");
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t + (long long)(col0 + 24 + 2) * (long long)ldo))), "f"(total[(26) + 0]), "r"((unsigned int)(col0 + 24 + 2 < N)) : "memory");
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t + (long long)(col0 + 24 + 3) * (long long)ldo))), "f"(total[(27) + 0]), "r"((unsigned int)(col0 + 24 + 3 < N)) : "memory");
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t + (long long)(col0 + 28) * (long long)ldo))), "f"(total[(28) + 0]), "r"((unsigned int)(col0 + 28 < N)) : "memory");
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t + (long long)(col0 + 28 + 1) * (long long)ldo))), "f"(total[(29) + 0]), "r"((unsigned int)(col0 + 28 + 1 < N)) : "memory");
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t + (long long)(col0 + 28 + 2) * (long long)ldo))), "f"(total[(30) + 0]), "r"((unsigned int)(col0 + 28 + 2 < N)) : "memory");
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t + (long long)(col0 + 28 + 3) * (long long)ldo))), "f"(total[(31) + 0]), "r"((unsigned int)(col0 + 28 + 3 < N)) : "memory");
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t + (long long)(col0 + 32) * (long long)ldo))), "f"(total[(32) + 0]), "r"((unsigned int)(col0 + 32 < N)) : "memory");
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t + (long long)(col0 + 32 + 1) * (long long)ldo))), "f"(total[(33) + 0]), "r"((unsigned int)(col0 + 32 + 1 < N)) : "memory");
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t + (long long)(col0 + 32 + 2) * (long long)ldo))), "f"(total[(34) + 0]), "r"((unsigned int)(col0 + 32 + 2 < N)) : "memory");
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t + (long long)(col0 + 32 + 3) * (long long)ldo))), "f"(total[(35) + 0]), "r"((unsigned int)(col0 + 32 + 3 < N)) : "memory");
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t + (long long)(col0 + 36) * (long long)ldo))), "f"(total[(36) + 0]), "r"((unsigned int)(col0 + 36 < N)) : "memory");
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t + (long long)(col0 + 36 + 1) * (long long)ldo))), "f"(total[(37) + 0]), "r"((unsigned int)(col0 + 36 + 1 < N)) : "memory");
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t + (long long)(col0 + 36 + 2) * (long long)ldo))), "f"(total[(38) + 0]), "r"((unsigned int)(col0 + 36 + 2 < N)) : "memory");
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t + (long long)(col0 + 36 + 3) * (long long)ldo))), "f"(total[(39) + 0]), "r"((unsigned int)(col0 + 36 + 3 < N)) : "memory");
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t + (long long)(col0 + 40) * (long long)ldo))), "f"(total[(40) + 0]), "r"((unsigned int)(col0 + 40 < N)) : "memory");
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t + (long long)(col0 + 40 + 1) * (long long)ldo))), "f"(total[(41) + 0]), "r"((unsigned int)(col0 + 40 + 1 < N)) : "memory");
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t + (long long)(col0 + 40 + 2) * (long long)ldo))), "f"(total[(42) + 0]), "r"((unsigned int)(col0 + 40 + 2 < N)) : "memory");
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t + (long long)(col0 + 40 + 3) * (long long)ldo))), "f"(total[(43) + 0]), "r"((unsigned int)(col0 + 40 + 3 < N)) : "memory");
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t + (long long)(col0 + 44) * (long long)ldo))), "f"(total[(44) + 0]), "r"((unsigned int)(col0 + 44 < N)) : "memory");
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t + (long long)(col0 + 44 + 1) * (long long)ldo))), "f"(total[(45) + 0]), "r"((unsigned int)(col0 + 44 + 1 < N)) : "memory");
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t + (long long)(col0 + 44 + 2) * (long long)ldo))), "f"(total[(46) + 0]), "r"((unsigned int)(col0 + 44 + 2 < N)) : "memory");
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t + (long long)(col0 + 44 + 3) * (long long)ldo))), "f"(total[(47) + 0]), "r"((unsigned int)(col0 + 44 + 3 < N)) : "memory");
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t + (long long)(col0 + 48) * (long long)ldo))), "f"(total[(48) + 0]), "r"((unsigned int)(col0 + 48 < N)) : "memory");
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t + (long long)(col0 + 48 + 1) * (long long)ldo))), "f"(total[(49) + 0]), "r"((unsigned int)(col0 + 48 + 1 < N)) : "memory");
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t + (long long)(col0 + 48 + 2) * (long long)ldo))), "f"(total[(50) + 0]), "r"((unsigned int)(col0 + 48 + 2 < N)) : "memory");
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t + (long long)(col0 + 48 + 3) * (long long)ldo))), "f"(total[(51) + 0]), "r"((unsigned int)(col0 + 48 + 3 < N)) : "memory");
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t + (long long)(col0 + 52) * (long long)ldo))), "f"(total[(52) + 0]), "r"((unsigned int)(col0 + 52 < N)) : "memory");
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t + (long long)(col0 + 52 + 1) * (long long)ldo))), "f"(total[(53) + 0]), "r"((unsigned int)(col0 + 52 + 1 < N)) : "memory");
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t + (long long)(col0 + 52 + 2) * (long long)ldo))), "f"(total[(54) + 0]), "r"((unsigned int)(col0 + 52 + 2 < N)) : "memory");
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t + (long long)(col0 + 52 + 3) * (long long)ldo))), "f"(total[(55) + 0]), "r"((unsigned int)(col0 + 52 + 3 < N)) : "memory");
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t + (long long)(col0 + 56) * (long long)ldo))), "f"(total[(56) + 0]), "r"((unsigned int)(col0 + 56 < N)) : "memory");
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t + (long long)(col0 + 56 + 1) * (long long)ldo))), "f"(total[(57) + 0]), "r"((unsigned int)(col0 + 56 + 1 < N)) : "memory");
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t + (long long)(col0 + 56 + 2) * (long long)ldo))), "f"(total[(58) + 0]), "r"((unsigned int)(col0 + 56 + 2 < N)) : "memory");
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t + (long long)(col0 + 56 + 3) * (long long)ldo))), "f"(total[(59) + 0]), "r"((unsigned int)(col0 + 56 + 3 < N)) : "memory");
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t + (long long)(col0 + 60) * (long long)ldo))), "f"(total[(60) + 0]), "r"((unsigned int)(col0 + 60 < N)) : "memory");
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t + (long long)(col0 + 60 + 1) * (long long)ldo))), "f"(total[(61) + 0]), "r"((unsigned int)(col0 + 60 + 1 < N)) : "memory");
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t + (long long)(col0 + 60 + 2) * (long long)ldo))), "f"(total[(62) + 0]), "r"((unsigned int)(col0 + 60 + 2 < N)) : "memory");
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t + (long long)(col0 + 60 + 3) * (long long)ldo))), "f"(total[(63) + 0]), "r"((unsigned int)(col0 + 60 + 3 < N)) : "memory");
                }
            }
            if (cta_rank == 1) {
                float total_1[64];
                float part_1[4];
                unsigned int addr_16 = red_buf_addr + (unsigned int)(tid_1 * 16);
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&total_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&total_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&total_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&total_1[(0) + 3]))
                    : "r"(addr_16));
                total_1[0] = total_1[0] + vals[64];
                total_1[1] = total_1[1] + vals[65];
                total_1[2] = total_1[2] + vals[66];
                total_1[3] = total_1[3] + vals[67];
                unsigned int addr_17 = red_buf_addr + (unsigned int)((128 + tid_1) * 16);
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&total_1[4])), "=r"(*reinterpret_cast<uint32_t*>(&total_1[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&total_1[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&total_1[(4) + 3]))
                    : "r"(addr_17));
                total_1[4] = total_1[4] + vals[68];
                total_1[5] = total_1[5] + vals[69];
                total_1[6] = total_1[6] + vals[70];
                total_1[7] = total_1[7] + vals[71];
                unsigned int addr_18 = red_buf_addr + (unsigned int)((256 + tid_1) * 16);
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&total_1[8])), "=r"(*reinterpret_cast<uint32_t*>(&total_1[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&total_1[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&total_1[(8) + 3]))
                    : "r"(addr_18));
                total_1[8] = total_1[8] + vals[72];
                total_1[9] = total_1[9] + vals[73];
                total_1[10] = total_1[10] + vals[74];
                total_1[11] = total_1[11] + vals[75];
                unsigned int addr_19 = red_buf_addr + (unsigned int)((384 + tid_1) * 16);
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&total_1[12])), "=r"(*reinterpret_cast<uint32_t*>(&total_1[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&total_1[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&total_1[(12) + 3]))
                    : "r"(addr_19));
                total_1[12] = total_1[12] + vals[76];
                total_1[13] = total_1[13] + vals[77];
                total_1[14] = total_1[14] + vals[78];
                total_1[15] = total_1[15] + vals[79];
                unsigned int addr_20 = red_buf_addr + (unsigned int)((512 + tid_1) * 16);
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&total_1[16])), "=r"(*reinterpret_cast<uint32_t*>(&total_1[(16) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&total_1[(16) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&total_1[(16) + 3]))
                    : "r"(addr_20));
                total_1[16] = total_1[16] + vals[80];
                total_1[17] = total_1[17] + vals[81];
                total_1[18] = total_1[18] + vals[82];
                total_1[19] = total_1[19] + vals[83];
                unsigned int addr_21 = red_buf_addr + (unsigned int)((640 + tid_1) * 16);
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&total_1[20])), "=r"(*reinterpret_cast<uint32_t*>(&total_1[(20) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&total_1[(20) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&total_1[(20) + 3]))
                    : "r"(addr_21));
                total_1[20] = total_1[20] + vals[84];
                total_1[21] = total_1[21] + vals[85];
                total_1[22] = total_1[22] + vals[86];
                total_1[23] = total_1[23] + vals[87];
                unsigned int addr_22 = red_buf_addr + (unsigned int)((768 + tid_1) * 16);
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&total_1[24])), "=r"(*reinterpret_cast<uint32_t*>(&total_1[(24) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&total_1[(24) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&total_1[(24) + 3]))
                    : "r"(addr_22));
                total_1[24] = total_1[24] + vals[88];
                total_1[25] = total_1[25] + vals[89];
                total_1[26] = total_1[26] + vals[90];
                total_1[27] = total_1[27] + vals[91];
                unsigned int addr_23 = red_buf_addr + (unsigned int)((896 + tid_1) * 16);
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&total_1[28])), "=r"(*reinterpret_cast<uint32_t*>(&total_1[(28) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&total_1[(28) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&total_1[(28) + 3]))
                    : "r"(addr_23));
                total_1[28] = total_1[28] + vals[92];
                total_1[29] = total_1[29] + vals[93];
                total_1[30] = total_1[30] + vals[94];
                total_1[31] = total_1[31] + vals[95];
                unsigned int addr_24 = red_buf_addr + (unsigned int)((1024 + tid_1) * 16);
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&total_1[32])), "=r"(*reinterpret_cast<uint32_t*>(&total_1[(32) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&total_1[(32) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&total_1[(32) + 3]))
                    : "r"(addr_24));
                total_1[32] = total_1[32] + vals[96];
                total_1[33] = total_1[33] + vals[97];
                total_1[34] = total_1[34] + vals[98];
                total_1[35] = total_1[35] + vals[99];
                unsigned int addr_25 = red_buf_addr + (unsigned int)((1152 + tid_1) * 16);
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&total_1[36])), "=r"(*reinterpret_cast<uint32_t*>(&total_1[(36) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&total_1[(36) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&total_1[(36) + 3]))
                    : "r"(addr_25));
                total_1[36] = total_1[36] + vals[100];
                total_1[37] = total_1[37] + vals[101];
                total_1[38] = total_1[38] + vals[102];
                total_1[39] = total_1[39] + vals[103];
                unsigned int addr_26 = red_buf_addr + (unsigned int)((1280 + tid_1) * 16);
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&total_1[40])), "=r"(*reinterpret_cast<uint32_t*>(&total_1[(40) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&total_1[(40) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&total_1[(40) + 3]))
                    : "r"(addr_26));
                total_1[40] = total_1[40] + vals[104];
                total_1[41] = total_1[41] + vals[105];
                total_1[42] = total_1[42] + vals[106];
                total_1[43] = total_1[43] + vals[107];
                unsigned int addr_27 = red_buf_addr + (unsigned int)((1408 + tid_1) * 16);
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&total_1[44])), "=r"(*reinterpret_cast<uint32_t*>(&total_1[(44) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&total_1[(44) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&total_1[(44) + 3]))
                    : "r"(addr_27));
                total_1[44] = total_1[44] + vals[108];
                total_1[45] = total_1[45] + vals[109];
                total_1[46] = total_1[46] + vals[110];
                total_1[47] = total_1[47] + vals[111];
                unsigned int addr_28 = red_buf_addr + (unsigned int)((1536 + tid_1) * 16);
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&total_1[48])), "=r"(*reinterpret_cast<uint32_t*>(&total_1[(48) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&total_1[(48) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&total_1[(48) + 3]))
                    : "r"(addr_28));
                total_1[48] = total_1[48] + vals[112];
                total_1[49] = total_1[49] + vals[113];
                total_1[50] = total_1[50] + vals[114];
                total_1[51] = total_1[51] + vals[115];
                unsigned int addr_29 = red_buf_addr + (unsigned int)((1664 + tid_1) * 16);
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&total_1[52])), "=r"(*reinterpret_cast<uint32_t*>(&total_1[(52) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&total_1[(52) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&total_1[(52) + 3]))
                    : "r"(addr_29));
                total_1[52] = total_1[52] + vals[116];
                total_1[53] = total_1[53] + vals[117];
                total_1[54] = total_1[54] + vals[118];
                total_1[55] = total_1[55] + vals[119];
                unsigned int addr_30 = red_buf_addr + (unsigned int)((1792 + tid_1) * 16);
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&total_1[56])), "=r"(*reinterpret_cast<uint32_t*>(&total_1[(56) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&total_1[(56) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&total_1[(56) + 3]))
                    : "r"(addr_30));
                total_1[56] = total_1[56] + vals[120];
                total_1[57] = total_1[57] + vals[121];
                total_1[58] = total_1[58] + vals[122];
                total_1[59] = total_1[59] + vals[123];
                unsigned int addr_31 = red_buf_addr + (unsigned int)((1920 + tid_1) * 16);
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&total_1[60])), "=r"(*reinterpret_cast<uint32_t*>(&total_1[(60) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&total_1[(60) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&total_1[(60) + 3]))
                    : "r"(addr_31));
                total_1[60] = total_1[60] + vals[124];
                total_1[61] = total_1[61] + vals[125];
                total_1[62] = total_1[62] + vals[126];
                total_1[63] = total_1[63] + vals[127];
                if (row < M) {
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t + (long long)(col0 + 64) * (long long)ldo))), "f"(total_1[(0) + 0]), "r"((unsigned int)(col0 + 64 < N)) : "memory");
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t + (long long)(col0 + 64 + 1) * (long long)ldo))), "f"(total_1[(1) + 0]), "r"((unsigned int)(col0 + 64 + 1 < N)) : "memory");
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t + (long long)(col0 + 64 + 2) * (long long)ldo))), "f"(total_1[(2) + 0]), "r"((unsigned int)(col0 + 64 + 2 < N)) : "memory");
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t + (long long)(col0 + 64 + 3) * (long long)ldo))), "f"(total_1[(3) + 0]), "r"((unsigned int)(col0 + 64 + 3 < N)) : "memory");
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t + (long long)(col0 + 68) * (long long)ldo))), "f"(total_1[(4) + 0]), "r"((unsigned int)(col0 + 68 < N)) : "memory");
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t + (long long)(col0 + 68 + 1) * (long long)ldo))), "f"(total_1[(5) + 0]), "r"((unsigned int)(col0 + 68 + 1 < N)) : "memory");
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t + (long long)(col0 + 68 + 2) * (long long)ldo))), "f"(total_1[(6) + 0]), "r"((unsigned int)(col0 + 68 + 2 < N)) : "memory");
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t + (long long)(col0 + 68 + 3) * (long long)ldo))), "f"(total_1[(7) + 0]), "r"((unsigned int)(col0 + 68 + 3 < N)) : "memory");
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t + (long long)(col0 + 72) * (long long)ldo))), "f"(total_1[(8) + 0]), "r"((unsigned int)(col0 + 72 < N)) : "memory");
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t + (long long)(col0 + 72 + 1) * (long long)ldo))), "f"(total_1[(9) + 0]), "r"((unsigned int)(col0 + 72 + 1 < N)) : "memory");
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t + (long long)(col0 + 72 + 2) * (long long)ldo))), "f"(total_1[(10) + 0]), "r"((unsigned int)(col0 + 72 + 2 < N)) : "memory");
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t + (long long)(col0 + 72 + 3) * (long long)ldo))), "f"(total_1[(11) + 0]), "r"((unsigned int)(col0 + 72 + 3 < N)) : "memory");
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t + (long long)(col0 + 76) * (long long)ldo))), "f"(total_1[(12) + 0]), "r"((unsigned int)(col0 + 76 < N)) : "memory");
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t + (long long)(col0 + 76 + 1) * (long long)ldo))), "f"(total_1[(13) + 0]), "r"((unsigned int)(col0 + 76 + 1 < N)) : "memory");
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t + (long long)(col0 + 76 + 2) * (long long)ldo))), "f"(total_1[(14) + 0]), "r"((unsigned int)(col0 + 76 + 2 < N)) : "memory");
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t + (long long)(col0 + 76 + 3) * (long long)ldo))), "f"(total_1[(15) + 0]), "r"((unsigned int)(col0 + 76 + 3 < N)) : "memory");
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t + (long long)(col0 + 80) * (long long)ldo))), "f"(total_1[(16) + 0]), "r"((unsigned int)(col0 + 80 < N)) : "memory");
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t + (long long)(col0 + 80 + 1) * (long long)ldo))), "f"(total_1[(17) + 0]), "r"((unsigned int)(col0 + 80 + 1 < N)) : "memory");
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t + (long long)(col0 + 80 + 2) * (long long)ldo))), "f"(total_1[(18) + 0]), "r"((unsigned int)(col0 + 80 + 2 < N)) : "memory");
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t + (long long)(col0 + 80 + 3) * (long long)ldo))), "f"(total_1[(19) + 0]), "r"((unsigned int)(col0 + 80 + 3 < N)) : "memory");
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t + (long long)(col0 + 84) * (long long)ldo))), "f"(total_1[(20) + 0]), "r"((unsigned int)(col0 + 84 < N)) : "memory");
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t + (long long)(col0 + 84 + 1) * (long long)ldo))), "f"(total_1[(21) + 0]), "r"((unsigned int)(col0 + 84 + 1 < N)) : "memory");
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t + (long long)(col0 + 84 + 2) * (long long)ldo))), "f"(total_1[(22) + 0]), "r"((unsigned int)(col0 + 84 + 2 < N)) : "memory");
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t + (long long)(col0 + 84 + 3) * (long long)ldo))), "f"(total_1[(23) + 0]), "r"((unsigned int)(col0 + 84 + 3 < N)) : "memory");
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t + (long long)(col0 + 88) * (long long)ldo))), "f"(total_1[(24) + 0]), "r"((unsigned int)(col0 + 88 < N)) : "memory");
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t + (long long)(col0 + 88 + 1) * (long long)ldo))), "f"(total_1[(25) + 0]), "r"((unsigned int)(col0 + 88 + 1 < N)) : "memory");
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t + (long long)(col0 + 88 + 2) * (long long)ldo))), "f"(total_1[(26) + 0]), "r"((unsigned int)(col0 + 88 + 2 < N)) : "memory");
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t + (long long)(col0 + 88 + 3) * (long long)ldo))), "f"(total_1[(27) + 0]), "r"((unsigned int)(col0 + 88 + 3 < N)) : "memory");
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t + (long long)(col0 + 92) * (long long)ldo))), "f"(total_1[(28) + 0]), "r"((unsigned int)(col0 + 92 < N)) : "memory");
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t + (long long)(col0 + 92 + 1) * (long long)ldo))), "f"(total_1[(29) + 0]), "r"((unsigned int)(col0 + 92 + 1 < N)) : "memory");
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t + (long long)(col0 + 92 + 2) * (long long)ldo))), "f"(total_1[(30) + 0]), "r"((unsigned int)(col0 + 92 + 2 < N)) : "memory");
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t + (long long)(col0 + 92 + 3) * (long long)ldo))), "f"(total_1[(31) + 0]), "r"((unsigned int)(col0 + 92 + 3 < N)) : "memory");
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t + (long long)(col0 + 96) * (long long)ldo))), "f"(total_1[(32) + 0]), "r"((unsigned int)(col0 + 96 < N)) : "memory");
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t + (long long)(col0 + 96 + 1) * (long long)ldo))), "f"(total_1[(33) + 0]), "r"((unsigned int)(col0 + 96 + 1 < N)) : "memory");
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t + (long long)(col0 + 96 + 2) * (long long)ldo))), "f"(total_1[(34) + 0]), "r"((unsigned int)(col0 + 96 + 2 < N)) : "memory");
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t + (long long)(col0 + 96 + 3) * (long long)ldo))), "f"(total_1[(35) + 0]), "r"((unsigned int)(col0 + 96 + 3 < N)) : "memory");
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t + (long long)(col0 + 100) * (long long)ldo))), "f"(total_1[(36) + 0]), "r"((unsigned int)(col0 + 100 < N)) : "memory");
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t + (long long)(col0 + 100 + 1) * (long long)ldo))), "f"(total_1[(37) + 0]), "r"((unsigned int)(col0 + 100 + 1 < N)) : "memory");
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t + (long long)(col0 + 100 + 2) * (long long)ldo))), "f"(total_1[(38) + 0]), "r"((unsigned int)(col0 + 100 + 2 < N)) : "memory");
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t + (long long)(col0 + 100 + 3) * (long long)ldo))), "f"(total_1[(39) + 0]), "r"((unsigned int)(col0 + 100 + 3 < N)) : "memory");
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t + (long long)(col0 + 104) * (long long)ldo))), "f"(total_1[(40) + 0]), "r"((unsigned int)(col0 + 104 < N)) : "memory");
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t + (long long)(col0 + 104 + 1) * (long long)ldo))), "f"(total_1[(41) + 0]), "r"((unsigned int)(col0 + 104 + 1 < N)) : "memory");
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t + (long long)(col0 + 104 + 2) * (long long)ldo))), "f"(total_1[(42) + 0]), "r"((unsigned int)(col0 + 104 + 2 < N)) : "memory");
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t + (long long)(col0 + 104 + 3) * (long long)ldo))), "f"(total_1[(43) + 0]), "r"((unsigned int)(col0 + 104 + 3 < N)) : "memory");
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t + (long long)(col0 + 108) * (long long)ldo))), "f"(total_1[(44) + 0]), "r"((unsigned int)(col0 + 108 < N)) : "memory");
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t + (long long)(col0 + 108 + 1) * (long long)ldo))), "f"(total_1[(45) + 0]), "r"((unsigned int)(col0 + 108 + 1 < N)) : "memory");
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t + (long long)(col0 + 108 + 2) * (long long)ldo))), "f"(total_1[(46) + 0]), "r"((unsigned int)(col0 + 108 + 2 < N)) : "memory");
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t + (long long)(col0 + 108 + 3) * (long long)ldo))), "f"(total_1[(47) + 0]), "r"((unsigned int)(col0 + 108 + 3 < N)) : "memory");
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t + (long long)(col0 + 112) * (long long)ldo))), "f"(total_1[(48) + 0]), "r"((unsigned int)(col0 + 112 < N)) : "memory");
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t + (long long)(col0 + 112 + 1) * (long long)ldo))), "f"(total_1[(49) + 0]), "r"((unsigned int)(col0 + 112 + 1 < N)) : "memory");
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t + (long long)(col0 + 112 + 2) * (long long)ldo))), "f"(total_1[(50) + 0]), "r"((unsigned int)(col0 + 112 + 2 < N)) : "memory");
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t + (long long)(col0 + 112 + 3) * (long long)ldo))), "f"(total_1[(51) + 0]), "r"((unsigned int)(col0 + 112 + 3 < N)) : "memory");
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t + (long long)(col0 + 116) * (long long)ldo))), "f"(total_1[(52) + 0]), "r"((unsigned int)(col0 + 116 < N)) : "memory");
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t + (long long)(col0 + 116 + 1) * (long long)ldo))), "f"(total_1[(53) + 0]), "r"((unsigned int)(col0 + 116 + 1 < N)) : "memory");
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t + (long long)(col0 + 116 + 2) * (long long)ldo))), "f"(total_1[(54) + 0]), "r"((unsigned int)(col0 + 116 + 2 < N)) : "memory");
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t + (long long)(col0 + 116 + 3) * (long long)ldo))), "f"(total_1[(55) + 0]), "r"((unsigned int)(col0 + 116 + 3 < N)) : "memory");
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t + (long long)(col0 + 120) * (long long)ldo))), "f"(total_1[(56) + 0]), "r"((unsigned int)(col0 + 120 < N)) : "memory");
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t + (long long)(col0 + 120 + 1) * (long long)ldo))), "f"(total_1[(57) + 0]), "r"((unsigned int)(col0 + 120 + 1 < N)) : "memory");
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t + (long long)(col0 + 120 + 2) * (long long)ldo))), "f"(total_1[(58) + 0]), "r"((unsigned int)(col0 + 120 + 2 < N)) : "memory");
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t + (long long)(col0 + 120 + 3) * (long long)ldo))), "f"(total_1[(59) + 0]), "r"((unsigned int)(col0 + 120 + 3 < N)) : "memory");
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t + (long long)(col0 + 124) * (long long)ldo))), "f"(total_1[(60) + 0]), "r"((unsigned int)(col0 + 124 < N)) : "memory");
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t + (long long)(col0 + 124 + 1) * (long long)ldo))), "f"(total_1[(61) + 0]), "r"((unsigned int)(col0 + 124 + 1 < N)) : "memory");
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t + (long long)(col0 + 124 + 2) * (long long)ldo))), "f"(total_1[(62) + 0]), "r"((unsigned int)(col0 + 124 + 2 < N)) : "memory");
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t + (long long)(col0 + 124 + 3) * (long long)ldo))), "f"(total_1[(63) + 0]), "r"((unsigned int)(col0 + 124 + 3 < N)) : "memory");
                }
            }
            asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
        }
    }

    // Cleanup
}

} // extern "C"
