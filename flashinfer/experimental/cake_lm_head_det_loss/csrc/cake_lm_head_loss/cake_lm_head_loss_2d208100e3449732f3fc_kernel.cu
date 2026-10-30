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
#define TMEM_NCOLS 256
#define TMEM_ACC_OFFSET 0
#define NUM_RING_STAGES 6
#define NUM_ACC_RING_STAGES 2
#define NUM_META_RING_STAGES 4
#define SMEM_SMEM_A_OFF 1024
#define SMEM_SMEM_A_STAGE_BYTES 16384
#define SMEM_SMEM_A_STRIDE 32768
#define SMEM_SMEM_B_OFF 17408
#define SMEM_SMEM_B_STAGE_BYTES 16384
#define SMEM_SMEM_B_STRIDE 32768
#define SMEM_META_IDS_OFF 197632
#define SMEM_META_IDS_STAGE_BYTES 2048
#define SMEM_META_IDS_STRIDE 2048
#define SMEM_META_TOK_OFF 199680
#define SMEM_META_TOK_STAGE_BYTES 2048
#define SMEM_META_TOK_STRIDE 2048
#define SMEM_META_COL_OFF 201728
#define SMEM_META_COL_STAGE_BYTES 2048
#define SMEM_META_COL_STRIDE 2048
#define SMEM_META_N_OFF 203776
#define SMEM_META_N_STAGE_BYTES 16
#define SMEM_META_N_STRIDE 16
#define SMEM_TOTAL 203904
#define THREADS 320
#if defined(__CUDACC_VER_MAJOR__) && (__CUDA_ARCH__ == 1000) && ((__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ == 9) || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ == 0))
#define CAKE_ARRIVE_FIRST 1
#else
#define CAKE_ARRIVE_FIRST 0
#endif

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



__device__ __forceinline__ void mbarrier_expect_tx(int mbar_addr, uint32_t bytes) {
    asm volatile(
        "mbarrier.expect_tx.relaxed.cta.shared::cta.b64 [%0], %1;"
        :: "r"(mbar_addr), "r"(bytes) : "memory");
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






__device__ __forceinline__ void tma_gather4_gmem2smem(
    int dst, const void *tmap_ptr,
    int col_idx, int row0, int row1, int row2, int row3,
    int mbar_addr) {
    // Non-multicast, single-CTA gather4 in the .shared::cta window (the
    // cuda::ptx form): the destination and the barrier are the executing
    // CTA's. A CTA pair's gather takes the .shared::cluster window
    // (tma_gather4_gmem2smem_cta2), its barrier being the pair leader's.
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
        ".mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3, %4, %5, %6}], [%7];"
        :: "r"(dst), "l"(tmap_ptr), "r"(col_idx),
           "r"(row0), "r"(row1), "r"(row2), "r"(row3),
           "r"(mbar_addr) : "memory");
}


__device__ __forceinline__ void tcgen05_commit(int mbar_addr) {
    asm volatile(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one"
        ".shared::cluster.b64 [%0];"
        :: "r"(mbar_addr) : "memory");
}

extern "C" {

__global__ __launch_bounds__(320) void
kernel_cake_lm_head_loss_2d208100e3449732f3fc(const __grid_constant__ CUtensorMap W, const __grid_constant__ CUtensorMap X, __nv_bfloat16* __restrict__ z, int* __restrict__ ids, int* __restrict__ offsets, int M, int T, int V, int H)
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
    #define full_addr (mbar_base + 0)
    #define empty_addr (mbar_base + 48)
    #define acc_full_addr (mbar_base + 96)
    #define acc_empty_addr (mbar_base + 112)
    #define meta_full_addr (mbar_base + 128)
    #define meta_empty_addr (mbar_base + 160)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    __nv_bfloat16* smem_a = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int smem_a_addr = smem + 1024;
    __nv_bfloat16* smem_b = reinterpret_cast<__nv_bfloat16*>(smem_raw + 17408);
    const int smem_b_addr = smem + 17408;
    int* meta_ids = reinterpret_cast<int*>(smem_raw + 197632);
    const int meta_ids_addr = smem + 197632;
    int* meta_tok = reinterpret_cast<int*>(smem_raw + 199680);
    const int meta_tok_addr = smem + 199680;
    int* meta_col = reinterpret_cast<int*>(smem_raw + 201728);
    const int meta_col_addr = smem + 201728;
    int* meta_n = reinterpret_cast<int*>(smem_raw + 203776);
    const int meta_n_addr = smem + 203776;

    // Mbarrier init (6 pipeline groups, 0 ordered-sequence groups, 24 barriers)
    // Mbarriers at smem_raw[0..192)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'ring' ---
            // full: 6 barriers, init_count=4
            mbarrier_init(smem + 0, 4);
            mbarrier_init(smem + 8, 4);
            mbarrier_init(smem + 16, 4);
            mbarrier_init(smem + 24, 4);
            mbarrier_init(smem + 32, 4);
            mbarrier_init(smem + 40, 4);
            // empty: 6 barriers, init_count=1
            mbarrier_init(smem + 48, 1);
            mbarrier_init(smem + 56, 1);
            mbarrier_init(smem + 64, 1);
            mbarrier_init(smem + 72, 1);
            mbarrier_init(smem + 80, 1);
            mbarrier_init(smem + 88, 1);
            // --- pipeline 'acc_ring' ---
            // acc_full: 2 barriers, init_count=1
            mbarrier_init(smem + 96, 1);
            mbarrier_init(smem + 104, 1);
            // acc_empty: 2 barriers, init_count=4
            mbarrier_init(smem + 112, 4);
            mbarrier_init(smem + 120, 4);
            // --- pipeline 'meta_ring' ---
            // meta_full: 4 barriers, init_count=32
            mbarrier_init(smem + 128, 32);
            mbarrier_init(smem + 136, 32);
            mbarrier_init(smem + 144, 32);
            mbarrier_init(smem + 152, 32);
            // meta_empty: 4 barriers, init_count=132
            mbarrier_init(smem + 160, 132);
            mbarrier_init(smem + 168, 132);
            mbarrier_init(smem + 176, 132);
            mbarrier_init(smem + 184, 132);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    // TMEM alloc (256 columns, 256 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 192);
    if (warp == 0) {
        int _tmem_hold = smem + 192;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(256) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_acc = taddr;

    // ---- Role: load ----
    if (warp <= 3) {
        { // load_main
            unsigned int slot = 0;
            int ms = 0;
            const int pw = warp;
            int a_idx[32];
            int b_idx[32];
            int n_tiles = (M + 127) / 128;
            int k_iters = H / 64;
            unsigned int _phase_meta_full = 0;
            unsigned int _phase_empty = 1;
            if (elect_sync()) {
                #pragma unroll 1
                for (int tile = bid; tile < n_tiles; tile += num_bids) {
                    mbarrier_wait(meta_full_addr + (ms) * 8, _phase_meta_full);
                    int _max_0 = ((meta_n[ms]) > (1) ? (meta_n[ms]) : (1));
                    int n_tok = ((_max_0) < (128) ? (_max_0) : (128));
                    a_idx[0] = meta_ids[ms * 128 + pw * 32];
                    a_idx[1] = meta_ids[ms * 128 + (pw * 32 + 1)];
                    a_idx[2] = meta_ids[ms * 128 + (pw * 32 + 2)];
                    a_idx[3] = meta_ids[ms * 128 + (pw * 32 + 3)];
                    a_idx[4] = meta_ids[ms * 128 + (pw * 32 + 4)];
                    a_idx[5] = meta_ids[ms * 128 + (pw * 32 + 5)];
                    a_idx[6] = meta_ids[ms * 128 + (pw * 32 + 6)];
                    a_idx[7] = meta_ids[ms * 128 + (pw * 32 + 7)];
                    a_idx[8] = meta_ids[ms * 128 + (pw * 32 + 8)];
                    a_idx[9] = meta_ids[ms * 128 + (pw * 32 + 9)];
                    a_idx[10] = meta_ids[ms * 128 + (pw * 32 + 10)];
                    a_idx[11] = meta_ids[ms * 128 + (pw * 32 + 11)];
                    a_idx[12] = meta_ids[ms * 128 + (pw * 32 + 12)];
                    a_idx[13] = meta_ids[ms * 128 + (pw * 32 + 13)];
                    a_idx[14] = meta_ids[ms * 128 + (pw * 32 + 14)];
                    a_idx[15] = meta_ids[ms * 128 + (pw * 32 + 15)];
                    a_idx[16] = meta_ids[ms * 128 + (pw * 32 + 16)];
                    a_idx[17] = meta_ids[ms * 128 + (pw * 32 + 17)];
                    a_idx[18] = meta_ids[ms * 128 + (pw * 32 + 18)];
                    a_idx[19] = meta_ids[ms * 128 + (pw * 32 + 19)];
                    a_idx[20] = meta_ids[ms * 128 + (pw * 32 + 20)];
                    a_idx[21] = meta_ids[ms * 128 + (pw * 32 + 21)];
                    a_idx[22] = meta_ids[ms * 128 + (pw * 32 + 22)];
                    a_idx[23] = meta_ids[ms * 128 + (pw * 32 + 23)];
                    a_idx[24] = meta_ids[ms * 128 + (pw * 32 + 24)];
                    a_idx[25] = meta_ids[ms * 128 + (pw * 32 + 25)];
                    a_idx[26] = meta_ids[ms * 128 + (pw * 32 + 26)];
                    a_idx[27] = meta_ids[ms * 128 + (pw * 32 + 27)];
                    a_idx[28] = meta_ids[ms * 128 + (pw * 32 + 28)];
                    a_idx[29] = meta_ids[ms * 128 + (pw * 32 + 29)];
                    a_idx[30] = meta_ids[ms * 128 + (pw * 32 + 30)];
                    a_idx[31] = meta_ids[ms * 128 + (pw * 32 + 31)];
                    b_idx[0] = meta_tok[ms * 128 + pw * 4];
                    b_idx[1] = meta_tok[ms * 128 + (pw * 4 + 1)];
                    b_idx[2] = meta_tok[ms * 128 + (pw * 4 + 2)];
                    b_idx[3] = meta_tok[ms * 128 + (pw * 4 + 3)];
                    b_idx[4] = meta_tok[ms * 128 + (pw + 4) * 4];
                    b_idx[5] = meta_tok[ms * 128 + ((pw + 4) * 4 + 1)];
                    b_idx[6] = meta_tok[ms * 128 + ((pw + 4) * 4 + 2)];
                    b_idx[7] = meta_tok[ms * 128 + ((pw + 4) * 4 + 3)];
                    b_idx[8] = meta_tok[ms * 128 + (pw + 8) * 4];
                    b_idx[9] = meta_tok[ms * 128 + ((pw + 8) * 4 + 1)];
                    b_idx[10] = meta_tok[ms * 128 + ((pw + 8) * 4 + 2)];
                    b_idx[11] = meta_tok[ms * 128 + ((pw + 8) * 4 + 3)];
                    b_idx[12] = meta_tok[ms * 128 + (pw + 12) * 4];
                    b_idx[13] = meta_tok[ms * 128 + ((pw + 12) * 4 + 1)];
                    b_idx[14] = meta_tok[ms * 128 + ((pw + 12) * 4 + 2)];
                    b_idx[15] = meta_tok[ms * 128 + ((pw + 12) * 4 + 3)];
                    b_idx[16] = meta_tok[ms * 128 + (pw + 16) * 4];
                    b_idx[17] = meta_tok[ms * 128 + ((pw + 16) * 4 + 1)];
                    b_idx[18] = meta_tok[ms * 128 + ((pw + 16) * 4 + 2)];
                    b_idx[19] = meta_tok[ms * 128 + ((pw + 16) * 4 + 3)];
                    b_idx[20] = meta_tok[ms * 128 + (pw + 20) * 4];
                    b_idx[21] = meta_tok[ms * 128 + ((pw + 20) * 4 + 1)];
                    b_idx[22] = meta_tok[ms * 128 + ((pw + 20) * 4 + 2)];
                    b_idx[23] = meta_tok[ms * 128 + ((pw + 20) * 4 + 3)];
                    b_idx[24] = meta_tok[ms * 128 + (pw + 24) * 4];
                    b_idx[25] = meta_tok[ms * 128 + ((pw + 24) * 4 + 1)];
                    b_idx[26] = meta_tok[ms * 128 + ((pw + 24) * 4 + 2)];
                    b_idx[27] = meta_tok[ms * 128 + ((pw + 24) * 4 + 3)];
                    b_idx[28] = meta_tok[ms * 128 + (pw + 28) * 4];
                    b_idx[29] = meta_tok[ms * 128 + ((pw + 28) * 4 + 1)];
                    b_idx[30] = meta_tok[ms * 128 + ((pw + 28) * 4 + 2)];
                    b_idx[31] = meta_tok[ms * 128 + ((pw + 28) * 4 + 3)];
                    mbarrier_arrive(meta_empty_addr + (ms) * 8);
                    ms += 1;
                    if (ms == 4) { ms = 0; _phase_meta_full ^= 1; }
                    int _max_1 = ((((n_tok + 3) / 4 - pw + 3) / 4) > (0) ? (((n_tok + 3) / 4 - pw + 3) / 4) : (0));
                    int b_ops = ((_max_1) < (8) ? (_max_1) : (8));
                    #pragma unroll 1
                    for (int kb = 0; kb < k_iters; kb++) {
                        mbarrier_wait(empty_addr + (slot) * 8, _phase_empty);
                        int col0 = kb * 64;
                        if (0 < b_ops) {
                            mbarrier_expect_tx(full_addr + (slot) * 8, 512);
                            tma_gather4_gmem2smem(smem_b_addr + slot * 32768 + (unsigned int)(pw * 512), (&X), col0, b_idx[0], b_idx[1], b_idx[2], b_idx[3], full_addr + (slot) * 8);
                        }
                        if (1 < b_ops) {
                            mbarrier_expect_tx(full_addr + (slot) * 8, 512);
                            tma_gather4_gmem2smem(smem_b_addr + slot * 32768 + (unsigned int)((pw + 4) * 512), (&X), col0, b_idx[4], b_idx[5], b_idx[6], b_idx[7], full_addr + (slot) * 8);
                        }
                        if (2 < b_ops) {
                            mbarrier_expect_tx(full_addr + (slot) * 8, 512);
                            tma_gather4_gmem2smem(smem_b_addr + slot * 32768 + (unsigned int)((pw + 8) * 512), (&X), col0, b_idx[8], b_idx[9], b_idx[10], b_idx[11], full_addr + (slot) * 8);
                        }
                        if (3 < b_ops) {
                            mbarrier_expect_tx(full_addr + (slot) * 8, 512);
                            tma_gather4_gmem2smem(smem_b_addr + slot * 32768 + (unsigned int)((pw + 12) * 512), (&X), col0, b_idx[12], b_idx[13], b_idx[14], b_idx[15], full_addr + (slot) * 8);
                        }
                        if (4 < b_ops) {
                            mbarrier_expect_tx(full_addr + (slot) * 8, 512);
                            tma_gather4_gmem2smem(smem_b_addr + slot * 32768 + (unsigned int)((pw + 16) * 512), (&X), col0, b_idx[16], b_idx[17], b_idx[18], b_idx[19], full_addr + (slot) * 8);
                        }
                        if (5 < b_ops) {
                            mbarrier_expect_tx(full_addr + (slot) * 8, 512);
                            tma_gather4_gmem2smem(smem_b_addr + slot * 32768 + (unsigned int)((pw + 20) * 512), (&X), col0, b_idx[20], b_idx[21], b_idx[22], b_idx[23], full_addr + (slot) * 8);
                        }
                        if (6 < b_ops) {
                            mbarrier_expect_tx(full_addr + (slot) * 8, 512);
                            tma_gather4_gmem2smem(smem_b_addr + slot * 32768 + (unsigned int)((pw + 24) * 512), (&X), col0, b_idx[24], b_idx[25], b_idx[26], b_idx[27], full_addr + (slot) * 8);
                        }
                        if (7 < b_ops) {
                            mbarrier_expect_tx(full_addr + (slot) * 8, 512);
                            tma_gather4_gmem2smem(smem_b_addr + slot * 32768 + (unsigned int)((pw + 28) * 512), (&X), col0, b_idx[28], b_idx[29], b_idx[30], b_idx[31], full_addr + (slot) * 8);
                        }
#if CAKE_ARRIVE_FIRST
                        mbarrier_arrive_expect_tx(full_addr + (slot) * 8, 4096);
#endif
                        tma_gather4_gmem2smem(smem_a_addr + slot * 32768 + (unsigned int)(pw * 32 * 128), (&W), col0, a_idx[0], a_idx[1], a_idx[2], a_idx[3], full_addr + (slot) * 8);
                        tma_gather4_gmem2smem(smem_a_addr + slot * 32768 + (unsigned int)((pw * 32 + 4) * 128), (&W), col0, a_idx[4], a_idx[5], a_idx[6], a_idx[7], full_addr + (slot) * 8);
                        tma_gather4_gmem2smem(smem_a_addr + slot * 32768 + (unsigned int)((pw * 32 + 8) * 128), (&W), col0, a_idx[8], a_idx[9], a_idx[10], a_idx[11], full_addr + (slot) * 8);
                        tma_gather4_gmem2smem(smem_a_addr + slot * 32768 + (unsigned int)((pw * 32 + 12) * 128), (&W), col0, a_idx[12], a_idx[13], a_idx[14], a_idx[15], full_addr + (slot) * 8);
                        tma_gather4_gmem2smem(smem_a_addr + slot * 32768 + (unsigned int)((pw * 32 + 16) * 128), (&W), col0, a_idx[16], a_idx[17], a_idx[18], a_idx[19], full_addr + (slot) * 8);
                        tma_gather4_gmem2smem(smem_a_addr + slot * 32768 + (unsigned int)((pw * 32 + 20) * 128), (&W), col0, a_idx[20], a_idx[21], a_idx[22], a_idx[23], full_addr + (slot) * 8);
                        tma_gather4_gmem2smem(smem_a_addr + slot * 32768 + (unsigned int)((pw * 32 + 24) * 128), (&W), col0, a_idx[24], a_idx[25], a_idx[26], a_idx[27], full_addr + (slot) * 8);
                        tma_gather4_gmem2smem(smem_a_addr + slot * 32768 + (unsigned int)((pw * 32 + 28) * 128), (&W), col0, a_idx[28], a_idx[29], a_idx[30], a_idx[31], full_addr + (slot) * 8);
#if CAKE_ARRIVE_FIRST
#else
                        mbarrier_arrive_expect_tx(full_addr + (slot) * 8, 4096);
#endif
                        slot += 1;
                        if (slot == 6) { slot = 0; _phase_empty ^= 1; }
                    }
                }
            }
        }
    }
    // ---- Role: mma ----
    if (warp == 4) {
        { // mma_main
            unsigned int slot_1 = 0;
            unsigned int acc = 0;
            int n_tiles_1 = (M + 127) / 128;
            int k_iters_1 = H / 64;
            unsigned int _phase_acc_empty = 1;
            unsigned int _phase_full = 0;
            if (elect_sync()) {
                #pragma unroll 1
                for (int tile_1 = bid; tile_1 < n_tiles_1; tile_1 += num_bids) {
                    mbarrier_wait(acc_empty_addr + (acc) * 8, _phase_acc_empty);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    #pragma unroll 1
                    for (int kb_1 = 0; kb_1 < k_iters_1; kb_1++) {
                        mbarrier_wait(full_addr + (slot_1) * 8, _phase_full);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int first = ((kb_1 == 0) ? 1 : 0);
                        int _smem_desc_lo_0 = (((smem_a_addr) >> 4) & 0x3FFF) + (slot_1) * 2048;
                        int _smem_desc_lo_1 = (((smem_b_addr) >> 4) & 0x3FFF) + (slot_1) * 2048;
                        asm volatile(
                    "{\n\t"
                    ".reg .pred p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    ""
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 136316048;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_smem_desc_lo_0), "r"(_smem_desc_lo_1), "r"((tmem_acc + (acc * 128))), "r"(((first) ? 0 : 1)) : "memory");
                        tcgen05_commit(empty_addr + (slot_1) * 8);
                        slot_1 += 1;
                        if (slot_1 == 6) { slot_1 = 0; _phase_full ^= 1; }
                    }
                    tcgen05_commit(acc_full_addr + (acc) * 8);
                    acc += 1;
                    if (acc == 2) { acc = 0; _phase_acc_empty ^= 1; }
                }
            }
        }
    }
    // ---- Role: meta ----
    if (warp == 5) {
        { // meta_main
            int ms_1 = 0;
            int n_tiles_2 = (M + 127) / 128;
            unsigned int lanes_below = (1 << (unsigned int)lane) - 1;
            unsigned int _phase_meta_empty = 1;
            #pragma unroll 1
            for (int tile_2 = bid; tile_2 < n_tiles_2; tile_2 += num_bids) {
                int key[4];
                int vid[4];
                int row[4];
                long long pair_0 = (long long)tile_2 * 128 + (long long)lane;
                long long _min_0 = ((pair_0) < ((long long)M - 1) ? (pair_0) : ((long long)M - 1));
                key[0] = (int)_min_0;
                int _max_2 = ((ids[key[0]]) > (0) ? (ids[key[0]]) : (0));
                int _min_1 = ((_max_2) < (V - 1) ? (_max_2) : (V - 1));
                vid[0] = _min_1;
                int const _table_search_3_value = key[0];
                int _table_search_3 = 0;
                int _table_search_3_hi = T - 1;
                while (_table_search_3 < _table_search_3_hi) {
                    int mid = _table_search_3 + _table_search_3_hi + 1 >> 1;
                    if (_table_search_3_value >= offsets[mid]) {
                        _table_search_3 = mid;
                    } else {
                        _table_search_3_hi = mid - 1;
                    }
                }
                row[0] = _table_search_3;
                long long pair_1 = (long long)tile_2 * 128 + (long long)(32 + lane);
                long long _min_4 = ((pair_1) < ((long long)M - 1) ? (pair_1) : ((long long)M - 1));
                key[1] = (int)_min_4;
                int _max_6 = ((ids[key[1]]) > (0) ? (ids[key[1]]) : (0));
                int _min_5 = ((_max_6) < (V - 1) ? (_max_6) : (V - 1));
                vid[1] = _min_5;
                int const _table_search_7_value = key[1];
                int _table_search_7 = 0;
                int _table_search_7_hi = T - 1;
                while (_table_search_7 < _table_search_7_hi) {
                    int mid_1 = _table_search_7 + _table_search_7_hi + 1 >> 1;
                    if (_table_search_7_value >= offsets[mid_1]) {
                        _table_search_7 = mid_1;
                    } else {
                        _table_search_7_hi = mid_1 - 1;
                    }
                }
                row[1] = _table_search_7;
                long long pair_2 = (long long)tile_2 * 128 + (long long)(64 + lane);
                long long _min_8 = ((pair_2) < ((long long)M - 1) ? (pair_2) : ((long long)M - 1));
                key[2] = (int)_min_8;
                int _max_10 = ((ids[key[2]]) > (0) ? (ids[key[2]]) : (0));
                int _min_9 = ((_max_10) < (V - 1) ? (_max_10) : (V - 1));
                vid[2] = _min_9;
                int const _table_search_11_value = key[2];
                int _table_search_11 = 0;
                int _table_search_11_hi = T - 1;
                while (_table_search_11 < _table_search_11_hi) {
                    int mid_2 = _table_search_11 + _table_search_11_hi + 1 >> 1;
                    if (_table_search_11_value >= offsets[mid_2]) {
                        _table_search_11 = mid_2;
                    } else {
                        _table_search_11_hi = mid_2 - 1;
                    }
                }
                row[2] = _table_search_11;
                long long pair_3 = (long long)tile_2 * 128 + (long long)(96 + lane);
                long long _min_12 = ((pair_3) < ((long long)M - 1) ? (pair_3) : ((long long)M - 1));
                key[3] = (int)_min_12;
                int _max_14 = ((ids[key[3]]) > (0) ? (ids[key[3]]) : (0));
                int _min_13 = ((_max_14) < (V - 1) ? (_max_14) : (V - 1));
                vid[3] = _min_13;
                int const _table_search_15_value = key[3];
                int _table_search_15 = 0;
                int _table_search_15_hi = T - 1;
                while (_table_search_15 < _table_search_15_hi) {
                    int mid_3 = _table_search_15 + _table_search_15_hi + 1 >> 1;
                    if (_table_search_15_value >= offsets[mid_3]) {
                        _table_search_15 = mid_3;
                    } else {
                        _table_search_15_hi = mid_3 - 1;
                    }
                }
                row[3] = _table_search_15;
                int col[4];
                int head[4];
                int n_tok_1 = 0;
                int prev_last = -1;
                int up_0 = __shfl_up_sync(0xFFFFFFFF, row[0], 1, 32);
                int prev_0 = ((lane == 0) ? prev_last : up_0);
                unsigned int heads_0 = __ballot_sync(0xFFFFFFFF, row[0] != prev_0);
                head[0] = ((row[0] != prev_0) ? 1 : 0);
                int _max_17 = ((n_tok_1 + (int)(unsigned int)__popc(heads_0 & lanes_below) + head[0] - 1) > (0) ? (n_tok_1 + (int)(unsigned int)__popc(heads_0 & lanes_below) + head[0] - 1) : (0));
                int _min_16 = ((_max_17) < (127) ? (_max_17) : (127));
                col[0] = _min_16;
                n_tok_1 = n_tok_1 + (int)(unsigned int)__popc(heads_0);
                int _shfl_sync_18 = __shfl_sync(0xFFFFFFFF, row[0], 31);
                prev_last = _shfl_sync_18;
                int up_1 = __shfl_up_sync(0xFFFFFFFF, row[1], 1, 32);
                int prev_1 = ((lane == 0) ? prev_last : up_1);
                unsigned int heads_1 = __ballot_sync(0xFFFFFFFF, row[1] != prev_1);
                head[1] = ((row[1] != prev_1) ? 1 : 0);
                int _max_20 = ((n_tok_1 + (int)(unsigned int)__popc(heads_1 & lanes_below) + head[1] - 1) > (0) ? (n_tok_1 + (int)(unsigned int)__popc(heads_1 & lanes_below) + head[1] - 1) : (0));
                int _min_19 = ((_max_20) < (127) ? (_max_20) : (127));
                col[1] = _min_19;
                n_tok_1 = n_tok_1 + (int)(unsigned int)__popc(heads_1);
                int _shfl_sync_21 = __shfl_sync(0xFFFFFFFF, row[1], 31);
                prev_last = _shfl_sync_21;
                int up_2 = __shfl_up_sync(0xFFFFFFFF, row[2], 1, 32);
                int prev_2 = ((lane == 0) ? prev_last : up_2);
                unsigned int heads_2 = __ballot_sync(0xFFFFFFFF, row[2] != prev_2);
                head[2] = ((row[2] != prev_2) ? 1 : 0);
                int _max_23 = ((n_tok_1 + (int)(unsigned int)__popc(heads_2 & lanes_below) + head[2] - 1) > (0) ? (n_tok_1 + (int)(unsigned int)__popc(heads_2 & lanes_below) + head[2] - 1) : (0));
                int _min_22 = ((_max_23) < (127) ? (_max_23) : (127));
                col[2] = _min_22;
                n_tok_1 = n_tok_1 + (int)(unsigned int)__popc(heads_2);
                int _shfl_sync_24 = __shfl_sync(0xFFFFFFFF, row[2], 31);
                prev_last = _shfl_sync_24;
                int up_3 = __shfl_up_sync(0xFFFFFFFF, row[3], 1, 32);
                int prev_3 = ((lane == 0) ? prev_last : up_3);
                unsigned int heads_3 = __ballot_sync(0xFFFFFFFF, row[3] != prev_3);
                head[3] = ((row[3] != prev_3) ? 1 : 0);
                int _max_26 = ((n_tok_1 + (int)(unsigned int)__popc(heads_3 & lanes_below) + head[3] - 1) > (0) ? (n_tok_1 + (int)(unsigned int)__popc(heads_3 & lanes_below) + head[3] - 1) : (0));
                int _min_25 = ((_max_26) < (127) ? (_max_26) : (127));
                col[3] = _min_25;
                n_tok_1 = n_tok_1 + (int)(unsigned int)__popc(heads_3);
                int _shfl_sync_27 = __shfl_sync(0xFFFFFFFF, row[3], 31);
                prev_last = _shfl_sync_27;
                mbarrier_wait(meta_empty_addr + (ms_1) * 8, _phase_meta_empty);
                meta_ids[ms_1 * 128 + lane] = vid[0];
                meta_col[ms_1 * 128 + lane] = col[0];
                if (head[0] != 0) {
                    meta_tok[ms_1 * 128 + col[0]] = row[0];
                }
                if (lane >= n_tok_1) {
                    meta_tok[ms_1 * 128 + lane] = prev_last;
                }
                meta_ids[ms_1 * 128 + (32 + lane)] = vid[1];
                meta_col[ms_1 * 128 + (32 + lane)] = col[1];
                if (head[1] != 0) {
                    meta_tok[ms_1 * 128 + col[1]] = row[1];
                }
                if (32 + lane >= n_tok_1) {
                    meta_tok[ms_1 * 128 + (32 + lane)] = prev_last;
                }
                meta_ids[ms_1 * 128 + (64 + lane)] = vid[2];
                meta_col[ms_1 * 128 + (64 + lane)] = col[2];
                if (head[2] != 0) {
                    meta_tok[ms_1 * 128 + col[2]] = row[2];
                }
                if (64 + lane >= n_tok_1) {
                    meta_tok[ms_1 * 128 + (64 + lane)] = prev_last;
                }
                meta_ids[ms_1 * 128 + (96 + lane)] = vid[3];
                meta_col[ms_1 * 128 + (96 + lane)] = col[3];
                if (head[3] != 0) {
                    meta_tok[ms_1 * 128 + col[3]] = row[3];
                }
                if (96 + lane >= n_tok_1) {
                    meta_tok[ms_1 * 128 + (96 + lane)] = prev_last;
                }
                if (lane == 0) {
                    meta_n[ms_1] = n_tok_1;
                }
                mbarrier_arrive(meta_full_addr + (ms_1) * 8);
                ms_1 += 1;
                if (ms_1 == 4) { ms_1 = 0; _phase_meta_empty ^= 1; }
            }
        }
    }
    // ---- Role: epilogue ----
    if (warp >= 6 && warp <= 9) {
        { // epilogue_main
            unsigned int acc_1 = 0;
            int ms_2 = 0;
            const int quarter = warp % 4;
            const int lane_row = quarter * 32 + lane;
            int n_tiles_3 = (M + 127) / 128;
            unsigned int _phase_meta_full_1 = 0;
            unsigned int _phase_acc_full = 0;
            #pragma unroll 1
            for (int tile_3 = bid; tile_3 < n_tiles_3; tile_3 += num_bids) {
                mbarrier_wait(meta_full_addr + (ms_2) * 8, _phase_meta_full_1);
                int own_col = meta_col[ms_2 * 128 + lane_row];
                mbarrier_arrive(meta_empty_addr + (ms_2) * 8);
                ms_2 += 1;
                if (ms_2 == 4) { ms_2 = 0; _phase_meta_full_1 ^= 1; }
                mbarrier_wait(acc_full_addr + (acc_1) * 8, _phase_acc_full);
                asm volatile("tcgen05.fence::after_thread_sync;");
                float zv = 0.0f;
                float tm_0[32];
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(tm_0[0]), "=f"(tm_0[1]), "=f"(tm_0[2]), "=f"(tm_0[3]), "=f"(tm_0[4]), "=f"(tm_0[5]), "=f"(tm_0[6]), "=f"(tm_0[7]), "=f"(tm_0[8]), "=f"(tm_0[9]), "=f"(tm_0[10]), "=f"(tm_0[11]), "=f"(tm_0[12]), "=f"(tm_0[13]), "=f"(tm_0[14]), "=f"(tm_0[15]), "=f"(tm_0[16]), "=f"(tm_0[17]), "=f"(tm_0[18]), "=f"(tm_0[19]), "=f"(tm_0[20]), "=f"(tm_0[21]), "=f"(tm_0[22]), "=f"(tm_0[23]), "=f"(tm_0[24]), "=f"(tm_0[25]), "=f"(tm_0[26]), "=f"(tm_0[27]), "=f"(tm_0[28]), "=f"(tm_0[29]), "=f"(tm_0[30]), "=f"(tm_0[31])
                    : "r"(taddr + (unsigned int)(quarter * 32 << 16) + acc_1 * 128 + 0));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                zv = ((own_col == 0) ? tm_0[0] : zv);
                zv = ((own_col == 1) ? tm_0[1] : zv);
                zv = ((own_col == 2) ? tm_0[2] : zv);
                zv = ((own_col == 3) ? tm_0[3] : zv);
                zv = ((own_col == 4) ? tm_0[4] : zv);
                zv = ((own_col == 5) ? tm_0[5] : zv);
                zv = ((own_col == 6) ? tm_0[6] : zv);
                zv = ((own_col == 7) ? tm_0[7] : zv);
                zv = ((own_col == 8) ? tm_0[8] : zv);
                zv = ((own_col == 9) ? tm_0[9] : zv);
                zv = ((own_col == 10) ? tm_0[10] : zv);
                zv = ((own_col == 11) ? tm_0[11] : zv);
                zv = ((own_col == 12) ? tm_0[12] : zv);
                zv = ((own_col == 13) ? tm_0[13] : zv);
                zv = ((own_col == 14) ? tm_0[14] : zv);
                zv = ((own_col == 15) ? tm_0[15] : zv);
                zv = ((own_col == 16) ? tm_0[16] : zv);
                zv = ((own_col == 17) ? tm_0[17] : zv);
                zv = ((own_col == 18) ? tm_0[18] : zv);
                zv = ((own_col == 19) ? tm_0[19] : zv);
                zv = ((own_col == 20) ? tm_0[20] : zv);
                zv = ((own_col == 21) ? tm_0[21] : zv);
                zv = ((own_col == 22) ? tm_0[22] : zv);
                zv = ((own_col == 23) ? tm_0[23] : zv);
                zv = ((own_col == 24) ? tm_0[24] : zv);
                zv = ((own_col == 25) ? tm_0[25] : zv);
                zv = ((own_col == 26) ? tm_0[26] : zv);
                zv = ((own_col == 27) ? tm_0[27] : zv);
                zv = ((own_col == 28) ? tm_0[28] : zv);
                zv = ((own_col == 29) ? tm_0[29] : zv);
                zv = ((own_col == 30) ? tm_0[30] : zv);
                zv = ((own_col == 31) ? tm_0[31] : zv);
                float tm_1[32];
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(tm_1[0]), "=f"(tm_1[1]), "=f"(tm_1[2]), "=f"(tm_1[3]), "=f"(tm_1[4]), "=f"(tm_1[5]), "=f"(tm_1[6]), "=f"(tm_1[7]), "=f"(tm_1[8]), "=f"(tm_1[9]), "=f"(tm_1[10]), "=f"(tm_1[11]), "=f"(tm_1[12]), "=f"(tm_1[13]), "=f"(tm_1[14]), "=f"(tm_1[15]), "=f"(tm_1[16]), "=f"(tm_1[17]), "=f"(tm_1[18]), "=f"(tm_1[19]), "=f"(tm_1[20]), "=f"(tm_1[21]), "=f"(tm_1[22]), "=f"(tm_1[23]), "=f"(tm_1[24]), "=f"(tm_1[25]), "=f"(tm_1[26]), "=f"(tm_1[27]), "=f"(tm_1[28]), "=f"(tm_1[29]), "=f"(tm_1[30]), "=f"(tm_1[31])
                    : "r"(taddr + (unsigned int)(quarter * 32 << 16) + acc_1 * 128 + 32));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                zv = ((own_col == 32) ? tm_1[0] : zv);
                zv = ((own_col == 33) ? tm_1[1] : zv);
                zv = ((own_col == 34) ? tm_1[2] : zv);
                zv = ((own_col == 35) ? tm_1[3] : zv);
                zv = ((own_col == 36) ? tm_1[4] : zv);
                zv = ((own_col == 37) ? tm_1[5] : zv);
                zv = ((own_col == 38) ? tm_1[6] : zv);
                zv = ((own_col == 39) ? tm_1[7] : zv);
                zv = ((own_col == 40) ? tm_1[8] : zv);
                zv = ((own_col == 41) ? tm_1[9] : zv);
                zv = ((own_col == 42) ? tm_1[10] : zv);
                zv = ((own_col == 43) ? tm_1[11] : zv);
                zv = ((own_col == 44) ? tm_1[12] : zv);
                zv = ((own_col == 45) ? tm_1[13] : zv);
                zv = ((own_col == 46) ? tm_1[14] : zv);
                zv = ((own_col == 47) ? tm_1[15] : zv);
                zv = ((own_col == 48) ? tm_1[16] : zv);
                zv = ((own_col == 49) ? tm_1[17] : zv);
                zv = ((own_col == 50) ? tm_1[18] : zv);
                zv = ((own_col == 51) ? tm_1[19] : zv);
                zv = ((own_col == 52) ? tm_1[20] : zv);
                zv = ((own_col == 53) ? tm_1[21] : zv);
                zv = ((own_col == 54) ? tm_1[22] : zv);
                zv = ((own_col == 55) ? tm_1[23] : zv);
                zv = ((own_col == 56) ? tm_1[24] : zv);
                zv = ((own_col == 57) ? tm_1[25] : zv);
                zv = ((own_col == 58) ? tm_1[26] : zv);
                zv = ((own_col == 59) ? tm_1[27] : zv);
                zv = ((own_col == 60) ? tm_1[28] : zv);
                zv = ((own_col == 61) ? tm_1[29] : zv);
                zv = ((own_col == 62) ? tm_1[30] : zv);
                zv = ((own_col == 63) ? tm_1[31] : zv);
                float tm_2[32];
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(tm_2[0]), "=f"(tm_2[1]), "=f"(tm_2[2]), "=f"(tm_2[3]), "=f"(tm_2[4]), "=f"(tm_2[5]), "=f"(tm_2[6]), "=f"(tm_2[7]), "=f"(tm_2[8]), "=f"(tm_2[9]), "=f"(tm_2[10]), "=f"(tm_2[11]), "=f"(tm_2[12]), "=f"(tm_2[13]), "=f"(tm_2[14]), "=f"(tm_2[15]), "=f"(tm_2[16]), "=f"(tm_2[17]), "=f"(tm_2[18]), "=f"(tm_2[19]), "=f"(tm_2[20]), "=f"(tm_2[21]), "=f"(tm_2[22]), "=f"(tm_2[23]), "=f"(tm_2[24]), "=f"(tm_2[25]), "=f"(tm_2[26]), "=f"(tm_2[27]), "=f"(tm_2[28]), "=f"(tm_2[29]), "=f"(tm_2[30]), "=f"(tm_2[31])
                    : "r"(taddr + (unsigned int)(quarter * 32 << 16) + acc_1 * 128 + 64));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                zv = ((own_col == 64) ? tm_2[0] : zv);
                zv = ((own_col == 65) ? tm_2[1] : zv);
                zv = ((own_col == 66) ? tm_2[2] : zv);
                zv = ((own_col == 67) ? tm_2[3] : zv);
                zv = ((own_col == 68) ? tm_2[4] : zv);
                zv = ((own_col == 69) ? tm_2[5] : zv);
                zv = ((own_col == 70) ? tm_2[6] : zv);
                zv = ((own_col == 71) ? tm_2[7] : zv);
                zv = ((own_col == 72) ? tm_2[8] : zv);
                zv = ((own_col == 73) ? tm_2[9] : zv);
                zv = ((own_col == 74) ? tm_2[10] : zv);
                zv = ((own_col == 75) ? tm_2[11] : zv);
                zv = ((own_col == 76) ? tm_2[12] : zv);
                zv = ((own_col == 77) ? tm_2[13] : zv);
                zv = ((own_col == 78) ? tm_2[14] : zv);
                zv = ((own_col == 79) ? tm_2[15] : zv);
                zv = ((own_col == 80) ? tm_2[16] : zv);
                zv = ((own_col == 81) ? tm_2[17] : zv);
                zv = ((own_col == 82) ? tm_2[18] : zv);
                zv = ((own_col == 83) ? tm_2[19] : zv);
                zv = ((own_col == 84) ? tm_2[20] : zv);
                zv = ((own_col == 85) ? tm_2[21] : zv);
                zv = ((own_col == 86) ? tm_2[22] : zv);
                zv = ((own_col == 87) ? tm_2[23] : zv);
                zv = ((own_col == 88) ? tm_2[24] : zv);
                zv = ((own_col == 89) ? tm_2[25] : zv);
                zv = ((own_col == 90) ? tm_2[26] : zv);
                zv = ((own_col == 91) ? tm_2[27] : zv);
                zv = ((own_col == 92) ? tm_2[28] : zv);
                zv = ((own_col == 93) ? tm_2[29] : zv);
                zv = ((own_col == 94) ? tm_2[30] : zv);
                zv = ((own_col == 95) ? tm_2[31] : zv);
                float tm_3[32];
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(tm_3[0]), "=f"(tm_3[1]), "=f"(tm_3[2]), "=f"(tm_3[3]), "=f"(tm_3[4]), "=f"(tm_3[5]), "=f"(tm_3[6]), "=f"(tm_3[7]), "=f"(tm_3[8]), "=f"(tm_3[9]), "=f"(tm_3[10]), "=f"(tm_3[11]), "=f"(tm_3[12]), "=f"(tm_3[13]), "=f"(tm_3[14]), "=f"(tm_3[15]), "=f"(tm_3[16]), "=f"(tm_3[17]), "=f"(tm_3[18]), "=f"(tm_3[19]), "=f"(tm_3[20]), "=f"(tm_3[21]), "=f"(tm_3[22]), "=f"(tm_3[23]), "=f"(tm_3[24]), "=f"(tm_3[25]), "=f"(tm_3[26]), "=f"(tm_3[27]), "=f"(tm_3[28]), "=f"(tm_3[29]), "=f"(tm_3[30]), "=f"(tm_3[31])
                    : "r"(taddr + (unsigned int)(quarter * 32 << 16) + acc_1 * 128 + 96));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                zv = ((own_col == 96) ? tm_3[0] : zv);
                zv = ((own_col == 97) ? tm_3[1] : zv);
                zv = ((own_col == 98) ? tm_3[2] : zv);
                zv = ((own_col == 99) ? tm_3[3] : zv);
                zv = ((own_col == 100) ? tm_3[4] : zv);
                zv = ((own_col == 101) ? tm_3[5] : zv);
                zv = ((own_col == 102) ? tm_3[6] : zv);
                zv = ((own_col == 103) ? tm_3[7] : zv);
                zv = ((own_col == 104) ? tm_3[8] : zv);
                zv = ((own_col == 105) ? tm_3[9] : zv);
                zv = ((own_col == 106) ? tm_3[10] : zv);
                zv = ((own_col == 107) ? tm_3[11] : zv);
                zv = ((own_col == 108) ? tm_3[12] : zv);
                zv = ((own_col == 109) ? tm_3[13] : zv);
                zv = ((own_col == 110) ? tm_3[14] : zv);
                zv = ((own_col == 111) ? tm_3[15] : zv);
                zv = ((own_col == 112) ? tm_3[16] : zv);
                zv = ((own_col == 113) ? tm_3[17] : zv);
                zv = ((own_col == 114) ? tm_3[18] : zv);
                zv = ((own_col == 115) ? tm_3[19] : zv);
                zv = ((own_col == 116) ? tm_3[20] : zv);
                zv = ((own_col == 117) ? tm_3[21] : zv);
                zv = ((own_col == 118) ? tm_3[22] : zv);
                zv = ((own_col == 119) ? tm_3[23] : zv);
                zv = ((own_col == 120) ? tm_3[24] : zv);
                zv = ((own_col == 121) ? tm_3[25] : zv);
                zv = ((own_col == 122) ? tm_3[26] : zv);
                zv = ((own_col == 123) ? tm_3[27] : zv);
                zv = ((own_col == 124) ? tm_3[28] : zv);
                zv = ((own_col == 125) ? tm_3[29] : zv);
                zv = ((own_col == 126) ? tm_3[30] : zv);
                zv = ((own_col == 127) ? tm_3[31] : zv);
                asm volatile("tcgen05.fence::before_thread_sync;");
                __syncwarp();
                if (elect_sync()) {
                    mbarrier_arrive(acc_empty_addr + (acc_1) * 8);
                }
                acc_1 += 1;
                if (acc_1 == 2) { acc_1 = 0; _phase_acc_full ^= 1; }
                long long pair = (long long)tile_3 * 128 + (long long)lane_row;
                if (pair < (long long)M) {
                    z[pair] = (__nv_bfloat16)zv;
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
