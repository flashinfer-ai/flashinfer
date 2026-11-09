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










__device__ __forceinline__ void tcgen05_commit_cg2_multicast(int mbar_addr, uint16_t cta_mask) {
    asm volatile(
        "{\n\t"
        ".reg .b16 lo, hi;\n\t"
        "mov.b32 {lo, hi}, %1;\n\t"
        "tcgen05.commit.cta_group::2.mbarrier::arrive::one"
        ".shared::cluster.multicast::cluster.b64 [%0], lo;\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"((uint32_t)cta_mask) : "memory");
}

extern "C" {

__global__ __launch_bounds__(192, LAUNCH_MIN_BLOCKS) __cluster_dims__(2,1,1) void
kernel_cake_gqa_train_f3ca6c2da5e891a0c2e0(const __grid_constant__ CUtensorMap dsw, const __grid_constant__ CUtensorMap K, int* __restrict__ tile_base, int* __restrict__ pair_base, int* __restrict__ pair_order, int* __restrict__ cu_seqlens_q, int* __restrict__ cu_seqlens_k, float* __restrict__ dq_accum, int num_segments, int group, int rows_pad, int n0, int n1, int h0, int hc, int tile_stride, int zero_tile, int accumulate)
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
    #define mma_done_addr (mbar_base + 24)
    #define acc_full_addr (mbar_base + 48)
    #define acc_empty_addr (mbar_base + 64)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    const unsigned int clusters_x = gridDim.x / 2;
    const unsigned int cluster_id = ((blockIdx.z * gridDim.y + blockIdx.y) * clusters_x) + blockIdx.x / 2;
    const unsigned int num_clusters = clusters_x * gridDim.y * gridDim.z;

    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));

    // Prologue
    __nv_bfloat16* sa = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int sa_addr = smem + 1024;
    __nv_bfloat16* sb = reinterpret_cast<__nv_bfloat16*>(smem_raw + 33792);
    const int sb_addr = smem + 33792;

    // Kernel setup ops
    if (warp == 5) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&dsw))) : "memory"); }
    if (warp == 5) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&K))) : "memory"); }

    // Mbarrier init (10 barriers at smem_raw[0..80))

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'ab_pipe' ---
            // tma_full: 3 barriers, init_count=2
            mbarrier_init(smem + 0, 2);
            mbarrier_init(smem + 8, 2);
            mbarrier_init(smem + 16, 2);
            // mma_done: 3 barriers, init_count=1
            mbarrier_init(smem + 24, 1);
            mbarrier_init(smem + 32, 1);
            mbarrier_init(smem + 40, 1);
            // --- pipeline 'acc_pipe' ---
            // acc_full: 2 barriers, init_count=1
            mbarrier_init(smem + 48, 1);
            mbarrier_init(smem + 56, 1);
            // acc_empty: 2 barriers, init_count=8
            mbarrier_init(smem + 64, 8);
            mbarrier_init(smem + 72, 8);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();
    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");

    // TMEM alloc (512 columns, 512 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 80);
    if (warp == 4) {
        int _tmem_hold = smem + 80;
        asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(512) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const unsigned int taddr = tmem_addr_storage[0];
    const unsigned int tmem_acc = taddr;

    // ---- Role: epilogue ----
    if (warp <= 3) {
        { // epilogue_main
            int items = pair_base[num_segments] * hc * (group >> 1);
            int rounds = (items + (int)num_clusters - 1) / (int)num_clusters;
            const int quarter = warp % 4;
            const int r_row = quarter * 32 + lane;
            const unsigned int lane_base = (unsigned int)(quarter * 32 << 16);
            unsigned int ec = 0;
            unsigned int _phase_acc_full = 0;
            #pragma unroll 1
            for (int r = 0; r < rounds; r++) {
                int cid = (int)cluster_id;
                int ncl = (int)num_clusters;
                int slot = (((r & 1) == 0) ? cid : ncl - 1 - cid);
                int pos = r * ncl + slot;
                if (pos < items) {
                    int n_groups = group >> 1;
                    int heads = hc * n_groups;
                    int pair = pair_order[pos / heads];
                    int hg = pos % heads;
                    int hl = hg / n_groups;
                    int g = hg % n_groups;
                    int lo = 0;
                    int hi = num_segments;
                    #pragma unroll 1
                    for (int _ = 0; _ < 31; _++) {
                        if (hi - lo > 1) {
                            int mid = lo + hi >> 1;
                            if (pair_base[mid] <= pair) {
                                lo = mid;
                            } else {
                                hi = mid;
                            }
                        }
                    }
                    int seg = lo;
                    int rank = cta_rank & 1;
                    int q0 = cu_seqlens_q[seg];
                    int q_len = cu_seqlens_q[seg + 1] - q0;
                    int k0 = cu_seqlens_k[seg];
                    int k_len = cu_seqlens_k[seg + 1] - k0;
                    int off = k_len - q_len;
                    int m_tiles = q_len + 63 >> 6;
                    int n_blocks = k_len + 127 >> 7;
                    int m_pair = (pair - pair_base[seg]) * 2;
                    int my_m = m_pair + rank;
                    int last_m = ((m_pair + 1 < m_tiles) ? m_pair + 1 : m_tiles - 1);
                    int n_end = 64 * (last_m + 1) + off + 127 >> 7;
                    if (n_end > n_blocks) {
                        n_end = n_blocks;
                    }
                    if (n_end > n1) {
                        n_end = n1;
                    }
                    int steps = n_end - n0;
                    if (steps < 0) {
                        steps = 0;
                    }
                    int mine = 64 * (my_m + 1) + off + 127 >> 7;
                    if (mine > n_blocks) {
                        mine = n_blocks;
                    }
                    if (my_m >= m_tiles) {
                        mine = 0;
                    }
                    int spans = 1;
                    if (steps > 0) {
                        spans = (n0 + steps - 1) / 64 - n0 / 64 + 1;
                    }
                    int h = h0 + hl;
                    int row_off = q0 * 2 + seg * 128 >> 7 << 7;
                    long long head_rows = (long long)((h * n_groups + g) * rows_pad + row_off);
                    long long tile_off = (head_rows + (long long)(my_m * 128)) * 256ll + (long long)((r_row >> 6) * 16384 + (r_row & 63) * 4);
                    #pragma unroll 1
                    for (int j = 0; j < spans; j++) {
                        int add = accumulate;
                        if (j > 0) {
                            add = 1;
                        }
                        mbarrier_wait(acc_full_addr + (ec) * 8, _phase_acc_full);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        if (my_m < m_tiles) {
                            if (steps > 0 || accumulate == 0) {
                                float acc_0[32];
                                if (steps > 0) {
                                    asm volatile(
                                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                                        : "=f"(acc_0[0]), "=f"(acc_0[1]), "=f"(acc_0[2]), "=f"(acc_0[3]), "=f"(acc_0[4]), "=f"(acc_0[5]), "=f"(acc_0[6]), "=f"(acc_0[7]), "=f"(acc_0[8]), "=f"(acc_0[9]), "=f"(acc_0[10]), "=f"(acc_0[11]), "=f"(acc_0[12]), "=f"(acc_0[13]), "=f"(acc_0[14]), "=f"(acc_0[15]), "=f"(acc_0[16]), "=f"(acc_0[17]), "=f"(acc_0[18]), "=f"(acc_0[19]), "=f"(acc_0[20]), "=f"(acc_0[21]), "=f"(acc_0[22]), "=f"(acc_0[23]), "=f"(acc_0[24]), "=f"(acc_0[25]), "=f"(acc_0[26]), "=f"(acc_0[27]), "=f"(acc_0[28]), "=f"(acc_0[29]), "=f"(acc_0[30]), "=f"(acc_0[31])
                                        : "r"(taddr + (ec * 256 + 0) + lane_base));
                                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                                } else {
                                    #pragma unroll
                                    for (int i_0 = 0; i_0 < 32; i_0++) {
                                        acc_0[i_0] = 0.0f;
                                    }
                                }
                                long long at_0 = tile_off + 0ll;
                                if (add != 0) {
                                    float old_0[4];
                                    {
                                        unsigned _v4_0_0;
                                        unsigned _v4_0_1;
                                        unsigned _v4_0_2;
                                        unsigned _v4_0_3;
                                        asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                            : "=r"(_v4_0_0), "=r"(_v4_0_1), "=r"(_v4_0_2), "=r"(_v4_0_3) : "l"((const void*)(dq_accum + at_0 + (0))), "l"(0x14F0000000000000ULL) : "memory");
                                        old_0[0 + 0] = __uint_as_float(_v4_0_0);
                                        old_0[0 + 1] = __uint_as_float(_v4_0_1);
                                        old_0[0 + 2] = __uint_as_float(_v4_0_2);
                                        old_0[0 + 3] = __uint_as_float(_v4_0_3);
                                    }
                                    acc_0[0] = old_0[0] + acc_0[0];
                                    acc_0[1] = old_0[1] + acc_0[1];
                                    acc_0[2] = old_0[2] + acc_0[2];
                                    acc_0[3] = old_0[3] + acc_0[3];
                                }
                                asm volatile("st.global.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(dq_accum + at_0), "f"(acc_0[(0) + 0]), "f"(acc_0[(0) + 1]), "f"(acc_0[(0) + 2]), "f"(acc_0[(0) + 3]), "l"(0x14F0000000000000ULL) : "memory");
                                long long at_1 = tile_off + 512ll;
                                if (add != 0) {
                                    float old_1[4];
                                    {
                                        unsigned _v4_1_0;
                                        unsigned _v4_1_1;
                                        unsigned _v4_1_2;
                                        unsigned _v4_1_3;
                                        asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                            : "=r"(_v4_1_0), "=r"(_v4_1_1), "=r"(_v4_1_2), "=r"(_v4_1_3) : "l"((const void*)(dq_accum + at_1 + (0))), "l"(0x14F0000000000000ULL) : "memory");
                                        old_1[0 + 0] = __uint_as_float(_v4_1_0);
                                        old_1[0 + 1] = __uint_as_float(_v4_1_1);
                                        old_1[0 + 2] = __uint_as_float(_v4_1_2);
                                        old_1[0 + 3] = __uint_as_float(_v4_1_3);
                                    }
                                    acc_0[4] = old_1[0] + acc_0[4];
                                    acc_0[5] = old_1[1] + acc_0[5];
                                    acc_0[6] = old_1[2] + acc_0[6];
                                    acc_0[7] = old_1[3] + acc_0[7];
                                }
                                asm volatile("st.global.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(dq_accum + at_1), "f"(acc_0[(4) + 0]), "f"(acc_0[(4) + 1]), "f"(acc_0[(4) + 2]), "f"(acc_0[(4) + 3]), "l"(0x14F0000000000000ULL) : "memory");
                                long long at_2 = tile_off + 1024ll;
                                if (add != 0) {
                                    float old_2[4];
                                    {
                                        unsigned _v4_2_0;
                                        unsigned _v4_2_1;
                                        unsigned _v4_2_2;
                                        unsigned _v4_2_3;
                                        asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                            : "=r"(_v4_2_0), "=r"(_v4_2_1), "=r"(_v4_2_2), "=r"(_v4_2_3) : "l"((const void*)(dq_accum + at_2 + (0))), "l"(0x14F0000000000000ULL) : "memory");
                                        old_2[0 + 0] = __uint_as_float(_v4_2_0);
                                        old_2[0 + 1] = __uint_as_float(_v4_2_1);
                                        old_2[0 + 2] = __uint_as_float(_v4_2_2);
                                        old_2[0 + 3] = __uint_as_float(_v4_2_3);
                                    }
                                    acc_0[8] = old_2[0] + acc_0[8];
                                    acc_0[9] = old_2[1] + acc_0[9];
                                    acc_0[10] = old_2[2] + acc_0[10];
                                    acc_0[11] = old_2[3] + acc_0[11];
                                }
                                asm volatile("st.global.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(dq_accum + at_2), "f"(acc_0[(8) + 0]), "f"(acc_0[(8) + 1]), "f"(acc_0[(8) + 2]), "f"(acc_0[(8) + 3]), "l"(0x14F0000000000000ULL) : "memory");
                                long long at_3 = tile_off + 1536ll;
                                if (add != 0) {
                                    float old_3[4];
                                    {
                                        unsigned _v4_3_0;
                                        unsigned _v4_3_1;
                                        unsigned _v4_3_2;
                                        unsigned _v4_3_3;
                                        asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                            : "=r"(_v4_3_0), "=r"(_v4_3_1), "=r"(_v4_3_2), "=r"(_v4_3_3) : "l"((const void*)(dq_accum + at_3 + (0))), "l"(0x14F0000000000000ULL) : "memory");
                                        old_3[0 + 0] = __uint_as_float(_v4_3_0);
                                        old_3[0 + 1] = __uint_as_float(_v4_3_1);
                                        old_3[0 + 2] = __uint_as_float(_v4_3_2);
                                        old_3[0 + 3] = __uint_as_float(_v4_3_3);
                                    }
                                    acc_0[12] = old_3[0] + acc_0[12];
                                    acc_0[13] = old_3[1] + acc_0[13];
                                    acc_0[14] = old_3[2] + acc_0[14];
                                    acc_0[15] = old_3[3] + acc_0[15];
                                }
                                asm volatile("st.global.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(dq_accum + at_3), "f"(acc_0[(12) + 0]), "f"(acc_0[(12) + 1]), "f"(acc_0[(12) + 2]), "f"(acc_0[(12) + 3]), "l"(0x14F0000000000000ULL) : "memory");
                                long long at_4 = tile_off + 2048ll;
                                if (add != 0) {
                                    float old_4[4];
                                    {
                                        unsigned _v4_4_0;
                                        unsigned _v4_4_1;
                                        unsigned _v4_4_2;
                                        unsigned _v4_4_3;
                                        asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                            : "=r"(_v4_4_0), "=r"(_v4_4_1), "=r"(_v4_4_2), "=r"(_v4_4_3) : "l"((const void*)(dq_accum + at_4 + (0))), "l"(0x14F0000000000000ULL) : "memory");
                                        old_4[0 + 0] = __uint_as_float(_v4_4_0);
                                        old_4[0 + 1] = __uint_as_float(_v4_4_1);
                                        old_4[0 + 2] = __uint_as_float(_v4_4_2);
                                        old_4[0 + 3] = __uint_as_float(_v4_4_3);
                                    }
                                    acc_0[16] = old_4[0] + acc_0[16];
                                    acc_0[17] = old_4[1] + acc_0[17];
                                    acc_0[18] = old_4[2] + acc_0[18];
                                    acc_0[19] = old_4[3] + acc_0[19];
                                }
                                asm volatile("st.global.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(dq_accum + at_4), "f"(acc_0[(16) + 0]), "f"(acc_0[(16) + 1]), "f"(acc_0[(16) + 2]), "f"(acc_0[(16) + 3]), "l"(0x14F0000000000000ULL) : "memory");
                                long long at_5 = tile_off + 2560ll;
                                if (add != 0) {
                                    float old_5[4];
                                    {
                                        unsigned _v4_5_0;
                                        unsigned _v4_5_1;
                                        unsigned _v4_5_2;
                                        unsigned _v4_5_3;
                                        asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                            : "=r"(_v4_5_0), "=r"(_v4_5_1), "=r"(_v4_5_2), "=r"(_v4_5_3) : "l"((const void*)(dq_accum + at_5 + (0))), "l"(0x14F0000000000000ULL) : "memory");
                                        old_5[0 + 0] = __uint_as_float(_v4_5_0);
                                        old_5[0 + 1] = __uint_as_float(_v4_5_1);
                                        old_5[0 + 2] = __uint_as_float(_v4_5_2);
                                        old_5[0 + 3] = __uint_as_float(_v4_5_3);
                                    }
                                    acc_0[20] = old_5[0] + acc_0[20];
                                    acc_0[21] = old_5[1] + acc_0[21];
                                    acc_0[22] = old_5[2] + acc_0[22];
                                    acc_0[23] = old_5[3] + acc_0[23];
                                }
                                asm volatile("st.global.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(dq_accum + at_5), "f"(acc_0[(20) + 0]), "f"(acc_0[(20) + 1]), "f"(acc_0[(20) + 2]), "f"(acc_0[(20) + 3]), "l"(0x14F0000000000000ULL) : "memory");
                                long long at_6 = tile_off + 3072ll;
                                if (add != 0) {
                                    float old_6[4];
                                    {
                                        unsigned _v4_6_0;
                                        unsigned _v4_6_1;
                                        unsigned _v4_6_2;
                                        unsigned _v4_6_3;
                                        asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                            : "=r"(_v4_6_0), "=r"(_v4_6_1), "=r"(_v4_6_2), "=r"(_v4_6_3) : "l"((const void*)(dq_accum + at_6 + (0))), "l"(0x14F0000000000000ULL) : "memory");
                                        old_6[0 + 0] = __uint_as_float(_v4_6_0);
                                        old_6[0 + 1] = __uint_as_float(_v4_6_1);
                                        old_6[0 + 2] = __uint_as_float(_v4_6_2);
                                        old_6[0 + 3] = __uint_as_float(_v4_6_3);
                                    }
                                    acc_0[24] = old_6[0] + acc_0[24];
                                    acc_0[25] = old_6[1] + acc_0[25];
                                    acc_0[26] = old_6[2] + acc_0[26];
                                    acc_0[27] = old_6[3] + acc_0[27];
                                }
                                asm volatile("st.global.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(dq_accum + at_6), "f"(acc_0[(24) + 0]), "f"(acc_0[(24) + 1]), "f"(acc_0[(24) + 2]), "f"(acc_0[(24) + 3]), "l"(0x14F0000000000000ULL) : "memory");
                                long long at_7 = tile_off + 3584ll;
                                if (add != 0) {
                                    float old_7[4];
                                    {
                                        unsigned _v4_7_0;
                                        unsigned _v4_7_1;
                                        unsigned _v4_7_2;
                                        unsigned _v4_7_3;
                                        asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                            : "=r"(_v4_7_0), "=r"(_v4_7_1), "=r"(_v4_7_2), "=r"(_v4_7_3) : "l"((const void*)(dq_accum + at_7 + (0))), "l"(0x14F0000000000000ULL) : "memory");
                                        old_7[0 + 0] = __uint_as_float(_v4_7_0);
                                        old_7[0 + 1] = __uint_as_float(_v4_7_1);
                                        old_7[0 + 2] = __uint_as_float(_v4_7_2);
                                        old_7[0 + 3] = __uint_as_float(_v4_7_3);
                                    }
                                    acc_0[28] = old_7[0] + acc_0[28];
                                    acc_0[29] = old_7[1] + acc_0[29];
                                    acc_0[30] = old_7[2] + acc_0[30];
                                    acc_0[31] = old_7[3] + acc_0[31];
                                }
                                asm volatile("st.global.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(dq_accum + at_7), "f"(acc_0[(28) + 0]), "f"(acc_0[(28) + 1]), "f"(acc_0[(28) + 2]), "f"(acc_0[(28) + 3]), "l"(0x14F0000000000000ULL) : "memory");
                                float acc_1[32];
                                if (steps > 0) {
                                    asm volatile(
                                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                                        : "=f"(acc_1[0]), "=f"(acc_1[1]), "=f"(acc_1[2]), "=f"(acc_1[3]), "=f"(acc_1[4]), "=f"(acc_1[5]), "=f"(acc_1[6]), "=f"(acc_1[7]), "=f"(acc_1[8]), "=f"(acc_1[9]), "=f"(acc_1[10]), "=f"(acc_1[11]), "=f"(acc_1[12]), "=f"(acc_1[13]), "=f"(acc_1[14]), "=f"(acc_1[15]), "=f"(acc_1[16]), "=f"(acc_1[17]), "=f"(acc_1[18]), "=f"(acc_1[19]), "=f"(acc_1[20]), "=f"(acc_1[21]), "=f"(acc_1[22]), "=f"(acc_1[23]), "=f"(acc_1[24]), "=f"(acc_1[25]), "=f"(acc_1[26]), "=f"(acc_1[27]), "=f"(acc_1[28]), "=f"(acc_1[29]), "=f"(acc_1[30]), "=f"(acc_1[31])
                                        : "r"(taddr + (ec * 256 + 32) + lane_base));
                                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                                } else {
                                    #pragma unroll
                                    for (int i_1 = 0; i_1 < 32; i_1++) {
                                        acc_1[i_1] = 0.0f;
                                    }
                                }
                                long long at_8 = tile_off + 4096ll;
                                if (add != 0) {
                                    float old_8[4];
                                    {
                                        unsigned _v4_8_0;
                                        unsigned _v4_8_1;
                                        unsigned _v4_8_2;
                                        unsigned _v4_8_3;
                                        asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                            : "=r"(_v4_8_0), "=r"(_v4_8_1), "=r"(_v4_8_2), "=r"(_v4_8_3) : "l"((const void*)(dq_accum + at_8 + (0))), "l"(0x14F0000000000000ULL) : "memory");
                                        old_8[0 + 0] = __uint_as_float(_v4_8_0);
                                        old_8[0 + 1] = __uint_as_float(_v4_8_1);
                                        old_8[0 + 2] = __uint_as_float(_v4_8_2);
                                        old_8[0 + 3] = __uint_as_float(_v4_8_3);
                                    }
                                    acc_1[0] = old_8[0] + acc_1[0];
                                    acc_1[1] = old_8[1] + acc_1[1];
                                    acc_1[2] = old_8[2] + acc_1[2];
                                    acc_1[3] = old_8[3] + acc_1[3];
                                }
                                asm volatile("st.global.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(dq_accum + at_8), "f"(acc_1[(0) + 0]), "f"(acc_1[(0) + 1]), "f"(acc_1[(0) + 2]), "f"(acc_1[(0) + 3]), "l"(0x14F0000000000000ULL) : "memory");
                                long long at_9 = tile_off + 4608ll;
                                if (add != 0) {
                                    float old_9[4];
                                    {
                                        unsigned _v4_9_0;
                                        unsigned _v4_9_1;
                                        unsigned _v4_9_2;
                                        unsigned _v4_9_3;
                                        asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                            : "=r"(_v4_9_0), "=r"(_v4_9_1), "=r"(_v4_9_2), "=r"(_v4_9_3) : "l"((const void*)(dq_accum + at_9 + (0))), "l"(0x14F0000000000000ULL) : "memory");
                                        old_9[0 + 0] = __uint_as_float(_v4_9_0);
                                        old_9[0 + 1] = __uint_as_float(_v4_9_1);
                                        old_9[0 + 2] = __uint_as_float(_v4_9_2);
                                        old_9[0 + 3] = __uint_as_float(_v4_9_3);
                                    }
                                    acc_1[4] = old_9[0] + acc_1[4];
                                    acc_1[5] = old_9[1] + acc_1[5];
                                    acc_1[6] = old_9[2] + acc_1[6];
                                    acc_1[7] = old_9[3] + acc_1[7];
                                }
                                asm volatile("st.global.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(dq_accum + at_9), "f"(acc_1[(4) + 0]), "f"(acc_1[(4) + 1]), "f"(acc_1[(4) + 2]), "f"(acc_1[(4) + 3]), "l"(0x14F0000000000000ULL) : "memory");
                                long long at_10 = tile_off + 5120ll;
                                if (add != 0) {
                                    float old_10[4];
                                    {
                                        unsigned _v4_10_0;
                                        unsigned _v4_10_1;
                                        unsigned _v4_10_2;
                                        unsigned _v4_10_3;
                                        asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                            : "=r"(_v4_10_0), "=r"(_v4_10_1), "=r"(_v4_10_2), "=r"(_v4_10_3) : "l"((const void*)(dq_accum + at_10 + (0))), "l"(0x14F0000000000000ULL) : "memory");
                                        old_10[0 + 0] = __uint_as_float(_v4_10_0);
                                        old_10[0 + 1] = __uint_as_float(_v4_10_1);
                                        old_10[0 + 2] = __uint_as_float(_v4_10_2);
                                        old_10[0 + 3] = __uint_as_float(_v4_10_3);
                                    }
                                    acc_1[8] = old_10[0] + acc_1[8];
                                    acc_1[9] = old_10[1] + acc_1[9];
                                    acc_1[10] = old_10[2] + acc_1[10];
                                    acc_1[11] = old_10[3] + acc_1[11];
                                }
                                asm volatile("st.global.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(dq_accum + at_10), "f"(acc_1[(8) + 0]), "f"(acc_1[(8) + 1]), "f"(acc_1[(8) + 2]), "f"(acc_1[(8) + 3]), "l"(0x14F0000000000000ULL) : "memory");
                                long long at_11 = tile_off + 5632ll;
                                if (add != 0) {
                                    float old_11[4];
                                    {
                                        unsigned _v4_11_0;
                                        unsigned _v4_11_1;
                                        unsigned _v4_11_2;
                                        unsigned _v4_11_3;
                                        asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                            : "=r"(_v4_11_0), "=r"(_v4_11_1), "=r"(_v4_11_2), "=r"(_v4_11_3) : "l"((const void*)(dq_accum + at_11 + (0))), "l"(0x14F0000000000000ULL) : "memory");
                                        old_11[0 + 0] = __uint_as_float(_v4_11_0);
                                        old_11[0 + 1] = __uint_as_float(_v4_11_1);
                                        old_11[0 + 2] = __uint_as_float(_v4_11_2);
                                        old_11[0 + 3] = __uint_as_float(_v4_11_3);
                                    }
                                    acc_1[12] = old_11[0] + acc_1[12];
                                    acc_1[13] = old_11[1] + acc_1[13];
                                    acc_1[14] = old_11[2] + acc_1[14];
                                    acc_1[15] = old_11[3] + acc_1[15];
                                }
                                asm volatile("st.global.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(dq_accum + at_11), "f"(acc_1[(12) + 0]), "f"(acc_1[(12) + 1]), "f"(acc_1[(12) + 2]), "f"(acc_1[(12) + 3]), "l"(0x14F0000000000000ULL) : "memory");
                                long long at_12 = tile_off + 6144ll;
                                if (add != 0) {
                                    float old_12[4];
                                    {
                                        unsigned _v4_12_0;
                                        unsigned _v4_12_1;
                                        unsigned _v4_12_2;
                                        unsigned _v4_12_3;
                                        asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                            : "=r"(_v4_12_0), "=r"(_v4_12_1), "=r"(_v4_12_2), "=r"(_v4_12_3) : "l"((const void*)(dq_accum + at_12 + (0))), "l"(0x14F0000000000000ULL) : "memory");
                                        old_12[0 + 0] = __uint_as_float(_v4_12_0);
                                        old_12[0 + 1] = __uint_as_float(_v4_12_1);
                                        old_12[0 + 2] = __uint_as_float(_v4_12_2);
                                        old_12[0 + 3] = __uint_as_float(_v4_12_3);
                                    }
                                    acc_1[16] = old_12[0] + acc_1[16];
                                    acc_1[17] = old_12[1] + acc_1[17];
                                    acc_1[18] = old_12[2] + acc_1[18];
                                    acc_1[19] = old_12[3] + acc_1[19];
                                }
                                asm volatile("st.global.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(dq_accum + at_12), "f"(acc_1[(16) + 0]), "f"(acc_1[(16) + 1]), "f"(acc_1[(16) + 2]), "f"(acc_1[(16) + 3]), "l"(0x14F0000000000000ULL) : "memory");
                                long long at_13 = tile_off + 6656ll;
                                if (add != 0) {
                                    float old_13[4];
                                    {
                                        unsigned _v4_13_0;
                                        unsigned _v4_13_1;
                                        unsigned _v4_13_2;
                                        unsigned _v4_13_3;
                                        asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                            : "=r"(_v4_13_0), "=r"(_v4_13_1), "=r"(_v4_13_2), "=r"(_v4_13_3) : "l"((const void*)(dq_accum + at_13 + (0))), "l"(0x14F0000000000000ULL) : "memory");
                                        old_13[0 + 0] = __uint_as_float(_v4_13_0);
                                        old_13[0 + 1] = __uint_as_float(_v4_13_1);
                                        old_13[0 + 2] = __uint_as_float(_v4_13_2);
                                        old_13[0 + 3] = __uint_as_float(_v4_13_3);
                                    }
                                    acc_1[20] = old_13[0] + acc_1[20];
                                    acc_1[21] = old_13[1] + acc_1[21];
                                    acc_1[22] = old_13[2] + acc_1[22];
                                    acc_1[23] = old_13[3] + acc_1[23];
                                }
                                asm volatile("st.global.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(dq_accum + at_13), "f"(acc_1[(20) + 0]), "f"(acc_1[(20) + 1]), "f"(acc_1[(20) + 2]), "f"(acc_1[(20) + 3]), "l"(0x14F0000000000000ULL) : "memory");
                                long long at_14 = tile_off + 7168ll;
                                if (add != 0) {
                                    float old_14[4];
                                    {
                                        unsigned _v4_14_0;
                                        unsigned _v4_14_1;
                                        unsigned _v4_14_2;
                                        unsigned _v4_14_3;
                                        asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                            : "=r"(_v4_14_0), "=r"(_v4_14_1), "=r"(_v4_14_2), "=r"(_v4_14_3) : "l"((const void*)(dq_accum + at_14 + (0))), "l"(0x14F0000000000000ULL) : "memory");
                                        old_14[0 + 0] = __uint_as_float(_v4_14_0);
                                        old_14[0 + 1] = __uint_as_float(_v4_14_1);
                                        old_14[0 + 2] = __uint_as_float(_v4_14_2);
                                        old_14[0 + 3] = __uint_as_float(_v4_14_3);
                                    }
                                    acc_1[24] = old_14[0] + acc_1[24];
                                    acc_1[25] = old_14[1] + acc_1[25];
                                    acc_1[26] = old_14[2] + acc_1[26];
                                    acc_1[27] = old_14[3] + acc_1[27];
                                }
                                asm volatile("st.global.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(dq_accum + at_14), "f"(acc_1[(24) + 0]), "f"(acc_1[(24) + 1]), "f"(acc_1[(24) + 2]), "f"(acc_1[(24) + 3]), "l"(0x14F0000000000000ULL) : "memory");
                                long long at_15 = tile_off + 7680ll;
                                if (add != 0) {
                                    float old_15[4];
                                    {
                                        unsigned _v4_15_0;
                                        unsigned _v4_15_1;
                                        unsigned _v4_15_2;
                                        unsigned _v4_15_3;
                                        asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                            : "=r"(_v4_15_0), "=r"(_v4_15_1), "=r"(_v4_15_2), "=r"(_v4_15_3) : "l"((const void*)(dq_accum + at_15 + (0))), "l"(0x14F0000000000000ULL) : "memory");
                                        old_15[0 + 0] = __uint_as_float(_v4_15_0);
                                        old_15[0 + 1] = __uint_as_float(_v4_15_1);
                                        old_15[0 + 2] = __uint_as_float(_v4_15_2);
                                        old_15[0 + 3] = __uint_as_float(_v4_15_3);
                                    }
                                    acc_1[28] = old_15[0] + acc_1[28];
                                    acc_1[29] = old_15[1] + acc_1[29];
                                    acc_1[30] = old_15[2] + acc_1[30];
                                    acc_1[31] = old_15[3] + acc_1[31];
                                }
                                asm volatile("st.global.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(dq_accum + at_15), "f"(acc_1[(28) + 0]), "f"(acc_1[(28) + 1]), "f"(acc_1[(28) + 2]), "f"(acc_1[(28) + 3]), "l"(0x14F0000000000000ULL) : "memory");
                                float acc_2[32];
                                if (steps > 0) {
                                    asm volatile(
                                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                                        : "=f"(acc_2[0]), "=f"(acc_2[1]), "=f"(acc_2[2]), "=f"(acc_2[3]), "=f"(acc_2[4]), "=f"(acc_2[5]), "=f"(acc_2[6]), "=f"(acc_2[7]), "=f"(acc_2[8]), "=f"(acc_2[9]), "=f"(acc_2[10]), "=f"(acc_2[11]), "=f"(acc_2[12]), "=f"(acc_2[13]), "=f"(acc_2[14]), "=f"(acc_2[15]), "=f"(acc_2[16]), "=f"(acc_2[17]), "=f"(acc_2[18]), "=f"(acc_2[19]), "=f"(acc_2[20]), "=f"(acc_2[21]), "=f"(acc_2[22]), "=f"(acc_2[23]), "=f"(acc_2[24]), "=f"(acc_2[25]), "=f"(acc_2[26]), "=f"(acc_2[27]), "=f"(acc_2[28]), "=f"(acc_2[29]), "=f"(acc_2[30]), "=f"(acc_2[31])
                                        : "r"(taddr + (ec * 256 + 64) + lane_base));
                                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                                } else {
                                    #pragma unroll
                                    for (int i_2 = 0; i_2 < 32; i_2++) {
                                        acc_2[i_2] = 0.0f;
                                    }
                                }
                                long long at_16 = tile_off + 8192ll;
                                if (add != 0) {
                                    float old_16[4];
                                    {
                                        unsigned _v4_16_0;
                                        unsigned _v4_16_1;
                                        unsigned _v4_16_2;
                                        unsigned _v4_16_3;
                                        asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                            : "=r"(_v4_16_0), "=r"(_v4_16_1), "=r"(_v4_16_2), "=r"(_v4_16_3) : "l"((const void*)(dq_accum + at_16 + (0))), "l"(0x14F0000000000000ULL) : "memory");
                                        old_16[0 + 0] = __uint_as_float(_v4_16_0);
                                        old_16[0 + 1] = __uint_as_float(_v4_16_1);
                                        old_16[0 + 2] = __uint_as_float(_v4_16_2);
                                        old_16[0 + 3] = __uint_as_float(_v4_16_3);
                                    }
                                    acc_2[0] = old_16[0] + acc_2[0];
                                    acc_2[1] = old_16[1] + acc_2[1];
                                    acc_2[2] = old_16[2] + acc_2[2];
                                    acc_2[3] = old_16[3] + acc_2[3];
                                }
                                asm volatile("st.global.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(dq_accum + at_16), "f"(acc_2[(0) + 0]), "f"(acc_2[(0) + 1]), "f"(acc_2[(0) + 2]), "f"(acc_2[(0) + 3]), "l"(0x14F0000000000000ULL) : "memory");
                                long long at_17 = tile_off + 8704ll;
                                if (add != 0) {
                                    float old_17[4];
                                    {
                                        unsigned _v4_17_0;
                                        unsigned _v4_17_1;
                                        unsigned _v4_17_2;
                                        unsigned _v4_17_3;
                                        asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                            : "=r"(_v4_17_0), "=r"(_v4_17_1), "=r"(_v4_17_2), "=r"(_v4_17_3) : "l"((const void*)(dq_accum + at_17 + (0))), "l"(0x14F0000000000000ULL) : "memory");
                                        old_17[0 + 0] = __uint_as_float(_v4_17_0);
                                        old_17[0 + 1] = __uint_as_float(_v4_17_1);
                                        old_17[0 + 2] = __uint_as_float(_v4_17_2);
                                        old_17[0 + 3] = __uint_as_float(_v4_17_3);
                                    }
                                    acc_2[4] = old_17[0] + acc_2[4];
                                    acc_2[5] = old_17[1] + acc_2[5];
                                    acc_2[6] = old_17[2] + acc_2[6];
                                    acc_2[7] = old_17[3] + acc_2[7];
                                }
                                asm volatile("st.global.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(dq_accum + at_17), "f"(acc_2[(4) + 0]), "f"(acc_2[(4) + 1]), "f"(acc_2[(4) + 2]), "f"(acc_2[(4) + 3]), "l"(0x14F0000000000000ULL) : "memory");
                                long long at_18 = tile_off + 9216ll;
                                if (add != 0) {
                                    float old_18[4];
                                    {
                                        unsigned _v4_18_0;
                                        unsigned _v4_18_1;
                                        unsigned _v4_18_2;
                                        unsigned _v4_18_3;
                                        asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                            : "=r"(_v4_18_0), "=r"(_v4_18_1), "=r"(_v4_18_2), "=r"(_v4_18_3) : "l"((const void*)(dq_accum + at_18 + (0))), "l"(0x14F0000000000000ULL) : "memory");
                                        old_18[0 + 0] = __uint_as_float(_v4_18_0);
                                        old_18[0 + 1] = __uint_as_float(_v4_18_1);
                                        old_18[0 + 2] = __uint_as_float(_v4_18_2);
                                        old_18[0 + 3] = __uint_as_float(_v4_18_3);
                                    }
                                    acc_2[8] = old_18[0] + acc_2[8];
                                    acc_2[9] = old_18[1] + acc_2[9];
                                    acc_2[10] = old_18[2] + acc_2[10];
                                    acc_2[11] = old_18[3] + acc_2[11];
                                }
                                asm volatile("st.global.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(dq_accum + at_18), "f"(acc_2[(8) + 0]), "f"(acc_2[(8) + 1]), "f"(acc_2[(8) + 2]), "f"(acc_2[(8) + 3]), "l"(0x14F0000000000000ULL) : "memory");
                                long long at_19 = tile_off + 9728ll;
                                if (add != 0) {
                                    float old_19[4];
                                    {
                                        unsigned _v4_19_0;
                                        unsigned _v4_19_1;
                                        unsigned _v4_19_2;
                                        unsigned _v4_19_3;
                                        asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                            : "=r"(_v4_19_0), "=r"(_v4_19_1), "=r"(_v4_19_2), "=r"(_v4_19_3) : "l"((const void*)(dq_accum + at_19 + (0))), "l"(0x14F0000000000000ULL) : "memory");
                                        old_19[0 + 0] = __uint_as_float(_v4_19_0);
                                        old_19[0 + 1] = __uint_as_float(_v4_19_1);
                                        old_19[0 + 2] = __uint_as_float(_v4_19_2);
                                        old_19[0 + 3] = __uint_as_float(_v4_19_3);
                                    }
                                    acc_2[12] = old_19[0] + acc_2[12];
                                    acc_2[13] = old_19[1] + acc_2[13];
                                    acc_2[14] = old_19[2] + acc_2[14];
                                    acc_2[15] = old_19[3] + acc_2[15];
                                }
                                asm volatile("st.global.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(dq_accum + at_19), "f"(acc_2[(12) + 0]), "f"(acc_2[(12) + 1]), "f"(acc_2[(12) + 2]), "f"(acc_2[(12) + 3]), "l"(0x14F0000000000000ULL) : "memory");
                                long long at_20 = tile_off + 10240ll;
                                if (add != 0) {
                                    float old_20[4];
                                    {
                                        unsigned _v4_20_0;
                                        unsigned _v4_20_1;
                                        unsigned _v4_20_2;
                                        unsigned _v4_20_3;
                                        asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                            : "=r"(_v4_20_0), "=r"(_v4_20_1), "=r"(_v4_20_2), "=r"(_v4_20_3) : "l"((const void*)(dq_accum + at_20 + (0))), "l"(0x14F0000000000000ULL) : "memory");
                                        old_20[0 + 0] = __uint_as_float(_v4_20_0);
                                        old_20[0 + 1] = __uint_as_float(_v4_20_1);
                                        old_20[0 + 2] = __uint_as_float(_v4_20_2);
                                        old_20[0 + 3] = __uint_as_float(_v4_20_3);
                                    }
                                    acc_2[16] = old_20[0] + acc_2[16];
                                    acc_2[17] = old_20[1] + acc_2[17];
                                    acc_2[18] = old_20[2] + acc_2[18];
                                    acc_2[19] = old_20[3] + acc_2[19];
                                }
                                asm volatile("st.global.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(dq_accum + at_20), "f"(acc_2[(16) + 0]), "f"(acc_2[(16) + 1]), "f"(acc_2[(16) + 2]), "f"(acc_2[(16) + 3]), "l"(0x14F0000000000000ULL) : "memory");
                                long long at_21 = tile_off + 10752ll;
                                if (add != 0) {
                                    float old_21[4];
                                    {
                                        unsigned _v4_21_0;
                                        unsigned _v4_21_1;
                                        unsigned _v4_21_2;
                                        unsigned _v4_21_3;
                                        asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                            : "=r"(_v4_21_0), "=r"(_v4_21_1), "=r"(_v4_21_2), "=r"(_v4_21_3) : "l"((const void*)(dq_accum + at_21 + (0))), "l"(0x14F0000000000000ULL) : "memory");
                                        old_21[0 + 0] = __uint_as_float(_v4_21_0);
                                        old_21[0 + 1] = __uint_as_float(_v4_21_1);
                                        old_21[0 + 2] = __uint_as_float(_v4_21_2);
                                        old_21[0 + 3] = __uint_as_float(_v4_21_3);
                                    }
                                    acc_2[20] = old_21[0] + acc_2[20];
                                    acc_2[21] = old_21[1] + acc_2[21];
                                    acc_2[22] = old_21[2] + acc_2[22];
                                    acc_2[23] = old_21[3] + acc_2[23];
                                }
                                asm volatile("st.global.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(dq_accum + at_21), "f"(acc_2[(20) + 0]), "f"(acc_2[(20) + 1]), "f"(acc_2[(20) + 2]), "f"(acc_2[(20) + 3]), "l"(0x14F0000000000000ULL) : "memory");
                                long long at_22 = tile_off + 11264ll;
                                if (add != 0) {
                                    float old_22[4];
                                    {
                                        unsigned _v4_22_0;
                                        unsigned _v4_22_1;
                                        unsigned _v4_22_2;
                                        unsigned _v4_22_3;
                                        asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                            : "=r"(_v4_22_0), "=r"(_v4_22_1), "=r"(_v4_22_2), "=r"(_v4_22_3) : "l"((const void*)(dq_accum + at_22 + (0))), "l"(0x14F0000000000000ULL) : "memory");
                                        old_22[0 + 0] = __uint_as_float(_v4_22_0);
                                        old_22[0 + 1] = __uint_as_float(_v4_22_1);
                                        old_22[0 + 2] = __uint_as_float(_v4_22_2);
                                        old_22[0 + 3] = __uint_as_float(_v4_22_3);
                                    }
                                    acc_2[24] = old_22[0] + acc_2[24];
                                    acc_2[25] = old_22[1] + acc_2[25];
                                    acc_2[26] = old_22[2] + acc_2[26];
                                    acc_2[27] = old_22[3] + acc_2[27];
                                }
                                asm volatile("st.global.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(dq_accum + at_22), "f"(acc_2[(24) + 0]), "f"(acc_2[(24) + 1]), "f"(acc_2[(24) + 2]), "f"(acc_2[(24) + 3]), "l"(0x14F0000000000000ULL) : "memory");
                                long long at_23 = tile_off + 11776ll;
                                if (add != 0) {
                                    float old_23[4];
                                    {
                                        unsigned _v4_23_0;
                                        unsigned _v4_23_1;
                                        unsigned _v4_23_2;
                                        unsigned _v4_23_3;
                                        asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                            : "=r"(_v4_23_0), "=r"(_v4_23_1), "=r"(_v4_23_2), "=r"(_v4_23_3) : "l"((const void*)(dq_accum + at_23 + (0))), "l"(0x14F0000000000000ULL) : "memory");
                                        old_23[0 + 0] = __uint_as_float(_v4_23_0);
                                        old_23[0 + 1] = __uint_as_float(_v4_23_1);
                                        old_23[0 + 2] = __uint_as_float(_v4_23_2);
                                        old_23[0 + 3] = __uint_as_float(_v4_23_3);
                                    }
                                    acc_2[28] = old_23[0] + acc_2[28];
                                    acc_2[29] = old_23[1] + acc_2[29];
                                    acc_2[30] = old_23[2] + acc_2[30];
                                    acc_2[31] = old_23[3] + acc_2[31];
                                }
                                asm volatile("st.global.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(dq_accum + at_23), "f"(acc_2[(28) + 0]), "f"(acc_2[(28) + 1]), "f"(acc_2[(28) + 2]), "f"(acc_2[(28) + 3]), "l"(0x14F0000000000000ULL) : "memory");
                                float acc_3[32];
                                if (steps > 0) {
                                    asm volatile(
                                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                                        : "=f"(acc_3[0]), "=f"(acc_3[1]), "=f"(acc_3[2]), "=f"(acc_3[3]), "=f"(acc_3[4]), "=f"(acc_3[5]), "=f"(acc_3[6]), "=f"(acc_3[7]), "=f"(acc_3[8]), "=f"(acc_3[9]), "=f"(acc_3[10]), "=f"(acc_3[11]), "=f"(acc_3[12]), "=f"(acc_3[13]), "=f"(acc_3[14]), "=f"(acc_3[15]), "=f"(acc_3[16]), "=f"(acc_3[17]), "=f"(acc_3[18]), "=f"(acc_3[19]), "=f"(acc_3[20]), "=f"(acc_3[21]), "=f"(acc_3[22]), "=f"(acc_3[23]), "=f"(acc_3[24]), "=f"(acc_3[25]), "=f"(acc_3[26]), "=f"(acc_3[27]), "=f"(acc_3[28]), "=f"(acc_3[29]), "=f"(acc_3[30]), "=f"(acc_3[31])
                                        : "r"(taddr + (ec * 256 + 96) + lane_base));
                                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                                } else {
                                    #pragma unroll
                                    for (int i_3 = 0; i_3 < 32; i_3++) {
                                        acc_3[i_3] = 0.0f;
                                    }
                                }
                                long long at_24 = tile_off + 12288ll;
                                if (add != 0) {
                                    float old_24[4];
                                    {
                                        unsigned _v4_24_0;
                                        unsigned _v4_24_1;
                                        unsigned _v4_24_2;
                                        unsigned _v4_24_3;
                                        asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                            : "=r"(_v4_24_0), "=r"(_v4_24_1), "=r"(_v4_24_2), "=r"(_v4_24_3) : "l"((const void*)(dq_accum + at_24 + (0))), "l"(0x14F0000000000000ULL) : "memory");
                                        old_24[0 + 0] = __uint_as_float(_v4_24_0);
                                        old_24[0 + 1] = __uint_as_float(_v4_24_1);
                                        old_24[0 + 2] = __uint_as_float(_v4_24_2);
                                        old_24[0 + 3] = __uint_as_float(_v4_24_3);
                                    }
                                    acc_3[0] = old_24[0] + acc_3[0];
                                    acc_3[1] = old_24[1] + acc_3[1];
                                    acc_3[2] = old_24[2] + acc_3[2];
                                    acc_3[3] = old_24[3] + acc_3[3];
                                }
                                asm volatile("st.global.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(dq_accum + at_24), "f"(acc_3[(0) + 0]), "f"(acc_3[(0) + 1]), "f"(acc_3[(0) + 2]), "f"(acc_3[(0) + 3]), "l"(0x14F0000000000000ULL) : "memory");
                                long long at_25 = tile_off + 12800ll;
                                if (add != 0) {
                                    float old_25[4];
                                    {
                                        unsigned _v4_25_0;
                                        unsigned _v4_25_1;
                                        unsigned _v4_25_2;
                                        unsigned _v4_25_3;
                                        asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                            : "=r"(_v4_25_0), "=r"(_v4_25_1), "=r"(_v4_25_2), "=r"(_v4_25_3) : "l"((const void*)(dq_accum + at_25 + (0))), "l"(0x14F0000000000000ULL) : "memory");
                                        old_25[0 + 0] = __uint_as_float(_v4_25_0);
                                        old_25[0 + 1] = __uint_as_float(_v4_25_1);
                                        old_25[0 + 2] = __uint_as_float(_v4_25_2);
                                        old_25[0 + 3] = __uint_as_float(_v4_25_3);
                                    }
                                    acc_3[4] = old_25[0] + acc_3[4];
                                    acc_3[5] = old_25[1] + acc_3[5];
                                    acc_3[6] = old_25[2] + acc_3[6];
                                    acc_3[7] = old_25[3] + acc_3[7];
                                }
                                asm volatile("st.global.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(dq_accum + at_25), "f"(acc_3[(4) + 0]), "f"(acc_3[(4) + 1]), "f"(acc_3[(4) + 2]), "f"(acc_3[(4) + 3]), "l"(0x14F0000000000000ULL) : "memory");
                                long long at_26 = tile_off + 13312ll;
                                if (add != 0) {
                                    float old_26[4];
                                    {
                                        unsigned _v4_26_0;
                                        unsigned _v4_26_1;
                                        unsigned _v4_26_2;
                                        unsigned _v4_26_3;
                                        asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                            : "=r"(_v4_26_0), "=r"(_v4_26_1), "=r"(_v4_26_2), "=r"(_v4_26_3) : "l"((const void*)(dq_accum + at_26 + (0))), "l"(0x14F0000000000000ULL) : "memory");
                                        old_26[0 + 0] = __uint_as_float(_v4_26_0);
                                        old_26[0 + 1] = __uint_as_float(_v4_26_1);
                                        old_26[0 + 2] = __uint_as_float(_v4_26_2);
                                        old_26[0 + 3] = __uint_as_float(_v4_26_3);
                                    }
                                    acc_3[8] = old_26[0] + acc_3[8];
                                    acc_3[9] = old_26[1] + acc_3[9];
                                    acc_3[10] = old_26[2] + acc_3[10];
                                    acc_3[11] = old_26[3] + acc_3[11];
                                }
                                asm volatile("st.global.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(dq_accum + at_26), "f"(acc_3[(8) + 0]), "f"(acc_3[(8) + 1]), "f"(acc_3[(8) + 2]), "f"(acc_3[(8) + 3]), "l"(0x14F0000000000000ULL) : "memory");
                                long long at_27 = tile_off + 13824ll;
                                if (add != 0) {
                                    float old_27[4];
                                    {
                                        unsigned _v4_27_0;
                                        unsigned _v4_27_1;
                                        unsigned _v4_27_2;
                                        unsigned _v4_27_3;
                                        asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                            : "=r"(_v4_27_0), "=r"(_v4_27_1), "=r"(_v4_27_2), "=r"(_v4_27_3) : "l"((const void*)(dq_accum + at_27 + (0))), "l"(0x14F0000000000000ULL) : "memory");
                                        old_27[0 + 0] = __uint_as_float(_v4_27_0);
                                        old_27[0 + 1] = __uint_as_float(_v4_27_1);
                                        old_27[0 + 2] = __uint_as_float(_v4_27_2);
                                        old_27[0 + 3] = __uint_as_float(_v4_27_3);
                                    }
                                    acc_3[12] = old_27[0] + acc_3[12];
                                    acc_3[13] = old_27[1] + acc_3[13];
                                    acc_3[14] = old_27[2] + acc_3[14];
                                    acc_3[15] = old_27[3] + acc_3[15];
                                }
                                asm volatile("st.global.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(dq_accum + at_27), "f"(acc_3[(12) + 0]), "f"(acc_3[(12) + 1]), "f"(acc_3[(12) + 2]), "f"(acc_3[(12) + 3]), "l"(0x14F0000000000000ULL) : "memory");
                                long long at_28 = tile_off + 14336ll;
                                if (add != 0) {
                                    float old_28[4];
                                    {
                                        unsigned _v4_28_0;
                                        unsigned _v4_28_1;
                                        unsigned _v4_28_2;
                                        unsigned _v4_28_3;
                                        asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                            : "=r"(_v4_28_0), "=r"(_v4_28_1), "=r"(_v4_28_2), "=r"(_v4_28_3) : "l"((const void*)(dq_accum + at_28 + (0))), "l"(0x14F0000000000000ULL) : "memory");
                                        old_28[0 + 0] = __uint_as_float(_v4_28_0);
                                        old_28[0 + 1] = __uint_as_float(_v4_28_1);
                                        old_28[0 + 2] = __uint_as_float(_v4_28_2);
                                        old_28[0 + 3] = __uint_as_float(_v4_28_3);
                                    }
                                    acc_3[16] = old_28[0] + acc_3[16];
                                    acc_3[17] = old_28[1] + acc_3[17];
                                    acc_3[18] = old_28[2] + acc_3[18];
                                    acc_3[19] = old_28[3] + acc_3[19];
                                }
                                asm volatile("st.global.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(dq_accum + at_28), "f"(acc_3[(16) + 0]), "f"(acc_3[(16) + 1]), "f"(acc_3[(16) + 2]), "f"(acc_3[(16) + 3]), "l"(0x14F0000000000000ULL) : "memory");
                                long long at_29 = tile_off + 14848ll;
                                if (add != 0) {
                                    float old_29[4];
                                    {
                                        unsigned _v4_29_0;
                                        unsigned _v4_29_1;
                                        unsigned _v4_29_2;
                                        unsigned _v4_29_3;
                                        asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                            : "=r"(_v4_29_0), "=r"(_v4_29_1), "=r"(_v4_29_2), "=r"(_v4_29_3) : "l"((const void*)(dq_accum + at_29 + (0))), "l"(0x14F0000000000000ULL) : "memory");
                                        old_29[0 + 0] = __uint_as_float(_v4_29_0);
                                        old_29[0 + 1] = __uint_as_float(_v4_29_1);
                                        old_29[0 + 2] = __uint_as_float(_v4_29_2);
                                        old_29[0 + 3] = __uint_as_float(_v4_29_3);
                                    }
                                    acc_3[20] = old_29[0] + acc_3[20];
                                    acc_3[21] = old_29[1] + acc_3[21];
                                    acc_3[22] = old_29[2] + acc_3[22];
                                    acc_3[23] = old_29[3] + acc_3[23];
                                }
                                asm volatile("st.global.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(dq_accum + at_29), "f"(acc_3[(20) + 0]), "f"(acc_3[(20) + 1]), "f"(acc_3[(20) + 2]), "f"(acc_3[(20) + 3]), "l"(0x14F0000000000000ULL) : "memory");
                                long long at_30 = tile_off + 15360ll;
                                if (add != 0) {
                                    float old_30[4];
                                    {
                                        unsigned _v4_30_0;
                                        unsigned _v4_30_1;
                                        unsigned _v4_30_2;
                                        unsigned _v4_30_3;
                                        asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                            : "=r"(_v4_30_0), "=r"(_v4_30_1), "=r"(_v4_30_2), "=r"(_v4_30_3) : "l"((const void*)(dq_accum + at_30 + (0))), "l"(0x14F0000000000000ULL) : "memory");
                                        old_30[0 + 0] = __uint_as_float(_v4_30_0);
                                        old_30[0 + 1] = __uint_as_float(_v4_30_1);
                                        old_30[0 + 2] = __uint_as_float(_v4_30_2);
                                        old_30[0 + 3] = __uint_as_float(_v4_30_3);
                                    }
                                    acc_3[24] = old_30[0] + acc_3[24];
                                    acc_3[25] = old_30[1] + acc_3[25];
                                    acc_3[26] = old_30[2] + acc_3[26];
                                    acc_3[27] = old_30[3] + acc_3[27];
                                }
                                asm volatile("st.global.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(dq_accum + at_30), "f"(acc_3[(24) + 0]), "f"(acc_3[(24) + 1]), "f"(acc_3[(24) + 2]), "f"(acc_3[(24) + 3]), "l"(0x14F0000000000000ULL) : "memory");
                                long long at_31 = tile_off + 15872ll;
                                if (add != 0) {
                                    float old_31[4];
                                    {
                                        unsigned _v4_31_0;
                                        unsigned _v4_31_1;
                                        unsigned _v4_31_2;
                                        unsigned _v4_31_3;
                                        asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                            : "=r"(_v4_31_0), "=r"(_v4_31_1), "=r"(_v4_31_2), "=r"(_v4_31_3) : "l"((const void*)(dq_accum + at_31 + (0))), "l"(0x14F0000000000000ULL) : "memory");
                                        old_31[0 + 0] = __uint_as_float(_v4_31_0);
                                        old_31[0 + 1] = __uint_as_float(_v4_31_1);
                                        old_31[0 + 2] = __uint_as_float(_v4_31_2);
                                        old_31[0 + 3] = __uint_as_float(_v4_31_3);
                                    }
                                    acc_3[28] = old_31[0] + acc_3[28];
                                    acc_3[29] = old_31[1] + acc_3[29];
                                    acc_3[30] = old_31[2] + acc_3[30];
                                    acc_3[31] = old_31[3] + acc_3[31];
                                }
                                asm volatile("st.global.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(dq_accum + at_31), "f"(acc_3[(28) + 0]), "f"(acc_3[(28) + 1]), "f"(acc_3[(28) + 2]), "f"(acc_3[(28) + 3]), "l"(0x14F0000000000000ULL) : "memory");
                                float acc_4[32];
                                if (steps > 0) {
                                    asm volatile(
                                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                                        : "=f"(acc_4[0]), "=f"(acc_4[1]), "=f"(acc_4[2]), "=f"(acc_4[3]), "=f"(acc_4[4]), "=f"(acc_4[5]), "=f"(acc_4[6]), "=f"(acc_4[7]), "=f"(acc_4[8]), "=f"(acc_4[9]), "=f"(acc_4[10]), "=f"(acc_4[11]), "=f"(acc_4[12]), "=f"(acc_4[13]), "=f"(acc_4[14]), "=f"(acc_4[15]), "=f"(acc_4[16]), "=f"(acc_4[17]), "=f"(acc_4[18]), "=f"(acc_4[19]), "=f"(acc_4[20]), "=f"(acc_4[21]), "=f"(acc_4[22]), "=f"(acc_4[23]), "=f"(acc_4[24]), "=f"(acc_4[25]), "=f"(acc_4[26]), "=f"(acc_4[27]), "=f"(acc_4[28]), "=f"(acc_4[29]), "=f"(acc_4[30]), "=f"(acc_4[31])
                                        : "r"(taddr + (ec * 256 + 128) + lane_base));
                                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                                } else {
                                    #pragma unroll
                                    for (int i_4 = 0; i_4 < 32; i_4++) {
                                        acc_4[i_4] = 0.0f;
                                    }
                                }
                                long long at_32 = tile_off + 256ll;
                                if (add != 0) {
                                    float old_32[4];
                                    {
                                        unsigned _v4_32_0;
                                        unsigned _v4_32_1;
                                        unsigned _v4_32_2;
                                        unsigned _v4_32_3;
                                        asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                            : "=r"(_v4_32_0), "=r"(_v4_32_1), "=r"(_v4_32_2), "=r"(_v4_32_3) : "l"((const void*)(dq_accum + at_32 + (0))), "l"(0x14F0000000000000ULL) : "memory");
                                        old_32[0 + 0] = __uint_as_float(_v4_32_0);
                                        old_32[0 + 1] = __uint_as_float(_v4_32_1);
                                        old_32[0 + 2] = __uint_as_float(_v4_32_2);
                                        old_32[0 + 3] = __uint_as_float(_v4_32_3);
                                    }
                                    acc_4[0] = old_32[0] + acc_4[0];
                                    acc_4[1] = old_32[1] + acc_4[1];
                                    acc_4[2] = old_32[2] + acc_4[2];
                                    acc_4[3] = old_32[3] + acc_4[3];
                                }
                                asm volatile("st.global.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(dq_accum + at_32), "f"(acc_4[(0) + 0]), "f"(acc_4[(0) + 1]), "f"(acc_4[(0) + 2]), "f"(acc_4[(0) + 3]), "l"(0x14F0000000000000ULL) : "memory");
                                long long at_33 = tile_off + 768ll;
                                if (add != 0) {
                                    float old_33[4];
                                    {
                                        unsigned _v4_33_0;
                                        unsigned _v4_33_1;
                                        unsigned _v4_33_2;
                                        unsigned _v4_33_3;
                                        asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                            : "=r"(_v4_33_0), "=r"(_v4_33_1), "=r"(_v4_33_2), "=r"(_v4_33_3) : "l"((const void*)(dq_accum + at_33 + (0))), "l"(0x14F0000000000000ULL) : "memory");
                                        old_33[0 + 0] = __uint_as_float(_v4_33_0);
                                        old_33[0 + 1] = __uint_as_float(_v4_33_1);
                                        old_33[0 + 2] = __uint_as_float(_v4_33_2);
                                        old_33[0 + 3] = __uint_as_float(_v4_33_3);
                                    }
                                    acc_4[4] = old_33[0] + acc_4[4];
                                    acc_4[5] = old_33[1] + acc_4[5];
                                    acc_4[6] = old_33[2] + acc_4[6];
                                    acc_4[7] = old_33[3] + acc_4[7];
                                }
                                asm volatile("st.global.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(dq_accum + at_33), "f"(acc_4[(4) + 0]), "f"(acc_4[(4) + 1]), "f"(acc_4[(4) + 2]), "f"(acc_4[(4) + 3]), "l"(0x14F0000000000000ULL) : "memory");
                                long long at_34 = tile_off + 1280ll;
                                if (add != 0) {
                                    float old_34[4];
                                    {
                                        unsigned _v4_34_0;
                                        unsigned _v4_34_1;
                                        unsigned _v4_34_2;
                                        unsigned _v4_34_3;
                                        asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                            : "=r"(_v4_34_0), "=r"(_v4_34_1), "=r"(_v4_34_2), "=r"(_v4_34_3) : "l"((const void*)(dq_accum + at_34 + (0))), "l"(0x14F0000000000000ULL) : "memory");
                                        old_34[0 + 0] = __uint_as_float(_v4_34_0);
                                        old_34[0 + 1] = __uint_as_float(_v4_34_1);
                                        old_34[0 + 2] = __uint_as_float(_v4_34_2);
                                        old_34[0 + 3] = __uint_as_float(_v4_34_3);
                                    }
                                    acc_4[8] = old_34[0] + acc_4[8];
                                    acc_4[9] = old_34[1] + acc_4[9];
                                    acc_4[10] = old_34[2] + acc_4[10];
                                    acc_4[11] = old_34[3] + acc_4[11];
                                }
                                asm volatile("st.global.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(dq_accum + at_34), "f"(acc_4[(8) + 0]), "f"(acc_4[(8) + 1]), "f"(acc_4[(8) + 2]), "f"(acc_4[(8) + 3]), "l"(0x14F0000000000000ULL) : "memory");
                                long long at_35 = tile_off + 1792ll;
                                if (add != 0) {
                                    float old_35[4];
                                    {
                                        unsigned _v4_35_0;
                                        unsigned _v4_35_1;
                                        unsigned _v4_35_2;
                                        unsigned _v4_35_3;
                                        asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                            : "=r"(_v4_35_0), "=r"(_v4_35_1), "=r"(_v4_35_2), "=r"(_v4_35_3) : "l"((const void*)(dq_accum + at_35 + (0))), "l"(0x14F0000000000000ULL) : "memory");
                                        old_35[0 + 0] = __uint_as_float(_v4_35_0);
                                        old_35[0 + 1] = __uint_as_float(_v4_35_1);
                                        old_35[0 + 2] = __uint_as_float(_v4_35_2);
                                        old_35[0 + 3] = __uint_as_float(_v4_35_3);
                                    }
                                    acc_4[12] = old_35[0] + acc_4[12];
                                    acc_4[13] = old_35[1] + acc_4[13];
                                    acc_4[14] = old_35[2] + acc_4[14];
                                    acc_4[15] = old_35[3] + acc_4[15];
                                }
                                asm volatile("st.global.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(dq_accum + at_35), "f"(acc_4[(12) + 0]), "f"(acc_4[(12) + 1]), "f"(acc_4[(12) + 2]), "f"(acc_4[(12) + 3]), "l"(0x14F0000000000000ULL) : "memory");
                                long long at_36 = tile_off + 2304ll;
                                if (add != 0) {
                                    float old_36[4];
                                    {
                                        unsigned _v4_36_0;
                                        unsigned _v4_36_1;
                                        unsigned _v4_36_2;
                                        unsigned _v4_36_3;
                                        asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                            : "=r"(_v4_36_0), "=r"(_v4_36_1), "=r"(_v4_36_2), "=r"(_v4_36_3) : "l"((const void*)(dq_accum + at_36 + (0))), "l"(0x14F0000000000000ULL) : "memory");
                                        old_36[0 + 0] = __uint_as_float(_v4_36_0);
                                        old_36[0 + 1] = __uint_as_float(_v4_36_1);
                                        old_36[0 + 2] = __uint_as_float(_v4_36_2);
                                        old_36[0 + 3] = __uint_as_float(_v4_36_3);
                                    }
                                    acc_4[16] = old_36[0] + acc_4[16];
                                    acc_4[17] = old_36[1] + acc_4[17];
                                    acc_4[18] = old_36[2] + acc_4[18];
                                    acc_4[19] = old_36[3] + acc_4[19];
                                }
                                asm volatile("st.global.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(dq_accum + at_36), "f"(acc_4[(16) + 0]), "f"(acc_4[(16) + 1]), "f"(acc_4[(16) + 2]), "f"(acc_4[(16) + 3]), "l"(0x14F0000000000000ULL) : "memory");
                                long long at_37 = tile_off + 2816ll;
                                if (add != 0) {
                                    float old_37[4];
                                    {
                                        unsigned _v4_37_0;
                                        unsigned _v4_37_1;
                                        unsigned _v4_37_2;
                                        unsigned _v4_37_3;
                                        asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                            : "=r"(_v4_37_0), "=r"(_v4_37_1), "=r"(_v4_37_2), "=r"(_v4_37_3) : "l"((const void*)(dq_accum + at_37 + (0))), "l"(0x14F0000000000000ULL) : "memory");
                                        old_37[0 + 0] = __uint_as_float(_v4_37_0);
                                        old_37[0 + 1] = __uint_as_float(_v4_37_1);
                                        old_37[0 + 2] = __uint_as_float(_v4_37_2);
                                        old_37[0 + 3] = __uint_as_float(_v4_37_3);
                                    }
                                    acc_4[20] = old_37[0] + acc_4[20];
                                    acc_4[21] = old_37[1] + acc_4[21];
                                    acc_4[22] = old_37[2] + acc_4[22];
                                    acc_4[23] = old_37[3] + acc_4[23];
                                }
                                asm volatile("st.global.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(dq_accum + at_37), "f"(acc_4[(20) + 0]), "f"(acc_4[(20) + 1]), "f"(acc_4[(20) + 2]), "f"(acc_4[(20) + 3]), "l"(0x14F0000000000000ULL) : "memory");
                                long long at_38 = tile_off + 3328ll;
                                if (add != 0) {
                                    float old_38[4];
                                    {
                                        unsigned _v4_38_0;
                                        unsigned _v4_38_1;
                                        unsigned _v4_38_2;
                                        unsigned _v4_38_3;
                                        asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                            : "=r"(_v4_38_0), "=r"(_v4_38_1), "=r"(_v4_38_2), "=r"(_v4_38_3) : "l"((const void*)(dq_accum + at_38 + (0))), "l"(0x14F0000000000000ULL) : "memory");
                                        old_38[0 + 0] = __uint_as_float(_v4_38_0);
                                        old_38[0 + 1] = __uint_as_float(_v4_38_1);
                                        old_38[0 + 2] = __uint_as_float(_v4_38_2);
                                        old_38[0 + 3] = __uint_as_float(_v4_38_3);
                                    }
                                    acc_4[24] = old_38[0] + acc_4[24];
                                    acc_4[25] = old_38[1] + acc_4[25];
                                    acc_4[26] = old_38[2] + acc_4[26];
                                    acc_4[27] = old_38[3] + acc_4[27];
                                }
                                asm volatile("st.global.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(dq_accum + at_38), "f"(acc_4[(24) + 0]), "f"(acc_4[(24) + 1]), "f"(acc_4[(24) + 2]), "f"(acc_4[(24) + 3]), "l"(0x14F0000000000000ULL) : "memory");
                                long long at_39 = tile_off + 3840ll;
                                if (add != 0) {
                                    float old_39[4];
                                    {
                                        unsigned _v4_39_0;
                                        unsigned _v4_39_1;
                                        unsigned _v4_39_2;
                                        unsigned _v4_39_3;
                                        asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                            : "=r"(_v4_39_0), "=r"(_v4_39_1), "=r"(_v4_39_2), "=r"(_v4_39_3) : "l"((const void*)(dq_accum + at_39 + (0))), "l"(0x14F0000000000000ULL) : "memory");
                                        old_39[0 + 0] = __uint_as_float(_v4_39_0);
                                        old_39[0 + 1] = __uint_as_float(_v4_39_1);
                                        old_39[0 + 2] = __uint_as_float(_v4_39_2);
                                        old_39[0 + 3] = __uint_as_float(_v4_39_3);
                                    }
                                    acc_4[28] = old_39[0] + acc_4[28];
                                    acc_4[29] = old_39[1] + acc_4[29];
                                    acc_4[30] = old_39[2] + acc_4[30];
                                    acc_4[31] = old_39[3] + acc_4[31];
                                }
                                asm volatile("st.global.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(dq_accum + at_39), "f"(acc_4[(28) + 0]), "f"(acc_4[(28) + 1]), "f"(acc_4[(28) + 2]), "f"(acc_4[(28) + 3]), "l"(0x14F0000000000000ULL) : "memory");
                                float acc_5[32];
                                if (steps > 0) {
                                    asm volatile(
                                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                                        : "=f"(acc_5[0]), "=f"(acc_5[1]), "=f"(acc_5[2]), "=f"(acc_5[3]), "=f"(acc_5[4]), "=f"(acc_5[5]), "=f"(acc_5[6]), "=f"(acc_5[7]), "=f"(acc_5[8]), "=f"(acc_5[9]), "=f"(acc_5[10]), "=f"(acc_5[11]), "=f"(acc_5[12]), "=f"(acc_5[13]), "=f"(acc_5[14]), "=f"(acc_5[15]), "=f"(acc_5[16]), "=f"(acc_5[17]), "=f"(acc_5[18]), "=f"(acc_5[19]), "=f"(acc_5[20]), "=f"(acc_5[21]), "=f"(acc_5[22]), "=f"(acc_5[23]), "=f"(acc_5[24]), "=f"(acc_5[25]), "=f"(acc_5[26]), "=f"(acc_5[27]), "=f"(acc_5[28]), "=f"(acc_5[29]), "=f"(acc_5[30]), "=f"(acc_5[31])
                                        : "r"(taddr + (ec * 256 + 160) + lane_base));
                                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                                } else {
                                    #pragma unroll
                                    for (int i_5 = 0; i_5 < 32; i_5++) {
                                        acc_5[i_5] = 0.0f;
                                    }
                                }
                                long long at_40 = tile_off + 4352ll;
                                if (add != 0) {
                                    float old_40[4];
                                    {
                                        unsigned _v4_40_0;
                                        unsigned _v4_40_1;
                                        unsigned _v4_40_2;
                                        unsigned _v4_40_3;
                                        asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                            : "=r"(_v4_40_0), "=r"(_v4_40_1), "=r"(_v4_40_2), "=r"(_v4_40_3) : "l"((const void*)(dq_accum + at_40 + (0))), "l"(0x14F0000000000000ULL) : "memory");
                                        old_40[0 + 0] = __uint_as_float(_v4_40_0);
                                        old_40[0 + 1] = __uint_as_float(_v4_40_1);
                                        old_40[0 + 2] = __uint_as_float(_v4_40_2);
                                        old_40[0 + 3] = __uint_as_float(_v4_40_3);
                                    }
                                    acc_5[0] = old_40[0] + acc_5[0];
                                    acc_5[1] = old_40[1] + acc_5[1];
                                    acc_5[2] = old_40[2] + acc_5[2];
                                    acc_5[3] = old_40[3] + acc_5[3];
                                }
                                asm volatile("st.global.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(dq_accum + at_40), "f"(acc_5[(0) + 0]), "f"(acc_5[(0) + 1]), "f"(acc_5[(0) + 2]), "f"(acc_5[(0) + 3]), "l"(0x14F0000000000000ULL) : "memory");
                                long long at_41 = tile_off + 4864ll;
                                if (add != 0) {
                                    float old_41[4];
                                    {
                                        unsigned _v4_41_0;
                                        unsigned _v4_41_1;
                                        unsigned _v4_41_2;
                                        unsigned _v4_41_3;
                                        asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                            : "=r"(_v4_41_0), "=r"(_v4_41_1), "=r"(_v4_41_2), "=r"(_v4_41_3) : "l"((const void*)(dq_accum + at_41 + (0))), "l"(0x14F0000000000000ULL) : "memory");
                                        old_41[0 + 0] = __uint_as_float(_v4_41_0);
                                        old_41[0 + 1] = __uint_as_float(_v4_41_1);
                                        old_41[0 + 2] = __uint_as_float(_v4_41_2);
                                        old_41[0 + 3] = __uint_as_float(_v4_41_3);
                                    }
                                    acc_5[4] = old_41[0] + acc_5[4];
                                    acc_5[5] = old_41[1] + acc_5[5];
                                    acc_5[6] = old_41[2] + acc_5[6];
                                    acc_5[7] = old_41[3] + acc_5[7];
                                }
                                asm volatile("st.global.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(dq_accum + at_41), "f"(acc_5[(4) + 0]), "f"(acc_5[(4) + 1]), "f"(acc_5[(4) + 2]), "f"(acc_5[(4) + 3]), "l"(0x14F0000000000000ULL) : "memory");
                                long long at_42 = tile_off + 5376ll;
                                if (add != 0) {
                                    float old_42[4];
                                    {
                                        unsigned _v4_42_0;
                                        unsigned _v4_42_1;
                                        unsigned _v4_42_2;
                                        unsigned _v4_42_3;
                                        asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                            : "=r"(_v4_42_0), "=r"(_v4_42_1), "=r"(_v4_42_2), "=r"(_v4_42_3) : "l"((const void*)(dq_accum + at_42 + (0))), "l"(0x14F0000000000000ULL) : "memory");
                                        old_42[0 + 0] = __uint_as_float(_v4_42_0);
                                        old_42[0 + 1] = __uint_as_float(_v4_42_1);
                                        old_42[0 + 2] = __uint_as_float(_v4_42_2);
                                        old_42[0 + 3] = __uint_as_float(_v4_42_3);
                                    }
                                    acc_5[8] = old_42[0] + acc_5[8];
                                    acc_5[9] = old_42[1] + acc_5[9];
                                    acc_5[10] = old_42[2] + acc_5[10];
                                    acc_5[11] = old_42[3] + acc_5[11];
                                }
                                asm volatile("st.global.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(dq_accum + at_42), "f"(acc_5[(8) + 0]), "f"(acc_5[(8) + 1]), "f"(acc_5[(8) + 2]), "f"(acc_5[(8) + 3]), "l"(0x14F0000000000000ULL) : "memory");
                                long long at_43 = tile_off + 5888ll;
                                if (add != 0) {
                                    float old_43[4];
                                    {
                                        unsigned _v4_43_0;
                                        unsigned _v4_43_1;
                                        unsigned _v4_43_2;
                                        unsigned _v4_43_3;
                                        asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                            : "=r"(_v4_43_0), "=r"(_v4_43_1), "=r"(_v4_43_2), "=r"(_v4_43_3) : "l"((const void*)(dq_accum + at_43 + (0))), "l"(0x14F0000000000000ULL) : "memory");
                                        old_43[0 + 0] = __uint_as_float(_v4_43_0);
                                        old_43[0 + 1] = __uint_as_float(_v4_43_1);
                                        old_43[0 + 2] = __uint_as_float(_v4_43_2);
                                        old_43[0 + 3] = __uint_as_float(_v4_43_3);
                                    }
                                    acc_5[12] = old_43[0] + acc_5[12];
                                    acc_5[13] = old_43[1] + acc_5[13];
                                    acc_5[14] = old_43[2] + acc_5[14];
                                    acc_5[15] = old_43[3] + acc_5[15];
                                }
                                asm volatile("st.global.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(dq_accum + at_43), "f"(acc_5[(12) + 0]), "f"(acc_5[(12) + 1]), "f"(acc_5[(12) + 2]), "f"(acc_5[(12) + 3]), "l"(0x14F0000000000000ULL) : "memory");
                                long long at_44 = tile_off + 6400ll;
                                if (add != 0) {
                                    float old_44[4];
                                    {
                                        unsigned _v4_44_0;
                                        unsigned _v4_44_1;
                                        unsigned _v4_44_2;
                                        unsigned _v4_44_3;
                                        asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                            : "=r"(_v4_44_0), "=r"(_v4_44_1), "=r"(_v4_44_2), "=r"(_v4_44_3) : "l"((const void*)(dq_accum + at_44 + (0))), "l"(0x14F0000000000000ULL) : "memory");
                                        old_44[0 + 0] = __uint_as_float(_v4_44_0);
                                        old_44[0 + 1] = __uint_as_float(_v4_44_1);
                                        old_44[0 + 2] = __uint_as_float(_v4_44_2);
                                        old_44[0 + 3] = __uint_as_float(_v4_44_3);
                                    }
                                    acc_5[16] = old_44[0] + acc_5[16];
                                    acc_5[17] = old_44[1] + acc_5[17];
                                    acc_5[18] = old_44[2] + acc_5[18];
                                    acc_5[19] = old_44[3] + acc_5[19];
                                }
                                asm volatile("st.global.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(dq_accum + at_44), "f"(acc_5[(16) + 0]), "f"(acc_5[(16) + 1]), "f"(acc_5[(16) + 2]), "f"(acc_5[(16) + 3]), "l"(0x14F0000000000000ULL) : "memory");
                                long long at_45 = tile_off + 6912ll;
                                if (add != 0) {
                                    float old_45[4];
                                    {
                                        unsigned _v4_45_0;
                                        unsigned _v4_45_1;
                                        unsigned _v4_45_2;
                                        unsigned _v4_45_3;
                                        asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                            : "=r"(_v4_45_0), "=r"(_v4_45_1), "=r"(_v4_45_2), "=r"(_v4_45_3) : "l"((const void*)(dq_accum + at_45 + (0))), "l"(0x14F0000000000000ULL) : "memory");
                                        old_45[0 + 0] = __uint_as_float(_v4_45_0);
                                        old_45[0 + 1] = __uint_as_float(_v4_45_1);
                                        old_45[0 + 2] = __uint_as_float(_v4_45_2);
                                        old_45[0 + 3] = __uint_as_float(_v4_45_3);
                                    }
                                    acc_5[20] = old_45[0] + acc_5[20];
                                    acc_5[21] = old_45[1] + acc_5[21];
                                    acc_5[22] = old_45[2] + acc_5[22];
                                    acc_5[23] = old_45[3] + acc_5[23];
                                }
                                asm volatile("st.global.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(dq_accum + at_45), "f"(acc_5[(20) + 0]), "f"(acc_5[(20) + 1]), "f"(acc_5[(20) + 2]), "f"(acc_5[(20) + 3]), "l"(0x14F0000000000000ULL) : "memory");
                                long long at_46 = tile_off + 7424ll;
                                if (add != 0) {
                                    float old_46[4];
                                    {
                                        unsigned _v4_46_0;
                                        unsigned _v4_46_1;
                                        unsigned _v4_46_2;
                                        unsigned _v4_46_3;
                                        asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                            : "=r"(_v4_46_0), "=r"(_v4_46_1), "=r"(_v4_46_2), "=r"(_v4_46_3) : "l"((const void*)(dq_accum + at_46 + (0))), "l"(0x14F0000000000000ULL) : "memory");
                                        old_46[0 + 0] = __uint_as_float(_v4_46_0);
                                        old_46[0 + 1] = __uint_as_float(_v4_46_1);
                                        old_46[0 + 2] = __uint_as_float(_v4_46_2);
                                        old_46[0 + 3] = __uint_as_float(_v4_46_3);
                                    }
                                    acc_5[24] = old_46[0] + acc_5[24];
                                    acc_5[25] = old_46[1] + acc_5[25];
                                    acc_5[26] = old_46[2] + acc_5[26];
                                    acc_5[27] = old_46[3] + acc_5[27];
                                }
                                asm volatile("st.global.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(dq_accum + at_46), "f"(acc_5[(24) + 0]), "f"(acc_5[(24) + 1]), "f"(acc_5[(24) + 2]), "f"(acc_5[(24) + 3]), "l"(0x14F0000000000000ULL) : "memory");
                                long long at_47 = tile_off + 7936ll;
                                if (add != 0) {
                                    float old_47[4];
                                    {
                                        unsigned _v4_47_0;
                                        unsigned _v4_47_1;
                                        unsigned _v4_47_2;
                                        unsigned _v4_47_3;
                                        asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                            : "=r"(_v4_47_0), "=r"(_v4_47_1), "=r"(_v4_47_2), "=r"(_v4_47_3) : "l"((const void*)(dq_accum + at_47 + (0))), "l"(0x14F0000000000000ULL) : "memory");
                                        old_47[0 + 0] = __uint_as_float(_v4_47_0);
                                        old_47[0 + 1] = __uint_as_float(_v4_47_1);
                                        old_47[0 + 2] = __uint_as_float(_v4_47_2);
                                        old_47[0 + 3] = __uint_as_float(_v4_47_3);
                                    }
                                    acc_5[28] = old_47[0] + acc_5[28];
                                    acc_5[29] = old_47[1] + acc_5[29];
                                    acc_5[30] = old_47[2] + acc_5[30];
                                    acc_5[31] = old_47[3] + acc_5[31];
                                }
                                asm volatile("st.global.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(dq_accum + at_47), "f"(acc_5[(28) + 0]), "f"(acc_5[(28) + 1]), "f"(acc_5[(28) + 2]), "f"(acc_5[(28) + 3]), "l"(0x14F0000000000000ULL) : "memory");
                                float acc_6[32];
                                if (steps > 0) {
                                    asm volatile(
                                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                                        : "=f"(acc_6[0]), "=f"(acc_6[1]), "=f"(acc_6[2]), "=f"(acc_6[3]), "=f"(acc_6[4]), "=f"(acc_6[5]), "=f"(acc_6[6]), "=f"(acc_6[7]), "=f"(acc_6[8]), "=f"(acc_6[9]), "=f"(acc_6[10]), "=f"(acc_6[11]), "=f"(acc_6[12]), "=f"(acc_6[13]), "=f"(acc_6[14]), "=f"(acc_6[15]), "=f"(acc_6[16]), "=f"(acc_6[17]), "=f"(acc_6[18]), "=f"(acc_6[19]), "=f"(acc_6[20]), "=f"(acc_6[21]), "=f"(acc_6[22]), "=f"(acc_6[23]), "=f"(acc_6[24]), "=f"(acc_6[25]), "=f"(acc_6[26]), "=f"(acc_6[27]), "=f"(acc_6[28]), "=f"(acc_6[29]), "=f"(acc_6[30]), "=f"(acc_6[31])
                                        : "r"(taddr + (ec * 256 + 192) + lane_base));
                                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                                } else {
                                    #pragma unroll
                                    for (int i_6 = 0; i_6 < 32; i_6++) {
                                        acc_6[i_6] = 0.0f;
                                    }
                                }
                                long long at_48 = tile_off + 8448ll;
                                if (add != 0) {
                                    float old_48[4];
                                    {
                                        unsigned _v4_48_0;
                                        unsigned _v4_48_1;
                                        unsigned _v4_48_2;
                                        unsigned _v4_48_3;
                                        asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                            : "=r"(_v4_48_0), "=r"(_v4_48_1), "=r"(_v4_48_2), "=r"(_v4_48_3) : "l"((const void*)(dq_accum + at_48 + (0))), "l"(0x14F0000000000000ULL) : "memory");
                                        old_48[0 + 0] = __uint_as_float(_v4_48_0);
                                        old_48[0 + 1] = __uint_as_float(_v4_48_1);
                                        old_48[0 + 2] = __uint_as_float(_v4_48_2);
                                        old_48[0 + 3] = __uint_as_float(_v4_48_3);
                                    }
                                    acc_6[0] = old_48[0] + acc_6[0];
                                    acc_6[1] = old_48[1] + acc_6[1];
                                    acc_6[2] = old_48[2] + acc_6[2];
                                    acc_6[3] = old_48[3] + acc_6[3];
                                }
                                asm volatile("st.global.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(dq_accum + at_48), "f"(acc_6[(0) + 0]), "f"(acc_6[(0) + 1]), "f"(acc_6[(0) + 2]), "f"(acc_6[(0) + 3]), "l"(0x14F0000000000000ULL) : "memory");
                                long long at_49 = tile_off + 8960ll;
                                if (add != 0) {
                                    float old_49[4];
                                    {
                                        unsigned _v4_49_0;
                                        unsigned _v4_49_1;
                                        unsigned _v4_49_2;
                                        unsigned _v4_49_3;
                                        asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                            : "=r"(_v4_49_0), "=r"(_v4_49_1), "=r"(_v4_49_2), "=r"(_v4_49_3) : "l"((const void*)(dq_accum + at_49 + (0))), "l"(0x14F0000000000000ULL) : "memory");
                                        old_49[0 + 0] = __uint_as_float(_v4_49_0);
                                        old_49[0 + 1] = __uint_as_float(_v4_49_1);
                                        old_49[0 + 2] = __uint_as_float(_v4_49_2);
                                        old_49[0 + 3] = __uint_as_float(_v4_49_3);
                                    }
                                    acc_6[4] = old_49[0] + acc_6[4];
                                    acc_6[5] = old_49[1] + acc_6[5];
                                    acc_6[6] = old_49[2] + acc_6[6];
                                    acc_6[7] = old_49[3] + acc_6[7];
                                }
                                asm volatile("st.global.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(dq_accum + at_49), "f"(acc_6[(4) + 0]), "f"(acc_6[(4) + 1]), "f"(acc_6[(4) + 2]), "f"(acc_6[(4) + 3]), "l"(0x14F0000000000000ULL) : "memory");
                                long long at_50 = tile_off + 9472ll;
                                if (add != 0) {
                                    float old_50[4];
                                    {
                                        unsigned _v4_50_0;
                                        unsigned _v4_50_1;
                                        unsigned _v4_50_2;
                                        unsigned _v4_50_3;
                                        asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                            : "=r"(_v4_50_0), "=r"(_v4_50_1), "=r"(_v4_50_2), "=r"(_v4_50_3) : "l"((const void*)(dq_accum + at_50 + (0))), "l"(0x14F0000000000000ULL) : "memory");
                                        old_50[0 + 0] = __uint_as_float(_v4_50_0);
                                        old_50[0 + 1] = __uint_as_float(_v4_50_1);
                                        old_50[0 + 2] = __uint_as_float(_v4_50_2);
                                        old_50[0 + 3] = __uint_as_float(_v4_50_3);
                                    }
                                    acc_6[8] = old_50[0] + acc_6[8];
                                    acc_6[9] = old_50[1] + acc_6[9];
                                    acc_6[10] = old_50[2] + acc_6[10];
                                    acc_6[11] = old_50[3] + acc_6[11];
                                }
                                asm volatile("st.global.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(dq_accum + at_50), "f"(acc_6[(8) + 0]), "f"(acc_6[(8) + 1]), "f"(acc_6[(8) + 2]), "f"(acc_6[(8) + 3]), "l"(0x14F0000000000000ULL) : "memory");
                                long long at_51 = tile_off + 9984ll;
                                if (add != 0) {
                                    float old_51[4];
                                    {
                                        unsigned _v4_51_0;
                                        unsigned _v4_51_1;
                                        unsigned _v4_51_2;
                                        unsigned _v4_51_3;
                                        asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                            : "=r"(_v4_51_0), "=r"(_v4_51_1), "=r"(_v4_51_2), "=r"(_v4_51_3) : "l"((const void*)(dq_accum + at_51 + (0))), "l"(0x14F0000000000000ULL) : "memory");
                                        old_51[0 + 0] = __uint_as_float(_v4_51_0);
                                        old_51[0 + 1] = __uint_as_float(_v4_51_1);
                                        old_51[0 + 2] = __uint_as_float(_v4_51_2);
                                        old_51[0 + 3] = __uint_as_float(_v4_51_3);
                                    }
                                    acc_6[12] = old_51[0] + acc_6[12];
                                    acc_6[13] = old_51[1] + acc_6[13];
                                    acc_6[14] = old_51[2] + acc_6[14];
                                    acc_6[15] = old_51[3] + acc_6[15];
                                }
                                asm volatile("st.global.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(dq_accum + at_51), "f"(acc_6[(12) + 0]), "f"(acc_6[(12) + 1]), "f"(acc_6[(12) + 2]), "f"(acc_6[(12) + 3]), "l"(0x14F0000000000000ULL) : "memory");
                                long long at_52 = tile_off + 10496ll;
                                if (add != 0) {
                                    float old_52[4];
                                    {
                                        unsigned _v4_52_0;
                                        unsigned _v4_52_1;
                                        unsigned _v4_52_2;
                                        unsigned _v4_52_3;
                                        asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                            : "=r"(_v4_52_0), "=r"(_v4_52_1), "=r"(_v4_52_2), "=r"(_v4_52_3) : "l"((const void*)(dq_accum + at_52 + (0))), "l"(0x14F0000000000000ULL) : "memory");
                                        old_52[0 + 0] = __uint_as_float(_v4_52_0);
                                        old_52[0 + 1] = __uint_as_float(_v4_52_1);
                                        old_52[0 + 2] = __uint_as_float(_v4_52_2);
                                        old_52[0 + 3] = __uint_as_float(_v4_52_3);
                                    }
                                    acc_6[16] = old_52[0] + acc_6[16];
                                    acc_6[17] = old_52[1] + acc_6[17];
                                    acc_6[18] = old_52[2] + acc_6[18];
                                    acc_6[19] = old_52[3] + acc_6[19];
                                }
                                asm volatile("st.global.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(dq_accum + at_52), "f"(acc_6[(16) + 0]), "f"(acc_6[(16) + 1]), "f"(acc_6[(16) + 2]), "f"(acc_6[(16) + 3]), "l"(0x14F0000000000000ULL) : "memory");
                                long long at_53 = tile_off + 11008ll;
                                if (add != 0) {
                                    float old_53[4];
                                    {
                                        unsigned _v4_53_0;
                                        unsigned _v4_53_1;
                                        unsigned _v4_53_2;
                                        unsigned _v4_53_3;
                                        asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                            : "=r"(_v4_53_0), "=r"(_v4_53_1), "=r"(_v4_53_2), "=r"(_v4_53_3) : "l"((const void*)(dq_accum + at_53 + (0))), "l"(0x14F0000000000000ULL) : "memory");
                                        old_53[0 + 0] = __uint_as_float(_v4_53_0);
                                        old_53[0 + 1] = __uint_as_float(_v4_53_1);
                                        old_53[0 + 2] = __uint_as_float(_v4_53_2);
                                        old_53[0 + 3] = __uint_as_float(_v4_53_3);
                                    }
                                    acc_6[20] = old_53[0] + acc_6[20];
                                    acc_6[21] = old_53[1] + acc_6[21];
                                    acc_6[22] = old_53[2] + acc_6[22];
                                    acc_6[23] = old_53[3] + acc_6[23];
                                }
                                asm volatile("st.global.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(dq_accum + at_53), "f"(acc_6[(20) + 0]), "f"(acc_6[(20) + 1]), "f"(acc_6[(20) + 2]), "f"(acc_6[(20) + 3]), "l"(0x14F0000000000000ULL) : "memory");
                                long long at_54 = tile_off + 11520ll;
                                if (add != 0) {
                                    float old_54[4];
                                    {
                                        unsigned _v4_54_0;
                                        unsigned _v4_54_1;
                                        unsigned _v4_54_2;
                                        unsigned _v4_54_3;
                                        asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                            : "=r"(_v4_54_0), "=r"(_v4_54_1), "=r"(_v4_54_2), "=r"(_v4_54_3) : "l"((const void*)(dq_accum + at_54 + (0))), "l"(0x14F0000000000000ULL) : "memory");
                                        old_54[0 + 0] = __uint_as_float(_v4_54_0);
                                        old_54[0 + 1] = __uint_as_float(_v4_54_1);
                                        old_54[0 + 2] = __uint_as_float(_v4_54_2);
                                        old_54[0 + 3] = __uint_as_float(_v4_54_3);
                                    }
                                    acc_6[24] = old_54[0] + acc_6[24];
                                    acc_6[25] = old_54[1] + acc_6[25];
                                    acc_6[26] = old_54[2] + acc_6[26];
                                    acc_6[27] = old_54[3] + acc_6[27];
                                }
                                asm volatile("st.global.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(dq_accum + at_54), "f"(acc_6[(24) + 0]), "f"(acc_6[(24) + 1]), "f"(acc_6[(24) + 2]), "f"(acc_6[(24) + 3]), "l"(0x14F0000000000000ULL) : "memory");
                                long long at_55 = tile_off + 12032ll;
                                if (add != 0) {
                                    float old_55[4];
                                    {
                                        unsigned _v4_55_0;
                                        unsigned _v4_55_1;
                                        unsigned _v4_55_2;
                                        unsigned _v4_55_3;
                                        asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                            : "=r"(_v4_55_0), "=r"(_v4_55_1), "=r"(_v4_55_2), "=r"(_v4_55_3) : "l"((const void*)(dq_accum + at_55 + (0))), "l"(0x14F0000000000000ULL) : "memory");
                                        old_55[0 + 0] = __uint_as_float(_v4_55_0);
                                        old_55[0 + 1] = __uint_as_float(_v4_55_1);
                                        old_55[0 + 2] = __uint_as_float(_v4_55_2);
                                        old_55[0 + 3] = __uint_as_float(_v4_55_3);
                                    }
                                    acc_6[28] = old_55[0] + acc_6[28];
                                    acc_6[29] = old_55[1] + acc_6[29];
                                    acc_6[30] = old_55[2] + acc_6[30];
                                    acc_6[31] = old_55[3] + acc_6[31];
                                }
                                asm volatile("st.global.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(dq_accum + at_55), "f"(acc_6[(28) + 0]), "f"(acc_6[(28) + 1]), "f"(acc_6[(28) + 2]), "f"(acc_6[(28) + 3]), "l"(0x14F0000000000000ULL) : "memory");
                                float acc_7[32];
                                if (steps > 0) {
                                    asm volatile(
                                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                                        : "=f"(acc_7[0]), "=f"(acc_7[1]), "=f"(acc_7[2]), "=f"(acc_7[3]), "=f"(acc_7[4]), "=f"(acc_7[5]), "=f"(acc_7[6]), "=f"(acc_7[7]), "=f"(acc_7[8]), "=f"(acc_7[9]), "=f"(acc_7[10]), "=f"(acc_7[11]), "=f"(acc_7[12]), "=f"(acc_7[13]), "=f"(acc_7[14]), "=f"(acc_7[15]), "=f"(acc_7[16]), "=f"(acc_7[17]), "=f"(acc_7[18]), "=f"(acc_7[19]), "=f"(acc_7[20]), "=f"(acc_7[21]), "=f"(acc_7[22]), "=f"(acc_7[23]), "=f"(acc_7[24]), "=f"(acc_7[25]), "=f"(acc_7[26]), "=f"(acc_7[27]), "=f"(acc_7[28]), "=f"(acc_7[29]), "=f"(acc_7[30]), "=f"(acc_7[31])
                                        : "r"(taddr + (ec * 256 + 224) + lane_base));
                                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                                } else {
                                    #pragma unroll
                                    for (int i_7 = 0; i_7 < 32; i_7++) {
                                        acc_7[i_7] = 0.0f;
                                    }
                                }
                                long long at_56 = tile_off + 12544ll;
                                if (add != 0) {
                                    float old_56[4];
                                    {
                                        unsigned _v4_56_0;
                                        unsigned _v4_56_1;
                                        unsigned _v4_56_2;
                                        unsigned _v4_56_3;
                                        asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                            : "=r"(_v4_56_0), "=r"(_v4_56_1), "=r"(_v4_56_2), "=r"(_v4_56_3) : "l"((const void*)(dq_accum + at_56 + (0))), "l"(0x14F0000000000000ULL) : "memory");
                                        old_56[0 + 0] = __uint_as_float(_v4_56_0);
                                        old_56[0 + 1] = __uint_as_float(_v4_56_1);
                                        old_56[0 + 2] = __uint_as_float(_v4_56_2);
                                        old_56[0 + 3] = __uint_as_float(_v4_56_3);
                                    }
                                    acc_7[0] = old_56[0] + acc_7[0];
                                    acc_7[1] = old_56[1] + acc_7[1];
                                    acc_7[2] = old_56[2] + acc_7[2];
                                    acc_7[3] = old_56[3] + acc_7[3];
                                }
                                asm volatile("st.global.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(dq_accum + at_56), "f"(acc_7[(0) + 0]), "f"(acc_7[(0) + 1]), "f"(acc_7[(0) + 2]), "f"(acc_7[(0) + 3]), "l"(0x14F0000000000000ULL) : "memory");
                                long long at_57 = tile_off + 13056ll;
                                if (add != 0) {
                                    float old_57[4];
                                    {
                                        unsigned _v4_57_0;
                                        unsigned _v4_57_1;
                                        unsigned _v4_57_2;
                                        unsigned _v4_57_3;
                                        asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                            : "=r"(_v4_57_0), "=r"(_v4_57_1), "=r"(_v4_57_2), "=r"(_v4_57_3) : "l"((const void*)(dq_accum + at_57 + (0))), "l"(0x14F0000000000000ULL) : "memory");
                                        old_57[0 + 0] = __uint_as_float(_v4_57_0);
                                        old_57[0 + 1] = __uint_as_float(_v4_57_1);
                                        old_57[0 + 2] = __uint_as_float(_v4_57_2);
                                        old_57[0 + 3] = __uint_as_float(_v4_57_3);
                                    }
                                    acc_7[4] = old_57[0] + acc_7[4];
                                    acc_7[5] = old_57[1] + acc_7[5];
                                    acc_7[6] = old_57[2] + acc_7[6];
                                    acc_7[7] = old_57[3] + acc_7[7];
                                }
                                asm volatile("st.global.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(dq_accum + at_57), "f"(acc_7[(4) + 0]), "f"(acc_7[(4) + 1]), "f"(acc_7[(4) + 2]), "f"(acc_7[(4) + 3]), "l"(0x14F0000000000000ULL) : "memory");
                                long long at_58 = tile_off + 13568ll;
                                if (add != 0) {
                                    float old_58[4];
                                    {
                                        unsigned _v4_58_0;
                                        unsigned _v4_58_1;
                                        unsigned _v4_58_2;
                                        unsigned _v4_58_3;
                                        asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                            : "=r"(_v4_58_0), "=r"(_v4_58_1), "=r"(_v4_58_2), "=r"(_v4_58_3) : "l"((const void*)(dq_accum + at_58 + (0))), "l"(0x14F0000000000000ULL) : "memory");
                                        old_58[0 + 0] = __uint_as_float(_v4_58_0);
                                        old_58[0 + 1] = __uint_as_float(_v4_58_1);
                                        old_58[0 + 2] = __uint_as_float(_v4_58_2);
                                        old_58[0 + 3] = __uint_as_float(_v4_58_3);
                                    }
                                    acc_7[8] = old_58[0] + acc_7[8];
                                    acc_7[9] = old_58[1] + acc_7[9];
                                    acc_7[10] = old_58[2] + acc_7[10];
                                    acc_7[11] = old_58[3] + acc_7[11];
                                }
                                asm volatile("st.global.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(dq_accum + at_58), "f"(acc_7[(8) + 0]), "f"(acc_7[(8) + 1]), "f"(acc_7[(8) + 2]), "f"(acc_7[(8) + 3]), "l"(0x14F0000000000000ULL) : "memory");
                                long long at_59 = tile_off + 14080ll;
                                if (add != 0) {
                                    float old_59[4];
                                    {
                                        unsigned _v4_59_0;
                                        unsigned _v4_59_1;
                                        unsigned _v4_59_2;
                                        unsigned _v4_59_3;
                                        asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                            : "=r"(_v4_59_0), "=r"(_v4_59_1), "=r"(_v4_59_2), "=r"(_v4_59_3) : "l"((const void*)(dq_accum + at_59 + (0))), "l"(0x14F0000000000000ULL) : "memory");
                                        old_59[0 + 0] = __uint_as_float(_v4_59_0);
                                        old_59[0 + 1] = __uint_as_float(_v4_59_1);
                                        old_59[0 + 2] = __uint_as_float(_v4_59_2);
                                        old_59[0 + 3] = __uint_as_float(_v4_59_3);
                                    }
                                    acc_7[12] = old_59[0] + acc_7[12];
                                    acc_7[13] = old_59[1] + acc_7[13];
                                    acc_7[14] = old_59[2] + acc_7[14];
                                    acc_7[15] = old_59[3] + acc_7[15];
                                }
                                asm volatile("st.global.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(dq_accum + at_59), "f"(acc_7[(12) + 0]), "f"(acc_7[(12) + 1]), "f"(acc_7[(12) + 2]), "f"(acc_7[(12) + 3]), "l"(0x14F0000000000000ULL) : "memory");
                                long long at_60 = tile_off + 14592ll;
                                if (add != 0) {
                                    float old_60[4];
                                    {
                                        unsigned _v4_60_0;
                                        unsigned _v4_60_1;
                                        unsigned _v4_60_2;
                                        unsigned _v4_60_3;
                                        asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                            : "=r"(_v4_60_0), "=r"(_v4_60_1), "=r"(_v4_60_2), "=r"(_v4_60_3) : "l"((const void*)(dq_accum + at_60 + (0))), "l"(0x14F0000000000000ULL) : "memory");
                                        old_60[0 + 0] = __uint_as_float(_v4_60_0);
                                        old_60[0 + 1] = __uint_as_float(_v4_60_1);
                                        old_60[0 + 2] = __uint_as_float(_v4_60_2);
                                        old_60[0 + 3] = __uint_as_float(_v4_60_3);
                                    }
                                    acc_7[16] = old_60[0] + acc_7[16];
                                    acc_7[17] = old_60[1] + acc_7[17];
                                    acc_7[18] = old_60[2] + acc_7[18];
                                    acc_7[19] = old_60[3] + acc_7[19];
                                }
                                asm volatile("st.global.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(dq_accum + at_60), "f"(acc_7[(16) + 0]), "f"(acc_7[(16) + 1]), "f"(acc_7[(16) + 2]), "f"(acc_7[(16) + 3]), "l"(0x14F0000000000000ULL) : "memory");
                                long long at_61 = tile_off + 15104ll;
                                if (add != 0) {
                                    float old_61[4];
                                    {
                                        unsigned _v4_61_0;
                                        unsigned _v4_61_1;
                                        unsigned _v4_61_2;
                                        unsigned _v4_61_3;
                                        asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                            : "=r"(_v4_61_0), "=r"(_v4_61_1), "=r"(_v4_61_2), "=r"(_v4_61_3) : "l"((const void*)(dq_accum + at_61 + (0))), "l"(0x14F0000000000000ULL) : "memory");
                                        old_61[0 + 0] = __uint_as_float(_v4_61_0);
                                        old_61[0 + 1] = __uint_as_float(_v4_61_1);
                                        old_61[0 + 2] = __uint_as_float(_v4_61_2);
                                        old_61[0 + 3] = __uint_as_float(_v4_61_3);
                                    }
                                    acc_7[20] = old_61[0] + acc_7[20];
                                    acc_7[21] = old_61[1] + acc_7[21];
                                    acc_7[22] = old_61[2] + acc_7[22];
                                    acc_7[23] = old_61[3] + acc_7[23];
                                }
                                asm volatile("st.global.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(dq_accum + at_61), "f"(acc_7[(20) + 0]), "f"(acc_7[(20) + 1]), "f"(acc_7[(20) + 2]), "f"(acc_7[(20) + 3]), "l"(0x14F0000000000000ULL) : "memory");
                                long long at_62 = tile_off + 15616ll;
                                if (add != 0) {
                                    float old_62[4];
                                    {
                                        unsigned _v4_62_0;
                                        unsigned _v4_62_1;
                                        unsigned _v4_62_2;
                                        unsigned _v4_62_3;
                                        asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                            : "=r"(_v4_62_0), "=r"(_v4_62_1), "=r"(_v4_62_2), "=r"(_v4_62_3) : "l"((const void*)(dq_accum + at_62 + (0))), "l"(0x14F0000000000000ULL) : "memory");
                                        old_62[0 + 0] = __uint_as_float(_v4_62_0);
                                        old_62[0 + 1] = __uint_as_float(_v4_62_1);
                                        old_62[0 + 2] = __uint_as_float(_v4_62_2);
                                        old_62[0 + 3] = __uint_as_float(_v4_62_3);
                                    }
                                    acc_7[24] = old_62[0] + acc_7[24];
                                    acc_7[25] = old_62[1] + acc_7[25];
                                    acc_7[26] = old_62[2] + acc_7[26];
                                    acc_7[27] = old_62[3] + acc_7[27];
                                }
                                asm volatile("st.global.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(dq_accum + at_62), "f"(acc_7[(24) + 0]), "f"(acc_7[(24) + 1]), "f"(acc_7[(24) + 2]), "f"(acc_7[(24) + 3]), "l"(0x14F0000000000000ULL) : "memory");
                                long long at_63 = tile_off + 16128ll;
                                if (add != 0) {
                                    float old_63[4];
                                    {
                                        unsigned _v4_63_0;
                                        unsigned _v4_63_1;
                                        unsigned _v4_63_2;
                                        unsigned _v4_63_3;
                                        asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                            : "=r"(_v4_63_0), "=r"(_v4_63_1), "=r"(_v4_63_2), "=r"(_v4_63_3) : "l"((const void*)(dq_accum + at_63 + (0))), "l"(0x14F0000000000000ULL) : "memory");
                                        old_63[0 + 0] = __uint_as_float(_v4_63_0);
                                        old_63[0 + 1] = __uint_as_float(_v4_63_1);
                                        old_63[0 + 2] = __uint_as_float(_v4_63_2);
                                        old_63[0 + 3] = __uint_as_float(_v4_63_3);
                                    }
                                    acc_7[28] = old_63[0] + acc_7[28];
                                    acc_7[29] = old_63[1] + acc_7[29];
                                    acc_7[30] = old_63[2] + acc_7[30];
                                    acc_7[31] = old_63[3] + acc_7[31];
                                }
                                asm volatile("st.global.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(dq_accum + at_63), "f"(acc_7[(28) + 0]), "f"(acc_7[(28) + 1]), "f"(acc_7[(28) + 2]), "f"(acc_7[(28) + 3]), "l"(0x14F0000000000000ULL) : "memory");
                            }
                        }
                        asm volatile("tcgen05.fence::before_thread_sync;");
                        __syncwarp();
                        if (elect_sync()) {
                            asm volatile(
                                "{\n\t"
                                ".reg .b32 remAddr32;\n\t"
                                "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                                "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                                "}"
                                :: "r"(acc_empty_addr + (ec) * 8), "r"(0) : "memory");
                        }
                        ec += 1;
                        if (ec == 2) { ec = 0; _phase_acc_full ^= 1; }
                    }
                }
            }
        }
    }
    // ---- Role: mma ----
    if (warp == 4) {
        { // mma_main
            int items_1 = pair_base[num_segments] * hc * (group >> 1);
            int rounds_1 = (items_1 + (int)num_clusters - 1) / (int)num_clusters;
            unsigned int mc = 0;
            unsigned int ac = 0;
            unsigned int _phase_acc_empty = 1;
            unsigned int _phase_tma_full = 0;
            if ((cta_rank & 1) == 0) {
                if (elect_sync()) {
                    #pragma unroll 1
                    for (int r_1 = 0; r_1 < rounds_1; r_1++) {
                        int cid_1 = (int)cluster_id;
                        int ncl_1 = (int)num_clusters;
                        int slot_1 = (((r_1 & 1) == 0) ? cid_1 : ncl_1 - 1 - cid_1);
                        int pos_1 = r_1 * ncl_1 + slot_1;
                        if (pos_1 < items_1) {
                            int n_groups_1 = group >> 1;
                            int heads_1 = hc * n_groups_1;
                            int pair_1 = pair_order[pos_1 / heads_1];
                            int hg_1 = pos_1 % heads_1;
                            int hl_1 = hg_1 / n_groups_1;
                            int g_1 = hg_1 % n_groups_1;
                            int lo_1 = 0;
                            int hi_1 = num_segments;
                            #pragma unroll 1
                            for (int __1 = 0; __1 < 31; __1++) {
                                if (hi_1 - lo_1 > 1) {
                                    int mid_1 = lo_1 + hi_1 >> 1;
                                    if (pair_base[mid_1] <= pair_1) {
                                        lo_1 = mid_1;
                                    } else {
                                        hi_1 = mid_1;
                                    }
                                }
                            }
                            int seg_1 = lo_1;
                            int rank_1 = cta_rank & 1;
                            int q0_1 = cu_seqlens_q[seg_1];
                            int q_len_1 = cu_seqlens_q[seg_1 + 1] - q0_1;
                            int k0_1 = cu_seqlens_k[seg_1];
                            int k_len_1 = cu_seqlens_k[seg_1 + 1] - k0_1;
                            int off_1 = k_len_1 - q_len_1;
                            int m_tiles_1 = q_len_1 + 63 >> 6;
                            int n_blocks_1 = k_len_1 + 127 >> 7;
                            int m_pair_1 = (pair_1 - pair_base[seg_1]) * 2;
                            int my_m_1 = m_pair_1 + rank_1;
                            int last_m_1 = ((m_pair_1 + 1 < m_tiles_1) ? m_pair_1 + 1 : m_tiles_1 - 1);
                            int n_end_1 = 64 * (last_m_1 + 1) + off_1 + 127 >> 7;
                            if (n_end_1 > n_blocks_1) {
                                n_end_1 = n_blocks_1;
                            }
                            if (n_end_1 > n1) {
                                n_end_1 = n1;
                            }
                            int steps_1 = n_end_1 - n0;
                            if (steps_1 < 0) {
                                steps_1 = 0;
                            }
                            int mine_1 = 64 * (my_m_1 + 1) + off_1 + 127 >> 7;
                            if (mine_1 > n_blocks_1) {
                                mine_1 = n_blocks_1;
                            }
                            if (my_m_1 >= m_tiles_1) {
                                mine_1 = 0;
                            }
                            int spans_1 = 1;
                            if (steps_1 > 0) {
                                spans_1 = (n0 + steps_1 - 1) / 64 - n0 / 64 + 1;
                            }
                            #pragma unroll 1
                            for (int j_1 = 0; j_1 < spans_1; j_1++) {
                                int n_lo = (n0 / 64 + j_1) * 64;
                                if (n_lo < n0) {
                                    n_lo = n0;
                                }
                                int n_hi = (n0 / 64 + j_1 + 1) * 64;
                                if (n_hi > n0 + steps_1) {
                                    n_hi = n0 + steps_1;
                                }
                                mbarrier_wait(acc_empty_addr + (ac) * 8, _phase_acc_empty);
                                asm volatile("tcgen05.fence::after_thread_sync;");
                                #pragma unroll 1
                                for (int n = n_lo; n < n_hi; n++) {
                                    int first = ((n == n_lo) ? 1 : 0);
                                    mbarrier_wait_cluster_hint(tma_full_addr + (mc) * 8, _phase_tma_full, 10000000);
                                    asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                                    int _smem_desc_lo_0 = ((((sa_addr) >> 4) & 0x7FFF) | 0x4000000) + (mc) * 4096;
                                    int _smem_desc_lo_1 = ((((sb_addr) >> 4) & 0x7FFF) | 0x4000000) + (mc) * 4096;
#else
                                    int _smem_desc_lo_0 = ((((sa_addr) >> 4) & 0x3FFF) | 0x4000000) + (mc) * 4096;
                                    int _smem_desc_lo_1 = ((((sb_addr) >> 4) & 0x3FFF) | 0x4000000) + (mc) * 4096;
#endif
                                    asm volatile(
                    "{\n\t"
                    ".reg .pred p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id, m0, m1, m2, m3, m4, m5, m6, m7;\n\t"
                    ".reg .b64 da, db;\n\t"
                    ""
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    "mov.b32 m0, 0; mov.b32 m1, 0; mov.b32 m2, 0; mov.b32 m3, 0;\n\tmov.b32 m4, 0; mov.b32 m5, 0; mov.b32 m6, 0; mov.b32 m7, 0;\n\t"
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 272729232;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"(_smem_desc_lo_0), "r"(_smem_desc_lo_1), "r"((tmem_acc + (ac * 256))), "r"(((first) ? 0 : 1)) : "memory");
                                    tcgen05_commit_cg2_multicast(mma_done_addr + (mc) * 8, (uint16_t)(3));
                                    mc += 1;
                                    if (mc == 3) { mc = 0; _phase_tma_full ^= 1; }
                                }
                                tcgen05_commit_cg2_multicast(acc_full_addr + (ac) * 8, (uint16_t)(3));
                                ac += 1;
                                if (ac == 2) { ac = 0; _phase_acc_empty ^= 1; }
                            }
                        }
                    }
                    mbarrier_wait(acc_empty_addr + (ac) * 8, _phase_acc_empty);
                    ac += 1;
                    if (ac == 2) { ac = 0; _phase_acc_empty ^= 1; }
                    mbarrier_wait(acc_empty_addr + (ac) * 8, _phase_acc_empty);
                    ac += 1;
                    if (ac == 2) { ac = 0; _phase_acc_empty ^= 1; }
                }
            }
        }
    }
    // ---- Role: load ----
    if (warp == 5) {
        { // load_main
            int items_2 = pair_base[num_segments] * hc * (group >> 1);
            int rounds_2 = (items_2 + (int)num_clusters - 1) / (int)num_clusters;
            unsigned int lc = 0;
            unsigned int _phase_mma_done = 1;
            if (elect_sync()) {
                #pragma unroll 1
                for (int r_2 = 0; r_2 < rounds_2; r_2++) {
                    int cid_2 = (int)cluster_id;
                    int ncl_2 = (int)num_clusters;
                    int slot_2 = (((r_2 & 1) == 0) ? cid_2 : ncl_2 - 1 - cid_2);
                    int pos_2 = r_2 * ncl_2 + slot_2;
                    if (pos_2 < items_2) {
                        int n_groups_2 = group >> 1;
                        int heads_2 = hc * n_groups_2;
                        int pair_2 = pair_order[pos_2 / heads_2];
                        int hg_2 = pos_2 % heads_2;
                        int hl_2 = hg_2 / n_groups_2;
                        int g_2 = hg_2 % n_groups_2;
                        int lo_2 = 0;
                        int hi_2 = num_segments;
                        #pragma unroll 1
                        for (int __2 = 0; __2 < 31; __2++) {
                            if (hi_2 - lo_2 > 1) {
                                int mid_2 = lo_2 + hi_2 >> 1;
                                if (pair_base[mid_2] <= pair_2) {
                                    lo_2 = mid_2;
                                } else {
                                    hi_2 = mid_2;
                                }
                            }
                        }
                        int seg_2 = lo_2;
                        int rank_2 = cta_rank & 1;
                        int q0_2 = cu_seqlens_q[seg_2];
                        int q_len_2 = cu_seqlens_q[seg_2 + 1] - q0_2;
                        int k0_2 = cu_seqlens_k[seg_2];
                        int k_len_2 = cu_seqlens_k[seg_2 + 1] - k0_2;
                        int off_2 = k_len_2 - q_len_2;
                        int m_tiles_2 = q_len_2 + 63 >> 6;
                        int n_blocks_2 = k_len_2 + 127 >> 7;
                        int m_pair_2 = (pair_2 - pair_base[seg_2]) * 2;
                        int my_m_2 = m_pair_2 + rank_2;
                        int last_m_2 = ((m_pair_2 + 1 < m_tiles_2) ? m_pair_2 + 1 : m_tiles_2 - 1);
                        int n_end_2 = 64 * (last_m_2 + 1) + off_2 + 127 >> 7;
                        if (n_end_2 > n_blocks_2) {
                            n_end_2 = n_blocks_2;
                        }
                        if (n_end_2 > n1) {
                            n_end_2 = n1;
                        }
                        int steps_2 = n_end_2 - n0;
                        if (steps_2 < 0) {
                            steps_2 = 0;
                        }
                        int mine_2 = 64 * (my_m_2 + 1) + off_2 + 127 >> 7;
                        if (mine_2 > n_blocks_2) {
                            mine_2 = n_blocks_2;
                        }
                        if (my_m_2 >= m_tiles_2) {
                            mine_2 = 0;
                        }
                        int spans_2 = 1;
                        if (steps_2 > 0) {
                            spans_2 = (n0 + steps_2 - 1) / 64 - n0 / 64 + 1;
                        }
                        int h_1 = h0 + hl_2;
                        int head_tile = (hl_2 * n_groups_2 + g_2) * tile_stride - tile_base[n0 * num_segments];
                        #pragma unroll 1
                        for (int s = 0; s < steps_2; s++) {
                            int n_1 = n0 + s;
                            int tile = zero_tile;
                            if (n_1 < mine_2) {
                                int first_key = n_1 * 128 - off_2;
                                int m_min = ((first_key > 0) ? first_key : 0) >> 6;
                                tile = head_tile + tile_base[n_1 * num_segments + seg_2] + (my_m_2 - m_min);
                            }
                            mbarrier_wait(mma_done_addr + (lc) * 8, _phase_mma_done);
                            asm volatile(
                                "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                                :: "r"((tma_full_addr + (lc) * 8) & 0xFEFFFFFF), "r"((uint32_t)(65536)) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::2.L2::cache_hint"
                                " [%0], [%1, {%2, %3}], [%4], %5;"
                                :: "r"(sa_addr + lc * 65536), "l"((&dsw)), "r"(0), "r"(tile * 256),
                                   "r"(((tma_full_addr + (lc) * 8) & 0xFEFFFFFF)), "l"(0x12F0000000000000ULL) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::2.L2::cache_hint"
                                " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                :: "r"(sb_addr + lc * 65536), "l"((&K)), "r"(rank_2 * 128), "r"(h_1), "r"(k0_2 + n_1 * 128),
                                   "r"(((tma_full_addr + (lc) * 8) & 0xFEFFFFFF)), "l"(0x14F0000000000000ULL) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::2.L2::cache_hint"
                                " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                :: "r"(sb_addr + lc * 65536 + 8192), "l"((&K)), "r"(rank_2 * 128), "r"(h_1), "r"(k0_2 + n_1 * 128 + 64),
                                   "r"(((tma_full_addr + (lc) * 8) & 0xFEFFFFFF)), "l"(0x14F0000000000000ULL) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::2.L2::cache_hint"
                                " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                :: "r"(sb_addr + lc * 65536 + 16384), "l"((&K)), "r"(rank_2 * 128 + 64), "r"(h_1), "r"(k0_2 + n_1 * 128),
                                   "r"(((tma_full_addr + (lc) * 8) & 0xFEFFFFFF)), "l"(0x14F0000000000000ULL) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::2.L2::cache_hint"
                                " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                :: "r"(sb_addr + lc * 65536 + 24576), "l"((&K)), "r"(rank_2 * 128 + 64), "r"(h_1), "r"(k0_2 + n_1 * 128 + 64),
                                   "r"(((tma_full_addr + (lc) * 8) & 0xFEFFFFFF)), "l"(0x14F0000000000000ULL) : "memory");
                            lc += 1;
                            if (lc == 3) { lc = 0; _phase_mma_done ^= 1; }
                        }
                    }
                }
                mbarrier_wait(mma_done_addr + (lc) * 8, _phase_mma_done);
                lc += 1;
                if (lc == 3) { lc = 0; _phase_mma_done ^= 1; }
                mbarrier_wait(mma_done_addr + (lc) * 8, _phase_mma_done);
                lc += 1;
                if (lc == 3) { lc = 0; _phase_mma_done ^= 1; }
                mbarrier_wait(mma_done_addr + (lc) * 8, _phase_mma_done);
                lc += 1;
                if (lc == 3) { lc = 0; _phase_mma_done ^= 1; }
            }
        }
    }

    // Cleanup
    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
    asm volatile("barrier.cluster.arrive.release.aligned;");
    asm volatile("barrier.cluster.wait.acquire.aligned;");

    if (warp == 4) {
        asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(512));
    }
}

} // extern "C"
