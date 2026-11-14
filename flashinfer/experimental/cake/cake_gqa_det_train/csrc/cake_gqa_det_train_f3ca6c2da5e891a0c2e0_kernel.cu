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







// ex2_emulation_f32x2 defined in softmax_frag_exp2_cast helper (or standalone)






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

__global__ __launch_bounds__(192, LAUNCH_MIN_BLOCKS) __cluster_dims__(2,1,1) void
kernel_cake_gqa_det_train_f3ca6c2da5e891a0c2e0(const __grid_constant__ CUtensorMap dsw, const __grid_constant__ CUtensorMap K, int* __restrict__ tile_base, int* __restrict__ pair_base, int* __restrict__ pair_order, int* __restrict__ cu_seqlens_q, int* __restrict__ cu_seqlens_k, float* __restrict__ dq_accum, __nv_bfloat16* __restrict__ dQ, int num_segments, int group, int rows_pad, int n0, int n1, int h0, int hc, int tile_stride, int zero_tile, int accumulate, int H_q, float softmax_scale, int finalize)
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
            float2 scale2 = make_float2(softmax_scale, softmax_scale);
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
                    int n_ex = n_end;
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
                    int fin = ((finalize == 2 || finalize == 1 && n_ex <= n1) ? 1 : 0);
                    int tok = my_m * 64 + (r_row >> 1);
                    int hq = h * group + 2 * g + (r_row & 1);
                    unsigned long long dq_row = ((unsigned long long)(q0 + tok) * (unsigned long long)H_q + (unsigned long long)hq) * 256ull;
                    #pragma unroll 1
                    for (int j = 0; j < spans; j++) {
                        int add = accumulate;
                        if (j > 0) {
                            add = 1;
                        }
                        int last = ((j == spans - 1) ? fin : 0);
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
                                if (last == 0) {
                                    asm volatile("st.global.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(dq_accum + at_0), "f"(acc_0[(0) + 0]), "f"(acc_0[(0) + 1]), "f"(acc_0[(0) + 2]), "f"(acc_0[(0) + 3]), "l"(0x14F0000000000000ULL) : "memory");
                                }
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
                                if (last == 0) {
                                    asm volatile("st.global.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(dq_accum + at_1), "f"(acc_0[(4) + 0]), "f"(acc_0[(4) + 1]), "f"(acc_0[(4) + 2]), "f"(acc_0[(4) + 3]), "l"(0x14F0000000000000ULL) : "memory");
                                }
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
                                if (last == 0) {
                                    asm volatile("st.global.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(dq_accum + at_2), "f"(acc_0[(8) + 0]), "f"(acc_0[(8) + 1]), "f"(acc_0[(8) + 2]), "f"(acc_0[(8) + 3]), "l"(0x14F0000000000000ULL) : "memory");
                                }
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
                                if (last == 0) {
                                    asm volatile("st.global.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(dq_accum + at_3), "f"(acc_0[(12) + 0]), "f"(acc_0[(12) + 1]), "f"(acc_0[(12) + 2]), "f"(acc_0[(12) + 3]), "l"(0x14F0000000000000ULL) : "memory");
                                }
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
                                if (last == 0) {
                                    asm volatile("st.global.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(dq_accum + at_4), "f"(acc_0[(16) + 0]), "f"(acc_0[(16) + 1]), "f"(acc_0[(16) + 2]), "f"(acc_0[(16) + 3]), "l"(0x14F0000000000000ULL) : "memory");
                                }
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
                                if (last == 0) {
                                    asm volatile("st.global.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(dq_accum + at_5), "f"(acc_0[(20) + 0]), "f"(acc_0[(20) + 1]), "f"(acc_0[(20) + 2]), "f"(acc_0[(20) + 3]), "l"(0x14F0000000000000ULL) : "memory");
                                }
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
                                if (last == 0) {
                                    asm volatile("st.global.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(dq_accum + at_6), "f"(acc_0[(24) + 0]), "f"(acc_0[(24) + 1]), "f"(acc_0[(24) + 2]), "f"(acc_0[(24) + 3]), "l"(0x14F0000000000000ULL) : "memory");
                                }
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
                                if (last == 0) {
                                    asm volatile("st.global.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(dq_accum + at_7), "f"(acc_0[(28) + 0]), "f"(acc_0[(28) + 1]), "f"(acc_0[(28) + 2]), "f"(acc_0[(28) + 3]), "l"(0x14F0000000000000ULL) : "memory");
                                }
                                if (last != 0) {
                                    unsigned int out_0[16];
                                    float2 _f32x2_mul_a_8 = make_float2(acc_0[0], acc_0[1]);
                                    float2 scaled_0;
                                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_0) : "l"(*(const unsigned long long*)&_f32x2_mul_a_8), "l"(*(const unsigned long long*)&scale2));
                                    out_0[0] = __as_u32(__float22bfloat162_rn(scaled_0));
                                    float2 _f32x2_mul_a_9 = make_float2(acc_0[2], acc_0[3]);
                                    float2 scaled_1;
                                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_9), "l"(*(const unsigned long long*)&scale2));
                                    out_0[1] = __as_u32(__float22bfloat162_rn(scaled_1));
                                    float2 _f32x2_mul_a_10 = make_float2(acc_0[4], acc_0[5]);
                                    float2 scaled_2;
                                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_2) : "l"(*(const unsigned long long*)&_f32x2_mul_a_10), "l"(*(const unsigned long long*)&scale2));
                                    out_0[2] = __as_u32(__float22bfloat162_rn(scaled_2));
                                    float2 _f32x2_mul_a_11 = make_float2(acc_0[6], acc_0[7]);
                                    float2 scaled_3;
                                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_3) : "l"(*(const unsigned long long*)&_f32x2_mul_a_11), "l"(*(const unsigned long long*)&scale2));
                                    out_0[3] = __as_u32(__float22bfloat162_rn(scaled_3));
                                    float2 _f32x2_mul_a_12 = make_float2(acc_0[8], acc_0[9]);
                                    float2 scaled_4;
                                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_4) : "l"(*(const unsigned long long*)&_f32x2_mul_a_12), "l"(*(const unsigned long long*)&scale2));
                                    out_0[4] = __as_u32(__float22bfloat162_rn(scaled_4));
                                    float2 _f32x2_mul_a_13 = make_float2(acc_0[10], acc_0[11]);
                                    float2 scaled_5;
                                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_5) : "l"(*(const unsigned long long*)&_f32x2_mul_a_13), "l"(*(const unsigned long long*)&scale2));
                                    out_0[5] = __as_u32(__float22bfloat162_rn(scaled_5));
                                    float2 _f32x2_mul_a_14 = make_float2(acc_0[12], acc_0[13]);
                                    float2 scaled_6;
                                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_6) : "l"(*(const unsigned long long*)&_f32x2_mul_a_14), "l"(*(const unsigned long long*)&scale2));
                                    out_0[6] = __as_u32(__float22bfloat162_rn(scaled_6));
                                    float2 _f32x2_mul_a_15 = make_float2(acc_0[14], acc_0[15]);
                                    float2 scaled_7;
                                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_7) : "l"(*(const unsigned long long*)&_f32x2_mul_a_15), "l"(*(const unsigned long long*)&scale2));
                                    out_0[7] = __as_u32(__float22bfloat162_rn(scaled_7));
                                    float2 _f32x2_mul_a_16 = make_float2(acc_0[16], acc_0[17]);
                                    float2 scaled_8;
                                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_8) : "l"(*(const unsigned long long*)&_f32x2_mul_a_16), "l"(*(const unsigned long long*)&scale2));
                                    out_0[8] = __as_u32(__float22bfloat162_rn(scaled_8));
                                    float2 _f32x2_mul_a_17 = make_float2(acc_0[18], acc_0[19]);
                                    float2 scaled_9;
                                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_9) : "l"(*(const unsigned long long*)&_f32x2_mul_a_17), "l"(*(const unsigned long long*)&scale2));
                                    out_0[9] = __as_u32(__float22bfloat162_rn(scaled_9));
                                    float2 _f32x2_mul_a_18 = make_float2(acc_0[20], acc_0[21]);
                                    float2 scaled_10;
                                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_10) : "l"(*(const unsigned long long*)&_f32x2_mul_a_18), "l"(*(const unsigned long long*)&scale2));
                                    out_0[10] = __as_u32(__float22bfloat162_rn(scaled_10));
                                    float2 _f32x2_mul_a_19 = make_float2(acc_0[22], acc_0[23]);
                                    float2 scaled_11;
                                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_11) : "l"(*(const unsigned long long*)&_f32x2_mul_a_19), "l"(*(const unsigned long long*)&scale2));
                                    out_0[11] = __as_u32(__float22bfloat162_rn(scaled_11));
                                    float2 _f32x2_mul_a_20 = make_float2(acc_0[24], acc_0[25]);
                                    float2 scaled_12;
                                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_12) : "l"(*(const unsigned long long*)&_f32x2_mul_a_20), "l"(*(const unsigned long long*)&scale2));
                                    out_0[12] = __as_u32(__float22bfloat162_rn(scaled_12));
                                    float2 _f32x2_mul_a_21 = make_float2(acc_0[26], acc_0[27]);
                                    float2 scaled_13;
                                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_13) : "l"(*(const unsigned long long*)&_f32x2_mul_a_21), "l"(*(const unsigned long long*)&scale2));
                                    out_0[13] = __as_u32(__float22bfloat162_rn(scaled_13));
                                    float2 _f32x2_mul_a_22 = make_float2(acc_0[28], acc_0[29]);
                                    float2 scaled_14;
                                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_14) : "l"(*(const unsigned long long*)&_f32x2_mul_a_22), "l"(*(const unsigned long long*)&scale2));
                                    out_0[14] = __as_u32(__float22bfloat162_rn(scaled_14));
                                    float2 _f32x2_mul_a_23 = make_float2(acc_0[30], acc_0[31]);
                                    float2 scaled_15;
                                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_15) : "l"(*(const unsigned long long*)&_f32x2_mul_a_23), "l"(*(const unsigned long long*)&scale2));
                                    out_0[15] = __as_u32(__float22bfloat162_rn(scaled_15));
                                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %9, 0; @p_ st.global.L2::evict_first.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8}; }" :: "l"(reinterpret_cast<unsigned int*>(dQ + (dq_row + 0ull))), "r"(out_0[(0) + 0]), "r"(out_0[(0) + 1]), "r"(out_0[(0) + 2]), "r"(out_0[(0) + 3]), "r"(out_0[(0) + 4]), "r"(out_0[(0) + 5]), "r"(out_0[(0) + 6]), "r"(out_0[(0) + 7]), "r"((unsigned int)(tok < q_len)) : "memory");
                                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %9, 0; @p_ st.global.L2::evict_first.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8}; }" :: "l"(reinterpret_cast<unsigned int*>(dQ + (dq_row + 16ull))), "r"(out_0[(8) + 0]), "r"(out_0[(8) + 1]), "r"(out_0[(8) + 2]), "r"(out_0[(8) + 3]), "r"(out_0[(8) + 4]), "r"(out_0[(8) + 5]), "r"(out_0[(8) + 6]), "r"(out_0[(8) + 7]), "r"((unsigned int)(tok < q_len)) : "memory");
                                }
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
                                        unsigned _v4_24_0;
                                        unsigned _v4_24_1;
                                        unsigned _v4_24_2;
                                        unsigned _v4_24_3;
                                        asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                            : "=r"(_v4_24_0), "=r"(_v4_24_1), "=r"(_v4_24_2), "=r"(_v4_24_3) : "l"((const void*)(dq_accum + at_8 + (0))), "l"(0x14F0000000000000ULL) : "memory");
                                        old_8[0 + 0] = __uint_as_float(_v4_24_0);
                                        old_8[0 + 1] = __uint_as_float(_v4_24_1);
                                        old_8[0 + 2] = __uint_as_float(_v4_24_2);
                                        old_8[0 + 3] = __uint_as_float(_v4_24_3);
                                    }
                                    acc_1[0] = old_8[0] + acc_1[0];
                                    acc_1[1] = old_8[1] + acc_1[1];
                                    acc_1[2] = old_8[2] + acc_1[2];
                                    acc_1[3] = old_8[3] + acc_1[3];
                                }
                                if (last == 0) {
                                    asm volatile("st.global.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(dq_accum + at_8), "f"(acc_1[(0) + 0]), "f"(acc_1[(0) + 1]), "f"(acc_1[(0) + 2]), "f"(acc_1[(0) + 3]), "l"(0x14F0000000000000ULL) : "memory");
                                }
                                long long at_9 = tile_off + 4608ll;
                                if (add != 0) {
                                    float old_9[4];
                                    {
                                        unsigned _v4_25_0;
                                        unsigned _v4_25_1;
                                        unsigned _v4_25_2;
                                        unsigned _v4_25_3;
                                        asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                            : "=r"(_v4_25_0), "=r"(_v4_25_1), "=r"(_v4_25_2), "=r"(_v4_25_3) : "l"((const void*)(dq_accum + at_9 + (0))), "l"(0x14F0000000000000ULL) : "memory");
                                        old_9[0 + 0] = __uint_as_float(_v4_25_0);
                                        old_9[0 + 1] = __uint_as_float(_v4_25_1);
                                        old_9[0 + 2] = __uint_as_float(_v4_25_2);
                                        old_9[0 + 3] = __uint_as_float(_v4_25_3);
                                    }
                                    acc_1[4] = old_9[0] + acc_1[4];
                                    acc_1[5] = old_9[1] + acc_1[5];
                                    acc_1[6] = old_9[2] + acc_1[6];
                                    acc_1[7] = old_9[3] + acc_1[7];
                                }
                                if (last == 0) {
                                    asm volatile("st.global.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(dq_accum + at_9), "f"(acc_1[(4) + 0]), "f"(acc_1[(4) + 1]), "f"(acc_1[(4) + 2]), "f"(acc_1[(4) + 3]), "l"(0x14F0000000000000ULL) : "memory");
                                }
                                long long at_10 = tile_off + 5120ll;
                                if (add != 0) {
                                    float old_10[4];
                                    {
                                        unsigned _v4_26_0;
                                        unsigned _v4_26_1;
                                        unsigned _v4_26_2;
                                        unsigned _v4_26_3;
                                        asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                            : "=r"(_v4_26_0), "=r"(_v4_26_1), "=r"(_v4_26_2), "=r"(_v4_26_3) : "l"((const void*)(dq_accum + at_10 + (0))), "l"(0x14F0000000000000ULL) : "memory");
                                        old_10[0 + 0] = __uint_as_float(_v4_26_0);
                                        old_10[0 + 1] = __uint_as_float(_v4_26_1);
                                        old_10[0 + 2] = __uint_as_float(_v4_26_2);
                                        old_10[0 + 3] = __uint_as_float(_v4_26_3);
                                    }
                                    acc_1[8] = old_10[0] + acc_1[8];
                                    acc_1[9] = old_10[1] + acc_1[9];
                                    acc_1[10] = old_10[2] + acc_1[10];
                                    acc_1[11] = old_10[3] + acc_1[11];
                                }
                                if (last == 0) {
                                    asm volatile("st.global.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(dq_accum + at_10), "f"(acc_1[(8) + 0]), "f"(acc_1[(8) + 1]), "f"(acc_1[(8) + 2]), "f"(acc_1[(8) + 3]), "l"(0x14F0000000000000ULL) : "memory");
                                }
                                long long at_11 = tile_off + 5632ll;
                                if (add != 0) {
                                    float old_11[4];
                                    {
                                        unsigned _v4_27_0;
                                        unsigned _v4_27_1;
                                        unsigned _v4_27_2;
                                        unsigned _v4_27_3;
                                        asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                            : "=r"(_v4_27_0), "=r"(_v4_27_1), "=r"(_v4_27_2), "=r"(_v4_27_3) : "l"((const void*)(dq_accum + at_11 + (0))), "l"(0x14F0000000000000ULL) : "memory");
                                        old_11[0 + 0] = __uint_as_float(_v4_27_0);
                                        old_11[0 + 1] = __uint_as_float(_v4_27_1);
                                        old_11[0 + 2] = __uint_as_float(_v4_27_2);
                                        old_11[0 + 3] = __uint_as_float(_v4_27_3);
                                    }
                                    acc_1[12] = old_11[0] + acc_1[12];
                                    acc_1[13] = old_11[1] + acc_1[13];
                                    acc_1[14] = old_11[2] + acc_1[14];
                                    acc_1[15] = old_11[3] + acc_1[15];
                                }
                                if (last == 0) {
                                    asm volatile("st.global.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(dq_accum + at_11), "f"(acc_1[(12) + 0]), "f"(acc_1[(12) + 1]), "f"(acc_1[(12) + 2]), "f"(acc_1[(12) + 3]), "l"(0x14F0000000000000ULL) : "memory");
                                }
                                long long at_12 = tile_off + 6144ll;
                                if (add != 0) {
                                    float old_12[4];
                                    {
                                        unsigned _v4_28_0;
                                        unsigned _v4_28_1;
                                        unsigned _v4_28_2;
                                        unsigned _v4_28_3;
                                        asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                            : "=r"(_v4_28_0), "=r"(_v4_28_1), "=r"(_v4_28_2), "=r"(_v4_28_3) : "l"((const void*)(dq_accum + at_12 + (0))), "l"(0x14F0000000000000ULL) : "memory");
                                        old_12[0 + 0] = __uint_as_float(_v4_28_0);
                                        old_12[0 + 1] = __uint_as_float(_v4_28_1);
                                        old_12[0 + 2] = __uint_as_float(_v4_28_2);
                                        old_12[0 + 3] = __uint_as_float(_v4_28_3);
                                    }
                                    acc_1[16] = old_12[0] + acc_1[16];
                                    acc_1[17] = old_12[1] + acc_1[17];
                                    acc_1[18] = old_12[2] + acc_1[18];
                                    acc_1[19] = old_12[3] + acc_1[19];
                                }
                                if (last == 0) {
                                    asm volatile("st.global.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(dq_accum + at_12), "f"(acc_1[(16) + 0]), "f"(acc_1[(16) + 1]), "f"(acc_1[(16) + 2]), "f"(acc_1[(16) + 3]), "l"(0x14F0000000000000ULL) : "memory");
                                }
                                long long at_13 = tile_off + 6656ll;
                                if (add != 0) {
                                    float old_13[4];
                                    {
                                        unsigned _v4_29_0;
                                        unsigned _v4_29_1;
                                        unsigned _v4_29_2;
                                        unsigned _v4_29_3;
                                        asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                            : "=r"(_v4_29_0), "=r"(_v4_29_1), "=r"(_v4_29_2), "=r"(_v4_29_3) : "l"((const void*)(dq_accum + at_13 + (0))), "l"(0x14F0000000000000ULL) : "memory");
                                        old_13[0 + 0] = __uint_as_float(_v4_29_0);
                                        old_13[0 + 1] = __uint_as_float(_v4_29_1);
                                        old_13[0 + 2] = __uint_as_float(_v4_29_2);
                                        old_13[0 + 3] = __uint_as_float(_v4_29_3);
                                    }
                                    acc_1[20] = old_13[0] + acc_1[20];
                                    acc_1[21] = old_13[1] + acc_1[21];
                                    acc_1[22] = old_13[2] + acc_1[22];
                                    acc_1[23] = old_13[3] + acc_1[23];
                                }
                                if (last == 0) {
                                    asm volatile("st.global.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(dq_accum + at_13), "f"(acc_1[(20) + 0]), "f"(acc_1[(20) + 1]), "f"(acc_1[(20) + 2]), "f"(acc_1[(20) + 3]), "l"(0x14F0000000000000ULL) : "memory");
                                }
                                long long at_14 = tile_off + 7168ll;
                                if (add != 0) {
                                    float old_14[4];
                                    {
                                        unsigned _v4_30_0;
                                        unsigned _v4_30_1;
                                        unsigned _v4_30_2;
                                        unsigned _v4_30_3;
                                        asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                            : "=r"(_v4_30_0), "=r"(_v4_30_1), "=r"(_v4_30_2), "=r"(_v4_30_3) : "l"((const void*)(dq_accum + at_14 + (0))), "l"(0x14F0000000000000ULL) : "memory");
                                        old_14[0 + 0] = __uint_as_float(_v4_30_0);
                                        old_14[0 + 1] = __uint_as_float(_v4_30_1);
                                        old_14[0 + 2] = __uint_as_float(_v4_30_2);
                                        old_14[0 + 3] = __uint_as_float(_v4_30_3);
                                    }
                                    acc_1[24] = old_14[0] + acc_1[24];
                                    acc_1[25] = old_14[1] + acc_1[25];
                                    acc_1[26] = old_14[2] + acc_1[26];
                                    acc_1[27] = old_14[3] + acc_1[27];
                                }
                                if (last == 0) {
                                    asm volatile("st.global.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(dq_accum + at_14), "f"(acc_1[(24) + 0]), "f"(acc_1[(24) + 1]), "f"(acc_1[(24) + 2]), "f"(acc_1[(24) + 3]), "l"(0x14F0000000000000ULL) : "memory");
                                }
                                long long at_15 = tile_off + 7680ll;
                                if (add != 0) {
                                    float old_15[4];
                                    {
                                        unsigned _v4_31_0;
                                        unsigned _v4_31_1;
                                        unsigned _v4_31_2;
                                        unsigned _v4_31_3;
                                        asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                            : "=r"(_v4_31_0), "=r"(_v4_31_1), "=r"(_v4_31_2), "=r"(_v4_31_3) : "l"((const void*)(dq_accum + at_15 + (0))), "l"(0x14F0000000000000ULL) : "memory");
                                        old_15[0 + 0] = __uint_as_float(_v4_31_0);
                                        old_15[0 + 1] = __uint_as_float(_v4_31_1);
                                        old_15[0 + 2] = __uint_as_float(_v4_31_2);
                                        old_15[0 + 3] = __uint_as_float(_v4_31_3);
                                    }
                                    acc_1[28] = old_15[0] + acc_1[28];
                                    acc_1[29] = old_15[1] + acc_1[29];
                                    acc_1[30] = old_15[2] + acc_1[30];
                                    acc_1[31] = old_15[3] + acc_1[31];
                                }
                                if (last == 0) {
                                    asm volatile("st.global.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(dq_accum + at_15), "f"(acc_1[(28) + 0]), "f"(acc_1[(28) + 1]), "f"(acc_1[(28) + 2]), "f"(acc_1[(28) + 3]), "l"(0x14F0000000000000ULL) : "memory");
                                }
                                if (last != 0) {
                                    unsigned int out_1[16];
                                    float2 _f32x2_mul_a_32 = make_float2(acc_1[0], acc_1[1]);
                                    float2 scaled_16;
                                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_16) : "l"(*(const unsigned long long*)&_f32x2_mul_a_32), "l"(*(const unsigned long long*)&scale2));
                                    out_1[0] = __as_u32(__float22bfloat162_rn(scaled_16));
                                    float2 _f32x2_mul_a_33 = make_float2(acc_1[2], acc_1[3]);
                                    float2 scaled_17;
                                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_17) : "l"(*(const unsigned long long*)&_f32x2_mul_a_33), "l"(*(const unsigned long long*)&scale2));
                                    out_1[1] = __as_u32(__float22bfloat162_rn(scaled_17));
                                    float2 _f32x2_mul_a_34 = make_float2(acc_1[4], acc_1[5]);
                                    float2 scaled_18;
                                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_18) : "l"(*(const unsigned long long*)&_f32x2_mul_a_34), "l"(*(const unsigned long long*)&scale2));
                                    out_1[2] = __as_u32(__float22bfloat162_rn(scaled_18));
                                    float2 _f32x2_mul_a_35 = make_float2(acc_1[6], acc_1[7]);
                                    float2 scaled_19;
                                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_19) : "l"(*(const unsigned long long*)&_f32x2_mul_a_35), "l"(*(const unsigned long long*)&scale2));
                                    out_1[3] = __as_u32(__float22bfloat162_rn(scaled_19));
                                    float2 _f32x2_mul_a_36 = make_float2(acc_1[8], acc_1[9]);
                                    float2 scaled_20;
                                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_20) : "l"(*(const unsigned long long*)&_f32x2_mul_a_36), "l"(*(const unsigned long long*)&scale2));
                                    out_1[4] = __as_u32(__float22bfloat162_rn(scaled_20));
                                    float2 _f32x2_mul_a_37 = make_float2(acc_1[10], acc_1[11]);
                                    float2 scaled_21;
                                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_21) : "l"(*(const unsigned long long*)&_f32x2_mul_a_37), "l"(*(const unsigned long long*)&scale2));
                                    out_1[5] = __as_u32(__float22bfloat162_rn(scaled_21));
                                    float2 _f32x2_mul_a_38 = make_float2(acc_1[12], acc_1[13]);
                                    float2 scaled_22;
                                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_22) : "l"(*(const unsigned long long*)&_f32x2_mul_a_38), "l"(*(const unsigned long long*)&scale2));
                                    out_1[6] = __as_u32(__float22bfloat162_rn(scaled_22));
                                    float2 _f32x2_mul_a_39 = make_float2(acc_1[14], acc_1[15]);
                                    float2 scaled_23;
                                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_23) : "l"(*(const unsigned long long*)&_f32x2_mul_a_39), "l"(*(const unsigned long long*)&scale2));
                                    out_1[7] = __as_u32(__float22bfloat162_rn(scaled_23));
                                    float2 _f32x2_mul_a_40 = make_float2(acc_1[16], acc_1[17]);
                                    float2 scaled_24;
                                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_24) : "l"(*(const unsigned long long*)&_f32x2_mul_a_40), "l"(*(const unsigned long long*)&scale2));
                                    out_1[8] = __as_u32(__float22bfloat162_rn(scaled_24));
                                    float2 _f32x2_mul_a_41 = make_float2(acc_1[18], acc_1[19]);
                                    float2 scaled_25;
                                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_25) : "l"(*(const unsigned long long*)&_f32x2_mul_a_41), "l"(*(const unsigned long long*)&scale2));
                                    out_1[9] = __as_u32(__float22bfloat162_rn(scaled_25));
                                    float2 _f32x2_mul_a_42 = make_float2(acc_1[20], acc_1[21]);
                                    float2 scaled_26;
                                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_26) : "l"(*(const unsigned long long*)&_f32x2_mul_a_42), "l"(*(const unsigned long long*)&scale2));
                                    out_1[10] = __as_u32(__float22bfloat162_rn(scaled_26));
                                    float2 _f32x2_mul_a_43 = make_float2(acc_1[22], acc_1[23]);
                                    float2 scaled_27;
                                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_27) : "l"(*(const unsigned long long*)&_f32x2_mul_a_43), "l"(*(const unsigned long long*)&scale2));
                                    out_1[11] = __as_u32(__float22bfloat162_rn(scaled_27));
                                    float2 _f32x2_mul_a_44 = make_float2(acc_1[24], acc_1[25]);
                                    float2 scaled_28;
                                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_28) : "l"(*(const unsigned long long*)&_f32x2_mul_a_44), "l"(*(const unsigned long long*)&scale2));
                                    out_1[12] = __as_u32(__float22bfloat162_rn(scaled_28));
                                    float2 _f32x2_mul_a_45 = make_float2(acc_1[26], acc_1[27]);
                                    float2 scaled_29;
                                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_29) : "l"(*(const unsigned long long*)&_f32x2_mul_a_45), "l"(*(const unsigned long long*)&scale2));
                                    out_1[13] = __as_u32(__float22bfloat162_rn(scaled_29));
                                    float2 _f32x2_mul_a_46 = make_float2(acc_1[28], acc_1[29]);
                                    float2 scaled_30;
                                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_30) : "l"(*(const unsigned long long*)&_f32x2_mul_a_46), "l"(*(const unsigned long long*)&scale2));
                                    out_1[14] = __as_u32(__float22bfloat162_rn(scaled_30));
                                    float2 _f32x2_mul_a_47 = make_float2(acc_1[30], acc_1[31]);
                                    float2 scaled_31;
                                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_31) : "l"(*(const unsigned long long*)&_f32x2_mul_a_47), "l"(*(const unsigned long long*)&scale2));
                                    out_1[15] = __as_u32(__float22bfloat162_rn(scaled_31));
                                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %9, 0; @p_ st.global.L2::evict_first.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8}; }" :: "l"(reinterpret_cast<unsigned int*>(dQ + (dq_row + 32ull))), "r"(out_1[(0) + 0]), "r"(out_1[(0) + 1]), "r"(out_1[(0) + 2]), "r"(out_1[(0) + 3]), "r"(out_1[(0) + 4]), "r"(out_1[(0) + 5]), "r"(out_1[(0) + 6]), "r"(out_1[(0) + 7]), "r"((unsigned int)(tok < q_len)) : "memory");
                                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %9, 0; @p_ st.global.L2::evict_first.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8}; }" :: "l"(reinterpret_cast<unsigned int*>(dQ + (dq_row + 48ull))), "r"(out_1[(8) + 0]), "r"(out_1[(8) + 1]), "r"(out_1[(8) + 2]), "r"(out_1[(8) + 3]), "r"(out_1[(8) + 4]), "r"(out_1[(8) + 5]), "r"(out_1[(8) + 6]), "r"(out_1[(8) + 7]), "r"((unsigned int)(tok < q_len)) : "memory");
                                }
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
                                        unsigned _v4_48_0;
                                        unsigned _v4_48_1;
                                        unsigned _v4_48_2;
                                        unsigned _v4_48_3;
                                        asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                            : "=r"(_v4_48_0), "=r"(_v4_48_1), "=r"(_v4_48_2), "=r"(_v4_48_3) : "l"((const void*)(dq_accum + at_16 + (0))), "l"(0x14F0000000000000ULL) : "memory");
                                        old_16[0 + 0] = __uint_as_float(_v4_48_0);
                                        old_16[0 + 1] = __uint_as_float(_v4_48_1);
                                        old_16[0 + 2] = __uint_as_float(_v4_48_2);
                                        old_16[0 + 3] = __uint_as_float(_v4_48_3);
                                    }
                                    acc_2[0] = old_16[0] + acc_2[0];
                                    acc_2[1] = old_16[1] + acc_2[1];
                                    acc_2[2] = old_16[2] + acc_2[2];
                                    acc_2[3] = old_16[3] + acc_2[3];
                                }
                                if (last == 0) {
                                    asm volatile("st.global.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(dq_accum + at_16), "f"(acc_2[(0) + 0]), "f"(acc_2[(0) + 1]), "f"(acc_2[(0) + 2]), "f"(acc_2[(0) + 3]), "l"(0x14F0000000000000ULL) : "memory");
                                }
                                long long at_17 = tile_off + 8704ll;
                                if (add != 0) {
                                    float old_17[4];
                                    {
                                        unsigned _v4_49_0;
                                        unsigned _v4_49_1;
                                        unsigned _v4_49_2;
                                        unsigned _v4_49_3;
                                        asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                            : "=r"(_v4_49_0), "=r"(_v4_49_1), "=r"(_v4_49_2), "=r"(_v4_49_3) : "l"((const void*)(dq_accum + at_17 + (0))), "l"(0x14F0000000000000ULL) : "memory");
                                        old_17[0 + 0] = __uint_as_float(_v4_49_0);
                                        old_17[0 + 1] = __uint_as_float(_v4_49_1);
                                        old_17[0 + 2] = __uint_as_float(_v4_49_2);
                                        old_17[0 + 3] = __uint_as_float(_v4_49_3);
                                    }
                                    acc_2[4] = old_17[0] + acc_2[4];
                                    acc_2[5] = old_17[1] + acc_2[5];
                                    acc_2[6] = old_17[2] + acc_2[6];
                                    acc_2[7] = old_17[3] + acc_2[7];
                                }
                                if (last == 0) {
                                    asm volatile("st.global.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(dq_accum + at_17), "f"(acc_2[(4) + 0]), "f"(acc_2[(4) + 1]), "f"(acc_2[(4) + 2]), "f"(acc_2[(4) + 3]), "l"(0x14F0000000000000ULL) : "memory");
                                }
                                long long at_18 = tile_off + 9216ll;
                                if (add != 0) {
                                    float old_18[4];
                                    {
                                        unsigned _v4_50_0;
                                        unsigned _v4_50_1;
                                        unsigned _v4_50_2;
                                        unsigned _v4_50_3;
                                        asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                            : "=r"(_v4_50_0), "=r"(_v4_50_1), "=r"(_v4_50_2), "=r"(_v4_50_3) : "l"((const void*)(dq_accum + at_18 + (0))), "l"(0x14F0000000000000ULL) : "memory");
                                        old_18[0 + 0] = __uint_as_float(_v4_50_0);
                                        old_18[0 + 1] = __uint_as_float(_v4_50_1);
                                        old_18[0 + 2] = __uint_as_float(_v4_50_2);
                                        old_18[0 + 3] = __uint_as_float(_v4_50_3);
                                    }
                                    acc_2[8] = old_18[0] + acc_2[8];
                                    acc_2[9] = old_18[1] + acc_2[9];
                                    acc_2[10] = old_18[2] + acc_2[10];
                                    acc_2[11] = old_18[3] + acc_2[11];
                                }
                                if (last == 0) {
                                    asm volatile("st.global.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(dq_accum + at_18), "f"(acc_2[(8) + 0]), "f"(acc_2[(8) + 1]), "f"(acc_2[(8) + 2]), "f"(acc_2[(8) + 3]), "l"(0x14F0000000000000ULL) : "memory");
                                }
                                long long at_19 = tile_off + 9728ll;
                                if (add != 0) {
                                    float old_19[4];
                                    {
                                        unsigned _v4_51_0;
                                        unsigned _v4_51_1;
                                        unsigned _v4_51_2;
                                        unsigned _v4_51_3;
                                        asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                            : "=r"(_v4_51_0), "=r"(_v4_51_1), "=r"(_v4_51_2), "=r"(_v4_51_3) : "l"((const void*)(dq_accum + at_19 + (0))), "l"(0x14F0000000000000ULL) : "memory");
                                        old_19[0 + 0] = __uint_as_float(_v4_51_0);
                                        old_19[0 + 1] = __uint_as_float(_v4_51_1);
                                        old_19[0 + 2] = __uint_as_float(_v4_51_2);
                                        old_19[0 + 3] = __uint_as_float(_v4_51_3);
                                    }
                                    acc_2[12] = old_19[0] + acc_2[12];
                                    acc_2[13] = old_19[1] + acc_2[13];
                                    acc_2[14] = old_19[2] + acc_2[14];
                                    acc_2[15] = old_19[3] + acc_2[15];
                                }
                                if (last == 0) {
                                    asm volatile("st.global.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(dq_accum + at_19), "f"(acc_2[(12) + 0]), "f"(acc_2[(12) + 1]), "f"(acc_2[(12) + 2]), "f"(acc_2[(12) + 3]), "l"(0x14F0000000000000ULL) : "memory");
                                }
                                long long at_20 = tile_off + 10240ll;
                                if (add != 0) {
                                    float old_20[4];
                                    {
                                        unsigned _v4_52_0;
                                        unsigned _v4_52_1;
                                        unsigned _v4_52_2;
                                        unsigned _v4_52_3;
                                        asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                            : "=r"(_v4_52_0), "=r"(_v4_52_1), "=r"(_v4_52_2), "=r"(_v4_52_3) : "l"((const void*)(dq_accum + at_20 + (0))), "l"(0x14F0000000000000ULL) : "memory");
                                        old_20[0 + 0] = __uint_as_float(_v4_52_0);
                                        old_20[0 + 1] = __uint_as_float(_v4_52_1);
                                        old_20[0 + 2] = __uint_as_float(_v4_52_2);
                                        old_20[0 + 3] = __uint_as_float(_v4_52_3);
                                    }
                                    acc_2[16] = old_20[0] + acc_2[16];
                                    acc_2[17] = old_20[1] + acc_2[17];
                                    acc_2[18] = old_20[2] + acc_2[18];
                                    acc_2[19] = old_20[3] + acc_2[19];
                                }
                                if (last == 0) {
                                    asm volatile("st.global.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(dq_accum + at_20), "f"(acc_2[(16) + 0]), "f"(acc_2[(16) + 1]), "f"(acc_2[(16) + 2]), "f"(acc_2[(16) + 3]), "l"(0x14F0000000000000ULL) : "memory");
                                }
                                long long at_21 = tile_off + 10752ll;
                                if (add != 0) {
                                    float old_21[4];
                                    {
                                        unsigned _v4_53_0;
                                        unsigned _v4_53_1;
                                        unsigned _v4_53_2;
                                        unsigned _v4_53_3;
                                        asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                            : "=r"(_v4_53_0), "=r"(_v4_53_1), "=r"(_v4_53_2), "=r"(_v4_53_3) : "l"((const void*)(dq_accum + at_21 + (0))), "l"(0x14F0000000000000ULL) : "memory");
                                        old_21[0 + 0] = __uint_as_float(_v4_53_0);
                                        old_21[0 + 1] = __uint_as_float(_v4_53_1);
                                        old_21[0 + 2] = __uint_as_float(_v4_53_2);
                                        old_21[0 + 3] = __uint_as_float(_v4_53_3);
                                    }
                                    acc_2[20] = old_21[0] + acc_2[20];
                                    acc_2[21] = old_21[1] + acc_2[21];
                                    acc_2[22] = old_21[2] + acc_2[22];
                                    acc_2[23] = old_21[3] + acc_2[23];
                                }
                                if (last == 0) {
                                    asm volatile("st.global.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(dq_accum + at_21), "f"(acc_2[(20) + 0]), "f"(acc_2[(20) + 1]), "f"(acc_2[(20) + 2]), "f"(acc_2[(20) + 3]), "l"(0x14F0000000000000ULL) : "memory");
                                }
                                long long at_22 = tile_off + 11264ll;
                                if (add != 0) {
                                    float old_22[4];
                                    {
                                        unsigned _v4_54_0;
                                        unsigned _v4_54_1;
                                        unsigned _v4_54_2;
                                        unsigned _v4_54_3;
                                        asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                            : "=r"(_v4_54_0), "=r"(_v4_54_1), "=r"(_v4_54_2), "=r"(_v4_54_3) : "l"((const void*)(dq_accum + at_22 + (0))), "l"(0x14F0000000000000ULL) : "memory");
                                        old_22[0 + 0] = __uint_as_float(_v4_54_0);
                                        old_22[0 + 1] = __uint_as_float(_v4_54_1);
                                        old_22[0 + 2] = __uint_as_float(_v4_54_2);
                                        old_22[0 + 3] = __uint_as_float(_v4_54_3);
                                    }
                                    acc_2[24] = old_22[0] + acc_2[24];
                                    acc_2[25] = old_22[1] + acc_2[25];
                                    acc_2[26] = old_22[2] + acc_2[26];
                                    acc_2[27] = old_22[3] + acc_2[27];
                                }
                                if (last == 0) {
                                    asm volatile("st.global.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(dq_accum + at_22), "f"(acc_2[(24) + 0]), "f"(acc_2[(24) + 1]), "f"(acc_2[(24) + 2]), "f"(acc_2[(24) + 3]), "l"(0x14F0000000000000ULL) : "memory");
                                }
                                long long at_23 = tile_off + 11776ll;
                                if (add != 0) {
                                    float old_23[4];
                                    {
                                        unsigned _v4_55_0;
                                        unsigned _v4_55_1;
                                        unsigned _v4_55_2;
                                        unsigned _v4_55_3;
                                        asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                            : "=r"(_v4_55_0), "=r"(_v4_55_1), "=r"(_v4_55_2), "=r"(_v4_55_3) : "l"((const void*)(dq_accum + at_23 + (0))), "l"(0x14F0000000000000ULL) : "memory");
                                        old_23[0 + 0] = __uint_as_float(_v4_55_0);
                                        old_23[0 + 1] = __uint_as_float(_v4_55_1);
                                        old_23[0 + 2] = __uint_as_float(_v4_55_2);
                                        old_23[0 + 3] = __uint_as_float(_v4_55_3);
                                    }
                                    acc_2[28] = old_23[0] + acc_2[28];
                                    acc_2[29] = old_23[1] + acc_2[29];
                                    acc_2[30] = old_23[2] + acc_2[30];
                                    acc_2[31] = old_23[3] + acc_2[31];
                                }
                                if (last == 0) {
                                    asm volatile("st.global.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(dq_accum + at_23), "f"(acc_2[(28) + 0]), "f"(acc_2[(28) + 1]), "f"(acc_2[(28) + 2]), "f"(acc_2[(28) + 3]), "l"(0x14F0000000000000ULL) : "memory");
                                }
                                if (last != 0) {
                                    unsigned int out_2[16];
                                    float2 _f32x2_mul_a_56 = make_float2(acc_2[0], acc_2[1]);
                                    float2 scaled_32;
                                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_32) : "l"(*(const unsigned long long*)&_f32x2_mul_a_56), "l"(*(const unsigned long long*)&scale2));
                                    out_2[0] = __as_u32(__float22bfloat162_rn(scaled_32));
                                    float2 _f32x2_mul_a_57 = make_float2(acc_2[2], acc_2[3]);
                                    float2 scaled_33;
                                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_33) : "l"(*(const unsigned long long*)&_f32x2_mul_a_57), "l"(*(const unsigned long long*)&scale2));
                                    out_2[1] = __as_u32(__float22bfloat162_rn(scaled_33));
                                    float2 _f32x2_mul_a_58 = make_float2(acc_2[4], acc_2[5]);
                                    float2 scaled_34;
                                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_34) : "l"(*(const unsigned long long*)&_f32x2_mul_a_58), "l"(*(const unsigned long long*)&scale2));
                                    out_2[2] = __as_u32(__float22bfloat162_rn(scaled_34));
                                    float2 _f32x2_mul_a_59 = make_float2(acc_2[6], acc_2[7]);
                                    float2 scaled_35;
                                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_35) : "l"(*(const unsigned long long*)&_f32x2_mul_a_59), "l"(*(const unsigned long long*)&scale2));
                                    out_2[3] = __as_u32(__float22bfloat162_rn(scaled_35));
                                    float2 _f32x2_mul_a_60 = make_float2(acc_2[8], acc_2[9]);
                                    float2 scaled_36;
                                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_36) : "l"(*(const unsigned long long*)&_f32x2_mul_a_60), "l"(*(const unsigned long long*)&scale2));
                                    out_2[4] = __as_u32(__float22bfloat162_rn(scaled_36));
                                    float2 _f32x2_mul_a_61 = make_float2(acc_2[10], acc_2[11]);
                                    float2 scaled_37;
                                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_37) : "l"(*(const unsigned long long*)&_f32x2_mul_a_61), "l"(*(const unsigned long long*)&scale2));
                                    out_2[5] = __as_u32(__float22bfloat162_rn(scaled_37));
                                    float2 _f32x2_mul_a_62 = make_float2(acc_2[12], acc_2[13]);
                                    float2 scaled_38;
                                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_38) : "l"(*(const unsigned long long*)&_f32x2_mul_a_62), "l"(*(const unsigned long long*)&scale2));
                                    out_2[6] = __as_u32(__float22bfloat162_rn(scaled_38));
                                    float2 _f32x2_mul_a_63 = make_float2(acc_2[14], acc_2[15]);
                                    float2 scaled_39;
                                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_39) : "l"(*(const unsigned long long*)&_f32x2_mul_a_63), "l"(*(const unsigned long long*)&scale2));
                                    out_2[7] = __as_u32(__float22bfloat162_rn(scaled_39));
                                    float2 _f32x2_mul_a_64 = make_float2(acc_2[16], acc_2[17]);
                                    float2 scaled_40;
                                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_40) : "l"(*(const unsigned long long*)&_f32x2_mul_a_64), "l"(*(const unsigned long long*)&scale2));
                                    out_2[8] = __as_u32(__float22bfloat162_rn(scaled_40));
                                    float2 _f32x2_mul_a_65 = make_float2(acc_2[18], acc_2[19]);
                                    float2 scaled_41;
                                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_41) : "l"(*(const unsigned long long*)&_f32x2_mul_a_65), "l"(*(const unsigned long long*)&scale2));
                                    out_2[9] = __as_u32(__float22bfloat162_rn(scaled_41));
                                    float2 _f32x2_mul_a_66 = make_float2(acc_2[20], acc_2[21]);
                                    float2 scaled_42;
                                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_42) : "l"(*(const unsigned long long*)&_f32x2_mul_a_66), "l"(*(const unsigned long long*)&scale2));
                                    out_2[10] = __as_u32(__float22bfloat162_rn(scaled_42));
                                    float2 _f32x2_mul_a_67 = make_float2(acc_2[22], acc_2[23]);
                                    float2 scaled_43;
                                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_43) : "l"(*(const unsigned long long*)&_f32x2_mul_a_67), "l"(*(const unsigned long long*)&scale2));
                                    out_2[11] = __as_u32(__float22bfloat162_rn(scaled_43));
                                    float2 _f32x2_mul_a_68 = make_float2(acc_2[24], acc_2[25]);
                                    float2 scaled_44;
                                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_44) : "l"(*(const unsigned long long*)&_f32x2_mul_a_68), "l"(*(const unsigned long long*)&scale2));
                                    out_2[12] = __as_u32(__float22bfloat162_rn(scaled_44));
                                    float2 _f32x2_mul_a_69 = make_float2(acc_2[26], acc_2[27]);
                                    float2 scaled_45;
                                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_45) : "l"(*(const unsigned long long*)&_f32x2_mul_a_69), "l"(*(const unsigned long long*)&scale2));
                                    out_2[13] = __as_u32(__float22bfloat162_rn(scaled_45));
                                    float2 _f32x2_mul_a_70 = make_float2(acc_2[28], acc_2[29]);
                                    float2 scaled_46;
                                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_46) : "l"(*(const unsigned long long*)&_f32x2_mul_a_70), "l"(*(const unsigned long long*)&scale2));
                                    out_2[14] = __as_u32(__float22bfloat162_rn(scaled_46));
                                    float2 _f32x2_mul_a_71 = make_float2(acc_2[30], acc_2[31]);
                                    float2 scaled_47;
                                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_47) : "l"(*(const unsigned long long*)&_f32x2_mul_a_71), "l"(*(const unsigned long long*)&scale2));
                                    out_2[15] = __as_u32(__float22bfloat162_rn(scaled_47));
                                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %9, 0; @p_ st.global.L2::evict_first.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8}; }" :: "l"(reinterpret_cast<unsigned int*>(dQ + (dq_row + 64ull))), "r"(out_2[(0) + 0]), "r"(out_2[(0) + 1]), "r"(out_2[(0) + 2]), "r"(out_2[(0) + 3]), "r"(out_2[(0) + 4]), "r"(out_2[(0) + 5]), "r"(out_2[(0) + 6]), "r"(out_2[(0) + 7]), "r"((unsigned int)(tok < q_len)) : "memory");
                                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %9, 0; @p_ st.global.L2::evict_first.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8}; }" :: "l"(reinterpret_cast<unsigned int*>(dQ + (dq_row + 80ull))), "r"(out_2[(8) + 0]), "r"(out_2[(8) + 1]), "r"(out_2[(8) + 2]), "r"(out_2[(8) + 3]), "r"(out_2[(8) + 4]), "r"(out_2[(8) + 5]), "r"(out_2[(8) + 6]), "r"(out_2[(8) + 7]), "r"((unsigned int)(tok < q_len)) : "memory");
                                }
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
                                        unsigned _v4_72_0;
                                        unsigned _v4_72_1;
                                        unsigned _v4_72_2;
                                        unsigned _v4_72_3;
                                        asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                            : "=r"(_v4_72_0), "=r"(_v4_72_1), "=r"(_v4_72_2), "=r"(_v4_72_3) : "l"((const void*)(dq_accum + at_24 + (0))), "l"(0x14F0000000000000ULL) : "memory");
                                        old_24[0 + 0] = __uint_as_float(_v4_72_0);
                                        old_24[0 + 1] = __uint_as_float(_v4_72_1);
                                        old_24[0 + 2] = __uint_as_float(_v4_72_2);
                                        old_24[0 + 3] = __uint_as_float(_v4_72_3);
                                    }
                                    acc_3[0] = old_24[0] + acc_3[0];
                                    acc_3[1] = old_24[1] + acc_3[1];
                                    acc_3[2] = old_24[2] + acc_3[2];
                                    acc_3[3] = old_24[3] + acc_3[3];
                                }
                                if (last == 0) {
                                    asm volatile("st.global.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(dq_accum + at_24), "f"(acc_3[(0) + 0]), "f"(acc_3[(0) + 1]), "f"(acc_3[(0) + 2]), "f"(acc_3[(0) + 3]), "l"(0x14F0000000000000ULL) : "memory");
                                }
                                long long at_25 = tile_off + 12800ll;
                                if (add != 0) {
                                    float old_25[4];
                                    {
                                        unsigned _v4_73_0;
                                        unsigned _v4_73_1;
                                        unsigned _v4_73_2;
                                        unsigned _v4_73_3;
                                        asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                            : "=r"(_v4_73_0), "=r"(_v4_73_1), "=r"(_v4_73_2), "=r"(_v4_73_3) : "l"((const void*)(dq_accum + at_25 + (0))), "l"(0x14F0000000000000ULL) : "memory");
                                        old_25[0 + 0] = __uint_as_float(_v4_73_0);
                                        old_25[0 + 1] = __uint_as_float(_v4_73_1);
                                        old_25[0 + 2] = __uint_as_float(_v4_73_2);
                                        old_25[0 + 3] = __uint_as_float(_v4_73_3);
                                    }
                                    acc_3[4] = old_25[0] + acc_3[4];
                                    acc_3[5] = old_25[1] + acc_3[5];
                                    acc_3[6] = old_25[2] + acc_3[6];
                                    acc_3[7] = old_25[3] + acc_3[7];
                                }
                                if (last == 0) {
                                    asm volatile("st.global.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(dq_accum + at_25), "f"(acc_3[(4) + 0]), "f"(acc_3[(4) + 1]), "f"(acc_3[(4) + 2]), "f"(acc_3[(4) + 3]), "l"(0x14F0000000000000ULL) : "memory");
                                }
                                long long at_26 = tile_off + 13312ll;
                                if (add != 0) {
                                    float old_26[4];
                                    {
                                        unsigned _v4_74_0;
                                        unsigned _v4_74_1;
                                        unsigned _v4_74_2;
                                        unsigned _v4_74_3;
                                        asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                            : "=r"(_v4_74_0), "=r"(_v4_74_1), "=r"(_v4_74_2), "=r"(_v4_74_3) : "l"((const void*)(dq_accum + at_26 + (0))), "l"(0x14F0000000000000ULL) : "memory");
                                        old_26[0 + 0] = __uint_as_float(_v4_74_0);
                                        old_26[0 + 1] = __uint_as_float(_v4_74_1);
                                        old_26[0 + 2] = __uint_as_float(_v4_74_2);
                                        old_26[0 + 3] = __uint_as_float(_v4_74_3);
                                    }
                                    acc_3[8] = old_26[0] + acc_3[8];
                                    acc_3[9] = old_26[1] + acc_3[9];
                                    acc_3[10] = old_26[2] + acc_3[10];
                                    acc_3[11] = old_26[3] + acc_3[11];
                                }
                                if (last == 0) {
                                    asm volatile("st.global.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(dq_accum + at_26), "f"(acc_3[(8) + 0]), "f"(acc_3[(8) + 1]), "f"(acc_3[(8) + 2]), "f"(acc_3[(8) + 3]), "l"(0x14F0000000000000ULL) : "memory");
                                }
                                long long at_27 = tile_off + 13824ll;
                                if (add != 0) {
                                    float old_27[4];
                                    {
                                        unsigned _v4_75_0;
                                        unsigned _v4_75_1;
                                        unsigned _v4_75_2;
                                        unsigned _v4_75_3;
                                        asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                            : "=r"(_v4_75_0), "=r"(_v4_75_1), "=r"(_v4_75_2), "=r"(_v4_75_3) : "l"((const void*)(dq_accum + at_27 + (0))), "l"(0x14F0000000000000ULL) : "memory");
                                        old_27[0 + 0] = __uint_as_float(_v4_75_0);
                                        old_27[0 + 1] = __uint_as_float(_v4_75_1);
                                        old_27[0 + 2] = __uint_as_float(_v4_75_2);
                                        old_27[0 + 3] = __uint_as_float(_v4_75_3);
                                    }
                                    acc_3[12] = old_27[0] + acc_3[12];
                                    acc_3[13] = old_27[1] + acc_3[13];
                                    acc_3[14] = old_27[2] + acc_3[14];
                                    acc_3[15] = old_27[3] + acc_3[15];
                                }
                                if (last == 0) {
                                    asm volatile("st.global.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(dq_accum + at_27), "f"(acc_3[(12) + 0]), "f"(acc_3[(12) + 1]), "f"(acc_3[(12) + 2]), "f"(acc_3[(12) + 3]), "l"(0x14F0000000000000ULL) : "memory");
                                }
                                long long at_28 = tile_off + 14336ll;
                                if (add != 0) {
                                    float old_28[4];
                                    {
                                        unsigned _v4_76_0;
                                        unsigned _v4_76_1;
                                        unsigned _v4_76_2;
                                        unsigned _v4_76_3;
                                        asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                            : "=r"(_v4_76_0), "=r"(_v4_76_1), "=r"(_v4_76_2), "=r"(_v4_76_3) : "l"((const void*)(dq_accum + at_28 + (0))), "l"(0x14F0000000000000ULL) : "memory");
                                        old_28[0 + 0] = __uint_as_float(_v4_76_0);
                                        old_28[0 + 1] = __uint_as_float(_v4_76_1);
                                        old_28[0 + 2] = __uint_as_float(_v4_76_2);
                                        old_28[0 + 3] = __uint_as_float(_v4_76_3);
                                    }
                                    acc_3[16] = old_28[0] + acc_3[16];
                                    acc_3[17] = old_28[1] + acc_3[17];
                                    acc_3[18] = old_28[2] + acc_3[18];
                                    acc_3[19] = old_28[3] + acc_3[19];
                                }
                                if (last == 0) {
                                    asm volatile("st.global.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(dq_accum + at_28), "f"(acc_3[(16) + 0]), "f"(acc_3[(16) + 1]), "f"(acc_3[(16) + 2]), "f"(acc_3[(16) + 3]), "l"(0x14F0000000000000ULL) : "memory");
                                }
                                long long at_29 = tile_off + 14848ll;
                                if (add != 0) {
                                    float old_29[4];
                                    {
                                        unsigned _v4_77_0;
                                        unsigned _v4_77_1;
                                        unsigned _v4_77_2;
                                        unsigned _v4_77_3;
                                        asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                            : "=r"(_v4_77_0), "=r"(_v4_77_1), "=r"(_v4_77_2), "=r"(_v4_77_3) : "l"((const void*)(dq_accum + at_29 + (0))), "l"(0x14F0000000000000ULL) : "memory");
                                        old_29[0 + 0] = __uint_as_float(_v4_77_0);
                                        old_29[0 + 1] = __uint_as_float(_v4_77_1);
                                        old_29[0 + 2] = __uint_as_float(_v4_77_2);
                                        old_29[0 + 3] = __uint_as_float(_v4_77_3);
                                    }
                                    acc_3[20] = old_29[0] + acc_3[20];
                                    acc_3[21] = old_29[1] + acc_3[21];
                                    acc_3[22] = old_29[2] + acc_3[22];
                                    acc_3[23] = old_29[3] + acc_3[23];
                                }
                                if (last == 0) {
                                    asm volatile("st.global.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(dq_accum + at_29), "f"(acc_3[(20) + 0]), "f"(acc_3[(20) + 1]), "f"(acc_3[(20) + 2]), "f"(acc_3[(20) + 3]), "l"(0x14F0000000000000ULL) : "memory");
                                }
                                long long at_30 = tile_off + 15360ll;
                                if (add != 0) {
                                    float old_30[4];
                                    {
                                        unsigned _v4_78_0;
                                        unsigned _v4_78_1;
                                        unsigned _v4_78_2;
                                        unsigned _v4_78_3;
                                        asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                            : "=r"(_v4_78_0), "=r"(_v4_78_1), "=r"(_v4_78_2), "=r"(_v4_78_3) : "l"((const void*)(dq_accum + at_30 + (0))), "l"(0x14F0000000000000ULL) : "memory");
                                        old_30[0 + 0] = __uint_as_float(_v4_78_0);
                                        old_30[0 + 1] = __uint_as_float(_v4_78_1);
                                        old_30[0 + 2] = __uint_as_float(_v4_78_2);
                                        old_30[0 + 3] = __uint_as_float(_v4_78_3);
                                    }
                                    acc_3[24] = old_30[0] + acc_3[24];
                                    acc_3[25] = old_30[1] + acc_3[25];
                                    acc_3[26] = old_30[2] + acc_3[26];
                                    acc_3[27] = old_30[3] + acc_3[27];
                                }
                                if (last == 0) {
                                    asm volatile("st.global.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(dq_accum + at_30), "f"(acc_3[(24) + 0]), "f"(acc_3[(24) + 1]), "f"(acc_3[(24) + 2]), "f"(acc_3[(24) + 3]), "l"(0x14F0000000000000ULL) : "memory");
                                }
                                long long at_31 = tile_off + 15872ll;
                                if (add != 0) {
                                    float old_31[4];
                                    {
                                        unsigned _v4_79_0;
                                        unsigned _v4_79_1;
                                        unsigned _v4_79_2;
                                        unsigned _v4_79_3;
                                        asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                            : "=r"(_v4_79_0), "=r"(_v4_79_1), "=r"(_v4_79_2), "=r"(_v4_79_3) : "l"((const void*)(dq_accum + at_31 + (0))), "l"(0x14F0000000000000ULL) : "memory");
                                        old_31[0 + 0] = __uint_as_float(_v4_79_0);
                                        old_31[0 + 1] = __uint_as_float(_v4_79_1);
                                        old_31[0 + 2] = __uint_as_float(_v4_79_2);
                                        old_31[0 + 3] = __uint_as_float(_v4_79_3);
                                    }
                                    acc_3[28] = old_31[0] + acc_3[28];
                                    acc_3[29] = old_31[1] + acc_3[29];
                                    acc_3[30] = old_31[2] + acc_3[30];
                                    acc_3[31] = old_31[3] + acc_3[31];
                                }
                                if (last == 0) {
                                    asm volatile("st.global.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(dq_accum + at_31), "f"(acc_3[(28) + 0]), "f"(acc_3[(28) + 1]), "f"(acc_3[(28) + 2]), "f"(acc_3[(28) + 3]), "l"(0x14F0000000000000ULL) : "memory");
                                }
                                if (last != 0) {
                                    unsigned int out_3[16];
                                    float2 _f32x2_mul_a_80 = make_float2(acc_3[0], acc_3[1]);
                                    float2 scaled_48;
                                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_48) : "l"(*(const unsigned long long*)&_f32x2_mul_a_80), "l"(*(const unsigned long long*)&scale2));
                                    out_3[0] = __as_u32(__float22bfloat162_rn(scaled_48));
                                    float2 _f32x2_mul_a_81 = make_float2(acc_3[2], acc_3[3]);
                                    float2 scaled_49;
                                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_49) : "l"(*(const unsigned long long*)&_f32x2_mul_a_81), "l"(*(const unsigned long long*)&scale2));
                                    out_3[1] = __as_u32(__float22bfloat162_rn(scaled_49));
                                    float2 _f32x2_mul_a_82 = make_float2(acc_3[4], acc_3[5]);
                                    float2 scaled_50;
                                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_50) : "l"(*(const unsigned long long*)&_f32x2_mul_a_82), "l"(*(const unsigned long long*)&scale2));
                                    out_3[2] = __as_u32(__float22bfloat162_rn(scaled_50));
                                    float2 _f32x2_mul_a_83 = make_float2(acc_3[6], acc_3[7]);
                                    float2 scaled_51;
                                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_51) : "l"(*(const unsigned long long*)&_f32x2_mul_a_83), "l"(*(const unsigned long long*)&scale2));
                                    out_3[3] = __as_u32(__float22bfloat162_rn(scaled_51));
                                    float2 _f32x2_mul_a_84 = make_float2(acc_3[8], acc_3[9]);
                                    float2 scaled_52;
                                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_52) : "l"(*(const unsigned long long*)&_f32x2_mul_a_84), "l"(*(const unsigned long long*)&scale2));
                                    out_3[4] = __as_u32(__float22bfloat162_rn(scaled_52));
                                    float2 _f32x2_mul_a_85 = make_float2(acc_3[10], acc_3[11]);
                                    float2 scaled_53;
                                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_53) : "l"(*(const unsigned long long*)&_f32x2_mul_a_85), "l"(*(const unsigned long long*)&scale2));
                                    out_3[5] = __as_u32(__float22bfloat162_rn(scaled_53));
                                    float2 _f32x2_mul_a_86 = make_float2(acc_3[12], acc_3[13]);
                                    float2 scaled_54;
                                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_54) : "l"(*(const unsigned long long*)&_f32x2_mul_a_86), "l"(*(const unsigned long long*)&scale2));
                                    out_3[6] = __as_u32(__float22bfloat162_rn(scaled_54));
                                    float2 _f32x2_mul_a_87 = make_float2(acc_3[14], acc_3[15]);
                                    float2 scaled_55;
                                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_55) : "l"(*(const unsigned long long*)&_f32x2_mul_a_87), "l"(*(const unsigned long long*)&scale2));
                                    out_3[7] = __as_u32(__float22bfloat162_rn(scaled_55));
                                    float2 _f32x2_mul_a_88 = make_float2(acc_3[16], acc_3[17]);
                                    float2 scaled_56;
                                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_56) : "l"(*(const unsigned long long*)&_f32x2_mul_a_88), "l"(*(const unsigned long long*)&scale2));
                                    out_3[8] = __as_u32(__float22bfloat162_rn(scaled_56));
                                    float2 _f32x2_mul_a_89 = make_float2(acc_3[18], acc_3[19]);
                                    float2 scaled_57;
                                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_57) : "l"(*(const unsigned long long*)&_f32x2_mul_a_89), "l"(*(const unsigned long long*)&scale2));
                                    out_3[9] = __as_u32(__float22bfloat162_rn(scaled_57));
                                    float2 _f32x2_mul_a_90 = make_float2(acc_3[20], acc_3[21]);
                                    float2 scaled_58;
                                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_58) : "l"(*(const unsigned long long*)&_f32x2_mul_a_90), "l"(*(const unsigned long long*)&scale2));
                                    out_3[10] = __as_u32(__float22bfloat162_rn(scaled_58));
                                    float2 _f32x2_mul_a_91 = make_float2(acc_3[22], acc_3[23]);
                                    float2 scaled_59;
                                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_59) : "l"(*(const unsigned long long*)&_f32x2_mul_a_91), "l"(*(const unsigned long long*)&scale2));
                                    out_3[11] = __as_u32(__float22bfloat162_rn(scaled_59));
                                    float2 _f32x2_mul_a_92 = make_float2(acc_3[24], acc_3[25]);
                                    float2 scaled_60;
                                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_60) : "l"(*(const unsigned long long*)&_f32x2_mul_a_92), "l"(*(const unsigned long long*)&scale2));
                                    out_3[12] = __as_u32(__float22bfloat162_rn(scaled_60));
                                    float2 _f32x2_mul_a_93 = make_float2(acc_3[26], acc_3[27]);
                                    float2 scaled_61;
                                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_61) : "l"(*(const unsigned long long*)&_f32x2_mul_a_93), "l"(*(const unsigned long long*)&scale2));
                                    out_3[13] = __as_u32(__float22bfloat162_rn(scaled_61));
                                    float2 _f32x2_mul_a_94 = make_float2(acc_3[28], acc_3[29]);
                                    float2 scaled_62;
                                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_62) : "l"(*(const unsigned long long*)&_f32x2_mul_a_94), "l"(*(const unsigned long long*)&scale2));
                                    out_3[14] = __as_u32(__float22bfloat162_rn(scaled_62));
                                    float2 _f32x2_mul_a_95 = make_float2(acc_3[30], acc_3[31]);
                                    float2 scaled_63;
                                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_63) : "l"(*(const unsigned long long*)&_f32x2_mul_a_95), "l"(*(const unsigned long long*)&scale2));
                                    out_3[15] = __as_u32(__float22bfloat162_rn(scaled_63));
                                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %9, 0; @p_ st.global.L2::evict_first.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8}; }" :: "l"(reinterpret_cast<unsigned int*>(dQ + (dq_row + 96ull))), "r"(out_3[(0) + 0]), "r"(out_3[(0) + 1]), "r"(out_3[(0) + 2]), "r"(out_3[(0) + 3]), "r"(out_3[(0) + 4]), "r"(out_3[(0) + 5]), "r"(out_3[(0) + 6]), "r"(out_3[(0) + 7]), "r"((unsigned int)(tok < q_len)) : "memory");
                                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %9, 0; @p_ st.global.L2::evict_first.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8}; }" :: "l"(reinterpret_cast<unsigned int*>(dQ + (dq_row + 112ull))), "r"(out_3[(8) + 0]), "r"(out_3[(8) + 1]), "r"(out_3[(8) + 2]), "r"(out_3[(8) + 3]), "r"(out_3[(8) + 4]), "r"(out_3[(8) + 5]), "r"(out_3[(8) + 6]), "r"(out_3[(8) + 7]), "r"((unsigned int)(tok < q_len)) : "memory");
                                }
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
                                        unsigned _v4_96_0;
                                        unsigned _v4_96_1;
                                        unsigned _v4_96_2;
                                        unsigned _v4_96_3;
                                        asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                            : "=r"(_v4_96_0), "=r"(_v4_96_1), "=r"(_v4_96_2), "=r"(_v4_96_3) : "l"((const void*)(dq_accum + at_32 + (0))), "l"(0x14F0000000000000ULL) : "memory");
                                        old_32[0 + 0] = __uint_as_float(_v4_96_0);
                                        old_32[0 + 1] = __uint_as_float(_v4_96_1);
                                        old_32[0 + 2] = __uint_as_float(_v4_96_2);
                                        old_32[0 + 3] = __uint_as_float(_v4_96_3);
                                    }
                                    acc_4[0] = old_32[0] + acc_4[0];
                                    acc_4[1] = old_32[1] + acc_4[1];
                                    acc_4[2] = old_32[2] + acc_4[2];
                                    acc_4[3] = old_32[3] + acc_4[3];
                                }
                                if (last == 0) {
                                    asm volatile("st.global.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(dq_accum + at_32), "f"(acc_4[(0) + 0]), "f"(acc_4[(0) + 1]), "f"(acc_4[(0) + 2]), "f"(acc_4[(0) + 3]), "l"(0x14F0000000000000ULL) : "memory");
                                }
                                long long at_33 = tile_off + 768ll;
                                if (add != 0) {
                                    float old_33[4];
                                    {
                                        unsigned _v4_97_0;
                                        unsigned _v4_97_1;
                                        unsigned _v4_97_2;
                                        unsigned _v4_97_3;
                                        asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                            : "=r"(_v4_97_0), "=r"(_v4_97_1), "=r"(_v4_97_2), "=r"(_v4_97_3) : "l"((const void*)(dq_accum + at_33 + (0))), "l"(0x14F0000000000000ULL) : "memory");
                                        old_33[0 + 0] = __uint_as_float(_v4_97_0);
                                        old_33[0 + 1] = __uint_as_float(_v4_97_1);
                                        old_33[0 + 2] = __uint_as_float(_v4_97_2);
                                        old_33[0 + 3] = __uint_as_float(_v4_97_3);
                                    }
                                    acc_4[4] = old_33[0] + acc_4[4];
                                    acc_4[5] = old_33[1] + acc_4[5];
                                    acc_4[6] = old_33[2] + acc_4[6];
                                    acc_4[7] = old_33[3] + acc_4[7];
                                }
                                if (last == 0) {
                                    asm volatile("st.global.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(dq_accum + at_33), "f"(acc_4[(4) + 0]), "f"(acc_4[(4) + 1]), "f"(acc_4[(4) + 2]), "f"(acc_4[(4) + 3]), "l"(0x14F0000000000000ULL) : "memory");
                                }
                                long long at_34 = tile_off + 1280ll;
                                if (add != 0) {
                                    float old_34[4];
                                    {
                                        unsigned _v4_98_0;
                                        unsigned _v4_98_1;
                                        unsigned _v4_98_2;
                                        unsigned _v4_98_3;
                                        asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                            : "=r"(_v4_98_0), "=r"(_v4_98_1), "=r"(_v4_98_2), "=r"(_v4_98_3) : "l"((const void*)(dq_accum + at_34 + (0))), "l"(0x14F0000000000000ULL) : "memory");
                                        old_34[0 + 0] = __uint_as_float(_v4_98_0);
                                        old_34[0 + 1] = __uint_as_float(_v4_98_1);
                                        old_34[0 + 2] = __uint_as_float(_v4_98_2);
                                        old_34[0 + 3] = __uint_as_float(_v4_98_3);
                                    }
                                    acc_4[8] = old_34[0] + acc_4[8];
                                    acc_4[9] = old_34[1] + acc_4[9];
                                    acc_4[10] = old_34[2] + acc_4[10];
                                    acc_4[11] = old_34[3] + acc_4[11];
                                }
                                if (last == 0) {
                                    asm volatile("st.global.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(dq_accum + at_34), "f"(acc_4[(8) + 0]), "f"(acc_4[(8) + 1]), "f"(acc_4[(8) + 2]), "f"(acc_4[(8) + 3]), "l"(0x14F0000000000000ULL) : "memory");
                                }
                                long long at_35 = tile_off + 1792ll;
                                if (add != 0) {
                                    float old_35[4];
                                    {
                                        unsigned _v4_99_0;
                                        unsigned _v4_99_1;
                                        unsigned _v4_99_2;
                                        unsigned _v4_99_3;
                                        asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                            : "=r"(_v4_99_0), "=r"(_v4_99_1), "=r"(_v4_99_2), "=r"(_v4_99_3) : "l"((const void*)(dq_accum + at_35 + (0))), "l"(0x14F0000000000000ULL) : "memory");
                                        old_35[0 + 0] = __uint_as_float(_v4_99_0);
                                        old_35[0 + 1] = __uint_as_float(_v4_99_1);
                                        old_35[0 + 2] = __uint_as_float(_v4_99_2);
                                        old_35[0 + 3] = __uint_as_float(_v4_99_3);
                                    }
                                    acc_4[12] = old_35[0] + acc_4[12];
                                    acc_4[13] = old_35[1] + acc_4[13];
                                    acc_4[14] = old_35[2] + acc_4[14];
                                    acc_4[15] = old_35[3] + acc_4[15];
                                }
                                if (last == 0) {
                                    asm volatile("st.global.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(dq_accum + at_35), "f"(acc_4[(12) + 0]), "f"(acc_4[(12) + 1]), "f"(acc_4[(12) + 2]), "f"(acc_4[(12) + 3]), "l"(0x14F0000000000000ULL) : "memory");
                                }
                                long long at_36 = tile_off + 2304ll;
                                if (add != 0) {
                                    float old_36[4];
                                    {
                                        unsigned _v4_100_0;
                                        unsigned _v4_100_1;
                                        unsigned _v4_100_2;
                                        unsigned _v4_100_3;
                                        asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                            : "=r"(_v4_100_0), "=r"(_v4_100_1), "=r"(_v4_100_2), "=r"(_v4_100_3) : "l"((const void*)(dq_accum + at_36 + (0))), "l"(0x14F0000000000000ULL) : "memory");
                                        old_36[0 + 0] = __uint_as_float(_v4_100_0);
                                        old_36[0 + 1] = __uint_as_float(_v4_100_1);
                                        old_36[0 + 2] = __uint_as_float(_v4_100_2);
                                        old_36[0 + 3] = __uint_as_float(_v4_100_3);
                                    }
                                    acc_4[16] = old_36[0] + acc_4[16];
                                    acc_4[17] = old_36[1] + acc_4[17];
                                    acc_4[18] = old_36[2] + acc_4[18];
                                    acc_4[19] = old_36[3] + acc_4[19];
                                }
                                if (last == 0) {
                                    asm volatile("st.global.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(dq_accum + at_36), "f"(acc_4[(16) + 0]), "f"(acc_4[(16) + 1]), "f"(acc_4[(16) + 2]), "f"(acc_4[(16) + 3]), "l"(0x14F0000000000000ULL) : "memory");
                                }
                                long long at_37 = tile_off + 2816ll;
                                if (add != 0) {
                                    float old_37[4];
                                    {
                                        unsigned _v4_101_0;
                                        unsigned _v4_101_1;
                                        unsigned _v4_101_2;
                                        unsigned _v4_101_3;
                                        asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                            : "=r"(_v4_101_0), "=r"(_v4_101_1), "=r"(_v4_101_2), "=r"(_v4_101_3) : "l"((const void*)(dq_accum + at_37 + (0))), "l"(0x14F0000000000000ULL) : "memory");
                                        old_37[0 + 0] = __uint_as_float(_v4_101_0);
                                        old_37[0 + 1] = __uint_as_float(_v4_101_1);
                                        old_37[0 + 2] = __uint_as_float(_v4_101_2);
                                        old_37[0 + 3] = __uint_as_float(_v4_101_3);
                                    }
                                    acc_4[20] = old_37[0] + acc_4[20];
                                    acc_4[21] = old_37[1] + acc_4[21];
                                    acc_4[22] = old_37[2] + acc_4[22];
                                    acc_4[23] = old_37[3] + acc_4[23];
                                }
                                if (last == 0) {
                                    asm volatile("st.global.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(dq_accum + at_37), "f"(acc_4[(20) + 0]), "f"(acc_4[(20) + 1]), "f"(acc_4[(20) + 2]), "f"(acc_4[(20) + 3]), "l"(0x14F0000000000000ULL) : "memory");
                                }
                                long long at_38 = tile_off + 3328ll;
                                if (add != 0) {
                                    float old_38[4];
                                    {
                                        unsigned _v4_102_0;
                                        unsigned _v4_102_1;
                                        unsigned _v4_102_2;
                                        unsigned _v4_102_3;
                                        asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                            : "=r"(_v4_102_0), "=r"(_v4_102_1), "=r"(_v4_102_2), "=r"(_v4_102_3) : "l"((const void*)(dq_accum + at_38 + (0))), "l"(0x14F0000000000000ULL) : "memory");
                                        old_38[0 + 0] = __uint_as_float(_v4_102_0);
                                        old_38[0 + 1] = __uint_as_float(_v4_102_1);
                                        old_38[0 + 2] = __uint_as_float(_v4_102_2);
                                        old_38[0 + 3] = __uint_as_float(_v4_102_3);
                                    }
                                    acc_4[24] = old_38[0] + acc_4[24];
                                    acc_4[25] = old_38[1] + acc_4[25];
                                    acc_4[26] = old_38[2] + acc_4[26];
                                    acc_4[27] = old_38[3] + acc_4[27];
                                }
                                if (last == 0) {
                                    asm volatile("st.global.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(dq_accum + at_38), "f"(acc_4[(24) + 0]), "f"(acc_4[(24) + 1]), "f"(acc_4[(24) + 2]), "f"(acc_4[(24) + 3]), "l"(0x14F0000000000000ULL) : "memory");
                                }
                                long long at_39 = tile_off + 3840ll;
                                if (add != 0) {
                                    float old_39[4];
                                    {
                                        unsigned _v4_103_0;
                                        unsigned _v4_103_1;
                                        unsigned _v4_103_2;
                                        unsigned _v4_103_3;
                                        asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                            : "=r"(_v4_103_0), "=r"(_v4_103_1), "=r"(_v4_103_2), "=r"(_v4_103_3) : "l"((const void*)(dq_accum + at_39 + (0))), "l"(0x14F0000000000000ULL) : "memory");
                                        old_39[0 + 0] = __uint_as_float(_v4_103_0);
                                        old_39[0 + 1] = __uint_as_float(_v4_103_1);
                                        old_39[0 + 2] = __uint_as_float(_v4_103_2);
                                        old_39[0 + 3] = __uint_as_float(_v4_103_3);
                                    }
                                    acc_4[28] = old_39[0] + acc_4[28];
                                    acc_4[29] = old_39[1] + acc_4[29];
                                    acc_4[30] = old_39[2] + acc_4[30];
                                    acc_4[31] = old_39[3] + acc_4[31];
                                }
                                if (last == 0) {
                                    asm volatile("st.global.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(dq_accum + at_39), "f"(acc_4[(28) + 0]), "f"(acc_4[(28) + 1]), "f"(acc_4[(28) + 2]), "f"(acc_4[(28) + 3]), "l"(0x14F0000000000000ULL) : "memory");
                                }
                                if (last != 0) {
                                    unsigned int out_4[16];
                                    float2 _f32x2_mul_a_104 = make_float2(acc_4[0], acc_4[1]);
                                    float2 scaled_64;
                                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_64) : "l"(*(const unsigned long long*)&_f32x2_mul_a_104), "l"(*(const unsigned long long*)&scale2));
                                    out_4[0] = __as_u32(__float22bfloat162_rn(scaled_64));
                                    float2 _f32x2_mul_a_105 = make_float2(acc_4[2], acc_4[3]);
                                    float2 scaled_65;
                                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_65) : "l"(*(const unsigned long long*)&_f32x2_mul_a_105), "l"(*(const unsigned long long*)&scale2));
                                    out_4[1] = __as_u32(__float22bfloat162_rn(scaled_65));
                                    float2 _f32x2_mul_a_106 = make_float2(acc_4[4], acc_4[5]);
                                    float2 scaled_66;
                                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_66) : "l"(*(const unsigned long long*)&_f32x2_mul_a_106), "l"(*(const unsigned long long*)&scale2));
                                    out_4[2] = __as_u32(__float22bfloat162_rn(scaled_66));
                                    float2 _f32x2_mul_a_107 = make_float2(acc_4[6], acc_4[7]);
                                    float2 scaled_67;
                                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_67) : "l"(*(const unsigned long long*)&_f32x2_mul_a_107), "l"(*(const unsigned long long*)&scale2));
                                    out_4[3] = __as_u32(__float22bfloat162_rn(scaled_67));
                                    float2 _f32x2_mul_a_108 = make_float2(acc_4[8], acc_4[9]);
                                    float2 scaled_68;
                                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_68) : "l"(*(const unsigned long long*)&_f32x2_mul_a_108), "l"(*(const unsigned long long*)&scale2));
                                    out_4[4] = __as_u32(__float22bfloat162_rn(scaled_68));
                                    float2 _f32x2_mul_a_109 = make_float2(acc_4[10], acc_4[11]);
                                    float2 scaled_69;
                                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_69) : "l"(*(const unsigned long long*)&_f32x2_mul_a_109), "l"(*(const unsigned long long*)&scale2));
                                    out_4[5] = __as_u32(__float22bfloat162_rn(scaled_69));
                                    float2 _f32x2_mul_a_110 = make_float2(acc_4[12], acc_4[13]);
                                    float2 scaled_70;
                                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_70) : "l"(*(const unsigned long long*)&_f32x2_mul_a_110), "l"(*(const unsigned long long*)&scale2));
                                    out_4[6] = __as_u32(__float22bfloat162_rn(scaled_70));
                                    float2 _f32x2_mul_a_111 = make_float2(acc_4[14], acc_4[15]);
                                    float2 scaled_71;
                                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_71) : "l"(*(const unsigned long long*)&_f32x2_mul_a_111), "l"(*(const unsigned long long*)&scale2));
                                    out_4[7] = __as_u32(__float22bfloat162_rn(scaled_71));
                                    float2 _f32x2_mul_a_112 = make_float2(acc_4[16], acc_4[17]);
                                    float2 scaled_72;
                                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_72) : "l"(*(const unsigned long long*)&_f32x2_mul_a_112), "l"(*(const unsigned long long*)&scale2));
                                    out_4[8] = __as_u32(__float22bfloat162_rn(scaled_72));
                                    float2 _f32x2_mul_a_113 = make_float2(acc_4[18], acc_4[19]);
                                    float2 scaled_73;
                                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_73) : "l"(*(const unsigned long long*)&_f32x2_mul_a_113), "l"(*(const unsigned long long*)&scale2));
                                    out_4[9] = __as_u32(__float22bfloat162_rn(scaled_73));
                                    float2 _f32x2_mul_a_114 = make_float2(acc_4[20], acc_4[21]);
                                    float2 scaled_74;
                                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_74) : "l"(*(const unsigned long long*)&_f32x2_mul_a_114), "l"(*(const unsigned long long*)&scale2));
                                    out_4[10] = __as_u32(__float22bfloat162_rn(scaled_74));
                                    float2 _f32x2_mul_a_115 = make_float2(acc_4[22], acc_4[23]);
                                    float2 scaled_75;
                                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_75) : "l"(*(const unsigned long long*)&_f32x2_mul_a_115), "l"(*(const unsigned long long*)&scale2));
                                    out_4[11] = __as_u32(__float22bfloat162_rn(scaled_75));
                                    float2 _f32x2_mul_a_116 = make_float2(acc_4[24], acc_4[25]);
                                    float2 scaled_76;
                                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_76) : "l"(*(const unsigned long long*)&_f32x2_mul_a_116), "l"(*(const unsigned long long*)&scale2));
                                    out_4[12] = __as_u32(__float22bfloat162_rn(scaled_76));
                                    float2 _f32x2_mul_a_117 = make_float2(acc_4[26], acc_4[27]);
                                    float2 scaled_77;
                                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_77) : "l"(*(const unsigned long long*)&_f32x2_mul_a_117), "l"(*(const unsigned long long*)&scale2));
                                    out_4[13] = __as_u32(__float22bfloat162_rn(scaled_77));
                                    float2 _f32x2_mul_a_118 = make_float2(acc_4[28], acc_4[29]);
                                    float2 scaled_78;
                                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_78) : "l"(*(const unsigned long long*)&_f32x2_mul_a_118), "l"(*(const unsigned long long*)&scale2));
                                    out_4[14] = __as_u32(__float22bfloat162_rn(scaled_78));
                                    float2 _f32x2_mul_a_119 = make_float2(acc_4[30], acc_4[31]);
                                    float2 scaled_79;
                                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_79) : "l"(*(const unsigned long long*)&_f32x2_mul_a_119), "l"(*(const unsigned long long*)&scale2));
                                    out_4[15] = __as_u32(__float22bfloat162_rn(scaled_79));
                                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %9, 0; @p_ st.global.L2::evict_first.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8}; }" :: "l"(reinterpret_cast<unsigned int*>(dQ + (dq_row + 128ull))), "r"(out_4[(0) + 0]), "r"(out_4[(0) + 1]), "r"(out_4[(0) + 2]), "r"(out_4[(0) + 3]), "r"(out_4[(0) + 4]), "r"(out_4[(0) + 5]), "r"(out_4[(0) + 6]), "r"(out_4[(0) + 7]), "r"((unsigned int)(tok < q_len)) : "memory");
                                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %9, 0; @p_ st.global.L2::evict_first.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8}; }" :: "l"(reinterpret_cast<unsigned int*>(dQ + (dq_row + 144ull))), "r"(out_4[(8) + 0]), "r"(out_4[(8) + 1]), "r"(out_4[(8) + 2]), "r"(out_4[(8) + 3]), "r"(out_4[(8) + 4]), "r"(out_4[(8) + 5]), "r"(out_4[(8) + 6]), "r"(out_4[(8) + 7]), "r"((unsigned int)(tok < q_len)) : "memory");
                                }
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
                                        unsigned _v4_120_0;
                                        unsigned _v4_120_1;
                                        unsigned _v4_120_2;
                                        unsigned _v4_120_3;
                                        asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                            : "=r"(_v4_120_0), "=r"(_v4_120_1), "=r"(_v4_120_2), "=r"(_v4_120_3) : "l"((const void*)(dq_accum + at_40 + (0))), "l"(0x14F0000000000000ULL) : "memory");
                                        old_40[0 + 0] = __uint_as_float(_v4_120_0);
                                        old_40[0 + 1] = __uint_as_float(_v4_120_1);
                                        old_40[0 + 2] = __uint_as_float(_v4_120_2);
                                        old_40[0 + 3] = __uint_as_float(_v4_120_3);
                                    }
                                    acc_5[0] = old_40[0] + acc_5[0];
                                    acc_5[1] = old_40[1] + acc_5[1];
                                    acc_5[2] = old_40[2] + acc_5[2];
                                    acc_5[3] = old_40[3] + acc_5[3];
                                }
                                if (last == 0) {
                                    asm volatile("st.global.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(dq_accum + at_40), "f"(acc_5[(0) + 0]), "f"(acc_5[(0) + 1]), "f"(acc_5[(0) + 2]), "f"(acc_5[(0) + 3]), "l"(0x14F0000000000000ULL) : "memory");
                                }
                                long long at_41 = tile_off + 4864ll;
                                if (add != 0) {
                                    float old_41[4];
                                    {
                                        unsigned _v4_121_0;
                                        unsigned _v4_121_1;
                                        unsigned _v4_121_2;
                                        unsigned _v4_121_3;
                                        asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                            : "=r"(_v4_121_0), "=r"(_v4_121_1), "=r"(_v4_121_2), "=r"(_v4_121_3) : "l"((const void*)(dq_accum + at_41 + (0))), "l"(0x14F0000000000000ULL) : "memory");
                                        old_41[0 + 0] = __uint_as_float(_v4_121_0);
                                        old_41[0 + 1] = __uint_as_float(_v4_121_1);
                                        old_41[0 + 2] = __uint_as_float(_v4_121_2);
                                        old_41[0 + 3] = __uint_as_float(_v4_121_3);
                                    }
                                    acc_5[4] = old_41[0] + acc_5[4];
                                    acc_5[5] = old_41[1] + acc_5[5];
                                    acc_5[6] = old_41[2] + acc_5[6];
                                    acc_5[7] = old_41[3] + acc_5[7];
                                }
                                if (last == 0) {
                                    asm volatile("st.global.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(dq_accum + at_41), "f"(acc_5[(4) + 0]), "f"(acc_5[(4) + 1]), "f"(acc_5[(4) + 2]), "f"(acc_5[(4) + 3]), "l"(0x14F0000000000000ULL) : "memory");
                                }
                                long long at_42 = tile_off + 5376ll;
                                if (add != 0) {
                                    float old_42[4];
                                    {
                                        unsigned _v4_122_0;
                                        unsigned _v4_122_1;
                                        unsigned _v4_122_2;
                                        unsigned _v4_122_3;
                                        asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                            : "=r"(_v4_122_0), "=r"(_v4_122_1), "=r"(_v4_122_2), "=r"(_v4_122_3) : "l"((const void*)(dq_accum + at_42 + (0))), "l"(0x14F0000000000000ULL) : "memory");
                                        old_42[0 + 0] = __uint_as_float(_v4_122_0);
                                        old_42[0 + 1] = __uint_as_float(_v4_122_1);
                                        old_42[0 + 2] = __uint_as_float(_v4_122_2);
                                        old_42[0 + 3] = __uint_as_float(_v4_122_3);
                                    }
                                    acc_5[8] = old_42[0] + acc_5[8];
                                    acc_5[9] = old_42[1] + acc_5[9];
                                    acc_5[10] = old_42[2] + acc_5[10];
                                    acc_5[11] = old_42[3] + acc_5[11];
                                }
                                if (last == 0) {
                                    asm volatile("st.global.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(dq_accum + at_42), "f"(acc_5[(8) + 0]), "f"(acc_5[(8) + 1]), "f"(acc_5[(8) + 2]), "f"(acc_5[(8) + 3]), "l"(0x14F0000000000000ULL) : "memory");
                                }
                                long long at_43 = tile_off + 5888ll;
                                if (add != 0) {
                                    float old_43[4];
                                    {
                                        unsigned _v4_123_0;
                                        unsigned _v4_123_1;
                                        unsigned _v4_123_2;
                                        unsigned _v4_123_3;
                                        asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                            : "=r"(_v4_123_0), "=r"(_v4_123_1), "=r"(_v4_123_2), "=r"(_v4_123_3) : "l"((const void*)(dq_accum + at_43 + (0))), "l"(0x14F0000000000000ULL) : "memory");
                                        old_43[0 + 0] = __uint_as_float(_v4_123_0);
                                        old_43[0 + 1] = __uint_as_float(_v4_123_1);
                                        old_43[0 + 2] = __uint_as_float(_v4_123_2);
                                        old_43[0 + 3] = __uint_as_float(_v4_123_3);
                                    }
                                    acc_5[12] = old_43[0] + acc_5[12];
                                    acc_5[13] = old_43[1] + acc_5[13];
                                    acc_5[14] = old_43[2] + acc_5[14];
                                    acc_5[15] = old_43[3] + acc_5[15];
                                }
                                if (last == 0) {
                                    asm volatile("st.global.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(dq_accum + at_43), "f"(acc_5[(12) + 0]), "f"(acc_5[(12) + 1]), "f"(acc_5[(12) + 2]), "f"(acc_5[(12) + 3]), "l"(0x14F0000000000000ULL) : "memory");
                                }
                                long long at_44 = tile_off + 6400ll;
                                if (add != 0) {
                                    float old_44[4];
                                    {
                                        unsigned _v4_124_0;
                                        unsigned _v4_124_1;
                                        unsigned _v4_124_2;
                                        unsigned _v4_124_3;
                                        asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                            : "=r"(_v4_124_0), "=r"(_v4_124_1), "=r"(_v4_124_2), "=r"(_v4_124_3) : "l"((const void*)(dq_accum + at_44 + (0))), "l"(0x14F0000000000000ULL) : "memory");
                                        old_44[0 + 0] = __uint_as_float(_v4_124_0);
                                        old_44[0 + 1] = __uint_as_float(_v4_124_1);
                                        old_44[0 + 2] = __uint_as_float(_v4_124_2);
                                        old_44[0 + 3] = __uint_as_float(_v4_124_3);
                                    }
                                    acc_5[16] = old_44[0] + acc_5[16];
                                    acc_5[17] = old_44[1] + acc_5[17];
                                    acc_5[18] = old_44[2] + acc_5[18];
                                    acc_5[19] = old_44[3] + acc_5[19];
                                }
                                if (last == 0) {
                                    asm volatile("st.global.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(dq_accum + at_44), "f"(acc_5[(16) + 0]), "f"(acc_5[(16) + 1]), "f"(acc_5[(16) + 2]), "f"(acc_5[(16) + 3]), "l"(0x14F0000000000000ULL) : "memory");
                                }
                                long long at_45 = tile_off + 6912ll;
                                if (add != 0) {
                                    float old_45[4];
                                    {
                                        unsigned _v4_125_0;
                                        unsigned _v4_125_1;
                                        unsigned _v4_125_2;
                                        unsigned _v4_125_3;
                                        asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                            : "=r"(_v4_125_0), "=r"(_v4_125_1), "=r"(_v4_125_2), "=r"(_v4_125_3) : "l"((const void*)(dq_accum + at_45 + (0))), "l"(0x14F0000000000000ULL) : "memory");
                                        old_45[0 + 0] = __uint_as_float(_v4_125_0);
                                        old_45[0 + 1] = __uint_as_float(_v4_125_1);
                                        old_45[0 + 2] = __uint_as_float(_v4_125_2);
                                        old_45[0 + 3] = __uint_as_float(_v4_125_3);
                                    }
                                    acc_5[20] = old_45[0] + acc_5[20];
                                    acc_5[21] = old_45[1] + acc_5[21];
                                    acc_5[22] = old_45[2] + acc_5[22];
                                    acc_5[23] = old_45[3] + acc_5[23];
                                }
                                if (last == 0) {
                                    asm volatile("st.global.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(dq_accum + at_45), "f"(acc_5[(20) + 0]), "f"(acc_5[(20) + 1]), "f"(acc_5[(20) + 2]), "f"(acc_5[(20) + 3]), "l"(0x14F0000000000000ULL) : "memory");
                                }
                                long long at_46 = tile_off + 7424ll;
                                if (add != 0) {
                                    float old_46[4];
                                    {
                                        unsigned _v4_126_0;
                                        unsigned _v4_126_1;
                                        unsigned _v4_126_2;
                                        unsigned _v4_126_3;
                                        asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                            : "=r"(_v4_126_0), "=r"(_v4_126_1), "=r"(_v4_126_2), "=r"(_v4_126_3) : "l"((const void*)(dq_accum + at_46 + (0))), "l"(0x14F0000000000000ULL) : "memory");
                                        old_46[0 + 0] = __uint_as_float(_v4_126_0);
                                        old_46[0 + 1] = __uint_as_float(_v4_126_1);
                                        old_46[0 + 2] = __uint_as_float(_v4_126_2);
                                        old_46[0 + 3] = __uint_as_float(_v4_126_3);
                                    }
                                    acc_5[24] = old_46[0] + acc_5[24];
                                    acc_5[25] = old_46[1] + acc_5[25];
                                    acc_5[26] = old_46[2] + acc_5[26];
                                    acc_5[27] = old_46[3] + acc_5[27];
                                }
                                if (last == 0) {
                                    asm volatile("st.global.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(dq_accum + at_46), "f"(acc_5[(24) + 0]), "f"(acc_5[(24) + 1]), "f"(acc_5[(24) + 2]), "f"(acc_5[(24) + 3]), "l"(0x14F0000000000000ULL) : "memory");
                                }
                                long long at_47 = tile_off + 7936ll;
                                if (add != 0) {
                                    float old_47[4];
                                    {
                                        unsigned _v4_127_0;
                                        unsigned _v4_127_1;
                                        unsigned _v4_127_2;
                                        unsigned _v4_127_3;
                                        asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                            : "=r"(_v4_127_0), "=r"(_v4_127_1), "=r"(_v4_127_2), "=r"(_v4_127_3) : "l"((const void*)(dq_accum + at_47 + (0))), "l"(0x14F0000000000000ULL) : "memory");
                                        old_47[0 + 0] = __uint_as_float(_v4_127_0);
                                        old_47[0 + 1] = __uint_as_float(_v4_127_1);
                                        old_47[0 + 2] = __uint_as_float(_v4_127_2);
                                        old_47[0 + 3] = __uint_as_float(_v4_127_3);
                                    }
                                    acc_5[28] = old_47[0] + acc_5[28];
                                    acc_5[29] = old_47[1] + acc_5[29];
                                    acc_5[30] = old_47[2] + acc_5[30];
                                    acc_5[31] = old_47[3] + acc_5[31];
                                }
                                if (last == 0) {
                                    asm volatile("st.global.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(dq_accum + at_47), "f"(acc_5[(28) + 0]), "f"(acc_5[(28) + 1]), "f"(acc_5[(28) + 2]), "f"(acc_5[(28) + 3]), "l"(0x14F0000000000000ULL) : "memory");
                                }
                                if (last != 0) {
                                    unsigned int out_5[16];
                                    float2 _f32x2_mul_a_128 = make_float2(acc_5[0], acc_5[1]);
                                    float2 scaled_80;
                                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_80) : "l"(*(const unsigned long long*)&_f32x2_mul_a_128), "l"(*(const unsigned long long*)&scale2));
                                    out_5[0] = __as_u32(__float22bfloat162_rn(scaled_80));
                                    float2 _f32x2_mul_a_129 = make_float2(acc_5[2], acc_5[3]);
                                    float2 scaled_81;
                                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_81) : "l"(*(const unsigned long long*)&_f32x2_mul_a_129), "l"(*(const unsigned long long*)&scale2));
                                    out_5[1] = __as_u32(__float22bfloat162_rn(scaled_81));
                                    float2 _f32x2_mul_a_130 = make_float2(acc_5[4], acc_5[5]);
                                    float2 scaled_82;
                                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_82) : "l"(*(const unsigned long long*)&_f32x2_mul_a_130), "l"(*(const unsigned long long*)&scale2));
                                    out_5[2] = __as_u32(__float22bfloat162_rn(scaled_82));
                                    float2 _f32x2_mul_a_131 = make_float2(acc_5[6], acc_5[7]);
                                    float2 scaled_83;
                                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_83) : "l"(*(const unsigned long long*)&_f32x2_mul_a_131), "l"(*(const unsigned long long*)&scale2));
                                    out_5[3] = __as_u32(__float22bfloat162_rn(scaled_83));
                                    float2 _f32x2_mul_a_132 = make_float2(acc_5[8], acc_5[9]);
                                    float2 scaled_84;
                                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_84) : "l"(*(const unsigned long long*)&_f32x2_mul_a_132), "l"(*(const unsigned long long*)&scale2));
                                    out_5[4] = __as_u32(__float22bfloat162_rn(scaled_84));
                                    float2 _f32x2_mul_a_133 = make_float2(acc_5[10], acc_5[11]);
                                    float2 scaled_85;
                                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_85) : "l"(*(const unsigned long long*)&_f32x2_mul_a_133), "l"(*(const unsigned long long*)&scale2));
                                    out_5[5] = __as_u32(__float22bfloat162_rn(scaled_85));
                                    float2 _f32x2_mul_a_134 = make_float2(acc_5[12], acc_5[13]);
                                    float2 scaled_86;
                                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_86) : "l"(*(const unsigned long long*)&_f32x2_mul_a_134), "l"(*(const unsigned long long*)&scale2));
                                    out_5[6] = __as_u32(__float22bfloat162_rn(scaled_86));
                                    float2 _f32x2_mul_a_135 = make_float2(acc_5[14], acc_5[15]);
                                    float2 scaled_87;
                                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_87) : "l"(*(const unsigned long long*)&_f32x2_mul_a_135), "l"(*(const unsigned long long*)&scale2));
                                    out_5[7] = __as_u32(__float22bfloat162_rn(scaled_87));
                                    float2 _f32x2_mul_a_136 = make_float2(acc_5[16], acc_5[17]);
                                    float2 scaled_88;
                                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_88) : "l"(*(const unsigned long long*)&_f32x2_mul_a_136), "l"(*(const unsigned long long*)&scale2));
                                    out_5[8] = __as_u32(__float22bfloat162_rn(scaled_88));
                                    float2 _f32x2_mul_a_137 = make_float2(acc_5[18], acc_5[19]);
                                    float2 scaled_89;
                                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_89) : "l"(*(const unsigned long long*)&_f32x2_mul_a_137), "l"(*(const unsigned long long*)&scale2));
                                    out_5[9] = __as_u32(__float22bfloat162_rn(scaled_89));
                                    float2 _f32x2_mul_a_138 = make_float2(acc_5[20], acc_5[21]);
                                    float2 scaled_90;
                                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_90) : "l"(*(const unsigned long long*)&_f32x2_mul_a_138), "l"(*(const unsigned long long*)&scale2));
                                    out_5[10] = __as_u32(__float22bfloat162_rn(scaled_90));
                                    float2 _f32x2_mul_a_139 = make_float2(acc_5[22], acc_5[23]);
                                    float2 scaled_91;
                                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_91) : "l"(*(const unsigned long long*)&_f32x2_mul_a_139), "l"(*(const unsigned long long*)&scale2));
                                    out_5[11] = __as_u32(__float22bfloat162_rn(scaled_91));
                                    float2 _f32x2_mul_a_140 = make_float2(acc_5[24], acc_5[25]);
                                    float2 scaled_92;
                                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_92) : "l"(*(const unsigned long long*)&_f32x2_mul_a_140), "l"(*(const unsigned long long*)&scale2));
                                    out_5[12] = __as_u32(__float22bfloat162_rn(scaled_92));
                                    float2 _f32x2_mul_a_141 = make_float2(acc_5[26], acc_5[27]);
                                    float2 scaled_93;
                                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_93) : "l"(*(const unsigned long long*)&_f32x2_mul_a_141), "l"(*(const unsigned long long*)&scale2));
                                    out_5[13] = __as_u32(__float22bfloat162_rn(scaled_93));
                                    float2 _f32x2_mul_a_142 = make_float2(acc_5[28], acc_5[29]);
                                    float2 scaled_94;
                                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_94) : "l"(*(const unsigned long long*)&_f32x2_mul_a_142), "l"(*(const unsigned long long*)&scale2));
                                    out_5[14] = __as_u32(__float22bfloat162_rn(scaled_94));
                                    float2 _f32x2_mul_a_143 = make_float2(acc_5[30], acc_5[31]);
                                    float2 scaled_95;
                                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_95) : "l"(*(const unsigned long long*)&_f32x2_mul_a_143), "l"(*(const unsigned long long*)&scale2));
                                    out_5[15] = __as_u32(__float22bfloat162_rn(scaled_95));
                                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %9, 0; @p_ st.global.L2::evict_first.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8}; }" :: "l"(reinterpret_cast<unsigned int*>(dQ + (dq_row + 160ull))), "r"(out_5[(0) + 0]), "r"(out_5[(0) + 1]), "r"(out_5[(0) + 2]), "r"(out_5[(0) + 3]), "r"(out_5[(0) + 4]), "r"(out_5[(0) + 5]), "r"(out_5[(0) + 6]), "r"(out_5[(0) + 7]), "r"((unsigned int)(tok < q_len)) : "memory");
                                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %9, 0; @p_ st.global.L2::evict_first.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8}; }" :: "l"(reinterpret_cast<unsigned int*>(dQ + (dq_row + 176ull))), "r"(out_5[(8) + 0]), "r"(out_5[(8) + 1]), "r"(out_5[(8) + 2]), "r"(out_5[(8) + 3]), "r"(out_5[(8) + 4]), "r"(out_5[(8) + 5]), "r"(out_5[(8) + 6]), "r"(out_5[(8) + 7]), "r"((unsigned int)(tok < q_len)) : "memory");
                                }
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
                                        unsigned _v4_144_0;
                                        unsigned _v4_144_1;
                                        unsigned _v4_144_2;
                                        unsigned _v4_144_3;
                                        asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                            : "=r"(_v4_144_0), "=r"(_v4_144_1), "=r"(_v4_144_2), "=r"(_v4_144_3) : "l"((const void*)(dq_accum + at_48 + (0))), "l"(0x14F0000000000000ULL) : "memory");
                                        old_48[0 + 0] = __uint_as_float(_v4_144_0);
                                        old_48[0 + 1] = __uint_as_float(_v4_144_1);
                                        old_48[0 + 2] = __uint_as_float(_v4_144_2);
                                        old_48[0 + 3] = __uint_as_float(_v4_144_3);
                                    }
                                    acc_6[0] = old_48[0] + acc_6[0];
                                    acc_6[1] = old_48[1] + acc_6[1];
                                    acc_6[2] = old_48[2] + acc_6[2];
                                    acc_6[3] = old_48[3] + acc_6[3];
                                }
                                if (last == 0) {
                                    asm volatile("st.global.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(dq_accum + at_48), "f"(acc_6[(0) + 0]), "f"(acc_6[(0) + 1]), "f"(acc_6[(0) + 2]), "f"(acc_6[(0) + 3]), "l"(0x14F0000000000000ULL) : "memory");
                                }
                                long long at_49 = tile_off + 8960ll;
                                if (add != 0) {
                                    float old_49[4];
                                    {
                                        unsigned _v4_145_0;
                                        unsigned _v4_145_1;
                                        unsigned _v4_145_2;
                                        unsigned _v4_145_3;
                                        asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                            : "=r"(_v4_145_0), "=r"(_v4_145_1), "=r"(_v4_145_2), "=r"(_v4_145_3) : "l"((const void*)(dq_accum + at_49 + (0))), "l"(0x14F0000000000000ULL) : "memory");
                                        old_49[0 + 0] = __uint_as_float(_v4_145_0);
                                        old_49[0 + 1] = __uint_as_float(_v4_145_1);
                                        old_49[0 + 2] = __uint_as_float(_v4_145_2);
                                        old_49[0 + 3] = __uint_as_float(_v4_145_3);
                                    }
                                    acc_6[4] = old_49[0] + acc_6[4];
                                    acc_6[5] = old_49[1] + acc_6[5];
                                    acc_6[6] = old_49[2] + acc_6[6];
                                    acc_6[7] = old_49[3] + acc_6[7];
                                }
                                if (last == 0) {
                                    asm volatile("st.global.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(dq_accum + at_49), "f"(acc_6[(4) + 0]), "f"(acc_6[(4) + 1]), "f"(acc_6[(4) + 2]), "f"(acc_6[(4) + 3]), "l"(0x14F0000000000000ULL) : "memory");
                                }
                                long long at_50 = tile_off + 9472ll;
                                if (add != 0) {
                                    float old_50[4];
                                    {
                                        unsigned _v4_146_0;
                                        unsigned _v4_146_1;
                                        unsigned _v4_146_2;
                                        unsigned _v4_146_3;
                                        asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                            : "=r"(_v4_146_0), "=r"(_v4_146_1), "=r"(_v4_146_2), "=r"(_v4_146_3) : "l"((const void*)(dq_accum + at_50 + (0))), "l"(0x14F0000000000000ULL) : "memory");
                                        old_50[0 + 0] = __uint_as_float(_v4_146_0);
                                        old_50[0 + 1] = __uint_as_float(_v4_146_1);
                                        old_50[0 + 2] = __uint_as_float(_v4_146_2);
                                        old_50[0 + 3] = __uint_as_float(_v4_146_3);
                                    }
                                    acc_6[8] = old_50[0] + acc_6[8];
                                    acc_6[9] = old_50[1] + acc_6[9];
                                    acc_6[10] = old_50[2] + acc_6[10];
                                    acc_6[11] = old_50[3] + acc_6[11];
                                }
                                if (last == 0) {
                                    asm volatile("st.global.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(dq_accum + at_50), "f"(acc_6[(8) + 0]), "f"(acc_6[(8) + 1]), "f"(acc_6[(8) + 2]), "f"(acc_6[(8) + 3]), "l"(0x14F0000000000000ULL) : "memory");
                                }
                                long long at_51 = tile_off + 9984ll;
                                if (add != 0) {
                                    float old_51[4];
                                    {
                                        unsigned _v4_147_0;
                                        unsigned _v4_147_1;
                                        unsigned _v4_147_2;
                                        unsigned _v4_147_3;
                                        asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                            : "=r"(_v4_147_0), "=r"(_v4_147_1), "=r"(_v4_147_2), "=r"(_v4_147_3) : "l"((const void*)(dq_accum + at_51 + (0))), "l"(0x14F0000000000000ULL) : "memory");
                                        old_51[0 + 0] = __uint_as_float(_v4_147_0);
                                        old_51[0 + 1] = __uint_as_float(_v4_147_1);
                                        old_51[0 + 2] = __uint_as_float(_v4_147_2);
                                        old_51[0 + 3] = __uint_as_float(_v4_147_3);
                                    }
                                    acc_6[12] = old_51[0] + acc_6[12];
                                    acc_6[13] = old_51[1] + acc_6[13];
                                    acc_6[14] = old_51[2] + acc_6[14];
                                    acc_6[15] = old_51[3] + acc_6[15];
                                }
                                if (last == 0) {
                                    asm volatile("st.global.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(dq_accum + at_51), "f"(acc_6[(12) + 0]), "f"(acc_6[(12) + 1]), "f"(acc_6[(12) + 2]), "f"(acc_6[(12) + 3]), "l"(0x14F0000000000000ULL) : "memory");
                                }
                                long long at_52 = tile_off + 10496ll;
                                if (add != 0) {
                                    float old_52[4];
                                    {
                                        unsigned _v4_148_0;
                                        unsigned _v4_148_1;
                                        unsigned _v4_148_2;
                                        unsigned _v4_148_3;
                                        asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                            : "=r"(_v4_148_0), "=r"(_v4_148_1), "=r"(_v4_148_2), "=r"(_v4_148_3) : "l"((const void*)(dq_accum + at_52 + (0))), "l"(0x14F0000000000000ULL) : "memory");
                                        old_52[0 + 0] = __uint_as_float(_v4_148_0);
                                        old_52[0 + 1] = __uint_as_float(_v4_148_1);
                                        old_52[0 + 2] = __uint_as_float(_v4_148_2);
                                        old_52[0 + 3] = __uint_as_float(_v4_148_3);
                                    }
                                    acc_6[16] = old_52[0] + acc_6[16];
                                    acc_6[17] = old_52[1] + acc_6[17];
                                    acc_6[18] = old_52[2] + acc_6[18];
                                    acc_6[19] = old_52[3] + acc_6[19];
                                }
                                if (last == 0) {
                                    asm volatile("st.global.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(dq_accum + at_52), "f"(acc_6[(16) + 0]), "f"(acc_6[(16) + 1]), "f"(acc_6[(16) + 2]), "f"(acc_6[(16) + 3]), "l"(0x14F0000000000000ULL) : "memory");
                                }
                                long long at_53 = tile_off + 11008ll;
                                if (add != 0) {
                                    float old_53[4];
                                    {
                                        unsigned _v4_149_0;
                                        unsigned _v4_149_1;
                                        unsigned _v4_149_2;
                                        unsigned _v4_149_3;
                                        asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                            : "=r"(_v4_149_0), "=r"(_v4_149_1), "=r"(_v4_149_2), "=r"(_v4_149_3) : "l"((const void*)(dq_accum + at_53 + (0))), "l"(0x14F0000000000000ULL) : "memory");
                                        old_53[0 + 0] = __uint_as_float(_v4_149_0);
                                        old_53[0 + 1] = __uint_as_float(_v4_149_1);
                                        old_53[0 + 2] = __uint_as_float(_v4_149_2);
                                        old_53[0 + 3] = __uint_as_float(_v4_149_3);
                                    }
                                    acc_6[20] = old_53[0] + acc_6[20];
                                    acc_6[21] = old_53[1] + acc_6[21];
                                    acc_6[22] = old_53[2] + acc_6[22];
                                    acc_6[23] = old_53[3] + acc_6[23];
                                }
                                if (last == 0) {
                                    asm volatile("st.global.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(dq_accum + at_53), "f"(acc_6[(20) + 0]), "f"(acc_6[(20) + 1]), "f"(acc_6[(20) + 2]), "f"(acc_6[(20) + 3]), "l"(0x14F0000000000000ULL) : "memory");
                                }
                                long long at_54 = tile_off + 11520ll;
                                if (add != 0) {
                                    float old_54[4];
                                    {
                                        unsigned _v4_150_0;
                                        unsigned _v4_150_1;
                                        unsigned _v4_150_2;
                                        unsigned _v4_150_3;
                                        asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                            : "=r"(_v4_150_0), "=r"(_v4_150_1), "=r"(_v4_150_2), "=r"(_v4_150_3) : "l"((const void*)(dq_accum + at_54 + (0))), "l"(0x14F0000000000000ULL) : "memory");
                                        old_54[0 + 0] = __uint_as_float(_v4_150_0);
                                        old_54[0 + 1] = __uint_as_float(_v4_150_1);
                                        old_54[0 + 2] = __uint_as_float(_v4_150_2);
                                        old_54[0 + 3] = __uint_as_float(_v4_150_3);
                                    }
                                    acc_6[24] = old_54[0] + acc_6[24];
                                    acc_6[25] = old_54[1] + acc_6[25];
                                    acc_6[26] = old_54[2] + acc_6[26];
                                    acc_6[27] = old_54[3] + acc_6[27];
                                }
                                if (last == 0) {
                                    asm volatile("st.global.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(dq_accum + at_54), "f"(acc_6[(24) + 0]), "f"(acc_6[(24) + 1]), "f"(acc_6[(24) + 2]), "f"(acc_6[(24) + 3]), "l"(0x14F0000000000000ULL) : "memory");
                                }
                                long long at_55 = tile_off + 12032ll;
                                if (add != 0) {
                                    float old_55[4];
                                    {
                                        unsigned _v4_151_0;
                                        unsigned _v4_151_1;
                                        unsigned _v4_151_2;
                                        unsigned _v4_151_3;
                                        asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                            : "=r"(_v4_151_0), "=r"(_v4_151_1), "=r"(_v4_151_2), "=r"(_v4_151_3) : "l"((const void*)(dq_accum + at_55 + (0))), "l"(0x14F0000000000000ULL) : "memory");
                                        old_55[0 + 0] = __uint_as_float(_v4_151_0);
                                        old_55[0 + 1] = __uint_as_float(_v4_151_1);
                                        old_55[0 + 2] = __uint_as_float(_v4_151_2);
                                        old_55[0 + 3] = __uint_as_float(_v4_151_3);
                                    }
                                    acc_6[28] = old_55[0] + acc_6[28];
                                    acc_6[29] = old_55[1] + acc_6[29];
                                    acc_6[30] = old_55[2] + acc_6[30];
                                    acc_6[31] = old_55[3] + acc_6[31];
                                }
                                if (last == 0) {
                                    asm volatile("st.global.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(dq_accum + at_55), "f"(acc_6[(28) + 0]), "f"(acc_6[(28) + 1]), "f"(acc_6[(28) + 2]), "f"(acc_6[(28) + 3]), "l"(0x14F0000000000000ULL) : "memory");
                                }
                                if (last != 0) {
                                    unsigned int out_6[16];
                                    float2 _f32x2_mul_a_152 = make_float2(acc_6[0], acc_6[1]);
                                    float2 scaled_96;
                                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_96) : "l"(*(const unsigned long long*)&_f32x2_mul_a_152), "l"(*(const unsigned long long*)&scale2));
                                    out_6[0] = __as_u32(__float22bfloat162_rn(scaled_96));
                                    float2 _f32x2_mul_a_153 = make_float2(acc_6[2], acc_6[3]);
                                    float2 scaled_97;
                                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_97) : "l"(*(const unsigned long long*)&_f32x2_mul_a_153), "l"(*(const unsigned long long*)&scale2));
                                    out_6[1] = __as_u32(__float22bfloat162_rn(scaled_97));
                                    float2 _f32x2_mul_a_154 = make_float2(acc_6[4], acc_6[5]);
                                    float2 scaled_98;
                                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_98) : "l"(*(const unsigned long long*)&_f32x2_mul_a_154), "l"(*(const unsigned long long*)&scale2));
                                    out_6[2] = __as_u32(__float22bfloat162_rn(scaled_98));
                                    float2 _f32x2_mul_a_155 = make_float2(acc_6[6], acc_6[7]);
                                    float2 scaled_99;
                                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_99) : "l"(*(const unsigned long long*)&_f32x2_mul_a_155), "l"(*(const unsigned long long*)&scale2));
                                    out_6[3] = __as_u32(__float22bfloat162_rn(scaled_99));
                                    float2 _f32x2_mul_a_156 = make_float2(acc_6[8], acc_6[9]);
                                    float2 scaled_100;
                                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_100) : "l"(*(const unsigned long long*)&_f32x2_mul_a_156), "l"(*(const unsigned long long*)&scale2));
                                    out_6[4] = __as_u32(__float22bfloat162_rn(scaled_100));
                                    float2 _f32x2_mul_a_157 = make_float2(acc_6[10], acc_6[11]);
                                    float2 scaled_101;
                                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_101) : "l"(*(const unsigned long long*)&_f32x2_mul_a_157), "l"(*(const unsigned long long*)&scale2));
                                    out_6[5] = __as_u32(__float22bfloat162_rn(scaled_101));
                                    float2 _f32x2_mul_a_158 = make_float2(acc_6[12], acc_6[13]);
                                    float2 scaled_102;
                                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_102) : "l"(*(const unsigned long long*)&_f32x2_mul_a_158), "l"(*(const unsigned long long*)&scale2));
                                    out_6[6] = __as_u32(__float22bfloat162_rn(scaled_102));
                                    float2 _f32x2_mul_a_159 = make_float2(acc_6[14], acc_6[15]);
                                    float2 scaled_103;
                                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_103) : "l"(*(const unsigned long long*)&_f32x2_mul_a_159), "l"(*(const unsigned long long*)&scale2));
                                    out_6[7] = __as_u32(__float22bfloat162_rn(scaled_103));
                                    float2 _f32x2_mul_a_160 = make_float2(acc_6[16], acc_6[17]);
                                    float2 scaled_104;
                                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_104) : "l"(*(const unsigned long long*)&_f32x2_mul_a_160), "l"(*(const unsigned long long*)&scale2));
                                    out_6[8] = __as_u32(__float22bfloat162_rn(scaled_104));
                                    float2 _f32x2_mul_a_161 = make_float2(acc_6[18], acc_6[19]);
                                    float2 scaled_105;
                                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_105) : "l"(*(const unsigned long long*)&_f32x2_mul_a_161), "l"(*(const unsigned long long*)&scale2));
                                    out_6[9] = __as_u32(__float22bfloat162_rn(scaled_105));
                                    float2 _f32x2_mul_a_162 = make_float2(acc_6[20], acc_6[21]);
                                    float2 scaled_106;
                                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_106) : "l"(*(const unsigned long long*)&_f32x2_mul_a_162), "l"(*(const unsigned long long*)&scale2));
                                    out_6[10] = __as_u32(__float22bfloat162_rn(scaled_106));
                                    float2 _f32x2_mul_a_163 = make_float2(acc_6[22], acc_6[23]);
                                    float2 scaled_107;
                                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_107) : "l"(*(const unsigned long long*)&_f32x2_mul_a_163), "l"(*(const unsigned long long*)&scale2));
                                    out_6[11] = __as_u32(__float22bfloat162_rn(scaled_107));
                                    float2 _f32x2_mul_a_164 = make_float2(acc_6[24], acc_6[25]);
                                    float2 scaled_108;
                                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_108) : "l"(*(const unsigned long long*)&_f32x2_mul_a_164), "l"(*(const unsigned long long*)&scale2));
                                    out_6[12] = __as_u32(__float22bfloat162_rn(scaled_108));
                                    float2 _f32x2_mul_a_165 = make_float2(acc_6[26], acc_6[27]);
                                    float2 scaled_109;
                                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_109) : "l"(*(const unsigned long long*)&_f32x2_mul_a_165), "l"(*(const unsigned long long*)&scale2));
                                    out_6[13] = __as_u32(__float22bfloat162_rn(scaled_109));
                                    float2 _f32x2_mul_a_166 = make_float2(acc_6[28], acc_6[29]);
                                    float2 scaled_110;
                                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_110) : "l"(*(const unsigned long long*)&_f32x2_mul_a_166), "l"(*(const unsigned long long*)&scale2));
                                    out_6[14] = __as_u32(__float22bfloat162_rn(scaled_110));
                                    float2 _f32x2_mul_a_167 = make_float2(acc_6[30], acc_6[31]);
                                    float2 scaled_111;
                                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_111) : "l"(*(const unsigned long long*)&_f32x2_mul_a_167), "l"(*(const unsigned long long*)&scale2));
                                    out_6[15] = __as_u32(__float22bfloat162_rn(scaled_111));
                                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %9, 0; @p_ st.global.L2::evict_first.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8}; }" :: "l"(reinterpret_cast<unsigned int*>(dQ + (dq_row + 192ull))), "r"(out_6[(0) + 0]), "r"(out_6[(0) + 1]), "r"(out_6[(0) + 2]), "r"(out_6[(0) + 3]), "r"(out_6[(0) + 4]), "r"(out_6[(0) + 5]), "r"(out_6[(0) + 6]), "r"(out_6[(0) + 7]), "r"((unsigned int)(tok < q_len)) : "memory");
                                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %9, 0; @p_ st.global.L2::evict_first.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8}; }" :: "l"(reinterpret_cast<unsigned int*>(dQ + (dq_row + 208ull))), "r"(out_6[(8) + 0]), "r"(out_6[(8) + 1]), "r"(out_6[(8) + 2]), "r"(out_6[(8) + 3]), "r"(out_6[(8) + 4]), "r"(out_6[(8) + 5]), "r"(out_6[(8) + 6]), "r"(out_6[(8) + 7]), "r"((unsigned int)(tok < q_len)) : "memory");
                                }
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
                                        unsigned _v4_168_0;
                                        unsigned _v4_168_1;
                                        unsigned _v4_168_2;
                                        unsigned _v4_168_3;
                                        asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                            : "=r"(_v4_168_0), "=r"(_v4_168_1), "=r"(_v4_168_2), "=r"(_v4_168_3) : "l"((const void*)(dq_accum + at_56 + (0))), "l"(0x14F0000000000000ULL) : "memory");
                                        old_56[0 + 0] = __uint_as_float(_v4_168_0);
                                        old_56[0 + 1] = __uint_as_float(_v4_168_1);
                                        old_56[0 + 2] = __uint_as_float(_v4_168_2);
                                        old_56[0 + 3] = __uint_as_float(_v4_168_3);
                                    }
                                    acc_7[0] = old_56[0] + acc_7[0];
                                    acc_7[1] = old_56[1] + acc_7[1];
                                    acc_7[2] = old_56[2] + acc_7[2];
                                    acc_7[3] = old_56[3] + acc_7[3];
                                }
                                if (last == 0) {
                                    asm volatile("st.global.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(dq_accum + at_56), "f"(acc_7[(0) + 0]), "f"(acc_7[(0) + 1]), "f"(acc_7[(0) + 2]), "f"(acc_7[(0) + 3]), "l"(0x14F0000000000000ULL) : "memory");
                                }
                                long long at_57 = tile_off + 13056ll;
                                if (add != 0) {
                                    float old_57[4];
                                    {
                                        unsigned _v4_169_0;
                                        unsigned _v4_169_1;
                                        unsigned _v4_169_2;
                                        unsigned _v4_169_3;
                                        asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                            : "=r"(_v4_169_0), "=r"(_v4_169_1), "=r"(_v4_169_2), "=r"(_v4_169_3) : "l"((const void*)(dq_accum + at_57 + (0))), "l"(0x14F0000000000000ULL) : "memory");
                                        old_57[0 + 0] = __uint_as_float(_v4_169_0);
                                        old_57[0 + 1] = __uint_as_float(_v4_169_1);
                                        old_57[0 + 2] = __uint_as_float(_v4_169_2);
                                        old_57[0 + 3] = __uint_as_float(_v4_169_3);
                                    }
                                    acc_7[4] = old_57[0] + acc_7[4];
                                    acc_7[5] = old_57[1] + acc_7[5];
                                    acc_7[6] = old_57[2] + acc_7[6];
                                    acc_7[7] = old_57[3] + acc_7[7];
                                }
                                if (last == 0) {
                                    asm volatile("st.global.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(dq_accum + at_57), "f"(acc_7[(4) + 0]), "f"(acc_7[(4) + 1]), "f"(acc_7[(4) + 2]), "f"(acc_7[(4) + 3]), "l"(0x14F0000000000000ULL) : "memory");
                                }
                                long long at_58 = tile_off + 13568ll;
                                if (add != 0) {
                                    float old_58[4];
                                    {
                                        unsigned _v4_170_0;
                                        unsigned _v4_170_1;
                                        unsigned _v4_170_2;
                                        unsigned _v4_170_3;
                                        asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                            : "=r"(_v4_170_0), "=r"(_v4_170_1), "=r"(_v4_170_2), "=r"(_v4_170_3) : "l"((const void*)(dq_accum + at_58 + (0))), "l"(0x14F0000000000000ULL) : "memory");
                                        old_58[0 + 0] = __uint_as_float(_v4_170_0);
                                        old_58[0 + 1] = __uint_as_float(_v4_170_1);
                                        old_58[0 + 2] = __uint_as_float(_v4_170_2);
                                        old_58[0 + 3] = __uint_as_float(_v4_170_3);
                                    }
                                    acc_7[8] = old_58[0] + acc_7[8];
                                    acc_7[9] = old_58[1] + acc_7[9];
                                    acc_7[10] = old_58[2] + acc_7[10];
                                    acc_7[11] = old_58[3] + acc_7[11];
                                }
                                if (last == 0) {
                                    asm volatile("st.global.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(dq_accum + at_58), "f"(acc_7[(8) + 0]), "f"(acc_7[(8) + 1]), "f"(acc_7[(8) + 2]), "f"(acc_7[(8) + 3]), "l"(0x14F0000000000000ULL) : "memory");
                                }
                                long long at_59 = tile_off + 14080ll;
                                if (add != 0) {
                                    float old_59[4];
                                    {
                                        unsigned _v4_171_0;
                                        unsigned _v4_171_1;
                                        unsigned _v4_171_2;
                                        unsigned _v4_171_3;
                                        asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                            : "=r"(_v4_171_0), "=r"(_v4_171_1), "=r"(_v4_171_2), "=r"(_v4_171_3) : "l"((const void*)(dq_accum + at_59 + (0))), "l"(0x14F0000000000000ULL) : "memory");
                                        old_59[0 + 0] = __uint_as_float(_v4_171_0);
                                        old_59[0 + 1] = __uint_as_float(_v4_171_1);
                                        old_59[0 + 2] = __uint_as_float(_v4_171_2);
                                        old_59[0 + 3] = __uint_as_float(_v4_171_3);
                                    }
                                    acc_7[12] = old_59[0] + acc_7[12];
                                    acc_7[13] = old_59[1] + acc_7[13];
                                    acc_7[14] = old_59[2] + acc_7[14];
                                    acc_7[15] = old_59[3] + acc_7[15];
                                }
                                if (last == 0) {
                                    asm volatile("st.global.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(dq_accum + at_59), "f"(acc_7[(12) + 0]), "f"(acc_7[(12) + 1]), "f"(acc_7[(12) + 2]), "f"(acc_7[(12) + 3]), "l"(0x14F0000000000000ULL) : "memory");
                                }
                                long long at_60 = tile_off + 14592ll;
                                if (add != 0) {
                                    float old_60[4];
                                    {
                                        unsigned _v4_172_0;
                                        unsigned _v4_172_1;
                                        unsigned _v4_172_2;
                                        unsigned _v4_172_3;
                                        asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                            : "=r"(_v4_172_0), "=r"(_v4_172_1), "=r"(_v4_172_2), "=r"(_v4_172_3) : "l"((const void*)(dq_accum + at_60 + (0))), "l"(0x14F0000000000000ULL) : "memory");
                                        old_60[0 + 0] = __uint_as_float(_v4_172_0);
                                        old_60[0 + 1] = __uint_as_float(_v4_172_1);
                                        old_60[0 + 2] = __uint_as_float(_v4_172_2);
                                        old_60[0 + 3] = __uint_as_float(_v4_172_3);
                                    }
                                    acc_7[16] = old_60[0] + acc_7[16];
                                    acc_7[17] = old_60[1] + acc_7[17];
                                    acc_7[18] = old_60[2] + acc_7[18];
                                    acc_7[19] = old_60[3] + acc_7[19];
                                }
                                if (last == 0) {
                                    asm volatile("st.global.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(dq_accum + at_60), "f"(acc_7[(16) + 0]), "f"(acc_7[(16) + 1]), "f"(acc_7[(16) + 2]), "f"(acc_7[(16) + 3]), "l"(0x14F0000000000000ULL) : "memory");
                                }
                                long long at_61 = tile_off + 15104ll;
                                if (add != 0) {
                                    float old_61[4];
                                    {
                                        unsigned _v4_173_0;
                                        unsigned _v4_173_1;
                                        unsigned _v4_173_2;
                                        unsigned _v4_173_3;
                                        asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                            : "=r"(_v4_173_0), "=r"(_v4_173_1), "=r"(_v4_173_2), "=r"(_v4_173_3) : "l"((const void*)(dq_accum + at_61 + (0))), "l"(0x14F0000000000000ULL) : "memory");
                                        old_61[0 + 0] = __uint_as_float(_v4_173_0);
                                        old_61[0 + 1] = __uint_as_float(_v4_173_1);
                                        old_61[0 + 2] = __uint_as_float(_v4_173_2);
                                        old_61[0 + 3] = __uint_as_float(_v4_173_3);
                                    }
                                    acc_7[20] = old_61[0] + acc_7[20];
                                    acc_7[21] = old_61[1] + acc_7[21];
                                    acc_7[22] = old_61[2] + acc_7[22];
                                    acc_7[23] = old_61[3] + acc_7[23];
                                }
                                if (last == 0) {
                                    asm volatile("st.global.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(dq_accum + at_61), "f"(acc_7[(20) + 0]), "f"(acc_7[(20) + 1]), "f"(acc_7[(20) + 2]), "f"(acc_7[(20) + 3]), "l"(0x14F0000000000000ULL) : "memory");
                                }
                                long long at_62 = tile_off + 15616ll;
                                if (add != 0) {
                                    float old_62[4];
                                    {
                                        unsigned _v4_174_0;
                                        unsigned _v4_174_1;
                                        unsigned _v4_174_2;
                                        unsigned _v4_174_3;
                                        asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                            : "=r"(_v4_174_0), "=r"(_v4_174_1), "=r"(_v4_174_2), "=r"(_v4_174_3) : "l"((const void*)(dq_accum + at_62 + (0))), "l"(0x14F0000000000000ULL) : "memory");
                                        old_62[0 + 0] = __uint_as_float(_v4_174_0);
                                        old_62[0 + 1] = __uint_as_float(_v4_174_1);
                                        old_62[0 + 2] = __uint_as_float(_v4_174_2);
                                        old_62[0 + 3] = __uint_as_float(_v4_174_3);
                                    }
                                    acc_7[24] = old_62[0] + acc_7[24];
                                    acc_7[25] = old_62[1] + acc_7[25];
                                    acc_7[26] = old_62[2] + acc_7[26];
                                    acc_7[27] = old_62[3] + acc_7[27];
                                }
                                if (last == 0) {
                                    asm volatile("st.global.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(dq_accum + at_62), "f"(acc_7[(24) + 0]), "f"(acc_7[(24) + 1]), "f"(acc_7[(24) + 2]), "f"(acc_7[(24) + 3]), "l"(0x14F0000000000000ULL) : "memory");
                                }
                                long long at_63 = tile_off + 16128ll;
                                if (add != 0) {
                                    float old_63[4];
                                    {
                                        unsigned _v4_175_0;
                                        unsigned _v4_175_1;
                                        unsigned _v4_175_2;
                                        unsigned _v4_175_3;
                                        asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                            : "=r"(_v4_175_0), "=r"(_v4_175_1), "=r"(_v4_175_2), "=r"(_v4_175_3) : "l"((const void*)(dq_accum + at_63 + (0))), "l"(0x14F0000000000000ULL) : "memory");
                                        old_63[0 + 0] = __uint_as_float(_v4_175_0);
                                        old_63[0 + 1] = __uint_as_float(_v4_175_1);
                                        old_63[0 + 2] = __uint_as_float(_v4_175_2);
                                        old_63[0 + 3] = __uint_as_float(_v4_175_3);
                                    }
                                    acc_7[28] = old_63[0] + acc_7[28];
                                    acc_7[29] = old_63[1] + acc_7[29];
                                    acc_7[30] = old_63[2] + acc_7[30];
                                    acc_7[31] = old_63[3] + acc_7[31];
                                }
                                if (last == 0) {
                                    asm volatile("st.global.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(dq_accum + at_63), "f"(acc_7[(28) + 0]), "f"(acc_7[(28) + 1]), "f"(acc_7[(28) + 2]), "f"(acc_7[(28) + 3]), "l"(0x14F0000000000000ULL) : "memory");
                                }
                                if (last != 0) {
                                    unsigned int out_7[16];
                                    float2 _f32x2_mul_a_176 = make_float2(acc_7[0], acc_7[1]);
                                    float2 scaled_112;
                                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_112) : "l"(*(const unsigned long long*)&_f32x2_mul_a_176), "l"(*(const unsigned long long*)&scale2));
                                    out_7[0] = __as_u32(__float22bfloat162_rn(scaled_112));
                                    float2 _f32x2_mul_a_177 = make_float2(acc_7[2], acc_7[3]);
                                    float2 scaled_113;
                                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_113) : "l"(*(const unsigned long long*)&_f32x2_mul_a_177), "l"(*(const unsigned long long*)&scale2));
                                    out_7[1] = __as_u32(__float22bfloat162_rn(scaled_113));
                                    float2 _f32x2_mul_a_178 = make_float2(acc_7[4], acc_7[5]);
                                    float2 scaled_114;
                                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_114) : "l"(*(const unsigned long long*)&_f32x2_mul_a_178), "l"(*(const unsigned long long*)&scale2));
                                    out_7[2] = __as_u32(__float22bfloat162_rn(scaled_114));
                                    float2 _f32x2_mul_a_179 = make_float2(acc_7[6], acc_7[7]);
                                    float2 scaled_115;
                                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_115) : "l"(*(const unsigned long long*)&_f32x2_mul_a_179), "l"(*(const unsigned long long*)&scale2));
                                    out_7[3] = __as_u32(__float22bfloat162_rn(scaled_115));
                                    float2 _f32x2_mul_a_180 = make_float2(acc_7[8], acc_7[9]);
                                    float2 scaled_116;
                                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_116) : "l"(*(const unsigned long long*)&_f32x2_mul_a_180), "l"(*(const unsigned long long*)&scale2));
                                    out_7[4] = __as_u32(__float22bfloat162_rn(scaled_116));
                                    float2 _f32x2_mul_a_181 = make_float2(acc_7[10], acc_7[11]);
                                    float2 scaled_117;
                                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_117) : "l"(*(const unsigned long long*)&_f32x2_mul_a_181), "l"(*(const unsigned long long*)&scale2));
                                    out_7[5] = __as_u32(__float22bfloat162_rn(scaled_117));
                                    float2 _f32x2_mul_a_182 = make_float2(acc_7[12], acc_7[13]);
                                    float2 scaled_118;
                                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_118) : "l"(*(const unsigned long long*)&_f32x2_mul_a_182), "l"(*(const unsigned long long*)&scale2));
                                    out_7[6] = __as_u32(__float22bfloat162_rn(scaled_118));
                                    float2 _f32x2_mul_a_183 = make_float2(acc_7[14], acc_7[15]);
                                    float2 scaled_119;
                                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_119) : "l"(*(const unsigned long long*)&_f32x2_mul_a_183), "l"(*(const unsigned long long*)&scale2));
                                    out_7[7] = __as_u32(__float22bfloat162_rn(scaled_119));
                                    float2 _f32x2_mul_a_184 = make_float2(acc_7[16], acc_7[17]);
                                    float2 scaled_120;
                                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_120) : "l"(*(const unsigned long long*)&_f32x2_mul_a_184), "l"(*(const unsigned long long*)&scale2));
                                    out_7[8] = __as_u32(__float22bfloat162_rn(scaled_120));
                                    float2 _f32x2_mul_a_185 = make_float2(acc_7[18], acc_7[19]);
                                    float2 scaled_121;
                                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_121) : "l"(*(const unsigned long long*)&_f32x2_mul_a_185), "l"(*(const unsigned long long*)&scale2));
                                    out_7[9] = __as_u32(__float22bfloat162_rn(scaled_121));
                                    float2 _f32x2_mul_a_186 = make_float2(acc_7[20], acc_7[21]);
                                    float2 scaled_122;
                                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_122) : "l"(*(const unsigned long long*)&_f32x2_mul_a_186), "l"(*(const unsigned long long*)&scale2));
                                    out_7[10] = __as_u32(__float22bfloat162_rn(scaled_122));
                                    float2 _f32x2_mul_a_187 = make_float2(acc_7[22], acc_7[23]);
                                    float2 scaled_123;
                                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_123) : "l"(*(const unsigned long long*)&_f32x2_mul_a_187), "l"(*(const unsigned long long*)&scale2));
                                    out_7[11] = __as_u32(__float22bfloat162_rn(scaled_123));
                                    float2 _f32x2_mul_a_188 = make_float2(acc_7[24], acc_7[25]);
                                    float2 scaled_124;
                                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_124) : "l"(*(const unsigned long long*)&_f32x2_mul_a_188), "l"(*(const unsigned long long*)&scale2));
                                    out_7[12] = __as_u32(__float22bfloat162_rn(scaled_124));
                                    float2 _f32x2_mul_a_189 = make_float2(acc_7[26], acc_7[27]);
                                    float2 scaled_125;
                                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_125) : "l"(*(const unsigned long long*)&_f32x2_mul_a_189), "l"(*(const unsigned long long*)&scale2));
                                    out_7[13] = __as_u32(__float22bfloat162_rn(scaled_125));
                                    float2 _f32x2_mul_a_190 = make_float2(acc_7[28], acc_7[29]);
                                    float2 scaled_126;
                                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_126) : "l"(*(const unsigned long long*)&_f32x2_mul_a_190), "l"(*(const unsigned long long*)&scale2));
                                    out_7[14] = __as_u32(__float22bfloat162_rn(scaled_126));
                                    float2 _f32x2_mul_a_191 = make_float2(acc_7[30], acc_7[31]);
                                    float2 scaled_127;
                                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_127) : "l"(*(const unsigned long long*)&_f32x2_mul_a_191), "l"(*(const unsigned long long*)&scale2));
                                    out_7[15] = __as_u32(__float22bfloat162_rn(scaled_127));
                                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %9, 0; @p_ st.global.L2::evict_first.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8}; }" :: "l"(reinterpret_cast<unsigned int*>(dQ + (dq_row + 224ull))), "r"(out_7[(0) + 0]), "r"(out_7[(0) + 1]), "r"(out_7[(0) + 2]), "r"(out_7[(0) + 3]), "r"(out_7[(0) + 4]), "r"(out_7[(0) + 5]), "r"(out_7[(0) + 6]), "r"(out_7[(0) + 7]), "r"((unsigned int)(tok < q_len)) : "memory");
                                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %9, 0; @p_ st.global.L2::evict_first.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8}; }" :: "l"(reinterpret_cast<unsigned int*>(dQ + (dq_row + 240ull))), "r"(out_7[(8) + 0]), "r"(out_7[(8) + 1]), "r"(out_7[(8) + 2]), "r"(out_7[(8) + 3]), "r"(out_7[(8) + 4]), "r"(out_7[(8) + 5]), "r"(out_7[(8) + 6]), "r"(out_7[(8) + 7]), "r"((unsigned int)(tok < q_len)) : "memory");
                                }
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
                            int n_ex_1 = n_end_1;
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
                        int n_ex_2 = n_end_2;
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
