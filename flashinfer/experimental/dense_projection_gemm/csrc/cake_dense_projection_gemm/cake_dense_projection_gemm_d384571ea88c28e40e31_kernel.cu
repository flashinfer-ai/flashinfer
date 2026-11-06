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




__device__ __forceinline__ void elect_commit_cg2_multicast(int mbar_addr, uint16_t cta_mask) {
    asm volatile(
        "{\n\t"
        ".reg .pred leader;\n\t"
        "elect.sync _|leader, 0xFFFFFFFF;\n\t"
        "@leader tcgen05.commit.cta_group::2.mbarrier::arrive::one"
        ".shared::cluster.multicast::cluster.b64 [%0], %1;\n\t"
        "}\n"
        :: "r"(mbar_addr), "h"(cta_mask) : "memory");
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



__device__ __forceinline__ void tma_reduce_add_2d(
    const void* tmap, int x, int y, unsigned smem_addr) {
    asm volatile(
        "cp.reduce.async.bulk.tensor.2d.global.shared::cta.add.tile.bulk_group"
        " [%0, {%1, %2}], [%3];"
        :: "l"(tmap), "r"(x), "r"(y), "r"(smem_addr)
        : "memory");
}



__device__ __forceinline__ void tcgen05_commit_cg2_multicast_leader(
    int mbar_addr, uint16_t cta_mask, uint32_t leader) {
    asm volatile(
        "{\n\t"
        ".reg .pred leader;\n\t"
        "setp.ne.b32 leader, %2, 0;\n\t"
        "@leader tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%0], %1;\n\t"
        "}\n"
        :: "r"(mbar_addr), "h"(cta_mask), "r"(leader) : "memory");
}


__device__ __forceinline__ void tmem_st_x8_u32(int addr, const uint32_t* src) {
    asm volatile(
        "tcgen05.st.sync.aligned.32x32b.x8.b32"
        " [%0], {%1,%2,%3,%4,%5,%6,%7,%8};"
        :: "r"(addr),
           "r"(src[0]), "r"(src[1]), "r"(src[2]), "r"(src[3]),
           "r"(src[4]), "r"(src[5]), "r"(src[6]), "r"(src[7]));
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

__global__ __launch_bounds__(512, LAUNCH_MIN_BLOCKS) __cluster_dims__(2,1,1) void
kernel_cake_dense_projection_gemm_d384571ea88c28e40e31(const __grid_constant__ CUtensorMap A, const __grid_constant__ CUtensorMap B, float* __restrict__ out32, const __grid_constant__ CUtensorMap D, int M, int N, int m_tiles, int n_tiles, int k_iters, int ldo, int units, int head_spans, int head_steps)
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
    #define raw_full_addr (mbar_base + 0)
    #define raw_empty_addr (mbar_base + 48)
    #define b_full_addr (mbar_base + 96)
    #define b_empty_addr (mbar_base + 144)
    #define b_read_addr (mbar_base + 192)
    #define a_full_addr (mbar_base + 240)
    #define a_empty_addr (mbar_base + 256)
    #define chunk_full_addr (mbar_base + 272)
    #define chunk_empty_addr (mbar_base + 288)
    #define mainloop_done_addr (mbar_base + 304)
    #define epilogue_done_addr (mbar_base + 312)
    #define work_full_addr (mbar_base + 320)
    #define work_empty_addr (mbar_base + 352)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    const unsigned int clusters_x = gridDim.x / 2;
    const unsigned int cluster_id = ((blockIdx.z * gridDim.y + blockIdx.y) * clusters_x) + blockIdx.x / 2;
    const unsigned int num_clusters = clusters_x * gridDim.y * gridDim.z;

    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));

    // Prologue
    float* a32 = reinterpret_cast<float*>(smem_raw + 1024);
    const int a32_addr = smem + 1024;
    __nv_bfloat16* b1 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 99328);
    const int b1_addr = smem + 99328;
    __nv_bfloat16* b2 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 103424);
    const int b2_addr = smem + 103424;
    __nv_bfloat16* b3 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 107520);
    const int b3_addr = smem + 107520;
    float* box = reinterpret_cast<float*>(smem_raw + 173056);
    const int box_addr = smem + 173056;
    unsigned int* work_response = reinterpret_cast<unsigned int*>(smem_raw + 189440);
    const int work_response_addr = smem + 189440;

    // Kernel setup ops
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");

    // Mbarrier init (48 barriers at smem_raw[0..384))

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'raw_pipe' ---
            // raw_full: 6 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            mbarrier_init(smem + 8, 1);
            mbarrier_init(smem + 16, 1);
            mbarrier_init(smem + 24, 1);
            mbarrier_init(smem + 32, 1);
            mbarrier_init(smem + 40, 1);
            // raw_empty: 6 barriers, init_count=4
            mbarrier_init(smem + 48, 4);
            mbarrier_init(smem + 56, 4);
            mbarrier_init(smem + 64, 4);
            mbarrier_init(smem + 72, 4);
            mbarrier_init(smem + 80, 4);
            mbarrier_init(smem + 88, 4);
            // --- pipeline 'b_pipe' ---
            // b_full: 6 barriers, init_count=1
            mbarrier_init(smem + 96, 1);
            mbarrier_init(smem + 104, 1);
            mbarrier_init(smem + 112, 1);
            mbarrier_init(smem + 120, 1);
            mbarrier_init(smem + 128, 1);
            mbarrier_init(smem + 136, 1);
            // b_empty: 6 barriers, init_count=1
            mbarrier_init(smem + 144, 1);
            mbarrier_init(smem + 152, 1);
            mbarrier_init(smem + 160, 1);
            mbarrier_init(smem + 168, 1);
            mbarrier_init(smem + 176, 1);
            mbarrier_init(smem + 184, 1);
            // b_read: 6 barriers, init_count=4
            mbarrier_init(smem + 192, 4);
            mbarrier_init(smem + 200, 4);
            mbarrier_init(smem + 208, 4);
            mbarrier_init(smem + 216, 4);
            mbarrier_init(smem + 224, 4);
            mbarrier_init(smem + 232, 4);
            // --- pipeline 'a_pipe' ---
            // a_full: 2 barriers, init_count=8
            mbarrier_init(smem + 240, 8);
            mbarrier_init(smem + 248, 8);
            // a_empty: 2 barriers, init_count=1
            mbarrier_init(smem + 256, 1);
            mbarrier_init(smem + 264, 1);
            // --- pipeline 'chunk_pipe' ---
            // chunk_full: 2 barriers, init_count=1
            mbarrier_init(smem + 272, 1);
            mbarrier_init(smem + 280, 1);
            // chunk_empty: 2 barriers, init_count=16
            mbarrier_init(smem + 288, 16);
            mbarrier_init(smem + 296, 16);
            // --- pipeline 'mainloop_pipe' ---
            // mainloop_done: 1 barriers, init_count=1
            mbarrier_init(smem + 304, 1);
            // epilogue_done: 1 barriers, init_count=16
            mbarrier_init(smem + 312, 16);
            // --- pipeline 'work_pipe' ---
            // work_full: 4 barriers, init_count=1
            mbarrier_init(smem + 320, 1);
            mbarrier_init(smem + 328, 1);
            mbarrier_init(smem + 336, 1);
            mbarrier_init(smem + 344, 1);
            // work_empty: 4 barriers, init_count=804
            mbarrier_init(smem + 352, 804);
            mbarrier_init(smem + 360, 804);
            mbarrier_init(smem + 368, 804);
            mbarrier_init(smem + 376, 804);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();
    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");

    // TMEM alloc (512 columns, 480 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 384);
    if (warp == 0) {
        int _tmem_hold = smem + 384;
        asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(512) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const unsigned int taddr = tmem_addr_storage[0];
    const unsigned int tmem_chunk_acc = taddr;
    const unsigned int tmem_small_acc = taddr + 256;
    const unsigned int tmem_a1 = taddr + 384;
    const unsigned int tmem_a2 = taddr + 416;
    const unsigned int tmem_a3 = taddr + 448;

    // Kernel post-init ops
    if (warp >= 0 && warp <= 3) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 56;");
    }

    // ---- Role: load ----
    if (warp == 0) {
        { // load_main
            unsigned int load_stage = 0;
            unsigned int work_stage = 0;
            unsigned int _phase_work_empty = 1;
            unsigned int _phase_raw_empty = 1;
            unsigned int _phase_work_full = 0;
            if (elect_sync()) {
                asm volatile("griddepcontrol.wait;" ::: "memory");
                unsigned int this_bid = bid;
                #pragma unroll 1
                for (unsigned int _unit_iter = 0; _unit_iter < units; _unit_iter++) {
                    if (cta_rank == 0) {
                        mbarrier_wait_cluster_hint(work_empty_addr + (work_stage) * 8, _phase_work_empty, 10000000);
                        asm volatile(
                            "{\n\t"
                            ".reg .b32 remAddr32;\n\t"
                            "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                            "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [remAddr32], %2;\n\t"
                            "}"
                            :: "r"(work_full_addr + (work_stage) * 8), "r"(0), "r"((uint32_t)(16)) : "memory");
                        asm volatile(
                            "{\n\t"
                            ".reg .b32 remAddr32;\n\t"
                            "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                            "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [remAddr32], %2;\n\t"
                            "}"
                            :: "r"(work_full_addr + (work_stage) * 8), "r"(1), "r"((uint32_t)(16)) : "memory");
                        asm volatile(
                            "fence.proxy.async.shared::cta;\n\t"
                            "clusterlaunchcontrol.try_cancel.async.shared::cta"
                                ".mbarrier::complete_tx::bytes.multicast::cluster::all.b128"
                                " [%0], [%1];"
                            :: "r"(work_response_addr + work_stage * 16 + 0 * 16), "r"(work_full_addr + work_stage * 8)
                            : "memory");
                    }
                    int unit = this_bid / 2;
                    int span = unit / n_tiles;
                    int steps = m_tiles / 2 * k_iters;
                    int spans = units / n_tiles;
                    int _max_0 = ((head_spans) > (1) ? (head_spans) : (1));
                    int in_head = span * head_steps / _max_0;
                    int in_tail = head_steps + (span - head_spans) * (steps - head_steps) / (spans - head_spans);
                    int first = ((span < head_spans) ? in_head : in_tail);
                    int first_step = first;
                    int _max_1 = ((head_spans) > (1) ? (head_spans) : (1));
                    int in_head_0 = (span + 1) * head_steps / _max_1;
                    int in_tail_0 = head_steps + (span + 1 - head_spans) * (steps - head_steps) / (spans - head_spans);
                    int first_0 = ((span + 1 < head_spans) ? in_head_0 : in_tail_0);
                    int last_step = first_0;
                    int first_pair = first_step / k_iters;
                    int num_segments = (last_step - 1) / k_iters - first_pair + 1;
                    #pragma unroll 1
                    for (int seg = 0; seg < num_segments; seg++) {
                        int rank = this_bid % 2;
                        int unit_0 = this_bid / 2;
                        int column_tile = unit_0 % n_tiles;
                        int pair_step = (first_pair + seg) * k_iters;
                        int _max_2 = ((first_step) > (pair_step) ? (first_step) : (pair_step));
                        int k_first = _max_2 - pair_step;
                        int _min_3 = ((last_step) < (pair_step + k_iters) ? (last_step) : (pair_step + k_iters));
                        int k_count = _min_3 - pair_step - k_first;
                        int off_m = ((first_pair + seg) * 2 + rank) * 128;
                        int off_n = column_tile * 128;
                        #pragma unroll 1
                        for (int iter_k = 0; iter_k < k_count; iter_k++) {
                            mbarrier_wait(raw_empty_addr + (load_stage) * 8, _phase_raw_empty);
                            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                            int k0 = (k_first + iter_k) * 32;
                            #pragma unroll
                            for (int p = 0; p < 4; p++) {
                                tma_3d_gmem2smem(a32_addr + load_stage * 16384 + (unsigned int)(p * 4096), (&A), off_m + 32 * p, k0, 0, raw_full_addr + (load_stage) * 8);
                            }
                            mbarrier_arrive_expect_tx(raw_full_addr + (load_stage) * 8, 16384);
                            load_stage += 1;
                            if (load_stage == 6) { load_stage = 0; _phase_raw_empty ^= 1; }
                        }
                    }
                    mbarrier_wait(work_full_addr + (work_stage) * 8, _phase_work_full);
                    uint32_t clc_valid = 0;
                    asm volatile(
                        "{\n\t"
                        ".reg .pred p1;\n\t"
                        ".reg .b128 clc_r;\n\t"
                        "ld.shared.b128 clc_r, [%1];\n\t"
                        "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                        "selp.u32 %0, 1, 0, p1;\n\t"
                        "}\n"
                        : "=r"(clc_valid)
                        : "r"(work_response_addr + work_stage * 16 + 0 * 16)
                        : "memory");
                    uint32_t clc_ctaid = 0;
                    asm volatile(
                        "{\n\t"
                        ".reg .pred p1;\n\t"
                        ".reg .b128 clc_r;\n\t"
                        "ld.shared.b128 clc_r, [%1];\n\t"
                        "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                        "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                        "}\n"
                        : "+r"(clc_ctaid)
                        : "r"(work_response_addr + work_stage * 16 + 0 * 16)
                        : "memory");
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    asm volatile(
                        "{\n\t"
                        ".reg .b32 remAddr32;\n\t"
                        "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                        "}"
                        :: "r"(work_empty_addr + (work_stage) * 8), "r"(0) : "memory");
                    work_stage += 1;
                    if (work_stage == 4) { work_stage = 0; _phase_work_empty ^= 1; _phase_work_full ^= 1; }
                    if (clc_valid == 0) {
                        break;
                    }
                    this_bid = clc_ctaid + (unsigned int)cta_rank;
                }
            }
        }
    }
    // ---- Role: mma ----
    if (warp == 1) {
        { // mma_main
            unsigned int a_stage = 0;
            unsigned int b_stage = 0;
            unsigned int chunk_stage = 0;
            unsigned int work_stage_1 = 0;
            unsigned int _phase_epilogue_done = 1;
            unsigned int _phase_chunk_empty = 1;
            unsigned int _phase_a_full = 0;
            unsigned int _phase_work_full_1 = 0;
            if (cta_rank == 0) {
                unsigned int this_bid_1 = bid;
                #pragma unroll 1
                for (unsigned int _unit_iter_1 = 0; _unit_iter_1 < units; _unit_iter_1++) {
                    int unit_1 = this_bid_1 / 2;
                    int span_1 = unit_1 / n_tiles;
                    int steps_1 = m_tiles / 2 * k_iters;
                    int spans_1 = units / n_tiles;
                    int _max_0_1 = ((head_spans) > (1) ? (head_spans) : (1));
                    int in_head_1 = span_1 * head_steps / _max_0_1;
                    int in_tail_1 = head_steps + (span_1 - head_spans) * (steps_1 - head_steps) / (spans_1 - head_spans);
                    int first_1 = ((span_1 < head_spans) ? in_head_1 : in_tail_1);
                    int first_step_1 = first_1;
                    int _max_1_1 = ((head_spans) > (1) ? (head_spans) : (1));
                    int in_head_3 = (span_1 + 1) * head_steps / _max_1_1;
                    int in_tail_3 = head_steps + (span_1 + 1 - head_spans) * (steps_1 - head_steps) / (spans_1 - head_spans);
                    int first_3 = ((span_1 + 1 < head_spans) ? in_head_3 : in_tail_3);
                    int last_step_1 = first_3;
                    int first_pair_1 = first_step_1 / k_iters;
                    int num_segments_1 = (last_step_1 - 1) / k_iters - first_pair_1 + 1;
                    #pragma unroll 1
                    for (int seg_1 = 0; seg_1 < num_segments_1; seg_1++) {
                        int rank_1 = this_bid_1 % 2;
                        int unit_3 = this_bid_1 / 2;
                        int column_tile_1 = unit_3 % n_tiles;
                        int pair_step_1 = (first_pair_1 + seg_1) * k_iters;
                        int _max_2_1 = ((first_step_1) > (pair_step_1) ? (first_step_1) : (pair_step_1));
                        int k_first_1 = _max_2_1 - pair_step_1;
                        int _min_3_1 = ((last_step_1) < (pair_step_1 + k_iters) ? (last_step_1) : (pair_step_1 + k_iters));
                        int k_count_1 = _min_3_1 - pair_step_1 - k_first_1;
                        int off_m_1 = ((first_pair_1 + seg_1) * 2 + rank_1) * 128;
                        int off_n_1 = column_tile_1 * 128;
                        mbarrier_wait(epilogue_done_addr, _phase_epilogue_done);
                        int num_chunks = (k_count_1 + 2 - 1) / 2;
                        #pragma unroll 1
                        for (int chunk = 0; chunk < num_chunks; chunk++) {
                            mbarrier_wait(chunk_empty_addr + (chunk_stage) * 8, _phase_chunk_empty);
                            #pragma unroll
                            for (int kc = 0; kc < 2; kc++) {
                                int iter_k_1 = chunk * 2 + kc;
                                if (iter_k_1 < k_count_1) {
                                    mbarrier_wait(a_full_addr + (a_stage) * 8, _phase_a_full);
                                    asm volatile("tcgen05.fence::after_thread_sync;");
                                    int big_init = ((kc == 0) ? 1 : 0);
                                    int small_init = ((iter_k_1 == 0) ? 1 : 0);
                                    int big_col = chunk_stage * 128;
                                    int small_col = 0 * 128;
                                    unsigned int a_part = a_stage * 16;
#if __CUDA_ARCH__ == 1070
                                    int _mma_b_lo = ((((b1_addr) >> 4) & 0x7FFF) | 0x800000) + (b_stage) * 768;
#else
                                    int _mma_b_lo = ((((b1_addr) >> 4) & 0x3FFF) | 0x800000) + (b_stage) * 768;
#endif
                                    asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 dhi, blo, ta, id, m0, m1, m2, m3, m4, m5, m6, m7;\n\t"
                    ".reg .b64 db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    "mov.b32 m0, 0; mov.b32 m1, 0; mov.b32 m2, 0; mov.b32 m3, 0;\n\tmov.b32 m4, 0; mov.b32 m5, 0; mov.b32 m6, 0; mov.b32 m7, 0;\n\t"
                    "mov.b32 dhi, 0x80004020;\n\t"
                    "mov.b32 id, 270599312;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 64;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"((tmem_chunk_acc + (big_col))), "r"(_mma_b_lo), "r"(tmem_a1 + a_part), "r"(((big_init) ? 0 : 1)) : "memory");
#if __CUDA_ARCH__ == 1070
                                    int _mma_b_lo_0 = ((((b2_addr) >> 4) & 0x7FFF) | 0x800000) + (b_stage) * 768;
#else
                                    int _mma_b_lo_0 = ((((b2_addr) >> 4) & 0x3FFF) | 0x800000) + (b_stage) * 768;
#endif
                                    asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 dhi, blo, ta, id, m0, m1, m2, m3, m4, m5, m6, m7;\n\t"
                    ".reg .b64 db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    "mov.b32 m0, 0; mov.b32 m1, 0; mov.b32 m2, 0; mov.b32 m3, 0;\n\tmov.b32 m4, 0; mov.b32 m5, 0; mov.b32 m6, 0; mov.b32 m7, 0;\n\t"
                    "mov.b32 dhi, 0x80004020;\n\t"
                    "mov.b32 id, 270599312;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 64;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"((tmem_small_acc + (small_col))), "r"(_mma_b_lo_0), "r"(tmem_a1 + a_part), "r"(((small_init) ? 0 : 1)) : "memory");
#if __CUDA_ARCH__ == 1070
                                    int _mma_b_lo_1 = ((((b1_addr) >> 4) & 0x7FFF) | 0x800000) + (b_stage) * 768;
#else
                                    int _mma_b_lo_1 = ((((b1_addr) >> 4) & 0x3FFF) | 0x800000) + (b_stage) * 768;
#endif
                                    asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 dhi, blo, ta, id, m0, m1, m2, m3, m4, m5, m6, m7;\n\t"
                    ".reg .b64 db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    "mov.b32 m0, 0; mov.b32 m1, 0; mov.b32 m2, 0; mov.b32 m3, 0;\n\tmov.b32 m4, 0; mov.b32 m5, 0; mov.b32 m6, 0; mov.b32 m7, 0;\n\t"
                    "mov.b32 dhi, 0x80004020;\n\t"
                    "mov.b32 id, 270599312;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 64;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"((tmem_small_acc + (small_col))), "r"(_mma_b_lo_1), "r"(tmem_a2 + a_part), "r"(1) : "memory");
#if __CUDA_ARCH__ == 1070
                                    int _mma_b_lo_2 = ((((b3_addr) >> 4) & 0x7FFF) | 0x800000) + (b_stage) * 768;
#else
                                    int _mma_b_lo_2 = ((((b3_addr) >> 4) & 0x3FFF) | 0x800000) + (b_stage) * 768;
#endif
                                    asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 dhi, blo, ta, id, m0, m1, m2, m3, m4, m5, m6, m7;\n\t"
                    ".reg .b64 db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    "mov.b32 m0, 0; mov.b32 m1, 0; mov.b32 m2, 0; mov.b32 m3, 0;\n\tmov.b32 m4, 0; mov.b32 m5, 0; mov.b32 m6, 0; mov.b32 m7, 0;\n\t"
                    "mov.b32 dhi, 0x80004020;\n\t"
                    "mov.b32 id, 270599312;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 64;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"((tmem_small_acc + (small_col))), "r"(_mma_b_lo_2), "r"(tmem_a1 + a_part), "r"(1) : "memory");
#if __CUDA_ARCH__ == 1070
                                    int _mma_b_lo_3 = ((((b2_addr) >> 4) & 0x7FFF) | 0x800000) + (b_stage) * 768;
#else
                                    int _mma_b_lo_3 = ((((b2_addr) >> 4) & 0x3FFF) | 0x800000) + (b_stage) * 768;
#endif
                                    asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 dhi, blo, ta, id, m0, m1, m2, m3, m4, m5, m6, m7;\n\t"
                    ".reg .b64 db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    "mov.b32 m0, 0; mov.b32 m1, 0; mov.b32 m2, 0; mov.b32 m3, 0;\n\tmov.b32 m4, 0; mov.b32 m5, 0; mov.b32 m6, 0; mov.b32 m7, 0;\n\t"
                    "mov.b32 dhi, 0x80004020;\n\t"
                    "mov.b32 id, 270599312;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 64;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"((tmem_small_acc + (small_col))), "r"(_mma_b_lo_3), "r"(tmem_a2 + a_part), "r"(1) : "memory");
#if __CUDA_ARCH__ == 1070
                                    int _mma_b_lo_4 = ((((b1_addr) >> 4) & 0x7FFF) | 0x800000) + (b_stage) * 768;
#else
                                    int _mma_b_lo_4 = ((((b1_addr) >> 4) & 0x3FFF) | 0x800000) + (b_stage) * 768;
#endif
                                    asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 dhi, blo, ta, id, m0, m1, m2, m3, m4, m5, m6, m7;\n\t"
                    ".reg .b64 db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    "mov.b32 m0, 0; mov.b32 m1, 0; mov.b32 m2, 0; mov.b32 m3, 0;\n\tmov.b32 m4, 0; mov.b32 m5, 0; mov.b32 m6, 0; mov.b32 m7, 0;\n\t"
                    "mov.b32 dhi, 0x80004020;\n\t"
                    "mov.b32 id, 270599312;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 64;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"((tmem_small_acc + (small_col))), "r"(_mma_b_lo_4), "r"(tmem_a3 + a_part), "r"(1) : "memory");
                                    uint32_t _elected_4 = elect_sync();
                                    tcgen05_commit_cg2_multicast_leader(a_empty_addr + (a_stage) * 8, (uint16_t)(3), _elected_4);
                                    tcgen05_commit_cg2_multicast_leader(b_empty_addr + (b_stage) * 8, (uint16_t)(3), _elected_4);
                                    a_stage += 1;
                                    if (a_stage == 2) { a_stage = 0; _phase_a_full ^= 1; }
                                    b_stage += 1;
                                    if (b_stage == 6) { b_stage = 0; }
                                }
                            }
                            elect_commit_cg2_multicast(chunk_full_addr + (chunk_stage) * 8, (uint16_t)(3));
                            chunk_stage += 1;
                            if (chunk_stage == 2) { chunk_stage = 0; _phase_chunk_empty ^= 1; }
                        }
                        elect_commit_cg2_multicast(mainloop_done_addr, (uint16_t)(3));
                        _phase_epilogue_done ^= 1;
                    }
                    mbarrier_wait(work_full_addr + (work_stage_1) * 8, _phase_work_full_1);
                    uint32_t clc_valid_1 = 0;
                    asm volatile(
                        "{\n\t"
                        ".reg .pred p1;\n\t"
                        ".reg .b128 clc_r;\n\t"
                        "ld.shared.b128 clc_r, [%1];\n\t"
                        "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                        "selp.u32 %0, 1, 0, p1;\n\t"
                        "}\n"
                        : "=r"(clc_valid_1)
                        : "r"(work_response_addr + work_stage_1 * 16 + 0 * 16)
                        : "memory");
                    uint32_t clc_ctaid_1 = 0;
                    asm volatile(
                        "{\n\t"
                        ".reg .pred p1;\n\t"
                        ".reg .b128 clc_r;\n\t"
                        "ld.shared.b128 clc_r, [%1];\n\t"
                        "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                        "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                        "}\n"
                        : "+r"(clc_ctaid_1)
                        : "r"(work_response_addr + work_stage_1 * 16 + 0 * 16)
                        : "memory");
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    asm volatile(
                        "{\n\t"
                        ".reg .b32 remAddr32;\n\t"
                        "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                        "}"
                        :: "r"(work_empty_addr + (work_stage_1) * 8), "r"(0) : "memory");
                    work_stage_1 += 1;
                    if (work_stage_1 == 4) { work_stage_1 = 0; _phase_work_full_1 ^= 1; }
                    if (clc_valid_1 == 0) {
                        break;
                    }
                    this_bid_1 = clc_ctaid_1 + (unsigned int)cta_rank;
                }
            }
        }
    }
    // ---- Role: bload ----
    if (warp == 2) {
        { // bload_main
            unsigned int b_stage_1 = 0;
            unsigned int work_stage_2 = 0;
            unsigned int _phase_b_empty = 1;
            unsigned int _phase_work_full_2 = 0;
            if (elect_sync()) {
                asm volatile("griddepcontrol.wait;" ::: "memory");
                unsigned int this_bid_2 = bid;
                #pragma unroll 1
                for (unsigned int _unit_iter_2 = 0; _unit_iter_2 < units; _unit_iter_2++) {
                    int unit_2 = this_bid_2 / 2;
                    int span_2 = unit_2 / n_tiles;
                    int steps_2 = m_tiles / 2 * k_iters;
                    int spans_2 = units / n_tiles;
                    int _max_0_2 = ((head_spans) > (1) ? (head_spans) : (1));
                    int in_head_2 = span_2 * head_steps / _max_0_2;
                    int in_tail_2 = head_steps + (span_2 - head_spans) * (steps_2 - head_steps) / (spans_2 - head_spans);
                    int first_2 = ((span_2 < head_spans) ? in_head_2 : in_tail_2);
                    int first_step_2 = first_2;
                    int _max_1_2 = ((head_spans) > (1) ? (head_spans) : (1));
                    int in_head_1_1 = (span_2 + 1) * head_steps / _max_1_2;
                    int in_tail_1_1 = head_steps + (span_2 + 1 - head_spans) * (steps_2 - head_steps) / (spans_2 - head_spans);
                    int first_1_1 = ((span_2 + 1 < head_spans) ? in_head_1_1 : in_tail_1_1);
                    int last_step_2 = first_1_1;
                    int first_pair_2 = first_step_2 / k_iters;
                    int num_segments_2 = (last_step_2 - 1) / k_iters - first_pair_2 + 1;
                    #pragma unroll 1
                    for (int seg_2 = 0; seg_2 < num_segments_2; seg_2++) {
                        int rank_2 = this_bid_2 % 2;
                        int unit_1_1 = this_bid_2 / 2;
                        int column_tile_2 = unit_1_1 % n_tiles;
                        int pair_step_2 = (first_pair_2 + seg_2) * k_iters;
                        int _max_2_2 = ((first_step_2) > (pair_step_2) ? (first_step_2) : (pair_step_2));
                        int k_first_2 = _max_2_2 - pair_step_2;
                        int _min_3_2 = ((last_step_2) < (pair_step_2 + k_iters) ? (last_step_2) : (pair_step_2 + k_iters));
                        int k_count_2 = _min_3_2 - pair_step_2 - k_first_2;
                        int off_m_2 = ((first_pair_2 + seg_2) * 2 + rank_2) * 128;
                        int off_n_2 = column_tile_2 * 128;
                        int b_col = off_n_2 + cta_rank * 64;
                        #pragma unroll 1
                        for (int iter_k_2 = 0; iter_k_2 < k_count_2; iter_k_2++) {
                            mbarrier_wait(b_empty_addr + (b_stage_1) * 8, _phase_b_empty);
                            mbarrier_wait(b_read_addr + (b_stage_1) * 8, _phase_b_empty);
                            int k0_1 = (k_first_2 + iter_k_2) * 32;
                            #pragma unroll
                            for (int p_1 = 0; p_1 < 2; p_1++) {
                                tma_3d_gmem2smem(b1_addr + b_stage_1 * 12288 + (unsigned int)(p_1 * 2048), (&B), b_col + 32 * p_1, k0_1, 0, b_full_addr + (b_stage_1) * 8);
                                tma_3d_gmem2smem(b2_addr + b_stage_1 * 12288 + (unsigned int)(p_1 * 2048), (&B), b_col + 32 * p_1, k0_1, 1, b_full_addr + (b_stage_1) * 8);
                                tma_3d_gmem2smem(b3_addr + b_stage_1 * 12288 + (unsigned int)(p_1 * 2048), (&B), b_col + 32 * p_1, k0_1, 2, b_full_addr + (b_stage_1) * 8);
                            }
                            mbarrier_arrive_expect_tx(b_full_addr + (b_stage_1) * 8, 12288);
                            b_stage_1 += 1;
                            if (b_stage_1 == 6) { b_stage_1 = 0; _phase_b_empty ^= 1; }
                        }
                    }
                    mbarrier_wait(work_full_addr + (work_stage_2) * 8, _phase_work_full_2);
                    uint32_t clc_valid_2 = 0;
                    asm volatile(
                        "{\n\t"
                        ".reg .pred p1;\n\t"
                        ".reg .b128 clc_r;\n\t"
                        "ld.shared.b128 clc_r, [%1];\n\t"
                        "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                        "selp.u32 %0, 1, 0, p1;\n\t"
                        "}\n"
                        : "=r"(clc_valid_2)
                        : "r"(work_response_addr + work_stage_2 * 16 + 0 * 16)
                        : "memory");
                    uint32_t clc_ctaid_2 = 0;
                    asm volatile(
                        "{\n\t"
                        ".reg .pred p1;\n\t"
                        ".reg .b128 clc_r;\n\t"
                        "ld.shared.b128 clc_r, [%1];\n\t"
                        "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                        "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                        "}\n"
                        : "+r"(clc_ctaid_2)
                        : "r"(work_response_addr + work_stage_2 * 16 + 0 * 16)
                        : "memory");
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    asm volatile(
                        "{\n\t"
                        ".reg .b32 remAddr32;\n\t"
                        "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                        "}"
                        :: "r"(work_empty_addr + (work_stage_2) * 8), "r"(0) : "memory");
                    work_stage_2 += 1;
                    if (work_stage_2 == 4) { work_stage_2 = 0; _phase_work_full_2 ^= 1; }
                    if (clc_valid_2 == 0) {
                        break;
                    }
                    this_bid_2 = clc_ctaid_2 + (unsigned int)cta_rank;
                }
                mbarrier_wait(b_empty_addr + (b_stage_1) * 8, _phase_b_empty);
                b_stage_1 += 1;
                if (b_stage_1 == 6) { b_stage_1 = 0; _phase_b_empty ^= 1; }
                mbarrier_wait(b_empty_addr + (b_stage_1) * 8, _phase_b_empty);
                b_stage_1 += 1;
                if (b_stage_1 == 6) { b_stage_1 = 0; _phase_b_empty ^= 1; }
                mbarrier_wait(b_empty_addr + (b_stage_1) * 8, _phase_b_empty);
                b_stage_1 += 1;
                if (b_stage_1 == 6) { b_stage_1 = 0; _phase_b_empty ^= 1; }
                mbarrier_wait(b_empty_addr + (b_stage_1) * 8, _phase_b_empty);
                b_stage_1 += 1;
                if (b_stage_1 == 6) { b_stage_1 = 0; _phase_b_empty ^= 1; }
                mbarrier_wait(b_empty_addr + (b_stage_1) * 8, _phase_b_empty);
                b_stage_1 += 1;
                if (b_stage_1 == 6) { b_stage_1 = 0; _phase_b_empty ^= 1; }
                mbarrier_wait(b_empty_addr + (b_stage_1) * 8, _phase_b_empty);
                b_stage_1 += 1;
                if (b_stage_1 == 6) { b_stage_1 = 0; _phase_b_empty ^= 1; }
            }
        }
    }
    // ---- Role: idle ----
    if (warp == 3) {
        { // idle_main
            __syncwarp();
        }
    }
    // ---- Role: epilogue ----
    if (warp >= 4 && warp <= 11) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 176;");
        { // epilogue_main
            asm volatile("griddepcontrol.wait;" ::: "memory");
            unsigned int chunk_stage_1 = 0;
            unsigned int work_stage_3 = 0;
            const int epi_warp = warp % 4;
            const int col_part = (warp - 4) / 4;
            const int local_row = epi_warp * 32 + lane;
            int lane_base = taddr + (unsigned int)(epi_warp * 32 << 16) + (unsigned int)(col_part * 64);
            unsigned int this_bid_3 = bid;
            unsigned int _phase_chunk_full = 0;
            unsigned int _phase_mainloop_done = 0;
            unsigned int _phase_work_full_3 = 0;
            #pragma unroll 1
            for (unsigned int _unit_iter_3 = 0; _unit_iter_3 < units; _unit_iter_3++) {
                int unit_4 = this_bid_3 / 2;
                int span_3 = unit_4 / n_tiles;
                int steps_3 = m_tiles / 2 * k_iters;
                int spans_3 = units / n_tiles;
                int _max_0_3 = ((head_spans) > (1) ? (head_spans) : (1));
                int in_head_4 = span_3 * head_steps / _max_0_3;
                int in_tail_4 = head_steps + (span_3 - head_spans) * (steps_3 - head_steps) / (spans_3 - head_spans);
                int first_4 = ((span_3 < head_spans) ? in_head_4 : in_tail_4);
                int first_step_3 = first_4;
                int _max_1_3 = ((head_spans) > (1) ? (head_spans) : (1));
                int in_head_4_1 = (span_3 + 1) * head_steps / _max_1_3;
                int in_tail_4_1 = head_steps + (span_3 + 1 - head_spans) * (steps_3 - head_steps) / (spans_3 - head_spans);
                int first_4_1 = ((span_3 + 1 < head_spans) ? in_head_4_1 : in_tail_4_1);
                int last_step_3 = first_4_1;
                int first_pair_3 = first_step_3 / k_iters;
                int num_segments_3 = (last_step_3 - 1) / k_iters - first_pair_3 + 1;
                #pragma unroll 1
                for (int seg_3 = 0; seg_3 < num_segments_3; seg_3++) {
                    int rank_3 = this_bid_3 % 2;
                    int unit_4_1 = this_bid_3 / 2;
                    int column_tile_3 = unit_4_1 % n_tiles;
                    int pair_step_3 = (first_pair_3 + seg_3) * k_iters;
                    int _max_2_3 = ((first_step_3) > (pair_step_3) ? (first_step_3) : (pair_step_3));
                    int k_first_3 = _max_2_3 - pair_step_3;
                    int _min_3_3 = ((last_step_3) < (pair_step_3 + k_iters) ? (last_step_3) : (pair_step_3 + k_iters));
                    int k_count_3 = _min_3_3 - pair_step_3 - k_first_3;
                    int off_m_3 = ((first_pair_3 + seg_3) * 2 + rank_3) * 128;
                    int off_n_3 = column_tile_3 * 128;
                    int num_chunks_1 = (k_count_3 + 2 - 1) / 2;
                    float run[64];
                    #pragma unroll
                    for (int j = 0; j < 64; j++) {
                        run[j] = 0.0f;
                    }
                    #pragma unroll 1
                    for (int _c = 0; _c < num_chunks_1; _c++) {
                        mbarrier_wait(chunk_full_addr + (chunk_stage_1) * 8, _phase_chunk_full);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int big_addr = (unsigned int)lane_base + chunk_stage_1 * 128;
                        #pragma unroll
                        for (int piece = 0; piece < 2; piece++) {
                            float _tmem_load_0[32];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                                : "=f"(_tmem_load_0[0]), "=f"(_tmem_load_0[1]), "=f"(_tmem_load_0[2]), "=f"(_tmem_load_0[3]), "=f"(_tmem_load_0[4]), "=f"(_tmem_load_0[5]), "=f"(_tmem_load_0[6]), "=f"(_tmem_load_0[7]), "=f"(_tmem_load_0[8]), "=f"(_tmem_load_0[9]), "=f"(_tmem_load_0[10]), "=f"(_tmem_load_0[11]), "=f"(_tmem_load_0[12]), "=f"(_tmem_load_0[13]), "=f"(_tmem_load_0[14]), "=f"(_tmem_load_0[15]), "=f"(_tmem_load_0[16]), "=f"(_tmem_load_0[17]), "=f"(_tmem_load_0[18]), "=f"(_tmem_load_0[19]), "=f"(_tmem_load_0[20]), "=f"(_tmem_load_0[21]), "=f"(_tmem_load_0[22]), "=f"(_tmem_load_0[23]), "=f"(_tmem_load_0[24]), "=f"(_tmem_load_0[25]), "=f"(_tmem_load_0[26]), "=f"(_tmem_load_0[27]), "=f"(_tmem_load_0[28]), "=f"(_tmem_load_0[29]), "=f"(_tmem_load_0[30]), "=f"(_tmem_load_0[31])
                                : "r"(big_addr + piece * 32));
                            asm volatile("tcgen05.wait::ld.sync.aligned;");
                            #pragma unroll
                            for (int j_1 = 0; j_1 < 32; j_1++) {
                                run[piece * 32 + j_1] = run[piece * 32 + j_1] + _tmem_load_0[j_1];
                            }
                        }
                        if (elect_sync()) {
                            asm volatile("tcgen05.fence::before_thread_sync;");
                            asm volatile(
                                "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                                :: "r"((chunk_empty_addr + (chunk_stage_1) * 8) & 0xFEFFFFFF) : "memory");
                        }
                        chunk_stage_1 += 1;
                        if (chunk_stage_1 == 2) { chunk_stage_1 = 0; _phase_chunk_full ^= 1; }
                    }
                    mbarrier_wait(mainloop_done_addr, _phase_mainloop_done);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int small_addr = (unsigned int)(lane_base + 256) + 0 * 128;
                    #pragma unroll
                    for (int piece_1 = 0; piece_1 < 2; piece_1++) {
                        float _tmem_load_1[32];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                            : "=f"(_tmem_load_1[0]), "=f"(_tmem_load_1[1]), "=f"(_tmem_load_1[2]), "=f"(_tmem_load_1[3]), "=f"(_tmem_load_1[4]), "=f"(_tmem_load_1[5]), "=f"(_tmem_load_1[6]), "=f"(_tmem_load_1[7]), "=f"(_tmem_load_1[8]), "=f"(_tmem_load_1[9]), "=f"(_tmem_load_1[10]), "=f"(_tmem_load_1[11]), "=f"(_tmem_load_1[12]), "=f"(_tmem_load_1[13]), "=f"(_tmem_load_1[14]), "=f"(_tmem_load_1[15]), "=f"(_tmem_load_1[16]), "=f"(_tmem_load_1[17]), "=f"(_tmem_load_1[18]), "=f"(_tmem_load_1[19]), "=f"(_tmem_load_1[20]), "=f"(_tmem_load_1[21]), "=f"(_tmem_load_1[22]), "=f"(_tmem_load_1[23]), "=f"(_tmem_load_1[24]), "=f"(_tmem_load_1[25]), "=f"(_tmem_load_1[26]), "=f"(_tmem_load_1[27]), "=f"(_tmem_load_1[28]), "=f"(_tmem_load_1[29]), "=f"(_tmem_load_1[30]), "=f"(_tmem_load_1[31])
                            : "r"(small_addr + piece_1 * 32));
                        asm volatile("tcgen05.wait::ld.sync.aligned;");
                        #pragma unroll
                        for (int j_2 = 0; j_2 < 32; j_2++) {
                            run[piece_1 * 32 + j_2] = run[piece_1 * 32 + j_2] + _tmem_load_1[j_2];
                        }
                    }
                    if (elect_sync()) {
                        asm volatile("tcgen05.fence::before_thread_sync;");
                        asm volatile(
                            "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                            :: "r"((epilogue_done_addr) & 0xFEFFFFFF) : "memory");
                    }
                    _phase_mainloop_done ^= 1;
                    int global_row = off_m_3 + local_row;
                    int col_base = off_n_3 + col_part * 64;
                    if (k_count_3 == k_iters) {
                        if (global_row < M) {
                            #pragma unroll
                            for (int j_3 = 0; j_3 < 64; j_3++) {
                                int n_idx = col_base + j_3;
                                if (n_idx < N) {
                                    out32[(unsigned long long)global_row + (unsigned long long)n_idx * (unsigned long long)ldo] = run[j_3];
                                }
                            }
                        }
                    } else {
                        if (warp == 4) {
                            if (elect_sync()) {
                                asm volatile("cp.async.bulk.wait_group.read 0;");
                            }
                        }
                        asm volatile("barrier.sync 1, 256;" ::: "memory");
                        if (epi_warp == 0) {
                            #pragma unroll
                            for (int j_0 = 0; j_0 < 64; j_0++) {
                                {
                                    uint32_t _addr_4 = static_cast<uint32_t>(box_addr + (unsigned int)((col_part * 64 + j_0) * 128 + lane * 4));
                                    asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_4), "f"(run[j_0]) : "memory");
                                }
                            }
                        }
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        asm volatile("barrier.sync 1, 256;" ::: "memory");
                        if (warp == 4) {
                            if (elect_sync()) {
                                tma_reduce_add_2d((&D), off_m_3, off_n_3, box_addr);
                                asm volatile("cp.async.bulk.commit_group;");
                            }
                        }
                        if (warp == 4) {
                            if (elect_sync()) {
                                asm volatile("cp.async.bulk.wait_group.read 0;");
                            }
                        }
                        asm volatile("barrier.sync 1, 256;" ::: "memory");
                        if (epi_warp == 1) {
                            #pragma unroll
                            for (int j_1_1 = 0; j_1_1 < 64; j_1_1++) {
                                {
                                    uint32_t _addr_5 = static_cast<uint32_t>(box_addr + (unsigned int)((col_part * 64 + j_1_1) * 128 + lane * 4));
                                    asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_5), "f"(run[j_1_1]) : "memory");
                                }
                            }
                        }
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        asm volatile("barrier.sync 1, 256;" ::: "memory");
                        if (warp == 4) {
                            if (elect_sync()) {
                                tma_reduce_add_2d((&D), off_m_3 + 32, off_n_3, box_addr);
                                asm volatile("cp.async.bulk.commit_group;");
                            }
                        }
                        if (warp == 4) {
                            if (elect_sync()) {
                                asm volatile("cp.async.bulk.wait_group.read 0;");
                            }
                        }
                        asm volatile("barrier.sync 1, 256;" ::: "memory");
                        if (epi_warp == 2) {
                            #pragma unroll
                            for (int j_2_1 = 0; j_2_1 < 64; j_2_1++) {
                                {
                                    uint32_t _addr_6 = static_cast<uint32_t>(box_addr + (unsigned int)((col_part * 64 + j_2_1) * 128 + lane * 4));
                                    asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_6), "f"(run[j_2_1]) : "memory");
                                }
                            }
                        }
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        asm volatile("barrier.sync 1, 256;" ::: "memory");
                        if (warp == 4) {
                            if (elect_sync()) {
                                tma_reduce_add_2d((&D), off_m_3 + 64, off_n_3, box_addr);
                                asm volatile("cp.async.bulk.commit_group;");
                            }
                        }
                        if (warp == 4) {
                            if (elect_sync()) {
                                asm volatile("cp.async.bulk.wait_group.read 0;");
                            }
                        }
                        asm volatile("barrier.sync 1, 256;" ::: "memory");
                        if (epi_warp == 3) {
                            #pragma unroll
                            for (int j_3_1 = 0; j_3_1 < 64; j_3_1++) {
                                {
                                    uint32_t _addr_7 = static_cast<uint32_t>(box_addr + (unsigned int)((col_part * 64 + j_3_1) * 128 + lane * 4));
                                    asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_7), "f"(run[j_3_1]) : "memory");
                                }
                            }
                        }
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        asm volatile("barrier.sync 1, 256;" ::: "memory");
                        if (warp == 4) {
                            if (elect_sync()) {
                                tma_reduce_add_2d((&D), off_m_3 + 96, off_n_3, box_addr);
                                asm volatile("cp.async.bulk.commit_group;");
                            }
                        }
                    }
                }
                mbarrier_wait(work_full_addr + (work_stage_3) * 8, _phase_work_full_3);
                uint32_t clc_valid_3 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "selp.u32 %0, 1, 0, p1;\n\t"
                    "}\n"
                    : "=r"(clc_valid_3)
                    : "r"(work_response_addr + work_stage_3 * 16 + 0 * 16)
                    : "memory");
                uint32_t clc_ctaid_3 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "+r"(clc_ctaid_3)
                    : "r"(work_response_addr + work_stage_3 * 16 + 0 * 16)
                    : "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("tcgen05.fence::before_thread_sync;");
                asm volatile(
                    "{\n\t"
                    ".reg .b32 remAddr32;\n\t"
                    "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                    "}"
                    :: "r"(work_empty_addr + (work_stage_3) * 8), "r"(0) : "memory");
                work_stage_3 += 1;
                if (work_stage_3 == 4) { work_stage_3 = 0; _phase_work_full_3 ^= 1; }
                if (clc_valid_3 == 0) {
                    break;
                }
                this_bid_3 = clc_ctaid_3 + (unsigned int)cta_rank;
            }
            if (warp == 4) {
                if (elect_sync()) {
                    asm volatile("cp.async.bulk.wait_group.read 0;");
                }
            }
        }
    }
    // ---- Role: producer ----
    if (warp >= 12 && warp <= 15) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 104;");
        { // producer_main
            unsigned int r_stage = 0;
            unsigned int b_stage_2 = 0;
            unsigned int a_stage_1 = 0;
            unsigned int work_stage_4 = 0;
            const int quarter = warp - 12;
            const int col_off = quarter * 4096 + (lane & 3) * 4;
            const int lane_chunk = lane >> 2;
            const unsigned int row_base = (unsigned int)(quarter * 32 << 16);
            unsigned int x[16];
            unsigned int y[16];
            unsigned int ph[8];
            unsigned int pm[8];
            unsigned int pl[8];
            unsigned int this_bid_4 = bid;
            unsigned int _phase_raw_full = 0;
            unsigned int _phase_a_empty = 1;
            unsigned int _phase_b_full = 0;
            unsigned int _phase_work_full_4 = 0;
            #pragma unroll 1
            for (unsigned int _unit_iter_4 = 0; _unit_iter_4 < units; _unit_iter_4++) {
                int unit_5 = this_bid_4 / 2;
                int span_4 = unit_5 / n_tiles;
                int steps_4 = m_tiles / 2 * k_iters;
                int spans_4 = units / n_tiles;
                int _max_0_4 = ((head_spans) > (1) ? (head_spans) : (1));
                int in_head_5 = span_4 * head_steps / _max_0_4;
                int in_tail_5 = head_steps + (span_4 - head_spans) * (steps_4 - head_steps) / (spans_4 - head_spans);
                int first_5 = ((span_4 < head_spans) ? in_head_5 : in_tail_5);
                int first_step_4 = first_5;
                int _max_1_4 = ((head_spans) > (1) ? (head_spans) : (1));
                int in_head_2_1 = (span_4 + 1) * head_steps / _max_1_4;
                int in_tail_2_1 = head_steps + (span_4 + 1 - head_spans) * (steps_4 - head_steps) / (spans_4 - head_spans);
                int first_2_1 = ((span_4 + 1 < head_spans) ? in_head_2_1 : in_tail_2_1);
                int last_step_4 = first_2_1;
                int first_pair_4 = first_step_4 / k_iters;
                int num_segments_4 = (last_step_4 - 1) / k_iters - first_pair_4 + 1;
                #pragma unroll 1
                for (int seg_4 = 0; seg_4 < num_segments_4; seg_4++) {
                    int rank_4 = this_bid_4 % 2;
                    int unit_2_1 = this_bid_4 / 2;
                    int column_tile_4 = unit_2_1 % n_tiles;
                    int pair_step_4 = (first_pair_4 + seg_4) * k_iters;
                    int _max_2_4 = ((first_step_4) > (pair_step_4) ? (first_step_4) : (pair_step_4));
                    int k_first_4 = _max_2_4 - pair_step_4;
                    int _min_3_4 = ((last_step_4) < (pair_step_4 + k_iters) ? (last_step_4) : (pair_step_4 + k_iters));
                    int k_count_4 = _min_3_4 - pair_step_4 - k_first_4;
                    int off_m_4 = ((first_pair_4 + seg_4) * 2 + rank_4) * 128;
                    int off_n_4 = column_tile_4 * 128;
                    #pragma unroll 1
                    for (int iter_k_3 = 0; iter_k_3 < k_count_4; iter_k_3++) {
                        mbarrier_wait(raw_full_addr + (r_stage) * 8, _phase_raw_full);
                        unsigned int src_col = a32_addr + r_stage * 16384 + (unsigned int)col_off;
                        if (iter_k_3 == 0) {
                            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&x[0])) : "r"(src_col + (unsigned int)((lane_chunk ^ 0) << 4)));
                            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&x[1])) : "r"(src_col + (unsigned int)(128 + ((lane_chunk ^ 1) << 4))));
                            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&x[2])) : "r"(src_col + (unsigned int)(256 + ((lane_chunk ^ 2) << 4))));
                            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&x[3])) : "r"(src_col + (unsigned int)(384 + ((lane_chunk ^ 3) << 4))));
                            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&x[4])) : "r"(src_col + (unsigned int)(512 + ((lane_chunk ^ 4) << 4))));
                            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&x[5])) : "r"(src_col + (unsigned int)(640 + ((lane_chunk ^ 5) << 4))));
                            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&x[6])) : "r"(src_col + (unsigned int)(768 + ((lane_chunk ^ 6) << 4))));
                            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&x[7])) : "r"(src_col + (unsigned int)(896 + ((lane_chunk ^ 7) << 4))));
                            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&x[8])) : "r"(src_col + (unsigned int)(1024 + ((lane_chunk ^ 0) << 4))));
                            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&x[9])) : "r"(src_col + (unsigned int)(1152 + ((lane_chunk ^ 1) << 4))));
                            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&x[10])) : "r"(src_col + (unsigned int)(1280 + ((lane_chunk ^ 2) << 4))));
                            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&x[11])) : "r"(src_col + (unsigned int)(1408 + ((lane_chunk ^ 3) << 4))));
                            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&x[12])) : "r"(src_col + (unsigned int)(1536 + ((lane_chunk ^ 4) << 4))));
                            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&x[13])) : "r"(src_col + (unsigned int)(1664 + ((lane_chunk ^ 5) << 4))));
                            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&x[14])) : "r"(src_col + (unsigned int)(1792 + ((lane_chunk ^ 6) << 4))));
                            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&x[15])) : "r"(src_col + (unsigned int)(1920 + ((lane_chunk ^ 7) << 4))));
                        }
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&y[0])) : "r"(src_col + (unsigned int)(2048 + ((lane_chunk ^ 0) << 4))));
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&y[1])) : "r"(src_col + (unsigned int)(2176 + ((lane_chunk ^ 1) << 4))));
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&y[2])) : "r"(src_col + (unsigned int)(2304 + ((lane_chunk ^ 2) << 4))));
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&y[3])) : "r"(src_col + (unsigned int)(2432 + ((lane_chunk ^ 3) << 4))));
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&y[4])) : "r"(src_col + (unsigned int)(2560 + ((lane_chunk ^ 4) << 4))));
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&y[5])) : "r"(src_col + (unsigned int)(2688 + ((lane_chunk ^ 5) << 4))));
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&y[6])) : "r"(src_col + (unsigned int)(2816 + ((lane_chunk ^ 6) << 4))));
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&y[7])) : "r"(src_col + (unsigned int)(2944 + ((lane_chunk ^ 7) << 4))));
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&y[8])) : "r"(src_col + (unsigned int)(3072 + ((lane_chunk ^ 0) << 4))));
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&y[9])) : "r"(src_col + (unsigned int)(3200 + ((lane_chunk ^ 1) << 4))));
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&y[10])) : "r"(src_col + (unsigned int)(3328 + ((lane_chunk ^ 2) << 4))));
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&y[11])) : "r"(src_col + (unsigned int)(3456 + ((lane_chunk ^ 3) << 4))));
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&y[12])) : "r"(src_col + (unsigned int)(3584 + ((lane_chunk ^ 4) << 4))));
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&y[13])) : "r"(src_col + (unsigned int)(3712 + ((lane_chunk ^ 5) << 4))));
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&y[14])) : "r"(src_col + (unsigned int)(3840 + ((lane_chunk ^ 6) << 4))));
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&y[15])) : "r"(src_col + (unsigned int)(3968 + ((lane_chunk ^ 7) << 4))));
                        float v0 = __uint_as_float(x[0]);
                        float v1 = __uint_as_float(x[1]);
                        unsigned int pk = __as_u32(__float22bfloat162_rn(make_float2(v0, v1)));
                        float r0 = v0 - __uint_as_float(pk << 16);
                        float r1 = v1 - __uint_as_float(pk & 4294901760u);
                        unsigned int m0 = __as_u32(r0) & 4294901760u;
                        unsigned int m1 = __as_u32(r1) & 4294901760u;
                        float s0 = r0 - __uint_as_float(m0);
                        float s1 = r1 - __uint_as_float(m1);
                        ph[0] = pk;
                        uint32_t _prmt_4;
                        asm("prmt.b32 %0, %1, %2, 0x7632;" : "=r"(_prmt_4) : "r"(m0), "r"(m1));
                        pm[0] = _prmt_4;
                        uint32_t _prmt_5;
                        asm("prmt.b32 %0, %1, %2, 0x7632;" : "=r"(_prmt_5) : "r"(__as_u32(s0)), "r"(__as_u32(s1)));
                        pl[0] = _prmt_5;
                        float v0_0 = __uint_as_float(x[2]);
                        float v1_0 = __uint_as_float(x[3]);
                        unsigned int pk_0 = __as_u32(__float22bfloat162_rn(make_float2(v0_0, v1_0)));
                        float r0_0 = v0_0 - __uint_as_float(pk_0 << 16);
                        float r1_0 = v1_0 - __uint_as_float(pk_0 & 4294901760u);
                        unsigned int m0_0 = __as_u32(r0_0) & 4294901760u;
                        unsigned int m1_0 = __as_u32(r1_0) & 4294901760u;
                        float s0_0 = r0_0 - __uint_as_float(m0_0);
                        float s1_0 = r1_0 - __uint_as_float(m1_0);
                        ph[1] = pk_0;
                        uint32_t _prmt_6;
                        asm("prmt.b32 %0, %1, %2, 0x7632;" : "=r"(_prmt_6) : "r"(m0_0), "r"(m1_0));
                        pm[1] = _prmt_6;
                        uint32_t _prmt_7;
                        asm("prmt.b32 %0, %1, %2, 0x7632;" : "=r"(_prmt_7) : "r"(__as_u32(s0_0)), "r"(__as_u32(s1_0)));
                        pl[1] = _prmt_7;
                        float v0_1 = __uint_as_float(x[4]);
                        float v1_1 = __uint_as_float(x[5]);
                        unsigned int pk_1 = __as_u32(__float22bfloat162_rn(make_float2(v0_1, v1_1)));
                        float r0_1 = v0_1 - __uint_as_float(pk_1 << 16);
                        float r1_1 = v1_1 - __uint_as_float(pk_1 & 4294901760u);
                        unsigned int m0_1 = __as_u32(r0_1) & 4294901760u;
                        unsigned int m1_1 = __as_u32(r1_1) & 4294901760u;
                        float s0_1 = r0_1 - __uint_as_float(m0_1);
                        float s1_1 = r1_1 - __uint_as_float(m1_1);
                        ph[2] = pk_1;
                        uint32_t _prmt_8;
                        asm("prmt.b32 %0, %1, %2, 0x7632;" : "=r"(_prmt_8) : "r"(m0_1), "r"(m1_1));
                        pm[2] = _prmt_8;
                        uint32_t _prmt_9;
                        asm("prmt.b32 %0, %1, %2, 0x7632;" : "=r"(_prmt_9) : "r"(__as_u32(s0_1)), "r"(__as_u32(s1_1)));
                        pl[2] = _prmt_9;
                        float v0_2 = __uint_as_float(x[6]);
                        float v1_2 = __uint_as_float(x[7]);
                        unsigned int pk_2 = __as_u32(__float22bfloat162_rn(make_float2(v0_2, v1_2)));
                        float r0_2 = v0_2 - __uint_as_float(pk_2 << 16);
                        float r1_2 = v1_2 - __uint_as_float(pk_2 & 4294901760u);
                        unsigned int m0_2 = __as_u32(r0_2) & 4294901760u;
                        unsigned int m1_2 = __as_u32(r1_2) & 4294901760u;
                        float s0_2 = r0_2 - __uint_as_float(m0_2);
                        float s1_2 = r1_2 - __uint_as_float(m1_2);
                        ph[3] = pk_2;
                        uint32_t _prmt_10;
                        asm("prmt.b32 %0, %1, %2, 0x7632;" : "=r"(_prmt_10) : "r"(m0_2), "r"(m1_2));
                        pm[3] = _prmt_10;
                        uint32_t _prmt_11;
                        asm("prmt.b32 %0, %1, %2, 0x7632;" : "=r"(_prmt_11) : "r"(__as_u32(s0_2)), "r"(__as_u32(s1_2)));
                        pl[3] = _prmt_11;
                        float v0_3 = __uint_as_float(x[8]);
                        float v1_3 = __uint_as_float(x[9]);
                        unsigned int pk_3 = __as_u32(__float22bfloat162_rn(make_float2(v0_3, v1_3)));
                        float r0_3 = v0_3 - __uint_as_float(pk_3 << 16);
                        float r1_3 = v1_3 - __uint_as_float(pk_3 & 4294901760u);
                        unsigned int m0_3 = __as_u32(r0_3) & 4294901760u;
                        unsigned int m1_3 = __as_u32(r1_3) & 4294901760u;
                        float s0_3 = r0_3 - __uint_as_float(m0_3);
                        float s1_3 = r1_3 - __uint_as_float(m1_3);
                        ph[4] = pk_3;
                        uint32_t _prmt_12;
                        asm("prmt.b32 %0, %1, %2, 0x7632;" : "=r"(_prmt_12) : "r"(m0_3), "r"(m1_3));
                        pm[4] = _prmt_12;
                        uint32_t _prmt_13;
                        asm("prmt.b32 %0, %1, %2, 0x7632;" : "=r"(_prmt_13) : "r"(__as_u32(s0_3)), "r"(__as_u32(s1_3)));
                        pl[4] = _prmt_13;
                        float v0_4 = __uint_as_float(x[10]);
                        float v1_4 = __uint_as_float(x[11]);
                        unsigned int pk_4 = __as_u32(__float22bfloat162_rn(make_float2(v0_4, v1_4)));
                        float r0_4 = v0_4 - __uint_as_float(pk_4 << 16);
                        float r1_4 = v1_4 - __uint_as_float(pk_4 & 4294901760u);
                        unsigned int m0_4 = __as_u32(r0_4) & 4294901760u;
                        unsigned int m1_4 = __as_u32(r1_4) & 4294901760u;
                        float s0_4 = r0_4 - __uint_as_float(m0_4);
                        float s1_4 = r1_4 - __uint_as_float(m1_4);
                        ph[5] = pk_4;
                        uint32_t _prmt_14;
                        asm("prmt.b32 %0, %1, %2, 0x7632;" : "=r"(_prmt_14) : "r"(m0_4), "r"(m1_4));
                        pm[5] = _prmt_14;
                        uint32_t _prmt_15;
                        asm("prmt.b32 %0, %1, %2, 0x7632;" : "=r"(_prmt_15) : "r"(__as_u32(s0_4)), "r"(__as_u32(s1_4)));
                        pl[5] = _prmt_15;
                        float v0_5 = __uint_as_float(x[12]);
                        float v1_5 = __uint_as_float(x[13]);
                        unsigned int pk_5 = __as_u32(__float22bfloat162_rn(make_float2(v0_5, v1_5)));
                        float r0_5 = v0_5 - __uint_as_float(pk_5 << 16);
                        float r1_5 = v1_5 - __uint_as_float(pk_5 & 4294901760u);
                        unsigned int m0_5 = __as_u32(r0_5) & 4294901760u;
                        unsigned int m1_5 = __as_u32(r1_5) & 4294901760u;
                        float s0_5 = r0_5 - __uint_as_float(m0_5);
                        float s1_5 = r1_5 - __uint_as_float(m1_5);
                        ph[6] = pk_5;
                        uint32_t _prmt_16;
                        asm("prmt.b32 %0, %1, %2, 0x7632;" : "=r"(_prmt_16) : "r"(m0_5), "r"(m1_5));
                        pm[6] = _prmt_16;
                        uint32_t _prmt_17;
                        asm("prmt.b32 %0, %1, %2, 0x7632;" : "=r"(_prmt_17) : "r"(__as_u32(s0_5)), "r"(__as_u32(s1_5)));
                        pl[6] = _prmt_17;
                        float v0_6 = __uint_as_float(x[14]);
                        float v1_6 = __uint_as_float(x[15]);
                        unsigned int pk_6 = __as_u32(__float22bfloat162_rn(make_float2(v0_6, v1_6)));
                        float r0_6 = v0_6 - __uint_as_float(pk_6 << 16);
                        float r1_6 = v1_6 - __uint_as_float(pk_6 & 4294901760u);
                        unsigned int m0_6 = __as_u32(r0_6) & 4294901760u;
                        unsigned int m1_6 = __as_u32(r1_6) & 4294901760u;
                        float s0_6 = r0_6 - __uint_as_float(m0_6);
                        float s1_6 = r1_6 - __uint_as_float(m1_6);
                        ph[7] = pk_6;
                        uint32_t _prmt_18;
                        asm("prmt.b32 %0, %1, %2, 0x7632;" : "=r"(_prmt_18) : "r"(m0_6), "r"(m1_6));
                        pm[7] = _prmt_18;
                        uint32_t _prmt_19;
                        asm("prmt.b32 %0, %1, %2, 0x7632;" : "=r"(_prmt_19) : "r"(__as_u32(s0_6)), "r"(__as_u32(s1_6)));
                        pl[7] = _prmt_19;
                        mbarrier_wait(a_empty_addr + (a_stage_1) * 8, _phase_a_empty);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        unsigned int a_col = taddr + row_base + a_stage_1 * 16;
                        tmem_st_x8_u32(a_col + 384, (const uint32_t*)ph);
                        tmem_st_x8_u32(a_col + 416, (const uint32_t*)pm);
                        tmem_st_x8_u32(a_col + 448, (const uint32_t*)pl);
                        if (iter_k_3 + 1 < k_count_4) {
                            mbarrier_wait(raw_full_addr + (((r_stage + 1 < 6) ? r_stage + 1 : (unsigned int)0)) * 8, ((r_stage + 1 < 6) ? _phase_raw_full : _phase_raw_full ^ 1));
                            unsigned int next_slot = ((r_stage + 1 < 6) ? r_stage + 1 : (unsigned int)0);
                            unsigned int next_col = a32_addr + next_slot * 16384 + (unsigned int)col_off;
                            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&x[0])) : "r"(next_col + (unsigned int)((lane_chunk ^ 0) << 4)));
                            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&x[1])) : "r"(next_col + (unsigned int)(128 + ((lane_chunk ^ 1) << 4))));
                            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&x[2])) : "r"(next_col + (unsigned int)(256 + ((lane_chunk ^ 2) << 4))));
                            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&x[3])) : "r"(next_col + (unsigned int)(384 + ((lane_chunk ^ 3) << 4))));
                            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&x[4])) : "r"(next_col + (unsigned int)(512 + ((lane_chunk ^ 4) << 4))));
                            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&x[5])) : "r"(next_col + (unsigned int)(640 + ((lane_chunk ^ 5) << 4))));
                            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&x[6])) : "r"(next_col + (unsigned int)(768 + ((lane_chunk ^ 6) << 4))));
                            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&x[7])) : "r"(next_col + (unsigned int)(896 + ((lane_chunk ^ 7) << 4))));
                            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&x[8])) : "r"(next_col + (unsigned int)(1024 + ((lane_chunk ^ 0) << 4))));
                            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&x[9])) : "r"(next_col + (unsigned int)(1152 + ((lane_chunk ^ 1) << 4))));
                            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&x[10])) : "r"(next_col + (unsigned int)(1280 + ((lane_chunk ^ 2) << 4))));
                            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&x[11])) : "r"(next_col + (unsigned int)(1408 + ((lane_chunk ^ 3) << 4))));
                            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&x[12])) : "r"(next_col + (unsigned int)(1536 + ((lane_chunk ^ 4) << 4))));
                            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&x[13])) : "r"(next_col + (unsigned int)(1664 + ((lane_chunk ^ 5) << 4))));
                            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&x[14])) : "r"(next_col + (unsigned int)(1792 + ((lane_chunk ^ 6) << 4))));
                            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&x[15])) : "r"(next_col + (unsigned int)(1920 + ((lane_chunk ^ 7) << 4))));
                        }
                        float v0_7 = __uint_as_float(y[0]);
                        float v1_7 = __uint_as_float(y[1]);
                        unsigned int pk_7 = __as_u32(__float22bfloat162_rn(make_float2(v0_7, v1_7)));
                        float r0_7 = v0_7 - __uint_as_float(pk_7 << 16);
                        float r1_7 = v1_7 - __uint_as_float(pk_7 & 4294901760u);
                        unsigned int m0_7 = __as_u32(r0_7) & 4294901760u;
                        unsigned int m1_7 = __as_u32(r1_7) & 4294901760u;
                        float s0_7 = r0_7 - __uint_as_float(m0_7);
                        float s1_7 = r1_7 - __uint_as_float(m1_7);
                        ph[0] = pk_7;
                        uint32_t _prmt_20;
                        asm("prmt.b32 %0, %1, %2, 0x7632;" : "=r"(_prmt_20) : "r"(m0_7), "r"(m1_7));
                        pm[0] = _prmt_20;
                        uint32_t _prmt_21;
                        asm("prmt.b32 %0, %1, %2, 0x7632;" : "=r"(_prmt_21) : "r"(__as_u32(s0_7)), "r"(__as_u32(s1_7)));
                        pl[0] = _prmt_21;
                        float v0_8 = __uint_as_float(y[2]);
                        float v1_8 = __uint_as_float(y[3]);
                        unsigned int pk_8 = __as_u32(__float22bfloat162_rn(make_float2(v0_8, v1_8)));
                        float r0_8 = v0_8 - __uint_as_float(pk_8 << 16);
                        float r1_8 = v1_8 - __uint_as_float(pk_8 & 4294901760u);
                        unsigned int m0_8 = __as_u32(r0_8) & 4294901760u;
                        unsigned int m1_8 = __as_u32(r1_8) & 4294901760u;
                        float s0_8 = r0_8 - __uint_as_float(m0_8);
                        float s1_8 = r1_8 - __uint_as_float(m1_8);
                        ph[1] = pk_8;
                        uint32_t _prmt_22;
                        asm("prmt.b32 %0, %1, %2, 0x7632;" : "=r"(_prmt_22) : "r"(m0_8), "r"(m1_8));
                        pm[1] = _prmt_22;
                        uint32_t _prmt_23;
                        asm("prmt.b32 %0, %1, %2, 0x7632;" : "=r"(_prmt_23) : "r"(__as_u32(s0_8)), "r"(__as_u32(s1_8)));
                        pl[1] = _prmt_23;
                        float v0_9 = __uint_as_float(y[4]);
                        float v1_9 = __uint_as_float(y[5]);
                        unsigned int pk_9 = __as_u32(__float22bfloat162_rn(make_float2(v0_9, v1_9)));
                        float r0_9 = v0_9 - __uint_as_float(pk_9 << 16);
                        float r1_9 = v1_9 - __uint_as_float(pk_9 & 4294901760u);
                        unsigned int m0_9 = __as_u32(r0_9) & 4294901760u;
                        unsigned int m1_9 = __as_u32(r1_9) & 4294901760u;
                        float s0_9 = r0_9 - __uint_as_float(m0_9);
                        float s1_9 = r1_9 - __uint_as_float(m1_9);
                        ph[2] = pk_9;
                        uint32_t _prmt_24;
                        asm("prmt.b32 %0, %1, %2, 0x7632;" : "=r"(_prmt_24) : "r"(m0_9), "r"(m1_9));
                        pm[2] = _prmt_24;
                        uint32_t _prmt_25;
                        asm("prmt.b32 %0, %1, %2, 0x7632;" : "=r"(_prmt_25) : "r"(__as_u32(s0_9)), "r"(__as_u32(s1_9)));
                        pl[2] = _prmt_25;
                        float v0_10 = __uint_as_float(y[6]);
                        float v1_10 = __uint_as_float(y[7]);
                        unsigned int pk_10 = __as_u32(__float22bfloat162_rn(make_float2(v0_10, v1_10)));
                        float r0_10 = v0_10 - __uint_as_float(pk_10 << 16);
                        float r1_10 = v1_10 - __uint_as_float(pk_10 & 4294901760u);
                        unsigned int m0_10 = __as_u32(r0_10) & 4294901760u;
                        unsigned int m1_10 = __as_u32(r1_10) & 4294901760u;
                        float s0_10 = r0_10 - __uint_as_float(m0_10);
                        float s1_10 = r1_10 - __uint_as_float(m1_10);
                        ph[3] = pk_10;
                        uint32_t _prmt_26;
                        asm("prmt.b32 %0, %1, %2, 0x7632;" : "=r"(_prmt_26) : "r"(m0_10), "r"(m1_10));
                        pm[3] = _prmt_26;
                        uint32_t _prmt_27;
                        asm("prmt.b32 %0, %1, %2, 0x7632;" : "=r"(_prmt_27) : "r"(__as_u32(s0_10)), "r"(__as_u32(s1_10)));
                        pl[3] = _prmt_27;
                        float v0_11 = __uint_as_float(y[8]);
                        float v1_11 = __uint_as_float(y[9]);
                        unsigned int pk_11 = __as_u32(__float22bfloat162_rn(make_float2(v0_11, v1_11)));
                        float r0_11 = v0_11 - __uint_as_float(pk_11 << 16);
                        float r1_11 = v1_11 - __uint_as_float(pk_11 & 4294901760u);
                        unsigned int m0_11 = __as_u32(r0_11) & 4294901760u;
                        unsigned int m1_11 = __as_u32(r1_11) & 4294901760u;
                        float s0_11 = r0_11 - __uint_as_float(m0_11);
                        float s1_11 = r1_11 - __uint_as_float(m1_11);
                        ph[4] = pk_11;
                        uint32_t _prmt_28;
                        asm("prmt.b32 %0, %1, %2, 0x7632;" : "=r"(_prmt_28) : "r"(m0_11), "r"(m1_11));
                        pm[4] = _prmt_28;
                        uint32_t _prmt_29;
                        asm("prmt.b32 %0, %1, %2, 0x7632;" : "=r"(_prmt_29) : "r"(__as_u32(s0_11)), "r"(__as_u32(s1_11)));
                        pl[4] = _prmt_29;
                        float v0_12 = __uint_as_float(y[10]);
                        float v1_12 = __uint_as_float(y[11]);
                        unsigned int pk_12 = __as_u32(__float22bfloat162_rn(make_float2(v0_12, v1_12)));
                        float r0_12 = v0_12 - __uint_as_float(pk_12 << 16);
                        float r1_12 = v1_12 - __uint_as_float(pk_12 & 4294901760u);
                        unsigned int m0_12 = __as_u32(r0_12) & 4294901760u;
                        unsigned int m1_12 = __as_u32(r1_12) & 4294901760u;
                        float s0_12 = r0_12 - __uint_as_float(m0_12);
                        float s1_12 = r1_12 - __uint_as_float(m1_12);
                        ph[5] = pk_12;
                        uint32_t _prmt_30;
                        asm("prmt.b32 %0, %1, %2, 0x7632;" : "=r"(_prmt_30) : "r"(m0_12), "r"(m1_12));
                        pm[5] = _prmt_30;
                        uint32_t _prmt_31;
                        asm("prmt.b32 %0, %1, %2, 0x7632;" : "=r"(_prmt_31) : "r"(__as_u32(s0_12)), "r"(__as_u32(s1_12)));
                        pl[5] = _prmt_31;
                        float v0_13 = __uint_as_float(y[12]);
                        float v1_13 = __uint_as_float(y[13]);
                        unsigned int pk_13 = __as_u32(__float22bfloat162_rn(make_float2(v0_13, v1_13)));
                        float r0_13 = v0_13 - __uint_as_float(pk_13 << 16);
                        float r1_13 = v1_13 - __uint_as_float(pk_13 & 4294901760u);
                        unsigned int m0_13 = __as_u32(r0_13) & 4294901760u;
                        unsigned int m1_13 = __as_u32(r1_13) & 4294901760u;
                        float s0_13 = r0_13 - __uint_as_float(m0_13);
                        float s1_13 = r1_13 - __uint_as_float(m1_13);
                        ph[6] = pk_13;
                        uint32_t _prmt_32;
                        asm("prmt.b32 %0, %1, %2, 0x7632;" : "=r"(_prmt_32) : "r"(m0_13), "r"(m1_13));
                        pm[6] = _prmt_32;
                        uint32_t _prmt_33;
                        asm("prmt.b32 %0, %1, %2, 0x7632;" : "=r"(_prmt_33) : "r"(__as_u32(s0_13)), "r"(__as_u32(s1_13)));
                        pl[6] = _prmt_33;
                        float v0_14 = __uint_as_float(y[14]);
                        float v1_14 = __uint_as_float(y[15]);
                        unsigned int pk_14 = __as_u32(__float22bfloat162_rn(make_float2(v0_14, v1_14)));
                        float r0_14 = v0_14 - __uint_as_float(pk_14 << 16);
                        float r1_14 = v1_14 - __uint_as_float(pk_14 & 4294901760u);
                        unsigned int m0_14 = __as_u32(r0_14) & 4294901760u;
                        unsigned int m1_14 = __as_u32(r1_14) & 4294901760u;
                        float s0_14 = r0_14 - __uint_as_float(m0_14);
                        float s1_14 = r1_14 - __uint_as_float(m1_14);
                        ph[7] = pk_14;
                        uint32_t _prmt_34;
                        asm("prmt.b32 %0, %1, %2, 0x7632;" : "=r"(_prmt_34) : "r"(m0_14), "r"(m1_14));
                        pm[7] = _prmt_34;
                        uint32_t _prmt_35;
                        asm("prmt.b32 %0, %1, %2, 0x7632;" : "=r"(_prmt_35) : "r"(__as_u32(s0_14)), "r"(__as_u32(s1_14)));
                        pl[7] = _prmt_35;
                        if (elect_sync()) {
                            mbarrier_arrive(raw_empty_addr + (r_stage) * 8);
                        }
                        tmem_st_x8_u32(a_col + 392, (const uint32_t*)ph);
                        tmem_st_x8_u32(a_col + 424, (const uint32_t*)pm);
                        tmem_st_x8_u32(a_col + 456, (const uint32_t*)pl);
                        mbarrier_wait(b_full_addr + (b_stage_2) * 8, _phase_b_full);
                        if (elect_sync()) {
                            mbarrier_arrive(b_read_addr + (b_stage_2) * 8);
                        }
                        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                        asm volatile("tcgen05.fence::before_thread_sync;");
                        if (elect_sync()) {
                            asm volatile(
                                "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                                :: "r"((a_full_addr + (a_stage_1) * 8) & 0xFEFFFFFF) : "memory");
                        }
                        r_stage += 1;
                        if (r_stage == 6) { r_stage = 0; _phase_raw_full ^= 1; }
                        b_stage_2 += 1;
                        if (b_stage_2 == 6) { b_stage_2 = 0; _phase_b_full ^= 1; }
                        a_stage_1 += 1;
                        if (a_stage_1 == 2) { a_stage_1 = 0; _phase_a_empty ^= 1; }
                    }
                }
                mbarrier_wait(work_full_addr + (work_stage_4) * 8, _phase_work_full_4);
                uint32_t clc_valid_4 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "selp.u32 %0, 1, 0, p1;\n\t"
                    "}\n"
                    : "=r"(clc_valid_4)
                    : "r"(work_response_addr + work_stage_4 * 16 + 0 * 16)
                    : "memory");
                uint32_t clc_ctaid_4 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "+r"(clc_ctaid_4)
                    : "r"(work_response_addr + work_stage_4 * 16 + 0 * 16)
                    : "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile(
                    "{\n\t"
                    ".reg .b32 remAddr32;\n\t"
                    "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                    "}"
                    :: "r"(work_empty_addr + (work_stage_4) * 8), "r"(0) : "memory");
                work_stage_4 += 1;
                if (work_stage_4 == 4) { work_stage_4 = 0; _phase_work_full_4 ^= 1; }
                if (clc_valid_4 == 0) {
                    break;
                }
                this_bid_4 = clc_ctaid_4 + (unsigned int)cta_rank;
            }
            mbarrier_wait(a_empty_addr + (a_stage_1) * 8, _phase_a_empty);
            a_stage_1 += 1;
            if (a_stage_1 == 2) { a_stage_1 = 0; _phase_a_empty ^= 1; }
            mbarrier_wait(a_empty_addr + (a_stage_1) * 8, _phase_a_empty);
            a_stage_1 += 1;
            if (a_stage_1 == 2) { a_stage_1 = 0; _phase_a_empty ^= 1; }
        }
    }

    // Cleanup
    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
    asm volatile("barrier.cluster.arrive.release.aligned;");
    asm volatile("barrier.cluster.wait.acquire.aligned;");

    if (warp == 0) {
        asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(512));
    }
}

} // extern "C"
