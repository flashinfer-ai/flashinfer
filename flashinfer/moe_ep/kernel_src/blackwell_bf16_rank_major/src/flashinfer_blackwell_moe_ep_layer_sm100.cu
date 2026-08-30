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
// Bundle: Blackwell BF16 rank-major MoE expert-parallel layer.
// Target: sm_100a; compile flags: --use_fast_math.
// Generated file; do not edit manually.
#include <stdint.h>
#include <cuda.h>
#include <cuda_bf16.h>

struct __align__(128) FlashInferTensorMap { uint64_t opaque[16]; };
template <int N>
struct __align__(128) FlashInferTensorMapPack { FlashInferTensorMap maps[N]; };

__device__ __forceinline__ int make_warp_uniform(int x) {
    int result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1F, 0xFFFFFFFF;"
                 : "=r"(result) : "r"(x));
    return result;
}

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

__device__ __forceinline__ void fence_async_shared() {
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}

__device__ __forceinline__ void tcgen05_mma_f16_cta2(
    int taddr, uint64_t a_desc, uint64_t b_desc,
    uint32_t i_desc, int enable_input_d) {
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        ".reg .b32 m0, m1, m2, m3, m4, m5, m6, m7;\n\t"
        "setp.ne.b32 p, %4, 0;\n\t"
        "mov.b32 m0, 0; mov.b32 m1, 0; mov.b32 m2, 0; mov.b32 m3, 0;\n\t"
        "mov.b32 m4, 0; mov.b32 m5, 0; mov.b32 m6, 0; mov.b32 m7, 0;\n\t"
        "tcgen05.mma.cta_group::2.kind::f16 [%0], %1, %2, %3, {m0, m1, m2, m3, m4, m5, m6, m7}, p;\n\t"
        "}\n"
        :: "r"(taddr), "l"(a_desc), "l"(b_desc),
           "r"(i_desc), "r"(enable_input_d));
}

__device__ __forceinline__ uint64_t desc_encode(uint64_t x) {
    return (x & 0x3FFFFULL) >> 4ULL;
}

__device__ __forceinline__ void mma_ss_step_cg2(
    int a_lo, int b_lo, int taddr, uint32_t i_desc, int enable_d,
    uint32_t a_dhi, uint32_t b_dhi) {
    asm volatile(
        "{\n\t"
        ".reg .pred leader, p;\n\t"
        ".reg .b32 adhi, bdhi, m0, m1, m2, m3, m4, m5, m6, m7;\n\t"
        ".reg .b64 da, db;\n\t"
        "elect.sync _|leader, 0xFFFFFFFF;\n\t"
        "setp.ne.b32 p, %4, 0;\n\t"
        "mov.b32 m0, 0; mov.b32 m1, 0; mov.b32 m2, 0; mov.b32 m3, 0;\n\t"
        "mov.b32 m4, 0; mov.b32 m5, 0; mov.b32 m6, 0; mov.b32 m7, 0;\n\t"
        "mov.b32 adhi, %5;\n\t"
        "mov.b32 bdhi, %6;\n\t"
        "mov.b64 da, {%0, adhi};\n\t"
        "mov.b64 db, {%1, bdhi};\n\t"
        "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, %3, "
        "{m0, m1, m2, m3, m4, m5, m6, m7}, p;\n\t"
        "}\n"
        :: "r"(a_lo), "r"(b_lo), "r"(taddr), "r"(i_desc), "r"(enable_d), "r"(a_dhi), "r"(b_dhi));
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

__device__ __forceinline__ void tmem_ld_x32(float* dst, int tmem_addr) {
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
        " {%0, %1, %2, %3, %4, %5, %6, %7,"
        "  %8, %9, %10, %11, %12, %13, %14, %15,"
        "  %16, %17, %18, %19, %20, %21, %22, %23,"
        "  %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
        : "=f"(dst[0]),  "=f"(dst[1]),  "=f"(dst[2]),  "=f"(dst[3]),
          "=f"(dst[4]),  "=f"(dst[5]),  "=f"(dst[6]),  "=f"(dst[7]),
          "=f"(dst[8]),  "=f"(dst[9]),  "=f"(dst[10]), "=f"(dst[11]),
          "=f"(dst[12]), "=f"(dst[13]), "=f"(dst[14]), "=f"(dst[15]),
          "=f"(dst[16]), "=f"(dst[17]), "=f"(dst[18]), "=f"(dst[19]),
          "=f"(dst[20]), "=f"(dst[21]), "=f"(dst[22]), "=f"(dst[23]),
          "=f"(dst[24]), "=f"(dst[25]), "=f"(dst[26]), "=f"(dst[27]),
          "=f"(dst[28]), "=f"(dst[29]), "=f"(dst[30]), "=f"(dst[31])
        : "r"(tmem_addr));
}

__device__ __forceinline__ uint32_t smem_addr(const void* ptr) {
    uint32_t addr;
    asm("{\n\t"
        ".reg .u64 u64addr;\n\t"
        "cvta.to.shared.u64 u64addr, %1;\n\t"
        "cvt.u32.u64 %0, u64addr;\n\t"
        "}\n" : "=r"(addr) : "l"(ptr));
    return addr;
}

__device__ __forceinline__ uint32_t mapa_to_rank(uint32_t local_addr, uint32_t rank) {
    uint32_t remote;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;"
        : "=r"(remote) : "r"(local_addr), "r"(rank));
    return remote;
}

__device__ __forceinline__ float approx_exp2(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

__device__ __forceinline__ uint64_t make_smem_desc(int addr) {
    const int SBO = 1024;
    return desc_encode(addr)
         | (desc_encode(SBO) << 32ULL)
         | (1ULL << 46ULL)
         | (2ULL << 61ULL);
}

__device__ __forceinline__ void tma_2d_gmem2smem(
    int dst, const void *tmap_ptr, int x, int y, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global"
        ".mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3}], [%4];"
        :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y),
           "r"(mbar_addr) : "memory");
}

__device__ __forceinline__ void tma_4d_gmem2smem_cta2(
    int dst, const void *tmap_ptr, int x, int y, int z, int w, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.4d.shared::cluster.global"
        ".mbarrier::complete_tx::bytes.cta_group::2"
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

__device__ __forceinline__ void tma_gather4_gmem2smem_cta2(
    int dst, const void *tmap_ptr,
    int col_idx, int row0, int row1, int row2, int row3,
    int mbar_addr) {
    // Canonical .shared::cta form; see tma_gather4_gmem2smem above.
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
        ".mbarrier::complete_tx::bytes.cta_group::2"
        " [%0], [%1, {%2, %3, %4, %5, %6}], [%7];"
        :: "r"(dst), "l"(tmap_ptr), "r"(col_idx),
           "r"(row0), "r"(row1), "r"(row2), "r"(row3),
           "r"(mbar_addr) : "memory");
}

__device__ __forceinline__ void tma_gather4_gmem2smem_mc(
    int dst, const void *tmap_ptr,
    int col_idx, int row0, int row1, int row2, int row3,
    int mbar_addr, unsigned short cta_mask) {
    // Multicast variant: the PTX grammar ties the .shared::cluster
    // destination to .multicast::cluster + ctaMask (cf. cuda_ptx /
    // SM100_TMA_LOAD_MULTICAST_2D_GATHER4 in CUTLASS).
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.tile::gather4"
        ".mbarrier::complete_tx::bytes.multicast::cluster"
        " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
        :: "r"(dst), "l"(tmap_ptr), "r"(col_idx),
           "r"(row0), "r"(row1), "r"(row2), "r"(row3),
           "r"(mbar_addr), "h"(cta_mask) : "memory");
}

__device__ __forceinline__ void tma_gather4_gmem2smem_mc_cta2(
    int dst, const void *tmap_ptr,
    int col_idx, int row0, int row1, int row2, int row3,
    int mbar_addr, unsigned short cta_mask) {
    // Multicast + cta_group::2 variant; see tma_gather4_gmem2smem_mc.
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.tile::gather4"
        ".mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
        " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
        :: "r"(dst), "l"(tmap_ptr), "r"(col_idx),
           "r"(row0), "r"(row1), "r"(row2), "r"(row3),
           "r"(mbar_addr), "h"(cta_mask) : "memory");
}

__device__ __forceinline__ void tma_store_4d(
    const void *tmap, int x, int y, int z, int w, unsigned smem_addr) {
    asm volatile(
        "cp.async.bulk.tensor.4d.global.shared::cta.tile.bulk_group"
        " [%0, {%1, %2, %3, %4}], [%5];"
        :: "l"(tmap), "r"(x), "r"(y), "r"(z), "r"(w), "r"(smem_addr) : "memory");
}

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

#define FLASHINFER_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define THREADS 32

extern "C" {

__global__ __launch_bounds__(32) void
kernel_rank_major_input_barrier_v1(long long* __restrict__ expert_ids, int* __restrict__ topk_ids, int32_t pg_world, int32_t pg_rank, unsigned* const* __restrict__ pg_flags)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // === Task calls (dependency order) ===
    #pragma unroll 1
    for (int route = tid; route < 1024; route += 32) {
        topk_ids[route] = (int)expert_ids[route];
    }
    if (warp == 0) {
        if (elect_sync()) {
            // nvlink_barrier(pg_flags) phase=0
            {
                const int __ws = pg_world;
                const int __me = pg_rank;
                const int __slot = 0;
                unsigned* __local_flag = pg_flags[__me] + __slot;
                unsigned __previous_epoch;
                asm volatile("ld.relaxed.sys.global.u32 %0, [%1];"
                    : "=r"(__previous_epoch) : "l"(__local_flag) : "memory");
                const unsigned __arrival_epoch = __previous_epoch + 1u;
                const unsigned __release_epoch = __previous_epoch + 2u;
                asm volatile("fence.proxy.async.global;" ::: "memory");
                asm volatile("st.release.sys.global.u32 [%0], %1;"
                    :: "l"(__local_flag), "r"(__arrival_epoch) : "memory");
                if (__me == 0) {
                    for (int __r = 0; __r < __ws; ++__r) {
                        unsigned* __peer_flag = pg_flags[__r] + __slot;
                        while (true) {
                            unsigned __v;
                            asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(__v) : "l"(__peer_flag) : "memory");
                            if (__v == __arrival_epoch) break;
                        }
                    }
                    asm volatile("fence.proxy.alias;" ::: "memory");
                    for (int __r = 0; __r < __ws; ++__r) {
                        unsigned* __peer_flag = pg_flags[__r] + __slot;
                        asm volatile("st.release.sys.global.u32 [%0], %1;"
                            :: "l"(__peer_flag), "r"(__release_epoch) : "memory");
                    }
                } else {
                    while (true) {
                        unsigned __v;
                        asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(__v) : "l"(__local_flag) : "memory");
                        if (__v == __release_epoch) break;
                    }
                    asm volatile("fence.proxy.alias;" ::: "memory");
                }
                asm volatile("fence.proxy.async.global;" ::: "memory");
            }
        }
    }
}

} // extern "C"

#undef FLASHINFER_INF
#undef NUM_MAIN_STAGES
#undef THREADS

#define FLASHINFER_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define SMEM_ROW_STAGE_OFF 1024
#define SMEM_ROW_STAGE_STAGE_BYTES 14336
#define SMEM_ROW_STAGE_STRIDE 14336
#define SMEM_IDS_STAGE_OFF 15360
#define SMEM_IDS_STAGE_STAGE_BYTES 32
#define SMEM_IDS_STAGE_STRIDE 32
#define SMEM_WEIGHTS_STAGE_OFF 15392
#define SMEM_WEIGHTS_STAGE_STAGE_BYTES 32
#define SMEM_WEIGHTS_STAGE_STRIDE 32
#define SMEM_TOTAL 15488
#define THREADS 256

extern "C" {

__global__ __launch_bounds__(256) void
kernel_rank_major_dispatch_v1(__nv_bfloat16* __restrict__ recv_hidden, int* __restrict__ recv_local_ids, float* __restrict__ recv_weights, int32_t pg_world, int32_t pg_rank, unsigned* const* __restrict__ pg_flags, __nv_bfloat16* __restrict__ hidden_states, __nv_bfloat16* const* __restrict__ hidden_states_peers, int* __restrict__ topk_ids, int* const* __restrict__ topk_ids_peers, float* __restrict__ topk_weights, float* const* __restrict__ topk_weights_peers)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // Kernel setup ops
    __nv_bfloat16* row_stage = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int row_stage_addr = smem + 1024;
    int* ids_stage = reinterpret_cast<int*>(smem_raw + 15360);
    const int ids_stage_addr = smem + 15360;
    float* weights_stage = reinterpret_cast<float*>(smem_raw + 15392);
    const int weights_stage_addr = smem + 15392;

    // Mbarrier init (1 groups, 1 barriers)
    // Mbarriers at smem_raw[0..8)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // row_full: 1 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncthreads();

    const int mbar_base = smem;
    #define row_full_addr (mbar_base + 0)

    // === Task calls (dependency order) ===
    int recv_token = bid;
    int src_rank = recv_token / 128;
    int src_token = recv_token - src_rank * 128;
    unsigned long long hidden_src_offset = (unsigned long long)src_token * 14336;
    unsigned long long routing_src_offset = (unsigned long long)src_token * 8 * 4;
    if (warp == 0) {
        if (elect_sync()) {
            mbarrier_arrive_expect_tx(row_full_addr, 14400);
            // nvlink_pull: smem(row_stage_addr) <- peers[src_rank] + hidden_src_offset, 14336B
            {
                const void* __remote = (const void*)((const char*)((hidden_states_peers)[src_rank]) + (uint64_t)(hidden_src_offset));
                asm volatile(
                    "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes"
                    " [%0], [%1], %2, [%3];"
                    :: "r"(row_stage_addr), "l"(__remote), "r"((uint32_t)(14336)), "r"(row_full_addr)
                    : "memory");
            }
            // nvlink_pull: smem(ids_stage_addr) <- peers[src_rank] + routing_src_offset, 32B
            {
                const void* __remote = (const void*)((const char*)((topk_ids_peers)[src_rank]) + (uint64_t)(routing_src_offset));
                asm volatile(
                    "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes"
                    " [%0], [%1], %2, [%3];"
                    :: "r"(ids_stage_addr), "l"(__remote), "r"((uint32_t)(32)), "r"(row_full_addr)
                    : "memory");
            }
            // nvlink_pull: smem(weights_stage_addr) <- peers[src_rank] + routing_src_offset, 32B
            {
                const void* __remote = (const void*)((const char*)((topk_weights_peers)[src_rank]) + (uint64_t)(routing_src_offset));
                asm volatile(
                    "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes"
                    " [%0], [%1], %2, [%3];"
                    :: "r"(weights_stage_addr), "l"(__remote), "r"((uint32_t)(32)), "r"(row_full_addr)
                    : "memory");
            }
        }
    }
    unsigned int _phase_row_full_0 = 0;
    mbarrier_wait(row_full_addr, _phase_row_full_0);
    _phase_row_full_0 ^= 1;
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    unsigned long long dst_row = (unsigned long long)recv_token * 7168;
    #pragma unroll 1
    for (int base = tid * 8; base < 7168; base += 2048) {
        unsigned int packed[4];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&packed[0])), "=r"(*reinterpret_cast<uint32_t*>(&packed[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&packed[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&packed[(0) + 3]))
            : "r"(row_stage_addr + (unsigned int)(base * 2)));
        reinterpret_cast<int4*>(recv_hidden + (dst_row + (unsigned long long)base))[0] = reinterpret_cast<int4*>(packed)[0];
    }
    if (tid < 8) {
        int global_expert = ids_stage[tid];
        int owner_begin = pg_rank * 32;
        unsigned long long route_index = (unsigned long long)recv_token * 8 + (unsigned long long)tid;
        if (global_expert >= owner_begin && global_expert < owner_begin + 32) {
            recv_local_ids[route_index] = global_expert - owner_begin;
            recv_weights[route_index] = weights_stage[tid];
        } else {
            recv_local_ids[route_index] = -1;
            recv_weights[route_index] = 0.0f;
        }
    }

    // Cleanup
    __syncthreads();
}

} // extern "C"

#undef FLASHINFER_INF
#undef NUM_MAIN_STAGES
#undef SMEM_IDS_STAGE_OFF
#undef SMEM_IDS_STAGE_STAGE_BYTES
#undef SMEM_IDS_STAGE_STRIDE
#undef SMEM_ROW_STAGE_OFF
#undef SMEM_ROW_STAGE_STAGE_BYTES
#undef SMEM_ROW_STAGE_STRIDE
#undef SMEM_TOTAL
#undef SMEM_WEIGHTS_STAGE_OFF
#undef SMEM_WEIGHTS_STAGE_STAGE_BYTES
#undef SMEM_WEIGHTS_STAGE_STRIDE
#undef THREADS
#undef ids_stage_addr
#undef row_full_addr
#undef row_stage_addr
#undef weights_stage_addr

#define FLASHINFER_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define THREADS 256

extern "C" {

__global__ __launch_bounds__(256) void
kernel_rank_major_route_reset_exact_v1(int* __restrict__ expert_scatter_offsets, __nv_bfloat16* __restrict__ zero_sentinel)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // === Task calls (dependency order) ===
    if (tid < 32) {
        expert_scatter_offsets[tid] = 0;
    }
    #pragma unroll 1
    for (int elem = tid; elem < 7168; elem += 256) {
        zero_sentinel[elem] = 0.0f;
    }
}

} // extern "C"

#undef FLASHINFER_INF
#undef NUM_MAIN_STAGES
#undef THREADS

#define FLASHINFER_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define THREADS 256

extern "C" {

__global__ __launch_bounds__(256) void
kernel_rank_major_route_count_exact_v1(int* __restrict__ recv_local_ids, int* __restrict__ expert_scatter_offsets, int* __restrict__ token_to_permuted)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // === Task calls (dependency order) ===
    int route = bid * 256 + tid;
    int local_expert = recv_local_ids[route];
    token_to_permuted[route] = 32768;
    if (local_expert >= 0) {
        int _atomic_old_0 = atomicAdd(&expert_scatter_offsets[local_expert], 1);
        int local_row = _atomic_old_0;
        token_to_permuted[route] = local_row;
    }
}

} // extern "C"

#undef FLASHINFER_INF
#undef NUM_MAIN_STAGES
#undef THREADS

#define FLASHINFER_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define THREADS 32

extern "C" {

__global__ __launch_bounds__(32) void
kernel_rank_major_route_finalize_exact_v1(int* __restrict__ expert_scatter_offsets, int* __restrict__ cta_to_expert, int* __restrict__ cta_to_mn_limit, int* __restrict__ expert_padded_row_offsets, int* __restrict__ num_non_exiting_ctas, int* __restrict__ total_padded_rows, int* __restrict__ route_map)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // === Task calls (dependency order) ===
    int expert = lane;
    int rows = expert_scatter_offsets[expert];
    int groups = (rows + 64 - 1) / 64;
    if (groups > 16) {
        groups = 16;
    }
    int inclusive = groups;
    int _shfl_up_0 = __shfl_up_sync(0xFFFFFFFF, inclusive, 1, 32);
    int peer = _shfl_up_0;
    if (lane >= 1) {
        inclusive = inclusive + peer;
    }
    int _shfl_up_1 = __shfl_up_sync(0xFFFFFFFF, inclusive, 2, 32);
    int peer_0 = _shfl_up_1;
    if (lane >= 2) {
        inclusive = inclusive + peer_0;
    }
    int _shfl_up_2 = __shfl_up_sync(0xFFFFFFFF, inclusive, 4, 32);
    int peer_1 = _shfl_up_2;
    if (lane >= 4) {
        inclusive = inclusive + peer_1;
    }
    int _shfl_up_3 = __shfl_up_sync(0xFFFFFFFF, inclusive, 8, 32);
    int peer_2 = _shfl_up_3;
    if (lane >= 8) {
        inclusive = inclusive + peer_2;
    }
    int _shfl_up_4 = __shfl_up_sync(0xFFFFFFFF, inclusive, 16, 32);
    int peer_3 = _shfl_up_4;
    if (lane >= 16) {
        inclusive = inclusive + peer_3;
    }
    int write_base = inclusive - groups;
    expert_padded_row_offsets[expert] = write_base * 64;
    #pragma unroll
    for (int local_group = 0; local_group < 16; local_group++) {
        if (groups > local_group) {
            int write_group = write_base + local_group;
            int valid_end = (local_group + 1) * 64;
            if (valid_end > rows) {
                valid_end = rows;
            }
            cta_to_expert[write_group] = expert;
            cta_to_mn_limit[write_group] = write_base * 64 + valid_end;
        }
    }
    int padded_end = groups * 64;
    #pragma unroll 1
    for (int padding_row = rows; padding_row < padded_end; padding_row++) {
        route_map[write_base * 64 + padding_row] = 0;
    }
    if (expert == 31) {
        num_non_exiting_ctas[0] = inclusive;
        total_padded_rows[0] = inclusive * 64;
    }
}

} // extern "C"

#undef FLASHINFER_INF
#undef NUM_MAIN_STAGES
#undef THREADS

#define FLASHINFER_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define THREADS 256

extern "C" {

__global__ __launch_bounds__(256) void
kernel_rank_major_route_scatter_exact_v1(int* __restrict__ recv_local_ids, int* __restrict__ expert_padded_row_offsets, int* __restrict__ route_map, int* __restrict__ token_to_permuted)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // === Task calls (dependency order) ===
    int route = bid * 256 + tid;
    int local_expert = recv_local_ids[route];
    int local_row = token_to_permuted[route];
    if (local_expert >= 0) {
        int compact_row = expert_padded_row_offsets[local_expert] + local_row;
        route_map[compact_row] = route / 8;
        token_to_permuted[route] = compact_row;
    } else {
        token_to_permuted[route] = 32768;
    }
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
}

} // extern "C"

#undef FLASHINFER_INF
#undef NUM_MAIN_STAGES
#undef THREADS

#define FLASHINFER_INF CUDART_INF_F
#define TMEM_NCOLS 128
#define TMEM_ACCUM_OFFSET 0
#define NUM_K_PIPE_STAGES 5
#define NUM_MMA_PIPE_STAGES 2
#define NUM_WORK_PIPE_STAGES 3
#define NUM_THROTTLE_PIPE_STAGES 3
#define NUM_DRAIN_PIPE_STAGES 1
#define SMEM_SMEM_A_OFF 1024
#define SMEM_SMEM_A_STAGE_BYTES 32768
#define SMEM_SMEM_A_STRIDE 32768
#define SMEM_SMEM_B_OFF 164864
#define SMEM_SMEM_B_STAGE_BYTES 8192
#define SMEM_SMEM_B_STRIDE 8192
#define SMEM_EPI_STAGING_OFF 205824
#define SMEM_EPI_STAGING_STAGE_BYTES 8192
#define SMEM_EPI_STAGING_STRIDE 8192
#define SMEM_WORK_RESPONSE_OFF 214016
#define SMEM_WORK_RESPONSE_STAGE_BYTES 16
#define SMEM_WORK_RESPONSE_STRIDE 16
#define SMEM_FAST_DRAIN_RESPONSE_OFF 214064
#define SMEM_FAST_DRAIN_RESPONSE_STAGE_BYTES 16
#define SMEM_FAST_DRAIN_RESPONSE_STRIDE 16
#define SMEM_TOTAL 223232
#define THREADS 384

extern "C" {

__global__ __launch_bounds__(384, 1) __cluster_dims__(2,1,1) void
kernel_rank_major_exact_fc1_swiglu_v1(FlashInferTensorMap const* weights, FlashInferTensorMap const* recv_hidden, FlashInferTensorMap const* compact_intermediate, int* __restrict__ route_map, int* __restrict__ num_non_exiting_ctas, int* __restrict__ cta_idx_y_to_batch_idx, int* __restrict__ cta_idx_y_to_mn_limit, int K)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    const unsigned int clusters_x = gridDim.x / 2;
    const unsigned int cluster_id = ((blockIdx.z * gridDim.y + blockIdx.y) * clusters_x) + blockIdx.x / 2;
    const unsigned int num_clusters = clusters_x * gridDim.y * gridDim.z;

    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));
    if (tid == 0) {
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(weights)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(recv_hidden)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(compact_intermediate)) : "memory");
    }
    __syncthreads();


    // Kernel setup ops
    __nv_bfloat16* smem_a = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int smem_a_addr = smem + 1024;
    __nv_bfloat16* smem_b = reinterpret_cast<__nv_bfloat16*>(smem_raw + 164864);
    const int smem_b_addr = smem + 164864;
    __nv_bfloat16* epi_staging = reinterpret_cast<__nv_bfloat16*>(smem_raw + 205824);
    const int epi_staging_addr = smem + 205824;
    unsigned int* work_response = reinterpret_cast<unsigned int*>(smem_raw + 214016);
    const int work_response_addr = smem + 214016;
    unsigned int* fast_drain_response = reinterpret_cast<unsigned int*>(smem_raw + 214064);
    const int fast_drain_response_addr = smem + 214064;

    // Mbarrier init (11 groups, 37 barriers)
    // Mbarriers at smem_raw[0..296)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'k_pipe' ---
            // a_full: 5 barriers, init_count=2
            mbarrier_init(smem + 0, 2);
            mbarrier_init(smem + 8, 2);
            mbarrier_init(smem + 16, 2);
            mbarrier_init(smem + 24, 2);
            mbarrier_init(smem + 32, 2);
            // a_free: 5 barriers, init_count=1
            mbarrier_init(smem + 40, 1);
            mbarrier_init(smem + 48, 1);
            mbarrier_init(smem + 56, 1);
            mbarrier_init(smem + 64, 1);
            mbarrier_init(smem + 72, 1);
            // b_full: 5 barriers, init_count=2
            mbarrier_init(smem + 80, 2);
            mbarrier_init(smem + 88, 2);
            mbarrier_init(smem + 96, 2);
            mbarrier_init(smem + 104, 2);
            mbarrier_init(smem + 112, 2);
            // b_free: 5 barriers, init_count=1
            mbarrier_init(smem + 120, 1);
            mbarrier_init(smem + 128, 1);
            mbarrier_init(smem + 136, 1);
            mbarrier_init(smem + 144, 1);
            mbarrier_init(smem + 152, 1);
            // --- pipeline 'mma_pipe' ---
            // mma_full: 2 barriers, init_count=1
            mbarrier_init(smem + 160, 1);
            mbarrier_init(smem + 168, 1);
            // mma_free: 2 barriers, init_count=256
            mbarrier_init(smem + 176, 256);
            mbarrier_init(smem + 184, 256);
            // --- pipeline 'work_pipe' ---
            // work_full: 3 barriers, init_count=1
            mbarrier_init(smem + 192, 1);
            mbarrier_init(smem + 200, 1);
            mbarrier_init(smem + 208, 1);
            // work_empty: 3 barriers, init_count=704
            mbarrier_init(smem + 216, 704);
            mbarrier_init(smem + 224, 704);
            mbarrier_init(smem + 232, 704);
            // --- pipeline 'throttle_pipe' ---
            // throttle_full: 3 barriers, init_count=32
            mbarrier_init(smem + 240, 32);
            mbarrier_init(smem + 248, 32);
            mbarrier_init(smem + 256, 32);
            // throttle_empty: 3 barriers, init_count=32
            mbarrier_init(smem + 264, 32);
            mbarrier_init(smem + 272, 32);
            mbarrier_init(smem + 280, 32);
            // --- pipeline 'drain_pipe' ---
            // drain_full: 1 barriers, init_count=1
            mbarrier_init(smem + 288, 1);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    // TMEM alloc (128 columns, 128 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 296);
    if (warp == 0) {
        int _tmem_hold = smem + 296;
        asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(128) : "memory");
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;");
    }

    asm volatile("barrier.cluster.arrive.release.aligned;");
    asm volatile("barrier.cluster.wait.acquire.aligned;");
    asm volatile("tcgen05.fence::after_thread_sync;");

    const int mbar_base = smem;
    #define a_full_addr (mbar_base + 0)
    #define a_free_addr (mbar_base + 40)
    #define b_full_addr (mbar_base + 80)
    #define b_free_addr (mbar_base + 120)
    #define mma_full_addr (mbar_base + 160)
    #define mma_free_addr (mbar_base + 176)
    #define work_full_addr (mbar_base + 192)
    #define work_empty_addr (mbar_base + 216)
    #define throttle_full_addr (mbar_base + 240)
    #define throttle_empty_addr (mbar_base + 264)
    #define drain_full_addr (mbar_base + 288)
    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_accum = taddr;
    asm volatile("griddepcontrol.wait;" ::: "memory");

    // ---- Role: epilogue ----
    if (warp <= 3) {
        { // epilogue_main
            asm volatile("barrier.sync 6, 224;" ::: "memory");
            asm volatile("setmaxnreg.dec.sync.aligned.u32 168;");
            asm volatile("griddepcontrol.wait;" ::: "memory");
            unsigned int acc_stage = 0;
            unsigned int work_stage = 0;
            unsigned int m_tile = blockIdx.x;
            unsigned int n_tile = blockIdx.y;
            const int warp_0 = warp;
            const int lane_1 = lane;
            int base_feature = warp_0 * 16 + lane_1 / 4 * 2;
            int base_token = lane_1 % 4 * 2;
            float output_pair[2];
            unsigned int packed_pair[1];
            unsigned int _phase_mma_full = 0;
            unsigned int _phase_work_full = 0;
            #pragma unroll 1
            for (unsigned int _tile_iter = 0; _tile_iter < gridDim.x / 2 * gridDim.y; _tile_iter++) {
                if (n_tile < (unsigned int)num_non_exiting_ctas[0]) {
                    int mn_limit = cta_idx_y_to_mn_limit[n_tile];
                    int padding_rows = (64 - mn_limit % 64) % 64;
                    mbarrier_wait(mma_full_addr + (acc_stage) * 8, _phase_mma_full);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    float _tmem_load_0[32];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[15])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[16])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[17])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[18])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[19])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[20])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[21])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[22])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[23])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[24])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[25])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[26])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[27])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[28])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[29])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[30])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[31]))
                        : "r"(taddr + acc_stage * 64));
                    float _tmem_load_1[32];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[15])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[16])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[17])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[18])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[19])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[20])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[21])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[22])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[23])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[24])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[25])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[26])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[27])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[28])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[29])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[30])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[31]))
                        : "r"(taddr + 1048576 + acc_stage * 64));
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    asm volatile("cp.async.bulk.wait_group.read 0;");
                    asm volatile("barrier.sync 4, 128;" ::: "memory");
                    int token = base_token;
                    float x0_lo = _tmem_load_0[0];
                    float x1_lo = _tmem_load_0[2];
                    float x0_hi = _tmem_load_1[0];
                    float x1_hi = _tmem_load_1[2];
                    float _exp2_0 = approx_exp2(-(x1_lo * 1.442695f));
                    output_pair[0] = x0_lo * (x1_lo / (1.0f + _exp2_0));
                    float _exp2_1 = approx_exp2(-(x1_hi * 1.442695f));
                    output_pair[1] = x0_hi * (x1_hi / (1.0f + _exp2_1));
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(output_pair[_lp*2 + 0], output_pair[_lp*2+1 + 0]));
                        packed_pair[_lp] = *(uint32_t*)&_bf2;
                    }
                    int smem_linear = token * 64 + base_feature;
                    int smem_element = smem_linear ^ token % 8 * 8;
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"(epi_staging_addr + (unsigned int)(smem_element * 2)), "r"((packed_pair[0])));
                    int token_0 = base_token + 1;
                    float x0_lo_1 = _tmem_load_0[1];
                    float x1_lo_2 = _tmem_load_0[3];
                    float x0_hi_3 = _tmem_load_1[1];
                    float x1_hi_4 = _tmem_load_1[3];
                    float _exp2_2 = approx_exp2(-(x1_lo_2 * 1.442695f));
                    output_pair[0] = x0_lo_1 * (x1_lo_2 / (1.0f + _exp2_2));
                    float _exp2_3 = approx_exp2(-(x1_hi_4 * 1.442695f));
                    output_pair[1] = x0_hi_3 * (x1_hi_4 / (1.0f + _exp2_3));
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(output_pair[_lp*2 + 0], output_pair[_lp*2+1 + 0]));
                        packed_pair[_lp] = *(uint32_t*)&_bf2;
                    }
                    int smem_linear_5 = token_0 * 64 + base_feature;
                    int smem_element_6 = smem_linear_5 ^ token_0 % 8 * 8;
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"(epi_staging_addr + (unsigned int)(smem_element_6 * 2)), "r"((packed_pair[0])));
                    int token_7 = base_token + 8;
                    float x0_lo_8 = _tmem_load_0[4];
                    float x1_lo_9 = _tmem_load_0[6];
                    float x0_hi_10 = _tmem_load_1[4];
                    float x1_hi_11 = _tmem_load_1[6];
                    float _exp2_4 = approx_exp2(-(x1_lo_9 * 1.442695f));
                    output_pair[0] = x0_lo_8 * (x1_lo_9 / (1.0f + _exp2_4));
                    float _exp2_5 = approx_exp2(-(x1_hi_11 * 1.442695f));
                    output_pair[1] = x0_hi_10 * (x1_hi_11 / (1.0f + _exp2_5));
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(output_pair[_lp*2 + 0], output_pair[_lp*2+1 + 0]));
                        packed_pair[_lp] = *(uint32_t*)&_bf2;
                    }
                    int smem_linear_12 = token_7 * 64 + base_feature;
                    int smem_element_13 = smem_linear_12 ^ token_7 % 8 * 8;
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"(epi_staging_addr + (unsigned int)(smem_element_13 * 2)), "r"((packed_pair[0])));
                    int token_14 = base_token + 8 + 1;
                    float x0_lo_15 = _tmem_load_0[5];
                    float x1_lo_16 = _tmem_load_0[7];
                    float x0_hi_17 = _tmem_load_1[5];
                    float x1_hi_18 = _tmem_load_1[7];
                    float _exp2_6 = approx_exp2(-(x1_lo_16 * 1.442695f));
                    output_pair[0] = x0_lo_15 * (x1_lo_16 / (1.0f + _exp2_6));
                    float _exp2_7 = approx_exp2(-(x1_hi_18 * 1.442695f));
                    output_pair[1] = x0_hi_17 * (x1_hi_18 / (1.0f + _exp2_7));
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(output_pair[_lp*2 + 0], output_pair[_lp*2+1 + 0]));
                        packed_pair[_lp] = *(uint32_t*)&_bf2;
                    }
                    int smem_linear_19 = token_14 * 64 + base_feature;
                    int smem_element_20 = smem_linear_19 ^ token_14 % 8 * 8;
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"(epi_staging_addr + (unsigned int)(smem_element_20 * 2)), "r"((packed_pair[0])));
                    int token_21 = base_token + 16;
                    float x0_lo_22 = _tmem_load_0[8];
                    float x1_lo_23 = _tmem_load_0[10];
                    float x0_hi_24 = _tmem_load_1[8];
                    float x1_hi_25 = _tmem_load_1[10];
                    float _exp2_8 = approx_exp2(-(x1_lo_23 * 1.442695f));
                    output_pair[0] = x0_lo_22 * (x1_lo_23 / (1.0f + _exp2_8));
                    float _exp2_9 = approx_exp2(-(x1_hi_25 * 1.442695f));
                    output_pair[1] = x0_hi_24 * (x1_hi_25 / (1.0f + _exp2_9));
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(output_pair[_lp*2 + 0], output_pair[_lp*2+1 + 0]));
                        packed_pair[_lp] = *(uint32_t*)&_bf2;
                    }
                    int smem_linear_26 = token_21 * 64 + base_feature;
                    int smem_element_27 = smem_linear_26 ^ token_21 % 8 * 8;
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"(epi_staging_addr + (unsigned int)(smem_element_27 * 2)), "r"((packed_pair[0])));
                    int token_28 = base_token + 16 + 1;
                    float x0_lo_29 = _tmem_load_0[9];
                    float x1_lo_30 = _tmem_load_0[11];
                    float x0_hi_31 = _tmem_load_1[9];
                    float x1_hi_32 = _tmem_load_1[11];
                    float _exp2_10 = approx_exp2(-(x1_lo_30 * 1.442695f));
                    output_pair[0] = x0_lo_29 * (x1_lo_30 / (1.0f + _exp2_10));
                    float _exp2_11 = approx_exp2(-(x1_hi_32 * 1.442695f));
                    output_pair[1] = x0_hi_31 * (x1_hi_32 / (1.0f + _exp2_11));
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(output_pair[_lp*2 + 0], output_pair[_lp*2+1 + 0]));
                        packed_pair[_lp] = *(uint32_t*)&_bf2;
                    }
                    int smem_linear_33 = token_28 * 64 + base_feature;
                    int smem_element_34 = smem_linear_33 ^ token_28 % 8 * 8;
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"(epi_staging_addr + (unsigned int)(smem_element_34 * 2)), "r"((packed_pair[0])));
                    int token_35 = base_token + 24;
                    float x0_lo_36 = _tmem_load_0[12];
                    float x1_lo_37 = _tmem_load_0[14];
                    float x0_hi_38 = _tmem_load_1[12];
                    float x1_hi_39 = _tmem_load_1[14];
                    float _exp2_12 = approx_exp2(-(x1_lo_37 * 1.442695f));
                    output_pair[0] = x0_lo_36 * (x1_lo_37 / (1.0f + _exp2_12));
                    float _exp2_13 = approx_exp2(-(x1_hi_39 * 1.442695f));
                    output_pair[1] = x0_hi_38 * (x1_hi_39 / (1.0f + _exp2_13));
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(output_pair[_lp*2 + 0], output_pair[_lp*2+1 + 0]));
                        packed_pair[_lp] = *(uint32_t*)&_bf2;
                    }
                    int smem_linear_40 = token_35 * 64 + base_feature;
                    int smem_element_41 = smem_linear_40 ^ token_35 % 8 * 8;
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"(epi_staging_addr + (unsigned int)(smem_element_41 * 2)), "r"((packed_pair[0])));
                    int token_42 = base_token + 24 + 1;
                    float x0_lo_43 = _tmem_load_0[13];
                    float x1_lo_44 = _tmem_load_0[15];
                    float x0_hi_45 = _tmem_load_1[13];
                    float x1_hi_46 = _tmem_load_1[15];
                    float _exp2_14 = approx_exp2(-(x1_lo_44 * 1.442695f));
                    output_pair[0] = x0_lo_43 * (x1_lo_44 / (1.0f + _exp2_14));
                    float _exp2_15 = approx_exp2(-(x1_hi_46 * 1.442695f));
                    output_pair[1] = x0_hi_45 * (x1_hi_46 / (1.0f + _exp2_15));
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(output_pair[_lp*2 + 0], output_pair[_lp*2+1 + 0]));
                        packed_pair[_lp] = *(uint32_t*)&_bf2;
                    }
                    int smem_linear_47 = token_42 * 64 + base_feature;
                    int smem_element_48 = smem_linear_47 ^ token_42 % 8 * 8;
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"(epi_staging_addr + (unsigned int)(smem_element_48 * 2)), "r"((packed_pair[0])));
                    int token_49 = base_token + 32;
                    float x0_lo_50 = _tmem_load_0[16];
                    float x1_lo_51 = _tmem_load_0[18];
                    float x0_hi_52 = _tmem_load_1[16];
                    float x1_hi_53 = _tmem_load_1[18];
                    float _exp2_16 = approx_exp2(-(x1_lo_51 * 1.442695f));
                    output_pair[0] = x0_lo_50 * (x1_lo_51 / (1.0f + _exp2_16));
                    float _exp2_17 = approx_exp2(-(x1_hi_53 * 1.442695f));
                    output_pair[1] = x0_hi_52 * (x1_hi_53 / (1.0f + _exp2_17));
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(output_pair[_lp*2 + 0], output_pair[_lp*2+1 + 0]));
                        packed_pair[_lp] = *(uint32_t*)&_bf2;
                    }
                    int smem_linear_54 = token_49 * 64 + base_feature;
                    int smem_element_55 = smem_linear_54 ^ token_49 % 8 * 8;
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"(epi_staging_addr + (unsigned int)(smem_element_55 * 2)), "r"((packed_pair[0])));
                    int token_56 = base_token + 32 + 1;
                    float x0_lo_57 = _tmem_load_0[17];
                    float x1_lo_58 = _tmem_load_0[19];
                    float x0_hi_59 = _tmem_load_1[17];
                    float x1_hi_60 = _tmem_load_1[19];
                    float _exp2_18 = approx_exp2(-(x1_lo_58 * 1.442695f));
                    output_pair[0] = x0_lo_57 * (x1_lo_58 / (1.0f + _exp2_18));
                    float _exp2_19 = approx_exp2(-(x1_hi_60 * 1.442695f));
                    output_pair[1] = x0_hi_59 * (x1_hi_60 / (1.0f + _exp2_19));
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(output_pair[_lp*2 + 0], output_pair[_lp*2+1 + 0]));
                        packed_pair[_lp] = *(uint32_t*)&_bf2;
                    }
                    int smem_linear_61 = token_56 * 64 + base_feature;
                    int smem_element_62 = smem_linear_61 ^ token_56 % 8 * 8;
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"(epi_staging_addr + (unsigned int)(smem_element_62 * 2)), "r"((packed_pair[0])));
                    int token_63 = base_token + 40;
                    float x0_lo_64 = _tmem_load_0[20];
                    float x1_lo_65 = _tmem_load_0[22];
                    float x0_hi_66 = _tmem_load_1[20];
                    float x1_hi_67 = _tmem_load_1[22];
                    float _exp2_20 = approx_exp2(-(x1_lo_65 * 1.442695f));
                    output_pair[0] = x0_lo_64 * (x1_lo_65 / (1.0f + _exp2_20));
                    float _exp2_21 = approx_exp2(-(x1_hi_67 * 1.442695f));
                    output_pair[1] = x0_hi_66 * (x1_hi_67 / (1.0f + _exp2_21));
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(output_pair[_lp*2 + 0], output_pair[_lp*2+1 + 0]));
                        packed_pair[_lp] = *(uint32_t*)&_bf2;
                    }
                    int smem_linear_68 = token_63 * 64 + base_feature;
                    int smem_element_69 = smem_linear_68 ^ token_63 % 8 * 8;
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"(epi_staging_addr + (unsigned int)(smem_element_69 * 2)), "r"((packed_pair[0])));
                    int token_70 = base_token + 40 + 1;
                    float x0_lo_71 = _tmem_load_0[21];
                    float x1_lo_72 = _tmem_load_0[23];
                    float x0_hi_73 = _tmem_load_1[21];
                    float x1_hi_74 = _tmem_load_1[23];
                    float _exp2_22 = approx_exp2(-(x1_lo_72 * 1.442695f));
                    output_pair[0] = x0_lo_71 * (x1_lo_72 / (1.0f + _exp2_22));
                    float _exp2_23 = approx_exp2(-(x1_hi_74 * 1.442695f));
                    output_pair[1] = x0_hi_73 * (x1_hi_74 / (1.0f + _exp2_23));
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(output_pair[_lp*2 + 0], output_pair[_lp*2+1 + 0]));
                        packed_pair[_lp] = *(uint32_t*)&_bf2;
                    }
                    int smem_linear_75 = token_70 * 64 + base_feature;
                    int smem_element_76 = smem_linear_75 ^ token_70 % 8 * 8;
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"(epi_staging_addr + (unsigned int)(smem_element_76 * 2)), "r"((packed_pair[0])));
                    int token_77 = base_token + 48;
                    float x0_lo_78 = _tmem_load_0[24];
                    float x1_lo_79 = _tmem_load_0[26];
                    float x0_hi_80 = _tmem_load_1[24];
                    float x1_hi_81 = _tmem_load_1[26];
                    float _exp2_24 = approx_exp2(-(x1_lo_79 * 1.442695f));
                    output_pair[0] = x0_lo_78 * (x1_lo_79 / (1.0f + _exp2_24));
                    float _exp2_25 = approx_exp2(-(x1_hi_81 * 1.442695f));
                    output_pair[1] = x0_hi_80 * (x1_hi_81 / (1.0f + _exp2_25));
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(output_pair[_lp*2 + 0], output_pair[_lp*2+1 + 0]));
                        packed_pair[_lp] = *(uint32_t*)&_bf2;
                    }
                    int smem_linear_82 = token_77 * 64 + base_feature;
                    int smem_element_83 = smem_linear_82 ^ token_77 % 8 * 8;
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"(epi_staging_addr + (unsigned int)(smem_element_83 * 2)), "r"((packed_pair[0])));
                    int token_84 = base_token + 48 + 1;
                    float x0_lo_85 = _tmem_load_0[25];
                    float x1_lo_86 = _tmem_load_0[27];
                    float x0_hi_87 = _tmem_load_1[25];
                    float x1_hi_88 = _tmem_load_1[27];
                    float _exp2_26 = approx_exp2(-(x1_lo_86 * 1.442695f));
                    output_pair[0] = x0_lo_85 * (x1_lo_86 / (1.0f + _exp2_26));
                    float _exp2_27 = approx_exp2(-(x1_hi_88 * 1.442695f));
                    output_pair[1] = x0_hi_87 * (x1_hi_88 / (1.0f + _exp2_27));
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(output_pair[_lp*2 + 0], output_pair[_lp*2+1 + 0]));
                        packed_pair[_lp] = *(uint32_t*)&_bf2;
                    }
                    int smem_linear_89 = token_84 * 64 + base_feature;
                    int smem_element_90 = smem_linear_89 ^ token_84 % 8 * 8;
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"(epi_staging_addr + (unsigned int)(smem_element_90 * 2)), "r"((packed_pair[0])));
                    int token_91 = base_token + 56;
                    float x0_lo_92 = _tmem_load_0[28];
                    float x1_lo_93 = _tmem_load_0[30];
                    float x0_hi_94 = _tmem_load_1[28];
                    float x1_hi_95 = _tmem_load_1[30];
                    float _exp2_28 = approx_exp2(-(x1_lo_93 * 1.442695f));
                    output_pair[0] = x0_lo_92 * (x1_lo_93 / (1.0f + _exp2_28));
                    float _exp2_29 = approx_exp2(-(x1_hi_95 * 1.442695f));
                    output_pair[1] = x0_hi_94 * (x1_hi_95 / (1.0f + _exp2_29));
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(output_pair[_lp*2 + 0], output_pair[_lp*2+1 + 0]));
                        packed_pair[_lp] = *(uint32_t*)&_bf2;
                    }
                    int smem_linear_96 = token_91 * 64 + base_feature;
                    int smem_element_97 = smem_linear_96 ^ token_91 % 8 * 8;
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"(epi_staging_addr + (unsigned int)(smem_element_97 * 2)), "r"((packed_pair[0])));
                    int token_98 = base_token + 56 + 1;
                    float x0_lo_99 = _tmem_load_0[29];
                    float x1_lo_100 = _tmem_load_0[31];
                    float x0_hi_101 = _tmem_load_1[29];
                    float x1_hi_102 = _tmem_load_1[31];
                    float _exp2_30 = approx_exp2(-(x1_lo_100 * 1.442695f));
                    output_pair[0] = x0_lo_99 * (x1_lo_100 / (1.0f + _exp2_30));
                    float _exp2_31 = approx_exp2(-(x1_hi_102 * 1.442695f));
                    output_pair[1] = x0_hi_101 * (x1_hi_102 / (1.0f + _exp2_31));
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(output_pair[_lp*2 + 0], output_pair[_lp*2+1 + 0]));
                        packed_pair[_lp] = *(uint32_t*)&_bf2;
                    }
                    int smem_linear_103 = token_98 * 64 + base_feature;
                    int smem_element_104 = smem_linear_103 ^ token_98 % 8 * 8;
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"(epi_staging_addr + (unsigned int)(smem_element_104 * 2)), "r"((packed_pair[0])));
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    asm volatile("barrier.sync 4, 128;" ::: "memory");
                    if (warp == 0) {
                        if (elect_sync()) {
                            tma_store_4d(compact_intermediate, m_tile * 64, padding_rows, 1073741824, n_tile * 64 - (unsigned int)padding_rows + 1073741824, epi_staging_addr);
                        }
                    }
                    asm volatile("cp.async.bulk.commit_group;");
                    asm volatile("barrier.sync 4, 128;" ::: "memory");
                    asm volatile(
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                        :: "r"((mma_free_addr + (acc_stage) * 8) & 0xFEFFFFFF) : "memory");
                    acc_stage += 1;
                    if (acc_stage == 2) { acc_stage = 0; _phase_mma_full ^= 1; }
                    acc_stage += 1;
                    if (acc_stage == 2) { acc_stage = 0; _phase_mma_full ^= 1; }
                }
                mbarrier_wait(work_full_addr + (work_stage) * 8, _phase_work_full);
                uint32_t _clc_valid_3 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "selp.u32 %0, 1, 0, p1;\n\t"
                    "}\n"
                    : "=r"(_clc_valid_3)
                    : "r"(work_response_addr + work_stage * 16)
                    : "memory");
                uint32_t _clc_ctaid_6 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "=r"(_clc_ctaid_6)
                    : "r"(work_response_addr + work_stage * 16)
                    : "memory");
                uint32_t _clc_ctaid_7 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.get_first_ctaid::y.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "=r"(_clc_ctaid_7)
                    : "r"(work_response_addr + work_stage * 16)
                    : "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile(
                    "{\n\t"
                    ".reg .b32 remAddr32;\n\t"
                    "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                    "}"
                    :: "r"(work_empty_addr + work_stage * 8), "r"(0) : "memory");
                work_stage += 1;
                if (work_stage == 3) { work_stage = 0; _phase_work_full ^= 1; }
                if (_clc_valid_3 == 0) {
                    break;
                }
                m_tile = _clc_ctaid_6 + (unsigned int)cta_rank;
                n_tile = _clc_ctaid_7;
            }
            asm volatile("barrier.sync 7, 128;" ::: "memory");
            if (warp == 0) {
                asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
                asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
                int _tmem_dealloc_addr = *((volatile int*)tmem_addr_storage);
                asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;" :: "r"(_tmem_dealloc_addr), "r"(128));
            }
        }
    }
    // ---- Role: mma ----
    if (warp == 4) {
        { // mma_main
            asm volatile("barrier.sync 6, 224;" ::: "memory");
            asm volatile("setmaxnreg.dec.sync.aligned.u32 96;");
            unsigned int _phase_mma_free = 1;
            unsigned int _phase_a_full = 0;
            unsigned int _phase_b_full = 0;
            unsigned int _phase_work_full_1 = 0;
            if (cta_rank == 0) {
                unsigned int stage = 0;
                unsigned int acc_stage_1 = 0;
                unsigned int work_stage_1 = 0;
                unsigned int m_tile_1 = blockIdx.x;
                unsigned int n_tile_1 = blockIdx.y;
                #pragma unroll 1
                for (unsigned int _tile_iter_1 = 0; _tile_iter_1 < gridDim.x / 2 * gridDim.y; _tile_iter_1++) {
                    if (n_tile_1 < (unsigned int)num_non_exiting_ctas[0]) {
                        mbarrier_wait(mma_free_addr + (acc_stage_1) * 8, _phase_mma_free);
                        #pragma unroll 1
                        for (int k_pair = 0; k_pair < (K + 128 - 1) / 128; k_pair += 2) {
                            mbarrier_wait(a_full_addr + (stage) * 8, _phase_a_full);
                            mbarrier_wait(b_full_addr + (stage) * 8, _phase_b_full);
                            asm volatile("tcgen05.fence::after_thread_sync;");
                            int _mma_a_lo_0 = (((smem_a_addr) >> 4) & 0x3FFF) + (stage) * 2048;
                            int _mma_b_lo_0 = (((smem_b_addr) >> 4) & 0x3FFF) + (stage) * 512;
                            asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id, m0, m1, m2, m3, m4, m5, m6, m7;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    "mov.b32 m0, 0; mov.b32 m1, 0; mov.b32 m2, 0; mov.b32 m3, 0;\n\tmov.b32 m4, 0; mov.b32 m5, 0; mov.b32 m6, 0; mov.b32 m7, 0;\n\t"
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 269485200;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 1018;\n\t"
                    "add.u32 blo, blo, 250;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_0), "r"(_mma_b_lo_0), "r"((tmem_accum + (acc_stage_1 * 64))), "r"(((((k_pair == 0) ? 1 : 0)) ? 0 : 1)));
                            elect_commit_cg2_multicast(a_free_addr + (stage) * 8, (uint16_t)(3));
                            elect_commit_cg2_multicast(b_free_addr + (stage) * 8, (uint16_t)(3));
                            stage += 1;
                            if (stage == 5) { stage = 0; _phase_a_full ^= 1; _phase_b_full ^= 1; }
                            mbarrier_wait(a_full_addr + (stage) * 8, _phase_a_full);
                            mbarrier_wait(b_full_addr + (stage) * 8, _phase_b_full);
                            asm volatile("tcgen05.fence::after_thread_sync;");
                            int _mma_a_lo_1 = (((smem_a_addr) >> 4) & 0x3FFF) + (stage) * 2048;
                            int _mma_b_lo_1 = (((smem_b_addr) >> 4) & 0x3FFF) + (stage) * 512;
                            asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id, m0, m1, m2, m3, m4, m5, m6, m7;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    "mov.b32 m0, 0; mov.b32 m1, 0; mov.b32 m2, 0; mov.b32 m3, 0;\n\tmov.b32 m4, 0; mov.b32 m5, 0; mov.b32 m6, 0; mov.b32 m7, 0;\n\t"
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 269485200;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 1018;\n\t"
                    "add.u32 blo, blo, 250;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_1), "r"(_mma_b_lo_1), "r"((tmem_accum + (acc_stage_1 * 64))), "r"(((((0) ? 1 : 0)) ? 0 : 1)));
                            elect_commit_cg2_multicast(a_free_addr + (stage) * 8, (uint16_t)(3));
                            elect_commit_cg2_multicast(b_free_addr + (stage) * 8, (uint16_t)(3));
                            stage += 1;
                            if (stage == 5) { stage = 0; _phase_a_full ^= 1; _phase_b_full ^= 1; }
                        }
                        elect_commit_cg2_multicast(mma_full_addr + (acc_stage_1) * 8, (uint16_t)(3));
                        acc_stage_1 += 1;
                        if (acc_stage_1 == 2) { acc_stage_1 = 0; _phase_mma_free ^= 1; }
                        acc_stage_1 += 1;
                        if (acc_stage_1 == 2) { acc_stage_1 = 0; _phase_mma_free ^= 1; }
                    }
                    mbarrier_wait(work_full_addr + (work_stage_1) * 8, _phase_work_full_1);
                    uint32_t _clc_valid_2 = 0;
                    asm volatile(
                        "{\n\t"
                        ".reg .pred p1;\n\t"
                        ".reg .b128 clc_r;\n\t"
                        "ld.shared.b128 clc_r, [%1];\n\t"
                        "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                        "selp.u32 %0, 1, 0, p1;\n\t"
                        "}\n"
                        : "=r"(_clc_valid_2)
                        : "r"(work_response_addr + work_stage_1 * 16)
                        : "memory");
                    uint32_t _clc_ctaid_4 = 0;
                    asm volatile(
                        "{\n\t"
                        ".reg .b128 clc_r;\n\t"
                        "ld.shared.b128 clc_r, [%1];\n\t"
                        "clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                        "}\n"
                        : "=r"(_clc_ctaid_4)
                        : "r"(work_response_addr + work_stage_1 * 16)
                        : "memory");
                    uint32_t _clc_ctaid_5 = 0;
                    asm volatile(
                        "{\n\t"
                        ".reg .b128 clc_r;\n\t"
                        "ld.shared.b128 clc_r, [%1];\n\t"
                        "clusterlaunchcontrol.query_cancel.get_first_ctaid::y.b32.b128 %0, clc_r;\n\t"
                        "}\n"
                        : "=r"(_clc_ctaid_5)
                        : "r"(work_response_addr + work_stage_1 * 16)
                        : "memory");
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    asm volatile(
                        "{\n\t"
                        ".reg .b32 remAddr32;\n\t"
                        "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                        "}"
                        :: "r"(work_empty_addr + work_stage_1 * 8), "r"(0) : "memory");
                    work_stage_1 += 1;
                    if (work_stage_1 == 3) { work_stage_1 = 0; _phase_work_full_1 ^= 1; }
                    if (_clc_valid_2 == 0) {
                        break;
                    }
                    m_tile_1 = _clc_ctaid_4 + (unsigned int)cta_rank;
                    n_tile_1 = _clc_ctaid_5;
                }
            }
        }
    }
    // ---- Role: load_a ----
    if (warp == 5) {
        { // load_a_main
            asm volatile("setmaxnreg.dec.sync.aligned.u32 96;");
            asm volatile("griddepcontrol.wait;" ::: "memory");
            unsigned int stage_1 = 0;
            unsigned int work_stage_2 = 0;
            unsigned int throttle_stage = 0;
            unsigned int m_tile_2 = blockIdx.x;
            unsigned int n_tile_2 = blockIdx.y;
            unsigned int cta_mask = 1 << cta_rank;
            unsigned int _phase_throttle_empty = 1;
            unsigned int _phase_a_free = 1;
            unsigned int _phase_work_full_2 = 0;
            #pragma unroll 1
            for (unsigned int _tile_iter_2 = 0; _tile_iter_2 < gridDim.x / 2 * gridDim.y; _tile_iter_2++) {
                if (n_tile_2 < (unsigned int)num_non_exiting_ctas[0]) {
                    int expert = cta_idx_y_to_batch_idx[n_tile_2];
                    if (cta_rank == 0) {
                        mbarrier_wait(throttle_empty_addr + (throttle_stage) * 8, _phase_throttle_empty);
                        mbarrier_arrive(throttle_full_addr + (throttle_stage) * 8);
                        throttle_stage += 1;
                        if (throttle_stage == 3) { throttle_stage = 0; _phase_throttle_empty ^= 1; }
                    }
                    #pragma unroll 1
                    for (int iter_k = 0; iter_k < (K + 128 - 1) / 128; iter_k++) {
                        mbarrier_wait(a_free_addr + (stage_1) * 8, _phase_a_free);
                        if (elect_sync()) {
                            asm volatile(
                                "cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                " [%0], [%1, {%2, %3, %4, %5}], [%6], %7;"
                                :: "r"(smem_a_addr + stage_1 * 32768), "l"(weights), "r"(0), "r"(m_tile_2 * 128), "r"(iter_k * 2), "r"(expert),
                                   "r"(((a_full_addr + (stage_1) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(cta_mask)) : "memory");
                            asm volatile(
                                "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                                :: "r"((a_full_addr + (stage_1) * 8) & 0xFEFFFFFF), "r"((uint32_t)(32768)) : "memory");
                        }
                        stage_1 += 1;
                        if (stage_1 == 5) { stage_1 = 0; _phase_a_free ^= 1; }
                    }
                }
                mbarrier_wait(work_full_addr + (work_stage_2) * 8, _phase_work_full_2);
                uint32_t _clc_valid_0 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "selp.u32 %0, 1, 0, p1;\n\t"
                    "}\n"
                    : "=r"(_clc_valid_0)
                    : "r"(work_response_addr + work_stage_2 * 16)
                    : "memory");
                uint32_t _clc_ctaid_0 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "=r"(_clc_ctaid_0)
                    : "r"(work_response_addr + work_stage_2 * 16)
                    : "memory");
                uint32_t _clc_ctaid_1 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.get_first_ctaid::y.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "=r"(_clc_ctaid_1)
                    : "r"(work_response_addr + work_stage_2 * 16)
                    : "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile(
                    "{\n\t"
                    ".reg .b32 remAddr32;\n\t"
                    "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                    "}"
                    :: "r"(work_empty_addr + work_stage_2 * 8), "r"(0) : "memory");
                work_stage_2 += 1;
                if (work_stage_2 == 3) { work_stage_2 = 0; _phase_work_full_2 ^= 1; }
                if (_clc_valid_0 == 0) {
                    break;
                }
                m_tile_2 = _clc_ctaid_0 + (unsigned int)cta_rank;
                n_tile_2 = _clc_ctaid_1;
            }
        }
    }
    // ---- Role: load_b ----
    if (warp >= 6 && warp <= 9) {
        { // load_b_main
            asm volatile("setmaxnreg.dec.sync.aligned.u32 96;");
            asm volatile("griddepcontrol.wait;" ::: "memory");
            unsigned int stage_2 = 0;
            unsigned int work_stage_3 = 0;
            unsigned int m_tile_3 = blockIdx.x;
            unsigned int n_tile_3 = blockIdx.y;
            int warp_local = warp - 6;
            int routed[8];
            unsigned int cta_mask_1 = 1 << cta_rank;
            unsigned int _phase_b_free = 1;
            unsigned int _phase_work_full_3 = 0;
            #pragma unroll 1
            for (unsigned int _tile_iter_3 = 0; _tile_iter_3 < gridDim.x / 2 * gridDim.y; _tile_iter_3++) {
                if (n_tile_3 < (unsigned int)num_non_exiting_ctas[0]) {
                    int route_base = n_tile_3 * 64 + (unsigned int)(cta_rank * 32) + (unsigned int)(warp_local * 4);
                    routed[0] = route_map[route_base];
                    routed[1] = route_map[route_base + 1];
                    routed[2] = route_map[route_base + 2];
                    routed[3] = route_map[route_base + 3];
                    int route_base_0 = n_tile_3 * 64 + (unsigned int)(cta_rank * 32) + (unsigned int)((4 + warp_local) * 4);
                    routed[4] = route_map[route_base_0];
                    routed[5] = route_map[route_base_0 + 1];
                    routed[6] = route_map[route_base_0 + 2];
                    routed[7] = route_map[route_base_0 + 3];
                    #pragma unroll 1
                    for (int iter_k_1 = 0; iter_k_1 < (K + 128 - 1) / 128; iter_k_1++) {
                        mbarrier_wait(b_free_addr + (stage_2) * 8, _phase_b_free);
                        if (elect_sync()) {
                            int dst = smem_b_addr + stage_2 * 8192;
                            tma_gather4_gmem2smem_mc_cta2(dst + warp_local * 512, recv_hidden, iter_k_1 * 128, routed[0], routed[1], routed[2], routed[3], ((b_full_addr + (stage_2) * 8) & 0xFEFFFFFF), cta_mask_1);
                            tma_gather4_gmem2smem_mc_cta2(dst + 4096 + warp_local * 512, recv_hidden, iter_k_1 * 128 + 64, routed[0], routed[1], routed[2], routed[3], ((b_full_addr + (stage_2) * 8) & 0xFEFFFFFF), cta_mask_1);
                            tma_gather4_gmem2smem_mc_cta2(dst + (4 + warp_local) * 512, recv_hidden, iter_k_1 * 128, routed[4], routed[5], routed[6], routed[7], ((b_full_addr + (stage_2) * 8) & 0xFEFFFFFF), cta_mask_1);
                            tma_gather4_gmem2smem_mc_cta2(dst + 4096 + (4 + warp_local) * 512, recv_hidden, iter_k_1 * 128 + 64, routed[4], routed[5], routed[6], routed[7], ((b_full_addr + (stage_2) * 8) & 0xFEFFFFFF), cta_mask_1);
                        }
                        if (warp == 6) {
                            if (elect_sync()) {
                                asm volatile(
                                    "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                                    :: "r"((b_full_addr + (stage_2) * 8) & 0xFEFFFFFF), "r"((uint32_t)(8192)) : "memory");
                            }
                        }
                        stage_2 += 1;
                        if (stage_2 == 5) { stage_2 = 0; _phase_b_free ^= 1; }
                    }
                }
                mbarrier_wait(work_full_addr + (work_stage_3) * 8, _phase_work_full_3);
                uint32_t _clc_valid_1 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "selp.u32 %0, 1, 0, p1;\n\t"
                    "}\n"
                    : "=r"(_clc_valid_1)
                    : "r"(work_response_addr + work_stage_3 * 16)
                    : "memory");
                uint32_t _clc_ctaid_2 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "=r"(_clc_ctaid_2)
                    : "r"(work_response_addr + work_stage_3 * 16)
                    : "memory");
                uint32_t _clc_ctaid_3 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.get_first_ctaid::y.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "=r"(_clc_ctaid_3)
                    : "r"(work_response_addr + work_stage_3 * 16)
                    : "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile(
                    "{\n\t"
                    ".reg .b32 remAddr32;\n\t"
                    "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                    "}"
                    :: "r"(work_empty_addr + work_stage_3 * 8), "r"(0) : "memory");
                work_stage_3 += 1;
                if (work_stage_3 == 3) { work_stage_3 = 0; _phase_work_full_3 ^= 1; }
                if (_clc_valid_1 == 0) {
                    break;
                }
                m_tile_3 = _clc_ctaid_2 + (unsigned int)cta_rank;
                n_tile_3 = _clc_ctaid_3;
            }
            asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
        }
    }
    // ---- Role: work_id ----
    if (warp == 10) {
        { // work_id_main
            asm volatile("barrier.sync 6, 224;" ::: "memory");
            asm volatile("setmaxnreg.dec.sync.aligned.u32 96;");
            asm volatile("griddepcontrol.wait;" ::: "memory");
            unsigned int _phase_throttle_full = 0;
            unsigned int _phase_drain_full = 0;
            unsigned int _phase_work_empty = 1;
            unsigned int _phase_work_full_4 = 0;
            if (cta_rank == 0) {
                unsigned int work_stage_4 = 0;
                unsigned int throttle_stage_1 = 0;
                unsigned int drain_stage = 0;
                unsigned int n_tile_4 = blockIdx.y;
                #pragma unroll 1
                for (unsigned int _tile_iter_4 = 0; _tile_iter_4 < gridDim.x / 2 * gridDim.y; _tile_iter_4++) {
                    if (n_tile_4 < (unsigned int)num_non_exiting_ctas[0]) {
                        mbarrier_wait(throttle_full_addr + (throttle_stage_1) * 8, _phase_throttle_full);
                        mbarrier_arrive(throttle_empty_addr + (throttle_stage_1) * 8);
                        throttle_stage_1 += 1;
                        if (throttle_stage_1 == 3) { throttle_stage_1 = 0; _phase_throttle_full ^= 1; }
                    } else {
                        #pragma unroll 1
                        for (unsigned int _drain_iter = 0; _drain_iter < (gridDim.x / 2 * gridDim.y + 4 - 1) / 4 + 1; _drain_iter++) {
                            if (elect_sync()) {
                                mbarrier_arrive_expect_tx(drain_full_addr + (drain_stage) * 8, 64);
                                asm volatile(
                                    "fence.proxy.async.shared::cta;\n\t"
                                    "clusterlaunchcontrol.try_cancel.async.shared::cta"
                                        ".mbarrier::complete_tx::bytes.b128"
                                        " [%0], [%1];"
                                    :: "r"(fast_drain_response_addr), "r"(drain_full_addr + drain_stage * 8)
                                    : "memory");
                                asm volatile(
                                    "fence.proxy.async.shared::cta;\n\t"
                                    "clusterlaunchcontrol.try_cancel.async.shared::cta"
                                        ".mbarrier::complete_tx::bytes.b128"
                                        " [%0], [%1];"
                                    :: "r"(fast_drain_response_addr + 16), "r"(drain_full_addr + drain_stage * 8)
                                    : "memory");
                                asm volatile(
                                    "fence.proxy.async.shared::cta;\n\t"
                                    "clusterlaunchcontrol.try_cancel.async.shared::cta"
                                        ".mbarrier::complete_tx::bytes.b128"
                                        " [%0], [%1];"
                                    :: "r"(fast_drain_response_addr + 32), "r"(drain_full_addr + drain_stage * 8)
                                    : "memory");
                                asm volatile(
                                    "fence.proxy.async.shared::cta;\n\t"
                                    "clusterlaunchcontrol.try_cancel.async.shared::cta"
                                        ".mbarrier::complete_tx::bytes.b128"
                                        " [%0], [%1];"
                                    :: "r"(fast_drain_response_addr + 48), "r"(drain_full_addr + drain_stage * 8)
                                    : "memory");
                            }
                            mbarrier_wait(drain_full_addr + (drain_stage) * 8, _phase_drain_full);
                            unsigned int canceled = 0;
                            uint32_t _clc_valid_4 = 0;
                            asm volatile(
                                "{\n\t"
                                ".reg .pred p1;\n\t"
                                ".reg .b128 clc_r;\n\t"
                                "ld.shared.b128 clc_r, [%1];\n\t"
                                "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                                "selp.u32 %0, 1, 0, p1;\n\t"
                                "}\n"
                                : "=r"(_clc_valid_4)
                                : "r"(fast_drain_response_addr)
                                : "memory");
                            canceled += _clc_valid_4;
                            uint32_t _clc_valid_5 = 0;
                            asm volatile(
                                "{\n\t"
                                ".reg .pred p1;\n\t"
                                ".reg .b128 clc_r;\n\t"
                                "ld.shared.b128 clc_r, [%1];\n\t"
                                "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                                "selp.u32 %0, 1, 0, p1;\n\t"
                                "}\n"
                                : "=r"(_clc_valid_5)
                                : "r"(fast_drain_response_addr + 16)
                                : "memory");
                            canceled += _clc_valid_5;
                            uint32_t _clc_valid_6 = 0;
                            asm volatile(
                                "{\n\t"
                                ".reg .pred p1;\n\t"
                                ".reg .b128 clc_r;\n\t"
                                "ld.shared.b128 clc_r, [%1];\n\t"
                                "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                                "selp.u32 %0, 1, 0, p1;\n\t"
                                "}\n"
                                : "=r"(_clc_valid_6)
                                : "r"(fast_drain_response_addr + 32)
                                : "memory");
                            canceled += _clc_valid_6;
                            uint32_t _clc_valid_7 = 0;
                            asm volatile(
                                "{\n\t"
                                ".reg .pred p1;\n\t"
                                ".reg .b128 clc_r;\n\t"
                                "ld.shared.b128 clc_r, [%1];\n\t"
                                "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                                "selp.u32 %0, 1, 0, p1;\n\t"
                                "}\n"
                                : "=r"(_clc_valid_7)
                                : "r"(fast_drain_response_addr + 48)
                                : "memory");
                            canceled += _clc_valid_7;
                            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                            drain_stage += 1;
                            if (drain_stage == 1) { drain_stage = 0; _phase_drain_full ^= 1; }
                            if (canceled == 0) {
                                break;
                            }
                        }
                    }
                    mbarrier_wait_cluster(work_empty_addr + (work_stage_4) * 8, _phase_work_empty);
                    if (lane < 2) {
                        asm volatile(
                            "{\n\t"
                            ".reg .b32 remAddr32;\n\t"
                            "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                            "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [remAddr32], %2;\n\t"
                            "}"
                            :: "r"(work_full_addr + work_stage_4 * 8), "r"(lane), "r"((uint32_t)(16)) : "memory");
                    }
                    if (elect_sync()) {
                        asm volatile(
                            "fence.proxy.async.shared::cta;\n\t"
                            "clusterlaunchcontrol.try_cancel.async.shared::cta"
                                ".mbarrier::complete_tx::bytes.multicast::cluster::all.b128"
                                " [%0], [%1];"
                            :: "r"(work_response_addr + work_stage_4 * 16), "r"(work_full_addr + work_stage_4 * 8)
                            : "memory");
                    }
                    mbarrier_wait(work_full_addr + (work_stage_4) * 8, _phase_work_full_4);
                    uint32_t _clc_valid_8 = 0;
                    asm volatile(
                        "{\n\t"
                        ".reg .pred p1;\n\t"
                        ".reg .b128 clc_r;\n\t"
                        "ld.shared.b128 clc_r, [%1];\n\t"
                        "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                        "selp.u32 %0, 1, 0, p1;\n\t"
                        "}\n"
                        : "=r"(_clc_valid_8)
                        : "r"(work_response_addr + work_stage_4 * 16)
                        : "memory");
                    uint32_t _clc_ctaid_8 = 0;
                    asm volatile(
                        "{\n\t"
                        ".reg .b128 clc_r;\n\t"
                        "ld.shared.b128 clc_r, [%1];\n\t"
                        "clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                        "}\n"
                        : "=r"(_clc_ctaid_8)
                        : "r"(work_response_addr + work_stage_4 * 16)
                        : "memory");
                    uint32_t _clc_ctaid_9 = 0;
                    asm volatile(
                        "{\n\t"
                        ".reg .b128 clc_r;\n\t"
                        "ld.shared.b128 clc_r, [%1];\n\t"
                        "clusterlaunchcontrol.query_cancel.get_first_ctaid::y.b32.b128 %0, clc_r;\n\t"
                        "}\n"
                        : "=r"(_clc_ctaid_9)
                        : "r"(work_response_addr + work_stage_4 * 16)
                        : "memory");
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    asm volatile(
                        "{\n\t"
                        ".reg .b32 remAddr32;\n\t"
                        "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                        "}"
                        :: "r"(work_empty_addr + work_stage_4 * 8), "r"(0) : "memory");
                    work_stage_4 += 1;
                    if (work_stage_4 == 3) { work_stage_4 = 0; _phase_work_empty ^= 1; _phase_work_full_4 ^= 1; }
                    if (_clc_valid_8 == 0) {
                        break;
                    }
                    n_tile_4 = _clc_ctaid_9;
                }
                #pragma unroll
                for (int _tail = 0; _tail < 3; _tail++) {
                    mbarrier_wait_cluster(work_empty_addr + (work_stage_4) * 8, _phase_work_empty);
                    work_stage_4 += 1;
                    if (work_stage_4 == 3) { work_stage_4 = 0; _phase_work_empty ^= 1; _phase_work_full_4 ^= 1; }
                }
            }
        }
    }
    // ---- Role: padding ----
    if (warp == 11) {
        { // padding_main
            asm volatile("barrier.sync 6, 224;" ::: "memory");
            asm volatile("setmaxnreg.dec.sync.aligned.u32 96;");
            unsigned int work_stage_5 = 0;
            unsigned int n_tile_5 = blockIdx.y;
            unsigned int _phase_work_full_5 = 0;
            #pragma unroll 1
            for (unsigned int _tile_iter_5 = 0; _tile_iter_5 < gridDim.x / 2 * gridDim.y; _tile_iter_5++) {
                mbarrier_wait(work_full_addr + (work_stage_5) * 8, _phase_work_full_5);
                uint32_t _clc_valid_9 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "selp.u32 %0, 1, 0, p1;\n\t"
                    "}\n"
                    : "=r"(_clc_valid_9)
                    : "r"(work_response_addr + work_stage_5 * 16)
                    : "memory");
                uint32_t _clc_ctaid_10 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "=r"(_clc_ctaid_10)
                    : "r"(work_response_addr + work_stage_5 * 16)
                    : "memory");
                uint32_t _clc_ctaid_11 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.get_first_ctaid::y.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "=r"(_clc_ctaid_11)
                    : "r"(work_response_addr + work_stage_5 * 16)
                    : "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile(
                    "{\n\t"
                    ".reg .b32 remAddr32;\n\t"
                    "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                    "}"
                    :: "r"(work_empty_addr + work_stage_5 * 8), "r"(0) : "memory");
                work_stage_5 += 1;
                if (work_stage_5 == 3) { work_stage_5 = 0; _phase_work_full_5 ^= 1; }
                if (_clc_valid_9 == 0) {
                    break;
                }
                n_tile_5 = _clc_ctaid_11;
            }
        }
    }

    // Cleanup
}

} // extern "C"

#undef FLASHINFER_INF
#undef NUM_DRAIN_PIPE_STAGES
#undef NUM_K_PIPE_STAGES
#undef NUM_MMA_PIPE_STAGES
#undef NUM_THROTTLE_PIPE_STAGES
#undef NUM_WORK_PIPE_STAGES
#undef SMEM_EPI_STAGING_OFF
#undef SMEM_EPI_STAGING_STAGE_BYTES
#undef SMEM_EPI_STAGING_STRIDE
#undef SMEM_FAST_DRAIN_RESPONSE_OFF
#undef SMEM_FAST_DRAIN_RESPONSE_STAGE_BYTES
#undef SMEM_FAST_DRAIN_RESPONSE_STRIDE
#undef SMEM_SMEM_A_OFF
#undef SMEM_SMEM_A_STAGE_BYTES
#undef SMEM_SMEM_A_STRIDE
#undef SMEM_SMEM_B_OFF
#undef SMEM_SMEM_B_STAGE_BYTES
#undef SMEM_SMEM_B_STRIDE
#undef SMEM_TOTAL
#undef SMEM_WORK_RESPONSE_OFF
#undef SMEM_WORK_RESPONSE_STAGE_BYTES
#undef SMEM_WORK_RESPONSE_STRIDE
#undef THREADS
#undef TMEM_ACCUM_OFFSET
#undef TMEM_NCOLS
#undef a_free_addr
#undef a_full_addr
#undef b_free_addr
#undef b_full_addr
#undef drain_full_addr
#undef epi_staging_addr
#undef fast_drain_response_addr
#undef mma_free_addr
#undef mma_full_addr
#undef smem_a_addr
#undef smem_b_addr
#undef throttle_empty_addr
#undef throttle_full_addr
#undef work_empty_addr
#undef work_full_addr
#undef work_response_addr

#define FLASHINFER_INF CUDART_INF_F
#define TMEM_NCOLS 128
#define TMEM_ACCUM_OFFSET 0
#define NUM_K_PIPE_STAGES 5
#define NUM_MMA_PIPE_STAGES 2
#define NUM_WORK_PIPE_STAGES 3
#define NUM_THROTTLE_PIPE_STAGES 3
#define NUM_DRAIN_PIPE_STAGES 1
#define SMEM_SMEM_A_OFF 1024
#define SMEM_SMEM_A_STAGE_BYTES 32768
#define SMEM_SMEM_A_STRIDE 32768
#define SMEM_SMEM_B_OFF 164864
#define SMEM_SMEM_B_STAGE_BYTES 8192
#define SMEM_SMEM_B_STRIDE 8192
#define SMEM_EPI_STAGING_OFF 205824
#define SMEM_EPI_STAGING_STAGE_BYTES 16384
#define SMEM_EPI_STAGING_STRIDE 16384
#define SMEM_EPI_STAGING_U64_OFF 205824
#define SMEM_EPI_STAGING_U64_STAGE_BYTES 16384
#define SMEM_EPI_STAGING_U64_STRIDE 16384
#define SMEM_WORK_RESPONSE_OFF 223232
#define SMEM_WORK_RESPONSE_STAGE_BYTES 16
#define SMEM_WORK_RESPONSE_STRIDE 16
#define SMEM_FAST_DRAIN_RESPONSE_OFF 223280
#define SMEM_FAST_DRAIN_RESPONSE_STAGE_BYTES 16
#define SMEM_FAST_DRAIN_RESPONSE_STRIDE 16
#define SMEM_TOTAL 223360
#define THREADS 256

extern "C" {

__global__ __launch_bounds__(256, 1) __cluster_dims__(2,1,1) void
kernel_trtllm_moe_bmm_tile_n64_fc2_bf16(FlashInferTensorMap const* A, FlashInferTensorMap const* B, FlashInferTensorMap const* C, int* __restrict__ num_non_exiting_ctas, int* __restrict__ cta_idx_y_to_batch_idx, int* __restrict__ cta_idx_y_to_mn_limit, int K)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    const unsigned int clusters_x = gridDim.x / 2;
    const unsigned int cluster_id = ((blockIdx.z * gridDim.y + blockIdx.y) * clusters_x) + blockIdx.x / 2;
    const unsigned int num_clusters = clusters_x * gridDim.y * gridDim.z;

    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));
    if (tid == 0) {
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(A)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(B)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(C)) : "memory");
    }
    __syncthreads();


    // Kernel setup ops
    __nv_bfloat16* smem_a = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int smem_a_addr = smem + 1024;
    __nv_bfloat16* smem_b = reinterpret_cast<__nv_bfloat16*>(smem_raw + 164864);
    const int smem_b_addr = smem + 164864;
    __nv_bfloat16* epi_staging = reinterpret_cast<__nv_bfloat16*>(smem_raw + 205824);
    const int epi_staging_addr = smem + 205824;
    unsigned long long* epi_staging_u64 = reinterpret_cast<unsigned long long*>(smem_raw + 205824);
    const int epi_staging_u64_addr = smem + 205824;
    unsigned int* work_response = reinterpret_cast<unsigned int*>(smem_raw + 223232);
    const int work_response_addr = smem + 223232;
    unsigned int* fast_drain_response = reinterpret_cast<unsigned int*>(smem_raw + 223280);
    const int fast_drain_response_addr = smem + 223280;

    // Mbarrier init (11 groups, 37 barriers)
    // Mbarriers at smem_raw[0..296)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'k_pipe' ---
            // a_full: 5 barriers, init_count=2
            mbarrier_init(smem + 0, 2);
            mbarrier_init(smem + 8, 2);
            mbarrier_init(smem + 16, 2);
            mbarrier_init(smem + 24, 2);
            mbarrier_init(smem + 32, 2);
            // a_free: 5 barriers, init_count=1
            mbarrier_init(smem + 40, 1);
            mbarrier_init(smem + 48, 1);
            mbarrier_init(smem + 56, 1);
            mbarrier_init(smem + 64, 1);
            mbarrier_init(smem + 72, 1);
            // b_full: 5 barriers, init_count=2
            mbarrier_init(smem + 80, 2);
            mbarrier_init(smem + 88, 2);
            mbarrier_init(smem + 96, 2);
            mbarrier_init(smem + 104, 2);
            mbarrier_init(smem + 112, 2);
            // b_free: 5 barriers, init_count=1
            mbarrier_init(smem + 120, 1);
            mbarrier_init(smem + 128, 1);
            mbarrier_init(smem + 136, 1);
            mbarrier_init(smem + 144, 1);
            mbarrier_init(smem + 152, 1);
            // --- pipeline 'mma_pipe' ---
            // mma_full: 2 barriers, init_count=1
            mbarrier_init(smem + 160, 1);
            mbarrier_init(smem + 168, 1);
            // mma_free: 2 barriers, init_count=256
            mbarrier_init(smem + 176, 256);
            mbarrier_init(smem + 184, 256);
            // --- pipeline 'work_pipe' ---
            // work_full: 3 barriers, init_count=1
            mbarrier_init(smem + 192, 1);
            mbarrier_init(smem + 200, 1);
            mbarrier_init(smem + 208, 1);
            // work_empty: 3 barriers, init_count=448
            mbarrier_init(smem + 216, 448);
            mbarrier_init(smem + 224, 448);
            mbarrier_init(smem + 232, 448);
            // --- pipeline 'throttle_pipe' ---
            // throttle_full: 3 barriers, init_count=32
            mbarrier_init(smem + 240, 32);
            mbarrier_init(smem + 248, 32);
            mbarrier_init(smem + 256, 32);
            // throttle_empty: 3 barriers, init_count=32
            mbarrier_init(smem + 264, 32);
            mbarrier_init(smem + 272, 32);
            mbarrier_init(smem + 280, 32);
            // --- pipeline 'drain_pipe' ---
            // drain_full: 1 barriers, init_count=1
            mbarrier_init(smem + 288, 1);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    // TMEM alloc (128 columns, 128 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 296);
    if (warp == 0) {
        int _tmem_hold = smem + 296;
        asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(128) : "memory");
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;");
    }

    asm volatile("barrier.cluster.arrive.release.aligned;");
    asm volatile("barrier.cluster.wait.acquire.aligned;");
    asm volatile("tcgen05.fence::after_thread_sync;");

    const int mbar_base = smem;
    #define a_full_addr (mbar_base + 0)
    #define a_free_addr (mbar_base + 40)
    #define b_full_addr (mbar_base + 80)
    #define b_free_addr (mbar_base + 120)
    #define mma_full_addr (mbar_base + 160)
    #define mma_free_addr (mbar_base + 176)
    #define work_full_addr (mbar_base + 192)
    #define work_empty_addr (mbar_base + 216)
    #define throttle_full_addr (mbar_base + 240)
    #define throttle_empty_addr (mbar_base + 264)
    #define drain_full_addr (mbar_base + 288)
    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_accum = taddr;

    // ---- Role: epilogue ----
    if (warp <= 3) {
        { // epilogue_main
            asm volatile("barrier.sync 6, 192;" ::: "memory");
            asm volatile("setmaxnreg.dec.sync.aligned.u32 168;");
            asm volatile("griddepcontrol.wait;" ::: "memory");
            int num_cluster_tiles = gridDim.x / 2 * gridDim.y;
            unsigned int acc_stage = 0;
            unsigned int work_stage = 0;
            unsigned int m_tile = blockIdx.x;
            unsigned int n_tile = blockIdx.y;
            const int warp_0 = warp;
            const int lane_1 = lane;
            int base_feature = warp_0 * 32 + lane_1 / 4 * 4;
            int base_token = lane_1 % 4 * 2;
            float converted[4];
            unsigned int packed[2];
            unsigned long long packed_word = 0;
            unsigned int _phase_mma_full = 0;
            unsigned int _phase_work_full = 0;
            #pragma unroll 1
            for (unsigned int _tile_iter = 0; _tile_iter < num_cluster_tiles; _tile_iter++) {
                if (n_tile < (unsigned int)num_non_exiting_ctas[0]) {
                    int mn_limit = cta_idx_y_to_mn_limit[n_tile];
                    int padding_rows = (64 - mn_limit % 64) % 64;
                    mbarrier_wait(mma_full_addr + (acc_stage) * 8, _phase_mma_full);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    float _tmem_load_0[32];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[15])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[16])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[17])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[18])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[19])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[20])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[21])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[22])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[23])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[24])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[25])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[26])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[27])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[28])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[29])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[30])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[31]))
                        : "r"(taddr + acc_stage * 64));
                    float _tmem_load_1[32];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[15])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[16])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[17])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[18])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[19])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[20])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[21])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[22])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[23])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[24])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[25])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[26])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[27])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[28])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[29])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[30])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[31]))
                        : "r"(taddr + 1048576 + acc_stage * 64));
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    asm volatile("cp.async.bulk.wait_group.read 0;");
                    asm volatile("barrier.sync 4, 128;" ::: "memory");
                    int token = base_token;
                    int frag_base = 0;
                    converted[0] = _tmem_load_0[frag_base];
                    converted[1] = _tmem_load_0[frag_base + 2];
                    converted[2] = _tmem_load_1[frag_base];
                    converted[3] = _tmem_load_1[frag_base + 2];
                    #pragma unroll
                    for (int _lp = 0; _lp < 2; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                        packed[_lp] = *(uint32_t*)&_bf2;
                    }
                    packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                    int swizzle_feature = token % 8 * 8;
                    if (base_feature < 64) {
                        epi_staging_u64[(token * 64 + (base_feature ^ swizzle_feature)) / 4] = packed_word;
                    } else {
                        epi_staging_u64[(4096 + token * 64 + (base_feature - 64 ^ swizzle_feature)) / 4] = packed_word;
                    }
                    int token_0 = base_token + 1;
                    int frag_base_1 = 1;
                    converted[0] = _tmem_load_0[frag_base_1];
                    converted[1] = _tmem_load_0[frag_base_1 + 2];
                    converted[2] = _tmem_load_1[frag_base_1];
                    converted[3] = _tmem_load_1[frag_base_1 + 2];
                    #pragma unroll
                    for (int _lp = 0; _lp < 2; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                        packed[_lp] = *(uint32_t*)&_bf2;
                    }
                    packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                    int swizzle_feature_2 = token_0 % 8 * 8;
                    if (base_feature < 64) {
                        epi_staging_u64[(token_0 * 64 + (base_feature ^ swizzle_feature_2)) / 4] = packed_word;
                    } else {
                        epi_staging_u64[(4096 + token_0 * 64 + (base_feature - 64 ^ swizzle_feature_2)) / 4] = packed_word;
                    }
                    int token_3 = base_token + 8;
                    int frag_base_4 = 4;
                    converted[0] = _tmem_load_0[frag_base_4];
                    converted[1] = _tmem_load_0[frag_base_4 + 2];
                    converted[2] = _tmem_load_1[frag_base_4];
                    converted[3] = _tmem_load_1[frag_base_4 + 2];
                    #pragma unroll
                    for (int _lp = 0; _lp < 2; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                        packed[_lp] = *(uint32_t*)&_bf2;
                    }
                    packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                    int swizzle_feature_5 = token_3 % 8 * 8;
                    if (base_feature < 64) {
                        epi_staging_u64[(token_3 * 64 + (base_feature ^ swizzle_feature_5)) / 4] = packed_word;
                    } else {
                        epi_staging_u64[(4096 + token_3 * 64 + (base_feature - 64 ^ swizzle_feature_5)) / 4] = packed_word;
                    }
                    int token_6 = base_token + 8 + 1;
                    int frag_base_7 = 5;
                    converted[0] = _tmem_load_0[frag_base_7];
                    converted[1] = _tmem_load_0[frag_base_7 + 2];
                    converted[2] = _tmem_load_1[frag_base_7];
                    converted[3] = _tmem_load_1[frag_base_7 + 2];
                    #pragma unroll
                    for (int _lp = 0; _lp < 2; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                        packed[_lp] = *(uint32_t*)&_bf2;
                    }
                    packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                    int swizzle_feature_8 = token_6 % 8 * 8;
                    if (base_feature < 64) {
                        epi_staging_u64[(token_6 * 64 + (base_feature ^ swizzle_feature_8)) / 4] = packed_word;
                    } else {
                        epi_staging_u64[(4096 + token_6 * 64 + (base_feature - 64 ^ swizzle_feature_8)) / 4] = packed_word;
                    }
                    int token_9 = base_token + 16;
                    int frag_base_10 = 8;
                    converted[0] = _tmem_load_0[frag_base_10];
                    converted[1] = _tmem_load_0[frag_base_10 + 2];
                    converted[2] = _tmem_load_1[frag_base_10];
                    converted[3] = _tmem_load_1[frag_base_10 + 2];
                    #pragma unroll
                    for (int _lp = 0; _lp < 2; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                        packed[_lp] = *(uint32_t*)&_bf2;
                    }
                    packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                    int swizzle_feature_11 = token_9 % 8 * 8;
                    if (base_feature < 64) {
                        epi_staging_u64[(token_9 * 64 + (base_feature ^ swizzle_feature_11)) / 4] = packed_word;
                    } else {
                        epi_staging_u64[(4096 + token_9 * 64 + (base_feature - 64 ^ swizzle_feature_11)) / 4] = packed_word;
                    }
                    int token_12 = base_token + 16 + 1;
                    int frag_base_13 = 9;
                    converted[0] = _tmem_load_0[frag_base_13];
                    converted[1] = _tmem_load_0[frag_base_13 + 2];
                    converted[2] = _tmem_load_1[frag_base_13];
                    converted[3] = _tmem_load_1[frag_base_13 + 2];
                    #pragma unroll
                    for (int _lp = 0; _lp < 2; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                        packed[_lp] = *(uint32_t*)&_bf2;
                    }
                    packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                    int swizzle_feature_14 = token_12 % 8 * 8;
                    if (base_feature < 64) {
                        epi_staging_u64[(token_12 * 64 + (base_feature ^ swizzle_feature_14)) / 4] = packed_word;
                    } else {
                        epi_staging_u64[(4096 + token_12 * 64 + (base_feature - 64 ^ swizzle_feature_14)) / 4] = packed_word;
                    }
                    int token_15 = base_token + 24;
                    int frag_base_16 = 12;
                    converted[0] = _tmem_load_0[frag_base_16];
                    converted[1] = _tmem_load_0[frag_base_16 + 2];
                    converted[2] = _tmem_load_1[frag_base_16];
                    converted[3] = _tmem_load_1[frag_base_16 + 2];
                    #pragma unroll
                    for (int _lp = 0; _lp < 2; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                        packed[_lp] = *(uint32_t*)&_bf2;
                    }
                    packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                    int swizzle_feature_17 = token_15 % 8 * 8;
                    if (base_feature < 64) {
                        epi_staging_u64[(token_15 * 64 + (base_feature ^ swizzle_feature_17)) / 4] = packed_word;
                    } else {
                        epi_staging_u64[(4096 + token_15 * 64 + (base_feature - 64 ^ swizzle_feature_17)) / 4] = packed_word;
                    }
                    int token_18 = base_token + 24 + 1;
                    int frag_base_19 = 13;
                    converted[0] = _tmem_load_0[frag_base_19];
                    converted[1] = _tmem_load_0[frag_base_19 + 2];
                    converted[2] = _tmem_load_1[frag_base_19];
                    converted[3] = _tmem_load_1[frag_base_19 + 2];
                    #pragma unroll
                    for (int _lp = 0; _lp < 2; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                        packed[_lp] = *(uint32_t*)&_bf2;
                    }
                    packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                    int swizzle_feature_20 = token_18 % 8 * 8;
                    if (base_feature < 64) {
                        epi_staging_u64[(token_18 * 64 + (base_feature ^ swizzle_feature_20)) / 4] = packed_word;
                    } else {
                        epi_staging_u64[(4096 + token_18 * 64 + (base_feature - 64 ^ swizzle_feature_20)) / 4] = packed_word;
                    }
                    int token_21 = base_token + 32;
                    int frag_base_22 = 16;
                    converted[0] = _tmem_load_0[frag_base_22];
                    converted[1] = _tmem_load_0[frag_base_22 + 2];
                    converted[2] = _tmem_load_1[frag_base_22];
                    converted[3] = _tmem_load_1[frag_base_22 + 2];
                    #pragma unroll
                    for (int _lp = 0; _lp < 2; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                        packed[_lp] = *(uint32_t*)&_bf2;
                    }
                    packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                    int swizzle_feature_23 = token_21 % 8 * 8;
                    if (base_feature < 64) {
                        epi_staging_u64[(token_21 * 64 + (base_feature ^ swizzle_feature_23)) / 4] = packed_word;
                    } else {
                        epi_staging_u64[(4096 + token_21 * 64 + (base_feature - 64 ^ swizzle_feature_23)) / 4] = packed_word;
                    }
                    int token_24 = base_token + 32 + 1;
                    int frag_base_25 = 17;
                    converted[0] = _tmem_load_0[frag_base_25];
                    converted[1] = _tmem_load_0[frag_base_25 + 2];
                    converted[2] = _tmem_load_1[frag_base_25];
                    converted[3] = _tmem_load_1[frag_base_25 + 2];
                    #pragma unroll
                    for (int _lp = 0; _lp < 2; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                        packed[_lp] = *(uint32_t*)&_bf2;
                    }
                    packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                    int swizzle_feature_26 = token_24 % 8 * 8;
                    if (base_feature < 64) {
                        epi_staging_u64[(token_24 * 64 + (base_feature ^ swizzle_feature_26)) / 4] = packed_word;
                    } else {
                        epi_staging_u64[(4096 + token_24 * 64 + (base_feature - 64 ^ swizzle_feature_26)) / 4] = packed_word;
                    }
                    int token_27 = base_token + 40;
                    int frag_base_28 = 20;
                    converted[0] = _tmem_load_0[frag_base_28];
                    converted[1] = _tmem_load_0[frag_base_28 + 2];
                    converted[2] = _tmem_load_1[frag_base_28];
                    converted[3] = _tmem_load_1[frag_base_28 + 2];
                    #pragma unroll
                    for (int _lp = 0; _lp < 2; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                        packed[_lp] = *(uint32_t*)&_bf2;
                    }
                    packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                    int swizzle_feature_29 = token_27 % 8 * 8;
                    if (base_feature < 64) {
                        epi_staging_u64[(token_27 * 64 + (base_feature ^ swizzle_feature_29)) / 4] = packed_word;
                    } else {
                        epi_staging_u64[(4096 + token_27 * 64 + (base_feature - 64 ^ swizzle_feature_29)) / 4] = packed_word;
                    }
                    int token_30 = base_token + 40 + 1;
                    int frag_base_31 = 21;
                    converted[0] = _tmem_load_0[frag_base_31];
                    converted[1] = _tmem_load_0[frag_base_31 + 2];
                    converted[2] = _tmem_load_1[frag_base_31];
                    converted[3] = _tmem_load_1[frag_base_31 + 2];
                    #pragma unroll
                    for (int _lp = 0; _lp < 2; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                        packed[_lp] = *(uint32_t*)&_bf2;
                    }
                    packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                    int swizzle_feature_32 = token_30 % 8 * 8;
                    if (base_feature < 64) {
                        epi_staging_u64[(token_30 * 64 + (base_feature ^ swizzle_feature_32)) / 4] = packed_word;
                    } else {
                        epi_staging_u64[(4096 + token_30 * 64 + (base_feature - 64 ^ swizzle_feature_32)) / 4] = packed_word;
                    }
                    int token_33 = base_token + 48;
                    int frag_base_34 = 24;
                    converted[0] = _tmem_load_0[frag_base_34];
                    converted[1] = _tmem_load_0[frag_base_34 + 2];
                    converted[2] = _tmem_load_1[frag_base_34];
                    converted[3] = _tmem_load_1[frag_base_34 + 2];
                    #pragma unroll
                    for (int _lp = 0; _lp < 2; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                        packed[_lp] = *(uint32_t*)&_bf2;
                    }
                    packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                    int swizzle_feature_35 = token_33 % 8 * 8;
                    if (base_feature < 64) {
                        epi_staging_u64[(token_33 * 64 + (base_feature ^ swizzle_feature_35)) / 4] = packed_word;
                    } else {
                        epi_staging_u64[(4096 + token_33 * 64 + (base_feature - 64 ^ swizzle_feature_35)) / 4] = packed_word;
                    }
                    int token_36 = base_token + 48 + 1;
                    int frag_base_37 = 25;
                    converted[0] = _tmem_load_0[frag_base_37];
                    converted[1] = _tmem_load_0[frag_base_37 + 2];
                    converted[2] = _tmem_load_1[frag_base_37];
                    converted[3] = _tmem_load_1[frag_base_37 + 2];
                    #pragma unroll
                    for (int _lp = 0; _lp < 2; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                        packed[_lp] = *(uint32_t*)&_bf2;
                    }
                    packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                    int swizzle_feature_38 = token_36 % 8 * 8;
                    if (base_feature < 64) {
                        epi_staging_u64[(token_36 * 64 + (base_feature ^ swizzle_feature_38)) / 4] = packed_word;
                    } else {
                        epi_staging_u64[(4096 + token_36 * 64 + (base_feature - 64 ^ swizzle_feature_38)) / 4] = packed_word;
                    }
                    int token_39 = base_token + 56;
                    int frag_base_40 = 28;
                    converted[0] = _tmem_load_0[frag_base_40];
                    converted[1] = _tmem_load_0[frag_base_40 + 2];
                    converted[2] = _tmem_load_1[frag_base_40];
                    converted[3] = _tmem_load_1[frag_base_40 + 2];
                    #pragma unroll
                    for (int _lp = 0; _lp < 2; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                        packed[_lp] = *(uint32_t*)&_bf2;
                    }
                    packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                    int swizzle_feature_41 = token_39 % 8 * 8;
                    if (base_feature < 64) {
                        epi_staging_u64[(token_39 * 64 + (base_feature ^ swizzle_feature_41)) / 4] = packed_word;
                    } else {
                        epi_staging_u64[(4096 + token_39 * 64 + (base_feature - 64 ^ swizzle_feature_41)) / 4] = packed_word;
                    }
                    int token_42 = base_token + 56 + 1;
                    int frag_base_43 = 29;
                    converted[0] = _tmem_load_0[frag_base_43];
                    converted[1] = _tmem_load_0[frag_base_43 + 2];
                    converted[2] = _tmem_load_1[frag_base_43];
                    converted[3] = _tmem_load_1[frag_base_43 + 2];
                    #pragma unroll
                    for (int _lp = 0; _lp < 2; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                        packed[_lp] = *(uint32_t*)&_bf2;
                    }
                    packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                    int swizzle_feature_44 = token_42 % 8 * 8;
                    if (base_feature < 64) {
                        epi_staging_u64[(token_42 * 64 + (base_feature ^ swizzle_feature_44)) / 4] = packed_word;
                    } else {
                        epi_staging_u64[(4096 + token_42 * 64 + (base_feature - 64 ^ swizzle_feature_44)) / 4] = packed_word;
                    }
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    asm volatile("barrier.sync 4, 128;" ::: "memory");
                    if (warp == 0) {
                        if (elect_sync()) {
                            tma_store_4d(C, m_tile * 128, padding_rows, 1073741824, n_tile * 64 - (unsigned int)padding_rows + 1073741824, epi_staging_addr);
                            tma_store_4d(C, m_tile * 128 + 64, padding_rows, 1073741824, n_tile * 64 - (unsigned int)padding_rows + 1073741824, epi_staging_addr + 8192);
                        }
                    }
                    asm volatile("cp.async.bulk.commit_group;");
                    asm volatile("barrier.sync 4, 128;" ::: "memory");
                    asm volatile(
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                        :: "r"((mma_free_addr + (acc_stage) * 8) & 0xFEFFFFFF) : "memory");
                    acc_stage += 1;
                    if (acc_stage == 2) { acc_stage = 0; _phase_mma_full ^= 1; }
                    acc_stage += 1;
                    if (acc_stage == 2) { acc_stage = 0; _phase_mma_full ^= 1; }
                }
                mbarrier_wait(work_full_addr + (work_stage) * 8, _phase_work_full);
                uint32_t _clc_valid_3 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "selp.u32 %0, 1, 0, p1;\n\t"
                    "}\n"
                    : "=r"(_clc_valid_3)
                    : "r"(work_response_addr + work_stage * 16)
                    : "memory");
                uint32_t _clc_ctaid_9 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "=r"(_clc_ctaid_9)
                    : "r"(work_response_addr + work_stage * 16)
                    : "memory");
                uint32_t _clc_ctaid_10 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.get_first_ctaid::y.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "=r"(_clc_ctaid_10)
                    : "r"(work_response_addr + work_stage * 16)
                    : "memory");
                uint32_t _clc_ctaid_11 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.get_first_ctaid::z.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "=r"(_clc_ctaid_11)
                    : "r"(work_response_addr + work_stage * 16)
                    : "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile(
                    "{\n\t"
                    ".reg .b32 remAddr32;\n\t"
                    "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                    "}"
                    :: "r"(work_empty_addr + work_stage * 8), "r"(0) : "memory");
                work_stage += 1;
                if (work_stage == 3) { work_stage = 0; _phase_work_full ^= 1; }
                if (_clc_valid_3 == 0) {
                    break;
                }
                m_tile = _clc_ctaid_9 + (unsigned int)cta_rank;
                n_tile = _clc_ctaid_10;
            }
            asm volatile("barrier.sync 7, 128;" ::: "memory");
            if (warp == 0) {
                asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
                asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
                int _tmem_dealloc_addr = *((volatile int*)tmem_addr_storage);
                asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;" :: "r"(_tmem_dealloc_addr), "r"(128));
            }
        }
    }
    // ---- Role: mma ----
    if (warp == 4) {
        { // mma_main
            asm volatile("barrier.sync 6, 192;" ::: "memory");
            asm volatile("setmaxnreg.dec.sync.aligned.u32 96;");
            unsigned int _phase_mma_free = 1;
            unsigned int _phase_a_full = 0;
            unsigned int _phase_b_full = 0;
            unsigned int _phase_work_full_1 = 0;
            if (cta_rank == 0) {
                int num_k_tiles = (K + 128 - 1) / 128;
                int num_cluster_tiles_1 = gridDim.x / 2 * gridDim.y;
                unsigned int stage = 0;
                unsigned int acc_stage_1 = 0;
                unsigned int work_stage_1 = 0;
                unsigned int m_tile_1 = blockIdx.x;
                unsigned int n_tile_1 = blockIdx.y;
                #pragma unroll 1
                for (unsigned int _tile_iter_1 = 0; _tile_iter_1 < num_cluster_tiles_1; _tile_iter_1++) {
                    if (n_tile_1 < (unsigned int)num_non_exiting_ctas[0]) {
                        mbarrier_wait(mma_free_addr + (acc_stage_1) * 8, _phase_mma_free);
                        #pragma unroll 1
                        for (int k_pair = 0; k_pair < num_k_tiles; k_pair += 2) {
                            int iter_k = k_pair;
                            mbarrier_wait(a_full_addr + (stage) * 8, _phase_a_full);
                            mbarrier_wait(b_full_addr + (stage) * 8, _phase_b_full);
                            asm volatile("tcgen05.fence::after_thread_sync;");
                            int _mma_a_lo_0 = (((smem_a_addr) >> 4) & 0x3FFF) + (stage) * 2048;
                            int _mma_b_lo_0 = (((smem_b_addr) >> 4) & 0x3FFF) + (stage) * 512;
                            asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id, m0, m1, m2, m3, m4, m5, m6, m7;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    "mov.b32 m0, 0; mov.b32 m1, 0; mov.b32 m2, 0; mov.b32 m3, 0;\n\tmov.b32 m4, 0; mov.b32 m5, 0; mov.b32 m6, 0; mov.b32 m7, 0;\n\t"
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 269485200;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 1018;\n\t"
                    "add.u32 blo, blo, 250;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_0), "r"(_mma_b_lo_0), "r"((tmem_accum + (acc_stage_1 * 64))), "r"(((((1) ? ((k_pair == 0) ? 1 : 0) : 0)) ? 0 : 1)));
                            elect_commit_cg2_multicast(a_free_addr + (stage) * 8, (uint16_t)(3));
                            elect_commit_cg2_multicast(b_free_addr + (stage) * 8, (uint16_t)(3));
                            stage += 1;
                            if (stage == 5) { stage = 0; _phase_a_full ^= 1; _phase_b_full ^= 1; }
                            int iter_k_0 = k_pair + 1;
                            mbarrier_wait(a_full_addr + (stage) * 8, _phase_a_full);
                            mbarrier_wait(b_full_addr + (stage) * 8, _phase_b_full);
                            asm volatile("tcgen05.fence::after_thread_sync;");
                            int _mma_a_lo_1 = (((smem_a_addr) >> 4) & 0x3FFF) + (stage) * 2048;
                            int _mma_b_lo_1 = (((smem_b_addr) >> 4) & 0x3FFF) + (stage) * 512;
                            asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id, m0, m1, m2, m3, m4, m5, m6, m7;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    "mov.b32 m0, 0; mov.b32 m1, 0; mov.b32 m2, 0; mov.b32 m3, 0;\n\tmov.b32 m4, 0; mov.b32 m5, 0; mov.b32 m6, 0; mov.b32 m7, 0;\n\t"
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 269485200;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 1018;\n\t"
                    "add.u32 blo, blo, 250;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_1), "r"(_mma_b_lo_1), "r"((tmem_accum + (acc_stage_1 * 64))), "r"(((((0) ? ((k_pair == 0) ? 1 : 0) : 0)) ? 0 : 1)));
                            elect_commit_cg2_multicast(a_free_addr + (stage) * 8, (uint16_t)(3));
                            elect_commit_cg2_multicast(b_free_addr + (stage) * 8, (uint16_t)(3));
                            stage += 1;
                            if (stage == 5) { stage = 0; _phase_a_full ^= 1; _phase_b_full ^= 1; }
                        }
                        elect_commit_cg2_multicast(mma_full_addr + (acc_stage_1) * 8, (uint16_t)(3));
                        acc_stage_1 += 1;
                        if (acc_stage_1 == 2) { acc_stage_1 = 0; _phase_mma_free ^= 1; }
                        acc_stage_1 += 1;
                        if (acc_stage_1 == 2) { acc_stage_1 = 0; _phase_mma_free ^= 1; }
                    }
                    mbarrier_wait(work_full_addr + (work_stage_1) * 8, _phase_work_full_1);
                    uint32_t _clc_valid_2 = 0;
                    asm volatile(
                        "{\n\t"
                        ".reg .pred p1;\n\t"
                        ".reg .b128 clc_r;\n\t"
                        "ld.shared.b128 clc_r, [%1];\n\t"
                        "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                        "selp.u32 %0, 1, 0, p1;\n\t"
                        "}\n"
                        : "=r"(_clc_valid_2)
                        : "r"(work_response_addr + work_stage_1 * 16)
                        : "memory");
                    uint32_t _clc_ctaid_6 = 0;
                    asm volatile(
                        "{\n\t"
                        ".reg .b128 clc_r;\n\t"
                        "ld.shared.b128 clc_r, [%1];\n\t"
                        "clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                        "}\n"
                        : "=r"(_clc_ctaid_6)
                        : "r"(work_response_addr + work_stage_1 * 16)
                        : "memory");
                    uint32_t _clc_ctaid_7 = 0;
                    asm volatile(
                        "{\n\t"
                        ".reg .b128 clc_r;\n\t"
                        "ld.shared.b128 clc_r, [%1];\n\t"
                        "clusterlaunchcontrol.query_cancel.get_first_ctaid::y.b32.b128 %0, clc_r;\n\t"
                        "}\n"
                        : "=r"(_clc_ctaid_7)
                        : "r"(work_response_addr + work_stage_1 * 16)
                        : "memory");
                    uint32_t _clc_ctaid_8 = 0;
                    asm volatile(
                        "{\n\t"
                        ".reg .b128 clc_r;\n\t"
                        "ld.shared.b128 clc_r, [%1];\n\t"
                        "clusterlaunchcontrol.query_cancel.get_first_ctaid::z.b32.b128 %0, clc_r;\n\t"
                        "}\n"
                        : "=r"(_clc_ctaid_8)
                        : "r"(work_response_addr + work_stage_1 * 16)
                        : "memory");
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    asm volatile(
                        "{\n\t"
                        ".reg .b32 remAddr32;\n\t"
                        "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                        "}"
                        :: "r"(work_empty_addr + work_stage_1 * 8), "r"(0) : "memory");
                    work_stage_1 += 1;
                    if (work_stage_1 == 3) { work_stage_1 = 0; _phase_work_full_1 ^= 1; }
                    if (_clc_valid_2 == 0) {
                        break;
                    }
                    m_tile_1 = _clc_ctaid_6 + (unsigned int)cta_rank;
                    n_tile_1 = _clc_ctaid_7;
                }
            }
        }
    }
    // ---- Role: load_a ----
    if (warp == 5) {
        { // load_a_main
            asm volatile("setmaxnreg.dec.sync.aligned.u32 96;");
            int num_k_tiles_1 = (K + 128 - 1) / 128;
            int num_cluster_tiles_2 = gridDim.x / 2 * gridDim.y;
            unsigned int stage_1 = 0;
            unsigned int work_stage_2 = 0;
            unsigned int throttle_stage = 0;
            unsigned int m_tile_2 = blockIdx.x;
            unsigned int n_tile_2 = blockIdx.y;
            unsigned int cta_mask = 1 << cta_rank;
            unsigned int _phase_throttle_empty = 1;
            unsigned int _phase_a_free = 1;
            unsigned int _phase_work_full_2 = 0;
            #pragma unroll 1
            for (unsigned int _tile_iter_2 = 0; _tile_iter_2 < num_cluster_tiles_2; _tile_iter_2++) {
                if (n_tile_2 < (unsigned int)num_non_exiting_ctas[0]) {
                    int expert = cta_idx_y_to_batch_idx[n_tile_2];
                    if (cta_rank == 0) {
                        mbarrier_wait(throttle_empty_addr + (throttle_stage) * 8, _phase_throttle_empty);
                        mbarrier_arrive(throttle_full_addr + (throttle_stage) * 8);
                        throttle_stage += 1;
                        if (throttle_stage == 3) { throttle_stage = 0; _phase_throttle_empty ^= 1; }
                    }
                    #pragma unroll 1
                    for (int iter_k_1 = 0; iter_k_1 < num_k_tiles_1; iter_k_1++) {
                        mbarrier_wait(a_free_addr + (stage_1) * 8, _phase_a_free);
                        if (elect_sync()) {
                            asm volatile(
                                "cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                " [%0], [%1, {%2, %3, %4, %5}], [%6], %7;"
                                :: "r"(smem_a_addr + stage_1 * 32768), "l"(A), "r"(0), "r"(m_tile_2 * 128), "r"(iter_k_1 * 2), "r"(expert),
                                   "r"(((a_full_addr + (stage_1) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(cta_mask)) : "memory");
                            asm volatile(
                                "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                                :: "r"((a_full_addr + (stage_1) * 8) & 0xFEFFFFFF), "r"((uint32_t)(32768)) : "memory");
                        }
                        stage_1 += 1;
                        if (stage_1 == 5) { stage_1 = 0; _phase_a_free ^= 1; }
                    }
                }
                mbarrier_wait(work_full_addr + (work_stage_2) * 8, _phase_work_full_2);
                uint32_t _clc_valid_0 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "selp.u32 %0, 1, 0, p1;\n\t"
                    "}\n"
                    : "=r"(_clc_valid_0)
                    : "r"(work_response_addr + work_stage_2 * 16)
                    : "memory");
                uint32_t _clc_ctaid_0 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "=r"(_clc_ctaid_0)
                    : "r"(work_response_addr + work_stage_2 * 16)
                    : "memory");
                uint32_t _clc_ctaid_1 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.get_first_ctaid::y.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "=r"(_clc_ctaid_1)
                    : "r"(work_response_addr + work_stage_2 * 16)
                    : "memory");
                uint32_t _clc_ctaid_2 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.get_first_ctaid::z.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "=r"(_clc_ctaid_2)
                    : "r"(work_response_addr + work_stage_2 * 16)
                    : "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile(
                    "{\n\t"
                    ".reg .b32 remAddr32;\n\t"
                    "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                    "}"
                    :: "r"(work_empty_addr + work_stage_2 * 8), "r"(0) : "memory");
                work_stage_2 += 1;
                if (work_stage_2 == 3) { work_stage_2 = 0; _phase_work_full_2 ^= 1; }
                if (_clc_valid_0 == 0) {
                    break;
                }
                m_tile_2 = _clc_ctaid_0 + (unsigned int)cta_rank;
                n_tile_2 = _clc_ctaid_1;
            }
        }
    }
    // ---- Role: load_b ----
    if (warp == 6) {
        { // load_b_main
            asm volatile("setmaxnreg.dec.sync.aligned.u32 96;");
            asm volatile("griddepcontrol.wait;" ::: "memory");
            int num_k_tiles_2 = (K + 128 - 1) / 128;
            int num_cluster_tiles_3 = gridDim.x / 2 * gridDim.y;
            unsigned int stage_2 = 0;
            unsigned int work_stage_3 = 0;
            unsigned int m_tile_3 = blockIdx.x;
            unsigned int n_tile_3 = blockIdx.y;
            unsigned int cta_mask_1 = 1 << cta_rank;
            unsigned int _phase_b_free = 1;
            unsigned int _phase_work_full_3 = 0;
            #pragma unroll 1
            for (unsigned int _tile_iter_3 = 0; _tile_iter_3 < num_cluster_tiles_3; _tile_iter_3++) {
                if (n_tile_3 < (unsigned int)num_non_exiting_ctas[0]) {
                    int mn_limit_1 = cta_idx_y_to_mn_limit[n_tile_3];
                    int padding_rows_1 = (64 - mn_limit_1 % 64) % 64;
                    #pragma unroll 1
                    for (int iter_k_2 = 0; iter_k_2 < num_k_tiles_2; iter_k_2++) {
                        mbarrier_wait(b_free_addr + (stage_2) * 8, _phase_b_free);
                        if (elect_sync()) {
                            int dst = smem_b_addr + stage_2 * 8192;
                            asm volatile(
                                "cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                " [%0], [%1, {%2, %3, %4, %5}], [%6], %7;"
                                :: "r"(dst), "l"(B), "r"(iter_k_2 * 128), "r"(cta_rank * 32 + padding_rows_1), "r"(1073741824), "r"(n_tile_3 * 64 - (unsigned int)padding_rows_1 + 1073741824),
                                   "r"(((b_full_addr + (stage_2) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(cta_mask_1)) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                " [%0], [%1, {%2, %3, %4, %5}], [%6], %7;"
                                :: "r"(dst + 4096), "l"(B), "r"(iter_k_2 * 128 + 64), "r"(cta_rank * 32 + padding_rows_1), "r"(1073741824), "r"(n_tile_3 * 64 - (unsigned int)padding_rows_1 + 1073741824),
                                   "r"(((b_full_addr + (stage_2) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(cta_mask_1)) : "memory");
                            asm volatile(
                                "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                                :: "r"((b_full_addr + (stage_2) * 8) & 0xFEFFFFFF), "r"((uint32_t)(8192)) : "memory");
                        }
                        stage_2 += 1;
                        if (stage_2 == 5) { stage_2 = 0; _phase_b_free ^= 1; }
                    }
                }
                mbarrier_wait(work_full_addr + (work_stage_3) * 8, _phase_work_full_3);
                uint32_t _clc_valid_1 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "selp.u32 %0, 1, 0, p1;\n\t"
                    "}\n"
                    : "=r"(_clc_valid_1)
                    : "r"(work_response_addr + work_stage_3 * 16)
                    : "memory");
                uint32_t _clc_ctaid_3 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "=r"(_clc_ctaid_3)
                    : "r"(work_response_addr + work_stage_3 * 16)
                    : "memory");
                uint32_t _clc_ctaid_4 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.get_first_ctaid::y.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "=r"(_clc_ctaid_4)
                    : "r"(work_response_addr + work_stage_3 * 16)
                    : "memory");
                uint32_t _clc_ctaid_5 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.get_first_ctaid::z.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "=r"(_clc_ctaid_5)
                    : "r"(work_response_addr + work_stage_3 * 16)
                    : "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile(
                    "{\n\t"
                    ".reg .b32 remAddr32;\n\t"
                    "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                    "}"
                    :: "r"(work_empty_addr + work_stage_3 * 8), "r"(0) : "memory");
                work_stage_3 += 1;
                if (work_stage_3 == 3) { work_stage_3 = 0; _phase_work_full_3 ^= 1; }
                if (_clc_valid_1 == 0) {
                    break;
                }
                m_tile_3 = _clc_ctaid_3 + (unsigned int)cta_rank;
                n_tile_3 = _clc_ctaid_4;
            }
            asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
        }
    }
    // ---- Role: work_id ----
    if (warp == 7) {
        { // work_id_main
            asm volatile("barrier.sync 6, 192;" ::: "memory");
            asm volatile("setmaxnreg.dec.sync.aligned.u32 96;");
            asm volatile("griddepcontrol.wait;" ::: "memory");
            unsigned int _phase_throttle_full = 0;
            unsigned int _phase_drain_full = 0;
            unsigned int _phase_work_empty = 1;
            unsigned int _phase_work_full_4 = 0;
            if (cta_rank == 0) {
                int num_cluster_tiles_4 = gridDim.x / 2 * gridDim.y;
                int drain_rounds = (num_cluster_tiles_4 + 4 - 1) / 4 + 1;
                unsigned int work_stage_4 = 0;
                unsigned int throttle_stage_1 = 0;
                unsigned int drain_stage = 0;
                unsigned int m_tile_4 = blockIdx.x;
                unsigned int n_tile_4 = blockIdx.y;
                #pragma unroll 1
                for (unsigned int _tile_iter_4 = 0; _tile_iter_4 < num_cluster_tiles_4; _tile_iter_4++) {
                    if (n_tile_4 < (unsigned int)num_non_exiting_ctas[0]) {
                        mbarrier_wait(throttle_full_addr + (throttle_stage_1) * 8, _phase_throttle_full);
                        mbarrier_arrive(throttle_empty_addr + (throttle_stage_1) * 8);
                        throttle_stage_1 += 1;
                        if (throttle_stage_1 == 3) { throttle_stage_1 = 0; _phase_throttle_full ^= 1; }
                    } else {
                        #pragma unroll 1
                        for (unsigned int _drain_iter = 0; _drain_iter < drain_rounds; _drain_iter++) {
                            if (elect_sync()) {
                                mbarrier_arrive_expect_tx(drain_full_addr + (drain_stage) * 8, 64);
                                asm volatile(
                                    "fence.proxy.async.shared::cta;\n\t"
                                    "clusterlaunchcontrol.try_cancel.async.shared::cta"
                                        ".mbarrier::complete_tx::bytes.b128"
                                        " [%0], [%1];"
                                    :: "r"(fast_drain_response_addr), "r"(drain_full_addr + drain_stage * 8)
                                    : "memory");
                                asm volatile(
                                    "fence.proxy.async.shared::cta;\n\t"
                                    "clusterlaunchcontrol.try_cancel.async.shared::cta"
                                        ".mbarrier::complete_tx::bytes.b128"
                                        " [%0], [%1];"
                                    :: "r"(fast_drain_response_addr + 16), "r"(drain_full_addr + drain_stage * 8)
                                    : "memory");
                                asm volatile(
                                    "fence.proxy.async.shared::cta;\n\t"
                                    "clusterlaunchcontrol.try_cancel.async.shared::cta"
                                        ".mbarrier::complete_tx::bytes.b128"
                                        " [%0], [%1];"
                                    :: "r"(fast_drain_response_addr + 32), "r"(drain_full_addr + drain_stage * 8)
                                    : "memory");
                                asm volatile(
                                    "fence.proxy.async.shared::cta;\n\t"
                                    "clusterlaunchcontrol.try_cancel.async.shared::cta"
                                        ".mbarrier::complete_tx::bytes.b128"
                                        " [%0], [%1];"
                                    :: "r"(fast_drain_response_addr + 48), "r"(drain_full_addr + drain_stage * 8)
                                    : "memory");
                            }
                            mbarrier_wait(drain_full_addr + (drain_stage) * 8, _phase_drain_full);
                            unsigned int canceled = 0;
                            uint32_t _clc_valid_4 = 0;
                            asm volatile(
                                "{\n\t"
                                ".reg .pred p1;\n\t"
                                ".reg .b128 clc_r;\n\t"
                                "ld.shared.b128 clc_r, [%1];\n\t"
                                "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                                "selp.u32 %0, 1, 0, p1;\n\t"
                                "}\n"
                                : "=r"(_clc_valid_4)
                                : "r"(fast_drain_response_addr)
                                : "memory");
                            canceled += _clc_valid_4;
                            uint32_t _clc_valid_5 = 0;
                            asm volatile(
                                "{\n\t"
                                ".reg .pred p1;\n\t"
                                ".reg .b128 clc_r;\n\t"
                                "ld.shared.b128 clc_r, [%1];\n\t"
                                "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                                "selp.u32 %0, 1, 0, p1;\n\t"
                                "}\n"
                                : "=r"(_clc_valid_5)
                                : "r"(fast_drain_response_addr + 16)
                                : "memory");
                            canceled += _clc_valid_5;
                            uint32_t _clc_valid_6 = 0;
                            asm volatile(
                                "{\n\t"
                                ".reg .pred p1;\n\t"
                                ".reg .b128 clc_r;\n\t"
                                "ld.shared.b128 clc_r, [%1];\n\t"
                                "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                                "selp.u32 %0, 1, 0, p1;\n\t"
                                "}\n"
                                : "=r"(_clc_valid_6)
                                : "r"(fast_drain_response_addr + 32)
                                : "memory");
                            canceled += _clc_valid_6;
                            uint32_t _clc_valid_7 = 0;
                            asm volatile(
                                "{\n\t"
                                ".reg .pred p1;\n\t"
                                ".reg .b128 clc_r;\n\t"
                                "ld.shared.b128 clc_r, [%1];\n\t"
                                "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                                "selp.u32 %0, 1, 0, p1;\n\t"
                                "}\n"
                                : "=r"(_clc_valid_7)
                                : "r"(fast_drain_response_addr + 48)
                                : "memory");
                            canceled += _clc_valid_7;
                            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                            drain_stage += 1;
                            if (drain_stage == 1) { drain_stage = 0; _phase_drain_full ^= 1; }
                            if (canceled == 0) {
                                break;
                            }
                        }
                    }
                    mbarrier_wait_cluster(work_empty_addr + (work_stage_4) * 8, _phase_work_empty);
                    if (lane < 2) {
                        asm volatile(
                            "{\n\t"
                            ".reg .b32 remAddr32;\n\t"
                            "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                            "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [remAddr32], %2;\n\t"
                            "}"
                            :: "r"(work_full_addr + work_stage_4 * 8), "r"(lane), "r"((uint32_t)(16)) : "memory");
                    }
                    if (elect_sync()) {
                        asm volatile(
                            "fence.proxy.async.shared::cta;\n\t"
                            "clusterlaunchcontrol.try_cancel.async.shared::cta"
                                ".mbarrier::complete_tx::bytes.multicast::cluster::all.b128"
                                " [%0], [%1];"
                            :: "r"(work_response_addr + work_stage_4 * 16), "r"(work_full_addr + work_stage_4 * 8)
                            : "memory");
                    }
                    mbarrier_wait(work_full_addr + (work_stage_4) * 8, _phase_work_full_4);
                    uint32_t _clc_valid_8 = 0;
                    asm volatile(
                        "{\n\t"
                        ".reg .pred p1;\n\t"
                        ".reg .b128 clc_r;\n\t"
                        "ld.shared.b128 clc_r, [%1];\n\t"
                        "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                        "selp.u32 %0, 1, 0, p1;\n\t"
                        "}\n"
                        : "=r"(_clc_valid_8)
                        : "r"(work_response_addr + work_stage_4 * 16)
                        : "memory");
                    uint32_t _clc_ctaid_12 = 0;
                    asm volatile(
                        "{\n\t"
                        ".reg .b128 clc_r;\n\t"
                        "ld.shared.b128 clc_r, [%1];\n\t"
                        "clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                        "}\n"
                        : "=r"(_clc_ctaid_12)
                        : "r"(work_response_addr + work_stage_4 * 16)
                        : "memory");
                    uint32_t _clc_ctaid_13 = 0;
                    asm volatile(
                        "{\n\t"
                        ".reg .b128 clc_r;\n\t"
                        "ld.shared.b128 clc_r, [%1];\n\t"
                        "clusterlaunchcontrol.query_cancel.get_first_ctaid::y.b32.b128 %0, clc_r;\n\t"
                        "}\n"
                        : "=r"(_clc_ctaid_13)
                        : "r"(work_response_addr + work_stage_4 * 16)
                        : "memory");
                    uint32_t _clc_ctaid_14 = 0;
                    asm volatile(
                        "{\n\t"
                        ".reg .b128 clc_r;\n\t"
                        "ld.shared.b128 clc_r, [%1];\n\t"
                        "clusterlaunchcontrol.query_cancel.get_first_ctaid::z.b32.b128 %0, clc_r;\n\t"
                        "}\n"
                        : "=r"(_clc_ctaid_14)
                        : "r"(work_response_addr + work_stage_4 * 16)
                        : "memory");
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    asm volatile(
                        "{\n\t"
                        ".reg .b32 remAddr32;\n\t"
                        "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                        "}"
                        :: "r"(work_empty_addr + work_stage_4 * 8), "r"(0) : "memory");
                    work_stage_4 += 1;
                    if (work_stage_4 == 3) { work_stage_4 = 0; _phase_work_empty ^= 1; _phase_work_full_4 ^= 1; }
                    if (_clc_valid_8 == 0) {
                        break;
                    }
                    m_tile_4 = _clc_ctaid_12 + (unsigned int)cta_rank;
                    n_tile_4 = _clc_ctaid_13;
                }
                #pragma unroll
                for (int _tail = 0; _tail < 3; _tail++) {
                    mbarrier_wait_cluster(work_empty_addr + (work_stage_4) * 8, _phase_work_empty);
                    work_stage_4 += 1;
                    if (work_stage_4 == 3) { work_stage_4 = 0; _phase_work_empty ^= 1; _phase_work_full_4 ^= 1; }
                }
            }
        }
    }

    // Cleanup
}

} // extern "C"

#undef FLASHINFER_INF
#undef NUM_DRAIN_PIPE_STAGES
#undef NUM_K_PIPE_STAGES
#undef NUM_MMA_PIPE_STAGES
#undef NUM_THROTTLE_PIPE_STAGES
#undef NUM_WORK_PIPE_STAGES
#undef SMEM_EPI_STAGING_OFF
#undef SMEM_EPI_STAGING_STAGE_BYTES
#undef SMEM_EPI_STAGING_STRIDE
#undef SMEM_EPI_STAGING_U64_OFF
#undef SMEM_EPI_STAGING_U64_STAGE_BYTES
#undef SMEM_EPI_STAGING_U64_STRIDE
#undef SMEM_FAST_DRAIN_RESPONSE_OFF
#undef SMEM_FAST_DRAIN_RESPONSE_STAGE_BYTES
#undef SMEM_FAST_DRAIN_RESPONSE_STRIDE
#undef SMEM_SMEM_A_OFF
#undef SMEM_SMEM_A_STAGE_BYTES
#undef SMEM_SMEM_A_STRIDE
#undef SMEM_SMEM_B_OFF
#undef SMEM_SMEM_B_STAGE_BYTES
#undef SMEM_SMEM_B_STRIDE
#undef SMEM_TOTAL
#undef SMEM_WORK_RESPONSE_OFF
#undef SMEM_WORK_RESPONSE_STAGE_BYTES
#undef SMEM_WORK_RESPONSE_STRIDE
#undef THREADS
#undef TMEM_ACCUM_OFFSET
#undef TMEM_NCOLS
#undef a_free_addr
#undef a_full_addr
#undef b_free_addr
#undef b_full_addr
#undef drain_full_addr
#undef epi_staging_addr
#undef epi_staging_u64_addr
#undef fast_drain_response_addr
#undef mma_free_addr
#undef mma_full_addr
#undef smem_a_addr
#undef smem_b_addr
#undef throttle_empty_addr
#undef throttle_full_addr
#undef work_empty_addr
#undef work_full_addr
#undef work_response_addr

#define FLASHINFER_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define SMEM_WEIGHTS_STAGE_OFF 0
#define SMEM_WEIGHTS_STAGE_STAGE_BYTES 32
#define SMEM_WEIGHTS_STAGE_STRIDE 32
#define SMEM_INVERSE_STAGE_OFF 32
#define SMEM_INVERSE_STAGE_STAGE_BYTES 32
#define SMEM_INVERSE_STAGE_STRIDE 32
#define SMEM_TOTAL 128
#define THREADS 128

extern "C" {

__global__ __launch_bounds__(128) void
kernel_rank_major_exact_unpermute_v1(__nv_bfloat16* __restrict__ expert_output, float* __restrict__ topk_weights, int* __restrict__ token_to_permuted, __nv_bfloat16* __restrict__ final_output, unsigned int hidden_size)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // Kernel setup ops
    float* weights_stage = reinterpret_cast<float*>(smem_raw + 0);
    const int weights_stage_addr = smem + 0;
    int* inverse_stage = reinterpret_cast<int*>(smem_raw + 32);
    const int inverse_stage_addr = smem + 32;

    // === Task calls (dependency order) ===
    unsigned int token = bid;
    unsigned int route_base = token * 8;
    if (tid < 8) {
        weights_stage[tid] = topk_weights[route_base + (unsigned int)tid];
        inverse_stage[tid] = token_to_permuted[route_base + (unsigned int)tid];
    }
    asm volatile("griddepcontrol.wait;" ::: "memory");
    __syncthreads();
    float weights[8];
    int rows[8];
    unsigned long long row_offsets[8];
    #pragma unroll
    for (int route_slot = 0; route_slot < 8; route_slot++) {
        float staged_weight = weights_stage[route_slot];
        __nv_bfloat16 _cvt_bf16_0 = __float2bfloat16(staged_weight);
        float _cvt_f32_0 = __bfloat162float(_cvt_bf16_0);
        weights[route_slot] = _cvt_f32_0;
        rows[route_slot] = inverse_stage[route_slot];
        row_offsets[route_slot] = (unsigned long long)rows[route_slot] * (unsigned long long)hidden_size;
    }
    unsigned long long output_base = (unsigned long long)token * (unsigned long long)hidden_size;
    unsigned int vector_elements = hidden_size / 8 * 8;
    #pragma unroll 1
    for (unsigned int base = tid * 8; base < vector_elements; base += 1024) {
        float accum[8];
        accum[0] = 0.0f;
        accum[1] = 0.0f;
        accum[2] = 0.0f;
        accum[3] = 0.0f;
        accum[4] = 0.0f;
        accum[5] = 0.0f;
        accum[6] = 0.0f;
        accum[7] = 0.0f;
        #pragma unroll
        for (int route_slot_1 = 0; route_slot_1 < 8; route_slot_1++) {
            float _vec_load_0[8];
            {
                const uint4* _vptr_0 = reinterpret_cast<const uint4*>(expert_output + (row_offsets[route_slot_1] + (unsigned long long)base) + 0);
                uint4 _vld_0[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
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
            for (int element = 0; element < 8; element++) {
                accum[element] = accum[element] + weights[route_slot_1] * _vec_load_0[element];
            }
        }
        {
            __nv_bfloat162 _pk[4];
            _pk[0] = __floats2bfloat162_rn(accum[0 + 0], accum[0 + 1]);
            _pk[1] = __floats2bfloat162_rn(accum[0 + 2], accum[0 + 3]);
            _pk[2] = __floats2bfloat162_rn(accum[0 + 4], accum[0 + 5]);
            _pk[3] = __floats2bfloat162_rn(accum[0 + 6], accum[0 + 7]);
            *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(final_output + (output_base + (unsigned long long)base)))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
        }
    }
}

} // extern "C"

#undef FLASHINFER_INF
#undef NUM_MAIN_STAGES
#undef SMEM_INVERSE_STAGE_OFF
#undef SMEM_INVERSE_STAGE_STAGE_BYTES
#undef SMEM_INVERSE_STAGE_STRIDE
#undef SMEM_TOTAL
#undef SMEM_WEIGHTS_STAGE_OFF
#undef SMEM_WEIGHTS_STAGE_STAGE_BYTES
#undef SMEM_WEIGHTS_STAGE_STRIDE
#undef THREADS
#undef inverse_stage_addr
#undef weights_stage_addr

#define FLASHINFER_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define THREADS 32

extern "C" {

__global__ __launch_bounds__(32) void
kernel_rank_major_partial_barrier_v1(int32_t pg_world, int32_t pg_rank, unsigned* const* __restrict__ pg_flags)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // === Task calls (dependency order) ===
    if (warp == 0) {
        if (elect_sync()) {
            // nvlink_barrier(pg_flags) phase=1
            {
                const int __ws = pg_world;
                const int __me = pg_rank;
                const int __slot = 1;
                unsigned* __local_flag = pg_flags[__me] + __slot;
                unsigned __previous_epoch;
                asm volatile("ld.relaxed.sys.global.u32 %0, [%1];"
                    : "=r"(__previous_epoch) : "l"(__local_flag) : "memory");
                const unsigned __arrival_epoch = __previous_epoch + 1u;
                const unsigned __release_epoch = __previous_epoch + 2u;
                asm volatile("fence.proxy.async.global;" ::: "memory");
                asm volatile("st.release.sys.global.u32 [%0], %1;"
                    :: "l"(__local_flag), "r"(__arrival_epoch) : "memory");
                if (__me == 0) {
                    for (int __r = 0; __r < __ws; ++__r) {
                        unsigned* __peer_flag = pg_flags[__r] + __slot;
                        while (true) {
                            unsigned __v;
                            asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(__v) : "l"(__peer_flag) : "memory");
                            if (__v == __arrival_epoch) break;
                        }
                    }
                    asm volatile("fence.proxy.alias;" ::: "memory");
                    for (int __r = 0; __r < __ws; ++__r) {
                        unsigned* __peer_flag = pg_flags[__r] + __slot;
                        asm volatile("st.release.sys.global.u32 [%0], %1;"
                            :: "l"(__peer_flag), "r"(__release_epoch) : "memory");
                    }
                } else {
                    while (true) {
                        unsigned __v;
                        asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(__v) : "l"(__local_flag) : "memory");
                        if (__v == __release_epoch) break;
                    }
                    asm volatile("fence.proxy.alias;" ::: "memory");
                }
                asm volatile("fence.proxy.async.global;" ::: "memory");
            }
        }
    }
}

} // extern "C"

#undef FLASHINFER_INF
#undef NUM_MAIN_STAGES
#undef THREADS

#define FLASHINFER_INF CUDART_INF_F
#define NUM_PULL_PIPE_STAGES 2
#define SMEM_PEER_ROW_OFF 1024
#define SMEM_PEER_ROW_STAGE_BYTES 14336
#define SMEM_PEER_ROW_STRIDE 14336
#define SMEM_TOTAL 29696
#define THREADS 256

extern "C" {

__global__ __launch_bounds__(256) void
kernel_rank_major_combine_v1(__nv_bfloat16* __restrict__ output, int32_t pg_world, int32_t pg_rank, unsigned* const* __restrict__ pg_flags, __nv_bfloat16* __restrict__ local_partials, __nv_bfloat16* const* __restrict__ local_partials_peers)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // Kernel setup ops
    __nv_bfloat16* peer_row = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int peer_row_addr = smem + 1024;

    // Mbarrier init (1 groups, 2 barriers)
    // Mbarriers at smem_raw[0..16)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'pull_pipe' ---
            // row_full: 2 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            mbarrier_init(smem + 8, 1);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncthreads();

    const int mbar_base = smem;
    #define row_full_addr (mbar_base + 0)

    // === Task calls (dependency order) ===
    int token = bid;
    int partial_row = pg_rank * 128 + token;
    float acc[32];
    acc[0] = 0.0f;
    acc[1] = 0.0f;
    acc[2] = 0.0f;
    acc[3] = 0.0f;
    acc[4] = 0.0f;
    acc[5] = 0.0f;
    acc[6] = 0.0f;
    acc[7] = 0.0f;
    acc[8] = 0.0f;
    acc[9] = 0.0f;
    acc[10] = 0.0f;
    acc[11] = 0.0f;
    acc[12] = 0.0f;
    acc[13] = 0.0f;
    acc[14] = 0.0f;
    acc[15] = 0.0f;
    acc[16] = 0.0f;
    acc[17] = 0.0f;
    acc[18] = 0.0f;
    acc[19] = 0.0f;
    acc[20] = 0.0f;
    acc[21] = 0.0f;
    acc[22] = 0.0f;
    acc[23] = 0.0f;
    acc[24] = 0.0f;
    acc[25] = 0.0f;
    acc[26] = 0.0f;
    acc[27] = 0.0f;
    acc[28] = 0.0f;
    acc[29] = 0.0f;
    acc[30] = 0.0f;
    acc[31] = 0.0f;
    int pull_stage = 0;
    unsigned int _phase_row_full = 0;
    #pragma unroll
    for (int peer = 0; peer < 8; peer++) {
        if (warp == 0) {
            if (elect_sync()) {
                mbarrier_arrive_expect_tx(row_full_addr + (pull_stage) * 8, 14336);
                // nvlink_pull: smem(peer_row_addr + (unsigned int)(pull_stage * 14336)) <- peers[peer] + (unsigned long long)partial_row * 14336, 14336B
                {
                    const void* __remote = (const void*)((const char*)((local_partials_peers)[peer]) + (uint64_t)((unsigned long long)partial_row * 14336));
                    asm volatile(
                        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes"
                        " [%0], [%1], %2, [%3];"
                        :: "r"(peer_row_addr + (unsigned int)(pull_stage * 14336)), "l"(__remote), "r"((uint32_t)(14336)), "r"(row_full_addr + (pull_stage) * 8)
                        : "memory");
                }
            }
        }
        mbarrier_wait(row_full_addr + (pull_stage) * 8, _phase_row_full);
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        #pragma unroll
        for (int item = 0; item < 4; item++) {
            int base = (tid + item * 256) * 8;
            if (base < 7168) {
                unsigned int packed[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&packed[0])), "=r"(*reinterpret_cast<uint32_t*>(&packed[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&packed[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&packed[(0) + 3]))
                    : "r"(peer_row_addr + (unsigned int)(pull_stage * 14336) + (unsigned int)(base * 2)));
                float packed_f32[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&packed_f32[_pair * 2])[0]), "=f"((&packed_f32[_pair * 2])[1])
                        : "r"(packed[_pair]));
                }
                #pragma unroll
                for (int element = 0; element < 8; element++) {
                    acc[item * 8 + element] = acc[item * 8 + element] + packed_f32[element];
                }
            }
        }
        __syncthreads();
        pull_stage += 1;
        if (pull_stage == 2) { pull_stage = 0; _phase_row_full ^= 1; }
    }
    unsigned long long out_row = (unsigned long long)token * 7168;
    #pragma unroll
    for (int item_1 = 0; item_1 < 4; item_1++) {
        int base_1 = (tid + item_1 * 256) * 8;
        if (base_1 < 7168) {
            {
                __nv_bfloat162 _pk[4];
                _pk[0] = __floats2bfloat162_rn(acc[item_1 * 8 + 0], acc[item_1 * 8 + 1]);
                _pk[1] = __floats2bfloat162_rn(acc[item_1 * 8 + 2], acc[item_1 * 8 + 3]);
                _pk[2] = __floats2bfloat162_rn(acc[item_1 * 8 + 4], acc[item_1 * 8 + 5]);
                _pk[3] = __floats2bfloat162_rn(acc[item_1 * 8 + 6], acc[item_1 * 8 + 7]);
                *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(output + (out_row + (unsigned long long)base_1)))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
            }
        }
    }

    // Cleanup
    __syncthreads();
}

} // extern "C"

#undef FLASHINFER_INF
#undef NUM_PULL_PIPE_STAGES
#undef SMEM_PEER_ROW_OFF
#undef SMEM_PEER_ROW_STAGE_BYTES
#undef SMEM_PEER_ROW_STRIDE
#undef SMEM_TOTAL
#undef THREADS
#undef peer_row_addr
#undef row_full_addr
