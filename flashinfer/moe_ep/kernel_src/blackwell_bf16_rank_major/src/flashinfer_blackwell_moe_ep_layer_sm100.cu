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
// Generated file, hand-edited here pending exporter support; change it only
// together with manifest.json (see VENDOR.md).
#include <stdint.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

struct __align__(128) FlashInferTensorMap { uint64_t opaque[16]; };
struct __align__(64) FlashInferTensorMap64 { uint64_t opaque[16]; };
static_assert(sizeof(FlashInferTensorMap64) == 128, "64-aligned tensor-map ABI size");
static_assert(alignof(FlashInferTensorMap64) == 64, "64-aligned tensor-map ABI alignment");
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

__device__ __forceinline__ void mbarrier_init_generic(void* mbar_addr, int count) {
    asm volatile("mbarrier.init.b64 [%0], %1;"
        :: "l"(mbar_addr), "r"(count));
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

// Source-faithful relaxed CTA wait used only by a typed protocol that does
// not attach the PTX acquire qualifier, such as FA4's interior P-ready edge.

__device__ __forceinline__ void mbarrier_wait_relaxed(int mbar_addr, int phase) {
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "LAB_WAIT_RELAXED:\n\t"
        "mbarrier.try_wait.parity.shared::cta.b64"
        " P1, [%0], %1, 10000000;\n\t"
        "@P1 bra.uni DONE_RELAXED;\n\t"
        "bra.uni LAB_WAIT_RELAXED;\n\t"
        "DONE_RELAXED:\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"(phase) : "memory");
}

// Exact source ports may request the PTX suspendTimeHint operand explicitly.
// The hint is expressed in nanoseconds and is kept separate from the canonical
// no-hint CTA helper so unrelated schedules retain their existing retry path.

__device__ __forceinline__ void mbarrier_wait_suspend(
        int mbar_addr, int phase, uint32_t suspend_time_hint) {
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "LAB_WAIT_SUSPEND:\n\t"
        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64"
        " P1, [%0], %1, %2;\n\t"
        "@P1 bra.uni DONE_SUSPEND;\n\t"
        "bra.uni LAB_WAIT_SUSPEND;\n\t"
        "DONE_SUSPEND:\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"(phase), "r"(suspend_time_hint) : "memory");
}

__device__ __forceinline__ void mbarrier_wait_cluster(int mbar_addr, int phase) {
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "LAB_WAIT_CLUSTER:\n\t"
        "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64"
        " P1, [%0], %1;\n\t"
        "@P1 bra.uni DONE_CLUSTER;\n\t"
        "bra.uni LAB_WAIT_CLUSTER;\n\t"
        "DONE_CLUSTER:\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"(phase) : "memory");
}

__device__ __forceinline__ void mbarrier_wait_hint(
        int mbar_addr, int phase, uint32_t suspend_time_hint) {
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        ".reg .u32 WAIT_ADDR;\n\t"
        "mov.u32 WAIT_ADDR, %0;\n\t"
        "LAB_WAIT_HINT:\n\t"
        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64"
        " P1, [WAIT_ADDR], %1, %2;\n\t"
        "@P1 bra.uni DONE_HINT;\n\t"
        "bra.uni LAB_WAIT_HINT;\n\t"
        "DONE_HINT:\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"(phase), "r"(suspend_time_hint) : "memory");
}

// Exact unqualified CTA wait used by source schedules whose PTX intentionally
// omits the acquire qualifier while retaining a typed suspendTimeHint operand.

__device__ __forceinline__ void mbarrier_wait_relaxed_hint(
        int mbar_addr, int phase, uint32_t suspend_time_hint) {
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "LAB_WAIT_RELAXED_HINT:\n\t"
        "mbarrier.try_wait.parity.shared::cta.b64"
        " P1, [%0], %1, %2;\n\t"
        "@P1 bra DONE_RELAXED_HINT;\n\t"
        "bra LAB_WAIT_RELAXED_HINT;\n\t"
        "DONE_RELAXED_HINT:\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"(phase), "r"(suspend_time_hint));
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

__device__ __forceinline__ void mbarrier_wait_token(int mbar_addr, int phase, uint32_t token) {
    if (token == 0) {
        mbarrier_wait(mbar_addr, phase);
    }
}

__device__ __forceinline__ void mbarrier_wait_token_suspend(
        int mbar_addr, int phase, uint32_t token, uint32_t suspend_time_hint) {
    if (token == 0) {
        mbarrier_wait_suspend(mbar_addr, phase, suspend_time_hint);
    }
}

__device__ __forceinline__ void mbarrier_wait_token_cluster(int mbar_addr, int phase, uint32_t token) {
    if (token == 0) {
        mbarrier_wait_cluster(mbar_addr, phase);
    }
}

__device__ __forceinline__ void mbarrier_wait_token_hint(
        int mbar_addr, int phase, uint32_t token, uint32_t suspend_time_hint) {
    if (token == 0) {
        mbarrier_wait_hint(mbar_addr, phase, suspend_time_hint);
    }
}

__device__ __forceinline__ void mbarrier_wait_token_cluster_hint(
        int mbar_addr, int phase, uint32_t token, uint32_t suspend_time_hint) {
    if (token == 0) {
        mbarrier_wait_cluster_hint(mbar_addr, phase, suspend_time_hint);
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
#define THREADS 256

extern "C" {

__global__ __launch_bounds__(256) void
kernel_rank_major_stage_inputs_v1(const __nv_bfloat16* __restrict__ hidden_in, const void* __restrict__ ids_in, const float* __restrict__ weights_in, __nv_bfloat16* __restrict__ hidden_states, int* __restrict__ topk_ids, float* __restrict__ topk_weights, int ids_is_int64, int hidden_copy, int active_tokens_per_rank)
{
    // One CTA per active token: the hidden row (7168 BF16 = 896 x 16 B) into the symmetric hidden_states row, the 8
    // routing ids narrowed to int32 (same value as the int64 hop: (int)(long long)id) and the 8 weights.
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    const int tid = threadIdx.x;
    const int token = blockIdx.x;
    if (hidden_copy) {
        const uint4* src = reinterpret_cast<const uint4*>(hidden_in + (unsigned long long)token * 7168);
        uint4* dst = reinterpret_cast<uint4*>(hidden_states + (unsigned long long)token * 7168);
        uint4 v[4];
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            const int idx = tid + i * 256;
            if (idx < 896) {
                v[i] = src[idx];
            }
        }
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            const int idx = tid + i * 256;
            if (idx < 896) {
                dst[idx] = v[i];
            }
        }
    }
    if (tid < 8) {
        const int route = token * 8 + tid;
        int id;
        if (ids_is_int64) {
            id = (int)(reinterpret_cast<const long long*>(ids_in)[route]);
        } else {
            id = (int)(long long)(reinterpret_cast<const int*>(ids_in)[route]);
        }
        topk_ids[route] = id;
        topk_weights[route] = weights_in[route];
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
kernel_rank_major_input_arrive_v2(int* __restrict__ topk_ids, int active_tokens_per_rank, int32_t pg_world, int32_t pg_rank, unsigned* const* __restrict__ pg_flags, unsigned* const* __restrict__ routed_peers, unsigned* const* __restrict__ route_codes_peers)
{
    const int tid = threadIdx.x;
    const uint32_t warp = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
    uint32_t lane;
    asm("mov.u32 %0, %%laneid;" : "=r"(lane));
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    asm volatile("griddepcontrol.wait;" ::: "memory");

    // === Task calls (dependency order) ===
    {
        // Owner masks of this rank's active tokens; lane o (< world) stores owner o's 4 words before releasing o.
        __syncwarp();
        unsigned int owner_words[4];
        #pragma unroll
        for (int k = 0; k < 4; k++) {
            const int t = (int)lane + 32 * k;
            unsigned int omask = 0u;
            if (t < active_tokens_per_rank) {
                #pragma unroll
                for (int slot = 0; slot < 8; slot++) {
                    const int expert = topk_ids[t * 8 + slot];
                    if ((unsigned int)expert < 256u) {
                        omask |= 1u << (expert >> 5);
                    }
                }
            }
            owner_words[k] = 0u;
            #pragma unroll
            for (int o = 0; o < 8; o++) {
                const unsigned int w = __ballot_sync(0xffffffff, (omask >> o) & 1u);
                if ((int)lane == o) {
                    owner_words[k] = w;
                }
            }
        }
        if ((int)lane < pg_world) {
            unsigned* dst = routed_peers[lane] + pg_rank * 4;
            #pragma unroll
            for (int k = 0; k < 4; k++) {
                dst[k] = owner_words[k];
            }
        }
    }
    {
        // Route codes: one byte per route of every active token for each owner o (o's local expert, or 0xff for a
        // route owned elsewhere), built in shared memory, then every lane stores a share of every owner's codes into
        // that owner's route_codes[this rank]; the __syncwarp after these stores orders them before lane o's release
        // to o below, so o's dispatch route CTA reads them locally after acquiring this rank's input mailbox.
        __shared__ __align__(16) uint2 s_codes[8][128];
        #pragma unroll 1
        for (int t = (int)lane; t < active_tokens_per_rank; t += 32) {
            int ids[8];
            #pragma unroll
            for (int slot = 0; slot < 8; slot++) {
                ids[slot] = topk_ids[t * 8 + slot];
            }
            #pragma unroll
            for (int o = 0; o < 8; o++) {
                unsigned int lo = 0u;
                unsigned int hi = 0u;
                #pragma unroll
                for (int slot = 0; slot < 8; slot++) {
                    const int local = ids[slot] - o * 32;
                    const unsigned int code = ((unsigned int)ids[slot] < 256u && (unsigned int)local < 32u) ? (unsigned int)local : 0xffu;
                    if (slot < 4) {
                        lo |= code << (8 * slot);
                    } else {
                        hi |= code << (8 * (slot - 4));
                    }
                }
                s_codes[o][t] = make_uint2(lo, hi);
            }
        }
        __syncwarp();
        // every lane stores 16 bytes (two tokens' codes) per instruction for every owner: a coalesced 512-byte warp
        // store per owner and step; the __syncwarp below orders all lanes' code stores before each lane's release
        #pragma unroll
        for (int o = 0; o < 8; o++) {
            if (o < pg_world) {
                uint4* dst = reinterpret_cast<uint4*>(route_codes_peers[o]) + pg_rank * 64;
                const uint4* src = reinterpret_cast<const uint4*>(&s_codes[o][0]);
                #pragma unroll 1
                for (int i = (int)lane; i < (active_tokens_per_rank + 1) / 2; i += 32) {
                    dst[i] = src[i];
                }
            }
        }
        __syncwarp();
    }
    // nvlink_arrive(pg_flags) phase=0: publish this rank's epoch to every peer's mailbox; consumers wait per peer.
    {
        const int __ws = pg_world;
        const int __me = pg_rank;
        const int __lane = lane;
        unsigned* __local_epoch = pg_flags[__me] + 0;
        unsigned __previous_epoch;
        asm volatile("ld.relaxed.sys.global.u32 %0, [%1];"
            : "=r"(__previous_epoch) : "l"(__local_epoch) : "memory");
        const unsigned __epoch = __previous_epoch + 1u;
        const int __bank = (int)(__epoch & 1u);
        const int __mailbox_base = 2 + (0 * 2 + __bank) * __ws;
        asm volatile("fence.proxy.async.global;" ::: "memory");
        asm volatile("fence.proxy.alias;" ::: "memory");
        if (__lane < __ws) {
            unsigned* __peer_mailbox = pg_flags[__lane] + __mailbox_base + __me;
            asm volatile("st.release.sys.global.u32 [%0], %1;"
                :: "l"(__peer_mailbox), "r"(__epoch) : "memory");
        }
        __syncwarp();
        if (__lane == 0) {
            asm volatile("st.relaxed.gpu.global.u32 [%0], %1;"
                :: "l"(__local_epoch), "r"(__epoch) : "memory");
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

// Pull path route: for every active count the route of this owner rank is computed by one extra dispatch CTA (block 0, 256
// threads) from the route codes every source stored into this rank's route_codes, concurrently with the row pulls; same tables as
// kernel_rank_major_route_fused_v1 (the compact row of a route inside its expert group may differ, which the bitwise
// output does not depend on). recv_local_ids / recv_weights stay with the row CTAs, as in the separate route kernel.
__device__ __noinline__ void rank_major_route_cta_ids(int active_tokens_per_rank, int32_t pg_world, int32_t pg_rank, unsigned* const* __restrict__ pg_flags, const unsigned* __restrict__ route_codes, int* __restrict__ expert_scatter_offsets, int* __restrict__ token_to_permuted, int* __restrict__ cta_to_expert, int* __restrict__ cta_to_mn_limit, int* __restrict__ expert_padded_row_offsets, int* __restrict__ num_non_exiting_ctas, int* __restrict__ total_padded_rows, int* __restrict__ route_map, __nv_bfloat16* __restrict__ zero_sentinel, int wait_all)
{
    __shared__ int s_rows[32];
    __shared__ int s_base[32];
    const int tid = threadIdx.x;
    uint32_t lane;
    asm("mov.u32 %0, %%laneid;" : "=r"(lane));
    if (wait_all && tid == 0) {
        unsigned __epoch;
        asm volatile("ld.relaxed.gpu.global.u32 %0, [%1];" : "=r"(__epoch) : "l"(pg_flags[pg_rank] + 0) : "memory");
        const unsigned* __mailbox = pg_flags[pg_rank] + 2 + (0 * 2 + (int)(__epoch & 1u)) * pg_world;
        for (int src = 0; src < pg_world; src++) {
            while (true) {
                unsigned __v;
                asm volatile("ld.relaxed.sys.global.u32 %0, [%1];" : "=r"(__v) : "l"(__mailbox + src) : "memory");
                if (__v != __epoch) continue;
                asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(__v) : "l"(__mailbox + src) : "memory");
                if (__v == __epoch) break;
            }
        }
    }
    if (tid < 32) {
        s_rows[tid] = 0;
    }
    #pragma unroll 1
    for (int elem = tid; elem < 7168; elem += 256) {
        zero_sentinel[elem] = 0.0f;
    }
    __syncthreads();
    const int owner_begin = pg_rank * 32;
    const int rows_total = 8 * active_tokens_per_rank;
    // pass 1: per-expert row counters; token_to_permuted holds (1 << 30 | local << 16 | row) until pass 3. The route
    // codes of every source were stored into this rank's route_codes by the sources' input_arrive before their
    // releases, which tid 0 acquired above (wait_all); all of a thread's code loads are issued before use.
    unsigned int cv[4][2];
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        const int r = tid + 256 * i;
        if (r < rows_total) {
            const int src = r / active_tokens_per_rank;
            const int tok = r - src * active_tokens_per_rank;
            asm volatile("ld.volatile.global.v2.u32 {%0,%1}, [%2];" : "=r"(cv[i][0]), "=r"(cv[i][1]) : "l"(route_codes + (src * 128 + tok) * 2) : "memory");
        }
    }
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        const int r = tid + 256 * i;
        if (r >= rows_total) {
            continue;
        }
        const int src = r / active_tokens_per_rank;
        const int tok = r - src * active_tokens_per_rank;
        const int route_base = (src * 128 + tok) * 8;
        #pragma unroll
        for (int k = 0; k < 8; k++) {
            const int code = (int)((cv[i][k >> 2] >> (8 * (k & 3))) & 0xffu);
            const int global_expert = code == 0xff ? -1 : owner_begin + code;
            if (global_expert >= owner_begin && global_expert < owner_begin + 32) {
                const int local = global_expert - owner_begin;
                token_to_permuted[route_base + k] = (1 << 30) | (local << 16) | atomicAdd(&s_rows[local], 1);
            } else {
                token_to_permuted[route_base + k] = 32768;
            }
        }
    }
    __syncthreads();
    if (tid < 32) {
        int expert = lane;
        int rows = s_rows[expert];
        expert_scatter_offsets[expert] = rows;
        int groups = (rows + 64 - 1) / 64;
        if (groups > 16) {
            groups = 16;
        }
        int inclusive = groups;
        #pragma unroll
        for (int delta = 1; delta < 32; delta <<= 1) {
            const int peer = __shfl_up_sync(0xFFFFFFFF, inclusive, delta, 32);
            if ((int)lane >= delta) {
                inclusive = inclusive + peer;
            }
        }
        int write_base = inclusive - groups;
        expert_padded_row_offsets[expert] = write_base * 64;
        s_base[expert] = write_base * 64;
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
    __syncthreads();
    // pass 3: compact rows (the thread re-reads its own pass-1 entries)
    #pragma unroll 1
    for (int r = tid; r < rows_total; r += 256) {
        const int src = r / active_tokens_per_rank;
        const int tok = r - src * active_tokens_per_rank;
        const int route_base = (src * 128 + tok) * 8;
        #pragma unroll
        for (int k = 0; k < 8; k++) {
            const int v = token_to_permuted[route_base + k];
            if (v & (1 << 30)) {
                const int compact_row = s_base[(v >> 16) & 0xff] + (v & 0xffff);
                route_map[compact_row] = route_base / 8;
                token_to_permuted[route_base + k] = compact_row;
            }
        }
    }
}

extern "C" {

__global__ __launch_bounds__(256, 8) void
kernel_rank_major_dispatch_v4(__nv_bfloat16* __restrict__ recv_hidden, int* __restrict__ recv_local_ids, float* __restrict__ recv_weights, int active_tokens_per_rank, int32_t pg_world, int32_t pg_rank, unsigned* const* __restrict__ pg_flags, __nv_bfloat16* __restrict__ hidden_states, __nv_bfloat16* const* __restrict__ hidden_states_peers, int* __restrict__ topk_ids, int* const* __restrict__ topk_ids_peers, float* __restrict__ topk_weights, float* const* __restrict__ topk_weights_peers, const unsigned* __restrict__ routed, int* __restrict__ expert_scatter_offsets, int* __restrict__ token_to_permuted, int* __restrict__ cta_to_expert, int* __restrict__ cta_to_mn_limit, int* __restrict__ expert_padded_row_offsets, int* __restrict__ num_non_exiting_ctas, int* __restrict__ total_padded_rows, int* __restrict__ route_map, __nv_bfloat16* __restrict__ zero_sentinel, const unsigned* __restrict__ route_codes)
{
    const int tid = threadIdx.x;
    const uint32_t warp = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
    uint32_t lane;
    asm("mov.u32 %0, %%laneid;" : "=r"(lane));
    asm volatile("griddepcontrol.wait;" ::: "memory");
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int mbar_base = smem;
    #define row_full_addr (mbar_base + 0)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // Kernel setup ops
    __nv_bfloat16* row_stage = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int row_stage_addr = smem + 1024;
    int* ids_stage = reinterpret_cast<int*>(smem_raw + 15360);
    const int ids_stage_addr = smem + 15360;
    float* weights_stage = reinterpret_cast<float*>(smem_raw + 15392);
    const int weights_stage_addr = smem + 15392;

    // Mbarrier init (1 pipeline groups, 0 ordered-sequence groups, 1 barriers)
    // Mbarriers at smem_raw[0..8)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // row_full: 1 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    const int has_route_cta = num_bids == 8 * active_tokens_per_rank + 1;
    if (has_route_cta && bid == 0) {
        rank_major_route_cta_ids(active_tokens_per_rank, pg_world, pg_rank, pg_flags, route_codes, expert_scatter_offsets, token_to_permuted, cta_to_expert, cta_to_mn_limit, expert_padded_row_offsets, num_non_exiting_ctas, total_padded_rows, route_map, zero_sentinel, 1);
        return;
    }
    // === Task calls (dependency order) ===
    int active_token = bid - has_route_cta;
    int src_rank = active_token / active_tokens_per_rank;
    int src_token = active_token - src_rank * active_tokens_per_rank;
    int recv_token = src_rank * 128 + src_token;
    if (tid == 0) {
        // Point-to-point input barrier: wait until source src_rank has published this forward's inputs.
        unsigned __epoch;
        asm volatile("ld.relaxed.gpu.global.u32 %0, [%1];" : "=r"(__epoch) : "l"(pg_flags[pg_rank] + 0) : "memory");
        const unsigned* __mailbox = pg_flags[pg_rank] + 2 + (0 * 2 + (int)(__epoch & 1u)) * pg_world + src_rank;
        while (true) {
            unsigned __v;
            asm volatile("ld.relaxed.sys.global.u32 %0, [%1];" : "=r"(__v) : "l"(__mailbox) : "memory");
            if (__v != __epoch) continue;
            asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(__v) : "l"(__mailbox) : "memory");
            if (__v == __epoch) break;
        }
        asm volatile("fence.proxy.async.global;" ::: "memory");
    }
    __syncthreads();
    {
        unsigned int routed_word;
        asm volatile("ld.relaxed.sys.global.u32 %0, [%1];" : "=r"(routed_word) : "l"(routed + src_rank * 4 + (src_token >> 5)) : "memory");
        if (((routed_word >> (src_token & 31)) & 1u) == 0u) {
            // No route of this token lands here: same route entries as a pulled all-remote id row, no row pull.
            if (tid < 8) {
                unsigned long long route_index = (unsigned long long)recv_token * 8 + (unsigned long long)tid;
                recv_local_ids[route_index] = -1;
                recv_weights[route_index] = 0.0f;
            }
            return;
        }
    }
    unsigned long long hidden_src_offset = (unsigned long long)src_token * 14336;
    unsigned long long routing_src_offset = (unsigned long long)src_token * 8 * 4;
    if (warp == 0) {
        if (elect_sync()) {
            asm volatile("fence.proxy.async.global;" ::: "memory");
            mbarrier_arrive_expect_tx(row_full_addr, 14400);
            // nvlink_pull: smem(row_stage_addr) <- peers[src_rank] + hidden_src_offset, 14336B
            {
                const void* __remote = (const void*)((const char*)(src_rank == pg_rank ? (const void*)hidden_states : (const void*)(hidden_states_peers)[src_rank]) + (uint64_t)(hidden_src_offset));
                asm volatile(
                    "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes"
                    " [%0], [%1], %2, [%3];"
                    :: "r"(row_stage_addr), "l"(__remote), "r"((uint32_t)(14336)), "r"(row_full_addr)
                    : "memory");
            }
            // nvlink_pull: smem(ids_stage_addr) <- peers[src_rank] + routing_src_offset, 32B
            {
                const void* __remote = (const void*)((const char*)(src_rank == pg_rank ? (const void*)topk_ids : (const void*)(topk_ids_peers)[src_rank]) + (uint64_t)(routing_src_offset));
                asm volatile(
                    "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes"
                    " [%0], [%1], %2, [%3];"
                    :: "r"(ids_stage_addr), "l"(__remote), "r"((uint32_t)(32)), "r"(row_full_addr)
                    : "memory");
            }
            // nvlink_pull: smem(weights_stage_addr) <- peers[src_rank] + routing_src_offset, 32B
            {
                const void* __remote = (const void*)((const char*)(src_rank == pg_rank ? (const void*)topk_weights : (const void*)(topk_weights_peers)[src_rank]) + (uint64_t)(routing_src_offset));
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

__global__ __launch_bounds__(1024) void
kernel_rank_major_route_fused_v1(int* __restrict__ recv_local_ids, int* __restrict__ expert_scatter_offsets, int* __restrict__ token_to_permuted, int* __restrict__ cta_to_expert, int* __restrict__ cta_to_mn_limit, int* __restrict__ expert_padded_row_offsets, int* __restrict__ num_non_exiting_ctas, int* __restrict__ total_padded_rows, int* __restrict__ route_map, __nv_bfloat16* __restrict__ zero_sentinel, int active_tokens_per_rank)
{
    // route_reset + route_count + route_finalize + route_scatter in one CTA (shared-memory expert counters).
    __shared__ int s_rows[32];
    __shared__ int s_base[32];
    const int tid = threadIdx.x;
    uint32_t lane;
    asm("mov.u32 %0, %%laneid;" : "=r"(lane));
    asm volatile("griddepcontrol.wait;" ::: "memory");
    if (tid < 32) {
        s_rows[tid] = 0;
    }
    #pragma unroll 1
    for (int elem = tid; elem < 7168; elem += 1024) {
        zero_sentinel[elem] = 0.0f;
    }
    __syncthreads();
    const int routes_per_rank = active_tokens_per_rank * 8;
    const int active_routes = 8 * routes_per_rank;
    int my_route[8];
    int my_expert[8];
    int my_row[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        const int active_route = tid + i * 1024;
        my_route[i] = -1;
        my_expert[i] = -1;
        my_row[i] = 0;
        if (active_route < active_routes) {
            const int src_rank = active_route / routes_per_rank;
            const int src_route = active_route - src_rank * routes_per_rank;
            my_route[i] = src_rank * 128 * 8 + src_route;
            my_expert[i] = recv_local_ids[my_route[i]];
        }
    }
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        if (my_expert[i] >= 0) {
            my_row[i] = atomicAdd(&s_rows[my_expert[i]], 1);
        }
    }
    __syncthreads();
    if (tid < 32) {
        int expert = lane;
        int rows = s_rows[expert];
        expert_scatter_offsets[expert] = rows;
        int groups = (rows + 64 - 1) / 64;
        if (groups > 16) {
            groups = 16;
        }
        int inclusive = groups;
        #pragma unroll
        for (int delta = 1; delta < 32; delta <<= 1) {
            const int peer = __shfl_up_sync(0xFFFFFFFF, inclusive, delta, 32);
            if ((int)lane >= delta) {
                inclusive = inclusive + peer;
            }
        }
        int write_base = inclusive - groups;
        expert_padded_row_offsets[expert] = write_base * 64;
        s_base[expert] = write_base * 64;
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
    __syncthreads();
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        if (my_route[i] >= 0) {
            if (my_expert[i] >= 0) {
                const int compact_row = s_base[my_expert[i]] + my_row[i];
                route_map[compact_row] = my_route[i] / 8;
                token_to_permuted[my_route[i]] = compact_row;
            } else {
                token_to_permuted[my_route[i]] = 32768;
            }
        }
    }
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
}

} // extern "C"

#undef FLASHINFER_INF
#undef NUM_MAIN_STAGES
#undef THREADS

#define FLASHINFER_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define THREADS 256



#undef FLASHINFER_INF
#undef NUM_MAIN_STAGES
#undef THREADS

#define FLASHINFER_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define THREADS 32



#undef FLASHINFER_INF
#undef NUM_MAIN_STAGES
#undef THREADS

#define FLASHINFER_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define THREADS 256



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
#define SMEM_FAST_DRAIN_RESPONSE_STAGE_BYTES 64
#define SMEM_FAST_DRAIN_RESPONSE_STRIDE 64
#define SMEM_TOTAL 223232
#define THREADS 384

extern "C" {

__global__ __launch_bounds__(384, 1) __cluster_dims__(2,1,1) void
kernel_rank_major_exact_fc1_swiglu_v1(FlashInferTensorMap const* weights, FlashInferTensorMap const* recv_hidden, FlashInferTensorMap const* compact_intermediate, int* __restrict__ route_map, int* __restrict__ num_non_exiting_ctas, int* __restrict__ cta_idx_y_to_batch_idx, int* __restrict__ cta_idx_y_to_mn_limit, int K)
{
    const int tid = threadIdx.x;
    const uint32_t warp = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
    uint32_t lane;
    asm("mov.u32 %0, %%laneid;" : "=r"(lane));

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

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

    // Mbarrier init (11 pipeline groups, 0 ordered-sequence groups, 37 barriers)
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
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;");
    }

    __syncthreads();
    asm volatile("barrier.cluster.arrive.release.aligned;");
    asm volatile("barrier.cluster.wait.acquire.aligned;");
    asm volatile("tcgen05.fence::after_thread_sync;");

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
                    : "r"(work_response_addr + work_stage * 16 + 0 * 16)
                    : "memory");
                uint32_t _clc_ctaid_6 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "=r"(_clc_ctaid_6)
                    : "r"(work_response_addr + work_stage * 16 + 0 * 16)
                    : "memory");
                uint32_t _clc_ctaid_7 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.get_first_ctaid::y.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "=r"(_clc_ctaid_7)
                    : "r"(work_response_addr + work_stage * 16 + 0 * 16)
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
                        : "r"(work_response_addr + work_stage_1 * 16 + 0 * 16)
                        : "memory");
                    uint32_t _clc_ctaid_4 = 0;
                    asm volatile(
                        "{\n\t"
                        ".reg .b128 clc_r;\n\t"
                        "ld.shared.b128 clc_r, [%1];\n\t"
                        "clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                        "}\n"
                        : "=r"(_clc_ctaid_4)
                        : "r"(work_response_addr + work_stage_1 * 16 + 0 * 16)
                        : "memory");
                    uint32_t _clc_ctaid_5 = 0;
                    asm volatile(
                        "{\n\t"
                        ".reg .b128 clc_r;\n\t"
                        "ld.shared.b128 clc_r, [%1];\n\t"
                        "clusterlaunchcontrol.query_cancel.get_first_ctaid::y.b32.b128 %0, clc_r;\n\t"
                        "}\n"
                        : "=r"(_clc_ctaid_5)
                        : "r"(work_response_addr + work_stage_1 * 16 + 0 * 16)
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
                    : "r"(work_response_addr + work_stage_2 * 16 + 0 * 16)
                    : "memory");
                uint32_t _clc_ctaid_0 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "=r"(_clc_ctaid_0)
                    : "r"(work_response_addr + work_stage_2 * 16 + 0 * 16)
                    : "memory");
                uint32_t _clc_ctaid_1 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.get_first_ctaid::y.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "=r"(_clc_ctaid_1)
                    : "r"(work_response_addr + work_stage_2 * 16 + 0 * 16)
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
                    : "r"(work_response_addr + work_stage_3 * 16 + 0 * 16)
                    : "memory");
                uint32_t _clc_ctaid_2 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "=r"(_clc_ctaid_2)
                    : "r"(work_response_addr + work_stage_3 * 16 + 0 * 16)
                    : "memory");
                uint32_t _clc_ctaid_3 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.get_first_ctaid::y.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "=r"(_clc_ctaid_3)
                    : "r"(work_response_addr + work_stage_3 * 16 + 0 * 16)
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
                                    :: "r"(fast_drain_response_addr + drain_stage * 64 + 0 * 16), "r"(drain_full_addr + drain_stage * 8)
                                    : "memory");
                                asm volatile(
                                    "fence.proxy.async.shared::cta;\n\t"
                                    "clusterlaunchcontrol.try_cancel.async.shared::cta"
                                        ".mbarrier::complete_tx::bytes.b128"
                                        " [%0], [%1];"
                                    :: "r"(fast_drain_response_addr + drain_stage * 64 + 1 * 16), "r"(drain_full_addr + drain_stage * 8)
                                    : "memory");
                                asm volatile(
                                    "fence.proxy.async.shared::cta;\n\t"
                                    "clusterlaunchcontrol.try_cancel.async.shared::cta"
                                        ".mbarrier::complete_tx::bytes.b128"
                                        " [%0], [%1];"
                                    :: "r"(fast_drain_response_addr + drain_stage * 64 + 2 * 16), "r"(drain_full_addr + drain_stage * 8)
                                    : "memory");
                                asm volatile(
                                    "fence.proxy.async.shared::cta;\n\t"
                                    "clusterlaunchcontrol.try_cancel.async.shared::cta"
                                        ".mbarrier::complete_tx::bytes.b128"
                                        " [%0], [%1];"
                                    :: "r"(fast_drain_response_addr + drain_stage * 64 + 3 * 16), "r"(drain_full_addr + drain_stage * 8)
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
                                : "r"(fast_drain_response_addr + drain_stage * 64 + 0 * 16)
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
                                : "r"(fast_drain_response_addr + drain_stage * 64 + 1 * 16)
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
                                : "r"(fast_drain_response_addr + drain_stage * 64 + 2 * 16)
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
                                : "r"(fast_drain_response_addr + drain_stage * 64 + 3 * 16)
                                : "memory");
                            canceled += _clc_valid_7;
                            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                            _phase_drain_full ^= 1;
                            if (canceled == 0) {
                                break;
                            }
                        }
                    }
                    mbarrier_wait_cluster_hint(work_empty_addr + (work_stage_4) * 8, _phase_work_empty, 10000000);
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
                            :: "r"(work_response_addr + work_stage_4 * 16 + 0 * 16), "r"(work_full_addr + work_stage_4 * 8)
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
                        : "r"(work_response_addr + work_stage_4 * 16 + 0 * 16)
                        : "memory");
                    uint32_t _clc_ctaid_8 = 0;
                    asm volatile(
                        "{\n\t"
                        ".reg .b128 clc_r;\n\t"
                        "ld.shared.b128 clc_r, [%1];\n\t"
                        "clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                        "}\n"
                        : "=r"(_clc_ctaid_8)
                        : "r"(work_response_addr + work_stage_4 * 16 + 0 * 16)
                        : "memory");
                    uint32_t _clc_ctaid_9 = 0;
                    asm volatile(
                        "{\n\t"
                        ".reg .b128 clc_r;\n\t"
                        "ld.shared.b128 clc_r, [%1];\n\t"
                        "clusterlaunchcontrol.query_cancel.get_first_ctaid::y.b32.b128 %0, clc_r;\n\t"
                        "}\n"
                        : "=r"(_clc_ctaid_9)
                        : "r"(work_response_addr + work_stage_4 * 16 + 0 * 16)
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
                    mbarrier_wait_cluster_hint(work_empty_addr + (work_stage_4) * 8, _phase_work_empty, 10000000);
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
                    : "r"(work_response_addr + work_stage_5 * 16 + 0 * 16)
                    : "memory");
                uint32_t _clc_ctaid_10 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "=r"(_clc_ctaid_10)
                    : "r"(work_response_addr + work_stage_5 * 16 + 0 * 16)
                    : "memory");
                uint32_t _clc_ctaid_11 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.get_first_ctaid::y.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "=r"(_clc_ctaid_11)
                    : "r"(work_response_addr + work_stage_5 * 16 + 0 * 16)
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
#define SMEM_FAST_DRAIN_RESPONSE_STAGE_BYTES 64
#define SMEM_FAST_DRAIN_RESPONSE_STRIDE 64
#define SMEM_TOTAL 223360
#define THREADS 256

extern "C" {

__global__ __launch_bounds__(256, 1) __cluster_dims__(2,1,1) void
kernel_trtllm_moe_bmm_tile_n64_fc2_bf16(FlashInferTensorMap const* A, FlashInferTensorMap const* B, FlashInferTensorMap const* C, int* __restrict__ num_non_exiting_ctas, int* __restrict__ cta_idx_y_to_batch_idx, int* __restrict__ cta_idx_y_to_mn_limit, int K)
{
    const int tid = threadIdx.x;
    const uint32_t warp = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
    uint32_t lane;
    asm("mov.u32 %0, %%laneid;" : "=r"(lane));

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

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

    // Mbarrier init (11 pipeline groups, 0 ordered-sequence groups, 37 barriers)
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
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;");
    }

    __syncthreads();
    asm volatile("barrier.cluster.arrive.release.aligned;");
    asm volatile("barrier.cluster.wait.acquire.aligned;");
    asm volatile("tcgen05.fence::after_thread_sync;");

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
                    : "r"(work_response_addr + work_stage * 16 + 0 * 16)
                    : "memory");
                uint32_t _clc_ctaid_9 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "=r"(_clc_ctaid_9)
                    : "r"(work_response_addr + work_stage * 16 + 0 * 16)
                    : "memory");
                uint32_t _clc_ctaid_10 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.get_first_ctaid::y.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "=r"(_clc_ctaid_10)
                    : "r"(work_response_addr + work_stage * 16 + 0 * 16)
                    : "memory");
                uint32_t _clc_ctaid_11 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.get_first_ctaid::z.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "=r"(_clc_ctaid_11)
                    : "r"(work_response_addr + work_stage * 16 + 0 * 16)
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
                        : "r"(work_response_addr + work_stage_1 * 16 + 0 * 16)
                        : "memory");
                    uint32_t _clc_ctaid_6 = 0;
                    asm volatile(
                        "{\n\t"
                        ".reg .b128 clc_r;\n\t"
                        "ld.shared.b128 clc_r, [%1];\n\t"
                        "clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                        "}\n"
                        : "=r"(_clc_ctaid_6)
                        : "r"(work_response_addr + work_stage_1 * 16 + 0 * 16)
                        : "memory");
                    uint32_t _clc_ctaid_7 = 0;
                    asm volatile(
                        "{\n\t"
                        ".reg .b128 clc_r;\n\t"
                        "ld.shared.b128 clc_r, [%1];\n\t"
                        "clusterlaunchcontrol.query_cancel.get_first_ctaid::y.b32.b128 %0, clc_r;\n\t"
                        "}\n"
                        : "=r"(_clc_ctaid_7)
                        : "r"(work_response_addr + work_stage_1 * 16 + 0 * 16)
                        : "memory");
                    uint32_t _clc_ctaid_8 = 0;
                    asm volatile(
                        "{\n\t"
                        ".reg .b128 clc_r;\n\t"
                        "ld.shared.b128 clc_r, [%1];\n\t"
                        "clusterlaunchcontrol.query_cancel.get_first_ctaid::z.b32.b128 %0, clc_r;\n\t"
                        "}\n"
                        : "=r"(_clc_ctaid_8)
                        : "r"(work_response_addr + work_stage_1 * 16 + 0 * 16)
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
                    : "r"(work_response_addr + work_stage_2 * 16 + 0 * 16)
                    : "memory");
                uint32_t _clc_ctaid_0 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "=r"(_clc_ctaid_0)
                    : "r"(work_response_addr + work_stage_2 * 16 + 0 * 16)
                    : "memory");
                uint32_t _clc_ctaid_1 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.get_first_ctaid::y.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "=r"(_clc_ctaid_1)
                    : "r"(work_response_addr + work_stage_2 * 16 + 0 * 16)
                    : "memory");
                uint32_t _clc_ctaid_2 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.get_first_ctaid::z.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "=r"(_clc_ctaid_2)
                    : "r"(work_response_addr + work_stage_2 * 16 + 0 * 16)
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
                    : "r"(work_response_addr + work_stage_3 * 16 + 0 * 16)
                    : "memory");
                uint32_t _clc_ctaid_3 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "=r"(_clc_ctaid_3)
                    : "r"(work_response_addr + work_stage_3 * 16 + 0 * 16)
                    : "memory");
                uint32_t _clc_ctaid_4 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.get_first_ctaid::y.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "=r"(_clc_ctaid_4)
                    : "r"(work_response_addr + work_stage_3 * 16 + 0 * 16)
                    : "memory");
                uint32_t _clc_ctaid_5 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.get_first_ctaid::z.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "=r"(_clc_ctaid_5)
                    : "r"(work_response_addr + work_stage_3 * 16 + 0 * 16)
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
                                    :: "r"(fast_drain_response_addr + drain_stage * 64 + 0 * 16), "r"(drain_full_addr + drain_stage * 8)
                                    : "memory");
                                asm volatile(
                                    "fence.proxy.async.shared::cta;\n\t"
                                    "clusterlaunchcontrol.try_cancel.async.shared::cta"
                                        ".mbarrier::complete_tx::bytes.b128"
                                        " [%0], [%1];"
                                    :: "r"(fast_drain_response_addr + drain_stage * 64 + 1 * 16), "r"(drain_full_addr + drain_stage * 8)
                                    : "memory");
                                asm volatile(
                                    "fence.proxy.async.shared::cta;\n\t"
                                    "clusterlaunchcontrol.try_cancel.async.shared::cta"
                                        ".mbarrier::complete_tx::bytes.b128"
                                        " [%0], [%1];"
                                    :: "r"(fast_drain_response_addr + drain_stage * 64 + 2 * 16), "r"(drain_full_addr + drain_stage * 8)
                                    : "memory");
                                asm volatile(
                                    "fence.proxy.async.shared::cta;\n\t"
                                    "clusterlaunchcontrol.try_cancel.async.shared::cta"
                                        ".mbarrier::complete_tx::bytes.b128"
                                        " [%0], [%1];"
                                    :: "r"(fast_drain_response_addr + drain_stage * 64 + 3 * 16), "r"(drain_full_addr + drain_stage * 8)
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
                                : "r"(fast_drain_response_addr + drain_stage * 64 + 0 * 16)
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
                                : "r"(fast_drain_response_addr + drain_stage * 64 + 1 * 16)
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
                                : "r"(fast_drain_response_addr + drain_stage * 64 + 2 * 16)
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
                                : "r"(fast_drain_response_addr + drain_stage * 64 + 3 * 16)
                                : "memory");
                            canceled += _clc_valid_7;
                            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                            _phase_drain_full ^= 1;
                            if (canceled == 0) {
                                break;
                            }
                        }
                    }
                    mbarrier_wait_cluster_hint(work_empty_addr + (work_stage_4) * 8, _phase_work_empty, 10000000);
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
                            :: "r"(work_response_addr + work_stage_4 * 16 + 0 * 16), "r"(work_full_addr + work_stage_4 * 8)
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
                        : "r"(work_response_addr + work_stage_4 * 16 + 0 * 16)
                        : "memory");
                    uint32_t _clc_ctaid_12 = 0;
                    asm volatile(
                        "{\n\t"
                        ".reg .b128 clc_r;\n\t"
                        "ld.shared.b128 clc_r, [%1];\n\t"
                        "clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                        "}\n"
                        : "=r"(_clc_ctaid_12)
                        : "r"(work_response_addr + work_stage_4 * 16 + 0 * 16)
                        : "memory");
                    uint32_t _clc_ctaid_13 = 0;
                    asm volatile(
                        "{\n\t"
                        ".reg .b128 clc_r;\n\t"
                        "ld.shared.b128 clc_r, [%1];\n\t"
                        "clusterlaunchcontrol.query_cancel.get_first_ctaid::y.b32.b128 %0, clc_r;\n\t"
                        "}\n"
                        : "=r"(_clc_ctaid_13)
                        : "r"(work_response_addr + work_stage_4 * 16 + 0 * 16)
                        : "memory");
                    uint32_t _clc_ctaid_14 = 0;
                    asm volatile(
                        "{\n\t"
                        ".reg .b128 clc_r;\n\t"
                        "ld.shared.b128 clc_r, [%1];\n\t"
                        "clusterlaunchcontrol.query_cancel.get_first_ctaid::z.b32.b128 %0, clc_r;\n\t"
                        "}\n"
                        : "=r"(_clc_ctaid_14)
                        : "r"(work_response_addr + work_stage_4 * 16 + 0 * 16)
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
                    mbarrier_wait_cluster_hint(work_empty_addr + (work_stage_4) * 8, _phase_work_empty, 10000000);
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
kernel_rank_major_exact_unpermute_v8(__nv_bfloat16* __restrict__ expert_output, float* __restrict__ topk_weights, int* __restrict__ token_to_permuted, __nv_bfloat16* __restrict__ final_output, unsigned int hidden_size, int active_tokens_per_rank)
{
    const int tid = threadIdx.x;
    const uint32_t warp = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
    uint32_t lane;
    asm("mov.u32 %0, %%laneid;" : "=r"(lane));

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
    int active_token = bid;
    int src_rank = active_token / active_tokens_per_rank;
    int src_token = active_token - src_rank * active_tokens_per_rank;
    unsigned int token = (unsigned int)(src_rank * 128 + src_token);
    unsigned int route_base = token * 8;
    // topk_weights / token_to_permuted are written by earlier grids (route kernel or dispatch route CTA): every read of
    // them follows griddepcontrol.wait. Before the wait only an L2 prefetch hint of their two 32-byte lines is issued
    // (no memory-model effect; the values are loaded after the wait).
    if (tid == 0) {
        asm volatile("prefetch.global.L2 [%0];" :: "l"(topk_weights + route_base) : "memory");
    } else if (tid == 32) {
        asm volatile("prefetch.global.L2 [%0];" :: "l"(token_to_permuted + route_base) : "memory");
    }
    asm volatile("griddepcontrol.wait;" ::: "memory");
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    if (tid < 8) {
        weights_stage[tid] = topk_weights[route_base + (unsigned int)tid];
        inverse_stage[tid] = token_to_permuted[route_base + (unsigned int)tid];
    }
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
    // One 1024-column chunk per CTA (grid.y = 7): one loop trip instead of seven.
    #pragma unroll 1
    for (unsigned int base = blockIdx.y * 1024 + tid * 8; base < vector_elements; base += 1024 * gridDim.y) {
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
            if (rows[route_slot_1] == 32768) {
                // Non-local route: weight 0.0 times the all-+0.0 sentinel row; apply its +0.0 term without the load.
                #pragma unroll
                for (int element = 0; element < 8; element++) {
                    asm volatile("add.ftz.f32 %0, %0, 0f00000000;" : "+f"(accum[element]));
                }
                continue;
            }
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
kernel_rank_major_partial_arrive_v1(int32_t pg_world, int32_t pg_rank, unsigned* const* __restrict__ pg_flags)
{
    const int tid = threadIdx.x;
    const uint32_t warp = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
    uint32_t lane;
    asm("mov.u32 %0, %%laneid;" : "=r"(lane));
    asm volatile("griddepcontrol.wait;" ::: "memory");
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");

    // === Task calls (dependency order) ===
    // nvlink_arrive(pg_flags) phase=1: publish this rank's epoch to every peer's mailbox; consumers wait per peer.
    {
        const int __ws = pg_world;
        const int __me = pg_rank;
        const int __lane = lane;
        unsigned* __local_epoch = pg_flags[__me] + 1;
        unsigned __previous_epoch;
        asm volatile("ld.relaxed.sys.global.u32 %0, [%1];"
            : "=r"(__previous_epoch) : "l"(__local_epoch) : "memory");
        const unsigned __epoch = __previous_epoch + 1u;
        const int __bank = (int)(__epoch & 1u);
        const int __mailbox_base = 2 + (1 * 2 + __bank) * __ws;
        asm volatile("fence.proxy.async.global;" ::: "memory");
        asm volatile("fence.proxy.alias;" ::: "memory");
        if (__lane < __ws) {
            unsigned* __peer_mailbox = pg_flags[__lane] + __mailbox_base + __me;
            asm volatile("st.release.sys.global.u32 [%0], %1;"
                :: "l"(__peer_mailbox), "r"(__epoch) : "memory");
        }
        __syncwarp();
        if (__lane == 0) {
            asm volatile("st.relaxed.gpu.global.u32 [%0], %1;"
                :: "l"(__local_epoch), "r"(__epoch) : "memory");
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

__global__ __launch_bounds__(128) void
kernel_rank_major_combine_v3(__nv_bfloat16* __restrict__ output, int32_t pg_world, int32_t pg_rank, unsigned* const* __restrict__ pg_flags, __nv_bfloat16* __restrict__ local_partials, __nv_bfloat16* const* __restrict__ local_partials_peers, const int* __restrict__ topk_ids)
{
    // One CTA per (token, 1024-column chunk): all owner pulls in flight at once, then the FP32 sum in owner order.
    const int tid = threadIdx.x;
    extern __shared__ __align__(1024) char smem_raw[];
    const int smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);
    asm volatile("griddepcontrol.wait;" ::: "memory");
    const int token = blockIdx.x;
    const int chunk = blockIdx.y;
    const int partial_row = pg_rank * 128 + token;
    unsigned int owners = 0xffu;
    owners = 0u;
    #pragma unroll
    for (int slot = 0; slot < 8; slot++) {
        const int expert = topk_ids[token * 8 + slot];
        if ((unsigned int)expert < 256u) {
            owners |= 1u << (expert >> 5);
        }
    }
    if (tid == 0) {
        #pragma unroll
        for (int peer = 0; peer < 8; peer++) {
            mbarrier_init(smem + peer * 8, 1);
        }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    __syncthreads();
    if (tid == 0) {
        // Point-to-point partial barrier: pull each routed owner's chunk as soon as that owner has published its
        // partial rows; wait for every owner (the forward does not complete before all owners arrived).
        unsigned __epoch;
        asm volatile("ld.relaxed.gpu.global.u32 %0, [%1];" : "=r"(__epoch) : "l"(pg_flags[pg_rank] + 1) : "memory");
        const unsigned* __mailbox = pg_flags[pg_rank] + 2 + (1 * 2 + (int)(__epoch & 1u)) * pg_world;
        unsigned int pending = 0xffu;
        while (pending != 0u) {
            #pragma unroll 1
            for (int peer = 0; peer < 8; peer++) {
                if (((pending >> peer) & 1u) == 0u) {
                    continue;
                }
                unsigned __v;
                asm volatile("ld.relaxed.sys.global.u32 %0, [%1];" : "=r"(__v) : "l"(__mailbox + peer) : "memory");
                if (__v != __epoch) {
                    continue;
                }
                asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(__v) : "l"(__mailbox + peer) : "memory");
                if (__v != __epoch) {
                    continue;
                }
                pending &= ~(1u << peer);
                if ((owners >> peer) & 1u) {
                    asm volatile("fence.proxy.async.global;" ::: "memory");
                    mbarrier_arrive_expect_tx(smem + peer * 8, 2048);
                    const void* __remote = (const void*)((const char*)(peer == pg_rank ? local_partials : (local_partials_peers)[peer]) + (uint64_t)((unsigned long long)partial_row * 14336 + (unsigned long long)chunk * 2048));
                    asm volatile(
                        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes"
                        " [%0], [%1], %2, [%3];"
                        :: "r"(smem + 1024 + peer * 2048), "l"(__remote), "r"((uint32_t)(2048)), "r"(smem + peer * 8)
                        : "memory");
                }
            }
        }
    }
    float acc[8];
    #pragma unroll
    for (int element = 0; element < 8; element++) {
        acc[element] = 0.0f;
    }
    #pragma unroll
    for (int peer = 0; peer < 8; peer++) {
        if ((owners >> peer) & 1u) {
            mbarrier_wait(smem + peer * 8, 0);
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            unsigned int packed[4];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(packed[0]), "=r"(packed[1]), "=r"(packed[2]), "=r"(packed[3])
                : "r"(smem + 1024 + peer * 2048 + tid * 16));
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
                acc[element] = acc[element] + packed_f32[element];
            }
        } else {
            // The owner holds no route of this token: its partial row is all +0.0. Add that +0.0 explicitly.
            #pragma unroll
            for (int element = 0; element < 8; element++) {
                asm volatile("add.ftz.f32 %0, %0, 0f00000000;" : "+f"(acc[element]));
            }
        }
    }
    {
        __nv_bfloat162 _pk[4];
        _pk[0] = __floats2bfloat162_rn(acc[0], acc[1]);
        _pk[1] = __floats2bfloat162_rn(acc[2], acc[3]);
        _pk[2] = __floats2bfloat162_rn(acc[4], acc[5]);
        _pk[3] = __floats2bfloat162_rn(acc[6], acc[7]);
        *reinterpret_cast<uint4*>(output + ((unsigned long long)token * 7168 + (unsigned long long)chunk * 1024 + (unsigned long long)tid * 8)) = *reinterpret_cast<uint4*>(&_pk[0]);
    }
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

#define FLASHINFER_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define THREADS 256

// LL protocol note (small-T path: kernel_rank_major_ll_push_v4 / ll_recv / exact_unpermute_v6 / combine_v6).
// Data: every LL word is {low 4 bytes payload, high 4 bytes epoch}, stored with st.relaxed.sys (16-byte vectors of two
// independent 8-byte elements) and read with ld.relaxed.sys; a reader uses a word only when its epoch is this
// forward's, so the payload is the one stored in this forward (single-copy atomicity of morally strong 8-byte
// accesses). No flag or fence orders the data.
// Epoch: ll_control[0] = epoch of the last LL forward (read by ll_push, which uses e = [0] + 1 and stores [1] = e);
// ll_recv, exact_unpermute_v6 and combine_v6 read [1] after griddepcontrol.wait; ll_recv's CTA 0 stores [0] = e.
// Buffers start zeroed and e >= 1. Slots are double-banked by e & 1.
// Buffer reuse: a bank is rewritten two LL forwards later. At the start of every LL forward each rank releases its
// epoch to every rank (ll_push warp 8: fence.acq_rel.sys, then st.relaxed.sys into ll_reuse[this rank] of the peer);
// combine_v6 (CTA (0, 0), warp 4) acquires every peer's word (ld.acquire.sys until >= e). So a reader's last access to
// a bank (ll_recv / combine of forward k) precedes, in grid order, its release in forward k + 1, which the writer
// acquires in forward k + 1 before (grid order) its stores of forward k + 2.
// Epoch wrap: 0 is never an epoch (e = [0] + 1, and 0 maps to 2, keeping the bank parity). In the forwards with
// e = 0xfffffffe and e = 0xffffffff, ll_push warp 8 of CTA (0, d, 0) zeroes (st.relaxed.sys) this rank's receive slots
// that peer d writes in bank (e & 1) ^ 1 (dispatch and combine) before its reuse release to d. The zeroes follow every
// earlier write into that bank (forward k - 1 words were read by this rank: coherence after a morally strong read;
// older words through the release acquired in combine k - 1), and peer d writes that bank again only in forward k + 1,
// after acquiring this release. Between two zeroings a bank sees each epoch value once (bank 0: 2, 4, ..., 0xfffffffe;
// bank 1: 0xffffffff, 3, 5, ..., 0xfffffffd), so a word carrying the reader's epoch was written in this forward, for any
// number of forwards. The reuse acquire compares (int)(seen - e), exact across the wrap.
#define LL_MAX_T 32
#define LL_HDR_WORDS 16
#define LL_ROW_WORDS 3584
#define LL_SLOT_WORDS (LL_HDR_WORDS + LL_ROW_WORDS)

__device__ __forceinline__ void ll_store2(unsigned long long* p, unsigned a, unsigned b, unsigned epoch) {
    const unsigned long long hi = (unsigned long long)epoch << 32;
    asm volatile("st.relaxed.sys.global.v2.u64 [%0], {%1, %2};"
        :: "l"(p), "l"(hi | (unsigned long long)a), "l"(hi | (unsigned long long)b) : "memory");
}

__device__ __forceinline__ void ll_load2(const unsigned long long* p, unsigned long long& a, unsigned long long& b) {
    asm volatile("ld.relaxed.sys.global.v2.u64 {%0, %1}, [%2];" : "=l"(a), "=l"(b) : "l"(p) : "memory");
}

__device__ __forceinline__ bool ll_ready2(unsigned long long a, unsigned long long b, unsigned epoch) {
    return (unsigned)(a >> 32) == epoch && (unsigned)(b >> 32) == epoch;
}

extern "C" {

__global__ __launch_bounds__(288) void
kernel_rank_major_ll_push_v4(const __nv_bfloat16* __restrict__ hidden_in, const void* __restrict__ ids_in, const float* __restrict__ weights_in, int* __restrict__ topk_ids, int ids_is_int64, int active_tokens_per_rank, int32_t pg_world, int32_t pg_rank, unsigned long long* const* __restrict__ ll_dispatch_peers, unsigned* __restrict__ ll_control, unsigned* const* __restrict__ ll_reuse_peers, unsigned long long* const* __restrict__ ll_combine_peers)
{
    // CTA (token, destination d, row split): lanes 0-7 of warp 0 in split 0 push the token's header (ids, weights) to
    // every d; warps 0-7 push this split's share of the hidden row to d only when one of the token's routes is owned
    // by d. Warp 8 of CTA (0, d, 0) publishes the buffer-reuse release to d off the data path. Ids narrowed to int32
    // as in kernel_rank_major_stage_inputs_v1 ((int)(long long)id).
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int token = blockIdx.x;
    const int dst = blockIdx.y;
    const int split = blockIdx.z;
    const int splits = gridDim.z;
    unsigned previous;
    asm volatile("ld.relaxed.gpu.global.u32 %0, [%1];" : "=r"(previous) : "l"(ll_control + 0) : "memory");
    // epoch 0 is the zero word's value and never used: 0xffffffff is followed by 2 (same bank parity as 0)
    unsigned epoch = previous + 1u;
    if (epoch == 0u) {
        epoch = 2u;
    }
    if (warp == 8) {
        if (token == 0 && split == 0) {
            if (epoch >= 0xfffffffeu) {
                // wrap phase (see the LL protocol note): zero this rank's receive slots that peer dst writes in the
                // other bank, before this forward's reuse release to dst
                const int zero_bank = (int)(epoch & 1u) ^ 1;
                unsigned long long* zd = ll_dispatch_peers[pg_rank]
                    + (unsigned long long)((zero_bank * 8 + dst) * LL_MAX_T) * LL_SLOT_WORDS;
                unsigned long long* zc = ll_combine_peers[pg_rank]
                    + (unsigned long long)((zero_bank * 8 + dst) * LL_MAX_T) * LL_ROW_WORDS;
                #pragma unroll 1
                for (int i = lane; i < LL_MAX_T * LL_SLOT_WORDS / 2; i += 32) {
                    asm volatile("st.relaxed.sys.global.v2.u64 [%0], {%1, %1};" :: "l"(zd + 2 * i), "l"(0ull) : "memory");
                }
                #pragma unroll 1
                for (int i = lane; i < LL_MAX_T * LL_ROW_WORDS / 2; i += 32) {
                    asm volatile("st.relaxed.sys.global.v2.u64 [%0], {%1, %1};" :: "l"(zc + 2 * i), "l"(0ull) : "memory");
                }
                __syncwarp();
                asm volatile("fence.proxy.alias;" ::: "memory");
            }
            if (lane == 0) {
                if (dst == 0) {
                    asm volatile("st.relaxed.gpu.global.u32 [%0], %1;" :: "l"(ll_control + 1), "r"(epoch) : "memory");
                }
                // release pattern (buffer reuse, see the LL protocol note): fence, then the reuse epoch
                asm volatile("fence.acq_rel.sys;" ::: "memory");
                asm volatile("st.relaxed.sys.global.u32 [%0], %1;" :: "l"(ll_reuse_peers[dst] + pg_rank), "r"(epoch) : "memory");
            }
        }
        return;
    }
    unsigned long long* slot = ll_dispatch_peers[dst]
        + (unsigned long long)((((int)(epoch & 1u)) * 8 + pg_rank) * LL_MAX_T + token) * LL_SLOT_WORDS;
    int id = -1;
    float weight = 0.0f;
    const int route = token * 8 + lane;
    if (lane < 8) {
        if (ids_is_int64) {
            id = (int)(reinterpret_cast<const long long*>(ids_in)[route]);
        } else {
            id = (int)(long long)(reinterpret_cast<const int*>(ids_in)[route]);
        }
        weight = weights_in[route];
    }
    const int routed = __any_sync(0xffffffff, lane < 8 && (unsigned int)id < 256u && (id >> 5) == dst);
    if (warp == 0 && lane < 8 && split == 0) {
        if (dst == pg_rank) {
            topk_ids[route] = id;
        }
        ll_store2(slot + 2 * lane, (unsigned)id, __float_as_uint(weight), epoch);
    }
    if (routed) {
        // one 16-byte LL vector (two payload words = four BF16) per thread and store: a warp writes 512 contiguous bytes
        const int per = 1792 / splits;
        const int lo = split * per;
        const uint2* src = reinterpret_cast<const uint2*>(hidden_in + (unsigned long long)token * 7168) + lo;
        unsigned long long* row = slot + LL_HDR_WORDS + 2 * lo;
        uint2 v[7];
        #pragma unroll
        for (int i = 0; i < 7; i++) {
            const int j = tid + i * 256;
            if (j < per) {
                v[i] = src[j];
            }
        }
        #pragma unroll
        for (int i = 0; i < 7; i++) {
            const int j = tid + i * 256;
            if (j < per) {
                ll_store2(row + 2 * j, v[i].x, v[i].y, epoch);
            }
        }
    }
}
} // extern "C"

// Route CTA of the LL path: the tables of rank_major_route_cta_ids, built from the ids received in the LL headers.
__device__ __noinline__ void rank_major_route_cta_ll(int active_tokens_per_rank, int32_t pg_rank, const unsigned long long* __restrict__ bank_words, unsigned epoch, int* __restrict__ expert_scatter_offsets, int* __restrict__ token_to_permuted, int* __restrict__ cta_to_expert, int* __restrict__ cta_to_mn_limit, int* __restrict__ expert_padded_row_offsets, int* __restrict__ num_non_exiting_ctas, int* __restrict__ total_padded_rows, int* __restrict__ route_map, __nv_bfloat16* __restrict__ zero_sentinel)
{
    __shared__ int s_rows[32];
    __shared__ int s_base[32];
    __shared__ int s_ids[8 * LL_MAX_T * 8];
    const int tid = threadIdx.x;
    uint32_t lane;
    asm("mov.u32 %0, %%laneid;" : "=r"(lane));
    const int rows_total = 8 * active_tokens_per_rank;
    #pragma unroll 1
    for (int q = tid; q < rows_total * 8; q += 256) {
        const int r = q >> 3;
        const int k = q & 7;
        const int src = r / active_tokens_per_rank;
        const int tok = r - src * active_tokens_per_rank;
        const unsigned long long* w = bank_words + (unsigned long long)(src * LL_MAX_T + tok) * LL_SLOT_WORDS + 2 * k;
        unsigned long long v;
        do {
            asm volatile("ld.relaxed.sys.global.u64 %0, [%1];" : "=l"(v) : "l"(w) : "memory");
        } while ((unsigned)(v >> 32) != epoch);
        s_ids[q] = (int)(unsigned)v;
    }
    if (tid < 32) {
        s_rows[tid] = 0;
    }
    #pragma unroll 1
    for (int elem = tid; elem < 7168; elem += 256) {
        zero_sentinel[elem] = 0.0f;
    }
    __syncthreads();
    const int owner_begin = pg_rank * 32;
    // pass 1: per-expert row counters; token_to_permuted holds (1 << 30 | local << 16 | row) until pass 3
    #pragma unroll 1
    for (int r = tid; r < rows_total; r += 256) {
        const int src = r / active_tokens_per_rank;
        const int tok = r - src * active_tokens_per_rank;
        const int route_base = (src * 128 + tok) * 8;
        #pragma unroll
        for (int k = 0; k < 8; k++) {
            const int global_expert = s_ids[r * 8 + k];
            if (global_expert >= owner_begin && global_expert < owner_begin + 32) {
                const int local = global_expert - owner_begin;
                token_to_permuted[route_base + k] = (1 << 30) | (local << 16) | atomicAdd(&s_rows[local], 1);
            } else {
                token_to_permuted[route_base + k] = 32768;
            }
        }
    }
    __syncthreads();
    if (tid < 32) {
        int expert = lane;
        int rows = s_rows[expert];
        expert_scatter_offsets[expert] = rows;
        int groups = (rows + 64 - 1) / 64;
        if (groups > 16) {
            groups = 16;
        }
        int inclusive = groups;
        #pragma unroll
        for (int delta = 1; delta < 32; delta <<= 1) {
            const int peer = __shfl_up_sync(0xFFFFFFFF, inclusive, delta, 32);
            if ((int)lane >= delta) {
                inclusive = inclusive + peer;
            }
        }
        int write_base = inclusive - groups;
        expert_padded_row_offsets[expert] = write_base * 64;
        s_base[expert] = write_base * 64;
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
    __syncthreads();
    // pass 3: compact rows (the thread re-reads its own pass-1 entries)
    #pragma unroll 1
    for (int r = tid; r < rows_total; r += 256) {
        const int src = r / active_tokens_per_rank;
        const int tok = r - src * active_tokens_per_rank;
        const int route_base = (src * 128 + tok) * 8;
        #pragma unroll
        for (int k = 0; k < 8; k++) {
            const int v = token_to_permuted[route_base + k];
            if (v & (1 << 30)) {
                const int compact_row = s_base[(v >> 16) & 0xff] + (v & 0xffff);
                route_map[compact_row] = route_base / 8;
                token_to_permuted[route_base + k] = compact_row;
            }
        }
    }
}

extern "C" {

__global__ __launch_bounds__(256) void
kernel_rank_major_ll_recv_v2(__nv_bfloat16* __restrict__ recv_hidden, int* __restrict__ recv_local_ids, float* __restrict__ recv_weights, int active_tokens_per_rank, int32_t pg_world, int32_t pg_rank, const unsigned long long* __restrict__ ll_dispatch, unsigned* __restrict__ ll_control, int* __restrict__ expert_scatter_offsets, int* __restrict__ token_to_permuted, int* __restrict__ cta_to_expert, int* __restrict__ cta_to_mn_limit, int* __restrict__ expert_padded_row_offsets, int* __restrict__ num_non_exiting_ctas, int* __restrict__ total_padded_rows, int* __restrict__ route_map, __nv_bfloat16* __restrict__ zero_sentinel, int row_splits)
{
    const int tid = threadIdx.x;
    asm volatile("griddepcontrol.wait;" ::: "memory");
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    unsigned epoch;
    asm volatile("ld.relaxed.gpu.global.u32 %0, [%1];" : "=r"(epoch) : "l"(ll_control + 1) : "memory");
    const unsigned long long* bank_words = ll_dispatch + (unsigned long long)(((int)(epoch & 1u)) * 8 * LL_MAX_T) * LL_SLOT_WORDS;
    const int bid = blockIdx.x;
    if (bid == 0) {
        if (tid == 0) {
            asm volatile("st.relaxed.gpu.global.u32 [%0], %1;" :: "l"(ll_control + 0), "r"(epoch) : "memory");
        }
        rank_major_route_cta_ll(active_tokens_per_rank, pg_rank, bank_words, epoch, expert_scatter_offsets, token_to_permuted, cta_to_expert, cta_to_mn_limit, expert_padded_row_offsets, num_non_exiting_ctas, total_padded_rows, route_map, zero_sentinel);
        return;
    }
    const int unit = bid - 1;
    const int active_token = unit / row_splits;
    const int split = unit - active_token * row_splits;
    const int src_rank = active_token / active_tokens_per_rank;
    const int src_token = active_token - src_rank * active_tokens_per_rank;
    const int recv_token = src_rank * 128 + src_token;
    const unsigned long long* slot = bank_words + (unsigned long long)(src_rank * LL_MAX_T + src_token) * LL_SLOT_WORDS;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    // every warp reads the header itself (lanes 0-7: one {id, weight} pair each) and decides whether the row comes
    int hit = 0;
    if (lane < 8) {
        unsigned long long a, b;
        ll_load2(slot + 2 * lane, a, b);
        while (!ll_ready2(a, b, epoch)) {
            ll_load2(slot + 2 * lane, a, b);
        }
        const int global_expert = (int)(unsigned)a;
        const int owner_begin = pg_rank * 32;
        hit = (global_expert >= owner_begin && global_expert < owner_begin + 32) ? 1 : 0;
        if (warp == 0 && split == 0) {
            const unsigned long long route_index = (unsigned long long)recv_token * 8 + (unsigned long long)lane;
            if (hit) {
                recv_local_ids[route_index] = global_expert - owner_begin;
                recv_weights[route_index] = __uint_as_float((unsigned)b);
            } else {
                recv_local_ids[route_index] = -1;
                recv_weights[route_index] = 0.0f;
            }
        }
    }
    if (!__any_sync(0xffffffff, hit)) {
        return;
    }
    // 1792 16-byte LL vectors (two payload words = four BF16 each) per row; this split polls 1792 / row_splits
    const int per = 1792 / row_splits;
    const int lo = split * per;
    const unsigned long long* row = slot + LL_HDR_WORDS + 2 * lo;
    uint2* out = reinterpret_cast<uint2*>(recv_hidden + (unsigned long long)recv_token * 7168) + lo;
    unsigned long long a[7], b[7];
    #pragma unroll
    for (int k = 0; k < 7; k++) {
        if (tid + 256 * k < per) {
            ll_load2(row + 2 * (tid + 256 * k), a[k], b[k]);
        }
    }
    bool ready;
    do {
        ready = true;
        #pragma unroll
        for (int k = 0; k < 7; k++) {
            if (tid + 256 * k < per && !ll_ready2(a[k], b[k], epoch)) {
                ll_load2(row + 2 * (tid + 256 * k), a[k], b[k]);
                ready = false;
            }
        }
    } while (!ready);
    #pragma unroll
    for (int k = 0; k < 7; k++) {
        if (tid + 256 * k < per) {
            out[tid + 256 * k] = make_uint2((unsigned)a[k], (unsigned)b[k]);
        }
    }
    // recv_hidden is read by FC1 through TMA (async proxy) in a later grid
    asm volatile("fence.proxy.async.global;" ::: "memory");
}

} // extern "C"

#undef FLASHINFER_INF
#undef NUM_MAIN_STAGES
#undef THREADS

#define FLASHINFER_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define THREADS 128

extern "C" {

__global__ __launch_bounds__(128) void
kernel_rank_major_exact_unpermute_v6(__nv_bfloat16* __restrict__ expert_output, float* __restrict__ topk_weights, int* __restrict__ token_to_permuted, unsigned long long* const* __restrict__ ll_combine_peers, unsigned* __restrict__ ll_control, unsigned int hidden_size, int active_tokens_per_rank, int32_t pg_rank)
{
    // The arithmetic of kernel_rank_major_exact_unpermute_v8; the partial row goes as LL words into source src_rank's
    // LL combine buffer slot [bank][this rank][src_token] (not written when no route of the token is owned here).
    const int tid = threadIdx.x;
    __shared__ float weights_stage[8];
    __shared__ int inverse_stage[8];
    const int bid = blockIdx.x;
    int active_token = bid;
    int src_rank = active_token / active_tokens_per_rank;
    int src_token = active_token - src_rank * active_tokens_per_rank;
    unsigned int token = (unsigned int)(src_rank * 128 + src_token);
    unsigned int route_base = token * 8;
    // recv_weights / token_to_permuted come from ll_recv (an earlier grid): read them after griddepcontrol.wait
    asm volatile("griddepcontrol.wait;" ::: "memory");
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    if (tid < 8) {
        weights_stage[tid] = topk_weights[route_base + (unsigned int)tid];
        inverse_stage[tid] = token_to_permuted[route_base + (unsigned int)tid];
    }
    unsigned epoch;
    asm volatile("ld.relaxed.gpu.global.u32 %0, [%1];" : "=r"(epoch) : "l"(ll_control + 1) : "memory");
    __syncthreads();
    float weights[8];
    int rows[8];
    unsigned long long row_offsets[8];
    bool any_local = false;
    #pragma unroll
    for (int route_slot = 0; route_slot < 8; route_slot++) {
        float staged_weight = weights_stage[route_slot];
        __nv_bfloat16 _cvt_bf16_0 = __float2bfloat16(staged_weight);
        float _cvt_f32_0 = __bfloat162float(_cvt_bf16_0);
        weights[route_slot] = _cvt_f32_0;
        rows[route_slot] = inverse_stage[route_slot];
        row_offsets[route_slot] = (unsigned long long)rows[route_slot] * (unsigned long long)hidden_size;
        any_local = any_local || rows[route_slot] != 32768;
    }
    if (!any_local) {
        return;
    }
    unsigned long long* dst = ll_combine_peers[src_rank]
        + (unsigned long long)((((int)(epoch & 1u)) * 8 + pg_rank) * LL_MAX_T + src_token) * LL_ROW_WORDS;
    unsigned int vector_elements = hidden_size / 8 * 8;
    #pragma unroll 1
    for (unsigned int base = blockIdx.y * 1024 + tid * 8; base < vector_elements; base += 1024 * gridDim.y) {
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
            if (rows[route_slot_1] == 32768) {
                // Non-local route: weight 0.0 times the all-+0.0 sentinel row; apply its +0.0 term without the load.
                #pragma unroll
                for (int element = 0; element < 8; element++) {
                    asm volatile("add.ftz.f32 %0, %0, 0f00000000;" : "+f"(accum[element]));
                }
                continue;
            }
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
            const uint32_t* _pw = reinterpret_cast<const uint32_t*>(&_pk[0]);
            // lane l holds LL vectors 2l and 2l+1 of its warp's 64; exchange so each store instruction writes 32
            // consecutive vectors (512 contiguous bytes): vector l from lane l/2, vector 32+l from lane 16+l/2
            const unsigned lane_id = threadIdx.x & 31u;
            const unsigned from_a = lane_id >> 1;
            const unsigned from_b = 16u + (lane_id >> 1);
            const unsigned a0 = __shfl_sync(0xffffffffu, _pw[0], from_a);
            const unsigned a1 = __shfl_sync(0xffffffffu, _pw[1], from_a);
            const unsigned a2 = __shfl_sync(0xffffffffu, _pw[2], from_a);
            const unsigned a3 = __shfl_sync(0xffffffffu, _pw[3], from_a);
            const unsigned b0 = __shfl_sync(0xffffffffu, _pw[0], from_b);
            const unsigned b1 = __shfl_sync(0xffffffffu, _pw[1], from_b);
            const unsigned b2 = __shfl_sync(0xffffffffu, _pw[2], from_b);
            const unsigned b3 = __shfl_sync(0xffffffffu, _pw[3], from_b);
            const bool odd = (lane_id & 1u) != 0u;
            const unsigned warp_vector0 = (base - lane_id * 8u) / 4u;
            ll_store2(dst + 2 * (warp_vector0 + lane_id), odd ? a2 : a0, odd ? a3 : a1, epoch);
            ll_store2(dst + 2 * (warp_vector0 + 32u + lane_id), odd ? b2 : b0, odd ? b3 : b1, epoch);
        }
    }
}

} // extern "C"

extern "C" {

__global__ __launch_bounds__(160) void
kernel_rank_major_combine_v6(__nv_bfloat16* __restrict__ output, int32_t pg_rank, const unsigned long long* __restrict__ ll_combine, unsigned* __restrict__ ll_control, const int* __restrict__ topk_ids, const unsigned* __restrict__ ll_reuse)
{
    // One CTA per (token, 1024-column chunk): poll every routed owner's LL partial words, then the FP32 sum in owner
    // order of kernel_rank_major_combine_v3 (+0.0 for owners without a route of this token).
    const int tid = threadIdx.x;
    asm volatile("griddepcontrol.wait;" ::: "memory");
    const int token = blockIdx.x;
    const int chunk = blockIdx.y;
    unsigned epoch;
    asm volatile("ld.relaxed.gpu.global.u32 %0, [%1];" : "=r"(epoch) : "l"(ll_control + 1) : "memory");
    if (tid >= 128) {
        // warp 4: acquire pattern (buffer reuse, see the LL protocol note) of every peer's release of this forward,
        // CTA (0, 0) only, off the combine's data path
        if (token == 0 && chunk == 0 && tid < 136) {
            unsigned seen;
            do {
                asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(seen) : "l"(ll_reuse + (tid - 128)) : "memory");
            } while ((int)(seen - epoch) < 0);
        }
        return;
    }
    unsigned int owners = 0u;
    #pragma unroll
    for (int slot = 0; slot < 8; slot++) {
        const int expert = topk_ids[token * 8 + slot];
        if ((unsigned int)expert < 256u) {
            owners |= 1u << (expert >> 5);
        }
    }
    const unsigned long long* words = ll_combine
        + (unsigned long long)(((int)(epoch & 1u)) * 8 * LL_MAX_T + token) * LL_ROW_WORDS + chunk * 512 + tid * 4;
    unsigned long long v[8][4];
    #pragma unroll
    for (int peer = 0; peer < 8; peer++) {
        if ((owners >> peer) & 1u) {
            const unsigned long long* p = words + (unsigned long long)peer * LL_MAX_T * LL_ROW_WORDS;
            ll_load2(p, v[peer][0], v[peer][1]);
            ll_load2(p + 2, v[peer][2], v[peer][3]);
        }
    }
    bool ready;
    do {
        ready = true;
        #pragma unroll
        for (int peer = 0; peer < 8; peer++) {
            if ((owners >> peer) & 1u) {
                if (!ll_ready2(v[peer][0], v[peer][1], epoch) || !ll_ready2(v[peer][2], v[peer][3], epoch)) {
                    const unsigned long long* p = words + (unsigned long long)peer * LL_MAX_T * LL_ROW_WORDS;
                    ll_load2(p, v[peer][0], v[peer][1]);
                    ll_load2(p + 2, v[peer][2], v[peer][3]);
                    ready = false;
                }
            }
        }
    } while (!ready);
    float acc[8];
    #pragma unroll
    for (int element = 0; element < 8; element++) {
        acc[element] = 0.0f;
    }
    #pragma unroll
    for (int peer = 0; peer < 8; peer++) {
        if ((owners >> peer) & 1u) {
            unsigned int packed[4];
            packed[0] = (unsigned int)v[peer][0];
            packed[1] = (unsigned int)v[peer][1];
            packed[2] = (unsigned int)v[peer][2];
            packed[3] = (unsigned int)v[peer][3];
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
                acc[element] = acc[element] + packed_f32[element];
            }
        } else {
            // The owner holds no route of this token: its partial row is all +0.0. Add that +0.0 explicitly.
            #pragma unroll
            for (int element = 0; element < 8; element++) {
                asm volatile("add.ftz.f32 %0, %0, 0f00000000;" : "+f"(acc[element]));
            }
        }
    }
    {
        __nv_bfloat162 _pk[4];
        _pk[0] = __floats2bfloat162_rn(acc[0], acc[1]);
        _pk[1] = __floats2bfloat162_rn(acc[2], acc[3]);
        _pk[2] = __floats2bfloat162_rn(acc[4], acc[5]);
        _pk[3] = __floats2bfloat162_rn(acc[6], acc[7]);
        *reinterpret_cast<uint4*>(output + ((unsigned long long)token * 7168 + (unsigned long long)chunk * 1024 + (unsigned long long)tid * 8)) = *reinterpret_cast<uint4*>(&_pk[0]);
    }
}

} // extern "C"

#undef FLASHINFER_INF
#undef NUM_MAIN_STAGES
#undef THREADS
