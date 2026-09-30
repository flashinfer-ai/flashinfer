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

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <math_constants.h>

namespace nvfp4_qualified_c388_up {
#define kernel_alpha_moe_nvfp4_up_workspace_contiguous_gate_up_two_consumer kernel_alpha_moe_nvfp4_up_workspace_contiguous_gate_up_two_consumer_nvfp4_qualified_c388_up
#define LOOM_INF CUDART_INF_F
#define TMEM_NCOLS 176
#define TMEM_UP_ACC_OFFSET 0
#define TMEM_UP_GATE_SF_OFFSET 128
#define TMEM_UP_UP_SF_OFFSET 144
#define TMEM_UP_X_SF_OFFSET 160
#define NUM_UP_PIPE_STAGES 2
#define NUM_READY_PIPE_STAGES 2
#define SMEM_SMEM_W1_OFF 1024
#define SMEM_SMEM_W1_STAGE_BYTES 32768
#define SMEM_SMEM_W1_STRIDE 43008
#define SMEM_SMEM_X_OFF 33792
#define SMEM_SMEM_X_STAGE_BYTES 2048
#define SMEM_SMEM_X_STRIDE 43008
#define SMEM_SMEM_W1_GATE_SF_OFF 37888
#define SMEM_SMEM_W1_GATE_SF_STAGE_BYTES 2048
#define SMEM_SMEM_W1_GATE_SF_STRIDE 43008
#define SMEM_SMEM_W1_UP_SF_OFF 39936
#define SMEM_SMEM_W1_UP_SF_STAGE_BYTES 2048
#define SMEM_SMEM_W1_UP_SF_STRIDE 43008
#define SMEM_SMEM_X_SF_OFF 41984
#define SMEM_SMEM_X_SF_STAGE_BYTES 2048
#define SMEM_SMEM_X_SF_STRIDE 43008
#define SMEM_SMEM_X1_OFF 35840
#define SMEM_SMEM_X1_STAGE_BYTES 2048
#define SMEM_SMEM_X1_STRIDE 43008
#define SMEM_SMEM_X_WIDE_OFF 33792
#define SMEM_SMEM_X_WIDE_STAGE_BYTES 4096
#define SMEM_SMEM_X_WIDE_STRIDE 43008
#define SMEM_SMEM_ACT_SCALE_OFF 87040
#define SMEM_SMEM_ACT_SCALE_STAGE_BYTES 512
#define SMEM_SMEM_ACT_SCALE_STRIDE 512
#define SMEM_TOTAL 87552
#define THREADS 256
#define PACKED_SCALE_LOADS 1

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


__device__ __forceinline__ void tcgen05_mma_mxf4nvf4_bs(
    int taddr, uint64_t a_desc, uint64_t b_desc, uint32_t i_desc,
    int sfa_taddr, int sfb_taddr, int enable_input_d) {
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, %6, 0;\n\t"
        "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X"
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
        "@leader tcgen05.mma.cta_group::1.kind::mxf4nvf4 [%2], da, db, %3, p;\n\t"
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


__device__ __forceinline__ uint64_t make_sf_cp_desc_sbo512(int addr) {
    const int SBO = 512;
    return desc_encode(addr)
         | (desc_encode(SBO) << 32ULL)
         | (1ULL << 46ULL);
}


__device__ __forceinline__ uint64_t make_sf_cp_desc_lo_sbo512(int lo) {
    const int SBO = 512;
    return (uint64_t)(uint32_t)lo
         | (desc_encode(SBO) << 32ULL)
         | (1ULL << 46ULL);
}


__device__ __forceinline__ void tcgen05_cp_32x128b_warpx4(
    int taddr, uint64_t s_desc) {
    asm volatile(
        "tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;"
        :: "r"(taddr), "l"(s_desc));
}


__device__ __forceinline__ uint64_t make_smem_desc(int addr) {
    const int SBO = 1024;
    return desc_encode(addr)
         | (desc_encode(SBO) << 32ULL)
         | (1ULL << 46ULL)
         | (2ULL << 61ULL);
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


__device__ __forceinline__ void tma_2d_gmem2smem(
    int dst, const void *tmap_ptr, int x, int y, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global"
        ".mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3}], [%4];"
        :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y),
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


__device__ __forceinline__ void tcgen05_commit(int mbar_addr) {
    asm volatile(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one"
        ".shared::cluster.b64 [%0];"
        :: "r"(mbar_addr) : "memory");
}


__device__ __forceinline__ void tmem_ld_x16(float* dst, int tmem_addr) {
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x16.b32"
        " {%0, %1, %2, %3, %4, %5, %6, %7,"
        "  %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
        : "=f"(dst[0]),  "=f"(dst[1]),  "=f"(dst[2]),  "=f"(dst[3]),
          "=f"(dst[4]),  "=f"(dst[5]),  "=f"(dst[6]),  "=f"(dst[7]),
          "=f"(dst[8]),  "=f"(dst[9]),  "=f"(dst[10]), "=f"(dst[11]),
          "=f"(dst[12]), "=f"(dst[13]), "=f"(dst[14]), "=f"(dst[15])
        : "r"(tmem_addr));
}


__device__ __forceinline__ void tmem_ld_x16_wait(float* dst, int addr) {
    tmem_ld_x16(dst, addr);
    asm volatile("tcgen05.wait::ld.sync.aligned;");
}


__device__ __forceinline__ uint32_t make_warp_uniform(uint32_t val) {
    uint32_t result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1f, 0xffffffff;"
        : "=r"(result) : "r"(val));
    return result;
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

__global__ __launch_bounds__(256, 1) void
kernel_alpha_moe_nvfp4_up_workspace_contiguous_gate_up_two_consumer(const __grid_constant__ CUtensorMap x, const __grid_constant__ CUtensorMap W1, uint8_t* __restrict__ x_scale, const __grid_constant__ CUtensorMap w1_scale_prepared, float* __restrict__ output1_scale_gate_scalar, float* __restrict__ output1_scale_scalar, int* __restrict__ sorted_token_ids, int* __restrict__ expert_ids, int* __restrict__ num_tokens_post_padded, int* __restrict__ compact_owner_plan, int* __restrict__ compact_owner_count, uint8_t* __restrict__ act_workspace, uint8_t* __restrict__ sf_workspace, int M, int K, int top_k, int route_block_m, int intermediate_blocks_total)
{
    const int tid = threadIdx.x;
    const uint32_t warp = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
    uint32_t lane;
    asm("mov.u32 %0, %%laneid;" : "=r"(lane));

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int mbar_base = smem;
    #define up_full_addr (mbar_base + 0)
    #define up_free_addr (mbar_base + 16)
    #define up_ready_addr (mbar_base + 32)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // Kernel setup ops
    uint8_t* smem_w1 = reinterpret_cast<uint8_t*>(smem_raw + 1024);
    const int smem_w1_addr = smem + 1024;
    uint8_t* smem_x = reinterpret_cast<uint8_t*>(smem_raw + 33792);
    const int smem_x_addr = smem + 33792;
    uint8_t* smem_w1_gate_sf = reinterpret_cast<uint8_t*>(smem_raw + 37888);
    const int smem_w1_gate_sf_addr = smem + 37888;
    uint8_t* smem_w1_up_sf = reinterpret_cast<uint8_t*>(smem_raw + 39936);
    const int smem_w1_up_sf_addr = smem + 39936;
    uint8_t* smem_x_sf = reinterpret_cast<uint8_t*>(smem_raw + 41984);
    const int smem_x_sf_addr = smem + 41984;
    uint8_t* smem_x1 = reinterpret_cast<uint8_t*>(smem_raw + 35840);
    const int smem_x1_addr = smem + 35840;
    uint8_t* smem_x_wide = reinterpret_cast<uint8_t*>(smem_raw + 33792);
    const int smem_x_wide_addr = smem + 33792;
    float* smem_act_scale = reinterpret_cast<float*>(smem_raw + 87040);
    const int smem_act_scale_addr = smem + 87040;

    // Mbarrier init (3 pipeline groups, 0 ordered-sequence groups, 6 barriers)
    // Mbarriers at smem_raw[0..48)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'up_pipe' ---
            // up_full: 2 barriers, init_count=3
            mbarrier_init(smem + 0, 3);
            mbarrier_init(smem + 8, 3);
            // up_free: 2 barriers, init_count=1
            mbarrier_init(smem + 16, 1);
            mbarrier_init(smem + 24, 1);
            // --- pipeline 'ready_pipe' ---
            // up_ready: 2 barriers, init_count=1
            mbarrier_init(smem + 32, 1);
            mbarrier_init(smem + 40, 1);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    // TMEM alloc (256 columns, 176 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 48);
    if (warp == 0) {
        int _tmem_hold = smem + 48;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(256) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_up_acc = taddr;
    const int tmem_up_gate_sf = taddr + 128;
    const int tmem_up_up_sf = taddr + 144;
    const int tmem_up_x_sf = taddr + 160;

    // ---- Role: load ----
    if (warp == 0) {
        { // load_main
            bool owns_route_l = blockIdx.x < compact_owner_count[0];
            int original_tile_l = 0;
            int route_base_l = 0;
            bool pair_upper_l = 0;
            bool has_second_l = 0;
            bool second_upper_l = 0;
            int route_expert_l = 0;
            if (owns_route_l) {
                original_tile_l = compact_owner_plan[blockIdx.x * 3];
                route_base_l = original_tile_l * 8;
                route_expert_l = compact_owner_plan[blockIdx.x * 3 + 1];
                int live_subtiles_l = compact_owner_plan[blockIdx.x * 3 + 2];
                pair_upper_l = live_subtiles_l >= 2;
                has_second_l = live_subtiles_l >= 3;
                second_upper_l = live_subtiles_l == 4;
            }
            unsigned int _phase_up_free = 1;
            if (owns_route_l) {
                int intermediate_blocks = intermediate_blocks_total;
                int expert = route_expert_l;
                unsigned int up_stage = 0;
                #pragma unroll 1
                for (int feature_load = 0; feature_load < 2; feature_load++) {
                    int intermediate_block = 2 * blockIdx.y + feature_load;
                    #pragma unroll 1
                    for (int kb = 0; kb < K / 256; kb++) {
                        mbarrier_wait(up_free_addr + (up_stage) * 8, _phase_up_free);
                        if (elect_sync()) {
                            mbarrier_arrive_expect_tx(up_full_addr + (up_stage) * 8, 36864);
                            tma_3d_gmem2smem(smem_w1_gate_sf_addr + up_stage * 43008, (&w1_scale_prepared), 0, 0, (expert * intermediate_blocks + intermediate_block) * (K / 256) + kb, up_full_addr + (up_stage) * 8);
                            tma_3d_gmem2smem(smem_w1_addr + up_stage * 43008, (&W1), 0, 0, (expert * intermediate_blocks + intermediate_block) * (K / 256) + kb, up_full_addr + (up_stage) * 8);
                        }
                        up_stage += 1;
                        if (up_stage == 2) { up_stage = 0; _phase_up_free ^= 1; }
                    }
                }
                asm volatile("barrier.sync 14, 256;" ::: "memory");
            }
        }
    }
    // ---- Role: mma ----
    if (warp == 1) {
        { // mma_main
            bool owns_route_m = blockIdx.x < compact_owner_count[0];
            int original_tile_m = 0;
            int route_base_m = 0;
            bool pair_upper_m = 0;
            bool has_second_m = 0;
            bool second_upper_m = 0;
            int route_expert_m = 0;
            if (owns_route_m) {
                original_tile_m = compact_owner_plan[blockIdx.x * 3];
                route_base_m = original_tile_m * 8;
                route_expert_m = compact_owner_plan[blockIdx.x * 3 + 1];
                int live_subtiles_m = compact_owner_plan[blockIdx.x * 3 + 2];
                pair_upper_m = live_subtiles_m >= 2;
                has_second_m = live_subtiles_m >= 3;
                second_upper_m = live_subtiles_m == 4;
            }
            unsigned int _phase_up_full = 0;
            if (owns_route_m) {
                unsigned int up_stage_mma = 0;
                #pragma unroll 1
                for (int feature_mma = 0; feature_mma < 2; feature_mma++) {
                    if (has_second_m) {
                        #pragma unroll 1
                        for (int kb_mma = 0; kb_mma < K / 256; kb_mma++) {
                            mbarrier_wait(up_full_addr + (up_stage_mma) * 8, _phase_up_full);
                            asm volatile("tcgen05.fence::after_thread_sync;");
                            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                            if (elect_sync()) {
                                tcgen05_cp_32x128b_warpx4(tmem_up_gate_sf, make_sf_cp_desc_lo_sbo512((((smem_w1_gate_sf_addr) >> 4) + (up_stage_mma) * 2688)));
                                tcgen05_cp_32x128b_warpx4((tmem_up_gate_sf + 4), make_sf_cp_desc_lo_sbo512((((smem_w1_gate_sf_addr) >> 4) + (up_stage_mma) * 2688 + 8)));
                                tcgen05_cp_32x128b_warpx4((tmem_up_gate_sf + 8), make_sf_cp_desc_lo_sbo512((((smem_w1_gate_sf_addr) >> 4) + (up_stage_mma) * 2688 + 16)));
                                tcgen05_cp_32x128b_warpx4((tmem_up_gate_sf + 12), make_sf_cp_desc_lo_sbo512((((smem_w1_gate_sf_addr) >> 4) + (up_stage_mma) * 2688 + 24)));
                            }
                            if (elect_sync()) {
                                tcgen05_cp_32x128b_warpx4(tmem_up_up_sf, make_sf_cp_desc_lo_sbo512((((smem_w1_up_sf_addr) >> 4) + (up_stage_mma) * 2688)));
                                tcgen05_cp_32x128b_warpx4((tmem_up_up_sf + 4), make_sf_cp_desc_lo_sbo512((((smem_w1_up_sf_addr) >> 4) + (up_stage_mma) * 2688 + 8)));
                                tcgen05_cp_32x128b_warpx4((tmem_up_up_sf + 8), make_sf_cp_desc_lo_sbo512((((smem_w1_up_sf_addr) >> 4) + (up_stage_mma) * 2688 + 16)));
                                tcgen05_cp_32x128b_warpx4((tmem_up_up_sf + 12), make_sf_cp_desc_lo_sbo512((((smem_w1_up_sf_addr) >> 4) + (up_stage_mma) * 2688 + 24)));
                            }
                            if (elect_sync()) {
                                tcgen05_cp_32x128b_warpx4(tmem_up_x_sf, make_sf_cp_desc_lo_sbo512((((smem_x_sf_addr) >> 4) + (up_stage_mma) * 2688)));
                                tcgen05_cp_32x128b_warpx4((tmem_up_x_sf + 4), make_sf_cp_desc_lo_sbo512((((smem_x_sf_addr) >> 4) + (up_stage_mma) * 2688 + 8)));
                                tcgen05_cp_32x128b_warpx4((tmem_up_x_sf + 8), make_sf_cp_desc_lo_sbo512((((smem_x_sf_addr) >> 4) + (up_stage_mma) * 2688 + 16)));
                                tcgen05_cp_32x128b_warpx4((tmem_up_x_sf + 12), make_sf_cp_desc_lo_sbo512((((smem_x_sf_addr) >> 4) + (up_stage_mma) * 2688 + 24)));
                            }
                            int init_up = ((kb_mma == 0) ? 1 : 0);
                            int _mma_a_lo_0 = make_warp_uniform((((smem_w1_addr) >> 4) & 0x3FFF) + (up_stage_mma) * 2688);
                            int _mma_b_lo_0 = make_warp_uniform((((smem_x_wide_addr) >> 4) & 0x3FFF) + (up_stage_mma) * 2688);
                            if (elect_sync()) {
                                {
                                    uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_0) | ((uint64_t)0x40004040 << 32);
                                    uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_0) | ((uint64_t)0x40004040 << 32);

                                    tcgen05_mma_mxf4nvf4_bs((tmem_up_acc + (feature_mma * 4 * 16)), a_desc + 0, b_desc + 0,
                                        0x8080480U, tmem_up_gate_sf + 0, tmem_up_x_sf + 0, ((init_up) ? 0 : 1));
                                    tcgen05_mma_mxf4nvf4_bs((tmem_up_acc + (feature_mma * 4 * 16)), a_desc + 2, b_desc + 2,
                                        0x8080480U, tmem_up_gate_sf + 4, tmem_up_x_sf + 4, 1);
                                    tcgen05_mma_mxf4nvf4_bs((tmem_up_acc + (feature_mma * 4 * 16)), a_desc + 4, b_desc + 4,
                                        0x8080480U, tmem_up_gate_sf + 8, tmem_up_x_sf + 8, 1);
                                    tcgen05_mma_mxf4nvf4_bs((tmem_up_acc + (feature_mma * 4 * 16)), a_desc + 6, b_desc + 6,
                                        0x8080480U, tmem_up_gate_sf + 12, tmem_up_x_sf + 12, 1);
                                }
                            }
                            int _mma_a_lo_1 = make_warp_uniform((((smem_w1_addr + 16384) >> 4) & 0x3FFF) + (up_stage_mma) * 2688);
                            int _mma_b_lo_1 = make_warp_uniform((((smem_x_wide_addr) >> 4) & 0x3FFF) + (up_stage_mma) * 2688);
                            if (elect_sync()) {
                                {
                                    uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_1) | ((uint64_t)0x40004040 << 32);
                                    uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_1) | ((uint64_t)0x40004040 << 32);

                                    tcgen05_mma_mxf4nvf4_bs((tmem_up_acc + (feature_mma * 4 * 16 + 32)), a_desc + 0, b_desc + 0,
                                        0x8080480U, tmem_up_up_sf + 0, tmem_up_x_sf + 0, ((init_up) ? 0 : 1));
                                    tcgen05_mma_mxf4nvf4_bs((tmem_up_acc + (feature_mma * 4 * 16 + 32)), a_desc + 2, b_desc + 2,
                                        0x8080480U, tmem_up_up_sf + 4, tmem_up_x_sf + 4, 1);
                                    tcgen05_mma_mxf4nvf4_bs((tmem_up_acc + (feature_mma * 4 * 16 + 32)), a_desc + 4, b_desc + 4,
                                        0x8080480U, tmem_up_up_sf + 8, tmem_up_x_sf + 8, 1);
                                    tcgen05_mma_mxf4nvf4_bs((tmem_up_acc + (feature_mma * 4 * 16 + 32)), a_desc + 6, b_desc + 6,
                                        0x8080480U, tmem_up_up_sf + 12, tmem_up_x_sf + 12, 1);
                                }
                            }
                            elect_commit(up_free_addr + (up_stage_mma) * 8);
                            up_stage_mma += 1;
                            if (up_stage_mma == 2) { up_stage_mma = 0; _phase_up_full ^= 1; }
                        }
                    } else {
                        #pragma unroll 1
                        for (int kb_mma_1 = 0; kb_mma_1 < K / 256; kb_mma_1++) {
                            mbarrier_wait(up_full_addr + (up_stage_mma) * 8, _phase_up_full);
                            asm volatile("tcgen05.fence::after_thread_sync;");
                            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                            if (elect_sync()) {
                                tcgen05_cp_32x128b_warpx4(tmem_up_gate_sf, make_sf_cp_desc_lo_sbo512((((smem_w1_gate_sf_addr) >> 4) + (up_stage_mma) * 2688)));
                                tcgen05_cp_32x128b_warpx4((tmem_up_gate_sf + 4), make_sf_cp_desc_lo_sbo512((((smem_w1_gate_sf_addr) >> 4) + (up_stage_mma) * 2688 + 8)));
                                tcgen05_cp_32x128b_warpx4((tmem_up_gate_sf + 8), make_sf_cp_desc_lo_sbo512((((smem_w1_gate_sf_addr) >> 4) + (up_stage_mma) * 2688 + 16)));
                                tcgen05_cp_32x128b_warpx4((tmem_up_gate_sf + 12), make_sf_cp_desc_lo_sbo512((((smem_w1_gate_sf_addr) >> 4) + (up_stage_mma) * 2688 + 24)));
                            }
                            if (elect_sync()) {
                                tcgen05_cp_32x128b_warpx4(tmem_up_up_sf, make_sf_cp_desc_lo_sbo512((((smem_w1_up_sf_addr) >> 4) + (up_stage_mma) * 2688)));
                                tcgen05_cp_32x128b_warpx4((tmem_up_up_sf + 4), make_sf_cp_desc_lo_sbo512((((smem_w1_up_sf_addr) >> 4) + (up_stage_mma) * 2688 + 8)));
                                tcgen05_cp_32x128b_warpx4((tmem_up_up_sf + 8), make_sf_cp_desc_lo_sbo512((((smem_w1_up_sf_addr) >> 4) + (up_stage_mma) * 2688 + 16)));
                                tcgen05_cp_32x128b_warpx4((tmem_up_up_sf + 12), make_sf_cp_desc_lo_sbo512((((smem_w1_up_sf_addr) >> 4) + (up_stage_mma) * 2688 + 24)));
                            }
                            if (elect_sync()) {
                                tcgen05_cp_32x128b_warpx4(tmem_up_x_sf, make_sf_cp_desc_lo_sbo512((((smem_x_sf_addr) >> 4) + (up_stage_mma) * 2688)));
                                tcgen05_cp_32x128b_warpx4((tmem_up_x_sf + 4), make_sf_cp_desc_lo_sbo512((((smem_x_sf_addr) >> 4) + (up_stage_mma) * 2688 + 8)));
                                tcgen05_cp_32x128b_warpx4((tmem_up_x_sf + 8), make_sf_cp_desc_lo_sbo512((((smem_x_sf_addr) >> 4) + (up_stage_mma) * 2688 + 16)));
                                tcgen05_cp_32x128b_warpx4((tmem_up_x_sf + 12), make_sf_cp_desc_lo_sbo512((((smem_x_sf_addr) >> 4) + (up_stage_mma) * 2688 + 24)));
                            }
                            int init_up_1 = ((kb_mma_1 == 0) ? 1 : 0);
                            int _mma_a_lo_2 = make_warp_uniform((((smem_w1_addr) >> 4) & 0x3FFF) + (up_stage_mma) * 2688);
                            int _mma_b_lo_2 = make_warp_uniform((((smem_x_addr) >> 4) & 0x3FFF) + (up_stage_mma) * 2688);
                            if (elect_sync()) {
                                {
                                    uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_2) | ((uint64_t)0x40004040 << 32);
                                    uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_2) | ((uint64_t)0x40004040 << 32);

                                    tcgen05_mma_mxf4nvf4_bs((tmem_up_acc + (feature_mma * 4 * 16)), a_desc + 0, b_desc + 0,
                                        0x8040480U, tmem_up_gate_sf + 0, tmem_up_x_sf + 0, ((init_up_1) ? 0 : 1));
                                    tcgen05_mma_mxf4nvf4_bs((tmem_up_acc + (feature_mma * 4 * 16)), a_desc + 2, b_desc + 2,
                                        0x8040480U, tmem_up_gate_sf + 4, tmem_up_x_sf + 4, 1);
                                    tcgen05_mma_mxf4nvf4_bs((tmem_up_acc + (feature_mma * 4 * 16)), a_desc + 4, b_desc + 4,
                                        0x8040480U, tmem_up_gate_sf + 8, tmem_up_x_sf + 8, 1);
                                    tcgen05_mma_mxf4nvf4_bs((tmem_up_acc + (feature_mma * 4 * 16)), a_desc + 6, b_desc + 6,
                                        0x8040480U, tmem_up_gate_sf + 12, tmem_up_x_sf + 12, 1);
                                }
                            }
                            int _mma_a_lo_3 = make_warp_uniform((((smem_w1_addr + 16384) >> 4) & 0x3FFF) + (up_stage_mma) * 2688);
                            int _mma_b_lo_3 = make_warp_uniform((((smem_x_addr) >> 4) & 0x3FFF) + (up_stage_mma) * 2688);
                            if (elect_sync()) {
                                {
                                    uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_3) | ((uint64_t)0x40004040 << 32);
                                    uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_3) | ((uint64_t)0x40004040 << 32);

                                    tcgen05_mma_mxf4nvf4_bs((tmem_up_acc + (feature_mma * 4 * 16 + 32)), a_desc + 0, b_desc + 0,
                                        0x8040480U, tmem_up_up_sf + 0, tmem_up_x_sf + 0, ((init_up_1) ? 0 : 1));
                                    tcgen05_mma_mxf4nvf4_bs((tmem_up_acc + (feature_mma * 4 * 16 + 32)), a_desc + 2, b_desc + 2,
                                        0x8040480U, tmem_up_up_sf + 4, tmem_up_x_sf + 4, 1);
                                    tcgen05_mma_mxf4nvf4_bs((tmem_up_acc + (feature_mma * 4 * 16 + 32)), a_desc + 4, b_desc + 4,
                                        0x8040480U, tmem_up_up_sf + 8, tmem_up_x_sf + 8, 1);
                                    tcgen05_mma_mxf4nvf4_bs((tmem_up_acc + (feature_mma * 4 * 16 + 32)), a_desc + 6, b_desc + 6,
                                        0x8040480U, tmem_up_up_sf + 12, tmem_up_x_sf + 12, 1);
                                }
                            }
                            elect_commit(up_free_addr + (up_stage_mma) * 8);
                            up_stage_mma += 1;
                            if (up_stage_mma == 2) { up_stage_mma = 0; _phase_up_full ^= 1; }
                        }
                    }
                    elect_commit(up_ready_addr + (feature_mma) * 8);
                }
                asm volatile("barrier.sync 14, 256;" ::: "memory");
            }
        }
    }
    // ---- Role: consumer ----
    if (warp >= 2 && warp <= 3) {
        { // consumer_main
            bool owns_route_c = blockIdx.x < compact_owner_count[0];
            int original_tile_c = 0;
            int route_base_c = 0;
            bool pair_upper_c = 0;
            bool has_second_c = 0;
            bool second_upper_c = 0;
            int route_expert_c = 0;
            if (owns_route_c) {
                original_tile_c = compact_owner_plan[blockIdx.x * 3];
                route_base_c = original_tile_c * 8;
                route_expert_c = compact_owner_plan[blockIdx.x * 3 + 1];
                int live_subtiles_c = compact_owner_plan[blockIdx.x * 3 + 2];
                pair_upper_c = live_subtiles_c >= 2;
                has_second_c = live_subtiles_c >= 3;
                second_upper_c = live_subtiles_c == 4;
            }
            unsigned int _phase_up_free_1 = 1;
            unsigned int _phase_up_free_2 = 1;
            if (owns_route_c) {
                int expert_c = route_expert_c;
                const int consumer_warp = warp - 2;
                const int physical_feature = (unsigned int)(consumer_warp * 32) + lane;
                int half_count_c = ((has_second_c) ? 2 : 1);
                int sf_cols_c = K / 16;
                int sf_lane_row_c = (unsigned int)(consumer_warp * 32) + lane / 4;
                int sf_lane_token_lo_c = 0;
                int sf_lane_token_hi_c = 0;
                if (sf_lane_row_c < 8) {
                    int sf_lane_pair_lo_c = sorted_token_ids[route_base_c + sf_lane_row_c];
                    int sf_lane_pair_hi_c = M * top_k;
                    if (pair_upper_c) {
                        sf_lane_pair_hi_c = sorted_token_ids[route_base_c + 8 + sf_lane_row_c];
                    }
                    int _min_0 = ((sf_lane_pair_lo_c / top_k) < (M - 1) ? (sf_lane_pair_lo_c / top_k) : (M - 1));
                    sf_lane_token_lo_c = _min_0;
                    int _min_1 = ((sf_lane_pair_hi_c / top_k) < (M - 1) ? (sf_lane_pair_hi_c / top_k) : (M - 1));
                    sf_lane_token_hi_c = _min_1;
                }
                int sf_second_token_lo_c = 0;
                int sf_second_token_hi_c = 0;
                if (has_second_c) {
                    if (sf_lane_row_c < 8) {
                        int sf_second_pair_lo_c = sorted_token_ids[route_base_c + 16 + sf_lane_row_c];
                        int sf_second_pair_hi_c = M * top_k;
                        if (second_upper_c) {
                            sf_second_pair_hi_c = sorted_token_ids[route_base_c + 24 + sf_lane_row_c];
                        }
                        int _min_2 = ((sf_second_pair_lo_c / top_k) < (M - 1) ? (sf_second_pair_lo_c / top_k) : (M - 1));
                        sf_second_token_lo_c = _min_2;
                        int _min_3 = ((sf_second_pair_hi_c / top_k) < (M - 1) ? (sf_second_pair_hi_c / top_k) : (M - 1));
                        sf_second_token_hi_c = _min_3;
                    }
                }
                int _min_4 = ((2) < (K / 256) ? (2) : (K / 256));
                #pragma unroll 1
                for (int zero_stage = 0; zero_stage < _min_4; zero_stage++) {
                    int x_zero_base = smem_x_sf_addr + (unsigned int)(zero_stage * 43008);
                    int zero_sf_row = (unsigned int)(consumer_warp * 32) + lane / 4;
                    if (zero_sf_row >= half_count_c * 16) {
                        int zero_sf_c = zero_sf_row % 32 / 8;
                        int zero_sf_d = zero_sf_row % 8;
                        int zero_sf_g = zero_sf_row / 32;
                        int zero_kset = lane % 4;
                        int zero_sf_dst = ((zero_sf_c * 4 + zero_kset) * 8 + zero_sf_d) * 16 + zero_sf_g * 4;
                        unsigned int zero_word = 0;
                        asm volatile("st.shared.b32 [%0], %1;" :: "r"(x_zero_base + zero_sf_dst), "r"(zero_word));
                    }
                    int zero_sf_row_0 = (unsigned int)(consumer_warp * 32 + 8) + lane / 4;
                    if (zero_sf_row_0 >= half_count_c * 16) {
                        int zero_sf_c_1 = zero_sf_row_0 % 32 / 8;
                        int zero_sf_d_1 = zero_sf_row_0 % 8;
                        int zero_sf_g_1 = zero_sf_row_0 / 32;
                        int zero_kset_1 = lane % 4;
                        int zero_sf_dst_1 = ((zero_sf_c_1 * 4 + zero_kset_1) * 8 + zero_sf_d_1) * 16 + zero_sf_g_1 * 4;
                        unsigned int zero_word_1 = 0;
                        asm volatile("st.shared.b32 [%0], %1;" :: "r"(x_zero_base + zero_sf_dst_1), "r"(zero_word_1));
                    }
                    int zero_sf_row_1 = (unsigned int)(consumer_warp * 32 + 16) + lane / 4;
                    if (zero_sf_row_1 >= half_count_c * 16) {
                        int zero_sf_c_2 = zero_sf_row_1 % 32 / 8;
                        int zero_sf_d_2 = zero_sf_row_1 % 8;
                        int zero_sf_g_2 = zero_sf_row_1 / 32;
                        int zero_kset_2 = lane % 4;
                        int zero_sf_dst_2 = ((zero_sf_c_2 * 4 + zero_kset_2) * 8 + zero_sf_d_2) * 16 + zero_sf_g_2 * 4;
                        unsigned int zero_word_2 = 0;
                        asm volatile("st.shared.b32 [%0], %1;" :: "r"(x_zero_base + zero_sf_dst_2), "r"(zero_word_2));
                    }
                    int zero_sf_row_2 = (unsigned int)(consumer_warp * 32 + 24) + lane / 4;
                    if (zero_sf_row_2 >= half_count_c * 16) {
                        int zero_sf_c_3 = zero_sf_row_2 % 32 / 8;
                        int zero_sf_d_3 = zero_sf_row_2 % 8;
                        int zero_sf_g_3 = zero_sf_row_2 / 32;
                        int zero_kset_3 = lane % 4;
                        int zero_sf_dst_3 = ((zero_sf_c_3 * 4 + zero_kset_3) * 8 + zero_sf_d_3) * 16 + zero_sf_g_3 * 4;
                        unsigned int zero_word_3 = 0;
                        asm volatile("st.shared.b32 [%0], %1;" :: "r"(x_zero_base + zero_sf_dst_3), "r"(zero_word_3));
                    }
                    int zero_sf_row_3 = (unsigned int)((consumer_warp + 2) * 32) + lane / 4;
                    if (zero_sf_row_3 >= half_count_c * 16) {
                        int zero_sf_c_4 = zero_sf_row_3 % 32 / 8;
                        int zero_sf_d_4 = zero_sf_row_3 % 8;
                        int zero_sf_g_4 = zero_sf_row_3 / 32;
                        int zero_kset_4 = lane % 4;
                        int zero_sf_dst_4 = ((zero_sf_c_4 * 4 + zero_kset_4) * 8 + zero_sf_d_4) * 16 + zero_sf_g_4 * 4;
                        unsigned int zero_word_4 = 0;
                        asm volatile("st.shared.b32 [%0], %1;" :: "r"(x_zero_base + zero_sf_dst_4), "r"(zero_word_4));
                    }
                    int zero_sf_row_4 = (unsigned int)((consumer_warp + 2) * 32 + 8) + lane / 4;
                    if (zero_sf_row_4 >= half_count_c * 16) {
                        int zero_sf_c_5 = zero_sf_row_4 % 32 / 8;
                        int zero_sf_d_5 = zero_sf_row_4 % 8;
                        int zero_sf_g_5 = zero_sf_row_4 / 32;
                        int zero_kset_5 = lane % 4;
                        int zero_sf_dst_5 = ((zero_sf_c_5 * 4 + zero_kset_5) * 8 + zero_sf_d_5) * 16 + zero_sf_g_5 * 4;
                        unsigned int zero_word_5 = 0;
                        asm volatile("st.shared.b32 [%0], %1;" :: "r"(x_zero_base + zero_sf_dst_5), "r"(zero_word_5));
                    }
                    int zero_sf_row_5 = (unsigned int)((consumer_warp + 2) * 32 + 16) + lane / 4;
                    if (zero_sf_row_5 >= half_count_c * 16) {
                        int zero_sf_c_6 = zero_sf_row_5 % 32 / 8;
                        int zero_sf_d_6 = zero_sf_row_5 % 8;
                        int zero_sf_g_6 = zero_sf_row_5 / 32;
                        int zero_kset_6 = lane % 4;
                        int zero_sf_dst_6 = ((zero_sf_c_6 * 4 + zero_kset_6) * 8 + zero_sf_d_6) * 16 + zero_sf_g_6 * 4;
                        unsigned int zero_word_6 = 0;
                        asm volatile("st.shared.b32 [%0], %1;" :: "r"(x_zero_base + zero_sf_dst_6), "r"(zero_word_6));
                    }
                    int zero_sf_row_6 = (unsigned int)((consumer_warp + 2) * 32 + 24) + lane / 4;
                    if (zero_sf_row_6 >= half_count_c * 16) {
                        int zero_sf_c_7 = zero_sf_row_6 % 32 / 8;
                        int zero_sf_d_7 = zero_sf_row_6 % 8;
                        int zero_sf_g_7 = zero_sf_row_6 / 32;
                        int zero_kset_7 = lane % 4;
                        int zero_sf_dst_7 = ((zero_sf_c_7 * 4 + zero_kset_7) * 8 + zero_sf_d_7) * 16 + zero_sf_g_7 * 4;
                        unsigned int zero_word_7 = 0;
                        asm volatile("st.shared.b32 [%0], %1;" :: "r"(x_zero_base + zero_sf_dst_7), "r"(zero_word_7));
                    }
                }
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("barrier.sync 15, 64;" ::: "memory");
                if (consumer_warp == 1) {
                    int lane_pair_up = sorted_token_ids[(unsigned int)route_base_c + lane / 4];
                    int lane_pair_up_hi = M * top_k;
                    if (pair_upper_c) {
                        lane_pair_up_hi = sorted_token_ids[(unsigned int)(route_base_c + 8) + lane / 4];
                    }
                    int _min_5 = ((lane_pair_up / top_k) < (M - 1) ? (lane_pair_up / top_k) : (M - 1));
                    int lane_token_up = _min_5;
                    int _min_6 = ((lane_pair_up_hi / top_k) < (M - 1) ? (lane_pair_up_hi / top_k) : (M - 1));
                    int lane_token_up_hi = _min_6;
                    int _shfl_0 = __shfl_sync(0xFFFFFFFF, lane_token_up, 0);
                    int row0 = _shfl_0;
                    int _shfl_1 = __shfl_sync(0xFFFFFFFF, lane_token_up, 4);
                    int row1 = _shfl_1;
                    int _shfl_2 = __shfl_sync(0xFFFFFFFF, lane_token_up, 8);
                    int row2 = _shfl_2;
                    int _shfl_3 = __shfl_sync(0xFFFFFFFF, lane_token_up, 12);
                    int row3 = _shfl_3;
                    int _shfl_4 = __shfl_sync(0xFFFFFFFF, lane_token_up, 16);
                    int row4 = _shfl_4;
                    int _shfl_5 = __shfl_sync(0xFFFFFFFF, lane_token_up, 20);
                    int row5 = _shfl_5;
                    int _shfl_6 = __shfl_sync(0xFFFFFFFF, lane_token_up, 24);
                    int row6 = _shfl_6;
                    int _shfl_7 = __shfl_sync(0xFFFFFFFF, lane_token_up, 28);
                    int row7 = _shfl_7;
                    int _shfl_8 = __shfl_sync(0xFFFFFFFF, lane_token_up_hi, 0);
                    int row8 = _shfl_8;
                    int _shfl_9 = __shfl_sync(0xFFFFFFFF, lane_token_up_hi, 4);
                    int row9 = _shfl_9;
                    int _shfl_10 = __shfl_sync(0xFFFFFFFF, lane_token_up_hi, 8);
                    int row10 = _shfl_10;
                    int _shfl_11 = __shfl_sync(0xFFFFFFFF, lane_token_up_hi, 12);
                    int row11 = _shfl_11;
                    int _shfl_12 = __shfl_sync(0xFFFFFFFF, lane_token_up_hi, 16);
                    int row12 = _shfl_12;
                    int _shfl_13 = __shfl_sync(0xFFFFFFFF, lane_token_up_hi, 20);
                    int row13 = _shfl_13;
                    int _shfl_14 = __shfl_sync(0xFFFFFFFF, lane_token_up_hi, 24);
                    int row14 = _shfl_14;
                    int _shfl_15 = __shfl_sync(0xFFFFFFFF, lane_token_up_hi, 28);
                    int row15 = _shfl_15;
                    int second_rows[16];
                    #pragma unroll
                    for (int second_slice = 0; second_slice < 2; second_slice++) {
                        int second_pair = M * top_k;
                        if (has_second_c) {
                            if (second_slice == 0 || second_upper_c) {
                                second_pair = sorted_token_ids[(unsigned int)(route_base_c + (2 + second_slice) * 8) + lane / 4];
                            }
                        }
                        int _min_7 = ((second_pair / top_k) < (M - 1) ? (second_pair / top_k) : (M - 1));
                        int second_token = _min_7;
                        #pragma unroll
                        for (int second_row = 0; second_row < 8; second_row++) {
                            int _shfl_16 = __shfl_sync(0xFFFFFFFF, second_token, second_row * 4);
                            second_rows[second_slice * 8 + second_row] = _shfl_16;
                        }
                    }
                    int up_x_bytes = 2048;
                    if (has_second_c) {
                        up_x_bytes = 4096;
                    }
                    unsigned int up_x_stage = 0;
                    #pragma unroll 1
                    for (int feature_x = 0; feature_x < 2; feature_x++) {
                        #pragma unroll 1
                        for (int kb_x = 0; kb_x < K / 256; kb_x++) {
                            mbarrier_wait(up_free_addr + (up_x_stage) * 8, _phase_up_free_1);
                            if (elect_sync()) {
                                mbarrier_arrive_expect_tx(up_full_addr + (up_x_stage) * 8, up_x_bytes);
                                tma_gather4_gmem2smem(smem_x_addr + up_x_stage * 43008, (&x), kb_x * 128, row0, row1, row2, row3, up_full_addr + (up_x_stage) * 8);
                                tma_gather4_gmem2smem(smem_x_addr + up_x_stage * 43008 + 512, (&x), kb_x * 128, row4, row5, row6, row7, up_full_addr + (up_x_stage) * 8);
                                tma_gather4_gmem2smem(smem_x_addr + up_x_stage * 43008 + 1024, (&x), kb_x * 128, row8, row9, row10, row11, up_full_addr + (up_x_stage) * 8);
                                tma_gather4_gmem2smem(smem_x_addr + up_x_stage * 43008 + 1536, (&x), kb_x * 128, row12, row13, row14, row15, up_full_addr + (up_x_stage) * 8);
                                if (has_second_c) {
                                    #pragma unroll
                                    for (int second_group = 0; second_group < 4; second_group++) {
                                        tma_gather4_gmem2smem(smem_x1_addr + up_x_stage * 43008 + (unsigned int)(second_group * 4 * 128), (&x), kb_x * 128, second_rows[second_group * 4], second_rows[second_group * 4 + 1], second_rows[second_group * 4 + 2], second_rows[second_group * 4 + 3], up_full_addr + (up_x_stage) * 8);
                                    }
                                }
                            }
                            up_x_stage += 1;
                            if (up_x_stage == 2) { up_x_stage = 0; _phase_up_free_1 ^= 1; }
                        }
                    }
                }
                if (consumer_warp == 0) {
                    unsigned int up_scale_stage = 0;
                    #pragma unroll 1
                    for (int feature_scale = 0; feature_scale < 2; feature_scale++) {
                        #pragma unroll 1
                        for (int kb_scale = 0; kb_scale < K / 256; kb_scale++) {
                            mbarrier_wait(up_free_addr + (up_scale_stage) * 8, _phase_up_free_2);
                            int x_sf_base = smem_x_sf_addr + up_scale_stage * 43008;
                            int sf_row = (unsigned int)(consumer_warp * 32) + lane / 4;
                            int sf_c = sf_row % 32 / 8;
                            int sf_d = sf_row % 8;
                            int sf_g = sf_row / 32;
                            int kset = lane % 4;
                            int sf_dst = ((sf_c * 4 + kset) * 8 + sf_d) * 16 + sf_g * 4;
                            unsigned int x_word = 0;
                            if (sf_row < half_count_c * 16) {
                                int x_token_c = ((1) ? sf_lane_token_lo_c : sf_lane_token_hi_c);
                                int x_idx = x_token_c * sf_cols_c + kb_scale * 16 + kset * 4;
                                {
                                    x_word = reinterpret_cast<const unsigned int*>(x_scale)[x_idx / 4];
                                }
                                asm volatile("st.shared.b32 [%0], %1;" :: "r"(x_sf_base + sf_dst), "r"(x_word));
                            }
                            int sf_row_0 = (unsigned int)(consumer_warp * 32 + 8) + lane / 4;
                            int sf_c_1 = sf_row_0 % 32 / 8;
                            int sf_d_2 = sf_row_0 % 8;
                            int sf_g_3 = sf_row_0 / 32;
                            int kset_4 = lane % 4;
                            int sf_dst_5 = ((sf_c_1 * 4 + kset_4) * 8 + sf_d_2) * 16 + sf_g_3 * 4;
                            unsigned int x_word_6 = 0;
                            if (sf_row_0 < half_count_c * 16) {
                                int x_token_c_1 = ((0) ? sf_lane_token_lo_c : sf_lane_token_hi_c);
                                int x_idx_1 = x_token_c_1 * sf_cols_c + kb_scale * 16 + kset_4 * 4;
                                {
                                    x_word_6 = reinterpret_cast<const unsigned int*>(x_scale)[x_idx_1 / 4];
                                }
                                asm volatile("st.shared.b32 [%0], %1;" :: "r"(x_sf_base + sf_dst_5), "r"(x_word_6));
                            }
                            int sf_row_7 = (unsigned int)(consumer_warp * 32 + 16) + lane / 4;
                            int sf_c_8 = sf_row_7 % 32 / 8;
                            int sf_d_9 = sf_row_7 % 8;
                            int sf_g_10 = sf_row_7 / 32;
                            int kset_11 = lane % 4;
                            int sf_dst_12 = ((sf_c_8 * 4 + kset_11) * 8 + sf_d_9) * 16 + sf_g_10 * 4;
                            unsigned int x_word_13 = 0;
                            if (sf_row_7 < half_count_c * 16) {
                                int x_token_c_2 = ((0) ? sf_lane_token_lo_c : sf_lane_token_hi_c);
                                {
                                    x_token_c_2 = ((1) ? sf_second_token_lo_c : sf_second_token_hi_c);
                                }
                                int x_idx_2 = x_token_c_2 * sf_cols_c + kb_scale * 16 + kset_11 * 4;
                                {
                                    x_word_13 = reinterpret_cast<const unsigned int*>(x_scale)[x_idx_2 / 4];
                                }
                                asm volatile("st.shared.b32 [%0], %1;" :: "r"(x_sf_base + sf_dst_12), "r"(x_word_13));
                            }
                            int sf_row_14 = (unsigned int)(consumer_warp * 32 + 24) + lane / 4;
                            int sf_c_15 = sf_row_14 % 32 / 8;
                            int sf_d_16 = sf_row_14 % 8;
                            int sf_g_17 = sf_row_14 / 32;
                            int kset_18 = lane % 4;
                            int sf_dst_19 = ((sf_c_15 * 4 + kset_18) * 8 + sf_d_16) * 16 + sf_g_17 * 4;
                            unsigned int x_word_20 = 0;
                            if (sf_row_14 < half_count_c * 16) {
                                int x_token_c_3 = ((0) ? sf_lane_token_lo_c : sf_lane_token_hi_c);
                                {
                                    x_token_c_3 = ((0) ? sf_second_token_lo_c : sf_second_token_hi_c);
                                }
                                int x_idx_3 = x_token_c_3 * sf_cols_c + kb_scale * 16 + kset_18 * 4;
                                {
                                    x_word_20 = reinterpret_cast<const unsigned int*>(x_scale)[x_idx_3 / 4];
                                }
                                asm volatile("st.shared.b32 [%0], %1;" :: "r"(x_sf_base + sf_dst_19), "r"(x_word_20));
                            }
                            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                            __syncwarp();
                            if (elect_sync()) {
                                mbarrier_arrive(up_full_addr + (up_scale_stage) * 8);
                            }
                            up_scale_stage += 1;
                            if (up_scale_stage == 2) { up_scale_stage = 0; _phase_up_free_1 ^= 1; _phase_up_free_2 ^= 1; }
                        }
                    }
                }
                asm volatile("barrier.sync 15, 64;" ::: "memory");
                asm volatile("barrier.sync 14, 256;" ::: "memory");
            }
        }
    }
    // ---- Role: epilogue ----
    if (warp >= 4 && warp <= 7) {
        { // epilogue_main
            bool owns_route_c_1 = blockIdx.x < compact_owner_count[0];
            int original_tile_c_1 = 0;
            int route_base_c_1 = 0;
            bool pair_upper_c_1 = 0;
            bool has_second_c_1 = 0;
            bool second_upper_c_1 = 0;
            int route_expert_c_1 = 0;
            if (owns_route_c_1) {
                original_tile_c_1 = compact_owner_plan[blockIdx.x * 3];
                route_base_c_1 = original_tile_c_1 * 8;
                route_expert_c_1 = compact_owner_plan[blockIdx.x * 3 + 1];
                int live_subtiles_c_1 = compact_owner_plan[blockIdx.x * 3 + 2];
                pair_upper_c_1 = live_subtiles_c_1 >= 2;
                has_second_c_1 = live_subtiles_c_1 >= 3;
                second_upper_c_1 = live_subtiles_c_1 == 4;
            }
            unsigned int _phase_up_ready = 0;
            if (owns_route_c_1) {
                int expert_c_1 = route_expert_c_1;
                float gate_scale_scalar_c = output1_scale_gate_scalar[expert_c_1];
                float up_scale_scalar_c = output1_scale_scalar[expert_c_1];
                const int consumer_warp_1 = warp % 4;
                const int physical_feature_1 = (unsigned int)(consumer_warp_1 * 32) + lane;
                int half_count_c_1 = ((has_second_c_1) ? 2 : 1);
                #pragma unroll 1
                for (int feature_epilogue = 0; feature_epilogue < 2; feature_epilogue++) {
                    int intermediate_block_epi = 2 * blockIdx.y + feature_epilogue;
                    mbarrier_wait(up_ready_addr + (feature_epilogue) * 8, _phase_up_ready);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    asm volatile("barrier.sync 13, 128;" ::: "memory");
                    #pragma unroll 1
                    for (int record_half = 0; record_half < half_count_c_1; record_half++) {
                        int workspace_tile_c = original_tile_c_1 + 2 * record_half;
                        bool record_has_upper_c = ((record_half == 0) ? pair_upper_c_1 : second_upper_c_1);
                        int live_record_subtiles_c = ((record_has_upper_c) ? 2 : 1);
                        int gate_addr = taddr + (unsigned int)(feature_epilogue * 4 * 16) + (unsigned int)(physical_feature_1 << 16) + (unsigned int)(record_half * 16);
                        int up_addr = gate_addr + 32;
                        float _tmem_load_0[16];
                        tmem_ld_x16(&_tmem_load_0[0], gate_addr);
                        asm volatile("tcgen05.wait::ld.sync.aligned;");
                        float _tmem_load_1[16];
                        tmem_ld_x16(&_tmem_load_1[0], up_addr);
                        asm volatile("tcgen05.wait::ld.sync.aligned;");
                        float act[16];
                        #pragma unroll
                        for (int token_slot_act = 0; token_slot_act < 8; token_slot_act++) {
                            float gate = _tmem_load_0[token_slot_act] * gate_scale_scalar_c;
                            float up = _tmem_load_1[token_slot_act] * up_scale_scalar_c;
                            float _expf_0 = __expf(-gate);
                            float _rcp_0 = approx_rcp(1.0f + _expf_0);
                            float sigmoid = _rcp_0;
                            act[token_slot_act] = gate * sigmoid * up;
                            float _fabs_0 = fabsf(act[token_slot_act]);
                            float group_max = _fabs_0;
                            float _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, group_max, 1);
                            float _max_0 = max_noftz(group_max, _shfl_xor_0);
                            group_max = _max_0;
                            float _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, group_max, 2);
                            float _max_1 = max_noftz(group_max, _shfl_xor_1);
                            group_max = _max_1;
                            float _shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, group_max, 4);
                            float _max_2 = max_noftz(group_max, _shfl_xor_2);
                            group_max = _max_2;
                            float _shfl_xor_3 = __shfl_xor_sync(0xFFFFFFFF, group_max, 8);
                            float _max_3 = max_noftz(group_max, _shfl_xor_3);
                            group_max = _max_3;
                            if (lane % 16 == 0) {
                                float _max_4 = max_noftz(group_max * 0.16666666666666666f, 1e-08f);
                                float scale_value = _max_4;
                                asm volatile("st.shared.b32 [%0], %1;" :: "r"(smem_act_scale_addr + (unsigned int)((token_slot_act * 8 + physical_feature_1 / 16) * 4)), "r"((__as_u32(scale_value))));
                            }
                        }
                        if (record_has_upper_c) {
                            #pragma unroll
                            for (int token_slot_act_1 = 8; token_slot_act_1 < 16; token_slot_act_1++) {
                                float gate_1 = _tmem_load_0[token_slot_act_1] * gate_scale_scalar_c;
                                float up_1 = _tmem_load_1[token_slot_act_1] * up_scale_scalar_c;
                                float _expf_1 = __expf(-gate_1);
                                float _rcp_1 = approx_rcp(1.0f + _expf_1);
                                float sigmoid_1 = _rcp_1;
                                act[token_slot_act_1] = gate_1 * sigmoid_1 * up_1;
                                float _fabs_1 = fabsf(act[token_slot_act_1]);
                                float group_max_1 = _fabs_1;
                                float _shfl_xor_4 = __shfl_xor_sync(0xFFFFFFFF, group_max_1, 1);
                                float _max_5 = max_noftz(group_max_1, _shfl_xor_4);
                                group_max_1 = _max_5;
                                float _shfl_xor_5 = __shfl_xor_sync(0xFFFFFFFF, group_max_1, 2);
                                float _max_6 = max_noftz(group_max_1, _shfl_xor_5);
                                group_max_1 = _max_6;
                                float _shfl_xor_6 = __shfl_xor_sync(0xFFFFFFFF, group_max_1, 4);
                                float _max_7 = max_noftz(group_max_1, _shfl_xor_6);
                                group_max_1 = _max_7;
                                float _shfl_xor_7 = __shfl_xor_sync(0xFFFFFFFF, group_max_1, 8);
                                float _max_8 = max_noftz(group_max_1, _shfl_xor_7);
                                group_max_1 = _max_8;
                                if (lane % 16 == 0) {
                                    float _max_9 = max_noftz(group_max_1 * 0.16666666666666666f, 1e-08f);
                                    float scale_value_1 = _max_9;
                                    asm volatile("st.shared.b32 [%0], %1;" :: "r"(smem_act_scale_addr + (unsigned int)((token_slot_act_1 * 8 + physical_feature_1 / 16) * 4)), "r"((__as_u32(scale_value_1))));
                                }
                            }
                        }
                        asm volatile("barrier.sync 13, 128;" ::: "memory");
                        int feature_group_lane = lane - lane % 16;
                        #pragma unroll
                        for (int token_slot_quant = 0; token_slot_quant < 8; token_slot_quant++) {
                            float act_scale = smem_act_scale[token_slot_quant * 8 + physical_feature_1 / 16];
                            float rounded_act_scale = 0.0f;
                            if (lane % 16 == 0) {
                                float scale_pack_src[4];
                                scale_pack_src[0] = act_scale;
                                scale_pack_src[1] = 0.0f;
                                scale_pack_src[2] = 0.0f;
                                scale_pack_src[3] = 0.0f;
                                unsigned int scale_pack_dst[1];
                                {
                                    uint32_t _packed;
                                    asm volatile("{\n\t"
                                        ".reg .b16 _lo;\n\t"
                                        ".reg .b16 _hi;\n\t"
                                        "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                        "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                        "mov.b32 %0, {_lo, _hi};\n\t"
                                        "}"
                                        : "=r"(_packed) : "f"(scale_pack_src[0]), "f"(scale_pack_src[1]),
                                                           "f"(scale_pack_src[2]), "f"(scale_pack_src[3]));
                                    scale_pack_dst[0] = _packed;
                                }
                                unsigned int scale_code = scale_pack_dst[0] & 127;
                                unsigned int scale_exp = scale_code >> 3 & 15;
                                unsigned int scale_mant = scale_code & 7;
                                if (scale_exp == 0) {
                                    rounded_act_scale = (float)scale_mant * 0.001953125f;
                                } else {
                                    float _exp2_0 = approx_exp2((float)scale_exp - 7.0f);
                                    rounded_act_scale = _exp2_0 * (1.0f + (float)scale_mant * 0.125f);
                                }
                            }
                            float _shfl_17;
                            asm volatile("shfl.sync.idx.b32 %0, %1, %2, 0x1f, 0xffffffff;" : "=f"(_shfl_17) : "f"(rounded_act_scale), "r"(feature_group_lane));
                            rounded_act_scale = _shfl_17;
                            float safe_act_scale = ((rounded_act_scale == 0.0f) ? 1.0f : rounded_act_scale);
                            float _rcp_2 = approx_rcp(safe_act_scale);
                            float inv_safe_act_scale = _rcp_2;
                            float fp4_lo[8];
                            float fp4_hi[8];
                            #pragma unroll
                            for (int fp4_lane = 0; fp4_lane < 8; fp4_lane++) {
                                float _shfl_18;
                                asm volatile("shfl.sync.idx.b32 %0, %1, %2, 0x1f, 0xffffffff;" : "=f"(_shfl_18) : "f"(act[token_slot_quant]), "r"(feature_group_lane + fp4_lane));
                                fp4_lo[fp4_lane] = _shfl_18 * inv_safe_act_scale;
                                float _shfl_19;
                                asm volatile("shfl.sync.idx.b32 %0, %1, %2, 0x1f, 0xffffffff;" : "=f"(_shfl_19) : "f"(act[token_slot_quant]), "r"(feature_group_lane + 8 + fp4_lane));
                                fp4_hi[fp4_lane] = _shfl_19 * inv_safe_act_scale;
                            }
                            if (lane % 16 == 0) {
                                unsigned int packed_lo[1];
                                unsigned int packed_hi[1];
                                asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(packed_lo[0]) : "f"(fp4_lo[0]), "f"(fp4_lo[1]), "f"(fp4_lo[2]), "f"(fp4_lo[3]), "f"(fp4_lo[4]), "f"(fp4_lo[5]), "f"(fp4_lo[6]), "f"(fp4_lo[7]));
                                asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(packed_hi[0]) : "f"(fp4_hi[0]), "f"(fp4_hi[1]), "f"(fp4_hi[2]), "f"(fp4_hi[3]), "f"(fp4_hi[4]), "f"(fp4_hi[5]), "f"(fp4_hi[6]), "f"(fp4_hi[7]));
                                int workspace_act_byte_c = ((workspace_tile_c + token_slot_quant / 8) * intermediate_blocks_total + intermediate_block_epi) * 512 + token_slot_quant % 8 * 64 + physical_feature_1 / 2;
                                if (token_slot_quant < 8 || record_has_upper_c) {
                                    *(reinterpret_cast<unsigned int*>(act_workspace + workspace_act_byte_c) + (0)) = packed_lo[0];
                                    *(reinterpret_cast<unsigned int*>(act_workspace + (workspace_act_byte_c + 4)) + (0)) = packed_hi[0];
                                }
                            }
                        }
                        if (record_has_upper_c) {
                            #pragma unroll
                            for (int token_slot_quant_1 = 8; token_slot_quant_1 < 16; token_slot_quant_1++) {
                                float act_scale_1 = smem_act_scale[token_slot_quant_1 * 8 + physical_feature_1 / 16];
                                float rounded_act_scale_1 = 0.0f;
                                if (lane % 16 == 0) {
                                    float scale_pack_src_1[4];
                                    scale_pack_src_1[0] = act_scale_1;
                                    scale_pack_src_1[1] = 0.0f;
                                    scale_pack_src_1[2] = 0.0f;
                                    scale_pack_src_1[3] = 0.0f;
                                    unsigned int scale_pack_dst_1[1];
                                    {
                                        uint32_t _packed;
                                        asm volatile("{\n\t"
                                            ".reg .b16 _lo;\n\t"
                                            ".reg .b16 _hi;\n\t"
                                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                            "mov.b32 %0, {_lo, _hi};\n\t"
                                            "}"
                                            : "=r"(_packed) : "f"(scale_pack_src_1[0]), "f"(scale_pack_src_1[1]),
                                                               "f"(scale_pack_src_1[2]), "f"(scale_pack_src_1[3]));
                                        scale_pack_dst_1[0] = _packed;
                                    }
                                    unsigned int scale_code_1 = scale_pack_dst_1[0] & 127;
                                    unsigned int scale_exp_1 = scale_code_1 >> 3 & 15;
                                    unsigned int scale_mant_1 = scale_code_1 & 7;
                                    if (scale_exp_1 == 0) {
                                        rounded_act_scale_1 = (float)scale_mant_1 * 0.001953125f;
                                    } else {
                                        float _exp2_1 = approx_exp2((float)scale_exp_1 - 7.0f);
                                        rounded_act_scale_1 = _exp2_1 * (1.0f + (float)scale_mant_1 * 0.125f);
                                    }
                                }
                                float _shfl_20;
                                asm volatile("shfl.sync.idx.b32 %0, %1, %2, 0x1f, 0xffffffff;" : "=f"(_shfl_20) : "f"(rounded_act_scale_1), "r"(feature_group_lane));
                                rounded_act_scale_1 = _shfl_20;
                                float safe_act_scale_1 = ((rounded_act_scale_1 == 0.0f) ? 1.0f : rounded_act_scale_1);
                                float _rcp_3 = approx_rcp(safe_act_scale_1);
                                float inv_safe_act_scale_1 = _rcp_3;
                                float fp4_lo_1[8];
                                float fp4_hi_1[8];
                                #pragma unroll
                                for (int fp4_lane_1 = 0; fp4_lane_1 < 8; fp4_lane_1++) {
                                    float _shfl_21;
                                    asm volatile("shfl.sync.idx.b32 %0, %1, %2, 0x1f, 0xffffffff;" : "=f"(_shfl_21) : "f"(act[token_slot_quant_1]), "r"(feature_group_lane + fp4_lane_1));
                                    fp4_lo_1[fp4_lane_1] = _shfl_21 * inv_safe_act_scale_1;
                                    float _shfl_22;
                                    asm volatile("shfl.sync.idx.b32 %0, %1, %2, 0x1f, 0xffffffff;" : "=f"(_shfl_22) : "f"(act[token_slot_quant_1]), "r"(feature_group_lane + 8 + fp4_lane_1));
                                    fp4_hi_1[fp4_lane_1] = _shfl_22 * inv_safe_act_scale_1;
                                }
                                if (lane % 16 == 0) {
                                    unsigned int packed_lo_1[1];
                                    unsigned int packed_hi_1[1];
                                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(packed_lo_1[0]) : "f"(fp4_lo_1[0]), "f"(fp4_lo_1[1]), "f"(fp4_lo_1[2]), "f"(fp4_lo_1[3]), "f"(fp4_lo_1[4]), "f"(fp4_lo_1[5]), "f"(fp4_lo_1[6]), "f"(fp4_lo_1[7]));
                                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(packed_hi_1[0]) : "f"(fp4_hi_1[0]), "f"(fp4_hi_1[1]), "f"(fp4_hi_1[2]), "f"(fp4_hi_1[3]), "f"(fp4_hi_1[4]), "f"(fp4_hi_1[5]), "f"(fp4_hi_1[6]), "f"(fp4_hi_1[7]));
                                    int workspace_act_byte_c_1 = ((workspace_tile_c + token_slot_quant_1 / 8) * intermediate_blocks_total + intermediate_block_epi) * 512 + token_slot_quant_1 % 8 * 64 + physical_feature_1 / 2;
                                    if (token_slot_quant_1 < 8 || record_has_upper_c) {
                                        *(reinterpret_cast<unsigned int*>(act_workspace + workspace_act_byte_c_1) + (0)) = packed_lo_1[0];
                                        *(reinterpret_cast<unsigned int*>(act_workspace + (workspace_act_byte_c_1 + 4)) + (0)) = packed_hi_1[0];
                                    }
                                }
                            }
                        }
                        int sf_c_act = physical_feature_1 % 32 / 8;
                        int sf_d_act = physical_feature_1 % 8;
                        int sf_g_act = physical_feature_1 / 32;
                        #pragma unroll 1
                        for (int dense_half_c = 0; dense_half_c < live_record_subtiles_c; dense_half_c++) {
                            int dense_workspace_record_c = (workspace_tile_c + dense_half_c) * intermediate_blocks_total + intermediate_block_epi;
                            float act_sf_values[4];
                            act_sf_values[0] = 0.0f;
                            if (physical_feature_1 < 8) {
                                act_sf_values[0] = smem_act_scale[(physical_feature_1 + dense_half_c * 8) * 8];
                            }
                            act_sf_values[1] = 0.0f;
                            if (physical_feature_1 < 8) {
                                act_sf_values[1] = smem_act_scale[(physical_feature_1 + dense_half_c * 8) * 8 + 1];
                            }
                            act_sf_values[2] = 0.0f;
                            if (physical_feature_1 < 8) {
                                act_sf_values[2] = smem_act_scale[(physical_feature_1 + dense_half_c * 8) * 8 + 2];
                            }
                            act_sf_values[3] = 0.0f;
                            if (physical_feature_1 < 8) {
                                act_sf_values[3] = smem_act_scale[(physical_feature_1 + dense_half_c * 8) * 8 + 3];
                            }
                            unsigned int packed_act_sf[1];
                            {
                                uint32_t _packed;
                                asm volatile("{\n\t"
                                    ".reg .b16 _lo;\n\t"
                                    ".reg .b16 _hi;\n\t"
                                    "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                    "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                    "mov.b32 %0, {_lo, _hi};\n\t"
                                    "}"
                                    : "=r"(_packed) : "f"(act_sf_values[0]), "f"(act_sf_values[1]),
                                                       "f"(act_sf_values[2]), "f"(act_sf_values[3]));
                                packed_act_sf[0] = _packed;
                            }
                            int act_sf_dst = (sf_c_act * 2 * 8 + sf_d_act) * 16 + sf_g_act * 4;
                            *(reinterpret_cast<unsigned int*>(sf_workspace + (dense_workspace_record_c * 1024 + act_sf_dst)) + (0)) = packed_act_sf[0];
                            float act_sf_values_0[4];
                            act_sf_values_0[0] = 0.0f;
                            if (physical_feature_1 < 8) {
                                act_sf_values_0[0] = smem_act_scale[(physical_feature_1 + dense_half_c * 8) * 8 + 4];
                            }
                            act_sf_values_0[1] = 0.0f;
                            if (physical_feature_1 < 8) {
                                act_sf_values_0[1] = smem_act_scale[(physical_feature_1 + dense_half_c * 8) * 8 + 4 + 1];
                            }
                            act_sf_values_0[2] = 0.0f;
                            if (physical_feature_1 < 8) {
                                act_sf_values_0[2] = smem_act_scale[(physical_feature_1 + dense_half_c * 8) * 8 + 4 + 2];
                            }
                            act_sf_values_0[3] = 0.0f;
                            if (physical_feature_1 < 8) {
                                act_sf_values_0[3] = smem_act_scale[(physical_feature_1 + dense_half_c * 8) * 8 + 4 + 3];
                            }
                            unsigned int packed_act_sf_1[1];
                            {
                                uint32_t _packed;
                                asm volatile("{\n\t"
                                    ".reg .b16 _lo;\n\t"
                                    ".reg .b16 _hi;\n\t"
                                    "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                    "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                    "mov.b32 %0, {_lo, _hi};\n\t"
                                    "}"
                                    : "=r"(_packed) : "f"(act_sf_values_0[0]), "f"(act_sf_values_0[1]),
                                                       "f"(act_sf_values_0[2]), "f"(act_sf_values_0[3]));
                                packed_act_sf_1[0] = _packed;
                            }
                            int act_sf_dst_2 = ((sf_c_act * 2 + 1) * 8 + sf_d_act) * 16 + sf_g_act * 4;
                            *(reinterpret_cast<unsigned int*>(sf_workspace + (dense_workspace_record_c * 1024 + act_sf_dst_2)) + (0)) = packed_act_sf_1[0];
                        }
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        asm volatile("barrier.sync 13, 128;" ::: "memory");
                    }
                }
                asm volatile("barrier.sync 14, 256;" ::: "memory");
            }
        }
    }

    // Cleanup
    __syncthreads(); // barrier before TMEM dealloc

    if (warp == 0) {
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(256));
    }
}

} // extern "C"


constexpr int kGeneratedThreads = THREADS;
constexpr int kGeneratedSmemTotal = SMEM_TOTAL;
#undef LOOM_INF
#undef TMEM_NCOLS
#undef TMEM_UP_ACC_OFFSET
#undef TMEM_UP_GATE_SF_OFFSET
#undef TMEM_UP_UP_SF_OFFSET
#undef TMEM_UP_X_SF_OFFSET
#undef NUM_UP_PIPE_STAGES
#undef NUM_READY_PIPE_STAGES
#undef SMEM_SMEM_W1_OFF
#undef SMEM_SMEM_W1_STAGE_BYTES
#undef SMEM_SMEM_W1_STRIDE
#undef SMEM_SMEM_X_OFF
#undef SMEM_SMEM_X_STAGE_BYTES
#undef SMEM_SMEM_X_STRIDE
#undef SMEM_SMEM_W1_GATE_SF_OFF
#undef SMEM_SMEM_W1_GATE_SF_STAGE_BYTES
#undef SMEM_SMEM_W1_GATE_SF_STRIDE
#undef SMEM_SMEM_W1_UP_SF_OFF
#undef SMEM_SMEM_W1_UP_SF_STAGE_BYTES
#undef SMEM_SMEM_W1_UP_SF_STRIDE
#undef SMEM_SMEM_X_SF_OFF
#undef SMEM_SMEM_X_SF_STAGE_BYTES
#undef SMEM_SMEM_X_SF_STRIDE
#undef SMEM_SMEM_X1_OFF
#undef SMEM_SMEM_X1_STAGE_BYTES
#undef SMEM_SMEM_X1_STRIDE
#undef SMEM_SMEM_X_WIDE_OFF
#undef SMEM_SMEM_X_WIDE_STAGE_BYTES
#undef SMEM_SMEM_X_WIDE_STRIDE
#undef SMEM_SMEM_ACT_SCALE_OFF
#undef SMEM_SMEM_ACT_SCALE_STAGE_BYTES
#undef SMEM_SMEM_ACT_SCALE_STRIDE
#undef SMEM_TOTAL
#undef THREADS
#undef PACKED_SCALE_LOADS
#undef up_full_addr
#undef up_free_addr
#undef up_ready_addr
#undef kernel_alpha_moe_nvfp4_up_workspace_contiguous_gate_up_two_consumer
}  // namespace nvfp4_qualified_c388_up
