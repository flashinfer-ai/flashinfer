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
// MiniMax-H3 dense non-causal BF16 self-attention (56 heads x 128, contiguous [S, 7168] I/O)
// with the fused BF16 query scale, generated from the Cake Weave schedule for SM120 (GB202).
// The device code uses TMA tile loads, ldmatrix and mma.sync m16n8k16 only, so it also
// compiles for SM90a/SM100a/SM103a; SM120 is the performance target.
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <cstdint>

static_assert(sizeof(CUtensorMap) == 128, "CUDA tensor-map ABI size mismatch");
static_assert(alignof(CUtensorMap) >= 64, "CUDA tensor-map ABI requires at least 64-byte alignment");

__device__ __forceinline__ int make_warp_uniform(int x) {
    int result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1F, 0xFFFFFFFF;"
                 : "=r"(result) : "r"(x));
    return result;
}

#define MINIMAX_H3_ATTN_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define SMEM_Q_SMEM_OFF 1024
#define SMEM_Q_SMEM_STAGE_BYTES 16384
#define SMEM_Q_SMEM_STRIDE 16384
#define SMEM_K_SMEM_OFF 17408
#define SMEM_K_SMEM_STAGE_BYTES 16384
#define SMEM_K_SMEM_STRIDE 16384
#define SMEM_V_SMEM_OFF 33792
#define SMEM_V_SMEM_STAGE_BYTES 16384
#define SMEM_V_SMEM_STRIDE 16384
#define SMEM_TOTAL 50176
#define THREADS 128
#define NUM_HEADS_CONST 56
#define HEAD_MAJOR 1
#define KV_UNROLL 1

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


__device__ __forceinline__ uint64_t desc_encode(uint64_t x) {
    return (x & 0x3FFFFULL) >> 4ULL;
}


__device__ __forceinline__ uint64_t make_smem_desc(int addr) {
    const int SBO = 1024;
    return desc_encode(addr)
         | (desc_encode(SBO) << 32ULL)
         | (1ULL << 46ULL)
         | (2ULL << 61ULL);
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

extern "C" {

__global__ __launch_bounds__(128, 2) void
kernel_minimax_h3_dense_attention(const __grid_constant__ CUtensorMap Q_map, const __grid_constant__ CUtensorMap K_map, const __grid_constant__ CUtensorMap V_map, __nv_bfloat16* __restrict__ O, int tokens, int num_q_tiles, int num_kv_tiles, unsigned int total_work, float softmax_scale_log2, unsigned int q_scale_bf16x2)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int mbar_base = smem;
    #define q_full_addr (mbar_base + 0)
    #define k_full_addr (mbar_base + 8)
    #define v_full_addr (mbar_base + 16)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // Kernel setup ops
    __nv_bfloat16* q_smem = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int q_smem_addr = smem + 1024;
    __nv_bfloat16* k_smem = reinterpret_cast<__nv_bfloat16*>(smem_raw + 17408);
    const int k_smem_addr = smem + 17408;
    __nv_bfloat16* v_smem = reinterpret_cast<__nv_bfloat16*>(smem_raw + 33792);
    const int v_smem_addr = smem + 33792;

    // Mbarrier init (3 pipeline groups, 0 ordered-sequence groups, 3 barriers)
    // Mbarriers at smem_raw[0..24)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // q_full: 1 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            // k_full: 1 barriers, init_count=1
            mbarrier_init(smem + 8, 1);
            // v_full: 1 barriers, init_count=1
            mbarrier_init(smem + 16, 1);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncthreads();

    // === Task calls (dependency order) ===
    int last_kv_tile = num_kv_tiles - 1;
    int tail_rows = tokens - last_kv_tile * 64;
    unsigned int _phase_q_full_0 = 0;
    unsigned int _phase_k_full_0 = 0;
    unsigned int _phase_v_full_0 = 0;
    #pragma unroll 1
    for (unsigned int work_id = bid; work_id < total_work; work_id += num_bids) {
        int head_idx = 0;
        int q_tile_idx = 0;
        {
            head_idx = work_id / (unsigned int)num_q_tiles;
            q_tile_idx = work_id % (unsigned int)num_q_tiles;
        }
        int q_tile_base = q_tile_idx * 64;
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        if (warp == 0) {
            if (elect_sync()) {
                mbarrier_arrive_expect_tx(q_full_addr, 16384);
                asm volatile(
                    "cp.async.bulk.tensor.4d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                    " [%0], [%1, {%2, %3, %4, %5}], [%6], %7;"
                    :: "r"(q_smem_addr), "l"((&Q_map)), "r"(0), "r"(q_tile_base), "r"(head_idx), "r"(0),
                       "r"(q_full_addr), "l"(0x12F0000000000000ULL) : "memory");
            }
        }
        if (warp == 0) {
            if (elect_sync()) {
                mbarrier_arrive_expect_tx(k_full_addr, 16384);
                asm volatile(
                    "cp.async.bulk.tensor.4d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                    " [%0], [%1, {%2, %3, %4, %5}], [%6], %7;"
                    :: "r"(k_smem_addr), "l"((&K_map)), "r"(0), "r"(0), "r"(head_idx), "r"(0),
                       "r"(k_full_addr), "l"(0x14F0000000000000ULL) : "memory");
            }
        }
        if (warp == 0) {
            if (elect_sync()) {
                mbarrier_arrive_expect_tx(v_full_addr, 16384);
                asm volatile(
                    "cp.async.bulk.tensor.4d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                    " [%0], [%1, {%2, %3, %4, %5}], [%6], %7;"
                    :: "r"(v_smem_addr), "l"((&V_map)), "r"(0), "r"(0), "r"(head_idx), "r"(0),
                       "r"(v_full_addr), "l"(0x14F0000000000000ULL) : "memory");
            }
        }
        unsigned int q_frags[32];
        unsigned int k_frag[4];
        unsigned int v_frag[4];
        float qk_acc[32];
        unsigned int p_frag[16];
        float out_acc[64];
        float m_state[2];
        float d_state[2];
        out_acc[0] = 0.0f;
        out_acc[1] = 0.0f;
        out_acc[2] = 0.0f;
        out_acc[3] = 0.0f;
        out_acc[4] = 0.0f;
        out_acc[5] = 0.0f;
        out_acc[6] = 0.0f;
        out_acc[7] = 0.0f;
        out_acc[8] = 0.0f;
        out_acc[9] = 0.0f;
        out_acc[10] = 0.0f;
        out_acc[11] = 0.0f;
        out_acc[12] = 0.0f;
        out_acc[13] = 0.0f;
        out_acc[14] = 0.0f;
        out_acc[15] = 0.0f;
        out_acc[16] = 0.0f;
        out_acc[17] = 0.0f;
        out_acc[18] = 0.0f;
        out_acc[19] = 0.0f;
        out_acc[20] = 0.0f;
        out_acc[21] = 0.0f;
        out_acc[22] = 0.0f;
        out_acc[23] = 0.0f;
        out_acc[24] = 0.0f;
        out_acc[25] = 0.0f;
        out_acc[26] = 0.0f;
        out_acc[27] = 0.0f;
        out_acc[28] = 0.0f;
        out_acc[29] = 0.0f;
        out_acc[30] = 0.0f;
        out_acc[31] = 0.0f;
        out_acc[32] = 0.0f;
        out_acc[33] = 0.0f;
        out_acc[34] = 0.0f;
        out_acc[35] = 0.0f;
        out_acc[36] = 0.0f;
        out_acc[37] = 0.0f;
        out_acc[38] = 0.0f;
        out_acc[39] = 0.0f;
        out_acc[40] = 0.0f;
        out_acc[41] = 0.0f;
        out_acc[42] = 0.0f;
        out_acc[43] = 0.0f;
        out_acc[44] = 0.0f;
        out_acc[45] = 0.0f;
        out_acc[46] = 0.0f;
        out_acc[47] = 0.0f;
        out_acc[48] = 0.0f;
        out_acc[49] = 0.0f;
        out_acc[50] = 0.0f;
        out_acc[51] = 0.0f;
        out_acc[52] = 0.0f;
        out_acc[53] = 0.0f;
        out_acc[54] = 0.0f;
        out_acc[55] = 0.0f;
        out_acc[56] = 0.0f;
        out_acc[57] = 0.0f;
        out_acc[58] = 0.0f;
        out_acc[59] = 0.0f;
        out_acc[60] = 0.0f;
        out_acc[61] = 0.0f;
        out_acc[62] = 0.0f;
        out_acc[63] = 0.0f;
        m_state[0] = -MINIMAX_H3_ATTN_INF;
        m_state[1] = -MINIMAX_H3_ATTN_INF;
        d_state[0] = 1.0f;
        d_state[1] = 1.0f;
        mbarrier_wait(q_full_addr, _phase_q_full_0);
        _phase_q_full_0 ^= 1;
        unsigned int vec[4];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&vec[0])), "=r"(*reinterpret_cast<uint32_t*>(&vec[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&vec[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&vec[(0) + 3]))
            : "r"(q_smem_addr + (unsigned int)((warp * 32 + lane) * 16)));
        uint32_t _bf16x2_mul_0;
        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_0) : "r"(vec[0]), "r"(q_scale_bf16x2));
        vec[0] = _bf16x2_mul_0;
        uint32_t _bf16x2_mul_1;
        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_1) : "r"(vec[1]), "r"(q_scale_bf16x2));
        vec[1] = _bf16x2_mul_1;
        uint32_t _bf16x2_mul_2;
        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_2) : "r"(vec[2]), "r"(q_scale_bf16x2));
        vec[2] = _bf16x2_mul_2;
        uint32_t _bf16x2_mul_3;
        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_3) : "r"(vec[3]), "r"(q_scale_bf16x2));
        vec[3] = _bf16x2_mul_3;
        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
            "r"(q_smem_addr + (unsigned int)((warp * 32 + lane) * 16)), "r"(*reinterpret_cast<uint32_t*>(&vec[0])), "r"(*reinterpret_cast<uint32_t*>(&vec[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&vec[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&vec[(0) + 3])));
        unsigned int vec_0[4];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&vec_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&vec_0[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&vec_0[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&vec_0[(0) + 3]))
            : "r"(q_smem_addr + (unsigned int)((warp * 32 + lane) * 16 + 2048)));
        uint32_t _bf16x2_mul_4;
        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_4) : "r"(vec_0[0]), "r"(q_scale_bf16x2));
        vec_0[0] = _bf16x2_mul_4;
        uint32_t _bf16x2_mul_5;
        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_5) : "r"(vec_0[1]), "r"(q_scale_bf16x2));
        vec_0[1] = _bf16x2_mul_5;
        uint32_t _bf16x2_mul_6;
        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_6) : "r"(vec_0[2]), "r"(q_scale_bf16x2));
        vec_0[2] = _bf16x2_mul_6;
        uint32_t _bf16x2_mul_7;
        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_7) : "r"(vec_0[3]), "r"(q_scale_bf16x2));
        vec_0[3] = _bf16x2_mul_7;
        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
            "r"(q_smem_addr + (unsigned int)((warp * 32 + lane) * 16 + 2048)), "r"(*reinterpret_cast<uint32_t*>(&vec_0[0])), "r"(*reinterpret_cast<uint32_t*>(&vec_0[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&vec_0[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&vec_0[(0) + 3])));
        unsigned int vec_1[4];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&vec_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&vec_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&vec_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&vec_1[(0) + 3]))
            : "r"(q_smem_addr + (unsigned int)((warp * 32 + lane) * 16 + 4096)));
        uint32_t _bf16x2_mul_8;
        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_8) : "r"(vec_1[0]), "r"(q_scale_bf16x2));
        vec_1[0] = _bf16x2_mul_8;
        uint32_t _bf16x2_mul_9;
        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_9) : "r"(vec_1[1]), "r"(q_scale_bf16x2));
        vec_1[1] = _bf16x2_mul_9;
        uint32_t _bf16x2_mul_10;
        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_10) : "r"(vec_1[2]), "r"(q_scale_bf16x2));
        vec_1[2] = _bf16x2_mul_10;
        uint32_t _bf16x2_mul_11;
        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_11) : "r"(vec_1[3]), "r"(q_scale_bf16x2));
        vec_1[3] = _bf16x2_mul_11;
        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
            "r"(q_smem_addr + (unsigned int)((warp * 32 + lane) * 16 + 4096)), "r"(*reinterpret_cast<uint32_t*>(&vec_1[0])), "r"(*reinterpret_cast<uint32_t*>(&vec_1[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&vec_1[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&vec_1[(0) + 3])));
        unsigned int vec_2[4];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&vec_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&vec_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&vec_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&vec_2[(0) + 3]))
            : "r"(q_smem_addr + (unsigned int)((warp * 32 + lane) * 16 + 6144)));
        uint32_t _bf16x2_mul_12;
        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_12) : "r"(vec_2[0]), "r"(q_scale_bf16x2));
        vec_2[0] = _bf16x2_mul_12;
        uint32_t _bf16x2_mul_13;
        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_13) : "r"(vec_2[1]), "r"(q_scale_bf16x2));
        vec_2[1] = _bf16x2_mul_13;
        uint32_t _bf16x2_mul_14;
        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_14) : "r"(vec_2[2]), "r"(q_scale_bf16x2));
        vec_2[2] = _bf16x2_mul_14;
        uint32_t _bf16x2_mul_15;
        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_15) : "r"(vec_2[3]), "r"(q_scale_bf16x2));
        vec_2[3] = _bf16x2_mul_15;
        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
            "r"(q_smem_addr + (unsigned int)((warp * 32 + lane) * 16 + 6144)), "r"(*reinterpret_cast<uint32_t*>(&vec_2[0])), "r"(*reinterpret_cast<uint32_t*>(&vec_2[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&vec_2[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&vec_2[(0) + 3])));
        unsigned int vec_3[4];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&vec_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&vec_3[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&vec_3[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&vec_3[(0) + 3]))
            : "r"(q_smem_addr + (unsigned int)((warp * 32 + lane) * 16 + 8192)));
        uint32_t _bf16x2_mul_16;
        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_16) : "r"(vec_3[0]), "r"(q_scale_bf16x2));
        vec_3[0] = _bf16x2_mul_16;
        uint32_t _bf16x2_mul_17;
        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_17) : "r"(vec_3[1]), "r"(q_scale_bf16x2));
        vec_3[1] = _bf16x2_mul_17;
        uint32_t _bf16x2_mul_18;
        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_18) : "r"(vec_3[2]), "r"(q_scale_bf16x2));
        vec_3[2] = _bf16x2_mul_18;
        uint32_t _bf16x2_mul_19;
        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_19) : "r"(vec_3[3]), "r"(q_scale_bf16x2));
        vec_3[3] = _bf16x2_mul_19;
        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
            "r"(q_smem_addr + (unsigned int)((warp * 32 + lane) * 16 + 8192)), "r"(*reinterpret_cast<uint32_t*>(&vec_3[0])), "r"(*reinterpret_cast<uint32_t*>(&vec_3[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&vec_3[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&vec_3[(0) + 3])));
        unsigned int vec_4[4];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&vec_4[0])), "=r"(*reinterpret_cast<uint32_t*>(&vec_4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&vec_4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&vec_4[(0) + 3]))
            : "r"(q_smem_addr + (unsigned int)((warp * 32 + lane) * 16 + 10240)));
        uint32_t _bf16x2_mul_20;
        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_20) : "r"(vec_4[0]), "r"(q_scale_bf16x2));
        vec_4[0] = _bf16x2_mul_20;
        uint32_t _bf16x2_mul_21;
        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_21) : "r"(vec_4[1]), "r"(q_scale_bf16x2));
        vec_4[1] = _bf16x2_mul_21;
        uint32_t _bf16x2_mul_22;
        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_22) : "r"(vec_4[2]), "r"(q_scale_bf16x2));
        vec_4[2] = _bf16x2_mul_22;
        uint32_t _bf16x2_mul_23;
        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_23) : "r"(vec_4[3]), "r"(q_scale_bf16x2));
        vec_4[3] = _bf16x2_mul_23;
        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
            "r"(q_smem_addr + (unsigned int)((warp * 32 + lane) * 16 + 10240)), "r"(*reinterpret_cast<uint32_t*>(&vec_4[0])), "r"(*reinterpret_cast<uint32_t*>(&vec_4[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&vec_4[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&vec_4[(0) + 3])));
        unsigned int vec_5[4];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&vec_5[0])), "=r"(*reinterpret_cast<uint32_t*>(&vec_5[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&vec_5[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&vec_5[(0) + 3]))
            : "r"(q_smem_addr + (unsigned int)((warp * 32 + lane) * 16 + 12288)));
        uint32_t _bf16x2_mul_24;
        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_24) : "r"(vec_5[0]), "r"(q_scale_bf16x2));
        vec_5[0] = _bf16x2_mul_24;
        uint32_t _bf16x2_mul_25;
        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_25) : "r"(vec_5[1]), "r"(q_scale_bf16x2));
        vec_5[1] = _bf16x2_mul_25;
        uint32_t _bf16x2_mul_26;
        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_26) : "r"(vec_5[2]), "r"(q_scale_bf16x2));
        vec_5[2] = _bf16x2_mul_26;
        uint32_t _bf16x2_mul_27;
        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_27) : "r"(vec_5[3]), "r"(q_scale_bf16x2));
        vec_5[3] = _bf16x2_mul_27;
        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
            "r"(q_smem_addr + (unsigned int)((warp * 32 + lane) * 16 + 12288)), "r"(*reinterpret_cast<uint32_t*>(&vec_5[0])), "r"(*reinterpret_cast<uint32_t*>(&vec_5[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&vec_5[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&vec_5[(0) + 3])));
        unsigned int vec_6[4];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&vec_6[0])), "=r"(*reinterpret_cast<uint32_t*>(&vec_6[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&vec_6[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&vec_6[(0) + 3]))
            : "r"(q_smem_addr + (unsigned int)((warp * 32 + lane) * 16 + 14336)));
        uint32_t _bf16x2_mul_28;
        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_28) : "r"(vec_6[0]), "r"(q_scale_bf16x2));
        vec_6[0] = _bf16x2_mul_28;
        uint32_t _bf16x2_mul_29;
        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_29) : "r"(vec_6[1]), "r"(q_scale_bf16x2));
        vec_6[1] = _bf16x2_mul_29;
        uint32_t _bf16x2_mul_30;
        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_30) : "r"(vec_6[2]), "r"(q_scale_bf16x2));
        vec_6[2] = _bf16x2_mul_30;
        uint32_t _bf16x2_mul_31;
        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_31) : "r"(vec_6[3]), "r"(q_scale_bf16x2));
        vec_6[3] = _bf16x2_mul_31;
        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
            "r"(q_smem_addr + (unsigned int)((warp * 32 + lane) * 16 + 14336)), "r"(*reinterpret_cast<uint32_t*>(&vec_6[0])), "r"(*reinterpret_cast<uint32_t*>(&vec_6[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&vec_6[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&vec_6[(0) + 3])));
        __syncthreads();
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(q_frags[0]), "=r"(q_frags[1]), "=r"(q_frags[2]), "=r"(q_frags[3])
            : "r"(q_smem_addr + (unsigned int)((lane / 16 / 8 * 512 + (warp * 16 + lane % 16) * 8 + (lane / 16 % 8 * 16 ^ (warp * 16 + lane % 16 & 7) << 4) / 16) * 16))
            : "memory");
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(q_frags[4]), "=r"(q_frags[5]), "=r"(q_frags[6]), "=r"(q_frags[7])
            : "r"(q_smem_addr + (unsigned int)((lane / 16 / 8 * 512 + (warp * 16 + lane % 16) * 8 + (lane / 16 % 8 * 16 ^ (warp * 16 + lane % 16 & 7) << 4) / 16 ^ 2) * 16))
            : "memory");
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(q_frags[8]), "=r"(q_frags[9]), "=r"(q_frags[10]), "=r"(q_frags[11])
            : "r"(q_smem_addr + (unsigned int)((lane / 16 / 8 * 512 + (warp * 16 + lane % 16) * 8 + (lane / 16 % 8 * 16 ^ (warp * 16 + lane % 16 & 7) << 4) / 16 ^ 2 ^ 6) * 16))
            : "memory");
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(q_frags[12]), "=r"(q_frags[13]), "=r"(q_frags[14]), "=r"(q_frags[15])
            : "r"(q_smem_addr + (unsigned int)((lane / 16 / 8 * 512 + (warp * 16 + lane % 16) * 8 + (lane / 16 % 8 * 16 ^ (warp * 16 + lane % 16 & 7) << 4) / 16 ^ 2 ^ 6 ^ 2) * 16))
            : "memory");
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(q_frags[16]), "=r"(q_frags[17]), "=r"(q_frags[18]), "=r"(q_frags[19])
            : "r"(q_smem_addr + (unsigned int)(((lane / 16 / 8 * 512 + (warp * 16 + lane % 16) * 8 + (lane / 16 % 8 * 16 ^ (warp * 16 + lane % 16 & 7) << 4) / 16 ^ 2 ^ 6 ^ 2 ^ 6) + 512) * 16))
            : "memory");
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(q_frags[20]), "=r"(q_frags[21]), "=r"(q_frags[22]), "=r"(q_frags[23])
            : "r"(q_smem_addr + (unsigned int)(((lane / 16 / 8 * 512 + (warp * 16 + lane % 16) * 8 + (lane / 16 % 8 * 16 ^ (warp * 16 + lane % 16 & 7) << 4) / 16 ^ 2 ^ 6 ^ 2 ^ 6) + 512 ^ 2) * 16))
            : "memory");
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(q_frags[24]), "=r"(q_frags[25]), "=r"(q_frags[26]), "=r"(q_frags[27])
            : "r"(q_smem_addr + (unsigned int)(((lane / 16 / 8 * 512 + (warp * 16 + lane % 16) * 8 + (lane / 16 % 8 * 16 ^ (warp * 16 + lane % 16 & 7) << 4) / 16 ^ 2 ^ 6 ^ 2 ^ 6) + 512 ^ 2 ^ 6) * 16))
            : "memory");
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(q_frags[28]), "=r"(q_frags[29]), "=r"(q_frags[30]), "=r"(q_frags[31])
            : "r"(q_smem_addr + (unsigned int)(((lane / 16 / 8 * 512 + (warp * 16 + lane % 16) * 8 + (lane / 16 % 8 * 16 ^ (warp * 16 + lane % 16 & 7) << 4) / 16 ^ 2 ^ 6 ^ 2 ^ 6) + 512 ^ 2 ^ 6 ^ 2) * 16))
            : "memory");
        #pragma unroll 1
        for (int kv_tile_idx = 0; kv_tile_idx < num_kv_tiles; kv_tile_idx++) {
            int kv_tile_base = kv_tile_idx * 64;
            int next_kv_tile_base = kv_tile_base + 64;
            mbarrier_wait(k_full_addr, _phase_k_full_0);
            _phase_k_full_0 ^= 1;
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(k_smem_addr + (unsigned int)((lane % 16 / 8 / 8 * 512 + (8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (8 * (lane / 16) + lane % 8 & 7) << 4) / 16) * 16))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                : "=f"(qk_acc[0]), "=f"(qk_acc[1]), "=f"(qk_acc[2]), "=f"(qk_acc[3])
                : "r"(q_frags[0]), "r"(q_frags[1]), "r"(q_frags[2]), "r"(q_frags[3]), "r"(k_frag[0]), "r"(k_frag[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                : "=f"(qk_acc[4]), "=f"(qk_acc[(4) + 1]), "=f"(qk_acc[(4) + 2]), "=f"(qk_acc[(4) + 3])
                : "r"(q_frags[0]), "r"(q_frags[1]), "r"(q_frags[2]), "r"(q_frags[3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(k_smem_addr + (unsigned int)((lane % 16 / 8 / 8 * 512 + (8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 128) * 16))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                : "=f"(qk_acc[8]), "=f"(qk_acc[(8) + 1]), "=f"(qk_acc[(8) + 2]), "=f"(qk_acc[(8) + 3])
                : "r"(q_frags[0]), "r"(q_frags[1]), "r"(q_frags[2]), "r"(q_frags[3]), "r"(k_frag[0]), "r"(k_frag[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                : "=f"(qk_acc[12]), "=f"(qk_acc[(12) + 1]), "=f"(qk_acc[(12) + 2]), "=f"(qk_acc[(12) + 3])
                : "r"(q_frags[0]), "r"(q_frags[1]), "r"(q_frags[2]), "r"(q_frags[3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(k_smem_addr + (unsigned int)((lane % 16 / 8 / 8 * 512 + (8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 128 + 128) * 16))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                : "=f"(qk_acc[16]), "=f"(qk_acc[(16) + 1]), "=f"(qk_acc[(16) + 2]), "=f"(qk_acc[(16) + 3])
                : "r"(q_frags[0]), "r"(q_frags[1]), "r"(q_frags[2]), "r"(q_frags[3]), "r"(k_frag[0]), "r"(k_frag[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                : "=f"(qk_acc[20]), "=f"(qk_acc[(20) + 1]), "=f"(qk_acc[(20) + 2]), "=f"(qk_acc[(20) + 3])
                : "r"(q_frags[0]), "r"(q_frags[1]), "r"(q_frags[2]), "r"(q_frags[3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(k_smem_addr + (unsigned int)((lane % 16 / 8 / 8 * 512 + (8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 128 + 128 + 128) * 16))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                : "=f"(qk_acc[24]), "=f"(qk_acc[(24) + 1]), "=f"(qk_acc[(24) + 2]), "=f"(qk_acc[(24) + 3])
                : "r"(q_frags[0]), "r"(q_frags[1]), "r"(q_frags[2]), "r"(q_frags[3]), "r"(k_frag[0]), "r"(k_frag[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                : "=f"(qk_acc[28]), "=f"(qk_acc[(28) + 1]), "=f"(qk_acc[(28) + 2]), "=f"(qk_acc[(28) + 3])
                : "r"(q_frags[0]), "r"(q_frags[1]), "r"(q_frags[2]), "r"(q_frags[3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(k_smem_addr + (unsigned int)(((lane % 16 / 8 / 8 * 512 + (8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 128 + 128 + 128 + 128 ^ 2) - 512) * 16))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(qk_acc[0]), "+f"(qk_acc[1]), "+f"(qk_acc[2]), "+f"(qk_acc[3])
                : "r"(q_frags[4]), "r"(q_frags[(4) + 1]), "r"(q_frags[(4) + 2]), "r"(q_frags[(4) + 3]), "r"(k_frag[0]), "r"(k_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(qk_acc[4]), "+f"(qk_acc[(4) + 1]), "+f"(qk_acc[(4) + 2]), "+f"(qk_acc[(4) + 3])
                : "r"(q_frags[4]), "r"(q_frags[(4) + 1]), "r"(q_frags[(4) + 2]), "r"(q_frags[(4) + 3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(k_smem_addr + (unsigned int)(((lane % 16 / 8 / 8 * 512 + (8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 128 + 128 + 128 + 128 ^ 2) - 512 + 128) * 16))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(qk_acc[8]), "+f"(qk_acc[(8) + 1]), "+f"(qk_acc[(8) + 2]), "+f"(qk_acc[(8) + 3])
                : "r"(q_frags[4]), "r"(q_frags[(4) + 1]), "r"(q_frags[(4) + 2]), "r"(q_frags[(4) + 3]), "r"(k_frag[0]), "r"(k_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(qk_acc[12]), "+f"(qk_acc[(12) + 1]), "+f"(qk_acc[(12) + 2]), "+f"(qk_acc[(12) + 3])
                : "r"(q_frags[4]), "r"(q_frags[(4) + 1]), "r"(q_frags[(4) + 2]), "r"(q_frags[(4) + 3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(k_smem_addr + (unsigned int)(((lane % 16 / 8 / 8 * 512 + (8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 128 + 128 + 128 + 128 ^ 2) - 512 + 128 + 128) * 16))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(qk_acc[16]), "+f"(qk_acc[(16) + 1]), "+f"(qk_acc[(16) + 2]), "+f"(qk_acc[(16) + 3])
                : "r"(q_frags[4]), "r"(q_frags[(4) + 1]), "r"(q_frags[(4) + 2]), "r"(q_frags[(4) + 3]), "r"(k_frag[0]), "r"(k_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(qk_acc[20]), "+f"(qk_acc[(20) + 1]), "+f"(qk_acc[(20) + 2]), "+f"(qk_acc[(20) + 3])
                : "r"(q_frags[4]), "r"(q_frags[(4) + 1]), "r"(q_frags[(4) + 2]), "r"(q_frags[(4) + 3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(k_smem_addr + (unsigned int)(((lane % 16 / 8 / 8 * 512 + (8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 128 + 128 + 128 + 128 ^ 2) - 512 + 128 + 128 + 128) * 16))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(qk_acc[24]), "+f"(qk_acc[(24) + 1]), "+f"(qk_acc[(24) + 2]), "+f"(qk_acc[(24) + 3])
                : "r"(q_frags[4]), "r"(q_frags[(4) + 1]), "r"(q_frags[(4) + 2]), "r"(q_frags[(4) + 3]), "r"(k_frag[0]), "r"(k_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(qk_acc[28]), "+f"(qk_acc[(28) + 1]), "+f"(qk_acc[(28) + 2]), "+f"(qk_acc[(28) + 3])
                : "r"(q_frags[4]), "r"(q_frags[(4) + 1]), "r"(q_frags[(4) + 2]), "r"(q_frags[(4) + 3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(k_smem_addr + (unsigned int)((((lane % 16 / 8 / 8 * 512 + (8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 128 + 128 + 128 + 128 ^ 2) - 512 + 128 + 128 + 128 + 128 ^ 6) - 512) * 16))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(qk_acc[0]), "+f"(qk_acc[1]), "+f"(qk_acc[2]), "+f"(qk_acc[3])
                : "r"(q_frags[8]), "r"(q_frags[(8) + 1]), "r"(q_frags[(8) + 2]), "r"(q_frags[(8) + 3]), "r"(k_frag[0]), "r"(k_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(qk_acc[4]), "+f"(qk_acc[(4) + 1]), "+f"(qk_acc[(4) + 2]), "+f"(qk_acc[(4) + 3])
                : "r"(q_frags[8]), "r"(q_frags[(8) + 1]), "r"(q_frags[(8) + 2]), "r"(q_frags[(8) + 3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(k_smem_addr + (unsigned int)((((lane % 16 / 8 / 8 * 512 + (8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 128 + 128 + 128 + 128 ^ 2) - 512 + 128 + 128 + 128 + 128 ^ 6) - 512 + 128) * 16))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(qk_acc[8]), "+f"(qk_acc[(8) + 1]), "+f"(qk_acc[(8) + 2]), "+f"(qk_acc[(8) + 3])
                : "r"(q_frags[8]), "r"(q_frags[(8) + 1]), "r"(q_frags[(8) + 2]), "r"(q_frags[(8) + 3]), "r"(k_frag[0]), "r"(k_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(qk_acc[12]), "+f"(qk_acc[(12) + 1]), "+f"(qk_acc[(12) + 2]), "+f"(qk_acc[(12) + 3])
                : "r"(q_frags[8]), "r"(q_frags[(8) + 1]), "r"(q_frags[(8) + 2]), "r"(q_frags[(8) + 3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(k_smem_addr + (unsigned int)((((lane % 16 / 8 / 8 * 512 + (8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 128 + 128 + 128 + 128 ^ 2) - 512 + 128 + 128 + 128 + 128 ^ 6) - 512 + 128 + 128) * 16))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(qk_acc[16]), "+f"(qk_acc[(16) + 1]), "+f"(qk_acc[(16) + 2]), "+f"(qk_acc[(16) + 3])
                : "r"(q_frags[8]), "r"(q_frags[(8) + 1]), "r"(q_frags[(8) + 2]), "r"(q_frags[(8) + 3]), "r"(k_frag[0]), "r"(k_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(qk_acc[20]), "+f"(qk_acc[(20) + 1]), "+f"(qk_acc[(20) + 2]), "+f"(qk_acc[(20) + 3])
                : "r"(q_frags[8]), "r"(q_frags[(8) + 1]), "r"(q_frags[(8) + 2]), "r"(q_frags[(8) + 3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(k_smem_addr + (unsigned int)((((lane % 16 / 8 / 8 * 512 + (8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 128 + 128 + 128 + 128 ^ 2) - 512 + 128 + 128 + 128 + 128 ^ 6) - 512 + 128 + 128 + 128) * 16))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(qk_acc[24]), "+f"(qk_acc[(24) + 1]), "+f"(qk_acc[(24) + 2]), "+f"(qk_acc[(24) + 3])
                : "r"(q_frags[8]), "r"(q_frags[(8) + 1]), "r"(q_frags[(8) + 2]), "r"(q_frags[(8) + 3]), "r"(k_frag[0]), "r"(k_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(qk_acc[28]), "+f"(qk_acc[(28) + 1]), "+f"(qk_acc[(28) + 2]), "+f"(qk_acc[(28) + 3])
                : "r"(q_frags[8]), "r"(q_frags[(8) + 1]), "r"(q_frags[(8) + 2]), "r"(q_frags[(8) + 3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(k_smem_addr + (unsigned int)(((((lane % 16 / 8 / 8 * 512 + (8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 128 + 128 + 128 + 128 ^ 2) - 512 + 128 + 128 + 128 + 128 ^ 6) - 512 + 128 + 128 + 128 + 128 ^ 2) - 512) * 16))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(qk_acc[0]), "+f"(qk_acc[1]), "+f"(qk_acc[2]), "+f"(qk_acc[3])
                : "r"(q_frags[12]), "r"(q_frags[(12) + 1]), "r"(q_frags[(12) + 2]), "r"(q_frags[(12) + 3]), "r"(k_frag[0]), "r"(k_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(qk_acc[4]), "+f"(qk_acc[(4) + 1]), "+f"(qk_acc[(4) + 2]), "+f"(qk_acc[(4) + 3])
                : "r"(q_frags[12]), "r"(q_frags[(12) + 1]), "r"(q_frags[(12) + 2]), "r"(q_frags[(12) + 3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(k_smem_addr + (unsigned int)(((((lane % 16 / 8 / 8 * 512 + (8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 128 + 128 + 128 + 128 ^ 2) - 512 + 128 + 128 + 128 + 128 ^ 6) - 512 + 128 + 128 + 128 + 128 ^ 2) - 512 + 128) * 16))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(qk_acc[8]), "+f"(qk_acc[(8) + 1]), "+f"(qk_acc[(8) + 2]), "+f"(qk_acc[(8) + 3])
                : "r"(q_frags[12]), "r"(q_frags[(12) + 1]), "r"(q_frags[(12) + 2]), "r"(q_frags[(12) + 3]), "r"(k_frag[0]), "r"(k_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(qk_acc[12]), "+f"(qk_acc[(12) + 1]), "+f"(qk_acc[(12) + 2]), "+f"(qk_acc[(12) + 3])
                : "r"(q_frags[12]), "r"(q_frags[(12) + 1]), "r"(q_frags[(12) + 2]), "r"(q_frags[(12) + 3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(k_smem_addr + (unsigned int)(((((lane % 16 / 8 / 8 * 512 + (8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 128 + 128 + 128 + 128 ^ 2) - 512 + 128 + 128 + 128 + 128 ^ 6) - 512 + 128 + 128 + 128 + 128 ^ 2) - 512 + 128 + 128) * 16))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(qk_acc[16]), "+f"(qk_acc[(16) + 1]), "+f"(qk_acc[(16) + 2]), "+f"(qk_acc[(16) + 3])
                : "r"(q_frags[12]), "r"(q_frags[(12) + 1]), "r"(q_frags[(12) + 2]), "r"(q_frags[(12) + 3]), "r"(k_frag[0]), "r"(k_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(qk_acc[20]), "+f"(qk_acc[(20) + 1]), "+f"(qk_acc[(20) + 2]), "+f"(qk_acc[(20) + 3])
                : "r"(q_frags[12]), "r"(q_frags[(12) + 1]), "r"(q_frags[(12) + 2]), "r"(q_frags[(12) + 3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(k_smem_addr + (unsigned int)(((((lane % 16 / 8 / 8 * 512 + (8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 128 + 128 + 128 + 128 ^ 2) - 512 + 128 + 128 + 128 + 128 ^ 6) - 512 + 128 + 128 + 128 + 128 ^ 2) - 512 + 128 + 128 + 128) * 16))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(qk_acc[24]), "+f"(qk_acc[(24) + 1]), "+f"(qk_acc[(24) + 2]), "+f"(qk_acc[(24) + 3])
                : "r"(q_frags[12]), "r"(q_frags[(12) + 1]), "r"(q_frags[(12) + 2]), "r"(q_frags[(12) + 3]), "r"(k_frag[0]), "r"(k_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(qk_acc[28]), "+f"(qk_acc[(28) + 1]), "+f"(qk_acc[(28) + 2]), "+f"(qk_acc[(28) + 3])
                : "r"(q_frags[12]), "r"(q_frags[(12) + 1]), "r"(q_frags[(12) + 2]), "r"(q_frags[(12) + 3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(k_smem_addr + (unsigned int)((((((lane % 16 / 8 / 8 * 512 + (8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 128 + 128 + 128 + 128 ^ 2) - 512 + 128 + 128 + 128 + 128 ^ 6) - 512 + 128 + 128 + 128 + 128 ^ 2) - 512 + 128 + 128 + 128 + 128 ^ 6) + 512 - 512) * 16))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(qk_acc[0]), "+f"(qk_acc[1]), "+f"(qk_acc[2]), "+f"(qk_acc[3])
                : "r"(q_frags[16]), "r"(q_frags[(16) + 1]), "r"(q_frags[(16) + 2]), "r"(q_frags[(16) + 3]), "r"(k_frag[0]), "r"(k_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(qk_acc[4]), "+f"(qk_acc[(4) + 1]), "+f"(qk_acc[(4) + 2]), "+f"(qk_acc[(4) + 3])
                : "r"(q_frags[16]), "r"(q_frags[(16) + 1]), "r"(q_frags[(16) + 2]), "r"(q_frags[(16) + 3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(k_smem_addr + (unsigned int)((((((lane % 16 / 8 / 8 * 512 + (8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 128 + 128 + 128 + 128 ^ 2) - 512 + 128 + 128 + 128 + 128 ^ 6) - 512 + 128 + 128 + 128 + 128 ^ 2) - 512 + 128 + 128 + 128 + 128 ^ 6) + 512 - 512 + 128) * 16))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(qk_acc[8]), "+f"(qk_acc[(8) + 1]), "+f"(qk_acc[(8) + 2]), "+f"(qk_acc[(8) + 3])
                : "r"(q_frags[16]), "r"(q_frags[(16) + 1]), "r"(q_frags[(16) + 2]), "r"(q_frags[(16) + 3]), "r"(k_frag[0]), "r"(k_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(qk_acc[12]), "+f"(qk_acc[(12) + 1]), "+f"(qk_acc[(12) + 2]), "+f"(qk_acc[(12) + 3])
                : "r"(q_frags[16]), "r"(q_frags[(16) + 1]), "r"(q_frags[(16) + 2]), "r"(q_frags[(16) + 3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(k_smem_addr + (unsigned int)((((((lane % 16 / 8 / 8 * 512 + (8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 128 + 128 + 128 + 128 ^ 2) - 512 + 128 + 128 + 128 + 128 ^ 6) - 512 + 128 + 128 + 128 + 128 ^ 2) - 512 + 128 + 128 + 128 + 128 ^ 6) + 512 - 512 + 128 + 128) * 16))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(qk_acc[16]), "+f"(qk_acc[(16) + 1]), "+f"(qk_acc[(16) + 2]), "+f"(qk_acc[(16) + 3])
                : "r"(q_frags[16]), "r"(q_frags[(16) + 1]), "r"(q_frags[(16) + 2]), "r"(q_frags[(16) + 3]), "r"(k_frag[0]), "r"(k_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(qk_acc[20]), "+f"(qk_acc[(20) + 1]), "+f"(qk_acc[(20) + 2]), "+f"(qk_acc[(20) + 3])
                : "r"(q_frags[16]), "r"(q_frags[(16) + 1]), "r"(q_frags[(16) + 2]), "r"(q_frags[(16) + 3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(k_smem_addr + (unsigned int)((((((lane % 16 / 8 / 8 * 512 + (8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 128 + 128 + 128 + 128 ^ 2) - 512 + 128 + 128 + 128 + 128 ^ 6) - 512 + 128 + 128 + 128 + 128 ^ 2) - 512 + 128 + 128 + 128 + 128 ^ 6) + 512 - 512 + 128 + 128 + 128) * 16))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(qk_acc[24]), "+f"(qk_acc[(24) + 1]), "+f"(qk_acc[(24) + 2]), "+f"(qk_acc[(24) + 3])
                : "r"(q_frags[16]), "r"(q_frags[(16) + 1]), "r"(q_frags[(16) + 2]), "r"(q_frags[(16) + 3]), "r"(k_frag[0]), "r"(k_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(qk_acc[28]), "+f"(qk_acc[(28) + 1]), "+f"(qk_acc[(28) + 2]), "+f"(qk_acc[(28) + 3])
                : "r"(q_frags[16]), "r"(q_frags[(16) + 1]), "r"(q_frags[(16) + 2]), "r"(q_frags[(16) + 3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(k_smem_addr + (unsigned int)(((((((lane % 16 / 8 / 8 * 512 + (8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 128 + 128 + 128 + 128 ^ 2) - 512 + 128 + 128 + 128 + 128 ^ 6) - 512 + 128 + 128 + 128 + 128 ^ 2) - 512 + 128 + 128 + 128 + 128 ^ 6) + 512 - 512 + 128 + 128 + 128 + 128 ^ 2) - 512) * 16))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(qk_acc[0]), "+f"(qk_acc[1]), "+f"(qk_acc[2]), "+f"(qk_acc[3])
                : "r"(q_frags[20]), "r"(q_frags[(20) + 1]), "r"(q_frags[(20) + 2]), "r"(q_frags[(20) + 3]), "r"(k_frag[0]), "r"(k_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(qk_acc[4]), "+f"(qk_acc[(4) + 1]), "+f"(qk_acc[(4) + 2]), "+f"(qk_acc[(4) + 3])
                : "r"(q_frags[20]), "r"(q_frags[(20) + 1]), "r"(q_frags[(20) + 2]), "r"(q_frags[(20) + 3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(k_smem_addr + (unsigned int)(((((((lane % 16 / 8 / 8 * 512 + (8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 128 + 128 + 128 + 128 ^ 2) - 512 + 128 + 128 + 128 + 128 ^ 6) - 512 + 128 + 128 + 128 + 128 ^ 2) - 512 + 128 + 128 + 128 + 128 ^ 6) + 512 - 512 + 128 + 128 + 128 + 128 ^ 2) - 512 + 128) * 16))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(qk_acc[8]), "+f"(qk_acc[(8) + 1]), "+f"(qk_acc[(8) + 2]), "+f"(qk_acc[(8) + 3])
                : "r"(q_frags[20]), "r"(q_frags[(20) + 1]), "r"(q_frags[(20) + 2]), "r"(q_frags[(20) + 3]), "r"(k_frag[0]), "r"(k_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(qk_acc[12]), "+f"(qk_acc[(12) + 1]), "+f"(qk_acc[(12) + 2]), "+f"(qk_acc[(12) + 3])
                : "r"(q_frags[20]), "r"(q_frags[(20) + 1]), "r"(q_frags[(20) + 2]), "r"(q_frags[(20) + 3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(k_smem_addr + (unsigned int)(((((((lane % 16 / 8 / 8 * 512 + (8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 128 + 128 + 128 + 128 ^ 2) - 512 + 128 + 128 + 128 + 128 ^ 6) - 512 + 128 + 128 + 128 + 128 ^ 2) - 512 + 128 + 128 + 128 + 128 ^ 6) + 512 - 512 + 128 + 128 + 128 + 128 ^ 2) - 512 + 128 + 128) * 16))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(qk_acc[16]), "+f"(qk_acc[(16) + 1]), "+f"(qk_acc[(16) + 2]), "+f"(qk_acc[(16) + 3])
                : "r"(q_frags[20]), "r"(q_frags[(20) + 1]), "r"(q_frags[(20) + 2]), "r"(q_frags[(20) + 3]), "r"(k_frag[0]), "r"(k_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(qk_acc[20]), "+f"(qk_acc[(20) + 1]), "+f"(qk_acc[(20) + 2]), "+f"(qk_acc[(20) + 3])
                : "r"(q_frags[20]), "r"(q_frags[(20) + 1]), "r"(q_frags[(20) + 2]), "r"(q_frags[(20) + 3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(k_smem_addr + (unsigned int)(((((((lane % 16 / 8 / 8 * 512 + (8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 128 + 128 + 128 + 128 ^ 2) - 512 + 128 + 128 + 128 + 128 ^ 6) - 512 + 128 + 128 + 128 + 128 ^ 2) - 512 + 128 + 128 + 128 + 128 ^ 6) + 512 - 512 + 128 + 128 + 128 + 128 ^ 2) - 512 + 128 + 128 + 128) * 16))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(qk_acc[24]), "+f"(qk_acc[(24) + 1]), "+f"(qk_acc[(24) + 2]), "+f"(qk_acc[(24) + 3])
                : "r"(q_frags[20]), "r"(q_frags[(20) + 1]), "r"(q_frags[(20) + 2]), "r"(q_frags[(20) + 3]), "r"(k_frag[0]), "r"(k_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(qk_acc[28]), "+f"(qk_acc[(28) + 1]), "+f"(qk_acc[(28) + 2]), "+f"(qk_acc[(28) + 3])
                : "r"(q_frags[20]), "r"(q_frags[(20) + 1]), "r"(q_frags[(20) + 2]), "r"(q_frags[(20) + 3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(k_smem_addr + (unsigned int)((((((((lane % 16 / 8 / 8 * 512 + (8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 128 + 128 + 128 + 128 ^ 2) - 512 + 128 + 128 + 128 + 128 ^ 6) - 512 + 128 + 128 + 128 + 128 ^ 2) - 512 + 128 + 128 + 128 + 128 ^ 6) + 512 - 512 + 128 + 128 + 128 + 128 ^ 2) - 512 + 128 + 128 + 128 + 128 ^ 6) - 512) * 16))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(qk_acc[0]), "+f"(qk_acc[1]), "+f"(qk_acc[2]), "+f"(qk_acc[3])
                : "r"(q_frags[24]), "r"(q_frags[(24) + 1]), "r"(q_frags[(24) + 2]), "r"(q_frags[(24) + 3]), "r"(k_frag[0]), "r"(k_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(qk_acc[4]), "+f"(qk_acc[(4) + 1]), "+f"(qk_acc[(4) + 2]), "+f"(qk_acc[(4) + 3])
                : "r"(q_frags[24]), "r"(q_frags[(24) + 1]), "r"(q_frags[(24) + 2]), "r"(q_frags[(24) + 3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(k_smem_addr + (unsigned int)((((((((lane % 16 / 8 / 8 * 512 + (8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 128 + 128 + 128 + 128 ^ 2) - 512 + 128 + 128 + 128 + 128 ^ 6) - 512 + 128 + 128 + 128 + 128 ^ 2) - 512 + 128 + 128 + 128 + 128 ^ 6) + 512 - 512 + 128 + 128 + 128 + 128 ^ 2) - 512 + 128 + 128 + 128 + 128 ^ 6) - 512 + 128) * 16))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(qk_acc[8]), "+f"(qk_acc[(8) + 1]), "+f"(qk_acc[(8) + 2]), "+f"(qk_acc[(8) + 3])
                : "r"(q_frags[24]), "r"(q_frags[(24) + 1]), "r"(q_frags[(24) + 2]), "r"(q_frags[(24) + 3]), "r"(k_frag[0]), "r"(k_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(qk_acc[12]), "+f"(qk_acc[(12) + 1]), "+f"(qk_acc[(12) + 2]), "+f"(qk_acc[(12) + 3])
                : "r"(q_frags[24]), "r"(q_frags[(24) + 1]), "r"(q_frags[(24) + 2]), "r"(q_frags[(24) + 3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(k_smem_addr + (unsigned int)((((((((lane % 16 / 8 / 8 * 512 + (8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 128 + 128 + 128 + 128 ^ 2) - 512 + 128 + 128 + 128 + 128 ^ 6) - 512 + 128 + 128 + 128 + 128 ^ 2) - 512 + 128 + 128 + 128 + 128 ^ 6) + 512 - 512 + 128 + 128 + 128 + 128 ^ 2) - 512 + 128 + 128 + 128 + 128 ^ 6) - 512 + 128 + 128) * 16))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(qk_acc[16]), "+f"(qk_acc[(16) + 1]), "+f"(qk_acc[(16) + 2]), "+f"(qk_acc[(16) + 3])
                : "r"(q_frags[24]), "r"(q_frags[(24) + 1]), "r"(q_frags[(24) + 2]), "r"(q_frags[(24) + 3]), "r"(k_frag[0]), "r"(k_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(qk_acc[20]), "+f"(qk_acc[(20) + 1]), "+f"(qk_acc[(20) + 2]), "+f"(qk_acc[(20) + 3])
                : "r"(q_frags[24]), "r"(q_frags[(24) + 1]), "r"(q_frags[(24) + 2]), "r"(q_frags[(24) + 3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(k_smem_addr + (unsigned int)((((((((lane % 16 / 8 / 8 * 512 + (8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 128 + 128 + 128 + 128 ^ 2) - 512 + 128 + 128 + 128 + 128 ^ 6) - 512 + 128 + 128 + 128 + 128 ^ 2) - 512 + 128 + 128 + 128 + 128 ^ 6) + 512 - 512 + 128 + 128 + 128 + 128 ^ 2) - 512 + 128 + 128 + 128 + 128 ^ 6) - 512 + 128 + 128 + 128) * 16))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(qk_acc[24]), "+f"(qk_acc[(24) + 1]), "+f"(qk_acc[(24) + 2]), "+f"(qk_acc[(24) + 3])
                : "r"(q_frags[24]), "r"(q_frags[(24) + 1]), "r"(q_frags[(24) + 2]), "r"(q_frags[(24) + 3]), "r"(k_frag[0]), "r"(k_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(qk_acc[28]), "+f"(qk_acc[(28) + 1]), "+f"(qk_acc[(28) + 2]), "+f"(qk_acc[(28) + 3])
                : "r"(q_frags[24]), "r"(q_frags[(24) + 1]), "r"(q_frags[(24) + 2]), "r"(q_frags[(24) + 3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(k_smem_addr + (unsigned int)(((((((((lane % 16 / 8 / 8 * 512 + (8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 128 + 128 + 128 + 128 ^ 2) - 512 + 128 + 128 + 128 + 128 ^ 6) - 512 + 128 + 128 + 128 + 128 ^ 2) - 512 + 128 + 128 + 128 + 128 ^ 6) + 512 - 512 + 128 + 128 + 128 + 128 ^ 2) - 512 + 128 + 128 + 128 + 128 ^ 6) - 512 + 128 + 128 + 128 + 128 ^ 2) - 512) * 16))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(qk_acc[0]), "+f"(qk_acc[1]), "+f"(qk_acc[2]), "+f"(qk_acc[3])
                : "r"(q_frags[28]), "r"(q_frags[(28) + 1]), "r"(q_frags[(28) + 2]), "r"(q_frags[(28) + 3]), "r"(k_frag[0]), "r"(k_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(qk_acc[4]), "+f"(qk_acc[(4) + 1]), "+f"(qk_acc[(4) + 2]), "+f"(qk_acc[(4) + 3])
                : "r"(q_frags[28]), "r"(q_frags[(28) + 1]), "r"(q_frags[(28) + 2]), "r"(q_frags[(28) + 3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(k_smem_addr + (unsigned int)(((((((((lane % 16 / 8 / 8 * 512 + (8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 128 + 128 + 128 + 128 ^ 2) - 512 + 128 + 128 + 128 + 128 ^ 6) - 512 + 128 + 128 + 128 + 128 ^ 2) - 512 + 128 + 128 + 128 + 128 ^ 6) + 512 - 512 + 128 + 128 + 128 + 128 ^ 2) - 512 + 128 + 128 + 128 + 128 ^ 6) - 512 + 128 + 128 + 128 + 128 ^ 2) - 512 + 128) * 16))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(qk_acc[8]), "+f"(qk_acc[(8) + 1]), "+f"(qk_acc[(8) + 2]), "+f"(qk_acc[(8) + 3])
                : "r"(q_frags[28]), "r"(q_frags[(28) + 1]), "r"(q_frags[(28) + 2]), "r"(q_frags[(28) + 3]), "r"(k_frag[0]), "r"(k_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(qk_acc[12]), "+f"(qk_acc[(12) + 1]), "+f"(qk_acc[(12) + 2]), "+f"(qk_acc[(12) + 3])
                : "r"(q_frags[28]), "r"(q_frags[(28) + 1]), "r"(q_frags[(28) + 2]), "r"(q_frags[(28) + 3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(k_smem_addr + (unsigned int)(((((((((lane % 16 / 8 / 8 * 512 + (8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 128 + 128 + 128 + 128 ^ 2) - 512 + 128 + 128 + 128 + 128 ^ 6) - 512 + 128 + 128 + 128 + 128 ^ 2) - 512 + 128 + 128 + 128 + 128 ^ 6) + 512 - 512 + 128 + 128 + 128 + 128 ^ 2) - 512 + 128 + 128 + 128 + 128 ^ 6) - 512 + 128 + 128 + 128 + 128 ^ 2) - 512 + 128 + 128) * 16))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(qk_acc[16]), "+f"(qk_acc[(16) + 1]), "+f"(qk_acc[(16) + 2]), "+f"(qk_acc[(16) + 3])
                : "r"(q_frags[28]), "r"(q_frags[(28) + 1]), "r"(q_frags[(28) + 2]), "r"(q_frags[(28) + 3]), "r"(k_frag[0]), "r"(k_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(qk_acc[20]), "+f"(qk_acc[(20) + 1]), "+f"(qk_acc[(20) + 2]), "+f"(qk_acc[(20) + 3])
                : "r"(q_frags[28]), "r"(q_frags[(28) + 1]), "r"(q_frags[(28) + 2]), "r"(q_frags[(28) + 3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(k_smem_addr + (unsigned int)(((((((((lane % 16 / 8 / 8 * 512 + (8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 128 + 128 + 128 + 128 ^ 2) - 512 + 128 + 128 + 128 + 128 ^ 6) - 512 + 128 + 128 + 128 + 128 ^ 2) - 512 + 128 + 128 + 128 + 128 ^ 6) + 512 - 512 + 128 + 128 + 128 + 128 ^ 2) - 512 + 128 + 128 + 128 + 128 ^ 6) - 512 + 128 + 128 + 128 + 128 ^ 2) - 512 + 128 + 128 + 128) * 16))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(qk_acc[24]), "+f"(qk_acc[(24) + 1]), "+f"(qk_acc[(24) + 2]), "+f"(qk_acc[(24) + 3])
                : "r"(q_frags[28]), "r"(q_frags[(28) + 1]), "r"(q_frags[(28) + 2]), "r"(q_frags[(28) + 3]), "r"(k_frag[0]), "r"(k_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(qk_acc[28]), "+f"(qk_acc[(28) + 1]), "+f"(qk_acc[(28) + 2]), "+f"(qk_acc[(28) + 3])
                : "r"(q_frags[28]), "r"(q_frags[(28) + 1]), "r"(q_frags[(28) + 2]), "r"(q_frags[(28) + 3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]));
            if (kv_tile_idx == last_kv_tile && tail_rows < 64) {
                if (kv_tile_base + 2 * (lane % 4) >= tokens) {
                    qk_acc[0] = -MINIMAX_H3_ATTN_INF;
                }
                if (kv_tile_base + 2 * (lane % 4) + 1 >= tokens) {
                    qk_acc[1] = -MINIMAX_H3_ATTN_INF;
                }
                if (kv_tile_base + 2 * (lane % 4) >= tokens) {
                    qk_acc[2] = -MINIMAX_H3_ATTN_INF;
                }
                if (kv_tile_base + 2 * (lane % 4) + 1 >= tokens) {
                    qk_acc[3] = -MINIMAX_H3_ATTN_INF;
                }
                if (kv_tile_base + 2 * (lane % 4) + 8 >= tokens) {
                    qk_acc[4] = -MINIMAX_H3_ATTN_INF;
                }
                if (kv_tile_base + 2 * (lane % 4) + 8 + 1 >= tokens) {
                    qk_acc[5] = -MINIMAX_H3_ATTN_INF;
                }
                if (kv_tile_base + 2 * (lane % 4) + 8 >= tokens) {
                    qk_acc[6] = -MINIMAX_H3_ATTN_INF;
                }
                if (kv_tile_base + 2 * (lane % 4) + 8 + 1 >= tokens) {
                    qk_acc[7] = -MINIMAX_H3_ATTN_INF;
                }
                if (kv_tile_base + 16 + 2 * (lane % 4) >= tokens) {
                    qk_acc[8] = -MINIMAX_H3_ATTN_INF;
                }
                if (kv_tile_base + 16 + 2 * (lane % 4) + 1 >= tokens) {
                    qk_acc[9] = -MINIMAX_H3_ATTN_INF;
                }
                if (kv_tile_base + 16 + 2 * (lane % 4) >= tokens) {
                    qk_acc[10] = -MINIMAX_H3_ATTN_INF;
                }
                if (kv_tile_base + 16 + 2 * (lane % 4) + 1 >= tokens) {
                    qk_acc[11] = -MINIMAX_H3_ATTN_INF;
                }
                if (kv_tile_base + 16 + 2 * (lane % 4) + 8 >= tokens) {
                    qk_acc[12] = -MINIMAX_H3_ATTN_INF;
                }
                if (kv_tile_base + 16 + 2 * (lane % 4) + 8 + 1 >= tokens) {
                    qk_acc[13] = -MINIMAX_H3_ATTN_INF;
                }
                if (kv_tile_base + 16 + 2 * (lane % 4) + 8 >= tokens) {
                    qk_acc[14] = -MINIMAX_H3_ATTN_INF;
                }
                if (kv_tile_base + 16 + 2 * (lane % 4) + 8 + 1 >= tokens) {
                    qk_acc[15] = -MINIMAX_H3_ATTN_INF;
                }
                if (kv_tile_base + 32 + 2 * (lane % 4) >= tokens) {
                    qk_acc[16] = -MINIMAX_H3_ATTN_INF;
                }
                if (kv_tile_base + 32 + 2 * (lane % 4) + 1 >= tokens) {
                    qk_acc[17] = -MINIMAX_H3_ATTN_INF;
                }
                if (kv_tile_base + 32 + 2 * (lane % 4) >= tokens) {
                    qk_acc[18] = -MINIMAX_H3_ATTN_INF;
                }
                if (kv_tile_base + 32 + 2 * (lane % 4) + 1 >= tokens) {
                    qk_acc[19] = -MINIMAX_H3_ATTN_INF;
                }
                if (kv_tile_base + 32 + 2 * (lane % 4) + 8 >= tokens) {
                    qk_acc[20] = -MINIMAX_H3_ATTN_INF;
                }
                if (kv_tile_base + 32 + 2 * (lane % 4) + 8 + 1 >= tokens) {
                    qk_acc[21] = -MINIMAX_H3_ATTN_INF;
                }
                if (kv_tile_base + 32 + 2 * (lane % 4) + 8 >= tokens) {
                    qk_acc[22] = -MINIMAX_H3_ATTN_INF;
                }
                if (kv_tile_base + 32 + 2 * (lane % 4) + 8 + 1 >= tokens) {
                    qk_acc[23] = -MINIMAX_H3_ATTN_INF;
                }
                if (kv_tile_base + 48 + 2 * (lane % 4) >= tokens) {
                    qk_acc[24] = -MINIMAX_H3_ATTN_INF;
                }
                if (kv_tile_base + 48 + 2 * (lane % 4) + 1 >= tokens) {
                    qk_acc[25] = -MINIMAX_H3_ATTN_INF;
                }
                if (kv_tile_base + 48 + 2 * (lane % 4) >= tokens) {
                    qk_acc[26] = -MINIMAX_H3_ATTN_INF;
                }
                if (kv_tile_base + 48 + 2 * (lane % 4) + 1 >= tokens) {
                    qk_acc[27] = -MINIMAX_H3_ATTN_INF;
                }
                if (kv_tile_base + 48 + 2 * (lane % 4) + 8 >= tokens) {
                    qk_acc[28] = -MINIMAX_H3_ATTN_INF;
                }
                if (kv_tile_base + 48 + 2 * (lane % 4) + 8 + 1 >= tokens) {
                    qk_acc[29] = -MINIMAX_H3_ATTN_INF;
                }
                if (kv_tile_base + 48 + 2 * (lane % 4) + 8 >= tokens) {
                    qk_acc[30] = -MINIMAX_H3_ATTN_INF;
                }
                if (kv_tile_base + 48 + 2 * (lane % 4) + 8 + 1 >= tokens) {
                    qk_acc[31] = -MINIMAX_H3_ATTN_INF;
                }
            }
            float o_scale[2];
            float m_prev = m_state[0];
            float _max_0 = max_noftz(qk_acc[0], qk_acc[1]);
            float _max_1 = max_noftz(qk_acc[4], qk_acc[5]);
            float _max_2 = max_noftz(_max_0, _max_1);
            float _max_3 = max_noftz(m_state[0], _max_2);
            m_state[0] = _max_3;
            float _max_4 = max_noftz(qk_acc[8], qk_acc[9]);
            float _max_5 = max_noftz(qk_acc[12], qk_acc[13]);
            float _max_6 = max_noftz(_max_4, _max_5);
            float _max_7 = max_noftz(m_state[0], _max_6);
            m_state[0] = _max_7;
            float _max_8 = max_noftz(qk_acc[16], qk_acc[17]);
            float _max_9 = max_noftz(qk_acc[20], qk_acc[21]);
            float _max_10 = max_noftz(_max_8, _max_9);
            float _max_11 = max_noftz(m_state[0], _max_10);
            m_state[0] = _max_11;
            float _max_12 = max_noftz(qk_acc[24], qk_acc[25]);
            float _max_13 = max_noftz(qk_acc[28], qk_acc[29]);
            float _max_14 = max_noftz(_max_12, _max_13);
            float _max_15 = max_noftz(m_state[0], _max_14);
            m_state[0] = _max_15;
            float _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, m_state[0], 2);
            float _max_16 = max_noftz(m_state[0], _shfl_xor_0);
            m_state[0] = _max_16;
            float _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, m_state[0], 1);
            float _max_17 = max_noftz(m_state[0], _shfl_xor_1);
            m_state[0] = _max_17;
            float _exp2_0 = approx_exp2(m_prev * softmax_scale_log2 - m_state[0] * softmax_scale_log2);
            o_scale[0] = _exp2_0;
            float m_prev_0 = m_state[1];
            float _max_18 = max_noftz(qk_acc[2], qk_acc[3]);
            float _max_19 = max_noftz(qk_acc[6], qk_acc[7]);
            float _max_20 = max_noftz(_max_18, _max_19);
            float _max_21 = max_noftz(m_state[1], _max_20);
            m_state[1] = _max_21;
            float _max_22 = max_noftz(qk_acc[10], qk_acc[11]);
            float _max_23 = max_noftz(qk_acc[14], qk_acc[15]);
            float _max_24 = max_noftz(_max_22, _max_23);
            float _max_25 = max_noftz(m_state[1], _max_24);
            m_state[1] = _max_25;
            float _max_26 = max_noftz(qk_acc[18], qk_acc[19]);
            float _max_27 = max_noftz(qk_acc[22], qk_acc[23]);
            float _max_28 = max_noftz(_max_26, _max_27);
            float _max_29 = max_noftz(m_state[1], _max_28);
            m_state[1] = _max_29;
            float _max_30 = max_noftz(qk_acc[26], qk_acc[27]);
            float _max_31 = max_noftz(qk_acc[30], qk_acc[31]);
            float _max_32 = max_noftz(_max_30, _max_31);
            float _max_33 = max_noftz(m_state[1], _max_32);
            m_state[1] = _max_33;
            float _shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, m_state[1], 2);
            float _max_34 = max_noftz(m_state[1], _shfl_xor_2);
            m_state[1] = _max_34;
            float _shfl_xor_3 = __shfl_xor_sync(0xFFFFFFFF, m_state[1], 1);
            float _max_35 = max_noftz(m_state[1], _shfl_xor_3);
            m_state[1] = _max_35;
            float _exp2_1 = approx_exp2(m_prev_0 * softmax_scale_log2 - m_state[1] * softmax_scale_log2);
            o_scale[1] = _exp2_1;
            {
                float2 _pair_scale_even2_0 = make_float2(o_scale[0], o_scale[0]);
                float2 _pair_scale_odd2_0 = make_float2(o_scale[1], o_scale[1]);
                float2* _pair_scale_src2_0 = reinterpret_cast<float2*>(&(out_acc + 0)[0]);
                #if __CUDA_ARCH__ >= 1000
                asm volatile("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_0[0]) : "l"(*(unsigned long long*)&_pair_scale_even2_0));
                asm volatile("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_0[1]) : "l"(*(unsigned long long*)&_pair_scale_odd2_0));
                asm volatile("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_0[2]) : "l"(*(unsigned long long*)&_pair_scale_even2_0));
                asm volatile("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_0[3]) : "l"(*(unsigned long long*)&_pair_scale_odd2_0));
                #else
                (out_acc + 0)[0] *= o_scale[0];
                (out_acc + 0)[1] *= o_scale[0];
                (out_acc + 0)[2] *= o_scale[1];
                (out_acc + 0)[3] *= o_scale[1];
                (out_acc + 0)[4] *= o_scale[0];
                (out_acc + 0)[5] *= o_scale[0];
                (out_acc + 0)[6] *= o_scale[1];
                (out_acc + 0)[7] *= o_scale[1];
                #endif
            }
            {
                float2 _pair_scale_even2_1 = make_float2(o_scale[0], o_scale[0]);
                float2 _pair_scale_odd2_1 = make_float2(o_scale[1], o_scale[1]);
                float2* _pair_scale_src2_1 = reinterpret_cast<float2*>(&(out_acc + 8)[0]);
                #if __CUDA_ARCH__ >= 1000
                asm volatile("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_1[0]) : "l"(*(unsigned long long*)&_pair_scale_even2_1));
                asm volatile("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_1[1]) : "l"(*(unsigned long long*)&_pair_scale_odd2_1));
                asm volatile("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_1[2]) : "l"(*(unsigned long long*)&_pair_scale_even2_1));
                asm volatile("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_1[3]) : "l"(*(unsigned long long*)&_pair_scale_odd2_1));
                #else
                (out_acc + 8)[0] *= o_scale[0];
                (out_acc + 8)[1] *= o_scale[0];
                (out_acc + 8)[2] *= o_scale[1];
                (out_acc + 8)[3] *= o_scale[1];
                (out_acc + 8)[4] *= o_scale[0];
                (out_acc + 8)[5] *= o_scale[0];
                (out_acc + 8)[6] *= o_scale[1];
                (out_acc + 8)[7] *= o_scale[1];
                #endif
            }
            {
                float2 _pair_scale_even2_2 = make_float2(o_scale[0], o_scale[0]);
                float2 _pair_scale_odd2_2 = make_float2(o_scale[1], o_scale[1]);
                float2* _pair_scale_src2_2 = reinterpret_cast<float2*>(&(out_acc + 16)[0]);
                #if __CUDA_ARCH__ >= 1000
                asm volatile("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_2[0]) : "l"(*(unsigned long long*)&_pair_scale_even2_2));
                asm volatile("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_2[1]) : "l"(*(unsigned long long*)&_pair_scale_odd2_2));
                asm volatile("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_2[2]) : "l"(*(unsigned long long*)&_pair_scale_even2_2));
                asm volatile("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_2[3]) : "l"(*(unsigned long long*)&_pair_scale_odd2_2));
                #else
                (out_acc + 16)[0] *= o_scale[0];
                (out_acc + 16)[1] *= o_scale[0];
                (out_acc + 16)[2] *= o_scale[1];
                (out_acc + 16)[3] *= o_scale[1];
                (out_acc + 16)[4] *= o_scale[0];
                (out_acc + 16)[5] *= o_scale[0];
                (out_acc + 16)[6] *= o_scale[1];
                (out_acc + 16)[7] *= o_scale[1];
                #endif
            }
            {
                float2 _pair_scale_even2_3 = make_float2(o_scale[0], o_scale[0]);
                float2 _pair_scale_odd2_3 = make_float2(o_scale[1], o_scale[1]);
                float2* _pair_scale_src2_3 = reinterpret_cast<float2*>(&(out_acc + 24)[0]);
                #if __CUDA_ARCH__ >= 1000
                asm volatile("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_3[0]) : "l"(*(unsigned long long*)&_pair_scale_even2_3));
                asm volatile("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_3[1]) : "l"(*(unsigned long long*)&_pair_scale_odd2_3));
                asm volatile("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_3[2]) : "l"(*(unsigned long long*)&_pair_scale_even2_3));
                asm volatile("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_3[3]) : "l"(*(unsigned long long*)&_pair_scale_odd2_3));
                #else
                (out_acc + 24)[0] *= o_scale[0];
                (out_acc + 24)[1] *= o_scale[0];
                (out_acc + 24)[2] *= o_scale[1];
                (out_acc + 24)[3] *= o_scale[1];
                (out_acc + 24)[4] *= o_scale[0];
                (out_acc + 24)[5] *= o_scale[0];
                (out_acc + 24)[6] *= o_scale[1];
                (out_acc + 24)[7] *= o_scale[1];
                #endif
            }
            {
                float2 _pair_scale_even2_4 = make_float2(o_scale[0], o_scale[0]);
                float2 _pair_scale_odd2_4 = make_float2(o_scale[1], o_scale[1]);
                float2* _pair_scale_src2_4 = reinterpret_cast<float2*>(&(out_acc + 32)[0]);
                #if __CUDA_ARCH__ >= 1000
                asm volatile("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_4[0]) : "l"(*(unsigned long long*)&_pair_scale_even2_4));
                asm volatile("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_4[1]) : "l"(*(unsigned long long*)&_pair_scale_odd2_4));
                asm volatile("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_4[2]) : "l"(*(unsigned long long*)&_pair_scale_even2_4));
                asm volatile("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_4[3]) : "l"(*(unsigned long long*)&_pair_scale_odd2_4));
                #else
                (out_acc + 32)[0] *= o_scale[0];
                (out_acc + 32)[1] *= o_scale[0];
                (out_acc + 32)[2] *= o_scale[1];
                (out_acc + 32)[3] *= o_scale[1];
                (out_acc + 32)[4] *= o_scale[0];
                (out_acc + 32)[5] *= o_scale[0];
                (out_acc + 32)[6] *= o_scale[1];
                (out_acc + 32)[7] *= o_scale[1];
                #endif
            }
            {
                float2 _pair_scale_even2_5 = make_float2(o_scale[0], o_scale[0]);
                float2 _pair_scale_odd2_5 = make_float2(o_scale[1], o_scale[1]);
                float2* _pair_scale_src2_5 = reinterpret_cast<float2*>(&(out_acc + 40)[0]);
                #if __CUDA_ARCH__ >= 1000
                asm volatile("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_5[0]) : "l"(*(unsigned long long*)&_pair_scale_even2_5));
                asm volatile("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_5[1]) : "l"(*(unsigned long long*)&_pair_scale_odd2_5));
                asm volatile("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_5[2]) : "l"(*(unsigned long long*)&_pair_scale_even2_5));
                asm volatile("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_5[3]) : "l"(*(unsigned long long*)&_pair_scale_odd2_5));
                #else
                (out_acc + 40)[0] *= o_scale[0];
                (out_acc + 40)[1] *= o_scale[0];
                (out_acc + 40)[2] *= o_scale[1];
                (out_acc + 40)[3] *= o_scale[1];
                (out_acc + 40)[4] *= o_scale[0];
                (out_acc + 40)[5] *= o_scale[0];
                (out_acc + 40)[6] *= o_scale[1];
                (out_acc + 40)[7] *= o_scale[1];
                #endif
            }
            {
                float2 _pair_scale_even2_6 = make_float2(o_scale[0], o_scale[0]);
                float2 _pair_scale_odd2_6 = make_float2(o_scale[1], o_scale[1]);
                float2* _pair_scale_src2_6 = reinterpret_cast<float2*>(&(out_acc + 48)[0]);
                #if __CUDA_ARCH__ >= 1000
                asm volatile("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_6[0]) : "l"(*(unsigned long long*)&_pair_scale_even2_6));
                asm volatile("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_6[1]) : "l"(*(unsigned long long*)&_pair_scale_odd2_6));
                asm volatile("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_6[2]) : "l"(*(unsigned long long*)&_pair_scale_even2_6));
                asm volatile("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_6[3]) : "l"(*(unsigned long long*)&_pair_scale_odd2_6));
                #else
                (out_acc + 48)[0] *= o_scale[0];
                (out_acc + 48)[1] *= o_scale[0];
                (out_acc + 48)[2] *= o_scale[1];
                (out_acc + 48)[3] *= o_scale[1];
                (out_acc + 48)[4] *= o_scale[0];
                (out_acc + 48)[5] *= o_scale[0];
                (out_acc + 48)[6] *= o_scale[1];
                (out_acc + 48)[7] *= o_scale[1];
                #endif
            }
            {
                float2 _pair_scale_even2_7 = make_float2(o_scale[0], o_scale[0]);
                float2 _pair_scale_odd2_7 = make_float2(o_scale[1], o_scale[1]);
                float2* _pair_scale_src2_7 = reinterpret_cast<float2*>(&(out_acc + 56)[0]);
                #if __CUDA_ARCH__ >= 1000
                asm volatile("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_7[0]) : "l"(*(unsigned long long*)&_pair_scale_even2_7));
                asm volatile("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_7[1]) : "l"(*(unsigned long long*)&_pair_scale_odd2_7));
                asm volatile("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_7[2]) : "l"(*(unsigned long long*)&_pair_scale_even2_7));
                asm volatile("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_7[3]) : "l"(*(unsigned long long*)&_pair_scale_odd2_7));
                #else
                (out_acc + 56)[0] *= o_scale[0];
                (out_acc + 56)[1] *= o_scale[0];
                (out_acc + 56)[2] *= o_scale[1];
                (out_acc + 56)[3] *= o_scale[1];
                (out_acc + 56)[4] *= o_scale[0];
                (out_acc + 56)[5] *= o_scale[0];
                (out_acc + 56)[6] *= o_scale[1];
                (out_acc + 56)[7] *= o_scale[1];
                #endif
            }
            float row0_shift = m_state[0] * softmax_scale_log2;
            float row1_shift = m_state[1] * softmax_scale_log2;
            float sum_even = 0.0f;
            float sum_odd = 0.0f;
            {
                float _exp2_2 = approx_exp2(qk_acc[0] * softmax_scale_log2 - row0_shift);
                float e0 = _exp2_2;
                float _exp2_3 = approx_exp2(qk_acc[1] * softmax_scale_log2 - row0_shift);
                float e1 = _exp2_3;
                qk_acc[0] = e0;
                qk_acc[1] = e1;
                sum_even = sum_even + e0 + e1;
            }
            {
                float _exp2_8 = approx_exp2(qk_acc[2] * softmax_scale_log2 - row1_shift);
                float e0_1 = _exp2_8;
                float _exp2_9 = approx_exp2(qk_acc[3] * softmax_scale_log2 - row1_shift);
                float e1_1 = _exp2_9;
                qk_acc[2] = e0_1;
                qk_acc[3] = e1_1;
                sum_odd = sum_odd + e0_1 + e1_1;
            }
            {
                float _exp2_10 = approx_exp2(qk_acc[4] * softmax_scale_log2 - row0_shift);
                float e0_2 = _exp2_10;
                float _exp2_11 = approx_exp2(qk_acc[5] * softmax_scale_log2 - row0_shift);
                float e1_2 = _exp2_11;
                qk_acc[4] = e0_2;
                qk_acc[5] = e1_2;
                sum_even = sum_even + e0_2 + e1_2;
            }
            {
                float _exp2_16 = approx_exp2(qk_acc[6] * softmax_scale_log2 - row1_shift);
                float e0_3 = _exp2_16;
                float _exp2_17 = approx_exp2(qk_acc[7] * softmax_scale_log2 - row1_shift);
                float e1_3 = _exp2_17;
                qk_acc[6] = e0_3;
                qk_acc[7] = e1_3;
                sum_odd = sum_odd + e0_3 + e1_3;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(qk_acc[_lp*2 + 0], qk_acc[_lp*2+1 + 0]));
                p_frag[_lp] = *(uint32_t*)&_bf2;
            }
            {
                float _fma_0 = __fmaf_rn(d_state[0], o_scale[0], sum_even);
                d_state[0] = _fma_0;
                float _fma_1 = __fmaf_rn(d_state[1], o_scale[1], sum_odd);
                d_state[1] = _fma_1;
            }
            float sum_even_1 = 0.0f;
            float sum_odd_2 = 0.0f;
            {
                float _exp2_18 = approx_exp2(qk_acc[8] * softmax_scale_log2 - row0_shift);
                float e0_4 = _exp2_18;
                float _exp2_19 = approx_exp2(qk_acc[9] * softmax_scale_log2 - row0_shift);
                float e1_4 = _exp2_19;
                qk_acc[8] = e0_4;
                qk_acc[9] = e1_4;
                sum_even_1 = sum_even_1 + e0_4 + e1_4;
            }
            {
                float _exp2_24 = approx_exp2(qk_acc[10] * softmax_scale_log2 - row1_shift);
                float e0_5 = _exp2_24;
                float _exp2_25 = approx_exp2(qk_acc[11] * softmax_scale_log2 - row1_shift);
                float e1_5 = _exp2_25;
                qk_acc[10] = e0_5;
                qk_acc[11] = e1_5;
                sum_odd_2 = sum_odd_2 + e0_5 + e1_5;
            }
            {
                float _exp2_26 = approx_exp2(qk_acc[12] * softmax_scale_log2 - row0_shift);
                float e0_6 = _exp2_26;
                float _exp2_27 = approx_exp2(qk_acc[13] * softmax_scale_log2 - row0_shift);
                float e1_6 = _exp2_27;
                qk_acc[12] = e0_6;
                qk_acc[13] = e1_6;
                sum_even_1 = sum_even_1 + e0_6 + e1_6;
            }
            {
                float _exp2_32 = approx_exp2(qk_acc[14] * softmax_scale_log2 - row1_shift);
                float e0_7 = _exp2_32;
                float _exp2_33 = approx_exp2(qk_acc[15] * softmax_scale_log2 - row1_shift);
                float e1_7 = _exp2_33;
                qk_acc[14] = e0_7;
                qk_acc[15] = e1_7;
                sum_odd_2 = sum_odd_2 + e0_7 + e1_7;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(qk_acc[_lp*2 + 8], qk_acc[_lp*2+1 + 8]));
                p_frag[_lp + 4] = *(uint32_t*)&_bf2;
            }
            {
                d_state[0] = d_state[0] + sum_even_1;
                d_state[1] = d_state[1] + sum_odd_2;
            }
            float sum_even_3 = 0.0f;
            float sum_odd_4 = 0.0f;
            {
                float _exp2_34 = approx_exp2(qk_acc[16] * softmax_scale_log2 - row0_shift);
                float e0_8 = _exp2_34;
                float _exp2_35 = approx_exp2(qk_acc[17] * softmax_scale_log2 - row0_shift);
                float e1_8 = _exp2_35;
                qk_acc[16] = e0_8;
                qk_acc[17] = e1_8;
                sum_even_3 = sum_even_3 + e0_8 + e1_8;
            }
            {
                float _exp2_40 = approx_exp2(qk_acc[18] * softmax_scale_log2 - row1_shift);
                float e0_9 = _exp2_40;
                float _exp2_41 = approx_exp2(qk_acc[19] * softmax_scale_log2 - row1_shift);
                float e1_9 = _exp2_41;
                qk_acc[18] = e0_9;
                qk_acc[19] = e1_9;
                sum_odd_4 = sum_odd_4 + e0_9 + e1_9;
            }
            {
                float _exp2_42 = approx_exp2(qk_acc[20] * softmax_scale_log2 - row0_shift);
                float e0_10 = _exp2_42;
                float _exp2_43 = approx_exp2(qk_acc[21] * softmax_scale_log2 - row0_shift);
                float e1_10 = _exp2_43;
                qk_acc[20] = e0_10;
                qk_acc[21] = e1_10;
                sum_even_3 = sum_even_3 + e0_10 + e1_10;
            }
            {
                float _exp2_48 = approx_exp2(qk_acc[22] * softmax_scale_log2 - row1_shift);
                float e0_11 = _exp2_48;
                float _exp2_49 = approx_exp2(qk_acc[23] * softmax_scale_log2 - row1_shift);
                float e1_11 = _exp2_49;
                qk_acc[22] = e0_11;
                qk_acc[23] = e1_11;
                sum_odd_4 = sum_odd_4 + e0_11 + e1_11;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(qk_acc[_lp*2 + 16], qk_acc[_lp*2+1 + 16]));
                p_frag[_lp + 8] = *(uint32_t*)&_bf2;
            }
            {
                d_state[0] = d_state[0] + sum_even_3;
                d_state[1] = d_state[1] + sum_odd_4;
            }
            float sum_even_5 = 0.0f;
            float sum_odd_6 = 0.0f;
            {
                float _exp2_50 = approx_exp2(qk_acc[24] * softmax_scale_log2 - row0_shift);
                float e0_12 = _exp2_50;
                float _exp2_51 = approx_exp2(qk_acc[25] * softmax_scale_log2 - row0_shift);
                float e1_12 = _exp2_51;
                qk_acc[24] = e0_12;
                qk_acc[25] = e1_12;
                sum_even_5 = sum_even_5 + e0_12 + e1_12;
            }
            {
                float _exp2_56 = approx_exp2(qk_acc[26] * softmax_scale_log2 - row1_shift);
                float e0_13 = _exp2_56;
                float _exp2_57 = approx_exp2(qk_acc[27] * softmax_scale_log2 - row1_shift);
                float e1_13 = _exp2_57;
                qk_acc[26] = e0_13;
                qk_acc[27] = e1_13;
                sum_odd_6 = sum_odd_6 + e0_13 + e1_13;
            }
            {
                float _exp2_58 = approx_exp2(qk_acc[28] * softmax_scale_log2 - row0_shift);
                float e0_14 = _exp2_58;
                float _exp2_59 = approx_exp2(qk_acc[29] * softmax_scale_log2 - row0_shift);
                float e1_14 = _exp2_59;
                qk_acc[28] = e0_14;
                qk_acc[29] = e1_14;
                sum_even_5 = sum_even_5 + e0_14 + e1_14;
            }
            {
                float _exp2_64 = approx_exp2(qk_acc[30] * softmax_scale_log2 - row1_shift);
                float e0_15 = _exp2_64;
                float _exp2_65 = approx_exp2(qk_acc[31] * softmax_scale_log2 - row1_shift);
                float e1_15 = _exp2_65;
                qk_acc[30] = e0_15;
                qk_acc[31] = e1_15;
                sum_odd_6 = sum_odd_6 + e0_15 + e1_15;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(qk_acc[_lp*2 + 24], qk_acc[_lp*2+1 + 24]));
                p_frag[_lp + 12] = *(uint32_t*)&_bf2;
            }
            {
                d_state[0] = d_state[0] + sum_even_5;
                d_state[1] = d_state[1] + sum_odd_6;
            }
            __syncthreads();
            if (last_kv_tile > kv_tile_idx) {
                if (warp == 0) {
                    if (elect_sync()) {
                        mbarrier_arrive_expect_tx(k_full_addr, 16384);
                        asm volatile(
                            "cp.async.bulk.tensor.4d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                            " [%0], [%1, {%2, %3, %4, %5}], [%6], %7;"
                            :: "r"(k_smem_addr), "l"((&K_map)), "r"(0), "r"(next_kv_tile_base), "r"(head_idx), "r"(0),
                               "r"(k_full_addr), "l"(0x14F0000000000000ULL) : "memory");
                    }
                }
            }
            mbarrier_wait(v_full_addr, _phase_v_full_0);
            _phase_v_full_0 ^= 1;
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(v_smem_addr + (unsigned int)((lane / 16 / 8 * 512 + lane % 16 * 8 + (lane / 16 % 8 * 16 ^ (lane % 16 & 7) << 4) / 16) * 16))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(out_acc[0]), "+f"(out_acc[1]), "+f"(out_acc[2]), "+f"(out_acc[3])
                : "r"(p_frag[0]), "r"(p_frag[1]), "r"(p_frag[2]), "r"(p_frag[3]), "r"(v_frag[0]), "r"(v_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(out_acc[4]), "+f"(out_acc[(4) + 1]), "+f"(out_acc[(4) + 2]), "+f"(out_acc[(4) + 3])
                : "r"(p_frag[0]), "r"(p_frag[1]), "r"(p_frag[2]), "r"(p_frag[3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(v_smem_addr + (unsigned int)(((lane / 16 + 2) / 8 * 512 + lane % 16 * 8 + ((lane / 16 + 2) % 8 * 16 ^ (lane % 16 & 7) << 4) / 16) * 16))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(out_acc[8]), "+f"(out_acc[(8) + 1]), "+f"(out_acc[(8) + 2]), "+f"(out_acc[(8) + 3])
                : "r"(p_frag[0]), "r"(p_frag[1]), "r"(p_frag[2]), "r"(p_frag[3]), "r"(v_frag[0]), "r"(v_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(out_acc[12]), "+f"(out_acc[(12) + 1]), "+f"(out_acc[(12) + 2]), "+f"(out_acc[(12) + 3])
                : "r"(p_frag[0]), "r"(p_frag[1]), "r"(p_frag[2]), "r"(p_frag[3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(v_smem_addr + (unsigned int)(((lane / 16 + 4) / 8 * 512 + lane % 16 * 8 + ((lane / 16 + 4) % 8 * 16 ^ (lane % 16 & 7) << 4) / 16) * 16))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(out_acc[16]), "+f"(out_acc[(16) + 1]), "+f"(out_acc[(16) + 2]), "+f"(out_acc[(16) + 3])
                : "r"(p_frag[0]), "r"(p_frag[1]), "r"(p_frag[2]), "r"(p_frag[3]), "r"(v_frag[0]), "r"(v_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(out_acc[20]), "+f"(out_acc[(20) + 1]), "+f"(out_acc[(20) + 2]), "+f"(out_acc[(20) + 3])
                : "r"(p_frag[0]), "r"(p_frag[1]), "r"(p_frag[2]), "r"(p_frag[3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(v_smem_addr + (unsigned int)(((lane / 16 + 6) / 8 * 512 + lane % 16 * 8 + ((lane / 16 + 6) % 8 * 16 ^ (lane % 16 & 7) << 4) / 16) * 16))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(out_acc[24]), "+f"(out_acc[(24) + 1]), "+f"(out_acc[(24) + 2]), "+f"(out_acc[(24) + 3])
                : "r"(p_frag[0]), "r"(p_frag[1]), "r"(p_frag[2]), "r"(p_frag[3]), "r"(v_frag[0]), "r"(v_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(out_acc[28]), "+f"(out_acc[(28) + 1]), "+f"(out_acc[(28) + 2]), "+f"(out_acc[(28) + 3])
                : "r"(p_frag[0]), "r"(p_frag[1]), "r"(p_frag[2]), "r"(p_frag[3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(v_smem_addr + (unsigned int)(((lane / 16 + 8) / 8 * 512 + lane % 16 * 8 + ((lane / 16 + 8) % 8 * 16 ^ (lane % 16 & 7) << 4) / 16) * 16))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(out_acc[32]), "+f"(out_acc[(32) + 1]), "+f"(out_acc[(32) + 2]), "+f"(out_acc[(32) + 3])
                : "r"(p_frag[0]), "r"(p_frag[1]), "r"(p_frag[2]), "r"(p_frag[3]), "r"(v_frag[0]), "r"(v_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(out_acc[36]), "+f"(out_acc[(36) + 1]), "+f"(out_acc[(36) + 2]), "+f"(out_acc[(36) + 3])
                : "r"(p_frag[0]), "r"(p_frag[1]), "r"(p_frag[2]), "r"(p_frag[3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(v_smem_addr + (unsigned int)(((lane / 16 + 10) / 8 * 512 + lane % 16 * 8 + ((lane / 16 + 10) % 8 * 16 ^ (lane % 16 & 7) << 4) / 16) * 16))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(out_acc[40]), "+f"(out_acc[(40) + 1]), "+f"(out_acc[(40) + 2]), "+f"(out_acc[(40) + 3])
                : "r"(p_frag[0]), "r"(p_frag[1]), "r"(p_frag[2]), "r"(p_frag[3]), "r"(v_frag[0]), "r"(v_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(out_acc[44]), "+f"(out_acc[(44) + 1]), "+f"(out_acc[(44) + 2]), "+f"(out_acc[(44) + 3])
                : "r"(p_frag[0]), "r"(p_frag[1]), "r"(p_frag[2]), "r"(p_frag[3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(v_smem_addr + (unsigned int)(((lane / 16 + 12) / 8 * 512 + lane % 16 * 8 + ((lane / 16 + 12) % 8 * 16 ^ (lane % 16 & 7) << 4) / 16) * 16))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(out_acc[48]), "+f"(out_acc[(48) + 1]), "+f"(out_acc[(48) + 2]), "+f"(out_acc[(48) + 3])
                : "r"(p_frag[0]), "r"(p_frag[1]), "r"(p_frag[2]), "r"(p_frag[3]), "r"(v_frag[0]), "r"(v_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(out_acc[52]), "+f"(out_acc[(52) + 1]), "+f"(out_acc[(52) + 2]), "+f"(out_acc[(52) + 3])
                : "r"(p_frag[0]), "r"(p_frag[1]), "r"(p_frag[2]), "r"(p_frag[3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(v_smem_addr + (unsigned int)(((lane / 16 + 14) / 8 * 512 + lane % 16 * 8 + ((lane / 16 + 14) % 8 * 16 ^ (lane % 16 & 7) << 4) / 16) * 16))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(out_acc[56]), "+f"(out_acc[(56) + 1]), "+f"(out_acc[(56) + 2]), "+f"(out_acc[(56) + 3])
                : "r"(p_frag[0]), "r"(p_frag[1]), "r"(p_frag[2]), "r"(p_frag[3]), "r"(v_frag[0]), "r"(v_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(out_acc[60]), "+f"(out_acc[(60) + 1]), "+f"(out_acc[(60) + 2]), "+f"(out_acc[(60) + 3])
                : "r"(p_frag[0]), "r"(p_frag[1]), "r"(p_frag[2]), "r"(p_frag[3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(v_smem_addr + (unsigned int)((lane / 16 / 8 * 512 + (16 + lane % 16) * 8 + (lane / 16 % 8 * 16 ^ (16 + lane % 16 & 7) << 4) / 16) * 16))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(out_acc[0]), "+f"(out_acc[1]), "+f"(out_acc[2]), "+f"(out_acc[3])
                : "r"(p_frag[4]), "r"(p_frag[(4) + 1]), "r"(p_frag[(4) + 2]), "r"(p_frag[(4) + 3]), "r"(v_frag[0]), "r"(v_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(out_acc[4]), "+f"(out_acc[(4) + 1]), "+f"(out_acc[(4) + 2]), "+f"(out_acc[(4) + 3])
                : "r"(p_frag[4]), "r"(p_frag[(4) + 1]), "r"(p_frag[(4) + 2]), "r"(p_frag[(4) + 3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(v_smem_addr + (unsigned int)(((lane / 16 + 2) / 8 * 512 + (16 + lane % 16) * 8 + ((lane / 16 + 2) % 8 * 16 ^ (16 + lane % 16 & 7) << 4) / 16) * 16))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(out_acc[8]), "+f"(out_acc[(8) + 1]), "+f"(out_acc[(8) + 2]), "+f"(out_acc[(8) + 3])
                : "r"(p_frag[4]), "r"(p_frag[(4) + 1]), "r"(p_frag[(4) + 2]), "r"(p_frag[(4) + 3]), "r"(v_frag[0]), "r"(v_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(out_acc[12]), "+f"(out_acc[(12) + 1]), "+f"(out_acc[(12) + 2]), "+f"(out_acc[(12) + 3])
                : "r"(p_frag[4]), "r"(p_frag[(4) + 1]), "r"(p_frag[(4) + 2]), "r"(p_frag[(4) + 3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(v_smem_addr + (unsigned int)(((lane / 16 + 4) / 8 * 512 + (16 + lane % 16) * 8 + ((lane / 16 + 4) % 8 * 16 ^ (16 + lane % 16 & 7) << 4) / 16) * 16))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(out_acc[16]), "+f"(out_acc[(16) + 1]), "+f"(out_acc[(16) + 2]), "+f"(out_acc[(16) + 3])
                : "r"(p_frag[4]), "r"(p_frag[(4) + 1]), "r"(p_frag[(4) + 2]), "r"(p_frag[(4) + 3]), "r"(v_frag[0]), "r"(v_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(out_acc[20]), "+f"(out_acc[(20) + 1]), "+f"(out_acc[(20) + 2]), "+f"(out_acc[(20) + 3])
                : "r"(p_frag[4]), "r"(p_frag[(4) + 1]), "r"(p_frag[(4) + 2]), "r"(p_frag[(4) + 3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(v_smem_addr + (unsigned int)(((lane / 16 + 6) / 8 * 512 + (16 + lane % 16) * 8 + ((lane / 16 + 6) % 8 * 16 ^ (16 + lane % 16 & 7) << 4) / 16) * 16))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(out_acc[24]), "+f"(out_acc[(24) + 1]), "+f"(out_acc[(24) + 2]), "+f"(out_acc[(24) + 3])
                : "r"(p_frag[4]), "r"(p_frag[(4) + 1]), "r"(p_frag[(4) + 2]), "r"(p_frag[(4) + 3]), "r"(v_frag[0]), "r"(v_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(out_acc[28]), "+f"(out_acc[(28) + 1]), "+f"(out_acc[(28) + 2]), "+f"(out_acc[(28) + 3])
                : "r"(p_frag[4]), "r"(p_frag[(4) + 1]), "r"(p_frag[(4) + 2]), "r"(p_frag[(4) + 3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(v_smem_addr + (unsigned int)(((lane / 16 + 8) / 8 * 512 + (16 + lane % 16) * 8 + ((lane / 16 + 8) % 8 * 16 ^ (16 + lane % 16 & 7) << 4) / 16) * 16))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(out_acc[32]), "+f"(out_acc[(32) + 1]), "+f"(out_acc[(32) + 2]), "+f"(out_acc[(32) + 3])
                : "r"(p_frag[4]), "r"(p_frag[(4) + 1]), "r"(p_frag[(4) + 2]), "r"(p_frag[(4) + 3]), "r"(v_frag[0]), "r"(v_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(out_acc[36]), "+f"(out_acc[(36) + 1]), "+f"(out_acc[(36) + 2]), "+f"(out_acc[(36) + 3])
                : "r"(p_frag[4]), "r"(p_frag[(4) + 1]), "r"(p_frag[(4) + 2]), "r"(p_frag[(4) + 3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(v_smem_addr + (unsigned int)(((lane / 16 + 10) / 8 * 512 + (16 + lane % 16) * 8 + ((lane / 16 + 10) % 8 * 16 ^ (16 + lane % 16 & 7) << 4) / 16) * 16))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(out_acc[40]), "+f"(out_acc[(40) + 1]), "+f"(out_acc[(40) + 2]), "+f"(out_acc[(40) + 3])
                : "r"(p_frag[4]), "r"(p_frag[(4) + 1]), "r"(p_frag[(4) + 2]), "r"(p_frag[(4) + 3]), "r"(v_frag[0]), "r"(v_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(out_acc[44]), "+f"(out_acc[(44) + 1]), "+f"(out_acc[(44) + 2]), "+f"(out_acc[(44) + 3])
                : "r"(p_frag[4]), "r"(p_frag[(4) + 1]), "r"(p_frag[(4) + 2]), "r"(p_frag[(4) + 3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(v_smem_addr + (unsigned int)(((lane / 16 + 12) / 8 * 512 + (16 + lane % 16) * 8 + ((lane / 16 + 12) % 8 * 16 ^ (16 + lane % 16 & 7) << 4) / 16) * 16))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(out_acc[48]), "+f"(out_acc[(48) + 1]), "+f"(out_acc[(48) + 2]), "+f"(out_acc[(48) + 3])
                : "r"(p_frag[4]), "r"(p_frag[(4) + 1]), "r"(p_frag[(4) + 2]), "r"(p_frag[(4) + 3]), "r"(v_frag[0]), "r"(v_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(out_acc[52]), "+f"(out_acc[(52) + 1]), "+f"(out_acc[(52) + 2]), "+f"(out_acc[(52) + 3])
                : "r"(p_frag[4]), "r"(p_frag[(4) + 1]), "r"(p_frag[(4) + 2]), "r"(p_frag[(4) + 3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(v_smem_addr + (unsigned int)(((lane / 16 + 14) / 8 * 512 + (16 + lane % 16) * 8 + ((lane / 16 + 14) % 8 * 16 ^ (16 + lane % 16 & 7) << 4) / 16) * 16))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(out_acc[56]), "+f"(out_acc[(56) + 1]), "+f"(out_acc[(56) + 2]), "+f"(out_acc[(56) + 3])
                : "r"(p_frag[4]), "r"(p_frag[(4) + 1]), "r"(p_frag[(4) + 2]), "r"(p_frag[(4) + 3]), "r"(v_frag[0]), "r"(v_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(out_acc[60]), "+f"(out_acc[(60) + 1]), "+f"(out_acc[(60) + 2]), "+f"(out_acc[(60) + 3])
                : "r"(p_frag[4]), "r"(p_frag[(4) + 1]), "r"(p_frag[(4) + 2]), "r"(p_frag[(4) + 3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(v_smem_addr + (unsigned int)((lane / 16 / 8 * 512 + (32 + lane % 16) * 8 + (lane / 16 % 8 * 16 ^ (32 + lane % 16 & 7) << 4) / 16) * 16))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(out_acc[0]), "+f"(out_acc[1]), "+f"(out_acc[2]), "+f"(out_acc[3])
                : "r"(p_frag[8]), "r"(p_frag[(8) + 1]), "r"(p_frag[(8) + 2]), "r"(p_frag[(8) + 3]), "r"(v_frag[0]), "r"(v_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(out_acc[4]), "+f"(out_acc[(4) + 1]), "+f"(out_acc[(4) + 2]), "+f"(out_acc[(4) + 3])
                : "r"(p_frag[8]), "r"(p_frag[(8) + 1]), "r"(p_frag[(8) + 2]), "r"(p_frag[(8) + 3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(v_smem_addr + (unsigned int)(((lane / 16 + 2) / 8 * 512 + (32 + lane % 16) * 8 + ((lane / 16 + 2) % 8 * 16 ^ (32 + lane % 16 & 7) << 4) / 16) * 16))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(out_acc[8]), "+f"(out_acc[(8) + 1]), "+f"(out_acc[(8) + 2]), "+f"(out_acc[(8) + 3])
                : "r"(p_frag[8]), "r"(p_frag[(8) + 1]), "r"(p_frag[(8) + 2]), "r"(p_frag[(8) + 3]), "r"(v_frag[0]), "r"(v_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(out_acc[12]), "+f"(out_acc[(12) + 1]), "+f"(out_acc[(12) + 2]), "+f"(out_acc[(12) + 3])
                : "r"(p_frag[8]), "r"(p_frag[(8) + 1]), "r"(p_frag[(8) + 2]), "r"(p_frag[(8) + 3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(v_smem_addr + (unsigned int)(((lane / 16 + 4) / 8 * 512 + (32 + lane % 16) * 8 + ((lane / 16 + 4) % 8 * 16 ^ (32 + lane % 16 & 7) << 4) / 16) * 16))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(out_acc[16]), "+f"(out_acc[(16) + 1]), "+f"(out_acc[(16) + 2]), "+f"(out_acc[(16) + 3])
                : "r"(p_frag[8]), "r"(p_frag[(8) + 1]), "r"(p_frag[(8) + 2]), "r"(p_frag[(8) + 3]), "r"(v_frag[0]), "r"(v_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(out_acc[20]), "+f"(out_acc[(20) + 1]), "+f"(out_acc[(20) + 2]), "+f"(out_acc[(20) + 3])
                : "r"(p_frag[8]), "r"(p_frag[(8) + 1]), "r"(p_frag[(8) + 2]), "r"(p_frag[(8) + 3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(v_smem_addr + (unsigned int)(((lane / 16 + 6) / 8 * 512 + (32 + lane % 16) * 8 + ((lane / 16 + 6) % 8 * 16 ^ (32 + lane % 16 & 7) << 4) / 16) * 16))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(out_acc[24]), "+f"(out_acc[(24) + 1]), "+f"(out_acc[(24) + 2]), "+f"(out_acc[(24) + 3])
                : "r"(p_frag[8]), "r"(p_frag[(8) + 1]), "r"(p_frag[(8) + 2]), "r"(p_frag[(8) + 3]), "r"(v_frag[0]), "r"(v_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(out_acc[28]), "+f"(out_acc[(28) + 1]), "+f"(out_acc[(28) + 2]), "+f"(out_acc[(28) + 3])
                : "r"(p_frag[8]), "r"(p_frag[(8) + 1]), "r"(p_frag[(8) + 2]), "r"(p_frag[(8) + 3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(v_smem_addr + (unsigned int)(((lane / 16 + 8) / 8 * 512 + (32 + lane % 16) * 8 + ((lane / 16 + 8) % 8 * 16 ^ (32 + lane % 16 & 7) << 4) / 16) * 16))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(out_acc[32]), "+f"(out_acc[(32) + 1]), "+f"(out_acc[(32) + 2]), "+f"(out_acc[(32) + 3])
                : "r"(p_frag[8]), "r"(p_frag[(8) + 1]), "r"(p_frag[(8) + 2]), "r"(p_frag[(8) + 3]), "r"(v_frag[0]), "r"(v_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(out_acc[36]), "+f"(out_acc[(36) + 1]), "+f"(out_acc[(36) + 2]), "+f"(out_acc[(36) + 3])
                : "r"(p_frag[8]), "r"(p_frag[(8) + 1]), "r"(p_frag[(8) + 2]), "r"(p_frag[(8) + 3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(v_smem_addr + (unsigned int)(((lane / 16 + 10) / 8 * 512 + (32 + lane % 16) * 8 + ((lane / 16 + 10) % 8 * 16 ^ (32 + lane % 16 & 7) << 4) / 16) * 16))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(out_acc[40]), "+f"(out_acc[(40) + 1]), "+f"(out_acc[(40) + 2]), "+f"(out_acc[(40) + 3])
                : "r"(p_frag[8]), "r"(p_frag[(8) + 1]), "r"(p_frag[(8) + 2]), "r"(p_frag[(8) + 3]), "r"(v_frag[0]), "r"(v_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(out_acc[44]), "+f"(out_acc[(44) + 1]), "+f"(out_acc[(44) + 2]), "+f"(out_acc[(44) + 3])
                : "r"(p_frag[8]), "r"(p_frag[(8) + 1]), "r"(p_frag[(8) + 2]), "r"(p_frag[(8) + 3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(v_smem_addr + (unsigned int)(((lane / 16 + 12) / 8 * 512 + (32 + lane % 16) * 8 + ((lane / 16 + 12) % 8 * 16 ^ (32 + lane % 16 & 7) << 4) / 16) * 16))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(out_acc[48]), "+f"(out_acc[(48) + 1]), "+f"(out_acc[(48) + 2]), "+f"(out_acc[(48) + 3])
                : "r"(p_frag[8]), "r"(p_frag[(8) + 1]), "r"(p_frag[(8) + 2]), "r"(p_frag[(8) + 3]), "r"(v_frag[0]), "r"(v_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(out_acc[52]), "+f"(out_acc[(52) + 1]), "+f"(out_acc[(52) + 2]), "+f"(out_acc[(52) + 3])
                : "r"(p_frag[8]), "r"(p_frag[(8) + 1]), "r"(p_frag[(8) + 2]), "r"(p_frag[(8) + 3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(v_smem_addr + (unsigned int)(((lane / 16 + 14) / 8 * 512 + (32 + lane % 16) * 8 + ((lane / 16 + 14) % 8 * 16 ^ (32 + lane % 16 & 7) << 4) / 16) * 16))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(out_acc[56]), "+f"(out_acc[(56) + 1]), "+f"(out_acc[(56) + 2]), "+f"(out_acc[(56) + 3])
                : "r"(p_frag[8]), "r"(p_frag[(8) + 1]), "r"(p_frag[(8) + 2]), "r"(p_frag[(8) + 3]), "r"(v_frag[0]), "r"(v_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(out_acc[60]), "+f"(out_acc[(60) + 1]), "+f"(out_acc[(60) + 2]), "+f"(out_acc[(60) + 3])
                : "r"(p_frag[8]), "r"(p_frag[(8) + 1]), "r"(p_frag[(8) + 2]), "r"(p_frag[(8) + 3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(v_smem_addr + (unsigned int)((lane / 16 / 8 * 512 + (48 + lane % 16) * 8 + (lane / 16 % 8 * 16 ^ (48 + lane % 16 & 7) << 4) / 16) * 16))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(out_acc[0]), "+f"(out_acc[1]), "+f"(out_acc[2]), "+f"(out_acc[3])
                : "r"(p_frag[12]), "r"(p_frag[(12) + 1]), "r"(p_frag[(12) + 2]), "r"(p_frag[(12) + 3]), "r"(v_frag[0]), "r"(v_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(out_acc[4]), "+f"(out_acc[(4) + 1]), "+f"(out_acc[(4) + 2]), "+f"(out_acc[(4) + 3])
                : "r"(p_frag[12]), "r"(p_frag[(12) + 1]), "r"(p_frag[(12) + 2]), "r"(p_frag[(12) + 3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(v_smem_addr + (unsigned int)(((lane / 16 + 2) / 8 * 512 + (48 + lane % 16) * 8 + ((lane / 16 + 2) % 8 * 16 ^ (48 + lane % 16 & 7) << 4) / 16) * 16))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(out_acc[8]), "+f"(out_acc[(8) + 1]), "+f"(out_acc[(8) + 2]), "+f"(out_acc[(8) + 3])
                : "r"(p_frag[12]), "r"(p_frag[(12) + 1]), "r"(p_frag[(12) + 2]), "r"(p_frag[(12) + 3]), "r"(v_frag[0]), "r"(v_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(out_acc[12]), "+f"(out_acc[(12) + 1]), "+f"(out_acc[(12) + 2]), "+f"(out_acc[(12) + 3])
                : "r"(p_frag[12]), "r"(p_frag[(12) + 1]), "r"(p_frag[(12) + 2]), "r"(p_frag[(12) + 3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(v_smem_addr + (unsigned int)(((lane / 16 + 4) / 8 * 512 + (48 + lane % 16) * 8 + ((lane / 16 + 4) % 8 * 16 ^ (48 + lane % 16 & 7) << 4) / 16) * 16))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(out_acc[16]), "+f"(out_acc[(16) + 1]), "+f"(out_acc[(16) + 2]), "+f"(out_acc[(16) + 3])
                : "r"(p_frag[12]), "r"(p_frag[(12) + 1]), "r"(p_frag[(12) + 2]), "r"(p_frag[(12) + 3]), "r"(v_frag[0]), "r"(v_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(out_acc[20]), "+f"(out_acc[(20) + 1]), "+f"(out_acc[(20) + 2]), "+f"(out_acc[(20) + 3])
                : "r"(p_frag[12]), "r"(p_frag[(12) + 1]), "r"(p_frag[(12) + 2]), "r"(p_frag[(12) + 3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(v_smem_addr + (unsigned int)(((lane / 16 + 6) / 8 * 512 + (48 + lane % 16) * 8 + ((lane / 16 + 6) % 8 * 16 ^ (48 + lane % 16 & 7) << 4) / 16) * 16))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(out_acc[24]), "+f"(out_acc[(24) + 1]), "+f"(out_acc[(24) + 2]), "+f"(out_acc[(24) + 3])
                : "r"(p_frag[12]), "r"(p_frag[(12) + 1]), "r"(p_frag[(12) + 2]), "r"(p_frag[(12) + 3]), "r"(v_frag[0]), "r"(v_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(out_acc[28]), "+f"(out_acc[(28) + 1]), "+f"(out_acc[(28) + 2]), "+f"(out_acc[(28) + 3])
                : "r"(p_frag[12]), "r"(p_frag[(12) + 1]), "r"(p_frag[(12) + 2]), "r"(p_frag[(12) + 3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(v_smem_addr + (unsigned int)(((lane / 16 + 8) / 8 * 512 + (48 + lane % 16) * 8 + ((lane / 16 + 8) % 8 * 16 ^ (48 + lane % 16 & 7) << 4) / 16) * 16))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(out_acc[32]), "+f"(out_acc[(32) + 1]), "+f"(out_acc[(32) + 2]), "+f"(out_acc[(32) + 3])
                : "r"(p_frag[12]), "r"(p_frag[(12) + 1]), "r"(p_frag[(12) + 2]), "r"(p_frag[(12) + 3]), "r"(v_frag[0]), "r"(v_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(out_acc[36]), "+f"(out_acc[(36) + 1]), "+f"(out_acc[(36) + 2]), "+f"(out_acc[(36) + 3])
                : "r"(p_frag[12]), "r"(p_frag[(12) + 1]), "r"(p_frag[(12) + 2]), "r"(p_frag[(12) + 3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(v_smem_addr + (unsigned int)(((lane / 16 + 10) / 8 * 512 + (48 + lane % 16) * 8 + ((lane / 16 + 10) % 8 * 16 ^ (48 + lane % 16 & 7) << 4) / 16) * 16))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(out_acc[40]), "+f"(out_acc[(40) + 1]), "+f"(out_acc[(40) + 2]), "+f"(out_acc[(40) + 3])
                : "r"(p_frag[12]), "r"(p_frag[(12) + 1]), "r"(p_frag[(12) + 2]), "r"(p_frag[(12) + 3]), "r"(v_frag[0]), "r"(v_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(out_acc[44]), "+f"(out_acc[(44) + 1]), "+f"(out_acc[(44) + 2]), "+f"(out_acc[(44) + 3])
                : "r"(p_frag[12]), "r"(p_frag[(12) + 1]), "r"(p_frag[(12) + 2]), "r"(p_frag[(12) + 3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(v_smem_addr + (unsigned int)(((lane / 16 + 12) / 8 * 512 + (48 + lane % 16) * 8 + ((lane / 16 + 12) % 8 * 16 ^ (48 + lane % 16 & 7) << 4) / 16) * 16))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(out_acc[48]), "+f"(out_acc[(48) + 1]), "+f"(out_acc[(48) + 2]), "+f"(out_acc[(48) + 3])
                : "r"(p_frag[12]), "r"(p_frag[(12) + 1]), "r"(p_frag[(12) + 2]), "r"(p_frag[(12) + 3]), "r"(v_frag[0]), "r"(v_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(out_acc[52]), "+f"(out_acc[(52) + 1]), "+f"(out_acc[(52) + 2]), "+f"(out_acc[(52) + 3])
                : "r"(p_frag[12]), "r"(p_frag[(12) + 1]), "r"(p_frag[(12) + 2]), "r"(p_frag[(12) + 3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(v_smem_addr + (unsigned int)(((lane / 16 + 14) / 8 * 512 + (48 + lane % 16) * 8 + ((lane / 16 + 14) % 8 * 16 ^ (48 + lane % 16 & 7) << 4) / 16) * 16))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(out_acc[56]), "+f"(out_acc[(56) + 1]), "+f"(out_acc[(56) + 2]), "+f"(out_acc[(56) + 3])
                : "r"(p_frag[12]), "r"(p_frag[(12) + 1]), "r"(p_frag[(12) + 2]), "r"(p_frag[(12) + 3]), "r"(v_frag[0]), "r"(v_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(out_acc[60]), "+f"(out_acc[(60) + 1]), "+f"(out_acc[(60) + 2]), "+f"(out_acc[(60) + 3])
                : "r"(p_frag[12]), "r"(p_frag[(12) + 1]), "r"(p_frag[(12) + 2]), "r"(p_frag[(12) + 3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
            __syncthreads();
            if (last_kv_tile > kv_tile_idx) {
                if (warp == 0) {
                    if (elect_sync()) {
                        mbarrier_arrive_expect_tx(v_full_addr, 16384);
                        asm volatile(
                            "cp.async.bulk.tensor.4d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                            " [%0], [%1, {%2, %3, %4, %5}], [%6], %7;"
                            :: "r"(v_smem_addr), "l"((&V_map)), "r"(0), "r"(next_kv_tile_base), "r"(head_idx), "r"(0),
                               "r"(v_full_addr), "l"(0x14F0000000000000ULL) : "memory");
                    }
                }
            }
        }
        float _shfl_xor_4 = __shfl_xor_sync(0xFFFFFFFF, d_state[0], 2);
        float peer_d = _shfl_xor_4;
        d_state[0] = d_state[0] + peer_d;
        float _shfl_xor_5 = __shfl_xor_sync(0xFFFFFFFF, d_state[0], 1);
        float peer_d2 = _shfl_xor_5;
        d_state[0] = d_state[0] + peer_d2;
        float _shfl_xor_6 = __shfl_xor_sync(0xFFFFFFFF, d_state[1], 2);
        float peer_d_7 = _shfl_xor_6;
        d_state[1] = d_state[1] + peer_d_7;
        float _shfl_xor_7 = __shfl_xor_sync(0xFFFFFFFF, d_state[1], 1);
        float peer_d2_8 = _shfl_xor_7;
        d_state[1] = d_state[1] + peer_d2_8;
        float _rcp_0 = approx_rcp(d_state[0]);
        float _rcp_1 = approx_rcp(d_state[1]);
        {
            float2 _pair_scale_even2_8 = make_float2(_rcp_0, _rcp_0);
            float2 _pair_scale_odd2_8 = make_float2(_rcp_1, _rcp_1);
            float2* _pair_scale_src2_8 = reinterpret_cast<float2*>(&(out_acc + 0)[0]);
            #if __CUDA_ARCH__ >= 1000
            asm volatile("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_8[0]) : "l"(*(unsigned long long*)&_pair_scale_even2_8));
            asm volatile("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_8[1]) : "l"(*(unsigned long long*)&_pair_scale_odd2_8));
            asm volatile("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_8[2]) : "l"(*(unsigned long long*)&_pair_scale_even2_8));
            asm volatile("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_8[3]) : "l"(*(unsigned long long*)&_pair_scale_odd2_8));
            #else
            (out_acc + 0)[0] *= _rcp_0;
            (out_acc + 0)[1] *= _rcp_0;
            (out_acc + 0)[2] *= _rcp_1;
            (out_acc + 0)[3] *= _rcp_1;
            (out_acc + 0)[4] *= _rcp_0;
            (out_acc + 0)[5] *= _rcp_0;
            (out_acc + 0)[6] *= _rcp_1;
            (out_acc + 0)[7] *= _rcp_1;
            #endif
        }
        {
            float2 _pair_scale_even2_9 = make_float2(_rcp_0, _rcp_0);
            float2 _pair_scale_odd2_9 = make_float2(_rcp_1, _rcp_1);
            float2* _pair_scale_src2_9 = reinterpret_cast<float2*>(&(out_acc + 8)[0]);
            #if __CUDA_ARCH__ >= 1000
            asm volatile("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_9[0]) : "l"(*(unsigned long long*)&_pair_scale_even2_9));
            asm volatile("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_9[1]) : "l"(*(unsigned long long*)&_pair_scale_odd2_9));
            asm volatile("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_9[2]) : "l"(*(unsigned long long*)&_pair_scale_even2_9));
            asm volatile("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_9[3]) : "l"(*(unsigned long long*)&_pair_scale_odd2_9));
            #else
            (out_acc + 8)[0] *= _rcp_0;
            (out_acc + 8)[1] *= _rcp_0;
            (out_acc + 8)[2] *= _rcp_1;
            (out_acc + 8)[3] *= _rcp_1;
            (out_acc + 8)[4] *= _rcp_0;
            (out_acc + 8)[5] *= _rcp_0;
            (out_acc + 8)[6] *= _rcp_1;
            (out_acc + 8)[7] *= _rcp_1;
            #endif
        }
        {
            float2 _pair_scale_even2_10 = make_float2(_rcp_0, _rcp_0);
            float2 _pair_scale_odd2_10 = make_float2(_rcp_1, _rcp_1);
            float2* _pair_scale_src2_10 = reinterpret_cast<float2*>(&(out_acc + 16)[0]);
            #if __CUDA_ARCH__ >= 1000
            asm volatile("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_10[0]) : "l"(*(unsigned long long*)&_pair_scale_even2_10));
            asm volatile("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_10[1]) : "l"(*(unsigned long long*)&_pair_scale_odd2_10));
            asm volatile("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_10[2]) : "l"(*(unsigned long long*)&_pair_scale_even2_10));
            asm volatile("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_10[3]) : "l"(*(unsigned long long*)&_pair_scale_odd2_10));
            #else
            (out_acc + 16)[0] *= _rcp_0;
            (out_acc + 16)[1] *= _rcp_0;
            (out_acc + 16)[2] *= _rcp_1;
            (out_acc + 16)[3] *= _rcp_1;
            (out_acc + 16)[4] *= _rcp_0;
            (out_acc + 16)[5] *= _rcp_0;
            (out_acc + 16)[6] *= _rcp_1;
            (out_acc + 16)[7] *= _rcp_1;
            #endif
        }
        {
            float2 _pair_scale_even2_11 = make_float2(_rcp_0, _rcp_0);
            float2 _pair_scale_odd2_11 = make_float2(_rcp_1, _rcp_1);
            float2* _pair_scale_src2_11 = reinterpret_cast<float2*>(&(out_acc + 24)[0]);
            #if __CUDA_ARCH__ >= 1000
            asm volatile("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_11[0]) : "l"(*(unsigned long long*)&_pair_scale_even2_11));
            asm volatile("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_11[1]) : "l"(*(unsigned long long*)&_pair_scale_odd2_11));
            asm volatile("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_11[2]) : "l"(*(unsigned long long*)&_pair_scale_even2_11));
            asm volatile("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_11[3]) : "l"(*(unsigned long long*)&_pair_scale_odd2_11));
            #else
            (out_acc + 24)[0] *= _rcp_0;
            (out_acc + 24)[1] *= _rcp_0;
            (out_acc + 24)[2] *= _rcp_1;
            (out_acc + 24)[3] *= _rcp_1;
            (out_acc + 24)[4] *= _rcp_0;
            (out_acc + 24)[5] *= _rcp_0;
            (out_acc + 24)[6] *= _rcp_1;
            (out_acc + 24)[7] *= _rcp_1;
            #endif
        }
        {
            float2 _pair_scale_even2_12 = make_float2(_rcp_0, _rcp_0);
            float2 _pair_scale_odd2_12 = make_float2(_rcp_1, _rcp_1);
            float2* _pair_scale_src2_12 = reinterpret_cast<float2*>(&(out_acc + 32)[0]);
            #if __CUDA_ARCH__ >= 1000
            asm volatile("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_12[0]) : "l"(*(unsigned long long*)&_pair_scale_even2_12));
            asm volatile("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_12[1]) : "l"(*(unsigned long long*)&_pair_scale_odd2_12));
            asm volatile("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_12[2]) : "l"(*(unsigned long long*)&_pair_scale_even2_12));
            asm volatile("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_12[3]) : "l"(*(unsigned long long*)&_pair_scale_odd2_12));
            #else
            (out_acc + 32)[0] *= _rcp_0;
            (out_acc + 32)[1] *= _rcp_0;
            (out_acc + 32)[2] *= _rcp_1;
            (out_acc + 32)[3] *= _rcp_1;
            (out_acc + 32)[4] *= _rcp_0;
            (out_acc + 32)[5] *= _rcp_0;
            (out_acc + 32)[6] *= _rcp_1;
            (out_acc + 32)[7] *= _rcp_1;
            #endif
        }
        {
            float2 _pair_scale_even2_13 = make_float2(_rcp_0, _rcp_0);
            float2 _pair_scale_odd2_13 = make_float2(_rcp_1, _rcp_1);
            float2* _pair_scale_src2_13 = reinterpret_cast<float2*>(&(out_acc + 40)[0]);
            #if __CUDA_ARCH__ >= 1000
            asm volatile("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_13[0]) : "l"(*(unsigned long long*)&_pair_scale_even2_13));
            asm volatile("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_13[1]) : "l"(*(unsigned long long*)&_pair_scale_odd2_13));
            asm volatile("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_13[2]) : "l"(*(unsigned long long*)&_pair_scale_even2_13));
            asm volatile("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_13[3]) : "l"(*(unsigned long long*)&_pair_scale_odd2_13));
            #else
            (out_acc + 40)[0] *= _rcp_0;
            (out_acc + 40)[1] *= _rcp_0;
            (out_acc + 40)[2] *= _rcp_1;
            (out_acc + 40)[3] *= _rcp_1;
            (out_acc + 40)[4] *= _rcp_0;
            (out_acc + 40)[5] *= _rcp_0;
            (out_acc + 40)[6] *= _rcp_1;
            (out_acc + 40)[7] *= _rcp_1;
            #endif
        }
        {
            float2 _pair_scale_even2_14 = make_float2(_rcp_0, _rcp_0);
            float2 _pair_scale_odd2_14 = make_float2(_rcp_1, _rcp_1);
            float2* _pair_scale_src2_14 = reinterpret_cast<float2*>(&(out_acc + 48)[0]);
            #if __CUDA_ARCH__ >= 1000
            asm volatile("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_14[0]) : "l"(*(unsigned long long*)&_pair_scale_even2_14));
            asm volatile("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_14[1]) : "l"(*(unsigned long long*)&_pair_scale_odd2_14));
            asm volatile("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_14[2]) : "l"(*(unsigned long long*)&_pair_scale_even2_14));
            asm volatile("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_14[3]) : "l"(*(unsigned long long*)&_pair_scale_odd2_14));
            #else
            (out_acc + 48)[0] *= _rcp_0;
            (out_acc + 48)[1] *= _rcp_0;
            (out_acc + 48)[2] *= _rcp_1;
            (out_acc + 48)[3] *= _rcp_1;
            (out_acc + 48)[4] *= _rcp_0;
            (out_acc + 48)[5] *= _rcp_0;
            (out_acc + 48)[6] *= _rcp_1;
            (out_acc + 48)[7] *= _rcp_1;
            #endif
        }
        {
            float2 _pair_scale_even2_15 = make_float2(_rcp_0, _rcp_0);
            float2 _pair_scale_odd2_15 = make_float2(_rcp_1, _rcp_1);
            float2* _pair_scale_src2_15 = reinterpret_cast<float2*>(&(out_acc + 56)[0]);
            #if __CUDA_ARCH__ >= 1000
            asm volatile("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_15[0]) : "l"(*(unsigned long long*)&_pair_scale_even2_15));
            asm volatile("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_15[1]) : "l"(*(unsigned long long*)&_pair_scale_odd2_15));
            asm volatile("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_15[2]) : "l"(*(unsigned long long*)&_pair_scale_even2_15));
            asm volatile("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_15[3]) : "l"(*(unsigned long long*)&_pair_scale_odd2_15));
            #else
            (out_acc + 56)[0] *= _rcp_0;
            (out_acc + 56)[1] *= _rcp_0;
            (out_acc + 56)[2] *= _rcp_1;
            (out_acc + 56)[3] *= _rcp_1;
            (out_acc + 56)[4] *= _rcp_0;
            (out_acc + 56)[5] *= _rcp_0;
            (out_acc + 56)[6] *= _rcp_1;
            (out_acc + 56)[7] *= _rcp_1;
            #endif
        }
        unsigned int o_pack[4];
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_acc[_lp*2 + 0], out_acc[_lp*2+1 + 0]));
            o_pack[_lp] = *(uint32_t*)&_bf2;
        }
        uint32_t _stmatrix_addr_16 = static_cast<uint32_t>(q_smem_addr + (unsigned int)((lane / 16 / 8 * 512 + (warp * 16 + lane % 16) * 8 + (lane / 16 % 8 * 16 ^ (warp * 16 + lane % 16 & 7) << 4) / 16) * 16));
        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
            :: "r"(_stmatrix_addr_16), "r"(*reinterpret_cast<const uint32_t*>(&o_pack[0])), "r"(*reinterpret_cast<const uint32_t*>(&o_pack[1])), "r"(*reinterpret_cast<const uint32_t*>(&o_pack[2])), "r"(*reinterpret_cast<const uint32_t*>(&o_pack[3]))
            : "memory");
        unsigned int o_pack_9[4];
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_acc[_lp*2 + 8], out_acc[_lp*2+1 + 8]));
            o_pack_9[_lp] = *(uint32_t*)&_bf2;
        }
        uint32_t _stmatrix_addr_17 = static_cast<uint32_t>(q_smem_addr + (unsigned int)(((2 + lane / 16) / 8 * 512 + (warp * 16 + lane % 16) * 8 + ((2 + lane / 16) % 8 * 16 ^ (warp * 16 + lane % 16 & 7) << 4) / 16) * 16));
        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
            :: "r"(_stmatrix_addr_17), "r"(*reinterpret_cast<const uint32_t*>(&o_pack_9[0])), "r"(*reinterpret_cast<const uint32_t*>(&o_pack_9[1])), "r"(*reinterpret_cast<const uint32_t*>(&o_pack_9[2])), "r"(*reinterpret_cast<const uint32_t*>(&o_pack_9[3]))
            : "memory");
        unsigned int o_pack_10[4];
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_acc[_lp*2 + 16], out_acc[_lp*2+1 + 16]));
            o_pack_10[_lp] = *(uint32_t*)&_bf2;
        }
        uint32_t _stmatrix_addr_18 = static_cast<uint32_t>(q_smem_addr + (unsigned int)(((4 + lane / 16) / 8 * 512 + (warp * 16 + lane % 16) * 8 + ((4 + lane / 16) % 8 * 16 ^ (warp * 16 + lane % 16 & 7) << 4) / 16) * 16));
        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
            :: "r"(_stmatrix_addr_18), "r"(*reinterpret_cast<const uint32_t*>(&o_pack_10[0])), "r"(*reinterpret_cast<const uint32_t*>(&o_pack_10[1])), "r"(*reinterpret_cast<const uint32_t*>(&o_pack_10[2])), "r"(*reinterpret_cast<const uint32_t*>(&o_pack_10[3]))
            : "memory");
        unsigned int o_pack_11[4];
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_acc[_lp*2 + 24], out_acc[_lp*2+1 + 24]));
            o_pack_11[_lp] = *(uint32_t*)&_bf2;
        }
        uint32_t _stmatrix_addr_19 = static_cast<uint32_t>(q_smem_addr + (unsigned int)(((6 + lane / 16) / 8 * 512 + (warp * 16 + lane % 16) * 8 + ((6 + lane / 16) % 8 * 16 ^ (warp * 16 + lane % 16 & 7) << 4) / 16) * 16));
        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
            :: "r"(_stmatrix_addr_19), "r"(*reinterpret_cast<const uint32_t*>(&o_pack_11[0])), "r"(*reinterpret_cast<const uint32_t*>(&o_pack_11[1])), "r"(*reinterpret_cast<const uint32_t*>(&o_pack_11[2])), "r"(*reinterpret_cast<const uint32_t*>(&o_pack_11[3]))
            : "memory");
        unsigned int o_pack_12[4];
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_acc[_lp*2 + 32], out_acc[_lp*2+1 + 32]));
            o_pack_12[_lp] = *(uint32_t*)&_bf2;
        }
        uint32_t _stmatrix_addr_20 = static_cast<uint32_t>(q_smem_addr + (unsigned int)(((8 + lane / 16) / 8 * 512 + (warp * 16 + lane % 16) * 8 + ((8 + lane / 16) % 8 * 16 ^ (warp * 16 + lane % 16 & 7) << 4) / 16) * 16));
        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
            :: "r"(_stmatrix_addr_20), "r"(*reinterpret_cast<const uint32_t*>(&o_pack_12[0])), "r"(*reinterpret_cast<const uint32_t*>(&o_pack_12[1])), "r"(*reinterpret_cast<const uint32_t*>(&o_pack_12[2])), "r"(*reinterpret_cast<const uint32_t*>(&o_pack_12[3]))
            : "memory");
        unsigned int o_pack_13[4];
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_acc[_lp*2 + 40], out_acc[_lp*2+1 + 40]));
            o_pack_13[_lp] = *(uint32_t*)&_bf2;
        }
        uint32_t _stmatrix_addr_21 = static_cast<uint32_t>(q_smem_addr + (unsigned int)(((10 + lane / 16) / 8 * 512 + (warp * 16 + lane % 16) * 8 + ((10 + lane / 16) % 8 * 16 ^ (warp * 16 + lane % 16 & 7) << 4) / 16) * 16));
        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
            :: "r"(_stmatrix_addr_21), "r"(*reinterpret_cast<const uint32_t*>(&o_pack_13[0])), "r"(*reinterpret_cast<const uint32_t*>(&o_pack_13[1])), "r"(*reinterpret_cast<const uint32_t*>(&o_pack_13[2])), "r"(*reinterpret_cast<const uint32_t*>(&o_pack_13[3]))
            : "memory");
        unsigned int o_pack_14[4];
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_acc[_lp*2 + 48], out_acc[_lp*2+1 + 48]));
            o_pack_14[_lp] = *(uint32_t*)&_bf2;
        }
        uint32_t _stmatrix_addr_22 = static_cast<uint32_t>(q_smem_addr + (unsigned int)(((12 + lane / 16) / 8 * 512 + (warp * 16 + lane % 16) * 8 + ((12 + lane / 16) % 8 * 16 ^ (warp * 16 + lane % 16 & 7) << 4) / 16) * 16));
        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
            :: "r"(_stmatrix_addr_22), "r"(*reinterpret_cast<const uint32_t*>(&o_pack_14[0])), "r"(*reinterpret_cast<const uint32_t*>(&o_pack_14[1])), "r"(*reinterpret_cast<const uint32_t*>(&o_pack_14[2])), "r"(*reinterpret_cast<const uint32_t*>(&o_pack_14[3]))
            : "memory");
        unsigned int o_pack_15[4];
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_acc[_lp*2 + 56], out_acc[_lp*2+1 + 56]));
            o_pack_15[_lp] = *(uint32_t*)&_bf2;
        }
        uint32_t _stmatrix_addr_23 = static_cast<uint32_t>(q_smem_addr + (unsigned int)(((14 + lane / 16) / 8 * 512 + (warp * 16 + lane % 16) * 8 + ((14 + lane / 16) % 8 * 16 ^ (warp * 16 + lane % 16 & 7) << 4) / 16) * 16));
        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
            :: "r"(_stmatrix_addr_23), "r"(*reinterpret_cast<const uint32_t*>(&o_pack_15[0])), "r"(*reinterpret_cast<const uint32_t*>(&o_pack_15[1])), "r"(*reinterpret_cast<const uint32_t*>(&o_pack_15[2])), "r"(*reinterpret_cast<const uint32_t*>(&o_pack_15[3]))
            : "memory");
        __syncthreads();
        unsigned int o_vec[4];
        int q_pos = q_tile_base + (warp * 16 + lane / 8);
        if (q_pos < tokens) {
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&o_vec[0])), "=r"(*reinterpret_cast<uint32_t*>(&o_vec[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&o_vec[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&o_vec[(0) + 3]))
                : "r"(q_smem_addr + (unsigned int)((lane % 8 / 8 * 512 + (warp * 16 + lane / 8) * 8 + (lane % 8 % 8 * 16 ^ (warp * 16 + lane / 8 & 7) << 4) / 16) * 16)));
            reinterpret_cast<int4*>(O + ((q_pos * NUM_HEADS_CONST + head_idx) * 128 + lane % 8 * 8))[0] = reinterpret_cast<int4*>(o_vec)[0];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&o_vec[0])), "=r"(*reinterpret_cast<uint32_t*>(&o_vec[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&o_vec[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&o_vec[(0) + 3]))
                : "r"(q_smem_addr + (unsigned int)(((lane % 8 + 8) / 8 * 512 + (warp * 16 + lane / 8) * 8 + ((lane % 8 + 8) % 8 * 16 ^ (warp * 16 + lane / 8 & 7) << 4) / 16) * 16)));
            reinterpret_cast<int4*>(O + ((q_pos * NUM_HEADS_CONST + head_idx) * 128 + lane % 8 * 8 + 64))[0] = reinterpret_cast<int4*>(o_vec)[0];
        }
        int q_pos_16 = q_tile_base + (warp * 16 + lane / 8 + 4);
        if (q_pos_16 < tokens) {
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&o_vec[0])), "=r"(*reinterpret_cast<uint32_t*>(&o_vec[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&o_vec[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&o_vec[(0) + 3]))
                : "r"(q_smem_addr + (unsigned int)((lane % 8 / 8 * 512 + (warp * 16 + lane / 8 + 4) * 8 + (lane % 8 % 8 * 16 ^ (warp * 16 + lane / 8 + 4 & 7) << 4) / 16) * 16)));
            reinterpret_cast<int4*>(O + ((q_pos_16 * NUM_HEADS_CONST + head_idx) * 128 + lane % 8 * 8))[0] = reinterpret_cast<int4*>(o_vec)[0];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&o_vec[0])), "=r"(*reinterpret_cast<uint32_t*>(&o_vec[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&o_vec[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&o_vec[(0) + 3]))
                : "r"(q_smem_addr + (unsigned int)(((lane % 8 + 8) / 8 * 512 + (warp * 16 + lane / 8 + 4) * 8 + ((lane % 8 + 8) % 8 * 16 ^ (warp * 16 + lane / 8 + 4 & 7) << 4) / 16) * 16)));
            reinterpret_cast<int4*>(O + ((q_pos_16 * NUM_HEADS_CONST + head_idx) * 128 + lane % 8 * 8 + 64))[0] = reinterpret_cast<int4*>(o_vec)[0];
        }
        int q_pos_17 = q_tile_base + (warp * 16 + lane / 8 + 8);
        if (q_pos_17 < tokens) {
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&o_vec[0])), "=r"(*reinterpret_cast<uint32_t*>(&o_vec[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&o_vec[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&o_vec[(0) + 3]))
                : "r"(q_smem_addr + (unsigned int)((lane % 8 / 8 * 512 + (warp * 16 + lane / 8 + 8) * 8 + (lane % 8 % 8 * 16 ^ (warp * 16 + lane / 8 + 8 & 7) << 4) / 16) * 16)));
            reinterpret_cast<int4*>(O + ((q_pos_17 * NUM_HEADS_CONST + head_idx) * 128 + lane % 8 * 8))[0] = reinterpret_cast<int4*>(o_vec)[0];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&o_vec[0])), "=r"(*reinterpret_cast<uint32_t*>(&o_vec[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&o_vec[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&o_vec[(0) + 3]))
                : "r"(q_smem_addr + (unsigned int)(((lane % 8 + 8) / 8 * 512 + (warp * 16 + lane / 8 + 8) * 8 + ((lane % 8 + 8) % 8 * 16 ^ (warp * 16 + lane / 8 + 8 & 7) << 4) / 16) * 16)));
            reinterpret_cast<int4*>(O + ((q_pos_17 * NUM_HEADS_CONST + head_idx) * 128 + lane % 8 * 8 + 64))[0] = reinterpret_cast<int4*>(o_vec)[0];
        }
        int q_pos_18 = q_tile_base + (warp * 16 + lane / 8 + 12);
        if (q_pos_18 < tokens) {
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&o_vec[0])), "=r"(*reinterpret_cast<uint32_t*>(&o_vec[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&o_vec[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&o_vec[(0) + 3]))
                : "r"(q_smem_addr + (unsigned int)((lane % 8 / 8 * 512 + (warp * 16 + lane / 8 + 12) * 8 + (lane % 8 % 8 * 16 ^ (warp * 16 + lane / 8 + 12 & 7) << 4) / 16) * 16)));
            reinterpret_cast<int4*>(O + ((q_pos_18 * NUM_HEADS_CONST + head_idx) * 128 + lane % 8 * 8))[0] = reinterpret_cast<int4*>(o_vec)[0];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&o_vec[0])), "=r"(*reinterpret_cast<uint32_t*>(&o_vec[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&o_vec[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&o_vec[(0) + 3]))
                : "r"(q_smem_addr + (unsigned int)(((lane % 8 + 8) / 8 * 512 + (warp * 16 + lane / 8 + 12) * 8 + ((lane % 8 + 8) % 8 * 16 ^ (warp * 16 + lane / 8 + 12 & 7) << 4) / 16) * 16)));
            reinterpret_cast<int4*>(O + ((q_pos_18 * NUM_HEADS_CONST + head_idx) * 128 + lane % 8 * 8 + 64))[0] = reinterpret_cast<int4*>(o_vec)[0];
        }
        __syncthreads();
    }

    // Cleanup
    __syncthreads();
}

} // extern "C"

#include <cuda_runtime.h>

#include <algorithm>
#include <mutex>
#include <vector>

#include "tvm_ffi_utils.h"

namespace {

constexpr int64_t kNumHeads = 56;
constexpr int64_t kHeadDim = 128;
constexpr int64_t kWidth = kNumHeads * kHeadDim;
constexpr int64_t kMaxTokens = 131072;
constexpr int kTileQ = 64;
constexpr int kTileKv = 64;
constexpr int kThreads = THREADS;
constexpr int kDynamicSmemBytes = SMEM_TOTAL;
// log2(e): the query is pre-scaled by BF16(1/sqrt(128)), so the softmax scale is 1.0.
constexpr float kSoftmaxScaleLog2 = 1.4426950408889634f;
// Two BF16 copies of 0.08837890625 = BF16(FP32(1/sqrt(128))), bits 0x3DB5.
constexpr unsigned int kQueryScaleBf16x2 = 0x3DB53DB5u;

void CheckRows(const TensorView& tensor, const char* name, int64_t tokens, DLDevice device) {
  TVM_FFI_CHECK(tensor.device().device_type == kDLCUDA, ValueError) << name << " must be a CUDA tensor";
  TVM_FFI_CHECK(tensor.device().device_id == device.device_id, ValueError)
      << name << " must be on the same CUDA device as q";
  TVM_FFI_CHECK(encode_dlpack_dtype(tensor.dtype()) == encode_dlpack_dtype(dl_bfloat16), ValueError)
      << name << " must be bfloat16";
  TVM_FFI_CHECK(tensor.ndim() == 2 && tensor.size(0) == tokens && tensor.size(1) == kWidth, ValueError)
      << name << " must have shape [tokens, 7168]";
  TVM_FFI_CHECK(tensor.stride(1) == 1 && tensor.stride(0) == kWidth, ValueError) << name << " must be contiguous";
  TVM_FFI_CHECK(reinterpret_cast<uintptr_t>(tensor.data_ptr()) % 16 == 0, ValueError)
      << name << " must be 16-byte aligned";
}

// One head's [64 tokens x 128 channels] BF16 tile per request, addressed as the 4-D view
// (64 channels, tokens, heads, 2 channel halves) of the contiguous [tokens, 7168] rows with
// a 128-byte swizzle.  The token box may exceed a short sequence: TMA zero-fills the rows
// beyond the tensor, the kernel masks those keys and never stores those query rows.
CUtensorMap EncodeRowsTile(const TensorView& rows, int64_t tokens, const char* name) {
  uint64_t global_dim[4] = {64, static_cast<uint64_t>(tokens), static_cast<uint64_t>(kNumHeads), 2};
  uint64_t global_strides[3] = {static_cast<uint64_t>(kWidth * 2), static_cast<uint64_t>(kHeadDim * 2), 64 * 2};
  uint32_t box_dim[4] = {64, kTileQ, 1, 2};
  uint32_t element_strides[4] = {1, 1, 1, 1};
  CUtensorMap descriptor{};
  CUresult result = cuTensorMapEncodeTiled(
      &descriptor, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4, rows.data_ptr(), global_dim, global_strides, box_dim,
      element_strides, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_CHECK(result == CUDA_SUCCESS, RuntimeError)
      << "failed to encode the " << name << " tensor map: CUresult=" << static_cast<int>(result);
  return descriptor;
}

int ConfigureKernel() {
  static std::mutex mutex;
  static std::vector<std::pair<int, int>> configured_devices;
  int device = -1;
  cudaError_t status = cudaGetDevice(&device);
  TVM_FFI_CHECK(status == cudaSuccess, RuntimeError)
      << "failed to get the active CUDA device: " << cudaGetErrorString(status);
  std::lock_guard<std::mutex> lock(mutex);
  for (const auto& entry : configured_devices) {
    if (entry.first == device) return entry.second;
  }
  cudaDeviceProp properties{};
  status = cudaGetDeviceProperties(&properties, device);
  TVM_FFI_CHECK(status == cudaSuccess, RuntimeError)
      << "failed to query CUDA device properties: " << cudaGetErrorString(status);
  TVM_FFI_CHECK(properties.major == 12 || properties.major == 10 || properties.major == 9, RuntimeError)
      << "MiniMax-H3 dense attention requires compute capability 12.x (target), 10.x or 9.0";
  status = cudaFuncSetAttribute(kernel_minimax_h3_dense_attention, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                kDynamicSmemBytes);
  TVM_FFI_CHECK(status == cudaSuccess, RuntimeError)
      << "failed to opt in to dynamic shared memory: " << cudaGetErrorString(status);
  configured_devices.emplace_back(device, properties.multiProcessorCount);
  return properties.multiProcessorCount;
}

}  // namespace

// y = softmax(BF16(q * BF16(1/sqrt(128))) @ k^T) @ v per head, batch 1, no mask, no dropout.
// q, k, v, out: contiguous BF16 [tokens, 7168] rows (element [t, h * 128 + d]), 1 <= tokens <= 131072.
void minimax_h3_dense_attention(TensorView q, TensorView k, TensorView v, TensorView out, int64_t ctas_per_sm) {
  const int64_t tokens = q.size(0);
  TVM_FFI_CHECK(q.ndim() == 2 && tokens >= 1 && tokens <= kMaxTokens, ValueError)
      << "q must be [tokens, 7168] with 1 <= tokens <= 131072";
  TVM_FFI_CHECK(ctas_per_sm >= 1 && ctas_per_sm <= 2, ValueError) << "ctas_per_sm must be 1 or 2";
  const DLDevice device = q.device();
  CheckRows(q, "q", tokens, device);
  CheckRows(k, "k", tokens, device);
  CheckRows(v, "v", tokens, device);
  CheckRows(out, "out", tokens, device);

  ffi::CUDADeviceGuard device_guard(device.device_id);
  const cudaStream_t stream = get_stream(device);
  const int num_sms = ConfigureKernel();
  const int num_q_tiles = static_cast<int>((tokens + kTileQ - 1) / kTileQ);
  const int num_kv_tiles = static_cast<int>((tokens + kTileKv - 1) / kTileKv);
  const unsigned int total_work = static_cast<unsigned int>(num_q_tiles) * static_cast<unsigned int>(kNumHeads);
  const int grid = std::max(1, std::min<int>(static_cast<int>(total_work), static_cast<int>(ctas_per_sm) * num_sms));

  const CUtensorMap q_map = EncodeRowsTile(q, tokens, "q");
  const CUtensorMap k_map = EncodeRowsTile(k, tokens, "k");
  const CUtensorMap v_map = EncodeRowsTile(v, tokens, "v");
  kernel_minimax_h3_dense_attention<<<grid, kThreads, kDynamicSmemBytes, stream>>>(
      q_map, k_map, v_map, static_cast<__nv_bfloat16*>(out.data_ptr()), static_cast<int>(tokens), num_q_tiles,
      num_kv_tiles, total_work, kSoftmaxScaleLog2, kQueryScaleBf16x2);
  const cudaError_t status = cudaGetLastError();
  TVM_FFI_CHECK(status == cudaSuccess, RuntimeError)
      << "MiniMax-H3 dense attention launch failed: " << cudaGetErrorString(status);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(minimax_h3_dense_attention, minimax_h3_dense_attention);
