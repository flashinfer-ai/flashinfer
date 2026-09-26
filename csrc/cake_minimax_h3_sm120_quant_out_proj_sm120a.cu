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
// clang-format off
// MiniMax-H3 quantized attention output projection with the fused indexed gate + residual for
// SM120 (GB202: RTX 5090 / RTX PRO 6000 Blackwell), generated from the Cake kernel schedules.
// One launch per call (out_proj_gemm_fused_{fp8,nvfp4}): a persistent 256x128 GEMM (7168 -> 5376,
// mma.sync e4m3 / kind::mxf4nvf4 block-scaled, 3-stage 64-byte-K TMA ring, one 384-thread CTA per
// SM).  Warps 1-3 of the producer warpgroup quantize the BF16 attention output rows [M, 7168] in
// place of a separate kernel (per-token E4M3, scale = amax / 448, or block-16 NVFP4 with FlashInfer
// fp4_quantize semantics), counting finished rows per 256-row M tile in a zeroed flag array that the
// TMA producer acquires before it loads a tile; the MMA warps run the fused
// dequant -> BF16 -> out = BF16(residual + BF16(gate[gate_index] * o)) epilogue writing [M, 5376].
// Device code: TMA, ldmatrix, mma.sync (kind::f8f6f4 / kind::mxf4nvf4), mbarrier pipelines,
// setmaxnreg, gpu-scope release/acquire flags.
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <cstdint>

static_assert(sizeof(CUtensorMap) == 128, "CUDA tensor-map ABI size mismatch");
static_assert(alignof(CUtensorMap) >= 64, "CUDA tensor-map ABI requires at least 64-byte alignment");

namespace h3_out_proj_gemm_fp8_sm120a {

__device__ __forceinline__ int make_warp_uniform(int x) {
    int result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1F, 0xFFFFFFFF;"
                 : "=r"(result) : "r"(x));
    return result;
}

#define H3_QOP_INF CUDART_INF_F
#define NUM_AB_PIPE_STAGES 3
#define SMEM_A_STAGE_OFF 1024
#define SMEM_A_STAGE_STAGE_BYTES 16384
#define SMEM_A_STAGE_STRIDE 16384
#define SMEM_B_STAGE_OFF 50176
#define SMEM_B_STAGE_STAGE_BYTES 8192
#define SMEM_B_STAGE_STRIDE 8192
#define SMEM_STAGING_OFF 74752
#define SMEM_STAGING_STAGE_BYTES 8192
#define SMEM_STAGING_STRIDE 8192
#define SMEM_SFA_SLOT_OFF 82944
#define SMEM_SFA_SLOT_STAGE_BYTES 4096
#define SMEM_SFA_SLOT_STRIDE 4096
#define SMEM_SFB_STAGE_OFF 91136
#define SMEM_SFB_STAGE_STAGE_BYTES 1024
#define SMEM_SFB_STAGE_STRIDE 1024
#define SMEM_TOTAL 94208
#define THREADS 384
#define GROUP_M 16

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


__device__ __forceinline__ uint32_t mbarrier_try_wait_plain(int mbar_addr, int phase) {
    uint32_t token;
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "mbarrier.try_wait.parity.shared::cta.b64 P1, [%1], %2;\n\t"
        "selp.u32 %0, 1, 0, P1;\n\t"
        "}\n"
        : "=r"(token)
        : "r"(mbar_addr), "r"(phase) : "memory");
    return token;
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


__device__ __forceinline__ void mbarrier_wait(int mbar_addr, int phase) {
    uint32_t ticks = 0x989680;
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "LAB_WAIT:\n\t"
        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64"
        " P1, [%0], %1, %2;\n\t"
        "@P1 bra.uni DONE;\n\t"
        "bra.uni LAB_WAIT;\n\t"
        "DONE:\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"(phase), "r"(ticks) : "memory");
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


__device__ __forceinline__ void tma_2d_gmem2smem(
    int dst, const void *tmap_ptr, int x, int y, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global"
        ".mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3}], [%4];"
        :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y),
           "r"(mbar_addr) : "memory");
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


__global__ __launch_bounds__(384, 1) void
kernel_h3_out_proj_gemm_fused(const __grid_constant__ CUtensorMap A, const __grid_constant__ CUtensorMap B, const __grid_constant__ CUtensorMap SFA, const __grid_constant__ CUtensorMap SFB, unsigned int* __restrict__ attn_out_w, unsigned int* __restrict__ act_q_w, uint8_t* __restrict__ act_sf_w, float* __restrict__ act_scale, float* __restrict__ act_global_scale, float* __restrict__ w_scale, __nv_bfloat16* __restrict__ gate, int* __restrict__ gate_index, __nv_bfloat16* __restrict__ residual, unsigned int* __restrict__ out, unsigned int* __restrict__ flags, int M, int num_m_tiles, int total_tiles, float alpha)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int mbar_base = smem;
    #define ab_full_addr (mbar_base + 0)
    #define ab_empty_addr (mbar_base + 24)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // Kernel setup ops
    uint8_t* A_stage = reinterpret_cast<uint8_t*>(smem_raw + 1024);
    const int A_stage_addr = smem + 1024;
    uint8_t* B_stage = reinterpret_cast<uint8_t*>(smem_raw + 50176);
    const int B_stage_addr = smem + 50176;
    __nv_bfloat16* staging = reinterpret_cast<__nv_bfloat16*>(smem_raw + 74752);
    const int staging_addr = smem + 74752;
    unsigned int* SFA_slot = reinterpret_cast<unsigned int*>(smem_raw + 82944);
    const int SFA_slot_addr = smem + 82944;
    unsigned int* SFB_stage = reinterpret_cast<unsigned int*>(smem_raw + 91136);
    const int SFB_stage_addr = smem + 91136;

    // Mbarrier init (2 pipeline groups, 0 ordered-sequence groups, 6 barriers)
    // Mbarriers at smem_raw[0..48)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'ab_pipe' ---
            // ab_full: 3 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            mbarrier_init(smem + 8, 1);
            mbarrier_init(smem + 16, 1);
            // ab_empty: 3 barriers, init_count=8
            mbarrier_init(smem + 24, 8);
            mbarrier_init(smem + 32, 8);
            mbarrier_init(smem + 40, 8);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncthreads();

    // ---- Warpgroup: 0 ----
    if (warp >= 0 && warp <= 3) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 56;");
        // ---- Role: producer ----
        if (warp == 0) {
            { // producer_main
                unsigned int load_stage = 0;
                unsigned int _phase_ab_empty = 1;
                #pragma unroll 1
                for (int tile = bid; tile < total_tiles; tile += num_bids) {
                    int tile_m = tile / (GROUP_M * 42) * GROUP_M + (tile - tile / (GROUP_M * 42) * (GROUP_M * 42)) % ((num_m_tiles - tile / (GROUP_M * 42) * GROUP_M < GROUP_M) ? num_m_tiles - tile / (GROUP_M * 42) * GROUP_M : GROUP_M);
                    int tile_n = (tile - tile / (GROUP_M * 42) * (GROUP_M * 42)) / ((num_m_tiles - tile / (GROUP_M * 42) * GROUP_M < GROUP_M) ? num_m_tiles - tile / (GROUP_M * 42) * GROUP_M : GROUP_M);
                    int rows_left = M - tile_m * 256;
                    unsigned int rows_ready = ((rows_left < 256) ? rows_left : 256);
                    if (elect_sync()) {
                        {
                        unsigned int _acquire_observed;
                        do {
                        asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(_acquire_observed) : "l"((reinterpret_cast<unsigned int*>(flags) + (tile_m))) : "memory");
                        } while (static_cast<unsigned int>(_acquire_observed - static_cast<unsigned int>(rows_ready)) >= static_cast<unsigned int>(1));
                        }
                        asm volatile("fence.proxy.async;");
                        #pragma unroll 1
                        for (int kp = 0; kp < 56; kp++) {
                            mbarrier_wait(ab_empty_addr + (load_stage) * 8, _phase_ab_empty);
                            mbarrier_arrive_expect_tx(ab_full_addr + (load_stage) * 8, 24576);
                            asm volatile(
                                "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes"
                                " [%0], [%1, {%2, %3}], [%4];"
                                :: "r"(A_stage_addr + load_stage * 16384), "l"((&A)), "r"(kp * 2 * 64), "r"(tile_m * 256), "r"(ab_full_addr + (load_stage) * 8) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes"
                                " [%0], [%1, {%2, %3}], [%4];"
                                :: "r"(B_stage_addr + load_stage * 8192), "l"((&B)), "r"(kp * 2 * 64), "r"(tile_n * 128), "r"(ab_full_addr + (load_stage) * 8) : "memory");
                            load_stage += 1;
                            if (load_stage == 3) { load_stage = 0; _phase_ab_empty ^= 1; }
                            mbarrier_wait(ab_empty_addr + (load_stage) * 8, _phase_ab_empty);
                            mbarrier_arrive_expect_tx(ab_full_addr + (load_stage) * 8, 24576);
                            asm volatile(
                                "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes"
                                " [%0], [%1, {%2, %3}], [%4];"
                                :: "r"(A_stage_addr + load_stage * 16384), "l"((&A)), "r"((kp * 2 + 1) * 64), "r"(tile_m * 256), "r"(ab_full_addr + (load_stage) * 8) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes"
                                " [%0], [%1, {%2, %3}], [%4];"
                                :: "r"(B_stage_addr + load_stage * 8192), "l"((&B)), "r"((kp * 2 + 1) * 64), "r"(tile_n * 128), "r"(ab_full_addr + (load_stage) * 8) : "memory");
                            load_stage += 1;
                            if (load_stage == 3) { load_stage = 0; _phase_ab_empty ^= 1; }
                        }
                    }
                }
            }
        // ---- Role: quant ----
        } else if (warp >= 1 && warp <= 3) {
            { // quant_main
                unsigned int qw[16];
                float qx[16];
                float qacc[4];
                unsigned int qwords[4];
                float global_scale = act_global_scale[0];
                int warp_id_in_role = (warp - 1);
                int qrow0 = bid * 3 + warp_id_in_role;
                #pragma unroll 1
                for (int row = qrow0; row < M; row += num_bids * 3) {
                    qacc[0] = 0.0f;
                    #pragma unroll 2
                    for (int j = 0; j < 7; j++) {
                        {
                            const uint4* _ivptr_0 = reinterpret_cast<const uint4*>(attn_out_w + (row * 3584 + (lane + 32 * (j * 2)) * 8) + 0);
                            uint4 _ivld_0;
                            _ivld_0 = *_ivptr_0;
                            qw[0 + 0] = _ivld_0.x;
                            qw[0 + 1] = _ivld_0.y;
                            qw[0 + 2] = _ivld_0.z;
                            qw[0 + 3] = _ivld_0.w;
                        }
                        {
                            const uint4* _ivptr_1 = reinterpret_cast<const uint4*>(attn_out_w + (row * 3584 + (lane + 32 * (j * 2)) * 8 + 4) + 0);
                            uint4 _ivld_1;
                            _ivld_1 = *_ivptr_1;
                            qw[4 + 0] = _ivld_1.x;
                            qw[4 + 1] = _ivld_1.y;
                            qw[4 + 2] = _ivld_1.z;
                            qw[4 + 3] = _ivld_1.w;
                        }
                        {
                            const uint4* _ivptr_2 = reinterpret_cast<const uint4*>(attn_out_w + (row * 3584 + (lane + 32 * (j * 2 + 1)) * 8) + 0);
                            uint4 _ivld_2;
                            _ivld_2 = *_ivptr_2;
                            qw[8 + 0] = _ivld_2.x;
                            qw[8 + 1] = _ivld_2.y;
                            qw[8 + 2] = _ivld_2.z;
                            qw[8 + 3] = _ivld_2.w;
                        }
                        {
                            const uint4* _ivptr_3 = reinterpret_cast<const uint4*>(attn_out_w + (row * 3584 + (lane + 32 * (j * 2 + 1)) * 8 + 4) + 0);
                            uint4 _ivld_3;
                            _ivld_3 = *_ivptr_3;
                            qw[12 + 0] = _ivld_3.x;
                            qw[12 + 1] = _ivld_3.y;
                            qw[12 + 2] = _ivld_3.z;
                            qw[12 + 3] = _ivld_3.w;
                        }
                        float qw_f32[16];
                        #pragma unroll
                        for (int _pair = 0; _pair < 8; _pair++) {
                            asm volatile(
                                "{\n\t"
                                "shl.b32 %0, %2, 16;\n\t"
                                "and.b32 %1, %2, 0xffff0000;\n\t"
                                "}\n"
                                : "=f"((&qw_f32[_pair * 2])[0]), "=f"((&qw_f32[_pair * 2])[1])
                                : "r"(qw[_pair]));
                        }
                        float _fabs_0 = fabsf(qw_f32[0]);
                        float _fmax_0 = fmaxf(qacc[0], _fabs_0);
                        qacc[0] = _fmax_0;
                        float _fabs_1 = fabsf(qw_f32[1]);
                        float _fmax_1 = fmaxf(qacc[0], _fabs_1);
                        qacc[0] = _fmax_1;
                        float _fabs_2 = fabsf(qw_f32[2]);
                        float _fmax_2 = fmaxf(qacc[0], _fabs_2);
                        qacc[0] = _fmax_2;
                        float _fabs_3 = fabsf(qw_f32[3]);
                        float _fmax_3 = fmaxf(qacc[0], _fabs_3);
                        qacc[0] = _fmax_3;
                        float _fabs_4 = fabsf(qw_f32[4]);
                        float _fmax_4 = fmaxf(qacc[0], _fabs_4);
                        qacc[0] = _fmax_4;
                        float _fabs_5 = fabsf(qw_f32[5]);
                        float _fmax_5 = fmaxf(qacc[0], _fabs_5);
                        qacc[0] = _fmax_5;
                        float _fabs_6 = fabsf(qw_f32[6]);
                        float _fmax_6 = fmaxf(qacc[0], _fabs_6);
                        qacc[0] = _fmax_6;
                        float _fabs_7 = fabsf(qw_f32[7]);
                        float _fmax_7 = fmaxf(qacc[0], _fabs_7);
                        qacc[0] = _fmax_7;
                        float _fabs_8 = fabsf(qw_f32[8]);
                        float _fmax_8 = fmaxf(qacc[0], _fabs_8);
                        qacc[0] = _fmax_8;
                        float _fabs_9 = fabsf(qw_f32[9]);
                        float _fmax_9 = fmaxf(qacc[0], _fabs_9);
                        qacc[0] = _fmax_9;
                        float _fabs_10 = fabsf(qw_f32[10]);
                        float _fmax_10 = fmaxf(qacc[0], _fabs_10);
                        qacc[0] = _fmax_10;
                        float _fabs_11 = fabsf(qw_f32[11]);
                        float _fmax_11 = fmaxf(qacc[0], _fabs_11);
                        qacc[0] = _fmax_11;
                        float _fabs_12 = fabsf(qw_f32[12]);
                        float _fmax_12 = fmaxf(qacc[0], _fabs_12);
                        qacc[0] = _fmax_12;
                        float _fabs_13 = fabsf(qw_f32[13]);
                        float _fmax_13 = fmaxf(qacc[0], _fabs_13);
                        qacc[0] = _fmax_13;
                        float _fabs_14 = fabsf(qw_f32[14]);
                        float _fmax_14 = fmaxf(qacc[0], _fabs_14);
                        qacc[0] = _fmax_14;
                        float _fabs_15 = fabsf(qw_f32[15]);
                        float _fmax_15 = fmaxf(qacc[0], _fabs_15);
                        qacc[0] = _fmax_15;
                        float qw_f32_0[16];
                        #pragma unroll
                        for (int _pair = 0; _pair < 8; _pair++) {
                            asm volatile(
                                "{\n\t"
                                "shl.b32 %0, %2, 16;\n\t"
                                "and.b32 %1, %2, 0xffff0000;\n\t"
                                "}\n"
                                : "=f"((&qw_f32_0[_pair * 2])[0]), "=f"((&qw_f32_0[_pair * 2])[1])
                                : "r"(qw[8 + _pair]));
                        }
                        float _fabs_16 = fabsf(qw_f32_0[0]);
                        float _fmax_16 = fmaxf(qacc[0], _fabs_16);
                        qacc[0] = _fmax_16;
                        float _fabs_17 = fabsf(qw_f32_0[1]);
                        float _fmax_17 = fmaxf(qacc[0], _fabs_17);
                        qacc[0] = _fmax_17;
                        float _fabs_18 = fabsf(qw_f32_0[2]);
                        float _fmax_18 = fmaxf(qacc[0], _fabs_18);
                        qacc[0] = _fmax_18;
                        float _fabs_19 = fabsf(qw_f32_0[3]);
                        float _fmax_19 = fmaxf(qacc[0], _fabs_19);
                        qacc[0] = _fmax_19;
                        float _fabs_20 = fabsf(qw_f32_0[4]);
                        float _fmax_20 = fmaxf(qacc[0], _fabs_20);
                        qacc[0] = _fmax_20;
                        float _fabs_21 = fabsf(qw_f32_0[5]);
                        float _fmax_21 = fmaxf(qacc[0], _fabs_21);
                        qacc[0] = _fmax_21;
                        float _fabs_22 = fabsf(qw_f32_0[6]);
                        float _fmax_22 = fmaxf(qacc[0], _fabs_22);
                        qacc[0] = _fmax_22;
                        float _fabs_23 = fabsf(qw_f32_0[7]);
                        float _fmax_23 = fmaxf(qacc[0], _fabs_23);
                        qacc[0] = _fmax_23;
                        float _fabs_24 = fabsf(qw_f32_0[8]);
                        float _fmax_24 = fmaxf(qacc[0], _fabs_24);
                        qacc[0] = _fmax_24;
                        float _fabs_25 = fabsf(qw_f32_0[9]);
                        float _fmax_25 = fmaxf(qacc[0], _fabs_25);
                        qacc[0] = _fmax_25;
                        float _fabs_26 = fabsf(qw_f32_0[10]);
                        float _fmax_26 = fmaxf(qacc[0], _fabs_26);
                        qacc[0] = _fmax_26;
                        float _fabs_27 = fabsf(qw_f32_0[11]);
                        float _fmax_27 = fmaxf(qacc[0], _fabs_27);
                        qacc[0] = _fmax_27;
                        float _fabs_28 = fabsf(qw_f32_0[12]);
                        float _fmax_28 = fmaxf(qacc[0], _fabs_28);
                        qacc[0] = _fmax_28;
                        float _fabs_29 = fabsf(qw_f32_0[13]);
                        float _fmax_29 = fmaxf(qacc[0], _fabs_29);
                        qacc[0] = _fmax_29;
                        float _fabs_30 = fabsf(qw_f32_0[14]);
                        float _fmax_30 = fmaxf(qacc[0], _fabs_30);
                        qacc[0] = _fmax_30;
                        float _fabs_31 = fabsf(qw_f32_0[15]);
                        float _fmax_31 = fmaxf(qacc[0], _fabs_31);
                        qacc[0] = _fmax_31;
                    }
                    float _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, qacc[0], 16);
                    float _fmax_32 = fmaxf(qacc[0], _shfl_xor_0);
                    qacc[0] = _fmax_32;
                    float _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, qacc[0], 8);
                    float _fmax_33 = fmaxf(qacc[0], _shfl_xor_1);
                    qacc[0] = _fmax_33;
                    float _shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, qacc[0], 4);
                    float _fmax_34 = fmaxf(qacc[0], _shfl_xor_2);
                    qacc[0] = _fmax_34;
                    float _shfl_xor_3 = __shfl_xor_sync(0xFFFFFFFF, qacc[0], 2);
                    float _fmax_35 = fmaxf(qacc[0], _shfl_xor_3);
                    qacc[0] = _fmax_35;
                    float _shfl_xor_4 = __shfl_xor_sync(0xFFFFFFFF, qacc[0], 1);
                    float _fmax_36 = fmaxf(qacc[0], _shfl_xor_4);
                    qacc[0] = _fmax_36;
                    float _fmax_37 = fmaxf(qacc[0], 1e-12f);
                    float _fdiv_rn_0 = __fdiv_rn(_fmax_37, 448.0f);
                    qacc[1] = _fdiv_rn_0;
                    if (elect_sync()) {
                        *(reinterpret_cast<float*>(act_scale + row) + (0)) = qacc[1];
                    }
                    #pragma unroll 2
                    for (int j_1 = 0; j_1 < 7; j_1++) {
                        {
                            const uint4* _ivptr_4 = reinterpret_cast<const uint4*>(attn_out_w + (row * 3584 + (lane + 32 * (j_1 * 2)) * 8) + 0);
                            uint4 _ivld_4;
                            _ivld_4 = *_ivptr_4;
                            qw[0 + 0] = _ivld_4.x;
                            qw[0 + 1] = _ivld_4.y;
                            qw[0 + 2] = _ivld_4.z;
                            qw[0 + 3] = _ivld_4.w;
                        }
                        {
                            const uint4* _ivptr_5 = reinterpret_cast<const uint4*>(attn_out_w + (row * 3584 + (lane + 32 * (j_1 * 2)) * 8 + 4) + 0);
                            uint4 _ivld_5;
                            _ivld_5 = *_ivptr_5;
                            qw[4 + 0] = _ivld_5.x;
                            qw[4 + 1] = _ivld_5.y;
                            qw[4 + 2] = _ivld_5.z;
                            qw[4 + 3] = _ivld_5.w;
                        }
                        {
                            const uint4* _ivptr_6 = reinterpret_cast<const uint4*>(attn_out_w + (row * 3584 + (lane + 32 * (j_1 * 2 + 1)) * 8) + 0);
                            uint4 _ivld_6;
                            _ivld_6 = *_ivptr_6;
                            qw[8 + 0] = _ivld_6.x;
                            qw[8 + 1] = _ivld_6.y;
                            qw[8 + 2] = _ivld_6.z;
                            qw[8 + 3] = _ivld_6.w;
                        }
                        {
                            const uint4* _ivptr_7 = reinterpret_cast<const uint4*>(attn_out_w + (row * 3584 + (lane + 32 * (j_1 * 2 + 1)) * 8 + 4) + 0);
                            uint4 _ivld_7;
                            _ivld_7 = *_ivptr_7;
                            qw[12 + 0] = _ivld_7.x;
                            qw[12 + 1] = _ivld_7.y;
                            qw[12 + 2] = _ivld_7.z;
                            qw[12 + 3] = _ivld_7.w;
                        }
                        float qw_f32_1[16];
                        #pragma unroll
                        for (int _pair = 0; _pair < 8; _pair++) {
                            asm volatile(
                                "{\n\t"
                                "shl.b32 %0, %2, 16;\n\t"
                                "and.b32 %1, %2, 0xffff0000;\n\t"
                                "}\n"
                                : "=f"((&qw_f32_1[_pair * 2])[0]), "=f"((&qw_f32_1[_pair * 2])[1])
                                : "r"(qw[_pair]));
                        }
                        float _fdiv_rn_1 = __fdiv_rn(qw_f32_1[0], qacc[1]);
                        float _fdiv_rn_2 = __fdiv_rn(qw_f32_1[1], qacc[1]);
                        float _fdiv_rn_3 = __fdiv_rn(qw_f32_1[2], qacc[1]);
                        float _fdiv_rn_4 = __fdiv_rn(qw_f32_1[3], qacc[1]);
                        uint16_t _e4m3x2_f32_0;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_0) : "f"(_fdiv_rn_2), "f"(_fdiv_rn_1));
                        uint16_t _e4m3x2_f32_1;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_1) : "f"(_fdiv_rn_4), "f"(_fdiv_rn_3));
                        uint32_t _pack_u16x2_0;
                        asm("mov.b32 %0, {%1, %2};" : "=r"(_pack_u16x2_0) : "h"(_e4m3x2_f32_0), "h"(_e4m3x2_f32_1));
                        qwords[0] = _pack_u16x2_0;
                        float _fdiv_rn_5 = __fdiv_rn(qw_f32_1[4], qacc[1]);
                        float _fdiv_rn_6 = __fdiv_rn(qw_f32_1[5], qacc[1]);
                        float _fdiv_rn_7 = __fdiv_rn(qw_f32_1[6], qacc[1]);
                        float _fdiv_rn_8 = __fdiv_rn(qw_f32_1[7], qacc[1]);
                        uint16_t _e4m3x2_f32_2;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_2) : "f"(_fdiv_rn_6), "f"(_fdiv_rn_5));
                        uint16_t _e4m3x2_f32_3;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_3) : "f"(_fdiv_rn_8), "f"(_fdiv_rn_7));
                        uint32_t _pack_u16x2_1;
                        asm("mov.b32 %0, {%1, %2};" : "=r"(_pack_u16x2_1) : "h"(_e4m3x2_f32_2), "h"(_e4m3x2_f32_3));
                        qwords[1] = _pack_u16x2_1;
                        float _fdiv_rn_9 = __fdiv_rn(qw_f32_1[8], qacc[1]);
                        float _fdiv_rn_10 = __fdiv_rn(qw_f32_1[9], qacc[1]);
                        float _fdiv_rn_11 = __fdiv_rn(qw_f32_1[10], qacc[1]);
                        float _fdiv_rn_12 = __fdiv_rn(qw_f32_1[11], qacc[1]);
                        uint16_t _e4m3x2_f32_4;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_4) : "f"(_fdiv_rn_10), "f"(_fdiv_rn_9));
                        uint16_t _e4m3x2_f32_5;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_5) : "f"(_fdiv_rn_12), "f"(_fdiv_rn_11));
                        uint32_t _pack_u16x2_2;
                        asm("mov.b32 %0, {%1, %2};" : "=r"(_pack_u16x2_2) : "h"(_e4m3x2_f32_4), "h"(_e4m3x2_f32_5));
                        qwords[2] = _pack_u16x2_2;
                        float _fdiv_rn_13 = __fdiv_rn(qw_f32_1[12], qacc[1]);
                        float _fdiv_rn_14 = __fdiv_rn(qw_f32_1[13], qacc[1]);
                        float _fdiv_rn_15 = __fdiv_rn(qw_f32_1[14], qacc[1]);
                        float _fdiv_rn_16 = __fdiv_rn(qw_f32_1[15], qacc[1]);
                        uint16_t _e4m3x2_f32_6;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_6) : "f"(_fdiv_rn_14), "f"(_fdiv_rn_13));
                        uint16_t _e4m3x2_f32_7;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_7) : "f"(_fdiv_rn_16), "f"(_fdiv_rn_15));
                        uint32_t _pack_u16x2_3;
                        asm("mov.b32 %0, {%1, %2};" : "=r"(_pack_u16x2_3) : "h"(_e4m3x2_f32_6), "h"(_e4m3x2_f32_7));
                        qwords[3] = _pack_u16x2_3;
                        reinterpret_cast<int4*>(act_q_w + ((row * 7168 + (lane + 32 * (j_1 * 2)) * 16) / 4))[0] = reinterpret_cast<int4*>(qwords)[0];
                        float qw_f32_0_1[16];
                        #pragma unroll
                        for (int _pair = 0; _pair < 8; _pair++) {
                            asm volatile(
                                "{\n\t"
                                "shl.b32 %0, %2, 16;\n\t"
                                "and.b32 %1, %2, 0xffff0000;\n\t"
                                "}\n"
                                : "=f"((&qw_f32_0_1[_pair * 2])[0]), "=f"((&qw_f32_0_1[_pair * 2])[1])
                                : "r"(qw[8 + _pair]));
                        }
                        float _fdiv_rn_17 = __fdiv_rn(qw_f32_0_1[0], qacc[1]);
                        float _fdiv_rn_18 = __fdiv_rn(qw_f32_0_1[1], qacc[1]);
                        float _fdiv_rn_19 = __fdiv_rn(qw_f32_0_1[2], qacc[1]);
                        float _fdiv_rn_20 = __fdiv_rn(qw_f32_0_1[3], qacc[1]);
                        uint16_t _e4m3x2_f32_8;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_8) : "f"(_fdiv_rn_18), "f"(_fdiv_rn_17));
                        uint16_t _e4m3x2_f32_9;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_9) : "f"(_fdiv_rn_20), "f"(_fdiv_rn_19));
                        uint32_t _pack_u16x2_4;
                        asm("mov.b32 %0, {%1, %2};" : "=r"(_pack_u16x2_4) : "h"(_e4m3x2_f32_8), "h"(_e4m3x2_f32_9));
                        qwords[0] = _pack_u16x2_4;
                        float _fdiv_rn_21 = __fdiv_rn(qw_f32_0_1[4], qacc[1]);
                        float _fdiv_rn_22 = __fdiv_rn(qw_f32_0_1[5], qacc[1]);
                        float _fdiv_rn_23 = __fdiv_rn(qw_f32_0_1[6], qacc[1]);
                        float _fdiv_rn_24 = __fdiv_rn(qw_f32_0_1[7], qacc[1]);
                        uint16_t _e4m3x2_f32_10;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_10) : "f"(_fdiv_rn_22), "f"(_fdiv_rn_21));
                        uint16_t _e4m3x2_f32_11;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_11) : "f"(_fdiv_rn_24), "f"(_fdiv_rn_23));
                        uint32_t _pack_u16x2_5;
                        asm("mov.b32 %0, {%1, %2};" : "=r"(_pack_u16x2_5) : "h"(_e4m3x2_f32_10), "h"(_e4m3x2_f32_11));
                        qwords[1] = _pack_u16x2_5;
                        float _fdiv_rn_25 = __fdiv_rn(qw_f32_0_1[8], qacc[1]);
                        float _fdiv_rn_26 = __fdiv_rn(qw_f32_0_1[9], qacc[1]);
                        float _fdiv_rn_27 = __fdiv_rn(qw_f32_0_1[10], qacc[1]);
                        float _fdiv_rn_28 = __fdiv_rn(qw_f32_0_1[11], qacc[1]);
                        uint16_t _e4m3x2_f32_12;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_12) : "f"(_fdiv_rn_26), "f"(_fdiv_rn_25));
                        uint16_t _e4m3x2_f32_13;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_13) : "f"(_fdiv_rn_28), "f"(_fdiv_rn_27));
                        uint32_t _pack_u16x2_6;
                        asm("mov.b32 %0, {%1, %2};" : "=r"(_pack_u16x2_6) : "h"(_e4m3x2_f32_12), "h"(_e4m3x2_f32_13));
                        qwords[2] = _pack_u16x2_6;
                        float _fdiv_rn_29 = __fdiv_rn(qw_f32_0_1[12], qacc[1]);
                        float _fdiv_rn_30 = __fdiv_rn(qw_f32_0_1[13], qacc[1]);
                        float _fdiv_rn_31 = __fdiv_rn(qw_f32_0_1[14], qacc[1]);
                        float _fdiv_rn_32 = __fdiv_rn(qw_f32_0_1[15], qacc[1]);
                        uint16_t _e4m3x2_f32_14;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_14) : "f"(_fdiv_rn_30), "f"(_fdiv_rn_29));
                        uint16_t _e4m3x2_f32_15;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_15) : "f"(_fdiv_rn_32), "f"(_fdiv_rn_31));
                        uint32_t _pack_u16x2_7;
                        asm("mov.b32 %0, {%1, %2};" : "=r"(_pack_u16x2_7) : "h"(_e4m3x2_f32_14), "h"(_e4m3x2_f32_15));
                        qwords[3] = _pack_u16x2_7;
                        reinterpret_cast<int4*>(act_q_w + ((row * 7168 + (lane + 32 * (j_1 * 2 + 1)) * 16) / 4))[0] = reinterpret_cast<int4*>(qwords)[0];
                    }
                    __threadfence();
                    __syncwarp();
                    if (elect_sync()) {
                        asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(flags) + (row / 256))), "r"(static_cast<unsigned int>(1)) : "memory");
                    }
                }
            }
        }
    }
    // ---- Role: mma ----
    if (warp >= 4 && warp <= 11) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 224;");
        { // mma_main
            unsigned int mma_stage = 0;
            int warp_id_in_role_1 = (warp - 4);
            int warp_m = warp_id_in_role_1 % 4;
            int warp_n = warp_id_in_role_1 / 4;
            int role_tid = warp_id_in_role_1 * 32 + lane;
            float accum[128];
            unsigned int a_frag[16];
            unsigned int b_frag[16];
            unsigned int _phase_ab_full = 0;
            #pragma unroll 1
            for (int tile_1 = bid; tile_1 < total_tiles; tile_1 += num_bids) {
                int tile_m_1 = tile_1 / (GROUP_M * 42) * GROUP_M + (tile_1 - tile_1 / (GROUP_M * 42) * (GROUP_M * 42)) % ((num_m_tiles - tile_1 / (GROUP_M * 42) * GROUP_M < GROUP_M) ? num_m_tiles - tile_1 / (GROUP_M * 42) * GROUP_M : GROUP_M);
                int tile_n_1 = (tile_1 - tile_1 / (GROUP_M * 42) * (GROUP_M * 42)) / ((num_m_tiles - tile_1 / (GROUP_M * 42) * GROUP_M < GROUP_M) ? num_m_tiles - tile_1 / (GROUP_M * 42) * GROUP_M : GROUP_M);
                accum[0] = 0.0f;
                accum[1] = 0.0f;
                accum[2] = 0.0f;
                accum[3] = 0.0f;
                accum[4] = 0.0f;
                accum[5] = 0.0f;
                accum[6] = 0.0f;
                accum[7] = 0.0f;
                accum[8] = 0.0f;
                accum[9] = 0.0f;
                accum[10] = 0.0f;
                accum[11] = 0.0f;
                accum[12] = 0.0f;
                accum[13] = 0.0f;
                accum[14] = 0.0f;
                accum[15] = 0.0f;
                accum[16] = 0.0f;
                accum[17] = 0.0f;
                accum[18] = 0.0f;
                accum[19] = 0.0f;
                accum[20] = 0.0f;
                accum[21] = 0.0f;
                accum[22] = 0.0f;
                accum[23] = 0.0f;
                accum[24] = 0.0f;
                accum[25] = 0.0f;
                accum[26] = 0.0f;
                accum[27] = 0.0f;
                accum[28] = 0.0f;
                accum[29] = 0.0f;
                accum[30] = 0.0f;
                accum[31] = 0.0f;
                accum[32] = 0.0f;
                accum[33] = 0.0f;
                accum[34] = 0.0f;
                accum[35] = 0.0f;
                accum[36] = 0.0f;
                accum[37] = 0.0f;
                accum[38] = 0.0f;
                accum[39] = 0.0f;
                accum[40] = 0.0f;
                accum[41] = 0.0f;
                accum[42] = 0.0f;
                accum[43] = 0.0f;
                accum[44] = 0.0f;
                accum[45] = 0.0f;
                accum[46] = 0.0f;
                accum[47] = 0.0f;
                accum[48] = 0.0f;
                accum[49] = 0.0f;
                accum[50] = 0.0f;
                accum[51] = 0.0f;
                accum[52] = 0.0f;
                accum[53] = 0.0f;
                accum[54] = 0.0f;
                accum[55] = 0.0f;
                accum[56] = 0.0f;
                accum[57] = 0.0f;
                accum[58] = 0.0f;
                accum[59] = 0.0f;
                accum[60] = 0.0f;
                accum[61] = 0.0f;
                accum[62] = 0.0f;
                accum[63] = 0.0f;
                accum[64] = 0.0f;
                accum[65] = 0.0f;
                accum[66] = 0.0f;
                accum[67] = 0.0f;
                accum[68] = 0.0f;
                accum[69] = 0.0f;
                accum[70] = 0.0f;
                accum[71] = 0.0f;
                accum[72] = 0.0f;
                accum[73] = 0.0f;
                accum[74] = 0.0f;
                accum[75] = 0.0f;
                accum[76] = 0.0f;
                accum[77] = 0.0f;
                accum[78] = 0.0f;
                accum[79] = 0.0f;
                accum[80] = 0.0f;
                accum[81] = 0.0f;
                accum[82] = 0.0f;
                accum[83] = 0.0f;
                accum[84] = 0.0f;
                accum[85] = 0.0f;
                accum[86] = 0.0f;
                accum[87] = 0.0f;
                accum[88] = 0.0f;
                accum[89] = 0.0f;
                accum[90] = 0.0f;
                accum[91] = 0.0f;
                accum[92] = 0.0f;
                accum[93] = 0.0f;
                accum[94] = 0.0f;
                accum[95] = 0.0f;
                accum[96] = 0.0f;
                accum[97] = 0.0f;
                accum[98] = 0.0f;
                accum[99] = 0.0f;
                accum[100] = 0.0f;
                accum[101] = 0.0f;
                accum[102] = 0.0f;
                accum[103] = 0.0f;
                accum[104] = 0.0f;
                accum[105] = 0.0f;
                accum[106] = 0.0f;
                accum[107] = 0.0f;
                accum[108] = 0.0f;
                accum[109] = 0.0f;
                accum[110] = 0.0f;
                accum[111] = 0.0f;
                accum[112] = 0.0f;
                accum[113] = 0.0f;
                accum[114] = 0.0f;
                accum[115] = 0.0f;
                accum[116] = 0.0f;
                accum[117] = 0.0f;
                accum[118] = 0.0f;
                accum[119] = 0.0f;
                accum[120] = 0.0f;
                accum[121] = 0.0f;
                accum[122] = 0.0f;
                accum[123] = 0.0f;
                accum[124] = 0.0f;
                accum[125] = 0.0f;
                accum[126] = 0.0f;
                accum[127] = 0.0f;
                int col_base = tile_n_1 * 128 + warp_n * 64 + (lane & 3) * 2;
                float sw[16];
                sw[0] = w_scale[col_base];
                sw[1] = w_scale[col_base + 1];
                sw[2] = w_scale[col_base + 8];
                sw[3] = w_scale[col_base + 8 + 1];
                sw[4] = w_scale[col_base + 16];
                sw[5] = w_scale[col_base + 16 + 1];
                sw[6] = w_scale[col_base + 24];
                sw[7] = w_scale[col_base + 24 + 1];
                sw[8] = w_scale[col_base + 32];
                sw[9] = w_scale[col_base + 32 + 1];
                sw[10] = w_scale[col_base + 40];
                sw[11] = w_scale[col_base + 40 + 1];
                sw[12] = w_scale[col_base + 48];
                sw[13] = w_scale[col_base + 48 + 1];
                sw[14] = w_scale[col_base + 56];
                sw[15] = w_scale[col_base + 56 + 1];
                float sa[8];
                #pragma unroll 1
                for (int kp_1 = 0; kp_1 < 56; kp_1++) {
                    mbarrier_wait(ab_full_addr + (mma_stage) * 8, _phase_ab_full);
                    if (kp_1 == 52) {
                        int arow = tile_m_1 * 256 + warp_m * 64 + (lane >> 2);
                        sa[0] = ((arow < M) ? act_scale[arow] : 0.0f);
                        int arow_0 = tile_m_1 * 256 + warp_m * 64 + 8 + (lane >> 2);
                        sa[1] = ((arow_0 < M) ? act_scale[arow_0] : 0.0f);
                        int arow_1 = tile_m_1 * 256 + warp_m * 64 + 16 + (lane >> 2);
                        sa[2] = ((arow_1 < M) ? act_scale[arow_1] : 0.0f);
                        int arow_2 = tile_m_1 * 256 + warp_m * 64 + 16 + 8 + (lane >> 2);
                        sa[3] = ((arow_2 < M) ? act_scale[arow_2] : 0.0f);
                        int arow_3 = tile_m_1 * 256 + warp_m * 64 + 32 + (lane >> 2);
                        sa[4] = ((arow_3 < M) ? act_scale[arow_3] : 0.0f);
                        int arow_4 = tile_m_1 * 256 + warp_m * 64 + 32 + 8 + (lane >> 2);
                        sa[5] = ((arow_4 < M) ? act_scale[arow_4] : 0.0f);
                        int arow_5 = tile_m_1 * 256 + warp_m * 64 + 48 + (lane >> 2);
                        sa[6] = ((arow_5 < M) ? act_scale[arow_5] : 0.0f);
                        int arow_6 = tile_m_1 * 256 + warp_m * 64 + 48 + 8 + (lane >> 2);
                        sa[7] = ((arow_6 < M) ? act_scale[arow_6] : 0.0f);
                    }
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                        : "r"(A_stage_addr + mma_stage * 16384 + (unsigned int)((warp_m * 64 + (lane >> 3 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)((lane >> 3 >> 1) * 16 ^ (warp_m * 64 + (lane >> 3 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_frag[4]), "=r"(a_frag[5]), "=r"(a_frag[6]), "=r"(a_frag[7])
                        : "r"(A_stage_addr + mma_stage * 16384 + (unsigned int)((warp_m * 64 + 16 + (lane >> 3 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)((lane >> 3 >> 1) * 16 ^ (warp_m * 64 + 16 + (lane >> 3 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_frag[8]), "=r"(a_frag[9]), "=r"(a_frag[10]), "=r"(a_frag[11])
                        : "r"(A_stage_addr + mma_stage * 16384 + (unsigned int)((warp_m * 64 + 32 + (lane >> 3 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)((lane >> 3 >> 1) * 16 ^ (warp_m * 64 + 32 + (lane >> 3 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_frag[12]), "=r"(a_frag[13]), "=r"(a_frag[14]), "=r"(a_frag[15])
                        : "r"(A_stage_addr + mma_stage * 16384 + (unsigned int)((warp_m * 64 + 48 + (lane >> 3 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)((lane >> 3 >> 1) * 16 ^ (warp_m * 64 + 48 + (lane >> 3 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                        : "r"(B_stage_addr + mma_stage * 8192 + (unsigned int)((warp_n * 64 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)((lane >> 3 & 1) * 16 ^ (warp_n * 64 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_frag[4]), "=r"(b_frag[5]), "=r"(b_frag[6]), "=r"(b_frag[7])
                        : "r"(B_stage_addr + mma_stage * 8192 + (unsigned int)((warp_n * 64 + 16 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)((lane >> 3 & 1) * 16 ^ (warp_n * 64 + 16 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_frag[8]), "=r"(b_frag[9]), "=r"(b_frag[10]), "=r"(b_frag[11])
                        : "r"(B_stage_addr + mma_stage * 8192 + (unsigned int)((warp_n * 64 + 32 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)((lane >> 3 & 1) * 16 ^ (warp_n * 64 + 32 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_frag[12]), "=r"(b_frag[13]), "=r"(b_frag[14]), "=r"(b_frag[15])
                        : "r"(B_stage_addr + mma_stage * 8192 + (unsigned int)((warp_n * 64 + 48 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)((lane >> 3 & 1) * 16 ^ (warp_n * 64 + 48 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[0]), "+f"(accum[1]), "+f"(accum[2]), "+f"(accum[3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[4]), "+f"(accum[(4) + 1]), "+f"(accum[(4) + 2]), "+f"(accum[(4) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[8]), "+f"(accum[(8) + 1]), "+f"(accum[(8) + 2]), "+f"(accum[(8) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[4]), "r"(b_frag[(4) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[12]), "+f"(accum[(12) + 1]), "+f"(accum[(12) + 2]), "+f"(accum[(12) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[6]), "r"(b_frag[(6) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[16]), "+f"(accum[(16) + 1]), "+f"(accum[(16) + 2]), "+f"(accum[(16) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[8]), "r"(b_frag[(8) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[20]), "+f"(accum[(20) + 1]), "+f"(accum[(20) + 2]), "+f"(accum[(20) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[10]), "r"(b_frag[(10) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[24]), "+f"(accum[(24) + 1]), "+f"(accum[(24) + 2]), "+f"(accum[(24) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[12]), "r"(b_frag[(12) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[28]), "+f"(accum[(28) + 1]), "+f"(accum[(28) + 2]), "+f"(accum[(28) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[14]), "r"(b_frag[(14) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[32]), "+f"(accum[(32) + 1]), "+f"(accum[(32) + 2]), "+f"(accum[(32) + 3])
                        : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[0]), "r"(b_frag[1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[36]), "+f"(accum[(36) + 1]), "+f"(accum[(36) + 2]), "+f"(accum[(36) + 3])
                        : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[40]), "+f"(accum[(40) + 1]), "+f"(accum[(40) + 2]), "+f"(accum[(40) + 3])
                        : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[4]), "r"(b_frag[(4) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[44]), "+f"(accum[(44) + 1]), "+f"(accum[(44) + 2]), "+f"(accum[(44) + 3])
                        : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[6]), "r"(b_frag[(6) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[48]), "+f"(accum[(48) + 1]), "+f"(accum[(48) + 2]), "+f"(accum[(48) + 3])
                        : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[8]), "r"(b_frag[(8) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[52]), "+f"(accum[(52) + 1]), "+f"(accum[(52) + 2]), "+f"(accum[(52) + 3])
                        : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[10]), "r"(b_frag[(10) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[56]), "+f"(accum[(56) + 1]), "+f"(accum[(56) + 2]), "+f"(accum[(56) + 3])
                        : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[12]), "r"(b_frag[(12) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[60]), "+f"(accum[(60) + 1]), "+f"(accum[(60) + 2]), "+f"(accum[(60) + 3])
                        : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[14]), "r"(b_frag[(14) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[64]), "+f"(accum[(64) + 1]), "+f"(accum[(64) + 2]), "+f"(accum[(64) + 3])
                        : "r"(a_frag[8]), "r"(a_frag[(8) + 1]), "r"(a_frag[(8) + 2]), "r"(a_frag[(8) + 3]), "r"(b_frag[0]), "r"(b_frag[1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[68]), "+f"(accum[(68) + 1]), "+f"(accum[(68) + 2]), "+f"(accum[(68) + 3])
                        : "r"(a_frag[8]), "r"(a_frag[(8) + 1]), "r"(a_frag[(8) + 2]), "r"(a_frag[(8) + 3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[72]), "+f"(accum[(72) + 1]), "+f"(accum[(72) + 2]), "+f"(accum[(72) + 3])
                        : "r"(a_frag[8]), "r"(a_frag[(8) + 1]), "r"(a_frag[(8) + 2]), "r"(a_frag[(8) + 3]), "r"(b_frag[4]), "r"(b_frag[(4) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[76]), "+f"(accum[(76) + 1]), "+f"(accum[(76) + 2]), "+f"(accum[(76) + 3])
                        : "r"(a_frag[8]), "r"(a_frag[(8) + 1]), "r"(a_frag[(8) + 2]), "r"(a_frag[(8) + 3]), "r"(b_frag[6]), "r"(b_frag[(6) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[80]), "+f"(accum[(80) + 1]), "+f"(accum[(80) + 2]), "+f"(accum[(80) + 3])
                        : "r"(a_frag[8]), "r"(a_frag[(8) + 1]), "r"(a_frag[(8) + 2]), "r"(a_frag[(8) + 3]), "r"(b_frag[8]), "r"(b_frag[(8) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[84]), "+f"(accum[(84) + 1]), "+f"(accum[(84) + 2]), "+f"(accum[(84) + 3])
                        : "r"(a_frag[8]), "r"(a_frag[(8) + 1]), "r"(a_frag[(8) + 2]), "r"(a_frag[(8) + 3]), "r"(b_frag[10]), "r"(b_frag[(10) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[88]), "+f"(accum[(88) + 1]), "+f"(accum[(88) + 2]), "+f"(accum[(88) + 3])
                        : "r"(a_frag[8]), "r"(a_frag[(8) + 1]), "r"(a_frag[(8) + 2]), "r"(a_frag[(8) + 3]), "r"(b_frag[12]), "r"(b_frag[(12) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[92]), "+f"(accum[(92) + 1]), "+f"(accum[(92) + 2]), "+f"(accum[(92) + 3])
                        : "r"(a_frag[8]), "r"(a_frag[(8) + 1]), "r"(a_frag[(8) + 2]), "r"(a_frag[(8) + 3]), "r"(b_frag[14]), "r"(b_frag[(14) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[96]), "+f"(accum[(96) + 1]), "+f"(accum[(96) + 2]), "+f"(accum[(96) + 3])
                        : "r"(a_frag[12]), "r"(a_frag[(12) + 1]), "r"(a_frag[(12) + 2]), "r"(a_frag[(12) + 3]), "r"(b_frag[0]), "r"(b_frag[1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[100]), "+f"(accum[(100) + 1]), "+f"(accum[(100) + 2]), "+f"(accum[(100) + 3])
                        : "r"(a_frag[12]), "r"(a_frag[(12) + 1]), "r"(a_frag[(12) + 2]), "r"(a_frag[(12) + 3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[104]), "+f"(accum[(104) + 1]), "+f"(accum[(104) + 2]), "+f"(accum[(104) + 3])
                        : "r"(a_frag[12]), "r"(a_frag[(12) + 1]), "r"(a_frag[(12) + 2]), "r"(a_frag[(12) + 3]), "r"(b_frag[4]), "r"(b_frag[(4) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[108]), "+f"(accum[(108) + 1]), "+f"(accum[(108) + 2]), "+f"(accum[(108) + 3])
                        : "r"(a_frag[12]), "r"(a_frag[(12) + 1]), "r"(a_frag[(12) + 2]), "r"(a_frag[(12) + 3]), "r"(b_frag[6]), "r"(b_frag[(6) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[112]), "+f"(accum[(112) + 1]), "+f"(accum[(112) + 2]), "+f"(accum[(112) + 3])
                        : "r"(a_frag[12]), "r"(a_frag[(12) + 1]), "r"(a_frag[(12) + 2]), "r"(a_frag[(12) + 3]), "r"(b_frag[8]), "r"(b_frag[(8) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[116]), "+f"(accum[(116) + 1]), "+f"(accum[(116) + 2]), "+f"(accum[(116) + 3])
                        : "r"(a_frag[12]), "r"(a_frag[(12) + 1]), "r"(a_frag[(12) + 2]), "r"(a_frag[(12) + 3]), "r"(b_frag[10]), "r"(b_frag[(10) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[120]), "+f"(accum[(120) + 1]), "+f"(accum[(120) + 2]), "+f"(accum[(120) + 3])
                        : "r"(a_frag[12]), "r"(a_frag[(12) + 1]), "r"(a_frag[(12) + 2]), "r"(a_frag[(12) + 3]), "r"(b_frag[12]), "r"(b_frag[(12) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[124]), "+f"(accum[(124) + 1]), "+f"(accum[(124) + 2]), "+f"(accum[(124) + 3])
                        : "r"(a_frag[12]), "r"(a_frag[(12) + 1]), "r"(a_frag[(12) + 2]), "r"(a_frag[(12) + 3]), "r"(b_frag[14]), "r"(b_frag[(14) + 1]));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                        : "r"(A_stage_addr + mma_stage * 16384 + (unsigned int)((warp_m * 64 + (lane >> 3 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(32 + (lane >> 3 >> 1) * 16 ^ (warp_m * 64 + (lane >> 3 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_frag[4]), "=r"(a_frag[5]), "=r"(a_frag[6]), "=r"(a_frag[7])
                        : "r"(A_stage_addr + mma_stage * 16384 + (unsigned int)((warp_m * 64 + 16 + (lane >> 3 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(32 + (lane >> 3 >> 1) * 16 ^ (warp_m * 64 + 16 + (lane >> 3 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_frag[8]), "=r"(a_frag[9]), "=r"(a_frag[10]), "=r"(a_frag[11])
                        : "r"(A_stage_addr + mma_stage * 16384 + (unsigned int)((warp_m * 64 + 32 + (lane >> 3 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(32 + (lane >> 3 >> 1) * 16 ^ (warp_m * 64 + 32 + (lane >> 3 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_frag[12]), "=r"(a_frag[13]), "=r"(a_frag[14]), "=r"(a_frag[15])
                        : "r"(A_stage_addr + mma_stage * 16384 + (unsigned int)((warp_m * 64 + 48 + (lane >> 3 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(32 + (lane >> 3 >> 1) * 16 ^ (warp_m * 64 + 48 + (lane >> 3 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                        : "r"(B_stage_addr + mma_stage * 8192 + (unsigned int)((warp_n * 64 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(32 + (lane >> 3 & 1) * 16 ^ (warp_n * 64 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_frag[4]), "=r"(b_frag[5]), "=r"(b_frag[6]), "=r"(b_frag[7])
                        : "r"(B_stage_addr + mma_stage * 8192 + (unsigned int)((warp_n * 64 + 16 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(32 + (lane >> 3 & 1) * 16 ^ (warp_n * 64 + 16 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_frag[8]), "=r"(b_frag[9]), "=r"(b_frag[10]), "=r"(b_frag[11])
                        : "r"(B_stage_addr + mma_stage * 8192 + (unsigned int)((warp_n * 64 + 32 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(32 + (lane >> 3 & 1) * 16 ^ (warp_n * 64 + 32 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_frag[12]), "=r"(b_frag[13]), "=r"(b_frag[14]), "=r"(b_frag[15])
                        : "r"(B_stage_addr + mma_stage * 8192 + (unsigned int)((warp_n * 64 + 48 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(32 + (lane >> 3 & 1) * 16 ^ (warp_n * 64 + 48 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[0]), "+f"(accum[1]), "+f"(accum[2]), "+f"(accum[3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[4]), "+f"(accum[(4) + 1]), "+f"(accum[(4) + 2]), "+f"(accum[(4) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[8]), "+f"(accum[(8) + 1]), "+f"(accum[(8) + 2]), "+f"(accum[(8) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[4]), "r"(b_frag[(4) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[12]), "+f"(accum[(12) + 1]), "+f"(accum[(12) + 2]), "+f"(accum[(12) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[6]), "r"(b_frag[(6) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[16]), "+f"(accum[(16) + 1]), "+f"(accum[(16) + 2]), "+f"(accum[(16) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[8]), "r"(b_frag[(8) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[20]), "+f"(accum[(20) + 1]), "+f"(accum[(20) + 2]), "+f"(accum[(20) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[10]), "r"(b_frag[(10) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[24]), "+f"(accum[(24) + 1]), "+f"(accum[(24) + 2]), "+f"(accum[(24) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[12]), "r"(b_frag[(12) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[28]), "+f"(accum[(28) + 1]), "+f"(accum[(28) + 2]), "+f"(accum[(28) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[14]), "r"(b_frag[(14) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[32]), "+f"(accum[(32) + 1]), "+f"(accum[(32) + 2]), "+f"(accum[(32) + 3])
                        : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[0]), "r"(b_frag[1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[36]), "+f"(accum[(36) + 1]), "+f"(accum[(36) + 2]), "+f"(accum[(36) + 3])
                        : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[40]), "+f"(accum[(40) + 1]), "+f"(accum[(40) + 2]), "+f"(accum[(40) + 3])
                        : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[4]), "r"(b_frag[(4) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[44]), "+f"(accum[(44) + 1]), "+f"(accum[(44) + 2]), "+f"(accum[(44) + 3])
                        : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[6]), "r"(b_frag[(6) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[48]), "+f"(accum[(48) + 1]), "+f"(accum[(48) + 2]), "+f"(accum[(48) + 3])
                        : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[8]), "r"(b_frag[(8) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[52]), "+f"(accum[(52) + 1]), "+f"(accum[(52) + 2]), "+f"(accum[(52) + 3])
                        : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[10]), "r"(b_frag[(10) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[56]), "+f"(accum[(56) + 1]), "+f"(accum[(56) + 2]), "+f"(accum[(56) + 3])
                        : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[12]), "r"(b_frag[(12) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[60]), "+f"(accum[(60) + 1]), "+f"(accum[(60) + 2]), "+f"(accum[(60) + 3])
                        : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[14]), "r"(b_frag[(14) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[64]), "+f"(accum[(64) + 1]), "+f"(accum[(64) + 2]), "+f"(accum[(64) + 3])
                        : "r"(a_frag[8]), "r"(a_frag[(8) + 1]), "r"(a_frag[(8) + 2]), "r"(a_frag[(8) + 3]), "r"(b_frag[0]), "r"(b_frag[1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[68]), "+f"(accum[(68) + 1]), "+f"(accum[(68) + 2]), "+f"(accum[(68) + 3])
                        : "r"(a_frag[8]), "r"(a_frag[(8) + 1]), "r"(a_frag[(8) + 2]), "r"(a_frag[(8) + 3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[72]), "+f"(accum[(72) + 1]), "+f"(accum[(72) + 2]), "+f"(accum[(72) + 3])
                        : "r"(a_frag[8]), "r"(a_frag[(8) + 1]), "r"(a_frag[(8) + 2]), "r"(a_frag[(8) + 3]), "r"(b_frag[4]), "r"(b_frag[(4) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[76]), "+f"(accum[(76) + 1]), "+f"(accum[(76) + 2]), "+f"(accum[(76) + 3])
                        : "r"(a_frag[8]), "r"(a_frag[(8) + 1]), "r"(a_frag[(8) + 2]), "r"(a_frag[(8) + 3]), "r"(b_frag[6]), "r"(b_frag[(6) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[80]), "+f"(accum[(80) + 1]), "+f"(accum[(80) + 2]), "+f"(accum[(80) + 3])
                        : "r"(a_frag[8]), "r"(a_frag[(8) + 1]), "r"(a_frag[(8) + 2]), "r"(a_frag[(8) + 3]), "r"(b_frag[8]), "r"(b_frag[(8) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[84]), "+f"(accum[(84) + 1]), "+f"(accum[(84) + 2]), "+f"(accum[(84) + 3])
                        : "r"(a_frag[8]), "r"(a_frag[(8) + 1]), "r"(a_frag[(8) + 2]), "r"(a_frag[(8) + 3]), "r"(b_frag[10]), "r"(b_frag[(10) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[88]), "+f"(accum[(88) + 1]), "+f"(accum[(88) + 2]), "+f"(accum[(88) + 3])
                        : "r"(a_frag[8]), "r"(a_frag[(8) + 1]), "r"(a_frag[(8) + 2]), "r"(a_frag[(8) + 3]), "r"(b_frag[12]), "r"(b_frag[(12) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[92]), "+f"(accum[(92) + 1]), "+f"(accum[(92) + 2]), "+f"(accum[(92) + 3])
                        : "r"(a_frag[8]), "r"(a_frag[(8) + 1]), "r"(a_frag[(8) + 2]), "r"(a_frag[(8) + 3]), "r"(b_frag[14]), "r"(b_frag[(14) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[96]), "+f"(accum[(96) + 1]), "+f"(accum[(96) + 2]), "+f"(accum[(96) + 3])
                        : "r"(a_frag[12]), "r"(a_frag[(12) + 1]), "r"(a_frag[(12) + 2]), "r"(a_frag[(12) + 3]), "r"(b_frag[0]), "r"(b_frag[1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[100]), "+f"(accum[(100) + 1]), "+f"(accum[(100) + 2]), "+f"(accum[(100) + 3])
                        : "r"(a_frag[12]), "r"(a_frag[(12) + 1]), "r"(a_frag[(12) + 2]), "r"(a_frag[(12) + 3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[104]), "+f"(accum[(104) + 1]), "+f"(accum[(104) + 2]), "+f"(accum[(104) + 3])
                        : "r"(a_frag[12]), "r"(a_frag[(12) + 1]), "r"(a_frag[(12) + 2]), "r"(a_frag[(12) + 3]), "r"(b_frag[4]), "r"(b_frag[(4) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[108]), "+f"(accum[(108) + 1]), "+f"(accum[(108) + 2]), "+f"(accum[(108) + 3])
                        : "r"(a_frag[12]), "r"(a_frag[(12) + 1]), "r"(a_frag[(12) + 2]), "r"(a_frag[(12) + 3]), "r"(b_frag[6]), "r"(b_frag[(6) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[112]), "+f"(accum[(112) + 1]), "+f"(accum[(112) + 2]), "+f"(accum[(112) + 3])
                        : "r"(a_frag[12]), "r"(a_frag[(12) + 1]), "r"(a_frag[(12) + 2]), "r"(a_frag[(12) + 3]), "r"(b_frag[8]), "r"(b_frag[(8) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[116]), "+f"(accum[(116) + 1]), "+f"(accum[(116) + 2]), "+f"(accum[(116) + 3])
                        : "r"(a_frag[12]), "r"(a_frag[(12) + 1]), "r"(a_frag[(12) + 2]), "r"(a_frag[(12) + 3]), "r"(b_frag[10]), "r"(b_frag[(10) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[120]), "+f"(accum[(120) + 1]), "+f"(accum[(120) + 2]), "+f"(accum[(120) + 3])
                        : "r"(a_frag[12]), "r"(a_frag[(12) + 1]), "r"(a_frag[(12) + 2]), "r"(a_frag[(12) + 3]), "r"(b_frag[12]), "r"(b_frag[(12) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[124]), "+f"(accum[(124) + 1]), "+f"(accum[(124) + 2]), "+f"(accum[(124) + 3])
                        : "r"(a_frag[12]), "r"(a_frag[(12) + 1]), "r"(a_frag[(12) + 2]), "r"(a_frag[(12) + 3]), "r"(b_frag[14]), "r"(b_frag[(14) + 1]));
                    __syncwarp();
                    if (elect_sync()) {
                        mbarrier_arrive(ab_empty_addr + (mma_stage) * 8);
                    }
                    mma_stage += 1;
                    if (mma_stage == 3) { mma_stage = 0; _phase_ab_full ^= 1; }
                    mbarrier_wait(ab_full_addr + (mma_stage) * 8, _phase_ab_full);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                        : "r"(A_stage_addr + mma_stage * 16384 + (unsigned int)((warp_m * 64 + (lane >> 3 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)((lane >> 3 >> 1) * 16 ^ (warp_m * 64 + (lane >> 3 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_frag[4]), "=r"(a_frag[5]), "=r"(a_frag[6]), "=r"(a_frag[7])
                        : "r"(A_stage_addr + mma_stage * 16384 + (unsigned int)((warp_m * 64 + 16 + (lane >> 3 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)((lane >> 3 >> 1) * 16 ^ (warp_m * 64 + 16 + (lane >> 3 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_frag[8]), "=r"(a_frag[9]), "=r"(a_frag[10]), "=r"(a_frag[11])
                        : "r"(A_stage_addr + mma_stage * 16384 + (unsigned int)((warp_m * 64 + 32 + (lane >> 3 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)((lane >> 3 >> 1) * 16 ^ (warp_m * 64 + 32 + (lane >> 3 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_frag[12]), "=r"(a_frag[13]), "=r"(a_frag[14]), "=r"(a_frag[15])
                        : "r"(A_stage_addr + mma_stage * 16384 + (unsigned int)((warp_m * 64 + 48 + (lane >> 3 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)((lane >> 3 >> 1) * 16 ^ (warp_m * 64 + 48 + (lane >> 3 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                        : "r"(B_stage_addr + mma_stage * 8192 + (unsigned int)((warp_n * 64 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)((lane >> 3 & 1) * 16 ^ (warp_n * 64 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_frag[4]), "=r"(b_frag[5]), "=r"(b_frag[6]), "=r"(b_frag[7])
                        : "r"(B_stage_addr + mma_stage * 8192 + (unsigned int)((warp_n * 64 + 16 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)((lane >> 3 & 1) * 16 ^ (warp_n * 64 + 16 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_frag[8]), "=r"(b_frag[9]), "=r"(b_frag[10]), "=r"(b_frag[11])
                        : "r"(B_stage_addr + mma_stage * 8192 + (unsigned int)((warp_n * 64 + 32 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)((lane >> 3 & 1) * 16 ^ (warp_n * 64 + 32 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_frag[12]), "=r"(b_frag[13]), "=r"(b_frag[14]), "=r"(b_frag[15])
                        : "r"(B_stage_addr + mma_stage * 8192 + (unsigned int)((warp_n * 64 + 48 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)((lane >> 3 & 1) * 16 ^ (warp_n * 64 + 48 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[0]), "+f"(accum[1]), "+f"(accum[2]), "+f"(accum[3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[4]), "+f"(accum[(4) + 1]), "+f"(accum[(4) + 2]), "+f"(accum[(4) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[8]), "+f"(accum[(8) + 1]), "+f"(accum[(8) + 2]), "+f"(accum[(8) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[4]), "r"(b_frag[(4) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[12]), "+f"(accum[(12) + 1]), "+f"(accum[(12) + 2]), "+f"(accum[(12) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[6]), "r"(b_frag[(6) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[16]), "+f"(accum[(16) + 1]), "+f"(accum[(16) + 2]), "+f"(accum[(16) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[8]), "r"(b_frag[(8) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[20]), "+f"(accum[(20) + 1]), "+f"(accum[(20) + 2]), "+f"(accum[(20) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[10]), "r"(b_frag[(10) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[24]), "+f"(accum[(24) + 1]), "+f"(accum[(24) + 2]), "+f"(accum[(24) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[12]), "r"(b_frag[(12) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[28]), "+f"(accum[(28) + 1]), "+f"(accum[(28) + 2]), "+f"(accum[(28) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[14]), "r"(b_frag[(14) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[32]), "+f"(accum[(32) + 1]), "+f"(accum[(32) + 2]), "+f"(accum[(32) + 3])
                        : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[0]), "r"(b_frag[1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[36]), "+f"(accum[(36) + 1]), "+f"(accum[(36) + 2]), "+f"(accum[(36) + 3])
                        : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[40]), "+f"(accum[(40) + 1]), "+f"(accum[(40) + 2]), "+f"(accum[(40) + 3])
                        : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[4]), "r"(b_frag[(4) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[44]), "+f"(accum[(44) + 1]), "+f"(accum[(44) + 2]), "+f"(accum[(44) + 3])
                        : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[6]), "r"(b_frag[(6) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[48]), "+f"(accum[(48) + 1]), "+f"(accum[(48) + 2]), "+f"(accum[(48) + 3])
                        : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[8]), "r"(b_frag[(8) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[52]), "+f"(accum[(52) + 1]), "+f"(accum[(52) + 2]), "+f"(accum[(52) + 3])
                        : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[10]), "r"(b_frag[(10) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[56]), "+f"(accum[(56) + 1]), "+f"(accum[(56) + 2]), "+f"(accum[(56) + 3])
                        : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[12]), "r"(b_frag[(12) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[60]), "+f"(accum[(60) + 1]), "+f"(accum[(60) + 2]), "+f"(accum[(60) + 3])
                        : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[14]), "r"(b_frag[(14) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[64]), "+f"(accum[(64) + 1]), "+f"(accum[(64) + 2]), "+f"(accum[(64) + 3])
                        : "r"(a_frag[8]), "r"(a_frag[(8) + 1]), "r"(a_frag[(8) + 2]), "r"(a_frag[(8) + 3]), "r"(b_frag[0]), "r"(b_frag[1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[68]), "+f"(accum[(68) + 1]), "+f"(accum[(68) + 2]), "+f"(accum[(68) + 3])
                        : "r"(a_frag[8]), "r"(a_frag[(8) + 1]), "r"(a_frag[(8) + 2]), "r"(a_frag[(8) + 3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[72]), "+f"(accum[(72) + 1]), "+f"(accum[(72) + 2]), "+f"(accum[(72) + 3])
                        : "r"(a_frag[8]), "r"(a_frag[(8) + 1]), "r"(a_frag[(8) + 2]), "r"(a_frag[(8) + 3]), "r"(b_frag[4]), "r"(b_frag[(4) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[76]), "+f"(accum[(76) + 1]), "+f"(accum[(76) + 2]), "+f"(accum[(76) + 3])
                        : "r"(a_frag[8]), "r"(a_frag[(8) + 1]), "r"(a_frag[(8) + 2]), "r"(a_frag[(8) + 3]), "r"(b_frag[6]), "r"(b_frag[(6) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[80]), "+f"(accum[(80) + 1]), "+f"(accum[(80) + 2]), "+f"(accum[(80) + 3])
                        : "r"(a_frag[8]), "r"(a_frag[(8) + 1]), "r"(a_frag[(8) + 2]), "r"(a_frag[(8) + 3]), "r"(b_frag[8]), "r"(b_frag[(8) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[84]), "+f"(accum[(84) + 1]), "+f"(accum[(84) + 2]), "+f"(accum[(84) + 3])
                        : "r"(a_frag[8]), "r"(a_frag[(8) + 1]), "r"(a_frag[(8) + 2]), "r"(a_frag[(8) + 3]), "r"(b_frag[10]), "r"(b_frag[(10) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[88]), "+f"(accum[(88) + 1]), "+f"(accum[(88) + 2]), "+f"(accum[(88) + 3])
                        : "r"(a_frag[8]), "r"(a_frag[(8) + 1]), "r"(a_frag[(8) + 2]), "r"(a_frag[(8) + 3]), "r"(b_frag[12]), "r"(b_frag[(12) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[92]), "+f"(accum[(92) + 1]), "+f"(accum[(92) + 2]), "+f"(accum[(92) + 3])
                        : "r"(a_frag[8]), "r"(a_frag[(8) + 1]), "r"(a_frag[(8) + 2]), "r"(a_frag[(8) + 3]), "r"(b_frag[14]), "r"(b_frag[(14) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[96]), "+f"(accum[(96) + 1]), "+f"(accum[(96) + 2]), "+f"(accum[(96) + 3])
                        : "r"(a_frag[12]), "r"(a_frag[(12) + 1]), "r"(a_frag[(12) + 2]), "r"(a_frag[(12) + 3]), "r"(b_frag[0]), "r"(b_frag[1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[100]), "+f"(accum[(100) + 1]), "+f"(accum[(100) + 2]), "+f"(accum[(100) + 3])
                        : "r"(a_frag[12]), "r"(a_frag[(12) + 1]), "r"(a_frag[(12) + 2]), "r"(a_frag[(12) + 3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[104]), "+f"(accum[(104) + 1]), "+f"(accum[(104) + 2]), "+f"(accum[(104) + 3])
                        : "r"(a_frag[12]), "r"(a_frag[(12) + 1]), "r"(a_frag[(12) + 2]), "r"(a_frag[(12) + 3]), "r"(b_frag[4]), "r"(b_frag[(4) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[108]), "+f"(accum[(108) + 1]), "+f"(accum[(108) + 2]), "+f"(accum[(108) + 3])
                        : "r"(a_frag[12]), "r"(a_frag[(12) + 1]), "r"(a_frag[(12) + 2]), "r"(a_frag[(12) + 3]), "r"(b_frag[6]), "r"(b_frag[(6) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[112]), "+f"(accum[(112) + 1]), "+f"(accum[(112) + 2]), "+f"(accum[(112) + 3])
                        : "r"(a_frag[12]), "r"(a_frag[(12) + 1]), "r"(a_frag[(12) + 2]), "r"(a_frag[(12) + 3]), "r"(b_frag[8]), "r"(b_frag[(8) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[116]), "+f"(accum[(116) + 1]), "+f"(accum[(116) + 2]), "+f"(accum[(116) + 3])
                        : "r"(a_frag[12]), "r"(a_frag[(12) + 1]), "r"(a_frag[(12) + 2]), "r"(a_frag[(12) + 3]), "r"(b_frag[10]), "r"(b_frag[(10) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[120]), "+f"(accum[(120) + 1]), "+f"(accum[(120) + 2]), "+f"(accum[(120) + 3])
                        : "r"(a_frag[12]), "r"(a_frag[(12) + 1]), "r"(a_frag[(12) + 2]), "r"(a_frag[(12) + 3]), "r"(b_frag[12]), "r"(b_frag[(12) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[124]), "+f"(accum[(124) + 1]), "+f"(accum[(124) + 2]), "+f"(accum[(124) + 3])
                        : "r"(a_frag[12]), "r"(a_frag[(12) + 1]), "r"(a_frag[(12) + 2]), "r"(a_frag[(12) + 3]), "r"(b_frag[14]), "r"(b_frag[(14) + 1]));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                        : "r"(A_stage_addr + mma_stage * 16384 + (unsigned int)((warp_m * 64 + (lane >> 3 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(32 + (lane >> 3 >> 1) * 16 ^ (warp_m * 64 + (lane >> 3 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_frag[4]), "=r"(a_frag[5]), "=r"(a_frag[6]), "=r"(a_frag[7])
                        : "r"(A_stage_addr + mma_stage * 16384 + (unsigned int)((warp_m * 64 + 16 + (lane >> 3 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(32 + (lane >> 3 >> 1) * 16 ^ (warp_m * 64 + 16 + (lane >> 3 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_frag[8]), "=r"(a_frag[9]), "=r"(a_frag[10]), "=r"(a_frag[11])
                        : "r"(A_stage_addr + mma_stage * 16384 + (unsigned int)((warp_m * 64 + 32 + (lane >> 3 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(32 + (lane >> 3 >> 1) * 16 ^ (warp_m * 64 + 32 + (lane >> 3 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_frag[12]), "=r"(a_frag[13]), "=r"(a_frag[14]), "=r"(a_frag[15])
                        : "r"(A_stage_addr + mma_stage * 16384 + (unsigned int)((warp_m * 64 + 48 + (lane >> 3 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(32 + (lane >> 3 >> 1) * 16 ^ (warp_m * 64 + 48 + (lane >> 3 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                        : "r"(B_stage_addr + mma_stage * 8192 + (unsigned int)((warp_n * 64 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(32 + (lane >> 3 & 1) * 16 ^ (warp_n * 64 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_frag[4]), "=r"(b_frag[5]), "=r"(b_frag[6]), "=r"(b_frag[7])
                        : "r"(B_stage_addr + mma_stage * 8192 + (unsigned int)((warp_n * 64 + 16 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(32 + (lane >> 3 & 1) * 16 ^ (warp_n * 64 + 16 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_frag[8]), "=r"(b_frag[9]), "=r"(b_frag[10]), "=r"(b_frag[11])
                        : "r"(B_stage_addr + mma_stage * 8192 + (unsigned int)((warp_n * 64 + 32 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(32 + (lane >> 3 & 1) * 16 ^ (warp_n * 64 + 32 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_frag[12]), "=r"(b_frag[13]), "=r"(b_frag[14]), "=r"(b_frag[15])
                        : "r"(B_stage_addr + mma_stage * 8192 + (unsigned int)((warp_n * 64 + 48 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(32 + (lane >> 3 & 1) * 16 ^ (warp_n * 64 + 48 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[0]), "+f"(accum[1]), "+f"(accum[2]), "+f"(accum[3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[4]), "+f"(accum[(4) + 1]), "+f"(accum[(4) + 2]), "+f"(accum[(4) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[8]), "+f"(accum[(8) + 1]), "+f"(accum[(8) + 2]), "+f"(accum[(8) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[4]), "r"(b_frag[(4) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[12]), "+f"(accum[(12) + 1]), "+f"(accum[(12) + 2]), "+f"(accum[(12) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[6]), "r"(b_frag[(6) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[16]), "+f"(accum[(16) + 1]), "+f"(accum[(16) + 2]), "+f"(accum[(16) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[8]), "r"(b_frag[(8) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[20]), "+f"(accum[(20) + 1]), "+f"(accum[(20) + 2]), "+f"(accum[(20) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[10]), "r"(b_frag[(10) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[24]), "+f"(accum[(24) + 1]), "+f"(accum[(24) + 2]), "+f"(accum[(24) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[12]), "r"(b_frag[(12) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[28]), "+f"(accum[(28) + 1]), "+f"(accum[(28) + 2]), "+f"(accum[(28) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[14]), "r"(b_frag[(14) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[32]), "+f"(accum[(32) + 1]), "+f"(accum[(32) + 2]), "+f"(accum[(32) + 3])
                        : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[0]), "r"(b_frag[1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[36]), "+f"(accum[(36) + 1]), "+f"(accum[(36) + 2]), "+f"(accum[(36) + 3])
                        : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[40]), "+f"(accum[(40) + 1]), "+f"(accum[(40) + 2]), "+f"(accum[(40) + 3])
                        : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[4]), "r"(b_frag[(4) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[44]), "+f"(accum[(44) + 1]), "+f"(accum[(44) + 2]), "+f"(accum[(44) + 3])
                        : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[6]), "r"(b_frag[(6) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[48]), "+f"(accum[(48) + 1]), "+f"(accum[(48) + 2]), "+f"(accum[(48) + 3])
                        : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[8]), "r"(b_frag[(8) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[52]), "+f"(accum[(52) + 1]), "+f"(accum[(52) + 2]), "+f"(accum[(52) + 3])
                        : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[10]), "r"(b_frag[(10) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[56]), "+f"(accum[(56) + 1]), "+f"(accum[(56) + 2]), "+f"(accum[(56) + 3])
                        : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[12]), "r"(b_frag[(12) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[60]), "+f"(accum[(60) + 1]), "+f"(accum[(60) + 2]), "+f"(accum[(60) + 3])
                        : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[14]), "r"(b_frag[(14) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[64]), "+f"(accum[(64) + 1]), "+f"(accum[(64) + 2]), "+f"(accum[(64) + 3])
                        : "r"(a_frag[8]), "r"(a_frag[(8) + 1]), "r"(a_frag[(8) + 2]), "r"(a_frag[(8) + 3]), "r"(b_frag[0]), "r"(b_frag[1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[68]), "+f"(accum[(68) + 1]), "+f"(accum[(68) + 2]), "+f"(accum[(68) + 3])
                        : "r"(a_frag[8]), "r"(a_frag[(8) + 1]), "r"(a_frag[(8) + 2]), "r"(a_frag[(8) + 3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[72]), "+f"(accum[(72) + 1]), "+f"(accum[(72) + 2]), "+f"(accum[(72) + 3])
                        : "r"(a_frag[8]), "r"(a_frag[(8) + 1]), "r"(a_frag[(8) + 2]), "r"(a_frag[(8) + 3]), "r"(b_frag[4]), "r"(b_frag[(4) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[76]), "+f"(accum[(76) + 1]), "+f"(accum[(76) + 2]), "+f"(accum[(76) + 3])
                        : "r"(a_frag[8]), "r"(a_frag[(8) + 1]), "r"(a_frag[(8) + 2]), "r"(a_frag[(8) + 3]), "r"(b_frag[6]), "r"(b_frag[(6) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[80]), "+f"(accum[(80) + 1]), "+f"(accum[(80) + 2]), "+f"(accum[(80) + 3])
                        : "r"(a_frag[8]), "r"(a_frag[(8) + 1]), "r"(a_frag[(8) + 2]), "r"(a_frag[(8) + 3]), "r"(b_frag[8]), "r"(b_frag[(8) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[84]), "+f"(accum[(84) + 1]), "+f"(accum[(84) + 2]), "+f"(accum[(84) + 3])
                        : "r"(a_frag[8]), "r"(a_frag[(8) + 1]), "r"(a_frag[(8) + 2]), "r"(a_frag[(8) + 3]), "r"(b_frag[10]), "r"(b_frag[(10) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[88]), "+f"(accum[(88) + 1]), "+f"(accum[(88) + 2]), "+f"(accum[(88) + 3])
                        : "r"(a_frag[8]), "r"(a_frag[(8) + 1]), "r"(a_frag[(8) + 2]), "r"(a_frag[(8) + 3]), "r"(b_frag[12]), "r"(b_frag[(12) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[92]), "+f"(accum[(92) + 1]), "+f"(accum[(92) + 2]), "+f"(accum[(92) + 3])
                        : "r"(a_frag[8]), "r"(a_frag[(8) + 1]), "r"(a_frag[(8) + 2]), "r"(a_frag[(8) + 3]), "r"(b_frag[14]), "r"(b_frag[(14) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[96]), "+f"(accum[(96) + 1]), "+f"(accum[(96) + 2]), "+f"(accum[(96) + 3])
                        : "r"(a_frag[12]), "r"(a_frag[(12) + 1]), "r"(a_frag[(12) + 2]), "r"(a_frag[(12) + 3]), "r"(b_frag[0]), "r"(b_frag[1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[100]), "+f"(accum[(100) + 1]), "+f"(accum[(100) + 2]), "+f"(accum[(100) + 3])
                        : "r"(a_frag[12]), "r"(a_frag[(12) + 1]), "r"(a_frag[(12) + 2]), "r"(a_frag[(12) + 3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[104]), "+f"(accum[(104) + 1]), "+f"(accum[(104) + 2]), "+f"(accum[(104) + 3])
                        : "r"(a_frag[12]), "r"(a_frag[(12) + 1]), "r"(a_frag[(12) + 2]), "r"(a_frag[(12) + 3]), "r"(b_frag[4]), "r"(b_frag[(4) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[108]), "+f"(accum[(108) + 1]), "+f"(accum[(108) + 2]), "+f"(accum[(108) + 3])
                        : "r"(a_frag[12]), "r"(a_frag[(12) + 1]), "r"(a_frag[(12) + 2]), "r"(a_frag[(12) + 3]), "r"(b_frag[6]), "r"(b_frag[(6) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[112]), "+f"(accum[(112) + 1]), "+f"(accum[(112) + 2]), "+f"(accum[(112) + 3])
                        : "r"(a_frag[12]), "r"(a_frag[(12) + 1]), "r"(a_frag[(12) + 2]), "r"(a_frag[(12) + 3]), "r"(b_frag[8]), "r"(b_frag[(8) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[116]), "+f"(accum[(116) + 1]), "+f"(accum[(116) + 2]), "+f"(accum[(116) + 3])
                        : "r"(a_frag[12]), "r"(a_frag[(12) + 1]), "r"(a_frag[(12) + 2]), "r"(a_frag[(12) + 3]), "r"(b_frag[10]), "r"(b_frag[(10) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[120]), "+f"(accum[(120) + 1]), "+f"(accum[(120) + 2]), "+f"(accum[(120) + 3])
                        : "r"(a_frag[12]), "r"(a_frag[(12) + 1]), "r"(a_frag[(12) + 2]), "r"(a_frag[(12) + 3]), "r"(b_frag[12]), "r"(b_frag[(12) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(accum[124]), "+f"(accum[(124) + 1]), "+f"(accum[(124) + 2]), "+f"(accum[(124) + 3])
                        : "r"(a_frag[12]), "r"(a_frag[(12) + 1]), "r"(a_frag[(12) + 2]), "r"(a_frag[(12) + 3]), "r"(b_frag[14]), "r"(b_frag[(14) + 1]));
                    __syncwarp();
                    if (elect_sync()) {
                        mbarrier_arrive(ab_empty_addr + (mma_stage) * 8);
                    }
                    mma_stage += 1;
                    if (mma_stage == 3) { mma_stage = 0; _phase_ab_full ^= 1; }
                }
                if (tile_m_1 > 2147483647) {
                    float chk = 0.0f;
                    chk = chk + accum[0];
                    chk = chk + accum[1];
                    chk = chk + accum[2];
                    chk = chk + accum[3];
                    chk = chk + accum[4];
                    chk = chk + accum[5];
                    chk = chk + accum[6];
                    chk = chk + accum[7];
                    chk = chk + accum[8];
                    chk = chk + accum[9];
                    chk = chk + accum[10];
                    chk = chk + accum[11];
                    chk = chk + accum[12];
                    chk = chk + accum[13];
                    chk = chk + accum[14];
                    chk = chk + accum[15];
                    chk = chk + accum[16];
                    chk = chk + accum[17];
                    chk = chk + accum[18];
                    chk = chk + accum[19];
                    chk = chk + accum[20];
                    chk = chk + accum[21];
                    chk = chk + accum[22];
                    chk = chk + accum[23];
                    chk = chk + accum[24];
                    chk = chk + accum[25];
                    chk = chk + accum[26];
                    chk = chk + accum[27];
                    chk = chk + accum[28];
                    chk = chk + accum[29];
                    chk = chk + accum[30];
                    chk = chk + accum[31];
                    chk = chk + accum[32];
                    chk = chk + accum[33];
                    chk = chk + accum[34];
                    chk = chk + accum[35];
                    chk = chk + accum[36];
                    chk = chk + accum[37];
                    chk = chk + accum[38];
                    chk = chk + accum[39];
                    chk = chk + accum[40];
                    chk = chk + accum[41];
                    chk = chk + accum[42];
                    chk = chk + accum[43];
                    chk = chk + accum[44];
                    chk = chk + accum[45];
                    chk = chk + accum[46];
                    chk = chk + accum[47];
                    chk = chk + accum[48];
                    chk = chk + accum[49];
                    chk = chk + accum[50];
                    chk = chk + accum[51];
                    chk = chk + accum[52];
                    chk = chk + accum[53];
                    chk = chk + accum[54];
                    chk = chk + accum[55];
                    chk = chk + accum[56];
                    chk = chk + accum[57];
                    chk = chk + accum[58];
                    chk = chk + accum[59];
                    chk = chk + accum[60];
                    chk = chk + accum[61];
                    chk = chk + accum[62];
                    chk = chk + accum[63];
                    chk = chk + accum[64];
                    chk = chk + accum[65];
                    chk = chk + accum[66];
                    chk = chk + accum[67];
                    chk = chk + accum[68];
                    chk = chk + accum[69];
                    chk = chk + accum[70];
                    chk = chk + accum[71];
                    chk = chk + accum[72];
                    chk = chk + accum[73];
                    chk = chk + accum[74];
                    chk = chk + accum[75];
                    chk = chk + accum[76];
                    chk = chk + accum[77];
                    chk = chk + accum[78];
                    chk = chk + accum[79];
                    chk = chk + accum[80];
                    chk = chk + accum[81];
                    chk = chk + accum[82];
                    chk = chk + accum[83];
                    chk = chk + accum[84];
                    chk = chk + accum[85];
                    chk = chk + accum[86];
                    chk = chk + accum[87];
                    chk = chk + accum[88];
                    chk = chk + accum[89];
                    chk = chk + accum[90];
                    chk = chk + accum[91];
                    chk = chk + accum[92];
                    chk = chk + accum[93];
                    chk = chk + accum[94];
                    chk = chk + accum[95];
                    chk = chk + accum[96];
                    chk = chk + accum[97];
                    chk = chk + accum[98];
                    chk = chk + accum[99];
                    chk = chk + accum[100];
                    chk = chk + accum[101];
                    chk = chk + accum[102];
                    chk = chk + accum[103];
                    chk = chk + accum[104];
                    chk = chk + accum[105];
                    chk = chk + accum[106];
                    chk = chk + accum[107];
                    chk = chk + accum[108];
                    chk = chk + accum[109];
                    chk = chk + accum[110];
                    chk = chk + accum[111];
                    chk = chk + accum[112];
                    chk = chk + accum[113];
                    chk = chk + accum[114];
                    chk = chk + accum[115];
                    chk = chk + accum[116];
                    chk = chk + accum[117];
                    chk = chk + accum[118];
                    chk = chk + accum[119];
                    chk = chk + accum[120];
                    chk = chk + accum[121];
                    chk = chk + accum[122];
                    chk = chk + accum[123];
                    chk = chk + accum[124];
                    chk = chk + accum[125];
                    chk = chk + accum[126];
                    chk = chk + accum[127];
                    out[tile_1] = __as_u32(chk);
                }
                int srow_st = warp_m * 8 + (lane & 7);
                int smat = lane >> 3;
                int srow = role_tid >> 3;
                int c = role_tid & 7;
                int col = tile_n_1 * 128 + c * 16;
                int grow_base = tile_m_1 * 256 + (srow >> 3) * 64 + (srow & 7);
                float ops[32];
                int meta[2];
                meta[0] = gate_index[((grow_base < M) ? grow_base : 0)];
                meta[1] = ((meta[0] >= 0 && meta[0] < 9) ? 1 : 0);
                {
                    const uint4* _vptr_0 = reinterpret_cast<const uint4*>(gate + (((meta[1] == 1) ? meta[0] : 0) * 5376 + col) + 0);
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
                                : "=f"((&ops[0 + _blk * 8 + _pair * 2])[0]), "=f"((&ops[0 + _blk * 8 + _pair * 2])[1])
                                : "r"(_vpairs_0[_pair]));
                        }
                    }
                }
                {
                    const uint4* _vptr_1 = reinterpret_cast<const uint4*>(gate + (((meta[1] == 1) ? meta[0] : 0) * 5376 + col + 8) + 0);
                    uint4 _vld_1[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_1[_blk] = _vptr_1[_blk];
                        uint32_t* _vpairs_1 = reinterpret_cast<uint32_t*>(&_vld_1[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            asm volatile(
                                "{\n\t"
                                "shl.b32 %0, %2, 16;\n\t"
                                "and.b32 %1, %2, 0xffff0000;\n\t"
                                "}\n"
                                : "=f"((&ops[8 + _blk * 8 + _pair * 2])[0]), "=f"((&ops[8 + _blk * 8 + _pair * 2])[1])
                                : "r"(_vpairs_1[_pair]));
                        }
                    }
                }
                {
                    const uint4* _vptr_2 = reinterpret_cast<const uint4*>(residual + (((grow_base < M) ? grow_base : 0) * 5376 + col) + 0);
                    uint4 _vld_2[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_2[_blk] = _vptr_2[_blk];
                        uint32_t* _vpairs_2 = reinterpret_cast<uint32_t*>(&_vld_2[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            asm volatile(
                                "{\n\t"
                                "shl.b32 %0, %2, 16;\n\t"
                                "and.b32 %1, %2, 0xffff0000;\n\t"
                                "}\n"
                                : "=f"((&ops[16 + _blk * 8 + _pair * 2])[0]), "=f"((&ops[16 + _blk * 8 + _pair * 2])[1])
                                : "r"(_vpairs_2[_pair]));
                        }
                    }
                }
                {
                    const uint4* _vptr_3 = reinterpret_cast<const uint4*>(residual + (((grow_base < M) ? grow_base : 0) * 5376 + col + 8) + 0);
                    uint4 _vld_3[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_3[_blk] = _vptr_3[_blk];
                        uint32_t* _vpairs_3 = reinterpret_cast<uint32_t*>(&_vld_3[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            asm volatile(
                                "{\n\t"
                                "shl.b32 %0, %2, 16;\n\t"
                                "and.b32 %1, %2, 0xffff0000;\n\t"
                                "}\n"
                                : "=f"((&ops[24 + _blk * 8 + _pair * 2])[0]), "=f"((&ops[24 + _blk * 8 + _pair * 2])[1])
                                : "r"(_vpairs_3[_pair]));
                        }
                    }
                }
                float scaled[16];
                scaled[0] = accum[0] * sa[0] * sw[0];
                scaled[1] = accum[1] * sa[0] * sw[1];
                scaled[2] = accum[4] * sa[0] * sw[2];
                scaled[3] = accum[5] * sa[0] * sw[3];
                scaled[4] = accum[8] * sa[0] * sw[4];
                scaled[5] = accum[9] * sa[0] * sw[5];
                scaled[6] = accum[12] * sa[0] * sw[6];
                scaled[7] = accum[13] * sa[0] * sw[7];
                scaled[8] = accum[16] * sa[0] * sw[8];
                scaled[9] = accum[17] * sa[0] * sw[9];
                scaled[10] = accum[20] * sa[0] * sw[10];
                scaled[11] = accum[21] * sa[0] * sw[11];
                scaled[12] = accum[24] * sa[0] * sw[12];
                scaled[13] = accum[25] * sa[0] * sw[13];
                scaled[14] = accum[28] * sa[0] * sw[14];
                scaled[15] = accum[29] * sa[0] * sw[15];
                uint32_t scaled_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled[_lp*2 + 0], scaled[_lp*2+1 + 0]));
                    scaled_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t _stmatrix_addr_4 = static_cast<uint32_t>(staging_addr + (unsigned int)(srow_st * 256) + (unsigned int)((warp_n * 64 + smat * 8) * 2 ^ (srow_st & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_4), "r"(*reinterpret_cast<const uint32_t*>(&scaled_bf16[0])), "r"(*reinterpret_cast<const uint32_t*>(&scaled_bf16[1])), "r"(*reinterpret_cast<const uint32_t*>(&scaled_bf16[2])), "r"(*reinterpret_cast<const uint32_t*>(&scaled_bf16[3]))
                    : "memory");
                uint32_t _stmatrix_addr_5 = static_cast<uint32_t>(staging_addr + (unsigned int)(srow_st * 256) + (unsigned int)((warp_n * 64 + (4 + smat) * 8) * 2 ^ (srow_st & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_5), "r"(*reinterpret_cast<const uint32_t*>(&scaled_bf16[4])), "r"(*reinterpret_cast<const uint32_t*>(&scaled_bf16[5])), "r"(*reinterpret_cast<const uint32_t*>(&scaled_bf16[6])), "r"(*reinterpret_cast<const uint32_t*>(&scaled_bf16[7]))
                    : "memory");
                asm volatile("barrier.sync 8, 256;" ::: "memory");
                int grow = grow_base;
                if (grow < M) {
                    unsigned int o_words[8];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&o_words[0])), "=r"(*reinterpret_cast<uint32_t*>(&o_words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&o_words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&o_words[(0) + 3]))
                        : "r"(staging_addr + (unsigned int)(srow * 256) + (unsigned int)(c * 32 ^ (srow & 7) << 4)));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&o_words[4])), "=r"(*reinterpret_cast<uint32_t*>(&o_words[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&o_words[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&o_words[(4) + 3]))
                        : "r"(staging_addr + (unsigned int)(srow * 256) + (unsigned int)(c * 32 + 16 ^ (srow & 7) << 4)));
                    float o_words_f32[16];
                    #pragma unroll
                    for (int _pair = 0; _pair < 8; _pair++) {
                        asm volatile(
                            "{\n\t"
                            "shl.b32 %0, %2, 16;\n\t"
                            "and.b32 %1, %2, 0xffff0000;\n\t"
                            "}\n"
                            : "=f"((&o_words_f32[_pair * 2])[0]), "=f"((&o_words_f32[_pair * 2])[1])
                            : "r"(o_words[_pair]));
                    }
                    int valid = meta[1];
                    float p_raw[16];
                    for (int e = 0; e < 8; e++) {
                        float gv0 = ops[e];
                        float gv1 = ops[8 + e];
                        float gs0 = ((valid == 1) ? gv0 : 0.0f);
                        float gs1 = ((valid == 1) ? gv1 : 0.0f);
                        p_raw[e] = gs0 * o_words_f32[e];
                        p_raw[8 + e] = gs1 * o_words_f32[8 + e];
                    }
                    uint32_t p_raw_bf16[8];
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(p_raw[_lp*2 + 0], p_raw[_lp*2+1 + 0]));
                        p_raw_bf16[_lp] = *(uint32_t*)&_bf2;
                    }
                    float p_raw_bf16_f32[16];
                    #pragma unroll
                    for (int _pair = 0; _pair < 8; _pair++) {
                        asm volatile(
                            "{\n\t"
                            "shl.b32 %0, %2, 16;\n\t"
                            "and.b32 %1, %2, 0xffff0000;\n\t"
                            "}\n"
                            : "=f"((&p_raw_bf16_f32[_pair * 2])[0]), "=f"((&p_raw_bf16_f32[_pair * 2])[1])
                            : "r"(p_raw_bf16[_pair]));
                    }
                    float out_raw[16];
                    for (int e_1 = 0; e_1 < 8; e_1++) {
                        float rv0 = ops[16 + e_1];
                        float rv1 = ops[24 + e_1];
                        out_raw[e_1] = rv0 + p_raw_bf16_f32[e_1];
                        out_raw[8 + e_1] = rv1 + p_raw_bf16_f32[8 + e_1];
                    }
                    uint32_t out_raw_bf16[8];
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_raw[_lp*2 + 0], out_raw[_lp*2+1 + 0]));
                        out_raw_bf16[_lp] = *(uint32_t*)&_bf2;
                    }
                    if (grow < M) {
                        reinterpret_cast<int4*>(out + ((grow * 5376 + col) / 2))[0] = reinterpret_cast<int4*>(out_raw_bf16 + 0)[0];
                        reinterpret_cast<int4*>(out + ((grow * 5376 + col) / 2 + 4))[0] = reinterpret_cast<int4*>(out_raw_bf16 + 4)[0];
                    }
                }
                asm volatile("barrier.sync 8, 256;" ::: "memory");
                meta[0] = gate_index[((grow_base + 8 < M) ? grow_base + 8 : 0)];
                meta[1] = ((meta[0] >= 0 && meta[0] < 9) ? 1 : 0);
                {
                    const uint4* _vptr_6 = reinterpret_cast<const uint4*>(gate + (((meta[1] == 1) ? meta[0] : 0) * 5376 + col) + 0);
                    uint4 _vld_6[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_6[_blk] = _vptr_6[_blk];
                        uint32_t* _vpairs_6 = reinterpret_cast<uint32_t*>(&_vld_6[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            asm volatile(
                                "{\n\t"
                                "shl.b32 %0, %2, 16;\n\t"
                                "and.b32 %1, %2, 0xffff0000;\n\t"
                                "}\n"
                                : "=f"((&ops[0 + _blk * 8 + _pair * 2])[0]), "=f"((&ops[0 + _blk * 8 + _pair * 2])[1])
                                : "r"(_vpairs_6[_pair]));
                        }
                    }
                }
                {
                    const uint4* _vptr_7 = reinterpret_cast<const uint4*>(gate + (((meta[1] == 1) ? meta[0] : 0) * 5376 + col + 8) + 0);
                    uint4 _vld_7[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_7[_blk] = _vptr_7[_blk];
                        uint32_t* _vpairs_7 = reinterpret_cast<uint32_t*>(&_vld_7[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            asm volatile(
                                "{\n\t"
                                "shl.b32 %0, %2, 16;\n\t"
                                "and.b32 %1, %2, 0xffff0000;\n\t"
                                "}\n"
                                : "=f"((&ops[8 + _blk * 8 + _pair * 2])[0]), "=f"((&ops[8 + _blk * 8 + _pair * 2])[1])
                                : "r"(_vpairs_7[_pair]));
                        }
                    }
                }
                {
                    const uint4* _vptr_8 = reinterpret_cast<const uint4*>(residual + (((grow_base + 8 < M) ? grow_base + 8 : 0) * 5376 + col) + 0);
                    uint4 _vld_8[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_8[_blk] = _vptr_8[_blk];
                        uint32_t* _vpairs_8 = reinterpret_cast<uint32_t*>(&_vld_8[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            asm volatile(
                                "{\n\t"
                                "shl.b32 %0, %2, 16;\n\t"
                                "and.b32 %1, %2, 0xffff0000;\n\t"
                                "}\n"
                                : "=f"((&ops[16 + _blk * 8 + _pair * 2])[0]), "=f"((&ops[16 + _blk * 8 + _pair * 2])[1])
                                : "r"(_vpairs_8[_pair]));
                        }
                    }
                }
                {
                    const uint4* _vptr_9 = reinterpret_cast<const uint4*>(residual + (((grow_base + 8 < M) ? grow_base + 8 : 0) * 5376 + col + 8) + 0);
                    uint4 _vld_9[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_9[_blk] = _vptr_9[_blk];
                        uint32_t* _vpairs_9 = reinterpret_cast<uint32_t*>(&_vld_9[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            asm volatile(
                                "{\n\t"
                                "shl.b32 %0, %2, 16;\n\t"
                                "and.b32 %1, %2, 0xffff0000;\n\t"
                                "}\n"
                                : "=f"((&ops[24 + _blk * 8 + _pair * 2])[0]), "=f"((&ops[24 + _blk * 8 + _pair * 2])[1])
                                : "r"(_vpairs_9[_pair]));
                        }
                    }
                }
                float scaled_0[16];
                scaled_0[0] = accum[2] * sa[1] * sw[0];
                scaled_0[1] = accum[3] * sa[1] * sw[1];
                scaled_0[2] = accum[6] * sa[1] * sw[2];
                scaled_0[3] = accum[7] * sa[1] * sw[3];
                scaled_0[4] = accum[10] * sa[1] * sw[4];
                scaled_0[5] = accum[11] * sa[1] * sw[5];
                scaled_0[6] = accum[14] * sa[1] * sw[6];
                scaled_0[7] = accum[15] * sa[1] * sw[7];
                scaled_0[8] = accum[18] * sa[1] * sw[8];
                scaled_0[9] = accum[19] * sa[1] * sw[9];
                scaled_0[10] = accum[22] * sa[1] * sw[10];
                scaled_0[11] = accum[23] * sa[1] * sw[11];
                scaled_0[12] = accum[26] * sa[1] * sw[12];
                scaled_0[13] = accum[27] * sa[1] * sw[13];
                scaled_0[14] = accum[30] * sa[1] * sw[14];
                scaled_0[15] = accum[31] * sa[1] * sw[15];
                uint32_t scaled_0_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_0[_lp*2 + 0], scaled_0[_lp*2+1 + 0]));
                    scaled_0_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t _stmatrix_addr_10 = static_cast<uint32_t>(staging_addr + (unsigned int)(srow_st * 256) + (unsigned int)((warp_n * 64 + smat * 8) * 2 ^ (srow_st & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_10), "r"(*reinterpret_cast<const uint32_t*>(&scaled_0_bf16[0])), "r"(*reinterpret_cast<const uint32_t*>(&scaled_0_bf16[1])), "r"(*reinterpret_cast<const uint32_t*>(&scaled_0_bf16[2])), "r"(*reinterpret_cast<const uint32_t*>(&scaled_0_bf16[3]))
                    : "memory");
                uint32_t _stmatrix_addr_11 = static_cast<uint32_t>(staging_addr + (unsigned int)(srow_st * 256) + (unsigned int)((warp_n * 64 + (4 + smat) * 8) * 2 ^ (srow_st & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_11), "r"(*reinterpret_cast<const uint32_t*>(&scaled_0_bf16[4])), "r"(*reinterpret_cast<const uint32_t*>(&scaled_0_bf16[5])), "r"(*reinterpret_cast<const uint32_t*>(&scaled_0_bf16[6])), "r"(*reinterpret_cast<const uint32_t*>(&scaled_0_bf16[7]))
                    : "memory");
                asm volatile("barrier.sync 8, 256;" ::: "memory");
                int grow_1 = grow_base + 8;
                if (grow_1 < M) {
                    unsigned int o_words_1[8];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&o_words_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&o_words_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&o_words_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&o_words_1[(0) + 3]))
                        : "r"(staging_addr + (unsigned int)(srow * 256) + (unsigned int)(c * 32 ^ (srow & 7) << 4)));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&o_words_1[4])), "=r"(*reinterpret_cast<uint32_t*>(&o_words_1[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&o_words_1[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&o_words_1[(4) + 3]))
                        : "r"(staging_addr + (unsigned int)(srow * 256) + (unsigned int)(c * 32 + 16 ^ (srow & 7) << 4)));
                    float o_words_f32_1[16];
                    #pragma unroll
                    for (int _pair = 0; _pair < 8; _pair++) {
                        asm volatile(
                            "{\n\t"
                            "shl.b32 %0, %2, 16;\n\t"
                            "and.b32 %1, %2, 0xffff0000;\n\t"
                            "}\n"
                            : "=f"((&o_words_f32_1[_pair * 2])[0]), "=f"((&o_words_f32_1[_pair * 2])[1])
                            : "r"(o_words_1[_pair]));
                    }
                    int valid_1 = meta[1];
                    float p_raw_1[16];
                    for (int e_2 = 0; e_2 < 8; e_2++) {
                        float gv0_1 = ops[e_2];
                        float gv1_1 = ops[8 + e_2];
                        float gs0_1 = ((valid_1 == 1) ? gv0_1 : 0.0f);
                        float gs1_1 = ((valid_1 == 1) ? gv1_1 : 0.0f);
                        p_raw_1[e_2] = gs0_1 * o_words_f32_1[e_2];
                        p_raw_1[8 + e_2] = gs1_1 * o_words_f32_1[8 + e_2];
                    }
                    uint32_t p_raw_bf16_1[8];
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(p_raw_1[_lp*2 + 0], p_raw_1[_lp*2+1 + 0]));
                        p_raw_bf16_1[_lp] = *(uint32_t*)&_bf2;
                    }
                    float p_raw_bf16_f32_1[16];
                    #pragma unroll
                    for (int _pair = 0; _pair < 8; _pair++) {
                        asm volatile(
                            "{\n\t"
                            "shl.b32 %0, %2, 16;\n\t"
                            "and.b32 %1, %2, 0xffff0000;\n\t"
                            "}\n"
                            : "=f"((&p_raw_bf16_f32_1[_pair * 2])[0]), "=f"((&p_raw_bf16_f32_1[_pair * 2])[1])
                            : "r"(p_raw_bf16_1[_pair]));
                    }
                    float out_raw_1[16];
                    for (int e_3 = 0; e_3 < 8; e_3++) {
                        float rv0_1 = ops[16 + e_3];
                        float rv1_1 = ops[24 + e_3];
                        out_raw_1[e_3] = rv0_1 + p_raw_bf16_f32_1[e_3];
                        out_raw_1[8 + e_3] = rv1_1 + p_raw_bf16_f32_1[8 + e_3];
                    }
                    uint32_t out_raw_bf16_1[8];
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_raw_1[_lp*2 + 0], out_raw_1[_lp*2+1 + 0]));
                        out_raw_bf16_1[_lp] = *(uint32_t*)&_bf2;
                    }
                    if (grow_1 < M) {
                        reinterpret_cast<int4*>(out + ((grow_1 * 5376 + col) / 2))[0] = reinterpret_cast<int4*>(out_raw_bf16_1 + 0)[0];
                        reinterpret_cast<int4*>(out + ((grow_1 * 5376 + col) / 2 + 4))[0] = reinterpret_cast<int4*>(out_raw_bf16_1 + 4)[0];
                    }
                }
                asm volatile("barrier.sync 8, 256;" ::: "memory");
                meta[0] = gate_index[((grow_base + 16 < M) ? grow_base + 16 : 0)];
                meta[1] = ((meta[0] >= 0 && meta[0] < 9) ? 1 : 0);
                {
                    const uint4* _vptr_12 = reinterpret_cast<const uint4*>(gate + (((meta[1] == 1) ? meta[0] : 0) * 5376 + col) + 0);
                    uint4 _vld_12[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_12[_blk] = _vptr_12[_blk];
                        uint32_t* _vpairs_12 = reinterpret_cast<uint32_t*>(&_vld_12[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            asm volatile(
                                "{\n\t"
                                "shl.b32 %0, %2, 16;\n\t"
                                "and.b32 %1, %2, 0xffff0000;\n\t"
                                "}\n"
                                : "=f"((&ops[0 + _blk * 8 + _pair * 2])[0]), "=f"((&ops[0 + _blk * 8 + _pair * 2])[1])
                                : "r"(_vpairs_12[_pair]));
                        }
                    }
                }
                {
                    const uint4* _vptr_13 = reinterpret_cast<const uint4*>(gate + (((meta[1] == 1) ? meta[0] : 0) * 5376 + col + 8) + 0);
                    uint4 _vld_13[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_13[_blk] = _vptr_13[_blk];
                        uint32_t* _vpairs_13 = reinterpret_cast<uint32_t*>(&_vld_13[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            asm volatile(
                                "{\n\t"
                                "shl.b32 %0, %2, 16;\n\t"
                                "and.b32 %1, %2, 0xffff0000;\n\t"
                                "}\n"
                                : "=f"((&ops[8 + _blk * 8 + _pair * 2])[0]), "=f"((&ops[8 + _blk * 8 + _pair * 2])[1])
                                : "r"(_vpairs_13[_pair]));
                        }
                    }
                }
                {
                    const uint4* _vptr_14 = reinterpret_cast<const uint4*>(residual + (((grow_base + 16 < M) ? grow_base + 16 : 0) * 5376 + col) + 0);
                    uint4 _vld_14[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_14[_blk] = _vptr_14[_blk];
                        uint32_t* _vpairs_14 = reinterpret_cast<uint32_t*>(&_vld_14[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            asm volatile(
                                "{\n\t"
                                "shl.b32 %0, %2, 16;\n\t"
                                "and.b32 %1, %2, 0xffff0000;\n\t"
                                "}\n"
                                : "=f"((&ops[16 + _blk * 8 + _pair * 2])[0]), "=f"((&ops[16 + _blk * 8 + _pair * 2])[1])
                                : "r"(_vpairs_14[_pair]));
                        }
                    }
                }
                {
                    const uint4* _vptr_15 = reinterpret_cast<const uint4*>(residual + (((grow_base + 16 < M) ? grow_base + 16 : 0) * 5376 + col + 8) + 0);
                    uint4 _vld_15[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_15[_blk] = _vptr_15[_blk];
                        uint32_t* _vpairs_15 = reinterpret_cast<uint32_t*>(&_vld_15[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            asm volatile(
                                "{\n\t"
                                "shl.b32 %0, %2, 16;\n\t"
                                "and.b32 %1, %2, 0xffff0000;\n\t"
                                "}\n"
                                : "=f"((&ops[24 + _blk * 8 + _pair * 2])[0]), "=f"((&ops[24 + _blk * 8 + _pair * 2])[1])
                                : "r"(_vpairs_15[_pair]));
                        }
                    }
                }
                float scaled_2[16];
                scaled_2[0] = accum[32] * sa[2] * sw[0];
                scaled_2[1] = accum[33] * sa[2] * sw[1];
                scaled_2[2] = accum[36] * sa[2] * sw[2];
                scaled_2[3] = accum[37] * sa[2] * sw[3];
                scaled_2[4] = accum[40] * sa[2] * sw[4];
                scaled_2[5] = accum[41] * sa[2] * sw[5];
                scaled_2[6] = accum[44] * sa[2] * sw[6];
                scaled_2[7] = accum[45] * sa[2] * sw[7];
                scaled_2[8] = accum[48] * sa[2] * sw[8];
                scaled_2[9] = accum[49] * sa[2] * sw[9];
                scaled_2[10] = accum[52] * sa[2] * sw[10];
                scaled_2[11] = accum[53] * sa[2] * sw[11];
                scaled_2[12] = accum[56] * sa[2] * sw[12];
                scaled_2[13] = accum[57] * sa[2] * sw[13];
                scaled_2[14] = accum[60] * sa[2] * sw[14];
                scaled_2[15] = accum[61] * sa[2] * sw[15];
                uint32_t scaled_2_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_2[_lp*2 + 0], scaled_2[_lp*2+1 + 0]));
                    scaled_2_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t _stmatrix_addr_16 = static_cast<uint32_t>(staging_addr + (unsigned int)(srow_st * 256) + (unsigned int)((warp_n * 64 + smat * 8) * 2 ^ (srow_st & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_16), "r"(*reinterpret_cast<const uint32_t*>(&scaled_2_bf16[0])), "r"(*reinterpret_cast<const uint32_t*>(&scaled_2_bf16[1])), "r"(*reinterpret_cast<const uint32_t*>(&scaled_2_bf16[2])), "r"(*reinterpret_cast<const uint32_t*>(&scaled_2_bf16[3]))
                    : "memory");
                uint32_t _stmatrix_addr_17 = static_cast<uint32_t>(staging_addr + (unsigned int)(srow_st * 256) + (unsigned int)((warp_n * 64 + (4 + smat) * 8) * 2 ^ (srow_st & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_17), "r"(*reinterpret_cast<const uint32_t*>(&scaled_2_bf16[4])), "r"(*reinterpret_cast<const uint32_t*>(&scaled_2_bf16[5])), "r"(*reinterpret_cast<const uint32_t*>(&scaled_2_bf16[6])), "r"(*reinterpret_cast<const uint32_t*>(&scaled_2_bf16[7]))
                    : "memory");
                asm volatile("barrier.sync 8, 256;" ::: "memory");
                int grow_3 = grow_base + 16;
                if (grow_3 < M) {
                    unsigned int o_words_2[8];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&o_words_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&o_words_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&o_words_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&o_words_2[(0) + 3]))
                        : "r"(staging_addr + (unsigned int)(srow * 256) + (unsigned int)(c * 32 ^ (srow & 7) << 4)));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&o_words_2[4])), "=r"(*reinterpret_cast<uint32_t*>(&o_words_2[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&o_words_2[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&o_words_2[(4) + 3]))
                        : "r"(staging_addr + (unsigned int)(srow * 256) + (unsigned int)(c * 32 + 16 ^ (srow & 7) << 4)));
                    float o_words_f32_2[16];
                    #pragma unroll
                    for (int _pair = 0; _pair < 8; _pair++) {
                        asm volatile(
                            "{\n\t"
                            "shl.b32 %0, %2, 16;\n\t"
                            "and.b32 %1, %2, 0xffff0000;\n\t"
                            "}\n"
                            : "=f"((&o_words_f32_2[_pair * 2])[0]), "=f"((&o_words_f32_2[_pair * 2])[1])
                            : "r"(o_words_2[_pair]));
                    }
                    int valid_2 = meta[1];
                    float p_raw_2[16];
                    for (int e_4 = 0; e_4 < 8; e_4++) {
                        float gv0_2 = ops[e_4];
                        float gv1_2 = ops[8 + e_4];
                        float gs0_2 = ((valid_2 == 1) ? gv0_2 : 0.0f);
                        float gs1_2 = ((valid_2 == 1) ? gv1_2 : 0.0f);
                        p_raw_2[e_4] = gs0_2 * o_words_f32_2[e_4];
                        p_raw_2[8 + e_4] = gs1_2 * o_words_f32_2[8 + e_4];
                    }
                    uint32_t p_raw_bf16_2[8];
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(p_raw_2[_lp*2 + 0], p_raw_2[_lp*2+1 + 0]));
                        p_raw_bf16_2[_lp] = *(uint32_t*)&_bf2;
                    }
                    float p_raw_bf16_f32_2[16];
                    #pragma unroll
                    for (int _pair = 0; _pair < 8; _pair++) {
                        asm volatile(
                            "{\n\t"
                            "shl.b32 %0, %2, 16;\n\t"
                            "and.b32 %1, %2, 0xffff0000;\n\t"
                            "}\n"
                            : "=f"((&p_raw_bf16_f32_2[_pair * 2])[0]), "=f"((&p_raw_bf16_f32_2[_pair * 2])[1])
                            : "r"(p_raw_bf16_2[_pair]));
                    }
                    float out_raw_2[16];
                    for (int e_5 = 0; e_5 < 8; e_5++) {
                        float rv0_2 = ops[16 + e_5];
                        float rv1_2 = ops[24 + e_5];
                        out_raw_2[e_5] = rv0_2 + p_raw_bf16_f32_2[e_5];
                        out_raw_2[8 + e_5] = rv1_2 + p_raw_bf16_f32_2[8 + e_5];
                    }
                    uint32_t out_raw_bf16_2[8];
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_raw_2[_lp*2 + 0], out_raw_2[_lp*2+1 + 0]));
                        out_raw_bf16_2[_lp] = *(uint32_t*)&_bf2;
                    }
                    if (grow_3 < M) {
                        reinterpret_cast<int4*>(out + ((grow_3 * 5376 + col) / 2))[0] = reinterpret_cast<int4*>(out_raw_bf16_2 + 0)[0];
                        reinterpret_cast<int4*>(out + ((grow_3 * 5376 + col) / 2 + 4))[0] = reinterpret_cast<int4*>(out_raw_bf16_2 + 4)[0];
                    }
                }
                asm volatile("barrier.sync 8, 256;" ::: "memory");
                meta[0] = gate_index[((grow_base + 24 < M) ? grow_base + 24 : 0)];
                meta[1] = ((meta[0] >= 0 && meta[0] < 9) ? 1 : 0);
                {
                    const uint4* _vptr_18 = reinterpret_cast<const uint4*>(gate + (((meta[1] == 1) ? meta[0] : 0) * 5376 + col) + 0);
                    uint4 _vld_18[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_18[_blk] = _vptr_18[_blk];
                        uint32_t* _vpairs_18 = reinterpret_cast<uint32_t*>(&_vld_18[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            asm volatile(
                                "{\n\t"
                                "shl.b32 %0, %2, 16;\n\t"
                                "and.b32 %1, %2, 0xffff0000;\n\t"
                                "}\n"
                                : "=f"((&ops[0 + _blk * 8 + _pair * 2])[0]), "=f"((&ops[0 + _blk * 8 + _pair * 2])[1])
                                : "r"(_vpairs_18[_pair]));
                        }
                    }
                }
                {
                    const uint4* _vptr_19 = reinterpret_cast<const uint4*>(gate + (((meta[1] == 1) ? meta[0] : 0) * 5376 + col + 8) + 0);
                    uint4 _vld_19[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_19[_blk] = _vptr_19[_blk];
                        uint32_t* _vpairs_19 = reinterpret_cast<uint32_t*>(&_vld_19[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            asm volatile(
                                "{\n\t"
                                "shl.b32 %0, %2, 16;\n\t"
                                "and.b32 %1, %2, 0xffff0000;\n\t"
                                "}\n"
                                : "=f"((&ops[8 + _blk * 8 + _pair * 2])[0]), "=f"((&ops[8 + _blk * 8 + _pair * 2])[1])
                                : "r"(_vpairs_19[_pair]));
                        }
                    }
                }
                {
                    const uint4* _vptr_20 = reinterpret_cast<const uint4*>(residual + (((grow_base + 24 < M) ? grow_base + 24 : 0) * 5376 + col) + 0);
                    uint4 _vld_20[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_20[_blk] = _vptr_20[_blk];
                        uint32_t* _vpairs_20 = reinterpret_cast<uint32_t*>(&_vld_20[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            asm volatile(
                                "{\n\t"
                                "shl.b32 %0, %2, 16;\n\t"
                                "and.b32 %1, %2, 0xffff0000;\n\t"
                                "}\n"
                                : "=f"((&ops[16 + _blk * 8 + _pair * 2])[0]), "=f"((&ops[16 + _blk * 8 + _pair * 2])[1])
                                : "r"(_vpairs_20[_pair]));
                        }
                    }
                }
                {
                    const uint4* _vptr_21 = reinterpret_cast<const uint4*>(residual + (((grow_base + 24 < M) ? grow_base + 24 : 0) * 5376 + col + 8) + 0);
                    uint4 _vld_21[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_21[_blk] = _vptr_21[_blk];
                        uint32_t* _vpairs_21 = reinterpret_cast<uint32_t*>(&_vld_21[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            asm volatile(
                                "{\n\t"
                                "shl.b32 %0, %2, 16;\n\t"
                                "and.b32 %1, %2, 0xffff0000;\n\t"
                                "}\n"
                                : "=f"((&ops[24 + _blk * 8 + _pair * 2])[0]), "=f"((&ops[24 + _blk * 8 + _pair * 2])[1])
                                : "r"(_vpairs_21[_pair]));
                        }
                    }
                }
                float scaled_4[16];
                scaled_4[0] = accum[34] * sa[3] * sw[0];
                scaled_4[1] = accum[35] * sa[3] * sw[1];
                scaled_4[2] = accum[38] * sa[3] * sw[2];
                scaled_4[3] = accum[39] * sa[3] * sw[3];
                scaled_4[4] = accum[42] * sa[3] * sw[4];
                scaled_4[5] = accum[43] * sa[3] * sw[5];
                scaled_4[6] = accum[46] * sa[3] * sw[6];
                scaled_4[7] = accum[47] * sa[3] * sw[7];
                scaled_4[8] = accum[50] * sa[3] * sw[8];
                scaled_4[9] = accum[51] * sa[3] * sw[9];
                scaled_4[10] = accum[54] * sa[3] * sw[10];
                scaled_4[11] = accum[55] * sa[3] * sw[11];
                scaled_4[12] = accum[58] * sa[3] * sw[12];
                scaled_4[13] = accum[59] * sa[3] * sw[13];
                scaled_4[14] = accum[62] * sa[3] * sw[14];
                scaled_4[15] = accum[63] * sa[3] * sw[15];
                uint32_t scaled_4_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_4[_lp*2 + 0], scaled_4[_lp*2+1 + 0]));
                    scaled_4_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t _stmatrix_addr_22 = static_cast<uint32_t>(staging_addr + (unsigned int)(srow_st * 256) + (unsigned int)((warp_n * 64 + smat * 8) * 2 ^ (srow_st & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_22), "r"(*reinterpret_cast<const uint32_t*>(&scaled_4_bf16[0])), "r"(*reinterpret_cast<const uint32_t*>(&scaled_4_bf16[1])), "r"(*reinterpret_cast<const uint32_t*>(&scaled_4_bf16[2])), "r"(*reinterpret_cast<const uint32_t*>(&scaled_4_bf16[3]))
                    : "memory");
                uint32_t _stmatrix_addr_23 = static_cast<uint32_t>(staging_addr + (unsigned int)(srow_st * 256) + (unsigned int)((warp_n * 64 + (4 + smat) * 8) * 2 ^ (srow_st & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_23), "r"(*reinterpret_cast<const uint32_t*>(&scaled_4_bf16[4])), "r"(*reinterpret_cast<const uint32_t*>(&scaled_4_bf16[5])), "r"(*reinterpret_cast<const uint32_t*>(&scaled_4_bf16[6])), "r"(*reinterpret_cast<const uint32_t*>(&scaled_4_bf16[7]))
                    : "memory");
                asm volatile("barrier.sync 8, 256;" ::: "memory");
                int grow_5 = grow_base + 24;
                if (grow_5 < M) {
                    unsigned int o_words_3[8];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&o_words_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&o_words_3[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&o_words_3[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&o_words_3[(0) + 3]))
                        : "r"(staging_addr + (unsigned int)(srow * 256) + (unsigned int)(c * 32 ^ (srow & 7) << 4)));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&o_words_3[4])), "=r"(*reinterpret_cast<uint32_t*>(&o_words_3[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&o_words_3[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&o_words_3[(4) + 3]))
                        : "r"(staging_addr + (unsigned int)(srow * 256) + (unsigned int)(c * 32 + 16 ^ (srow & 7) << 4)));
                    float o_words_f32_3[16];
                    #pragma unroll
                    for (int _pair = 0; _pair < 8; _pair++) {
                        asm volatile(
                            "{\n\t"
                            "shl.b32 %0, %2, 16;\n\t"
                            "and.b32 %1, %2, 0xffff0000;\n\t"
                            "}\n"
                            : "=f"((&o_words_f32_3[_pair * 2])[0]), "=f"((&o_words_f32_3[_pair * 2])[1])
                            : "r"(o_words_3[_pair]));
                    }
                    int valid_3 = meta[1];
                    float p_raw_3[16];
                    for (int e_6 = 0; e_6 < 8; e_6++) {
                        float gv0_3 = ops[e_6];
                        float gv1_3 = ops[8 + e_6];
                        float gs0_3 = ((valid_3 == 1) ? gv0_3 : 0.0f);
                        float gs1_3 = ((valid_3 == 1) ? gv1_3 : 0.0f);
                        p_raw_3[e_6] = gs0_3 * o_words_f32_3[e_6];
                        p_raw_3[8 + e_6] = gs1_3 * o_words_f32_3[8 + e_6];
                    }
                    uint32_t p_raw_bf16_3[8];
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(p_raw_3[_lp*2 + 0], p_raw_3[_lp*2+1 + 0]));
                        p_raw_bf16_3[_lp] = *(uint32_t*)&_bf2;
                    }
                    float p_raw_bf16_f32_3[16];
                    #pragma unroll
                    for (int _pair = 0; _pair < 8; _pair++) {
                        asm volatile(
                            "{\n\t"
                            "shl.b32 %0, %2, 16;\n\t"
                            "and.b32 %1, %2, 0xffff0000;\n\t"
                            "}\n"
                            : "=f"((&p_raw_bf16_f32_3[_pair * 2])[0]), "=f"((&p_raw_bf16_f32_3[_pair * 2])[1])
                            : "r"(p_raw_bf16_3[_pair]));
                    }
                    float out_raw_3[16];
                    for (int e_7 = 0; e_7 < 8; e_7++) {
                        float rv0_3 = ops[16 + e_7];
                        float rv1_3 = ops[24 + e_7];
                        out_raw_3[e_7] = rv0_3 + p_raw_bf16_f32_3[e_7];
                        out_raw_3[8 + e_7] = rv1_3 + p_raw_bf16_f32_3[8 + e_7];
                    }
                    uint32_t out_raw_bf16_3[8];
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_raw_3[_lp*2 + 0], out_raw_3[_lp*2+1 + 0]));
                        out_raw_bf16_3[_lp] = *(uint32_t*)&_bf2;
                    }
                    if (grow_5 < M) {
                        reinterpret_cast<int4*>(out + ((grow_5 * 5376 + col) / 2))[0] = reinterpret_cast<int4*>(out_raw_bf16_3 + 0)[0];
                        reinterpret_cast<int4*>(out + ((grow_5 * 5376 + col) / 2 + 4))[0] = reinterpret_cast<int4*>(out_raw_bf16_3 + 4)[0];
                    }
                }
                asm volatile("barrier.sync 8, 256;" ::: "memory");
                meta[0] = gate_index[((grow_base + 32 < M) ? grow_base + 32 : 0)];
                meta[1] = ((meta[0] >= 0 && meta[0] < 9) ? 1 : 0);
                {
                    const uint4* _vptr_24 = reinterpret_cast<const uint4*>(gate + (((meta[1] == 1) ? meta[0] : 0) * 5376 + col) + 0);
                    uint4 _vld_24[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_24[_blk] = _vptr_24[_blk];
                        uint32_t* _vpairs_24 = reinterpret_cast<uint32_t*>(&_vld_24[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            asm volatile(
                                "{\n\t"
                                "shl.b32 %0, %2, 16;\n\t"
                                "and.b32 %1, %2, 0xffff0000;\n\t"
                                "}\n"
                                : "=f"((&ops[0 + _blk * 8 + _pair * 2])[0]), "=f"((&ops[0 + _blk * 8 + _pair * 2])[1])
                                : "r"(_vpairs_24[_pair]));
                        }
                    }
                }
                {
                    const uint4* _vptr_25 = reinterpret_cast<const uint4*>(gate + (((meta[1] == 1) ? meta[0] : 0) * 5376 + col + 8) + 0);
                    uint4 _vld_25[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_25[_blk] = _vptr_25[_blk];
                        uint32_t* _vpairs_25 = reinterpret_cast<uint32_t*>(&_vld_25[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            asm volatile(
                                "{\n\t"
                                "shl.b32 %0, %2, 16;\n\t"
                                "and.b32 %1, %2, 0xffff0000;\n\t"
                                "}\n"
                                : "=f"((&ops[8 + _blk * 8 + _pair * 2])[0]), "=f"((&ops[8 + _blk * 8 + _pair * 2])[1])
                                : "r"(_vpairs_25[_pair]));
                        }
                    }
                }
                {
                    const uint4* _vptr_26 = reinterpret_cast<const uint4*>(residual + (((grow_base + 32 < M) ? grow_base + 32 : 0) * 5376 + col) + 0);
                    uint4 _vld_26[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_26[_blk] = _vptr_26[_blk];
                        uint32_t* _vpairs_26 = reinterpret_cast<uint32_t*>(&_vld_26[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            asm volatile(
                                "{\n\t"
                                "shl.b32 %0, %2, 16;\n\t"
                                "and.b32 %1, %2, 0xffff0000;\n\t"
                                "}\n"
                                : "=f"((&ops[16 + _blk * 8 + _pair * 2])[0]), "=f"((&ops[16 + _blk * 8 + _pair * 2])[1])
                                : "r"(_vpairs_26[_pair]));
                        }
                    }
                }
                {
                    const uint4* _vptr_27 = reinterpret_cast<const uint4*>(residual + (((grow_base + 32 < M) ? grow_base + 32 : 0) * 5376 + col + 8) + 0);
                    uint4 _vld_27[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_27[_blk] = _vptr_27[_blk];
                        uint32_t* _vpairs_27 = reinterpret_cast<uint32_t*>(&_vld_27[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            asm volatile(
                                "{\n\t"
                                "shl.b32 %0, %2, 16;\n\t"
                                "and.b32 %1, %2, 0xffff0000;\n\t"
                                "}\n"
                                : "=f"((&ops[24 + _blk * 8 + _pair * 2])[0]), "=f"((&ops[24 + _blk * 8 + _pair * 2])[1])
                                : "r"(_vpairs_27[_pair]));
                        }
                    }
                }
                float scaled_6[16];
                scaled_6[0] = accum[64] * sa[4] * sw[0];
                scaled_6[1] = accum[65] * sa[4] * sw[1];
                scaled_6[2] = accum[68] * sa[4] * sw[2];
                scaled_6[3] = accum[69] * sa[4] * sw[3];
                scaled_6[4] = accum[72] * sa[4] * sw[4];
                scaled_6[5] = accum[73] * sa[4] * sw[5];
                scaled_6[6] = accum[76] * sa[4] * sw[6];
                scaled_6[7] = accum[77] * sa[4] * sw[7];
                scaled_6[8] = accum[80] * sa[4] * sw[8];
                scaled_6[9] = accum[81] * sa[4] * sw[9];
                scaled_6[10] = accum[84] * sa[4] * sw[10];
                scaled_6[11] = accum[85] * sa[4] * sw[11];
                scaled_6[12] = accum[88] * sa[4] * sw[12];
                scaled_6[13] = accum[89] * sa[4] * sw[13];
                scaled_6[14] = accum[92] * sa[4] * sw[14];
                scaled_6[15] = accum[93] * sa[4] * sw[15];
                uint32_t scaled_6_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_6[_lp*2 + 0], scaled_6[_lp*2+1 + 0]));
                    scaled_6_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t _stmatrix_addr_28 = static_cast<uint32_t>(staging_addr + (unsigned int)(srow_st * 256) + (unsigned int)((warp_n * 64 + smat * 8) * 2 ^ (srow_st & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_28), "r"(*reinterpret_cast<const uint32_t*>(&scaled_6_bf16[0])), "r"(*reinterpret_cast<const uint32_t*>(&scaled_6_bf16[1])), "r"(*reinterpret_cast<const uint32_t*>(&scaled_6_bf16[2])), "r"(*reinterpret_cast<const uint32_t*>(&scaled_6_bf16[3]))
                    : "memory");
                uint32_t _stmatrix_addr_29 = static_cast<uint32_t>(staging_addr + (unsigned int)(srow_st * 256) + (unsigned int)((warp_n * 64 + (4 + smat) * 8) * 2 ^ (srow_st & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_29), "r"(*reinterpret_cast<const uint32_t*>(&scaled_6_bf16[4])), "r"(*reinterpret_cast<const uint32_t*>(&scaled_6_bf16[5])), "r"(*reinterpret_cast<const uint32_t*>(&scaled_6_bf16[6])), "r"(*reinterpret_cast<const uint32_t*>(&scaled_6_bf16[7]))
                    : "memory");
                asm volatile("barrier.sync 8, 256;" ::: "memory");
                int grow_7 = grow_base + 32;
                if (grow_7 < M) {
                    unsigned int o_words_4[8];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&o_words_4[0])), "=r"(*reinterpret_cast<uint32_t*>(&o_words_4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&o_words_4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&o_words_4[(0) + 3]))
                        : "r"(staging_addr + (unsigned int)(srow * 256) + (unsigned int)(c * 32 ^ (srow & 7) << 4)));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&o_words_4[4])), "=r"(*reinterpret_cast<uint32_t*>(&o_words_4[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&o_words_4[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&o_words_4[(4) + 3]))
                        : "r"(staging_addr + (unsigned int)(srow * 256) + (unsigned int)(c * 32 + 16 ^ (srow & 7) << 4)));
                    float o_words_f32_4[16];
                    #pragma unroll
                    for (int _pair = 0; _pair < 8; _pair++) {
                        asm volatile(
                            "{\n\t"
                            "shl.b32 %0, %2, 16;\n\t"
                            "and.b32 %1, %2, 0xffff0000;\n\t"
                            "}\n"
                            : "=f"((&o_words_f32_4[_pair * 2])[0]), "=f"((&o_words_f32_4[_pair * 2])[1])
                            : "r"(o_words_4[_pair]));
                    }
                    int valid_4 = meta[1];
                    float p_raw_4[16];
                    for (int e_8 = 0; e_8 < 8; e_8++) {
                        float gv0_4 = ops[e_8];
                        float gv1_4 = ops[8 + e_8];
                        float gs0_4 = ((valid_4 == 1) ? gv0_4 : 0.0f);
                        float gs1_4 = ((valid_4 == 1) ? gv1_4 : 0.0f);
                        p_raw_4[e_8] = gs0_4 * o_words_f32_4[e_8];
                        p_raw_4[8 + e_8] = gs1_4 * o_words_f32_4[8 + e_8];
                    }
                    uint32_t p_raw_bf16_4[8];
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(p_raw_4[_lp*2 + 0], p_raw_4[_lp*2+1 + 0]));
                        p_raw_bf16_4[_lp] = *(uint32_t*)&_bf2;
                    }
                    float p_raw_bf16_f32_4[16];
                    #pragma unroll
                    for (int _pair = 0; _pair < 8; _pair++) {
                        asm volatile(
                            "{\n\t"
                            "shl.b32 %0, %2, 16;\n\t"
                            "and.b32 %1, %2, 0xffff0000;\n\t"
                            "}\n"
                            : "=f"((&p_raw_bf16_f32_4[_pair * 2])[0]), "=f"((&p_raw_bf16_f32_4[_pair * 2])[1])
                            : "r"(p_raw_bf16_4[_pair]));
                    }
                    float out_raw_4[16];
                    for (int e_9 = 0; e_9 < 8; e_9++) {
                        float rv0_4 = ops[16 + e_9];
                        float rv1_4 = ops[24 + e_9];
                        out_raw_4[e_9] = rv0_4 + p_raw_bf16_f32_4[e_9];
                        out_raw_4[8 + e_9] = rv1_4 + p_raw_bf16_f32_4[8 + e_9];
                    }
                    uint32_t out_raw_bf16_4[8];
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_raw_4[_lp*2 + 0], out_raw_4[_lp*2+1 + 0]));
                        out_raw_bf16_4[_lp] = *(uint32_t*)&_bf2;
                    }
                    if (grow_7 < M) {
                        reinterpret_cast<int4*>(out + ((grow_7 * 5376 + col) / 2))[0] = reinterpret_cast<int4*>(out_raw_bf16_4 + 0)[0];
                        reinterpret_cast<int4*>(out + ((grow_7 * 5376 + col) / 2 + 4))[0] = reinterpret_cast<int4*>(out_raw_bf16_4 + 4)[0];
                    }
                }
                asm volatile("barrier.sync 8, 256;" ::: "memory");
                meta[0] = gate_index[((grow_base + 40 < M) ? grow_base + 40 : 0)];
                meta[1] = ((meta[0] >= 0 && meta[0] < 9) ? 1 : 0);
                {
                    const uint4* _vptr_30 = reinterpret_cast<const uint4*>(gate + (((meta[1] == 1) ? meta[0] : 0) * 5376 + col) + 0);
                    uint4 _vld_30[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_30[_blk] = _vptr_30[_blk];
                        uint32_t* _vpairs_30 = reinterpret_cast<uint32_t*>(&_vld_30[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            asm volatile(
                                "{\n\t"
                                "shl.b32 %0, %2, 16;\n\t"
                                "and.b32 %1, %2, 0xffff0000;\n\t"
                                "}\n"
                                : "=f"((&ops[0 + _blk * 8 + _pair * 2])[0]), "=f"((&ops[0 + _blk * 8 + _pair * 2])[1])
                                : "r"(_vpairs_30[_pair]));
                        }
                    }
                }
                {
                    const uint4* _vptr_31 = reinterpret_cast<const uint4*>(gate + (((meta[1] == 1) ? meta[0] : 0) * 5376 + col + 8) + 0);
                    uint4 _vld_31[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_31[_blk] = _vptr_31[_blk];
                        uint32_t* _vpairs_31 = reinterpret_cast<uint32_t*>(&_vld_31[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            asm volatile(
                                "{\n\t"
                                "shl.b32 %0, %2, 16;\n\t"
                                "and.b32 %1, %2, 0xffff0000;\n\t"
                                "}\n"
                                : "=f"((&ops[8 + _blk * 8 + _pair * 2])[0]), "=f"((&ops[8 + _blk * 8 + _pair * 2])[1])
                                : "r"(_vpairs_31[_pair]));
                        }
                    }
                }
                {
                    const uint4* _vptr_32 = reinterpret_cast<const uint4*>(residual + (((grow_base + 40 < M) ? grow_base + 40 : 0) * 5376 + col) + 0);
                    uint4 _vld_32[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_32[_blk] = _vptr_32[_blk];
                        uint32_t* _vpairs_32 = reinterpret_cast<uint32_t*>(&_vld_32[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            asm volatile(
                                "{\n\t"
                                "shl.b32 %0, %2, 16;\n\t"
                                "and.b32 %1, %2, 0xffff0000;\n\t"
                                "}\n"
                                : "=f"((&ops[16 + _blk * 8 + _pair * 2])[0]), "=f"((&ops[16 + _blk * 8 + _pair * 2])[1])
                                : "r"(_vpairs_32[_pair]));
                        }
                    }
                }
                {
                    const uint4* _vptr_33 = reinterpret_cast<const uint4*>(residual + (((grow_base + 40 < M) ? grow_base + 40 : 0) * 5376 + col + 8) + 0);
                    uint4 _vld_33[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_33[_blk] = _vptr_33[_blk];
                        uint32_t* _vpairs_33 = reinterpret_cast<uint32_t*>(&_vld_33[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            asm volatile(
                                "{\n\t"
                                "shl.b32 %0, %2, 16;\n\t"
                                "and.b32 %1, %2, 0xffff0000;\n\t"
                                "}\n"
                                : "=f"((&ops[24 + _blk * 8 + _pair * 2])[0]), "=f"((&ops[24 + _blk * 8 + _pair * 2])[1])
                                : "r"(_vpairs_33[_pair]));
                        }
                    }
                }
                float scaled_8[16];
                scaled_8[0] = accum[66] * sa[5] * sw[0];
                scaled_8[1] = accum[67] * sa[5] * sw[1];
                scaled_8[2] = accum[70] * sa[5] * sw[2];
                scaled_8[3] = accum[71] * sa[5] * sw[3];
                scaled_8[4] = accum[74] * sa[5] * sw[4];
                scaled_8[5] = accum[75] * sa[5] * sw[5];
                scaled_8[6] = accum[78] * sa[5] * sw[6];
                scaled_8[7] = accum[79] * sa[5] * sw[7];
                scaled_8[8] = accum[82] * sa[5] * sw[8];
                scaled_8[9] = accum[83] * sa[5] * sw[9];
                scaled_8[10] = accum[86] * sa[5] * sw[10];
                scaled_8[11] = accum[87] * sa[5] * sw[11];
                scaled_8[12] = accum[90] * sa[5] * sw[12];
                scaled_8[13] = accum[91] * sa[5] * sw[13];
                scaled_8[14] = accum[94] * sa[5] * sw[14];
                scaled_8[15] = accum[95] * sa[5] * sw[15];
                uint32_t scaled_8_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_8[_lp*2 + 0], scaled_8[_lp*2+1 + 0]));
                    scaled_8_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t _stmatrix_addr_34 = static_cast<uint32_t>(staging_addr + (unsigned int)(srow_st * 256) + (unsigned int)((warp_n * 64 + smat * 8) * 2 ^ (srow_st & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_34), "r"(*reinterpret_cast<const uint32_t*>(&scaled_8_bf16[0])), "r"(*reinterpret_cast<const uint32_t*>(&scaled_8_bf16[1])), "r"(*reinterpret_cast<const uint32_t*>(&scaled_8_bf16[2])), "r"(*reinterpret_cast<const uint32_t*>(&scaled_8_bf16[3]))
                    : "memory");
                uint32_t _stmatrix_addr_35 = static_cast<uint32_t>(staging_addr + (unsigned int)(srow_st * 256) + (unsigned int)((warp_n * 64 + (4 + smat) * 8) * 2 ^ (srow_st & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_35), "r"(*reinterpret_cast<const uint32_t*>(&scaled_8_bf16[4])), "r"(*reinterpret_cast<const uint32_t*>(&scaled_8_bf16[5])), "r"(*reinterpret_cast<const uint32_t*>(&scaled_8_bf16[6])), "r"(*reinterpret_cast<const uint32_t*>(&scaled_8_bf16[7]))
                    : "memory");
                asm volatile("barrier.sync 8, 256;" ::: "memory");
                int grow_9 = grow_base + 40;
                if (grow_9 < M) {
                    unsigned int o_words_5[8];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&o_words_5[0])), "=r"(*reinterpret_cast<uint32_t*>(&o_words_5[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&o_words_5[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&o_words_5[(0) + 3]))
                        : "r"(staging_addr + (unsigned int)(srow * 256) + (unsigned int)(c * 32 ^ (srow & 7) << 4)));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&o_words_5[4])), "=r"(*reinterpret_cast<uint32_t*>(&o_words_5[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&o_words_5[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&o_words_5[(4) + 3]))
                        : "r"(staging_addr + (unsigned int)(srow * 256) + (unsigned int)(c * 32 + 16 ^ (srow & 7) << 4)));
                    float o_words_f32_5[16];
                    #pragma unroll
                    for (int _pair = 0; _pair < 8; _pair++) {
                        asm volatile(
                            "{\n\t"
                            "shl.b32 %0, %2, 16;\n\t"
                            "and.b32 %1, %2, 0xffff0000;\n\t"
                            "}\n"
                            : "=f"((&o_words_f32_5[_pair * 2])[0]), "=f"((&o_words_f32_5[_pair * 2])[1])
                            : "r"(o_words_5[_pair]));
                    }
                    int valid_5 = meta[1];
                    float p_raw_5[16];
                    for (int e_10 = 0; e_10 < 8; e_10++) {
                        float gv0_5 = ops[e_10];
                        float gv1_5 = ops[8 + e_10];
                        float gs0_5 = ((valid_5 == 1) ? gv0_5 : 0.0f);
                        float gs1_5 = ((valid_5 == 1) ? gv1_5 : 0.0f);
                        p_raw_5[e_10] = gs0_5 * o_words_f32_5[e_10];
                        p_raw_5[8 + e_10] = gs1_5 * o_words_f32_5[8 + e_10];
                    }
                    uint32_t p_raw_bf16_5[8];
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(p_raw_5[_lp*2 + 0], p_raw_5[_lp*2+1 + 0]));
                        p_raw_bf16_5[_lp] = *(uint32_t*)&_bf2;
                    }
                    float p_raw_bf16_f32_5[16];
                    #pragma unroll
                    for (int _pair = 0; _pair < 8; _pair++) {
                        asm volatile(
                            "{\n\t"
                            "shl.b32 %0, %2, 16;\n\t"
                            "and.b32 %1, %2, 0xffff0000;\n\t"
                            "}\n"
                            : "=f"((&p_raw_bf16_f32_5[_pair * 2])[0]), "=f"((&p_raw_bf16_f32_5[_pair * 2])[1])
                            : "r"(p_raw_bf16_5[_pair]));
                    }
                    float out_raw_5[16];
                    for (int e_11 = 0; e_11 < 8; e_11++) {
                        float rv0_5 = ops[16 + e_11];
                        float rv1_5 = ops[24 + e_11];
                        out_raw_5[e_11] = rv0_5 + p_raw_bf16_f32_5[e_11];
                        out_raw_5[8 + e_11] = rv1_5 + p_raw_bf16_f32_5[8 + e_11];
                    }
                    uint32_t out_raw_bf16_5[8];
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_raw_5[_lp*2 + 0], out_raw_5[_lp*2+1 + 0]));
                        out_raw_bf16_5[_lp] = *(uint32_t*)&_bf2;
                    }
                    if (grow_9 < M) {
                        reinterpret_cast<int4*>(out + ((grow_9 * 5376 + col) / 2))[0] = reinterpret_cast<int4*>(out_raw_bf16_5 + 0)[0];
                        reinterpret_cast<int4*>(out + ((grow_9 * 5376 + col) / 2 + 4))[0] = reinterpret_cast<int4*>(out_raw_bf16_5 + 4)[0];
                    }
                }
                asm volatile("barrier.sync 8, 256;" ::: "memory");
                meta[0] = gate_index[((grow_base + 48 < M) ? grow_base + 48 : 0)];
                meta[1] = ((meta[0] >= 0 && meta[0] < 9) ? 1 : 0);
                {
                    const uint4* _vptr_36 = reinterpret_cast<const uint4*>(gate + (((meta[1] == 1) ? meta[0] : 0) * 5376 + col) + 0);
                    uint4 _vld_36[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_36[_blk] = _vptr_36[_blk];
                        uint32_t* _vpairs_36 = reinterpret_cast<uint32_t*>(&_vld_36[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            asm volatile(
                                "{\n\t"
                                "shl.b32 %0, %2, 16;\n\t"
                                "and.b32 %1, %2, 0xffff0000;\n\t"
                                "}\n"
                                : "=f"((&ops[0 + _blk * 8 + _pair * 2])[0]), "=f"((&ops[0 + _blk * 8 + _pair * 2])[1])
                                : "r"(_vpairs_36[_pair]));
                        }
                    }
                }
                {
                    const uint4* _vptr_37 = reinterpret_cast<const uint4*>(gate + (((meta[1] == 1) ? meta[0] : 0) * 5376 + col + 8) + 0);
                    uint4 _vld_37[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_37[_blk] = _vptr_37[_blk];
                        uint32_t* _vpairs_37 = reinterpret_cast<uint32_t*>(&_vld_37[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            asm volatile(
                                "{\n\t"
                                "shl.b32 %0, %2, 16;\n\t"
                                "and.b32 %1, %2, 0xffff0000;\n\t"
                                "}\n"
                                : "=f"((&ops[8 + _blk * 8 + _pair * 2])[0]), "=f"((&ops[8 + _blk * 8 + _pair * 2])[1])
                                : "r"(_vpairs_37[_pair]));
                        }
                    }
                }
                {
                    const uint4* _vptr_38 = reinterpret_cast<const uint4*>(residual + (((grow_base + 48 < M) ? grow_base + 48 : 0) * 5376 + col) + 0);
                    uint4 _vld_38[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_38[_blk] = _vptr_38[_blk];
                        uint32_t* _vpairs_38 = reinterpret_cast<uint32_t*>(&_vld_38[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            asm volatile(
                                "{\n\t"
                                "shl.b32 %0, %2, 16;\n\t"
                                "and.b32 %1, %2, 0xffff0000;\n\t"
                                "}\n"
                                : "=f"((&ops[16 + _blk * 8 + _pair * 2])[0]), "=f"((&ops[16 + _blk * 8 + _pair * 2])[1])
                                : "r"(_vpairs_38[_pair]));
                        }
                    }
                }
                {
                    const uint4* _vptr_39 = reinterpret_cast<const uint4*>(residual + (((grow_base + 48 < M) ? grow_base + 48 : 0) * 5376 + col + 8) + 0);
                    uint4 _vld_39[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_39[_blk] = _vptr_39[_blk];
                        uint32_t* _vpairs_39 = reinterpret_cast<uint32_t*>(&_vld_39[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            asm volatile(
                                "{\n\t"
                                "shl.b32 %0, %2, 16;\n\t"
                                "and.b32 %1, %2, 0xffff0000;\n\t"
                                "}\n"
                                : "=f"((&ops[24 + _blk * 8 + _pair * 2])[0]), "=f"((&ops[24 + _blk * 8 + _pair * 2])[1])
                                : "r"(_vpairs_39[_pair]));
                        }
                    }
                }
                float scaled_10[16];
                scaled_10[0] = accum[96] * sa[6] * sw[0];
                scaled_10[1] = accum[97] * sa[6] * sw[1];
                scaled_10[2] = accum[100] * sa[6] * sw[2];
                scaled_10[3] = accum[101] * sa[6] * sw[3];
                scaled_10[4] = accum[104] * sa[6] * sw[4];
                scaled_10[5] = accum[105] * sa[6] * sw[5];
                scaled_10[6] = accum[108] * sa[6] * sw[6];
                scaled_10[7] = accum[109] * sa[6] * sw[7];
                scaled_10[8] = accum[112] * sa[6] * sw[8];
                scaled_10[9] = accum[113] * sa[6] * sw[9];
                scaled_10[10] = accum[116] * sa[6] * sw[10];
                scaled_10[11] = accum[117] * sa[6] * sw[11];
                scaled_10[12] = accum[120] * sa[6] * sw[12];
                scaled_10[13] = accum[121] * sa[6] * sw[13];
                scaled_10[14] = accum[124] * sa[6] * sw[14];
                scaled_10[15] = accum[125] * sa[6] * sw[15];
                uint32_t scaled_10_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_10[_lp*2 + 0], scaled_10[_lp*2+1 + 0]));
                    scaled_10_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t _stmatrix_addr_40 = static_cast<uint32_t>(staging_addr + (unsigned int)(srow_st * 256) + (unsigned int)((warp_n * 64 + smat * 8) * 2 ^ (srow_st & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_40), "r"(*reinterpret_cast<const uint32_t*>(&scaled_10_bf16[0])), "r"(*reinterpret_cast<const uint32_t*>(&scaled_10_bf16[1])), "r"(*reinterpret_cast<const uint32_t*>(&scaled_10_bf16[2])), "r"(*reinterpret_cast<const uint32_t*>(&scaled_10_bf16[3]))
                    : "memory");
                uint32_t _stmatrix_addr_41 = static_cast<uint32_t>(staging_addr + (unsigned int)(srow_st * 256) + (unsigned int)((warp_n * 64 + (4 + smat) * 8) * 2 ^ (srow_st & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_41), "r"(*reinterpret_cast<const uint32_t*>(&scaled_10_bf16[4])), "r"(*reinterpret_cast<const uint32_t*>(&scaled_10_bf16[5])), "r"(*reinterpret_cast<const uint32_t*>(&scaled_10_bf16[6])), "r"(*reinterpret_cast<const uint32_t*>(&scaled_10_bf16[7]))
                    : "memory");
                asm volatile("barrier.sync 8, 256;" ::: "memory");
                int grow_11 = grow_base + 48;
                if (grow_11 < M) {
                    unsigned int o_words_6[8];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&o_words_6[0])), "=r"(*reinterpret_cast<uint32_t*>(&o_words_6[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&o_words_6[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&o_words_6[(0) + 3]))
                        : "r"(staging_addr + (unsigned int)(srow * 256) + (unsigned int)(c * 32 ^ (srow & 7) << 4)));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&o_words_6[4])), "=r"(*reinterpret_cast<uint32_t*>(&o_words_6[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&o_words_6[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&o_words_6[(4) + 3]))
                        : "r"(staging_addr + (unsigned int)(srow * 256) + (unsigned int)(c * 32 + 16 ^ (srow & 7) << 4)));
                    float o_words_f32_6[16];
                    #pragma unroll
                    for (int _pair = 0; _pair < 8; _pair++) {
                        asm volatile(
                            "{\n\t"
                            "shl.b32 %0, %2, 16;\n\t"
                            "and.b32 %1, %2, 0xffff0000;\n\t"
                            "}\n"
                            : "=f"((&o_words_f32_6[_pair * 2])[0]), "=f"((&o_words_f32_6[_pair * 2])[1])
                            : "r"(o_words_6[_pair]));
                    }
                    int valid_6 = meta[1];
                    float p_raw_6[16];
                    for (int e_12 = 0; e_12 < 8; e_12++) {
                        float gv0_6 = ops[e_12];
                        float gv1_6 = ops[8 + e_12];
                        float gs0_6 = ((valid_6 == 1) ? gv0_6 : 0.0f);
                        float gs1_6 = ((valid_6 == 1) ? gv1_6 : 0.0f);
                        p_raw_6[e_12] = gs0_6 * o_words_f32_6[e_12];
                        p_raw_6[8 + e_12] = gs1_6 * o_words_f32_6[8 + e_12];
                    }
                    uint32_t p_raw_bf16_6[8];
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(p_raw_6[_lp*2 + 0], p_raw_6[_lp*2+1 + 0]));
                        p_raw_bf16_6[_lp] = *(uint32_t*)&_bf2;
                    }
                    float p_raw_bf16_f32_6[16];
                    #pragma unroll
                    for (int _pair = 0; _pair < 8; _pair++) {
                        asm volatile(
                            "{\n\t"
                            "shl.b32 %0, %2, 16;\n\t"
                            "and.b32 %1, %2, 0xffff0000;\n\t"
                            "}\n"
                            : "=f"((&p_raw_bf16_f32_6[_pair * 2])[0]), "=f"((&p_raw_bf16_f32_6[_pair * 2])[1])
                            : "r"(p_raw_bf16_6[_pair]));
                    }
                    float out_raw_6[16];
                    for (int e_13 = 0; e_13 < 8; e_13++) {
                        float rv0_6 = ops[16 + e_13];
                        float rv1_6 = ops[24 + e_13];
                        out_raw_6[e_13] = rv0_6 + p_raw_bf16_f32_6[e_13];
                        out_raw_6[8 + e_13] = rv1_6 + p_raw_bf16_f32_6[8 + e_13];
                    }
                    uint32_t out_raw_bf16_6[8];
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_raw_6[_lp*2 + 0], out_raw_6[_lp*2+1 + 0]));
                        out_raw_bf16_6[_lp] = *(uint32_t*)&_bf2;
                    }
                    if (grow_11 < M) {
                        reinterpret_cast<int4*>(out + ((grow_11 * 5376 + col) / 2))[0] = reinterpret_cast<int4*>(out_raw_bf16_6 + 0)[0];
                        reinterpret_cast<int4*>(out + ((grow_11 * 5376 + col) / 2 + 4))[0] = reinterpret_cast<int4*>(out_raw_bf16_6 + 4)[0];
                    }
                }
                asm volatile("barrier.sync 8, 256;" ::: "memory");
                meta[0] = gate_index[((grow_base + 56 < M) ? grow_base + 56 : 0)];
                meta[1] = ((meta[0] >= 0 && meta[0] < 9) ? 1 : 0);
                {
                    const uint4* _vptr_42 = reinterpret_cast<const uint4*>(gate + (((meta[1] == 1) ? meta[0] : 0) * 5376 + col) + 0);
                    uint4 _vld_42[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_42[_blk] = _vptr_42[_blk];
                        uint32_t* _vpairs_42 = reinterpret_cast<uint32_t*>(&_vld_42[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            asm volatile(
                                "{\n\t"
                                "shl.b32 %0, %2, 16;\n\t"
                                "and.b32 %1, %2, 0xffff0000;\n\t"
                                "}\n"
                                : "=f"((&ops[0 + _blk * 8 + _pair * 2])[0]), "=f"((&ops[0 + _blk * 8 + _pair * 2])[1])
                                : "r"(_vpairs_42[_pair]));
                        }
                    }
                }
                {
                    const uint4* _vptr_43 = reinterpret_cast<const uint4*>(gate + (((meta[1] == 1) ? meta[0] : 0) * 5376 + col + 8) + 0);
                    uint4 _vld_43[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_43[_blk] = _vptr_43[_blk];
                        uint32_t* _vpairs_43 = reinterpret_cast<uint32_t*>(&_vld_43[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            asm volatile(
                                "{\n\t"
                                "shl.b32 %0, %2, 16;\n\t"
                                "and.b32 %1, %2, 0xffff0000;\n\t"
                                "}\n"
                                : "=f"((&ops[8 + _blk * 8 + _pair * 2])[0]), "=f"((&ops[8 + _blk * 8 + _pair * 2])[1])
                                : "r"(_vpairs_43[_pair]));
                        }
                    }
                }
                {
                    const uint4* _vptr_44 = reinterpret_cast<const uint4*>(residual + (((grow_base + 56 < M) ? grow_base + 56 : 0) * 5376 + col) + 0);
                    uint4 _vld_44[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_44[_blk] = _vptr_44[_blk];
                        uint32_t* _vpairs_44 = reinterpret_cast<uint32_t*>(&_vld_44[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            asm volatile(
                                "{\n\t"
                                "shl.b32 %0, %2, 16;\n\t"
                                "and.b32 %1, %2, 0xffff0000;\n\t"
                                "}\n"
                                : "=f"((&ops[16 + _blk * 8 + _pair * 2])[0]), "=f"((&ops[16 + _blk * 8 + _pair * 2])[1])
                                : "r"(_vpairs_44[_pair]));
                        }
                    }
                }
                {
                    const uint4* _vptr_45 = reinterpret_cast<const uint4*>(residual + (((grow_base + 56 < M) ? grow_base + 56 : 0) * 5376 + col + 8) + 0);
                    uint4 _vld_45[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_45[_blk] = _vptr_45[_blk];
                        uint32_t* _vpairs_45 = reinterpret_cast<uint32_t*>(&_vld_45[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            asm volatile(
                                "{\n\t"
                                "shl.b32 %0, %2, 16;\n\t"
                                "and.b32 %1, %2, 0xffff0000;\n\t"
                                "}\n"
                                : "=f"((&ops[24 + _blk * 8 + _pair * 2])[0]), "=f"((&ops[24 + _blk * 8 + _pair * 2])[1])
                                : "r"(_vpairs_45[_pair]));
                        }
                    }
                }
                float scaled_12[16];
                scaled_12[0] = accum[98] * sa[7] * sw[0];
                scaled_12[1] = accum[99] * sa[7] * sw[1];
                scaled_12[2] = accum[102] * sa[7] * sw[2];
                scaled_12[3] = accum[103] * sa[7] * sw[3];
                scaled_12[4] = accum[106] * sa[7] * sw[4];
                scaled_12[5] = accum[107] * sa[7] * sw[5];
                scaled_12[6] = accum[110] * sa[7] * sw[6];
                scaled_12[7] = accum[111] * sa[7] * sw[7];
                scaled_12[8] = accum[114] * sa[7] * sw[8];
                scaled_12[9] = accum[115] * sa[7] * sw[9];
                scaled_12[10] = accum[118] * sa[7] * sw[10];
                scaled_12[11] = accum[119] * sa[7] * sw[11];
                scaled_12[12] = accum[122] * sa[7] * sw[12];
                scaled_12[13] = accum[123] * sa[7] * sw[13];
                scaled_12[14] = accum[126] * sa[7] * sw[14];
                scaled_12[15] = accum[127] * sa[7] * sw[15];
                uint32_t scaled_12_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_12[_lp*2 + 0], scaled_12[_lp*2+1 + 0]));
                    scaled_12_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t _stmatrix_addr_46 = static_cast<uint32_t>(staging_addr + (unsigned int)(srow_st * 256) + (unsigned int)((warp_n * 64 + smat * 8) * 2 ^ (srow_st & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_46), "r"(*reinterpret_cast<const uint32_t*>(&scaled_12_bf16[0])), "r"(*reinterpret_cast<const uint32_t*>(&scaled_12_bf16[1])), "r"(*reinterpret_cast<const uint32_t*>(&scaled_12_bf16[2])), "r"(*reinterpret_cast<const uint32_t*>(&scaled_12_bf16[3]))
                    : "memory");
                uint32_t _stmatrix_addr_47 = static_cast<uint32_t>(staging_addr + (unsigned int)(srow_st * 256) + (unsigned int)((warp_n * 64 + (4 + smat) * 8) * 2 ^ (srow_st & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_47), "r"(*reinterpret_cast<const uint32_t*>(&scaled_12_bf16[4])), "r"(*reinterpret_cast<const uint32_t*>(&scaled_12_bf16[5])), "r"(*reinterpret_cast<const uint32_t*>(&scaled_12_bf16[6])), "r"(*reinterpret_cast<const uint32_t*>(&scaled_12_bf16[7]))
                    : "memory");
                asm volatile("barrier.sync 8, 256;" ::: "memory");
                int grow_13 = grow_base + 56;
                if (grow_13 < M) {
                    unsigned int o_words_7[8];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&o_words_7[0])), "=r"(*reinterpret_cast<uint32_t*>(&o_words_7[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&o_words_7[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&o_words_7[(0) + 3]))
                        : "r"(staging_addr + (unsigned int)(srow * 256) + (unsigned int)(c * 32 ^ (srow & 7) << 4)));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&o_words_7[4])), "=r"(*reinterpret_cast<uint32_t*>(&o_words_7[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&o_words_7[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&o_words_7[(4) + 3]))
                        : "r"(staging_addr + (unsigned int)(srow * 256) + (unsigned int)(c * 32 + 16 ^ (srow & 7) << 4)));
                    float o_words_f32_7[16];
                    #pragma unroll
                    for (int _pair = 0; _pair < 8; _pair++) {
                        asm volatile(
                            "{\n\t"
                            "shl.b32 %0, %2, 16;\n\t"
                            "and.b32 %1, %2, 0xffff0000;\n\t"
                            "}\n"
                            : "=f"((&o_words_f32_7[_pair * 2])[0]), "=f"((&o_words_f32_7[_pair * 2])[1])
                            : "r"(o_words_7[_pair]));
                    }
                    int valid_7 = meta[1];
                    float p_raw_7[16];
                    for (int e_14 = 0; e_14 < 8; e_14++) {
                        float gv0_7 = ops[e_14];
                        float gv1_7 = ops[8 + e_14];
                        float gs0_7 = ((valid_7 == 1) ? gv0_7 : 0.0f);
                        float gs1_7 = ((valid_7 == 1) ? gv1_7 : 0.0f);
                        p_raw_7[e_14] = gs0_7 * o_words_f32_7[e_14];
                        p_raw_7[8 + e_14] = gs1_7 * o_words_f32_7[8 + e_14];
                    }
                    uint32_t p_raw_bf16_7[8];
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(p_raw_7[_lp*2 + 0], p_raw_7[_lp*2+1 + 0]));
                        p_raw_bf16_7[_lp] = *(uint32_t*)&_bf2;
                    }
                    float p_raw_bf16_f32_7[16];
                    #pragma unroll
                    for (int _pair = 0; _pair < 8; _pair++) {
                        asm volatile(
                            "{\n\t"
                            "shl.b32 %0, %2, 16;\n\t"
                            "and.b32 %1, %2, 0xffff0000;\n\t"
                            "}\n"
                            : "=f"((&p_raw_bf16_f32_7[_pair * 2])[0]), "=f"((&p_raw_bf16_f32_7[_pair * 2])[1])
                            : "r"(p_raw_bf16_7[_pair]));
                    }
                    float out_raw_7[16];
                    for (int e_15 = 0; e_15 < 8; e_15++) {
                        float rv0_7 = ops[16 + e_15];
                        float rv1_7 = ops[24 + e_15];
                        out_raw_7[e_15] = rv0_7 + p_raw_bf16_f32_7[e_15];
                        out_raw_7[8 + e_15] = rv1_7 + p_raw_bf16_f32_7[8 + e_15];
                    }
                    uint32_t out_raw_bf16_7[8];
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_raw_7[_lp*2 + 0], out_raw_7[_lp*2+1 + 0]));
                        out_raw_bf16_7[_lp] = *(uint32_t*)&_bf2;
                    }
                    if (grow_13 < M) {
                        reinterpret_cast<int4*>(out + ((grow_13 * 5376 + col) / 2))[0] = reinterpret_cast<int4*>(out_raw_bf16_7 + 0)[0];
                        reinterpret_cast<int4*>(out + ((grow_13 * 5376 + col) / 2 + 4))[0] = reinterpret_cast<int4*>(out_raw_bf16_7 + 4)[0];
                    }
                }
                asm volatile("barrier.sync 8, 256;" ::: "memory");
            }
        }
    }

    // Cleanup
}

}  // namespace h3_out_proj_gemm_fp8_sm120a
#undef GROUP_M
#undef H3_QOP_INF
#undef NUM_AB_PIPE_STAGES
#undef SMEM_A_STAGE_OFF
#undef SMEM_A_STAGE_STAGE_BYTES
#undef SMEM_A_STAGE_STRIDE
#undef SMEM_B_STAGE_OFF
#undef SMEM_B_STAGE_STAGE_BYTES
#undef SMEM_B_STAGE_STRIDE
#undef SMEM_SFA_SLOT_OFF
#undef SMEM_SFA_SLOT_STAGE_BYTES
#undef SMEM_SFA_SLOT_STRIDE
#undef SMEM_SFB_STAGE_OFF
#undef SMEM_SFB_STAGE_STAGE_BYTES
#undef SMEM_SFB_STAGE_STRIDE
#undef SMEM_STAGING_OFF
#undef SMEM_STAGING_STAGE_BYTES
#undef SMEM_STAGING_STRIDE
#undef SMEM_TOTAL
#undef THREADS
#undef ab_empty_addr
#undef ab_full_addr

namespace h3_out_proj_gemm_nvfp4_sm120a {

__device__ __forceinline__ int make_warp_uniform(int x) {
    int result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1F, 0xFFFFFFFF;"
                 : "=r"(result) : "r"(x));
    return result;
}

#define H3_QOP_INF CUDART_INF_F
#define NUM_AB_PIPE_STAGES 3
#define SMEM_A_STAGE_OFF 1024
#define SMEM_A_STAGE_STAGE_BYTES 16384
#define SMEM_A_STAGE_STRIDE 16384
#define SMEM_B_STAGE_OFF 50176
#define SMEM_B_STAGE_STAGE_BYTES 8192
#define SMEM_B_STAGE_STRIDE 8192
#define SMEM_STAGING_OFF 74752
#define SMEM_STAGING_STAGE_BYTES 8192
#define SMEM_STAGING_STRIDE 8192
#define SMEM_SFA_SLOT_OFF 82944
#define SMEM_SFA_SLOT_STAGE_BYTES 4096
#define SMEM_SFA_SLOT_STRIDE 4096
#define SMEM_SFB_STAGE_OFF 91136
#define SMEM_SFB_STAGE_STAGE_BYTES 1024
#define SMEM_SFB_STAGE_STRIDE 1024
#define SMEM_TOTAL 94208
#define THREADS 384
#define GROUP_M 16

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


__device__ __forceinline__ uint32_t mbarrier_try_wait_plain(int mbar_addr, int phase) {
    uint32_t token;
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "mbarrier.try_wait.parity.shared::cta.b64 P1, [%1], %2;\n\t"
        "selp.u32 %0, 1, 0, P1;\n\t"
        "}\n"
        : "=r"(token)
        : "r"(mbar_addr), "r"(phase) : "memory");
    return token;
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


__device__ __forceinline__ void mbarrier_wait(int mbar_addr, int phase) {
    uint32_t ticks = 0x989680;
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "LAB_WAIT:\n\t"
        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64"
        " P1, [%0], %1, %2;\n\t"
        "@P1 bra.uni DONE;\n\t"
        "bra.uni LAB_WAIT;\n\t"
        "DONE:\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"(phase), "r"(ticks) : "memory");
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


__device__ __forceinline__ void tma_2d_gmem2smem(
    int dst, const void *tmap_ptr, int x, int y, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global"
        ".mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3}], [%4];"
        :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y),
           "r"(mbar_addr) : "memory");
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


__global__ __launch_bounds__(384, 1) void
kernel_h3_out_proj_gemm_fused(const __grid_constant__ CUtensorMap A, const __grid_constant__ CUtensorMap B, const __grid_constant__ CUtensorMap SFA, const __grid_constant__ CUtensorMap SFB, unsigned int* __restrict__ attn_out_w, unsigned int* __restrict__ act_q_w, uint8_t* __restrict__ act_sf_w, float* __restrict__ act_scale, float* __restrict__ act_global_scale, float* __restrict__ w_scale, __nv_bfloat16* __restrict__ gate, int* __restrict__ gate_index, __nv_bfloat16* __restrict__ residual, unsigned int* __restrict__ out, unsigned int* __restrict__ flags, int M, int num_m_tiles, int total_tiles, float alpha)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int mbar_base = smem;
    #define ab_full_addr (mbar_base + 0)
    #define ab_empty_addr (mbar_base + 24)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // Kernel setup ops
    uint8_t* A_stage = reinterpret_cast<uint8_t*>(smem_raw + 1024);
    const int A_stage_addr = smem + 1024;
    uint8_t* B_stage = reinterpret_cast<uint8_t*>(smem_raw + 50176);
    const int B_stage_addr = smem + 50176;
    __nv_bfloat16* staging = reinterpret_cast<__nv_bfloat16*>(smem_raw + 74752);
    const int staging_addr = smem + 74752;
    unsigned int* SFA_slot = reinterpret_cast<unsigned int*>(smem_raw + 82944);
    const int SFA_slot_addr = smem + 82944;
    unsigned int* SFB_stage = reinterpret_cast<unsigned int*>(smem_raw + 91136);
    const int SFB_stage_addr = smem + 91136;

    // Mbarrier init (2 pipeline groups, 0 ordered-sequence groups, 6 barriers)
    // Mbarriers at smem_raw[0..48)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'ab_pipe' ---
            // ab_full: 3 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            mbarrier_init(smem + 8, 1);
            mbarrier_init(smem + 16, 1);
            // ab_empty: 3 barriers, init_count=8
            mbarrier_init(smem + 24, 8);
            mbarrier_init(smem + 32, 8);
            mbarrier_init(smem + 40, 8);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncthreads();

    // ---- Warpgroup: 0 ----
    if (warp >= 0 && warp <= 3) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 56;");
        // ---- Role: producer ----
        if (warp == 0) {
            { // producer_main
                unsigned int load_stage = 0;
                unsigned int _phase_ab_empty = 1;
                #pragma unroll 1
                for (int tile = bid; tile < total_tiles; tile += num_bids) {
                    int tile_m = tile / (GROUP_M * 42) * GROUP_M + (tile - tile / (GROUP_M * 42) * (GROUP_M * 42)) % ((num_m_tiles - tile / (GROUP_M * 42) * GROUP_M < GROUP_M) ? num_m_tiles - tile / (GROUP_M * 42) * GROUP_M : GROUP_M);
                    int tile_n = (tile - tile / (GROUP_M * 42) * (GROUP_M * 42)) / ((num_m_tiles - tile / (GROUP_M * 42) * GROUP_M < GROUP_M) ? num_m_tiles - tile / (GROUP_M * 42) * GROUP_M : GROUP_M);
                    int rows_left = M - tile_m * 256;
                    unsigned int rows_ready = ((rows_left < 256) ? rows_left : 256);
                    if (elect_sync()) {
                        {
                        unsigned int _acquire_observed;
                        do {
                        asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(_acquire_observed) : "l"((reinterpret_cast<unsigned int*>(flags) + (tile_m))) : "memory");
                        } while (static_cast<unsigned int>(_acquire_observed - static_cast<unsigned int>(rows_ready)) >= static_cast<unsigned int>(1));
                        }
                        asm volatile("fence.proxy.async;");
                        #pragma unroll 1
                        for (int kp = 0; kp < 28; kp++) {
                            mbarrier_wait(ab_empty_addr + (load_stage) * 8, _phase_ab_empty);
                            mbarrier_arrive_expect_tx(ab_full_addr + (load_stage) * 8, 29696);
                            asm volatile(
                                "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes"
                                " [%0], [%1, {%2, %3}], [%4];"
                                :: "r"(A_stage_addr + load_stage * 16384), "l"((&A)), "r"(kp * 2 * 64), "r"(tile_m * 256), "r"(ab_full_addr + (load_stage) * 8) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes"
                                " [%0], [%1, {%2, %3}], [%4];"
                                :: "r"(B_stage_addr + load_stage * 8192), "l"((&B)), "r"(kp * 2 * 64), "r"(tile_n * 128), "r"(ab_full_addr + (load_stage) * 8) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes"
                                " [%0], [%1, {%2, %3}], [%4];"
                                :: "r"(SFB_stage_addr + load_stage * 1024), "l"((&SFB)), "r"(0), "r"(tile_n * 224 + kp * 2 * 4), "r"(ab_full_addr + (load_stage) * 8) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes"
                                " [%0], [%1, {%2, %3}], [%4];"
                                :: "r"(SFA_slot_addr + (unsigned int)((kp * 2 >> 1 & 1) * 4096)), "l"((&SFA)), "r"((kp * 2 >> 1) * 16), "r"(tile_m * 256), "r"(ab_full_addr + (load_stage) * 8) : "memory");
                            load_stage += 1;
                            if (load_stage == 3) { load_stage = 0; _phase_ab_empty ^= 1; }
                            mbarrier_wait(ab_empty_addr + (load_stage) * 8, _phase_ab_empty);
                            mbarrier_arrive_expect_tx(ab_full_addr + (load_stage) * 8, 25600);
                            asm volatile(
                                "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes"
                                " [%0], [%1, {%2, %3}], [%4];"
                                :: "r"(A_stage_addr + load_stage * 16384), "l"((&A)), "r"((kp * 2 + 1) * 64), "r"(tile_m * 256), "r"(ab_full_addr + (load_stage) * 8) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes"
                                " [%0], [%1, {%2, %3}], [%4];"
                                :: "r"(B_stage_addr + load_stage * 8192), "l"((&B)), "r"((kp * 2 + 1) * 64), "r"(tile_n * 128), "r"(ab_full_addr + (load_stage) * 8) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes"
                                " [%0], [%1, {%2, %3}], [%4];"
                                :: "r"(SFB_stage_addr + load_stage * 1024), "l"((&SFB)), "r"(0), "r"(tile_n * 224 + (kp * 2 + 1) * 4), "r"(ab_full_addr + (load_stage) * 8) : "memory");
                            load_stage += 1;
                            if (load_stage == 3) { load_stage = 0; _phase_ab_empty ^= 1; }
                        }
                    }
                }
            }
        // ---- Role: quant ----
        } else if (warp >= 1 && warp <= 3) {
            { // quant_main
                unsigned int qw[16];
                float qx[16];
                float qacc[4];
                unsigned int qwords[4];
                float global_scale = act_global_scale[0];
                int warp_id_in_role = (warp - 1);
                int qrow0 = bid * 3 + warp_id_in_role;
                #pragma unroll 1
                for (int row = qrow0; row < M; row += num_bids * 3) {
                    qacc[0] = 0.0f;
                    #pragma unroll 2
                    for (int j = 0; j < 7; j++) {
                        {
                            const uint4* _ivptr_0 = reinterpret_cast<const uint4*>(attn_out_w + (row * 3584 + (lane + 32 * j) * 16) + 0);
                            uint4 _ivld_0;
                            _ivld_0 = *_ivptr_0;
                            qw[0 + 0] = _ivld_0.x;
                            qw[0 + 1] = _ivld_0.y;
                            qw[0 + 2] = _ivld_0.z;
                            qw[0 + 3] = _ivld_0.w;
                        }
                        {
                            const uint4* _ivptr_1 = reinterpret_cast<const uint4*>(attn_out_w + (row * 3584 + (lane + 32 * j) * 16 + 4) + 0);
                            uint4 _ivld_1;
                            _ivld_1 = *_ivptr_1;
                            qw[4 + 0] = _ivld_1.x;
                            qw[4 + 1] = _ivld_1.y;
                            qw[4 + 2] = _ivld_1.z;
                            qw[4 + 3] = _ivld_1.w;
                        }
                        {
                            const uint4* _ivptr_2 = reinterpret_cast<const uint4*>(attn_out_w + (row * 3584 + (lane + 32 * j) * 16 + 8) + 0);
                            uint4 _ivld_2;
                            _ivld_2 = *_ivptr_2;
                            qw[8 + 0] = _ivld_2.x;
                            qw[8 + 1] = _ivld_2.y;
                            qw[8 + 2] = _ivld_2.z;
                            qw[8 + 3] = _ivld_2.w;
                        }
                        {
                            const uint4* _ivptr_3 = reinterpret_cast<const uint4*>(attn_out_w + (row * 3584 + (lane + 32 * j) * 16 + 12) + 0);
                            uint4 _ivld_3;
                            _ivld_3 = *_ivptr_3;
                            qw[12 + 0] = _ivld_3.x;
                            qw[12 + 1] = _ivld_3.y;
                            qw[12 + 2] = _ivld_3.z;
                            qw[12 + 3] = _ivld_3.w;
                        }
                        float qw_f32[16];
                        #pragma unroll
                        for (int _pair = 0; _pair < 8; _pair++) {
                            asm volatile(
                                "{\n\t"
                                "shl.b32 %0, %2, 16;\n\t"
                                "and.b32 %1, %2, 0xffff0000;\n\t"
                                "}\n"
                                : "=f"((&qw_f32[_pair * 2])[0]), "=f"((&qw_f32[_pair * 2])[1])
                                : "r"(qw[_pair]));
                        }
                        qacc[0] = 0.0f;
                        qx[0] = qw_f32[0];
                        float _fabs_0 = fabsf(qw_f32[0]);
                        float _fmax_0 = fmaxf(qacc[0], _fabs_0);
                        qacc[0] = _fmax_0;
                        qx[1] = qw_f32[1];
                        float _fabs_1 = fabsf(qw_f32[1]);
                        float _fmax_1 = fmaxf(qacc[0], _fabs_1);
                        qacc[0] = _fmax_1;
                        qx[2] = qw_f32[2];
                        float _fabs_2 = fabsf(qw_f32[2]);
                        float _fmax_2 = fmaxf(qacc[0], _fabs_2);
                        qacc[0] = _fmax_2;
                        qx[3] = qw_f32[3];
                        float _fabs_3 = fabsf(qw_f32[3]);
                        float _fmax_3 = fmaxf(qacc[0], _fabs_3);
                        qacc[0] = _fmax_3;
                        qx[4] = qw_f32[4];
                        float _fabs_4 = fabsf(qw_f32[4]);
                        float _fmax_4 = fmaxf(qacc[0], _fabs_4);
                        qacc[0] = _fmax_4;
                        qx[5] = qw_f32[5];
                        float _fabs_5 = fabsf(qw_f32[5]);
                        float _fmax_5 = fmaxf(qacc[0], _fabs_5);
                        qacc[0] = _fmax_5;
                        qx[6] = qw_f32[6];
                        float _fabs_6 = fabsf(qw_f32[6]);
                        float _fmax_6 = fmaxf(qacc[0], _fabs_6);
                        qacc[0] = _fmax_6;
                        qx[7] = qw_f32[7];
                        float _fabs_7 = fabsf(qw_f32[7]);
                        float _fmax_7 = fmaxf(qacc[0], _fabs_7);
                        qacc[0] = _fmax_7;
                        qx[8] = qw_f32[8];
                        float _fabs_8 = fabsf(qw_f32[8]);
                        float _fmax_8 = fmaxf(qacc[0], _fabs_8);
                        qacc[0] = _fmax_8;
                        qx[9] = qw_f32[9];
                        float _fabs_9 = fabsf(qw_f32[9]);
                        float _fmax_9 = fmaxf(qacc[0], _fabs_9);
                        qacc[0] = _fmax_9;
                        qx[10] = qw_f32[10];
                        float _fabs_10 = fabsf(qw_f32[10]);
                        float _fmax_10 = fmaxf(qacc[0], _fabs_10);
                        qacc[0] = _fmax_10;
                        qx[11] = qw_f32[11];
                        float _fabs_11 = fabsf(qw_f32[11]);
                        float _fmax_11 = fmaxf(qacc[0], _fabs_11);
                        qacc[0] = _fmax_11;
                        qx[12] = qw_f32[12];
                        float _fabs_12 = fabsf(qw_f32[12]);
                        float _fmax_12 = fmaxf(qacc[0], _fabs_12);
                        qacc[0] = _fmax_12;
                        qx[13] = qw_f32[13];
                        float _fabs_13 = fabsf(qw_f32[13]);
                        float _fmax_13 = fmaxf(qacc[0], _fabs_13);
                        qacc[0] = _fmax_13;
                        qx[14] = qw_f32[14];
                        float _fabs_14 = fabsf(qw_f32[14]);
                        float _fmax_14 = fmaxf(qacc[0], _fabs_14);
                        qacc[0] = _fmax_14;
                        qx[15] = qw_f32[15];
                        float _fabs_15 = fabsf(qw_f32[15]);
                        float _fmax_15 = fmaxf(qacc[0], _fabs_15);
                        qacc[0] = _fmax_15;
                        qacc[1] = qacc[0] * 0.16666666666666666f * global_scale;
                        {
                            unsigned short _sf_pair;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, 0f00000000, %1;" : "=h"(_sf_pair) : "f"(qacc[1]));
                            *(reinterpret_cast<unsigned char*>(act_sf_w + (row * 448 + (lane + 32 * j) * 2)) + (0)) = (unsigned char)(_sf_pair & 0x7F);
                        }
                        uint16_t _e4m3x2_f32_0;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_0) : "f"(0.0f), "f"(qacc[1]));
                        uint16_t _e4m3x2_decode_4 = (uint16_t)((unsigned int)_e4m3x2_f32_0 & 0xFFu);
                        uint32_t _f16x2_decode_4;
                        float _fp8_decode_0;
                        asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_decode_4) : "h"(_e4m3x2_decode_4));
                        uint16_t _f16_decode_4 = (uint16_t)_f16x2_decode_4;
                        asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_decode_0) : "h"(_f16_decode_4));
                        qacc[2] = _fp8_decode_0;
                        float _fdiv_rn_0 = __fdiv_rn(global_scale, qacc[2]);
                        qacc[3] = ((qacc[2] != 0.0f) ? _fdiv_rn_0 : 0.0f);
                        qx[0] = qx[0] * qacc[3];
                        qx[1] = qx[1] * qacc[3];
                        qx[2] = qx[2] * qacc[3];
                        qx[3] = qx[3] * qacc[3];
                        qx[4] = qx[4] * qacc[3];
                        qx[5] = qx[5] * qacc[3];
                        qx[6] = qx[6] * qacc[3];
                        qx[7] = qx[7] * qacc[3];
                        qx[8] = qx[8] * qacc[3];
                        qx[9] = qx[9] * qacc[3];
                        qx[10] = qx[10] * qacc[3];
                        qx[11] = qx[11] * qacc[3];
                        qx[12] = qx[12] * qacc[3];
                        qx[13] = qx[13] * qacc[3];
                        qx[14] = qx[14] * qacc[3];
                        qx[15] = qx[15] * qacc[3];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(qwords[0]) : "f"(qx[0]), "f"(qx[1]), "f"(qx[2]), "f"(qx[3]), "f"(qx[4]), "f"(qx[5]), "f"(qx[6]), "f"(qx[7]));
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(qwords[1]) : "f"(qx[8]), "f"(qx[9]), "f"(qx[10]), "f"(qx[11]), "f"(qx[12]), "f"(qx[13]), "f"(qx[14]), "f"(qx[15]));
                        float qw_f32_0[16];
                        #pragma unroll
                        for (int _pair = 0; _pair < 8; _pair++) {
                            asm volatile(
                                "{\n\t"
                                "shl.b32 %0, %2, 16;\n\t"
                                "and.b32 %1, %2, 0xffff0000;\n\t"
                                "}\n"
                                : "=f"((&qw_f32_0[_pair * 2])[0]), "=f"((&qw_f32_0[_pair * 2])[1])
                                : "r"(qw[8 + _pair]));
                        }
                        qacc[0] = 0.0f;
                        qx[0] = qw_f32_0[0];
                        float _fabs_16 = fabsf(qw_f32_0[0]);
                        float _fmax_16 = fmaxf(qacc[0], _fabs_16);
                        qacc[0] = _fmax_16;
                        qx[1] = qw_f32_0[1];
                        float _fabs_17 = fabsf(qw_f32_0[1]);
                        float _fmax_17 = fmaxf(qacc[0], _fabs_17);
                        qacc[0] = _fmax_17;
                        qx[2] = qw_f32_0[2];
                        float _fabs_18 = fabsf(qw_f32_0[2]);
                        float _fmax_18 = fmaxf(qacc[0], _fabs_18);
                        qacc[0] = _fmax_18;
                        qx[3] = qw_f32_0[3];
                        float _fabs_19 = fabsf(qw_f32_0[3]);
                        float _fmax_19 = fmaxf(qacc[0], _fabs_19);
                        qacc[0] = _fmax_19;
                        qx[4] = qw_f32_0[4];
                        float _fabs_20 = fabsf(qw_f32_0[4]);
                        float _fmax_20 = fmaxf(qacc[0], _fabs_20);
                        qacc[0] = _fmax_20;
                        qx[5] = qw_f32_0[5];
                        float _fabs_21 = fabsf(qw_f32_0[5]);
                        float _fmax_21 = fmaxf(qacc[0], _fabs_21);
                        qacc[0] = _fmax_21;
                        qx[6] = qw_f32_0[6];
                        float _fabs_22 = fabsf(qw_f32_0[6]);
                        float _fmax_22 = fmaxf(qacc[0], _fabs_22);
                        qacc[0] = _fmax_22;
                        qx[7] = qw_f32_0[7];
                        float _fabs_23 = fabsf(qw_f32_0[7]);
                        float _fmax_23 = fmaxf(qacc[0], _fabs_23);
                        qacc[0] = _fmax_23;
                        qx[8] = qw_f32_0[8];
                        float _fabs_24 = fabsf(qw_f32_0[8]);
                        float _fmax_24 = fmaxf(qacc[0], _fabs_24);
                        qacc[0] = _fmax_24;
                        qx[9] = qw_f32_0[9];
                        float _fabs_25 = fabsf(qw_f32_0[9]);
                        float _fmax_25 = fmaxf(qacc[0], _fabs_25);
                        qacc[0] = _fmax_25;
                        qx[10] = qw_f32_0[10];
                        float _fabs_26 = fabsf(qw_f32_0[10]);
                        float _fmax_26 = fmaxf(qacc[0], _fabs_26);
                        qacc[0] = _fmax_26;
                        qx[11] = qw_f32_0[11];
                        float _fabs_27 = fabsf(qw_f32_0[11]);
                        float _fmax_27 = fmaxf(qacc[0], _fabs_27);
                        qacc[0] = _fmax_27;
                        qx[12] = qw_f32_0[12];
                        float _fabs_28 = fabsf(qw_f32_0[12]);
                        float _fmax_28 = fmaxf(qacc[0], _fabs_28);
                        qacc[0] = _fmax_28;
                        qx[13] = qw_f32_0[13];
                        float _fabs_29 = fabsf(qw_f32_0[13]);
                        float _fmax_29 = fmaxf(qacc[0], _fabs_29);
                        qacc[0] = _fmax_29;
                        qx[14] = qw_f32_0[14];
                        float _fabs_30 = fabsf(qw_f32_0[14]);
                        float _fmax_30 = fmaxf(qacc[0], _fabs_30);
                        qacc[0] = _fmax_30;
                        qx[15] = qw_f32_0[15];
                        float _fabs_31 = fabsf(qw_f32_0[15]);
                        float _fmax_31 = fmaxf(qacc[0], _fabs_31);
                        qacc[0] = _fmax_31;
                        qacc[1] = qacc[0] * 0.16666666666666666f * global_scale;
                        {
                            unsigned short _sf_pair;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, 0f00000000, %1;" : "=h"(_sf_pair) : "f"(qacc[1]));
                            *(reinterpret_cast<unsigned char*>(act_sf_w + (row * 448 + (lane + 32 * j) * 2 + 1)) + (0)) = (unsigned char)(_sf_pair & 0x7F);
                        }
                        uint16_t _e4m3x2_f32_1;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_1) : "f"(0.0f), "f"(qacc[1]));
                        uint16_t _e4m3x2_decode_5 = (uint16_t)((unsigned int)_e4m3x2_f32_1 & 0xFFu);
                        uint32_t _f16x2_decode_5;
                        float _fp8_decode_1;
                        asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_decode_5) : "h"(_e4m3x2_decode_5));
                        uint16_t _f16_decode_5 = (uint16_t)_f16x2_decode_5;
                        asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_decode_1) : "h"(_f16_decode_5));
                        qacc[2] = _fp8_decode_1;
                        float _fdiv_rn_1 = __fdiv_rn(global_scale, qacc[2]);
                        qacc[3] = ((qacc[2] != 0.0f) ? _fdiv_rn_1 : 0.0f);
                        qx[0] = qx[0] * qacc[3];
                        qx[1] = qx[1] * qacc[3];
                        qx[2] = qx[2] * qacc[3];
                        qx[3] = qx[3] * qacc[3];
                        qx[4] = qx[4] * qacc[3];
                        qx[5] = qx[5] * qacc[3];
                        qx[6] = qx[6] * qacc[3];
                        qx[7] = qx[7] * qacc[3];
                        qx[8] = qx[8] * qacc[3];
                        qx[9] = qx[9] * qacc[3];
                        qx[10] = qx[10] * qacc[3];
                        qx[11] = qx[11] * qacc[3];
                        qx[12] = qx[12] * qacc[3];
                        qx[13] = qx[13] * qacc[3];
                        qx[14] = qx[14] * qacc[3];
                        qx[15] = qx[15] * qacc[3];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(qwords[2]) : "f"(qx[0]), "f"(qx[1]), "f"(qx[2]), "f"(qx[3]), "f"(qx[4]), "f"(qx[5]), "f"(qx[6]), "f"(qx[7]));
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(qwords[3]) : "f"(qx[8]), "f"(qx[9]), "f"(qx[10]), "f"(qx[11]), "f"(qx[12]), "f"(qx[13]), "f"(qx[14]), "f"(qx[15]));
                        reinterpret_cast<int4*>(act_q_w + ((row * 3584 + (lane + 32 * j) * 16) / 4))[0] = reinterpret_cast<int4*>(qwords)[0];
                    }
                    __threadfence();
                    __syncwarp();
                    if (elect_sync()) {
                        asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(flags) + (row / 256))), "r"(static_cast<unsigned int>(1)) : "memory");
                    }
                }
            }
        }
    }
    // ---- Role: mma ----
    if (warp >= 4 && warp <= 11) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 224;");
        { // mma_main
            unsigned int mma_stage = 0;
            int warp_id_in_role_1 = (warp - 4);
            int warp_m = warp_id_in_role_1 % 4;
            int warp_n = warp_id_in_role_1 / 4;
            int role_tid = warp_id_in_role_1 * 32 + lane;
            float accum[128];
            unsigned int a_frag[16];
            unsigned int b_frag[16];
            unsigned int _phase_ab_full = 0;
            #pragma unroll 1
            for (int tile_1 = bid; tile_1 < total_tiles; tile_1 += num_bids) {
                int tile_m_1 = tile_1 / (GROUP_M * 42) * GROUP_M + (tile_1 - tile_1 / (GROUP_M * 42) * (GROUP_M * 42)) % ((num_m_tiles - tile_1 / (GROUP_M * 42) * GROUP_M < GROUP_M) ? num_m_tiles - tile_1 / (GROUP_M * 42) * GROUP_M : GROUP_M);
                int tile_n_1 = (tile_1 - tile_1 / (GROUP_M * 42) * (GROUP_M * 42)) / ((num_m_tiles - tile_1 / (GROUP_M * 42) * GROUP_M < GROUP_M) ? num_m_tiles - tile_1 / (GROUP_M * 42) * GROUP_M : GROUP_M);
                accum[0] = 0.0f;
                accum[1] = 0.0f;
                accum[2] = 0.0f;
                accum[3] = 0.0f;
                accum[4] = 0.0f;
                accum[5] = 0.0f;
                accum[6] = 0.0f;
                accum[7] = 0.0f;
                accum[8] = 0.0f;
                accum[9] = 0.0f;
                accum[10] = 0.0f;
                accum[11] = 0.0f;
                accum[12] = 0.0f;
                accum[13] = 0.0f;
                accum[14] = 0.0f;
                accum[15] = 0.0f;
                accum[16] = 0.0f;
                accum[17] = 0.0f;
                accum[18] = 0.0f;
                accum[19] = 0.0f;
                accum[20] = 0.0f;
                accum[21] = 0.0f;
                accum[22] = 0.0f;
                accum[23] = 0.0f;
                accum[24] = 0.0f;
                accum[25] = 0.0f;
                accum[26] = 0.0f;
                accum[27] = 0.0f;
                accum[28] = 0.0f;
                accum[29] = 0.0f;
                accum[30] = 0.0f;
                accum[31] = 0.0f;
                accum[32] = 0.0f;
                accum[33] = 0.0f;
                accum[34] = 0.0f;
                accum[35] = 0.0f;
                accum[36] = 0.0f;
                accum[37] = 0.0f;
                accum[38] = 0.0f;
                accum[39] = 0.0f;
                accum[40] = 0.0f;
                accum[41] = 0.0f;
                accum[42] = 0.0f;
                accum[43] = 0.0f;
                accum[44] = 0.0f;
                accum[45] = 0.0f;
                accum[46] = 0.0f;
                accum[47] = 0.0f;
                accum[48] = 0.0f;
                accum[49] = 0.0f;
                accum[50] = 0.0f;
                accum[51] = 0.0f;
                accum[52] = 0.0f;
                accum[53] = 0.0f;
                accum[54] = 0.0f;
                accum[55] = 0.0f;
                accum[56] = 0.0f;
                accum[57] = 0.0f;
                accum[58] = 0.0f;
                accum[59] = 0.0f;
                accum[60] = 0.0f;
                accum[61] = 0.0f;
                accum[62] = 0.0f;
                accum[63] = 0.0f;
                accum[64] = 0.0f;
                accum[65] = 0.0f;
                accum[66] = 0.0f;
                accum[67] = 0.0f;
                accum[68] = 0.0f;
                accum[69] = 0.0f;
                accum[70] = 0.0f;
                accum[71] = 0.0f;
                accum[72] = 0.0f;
                accum[73] = 0.0f;
                accum[74] = 0.0f;
                accum[75] = 0.0f;
                accum[76] = 0.0f;
                accum[77] = 0.0f;
                accum[78] = 0.0f;
                accum[79] = 0.0f;
                accum[80] = 0.0f;
                accum[81] = 0.0f;
                accum[82] = 0.0f;
                accum[83] = 0.0f;
                accum[84] = 0.0f;
                accum[85] = 0.0f;
                accum[86] = 0.0f;
                accum[87] = 0.0f;
                accum[88] = 0.0f;
                accum[89] = 0.0f;
                accum[90] = 0.0f;
                accum[91] = 0.0f;
                accum[92] = 0.0f;
                accum[93] = 0.0f;
                accum[94] = 0.0f;
                accum[95] = 0.0f;
                accum[96] = 0.0f;
                accum[97] = 0.0f;
                accum[98] = 0.0f;
                accum[99] = 0.0f;
                accum[100] = 0.0f;
                accum[101] = 0.0f;
                accum[102] = 0.0f;
                accum[103] = 0.0f;
                accum[104] = 0.0f;
                accum[105] = 0.0f;
                accum[106] = 0.0f;
                accum[107] = 0.0f;
                accum[108] = 0.0f;
                accum[109] = 0.0f;
                accum[110] = 0.0f;
                accum[111] = 0.0f;
                accum[112] = 0.0f;
                accum[113] = 0.0f;
                accum[114] = 0.0f;
                accum[115] = 0.0f;
                accum[116] = 0.0f;
                accum[117] = 0.0f;
                accum[118] = 0.0f;
                accum[119] = 0.0f;
                accum[120] = 0.0f;
                accum[121] = 0.0f;
                accum[122] = 0.0f;
                accum[123] = 0.0f;
                accum[124] = 0.0f;
                accum[125] = 0.0f;
                accum[126] = 0.0f;
                accum[127] = 0.0f;
                int col_base = tile_n_1 * 128 + warp_n * 64 + (lane & 3) * 2;
                float sw[16];
                sw[0] = alpha;
                sw[1] = alpha;
                sw[2] = alpha;
                sw[3] = alpha;
                sw[4] = alpha;
                sw[5] = alpha;
                sw[6] = alpha;
                sw[7] = alpha;
                sw[8] = alpha;
                sw[9] = alpha;
                sw[10] = alpha;
                sw[11] = alpha;
                sw[12] = alpha;
                sw[13] = alpha;
                sw[14] = alpha;
                sw[15] = alpha;
                float sa[8];
                #pragma unroll 1
                for (int kp_1 = 0; kp_1 < 28; kp_1++) {
                    mbarrier_wait(ab_full_addr + (mma_stage) * 8, _phase_ab_full);
                    if (kp_1 == 24) {
                        int arow = tile_m_1 * 256 + warp_m * 64 + (lane >> 2);
                        sa[0] = 1.0f;
                        int arow_0 = tile_m_1 * 256 + warp_m * 64 + 8 + (lane >> 2);
                        sa[1] = 1.0f;
                        int arow_1 = tile_m_1 * 256 + warp_m * 64 + 16 + (lane >> 2);
                        sa[2] = 1.0f;
                        int arow_2 = tile_m_1 * 256 + warp_m * 64 + 16 + 8 + (lane >> 2);
                        sa[3] = 1.0f;
                        int arow_3 = tile_m_1 * 256 + warp_m * 64 + 32 + (lane >> 2);
                        sa[4] = 1.0f;
                        int arow_4 = tile_m_1 * 256 + warp_m * 64 + 32 + 8 + (lane >> 2);
                        sa[5] = 1.0f;
                        int arow_5 = tile_m_1 * 256 + warp_m * 64 + 48 + (lane >> 2);
                        sa[6] = 1.0f;
                        int arow_6 = tile_m_1 * 256 + warp_m * 64 + 48 + 8 + (lane >> 2);
                        sa[7] = 1.0f;
                    }
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                        : "r"(A_stage_addr + mma_stage * 16384 + (unsigned int)((warp_m * 64 + (lane >> 3 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)((lane >> 3 >> 1) * 16 ^ (warp_m * 64 + (lane >> 3 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_frag[4]), "=r"(a_frag[5]), "=r"(a_frag[6]), "=r"(a_frag[7])
                        : "r"(A_stage_addr + mma_stage * 16384 + (unsigned int)((warp_m * 64 + 16 + (lane >> 3 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)((lane >> 3 >> 1) * 16 ^ (warp_m * 64 + 16 + (lane >> 3 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_frag[8]), "=r"(a_frag[9]), "=r"(a_frag[10]), "=r"(a_frag[11])
                        : "r"(A_stage_addr + mma_stage * 16384 + (unsigned int)((warp_m * 64 + 32 + (lane >> 3 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)((lane >> 3 >> 1) * 16 ^ (warp_m * 64 + 32 + (lane >> 3 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_frag[12]), "=r"(a_frag[13]), "=r"(a_frag[14]), "=r"(a_frag[15])
                        : "r"(A_stage_addr + mma_stage * 16384 + (unsigned int)((warp_m * 64 + 48 + (lane >> 3 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)((lane >> 3 >> 1) * 16 ^ (warp_m * 64 + 48 + (lane >> 3 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                        : "r"(B_stage_addr + mma_stage * 8192 + (unsigned int)((warp_n * 64 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)((lane >> 3 & 1) * 16 ^ (warp_n * 64 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_frag[4]), "=r"(b_frag[5]), "=r"(b_frag[6]), "=r"(b_frag[7])
                        : "r"(B_stage_addr + mma_stage * 8192 + (unsigned int)((warp_n * 64 + 16 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)((lane >> 3 & 1) * 16 ^ (warp_n * 64 + 16 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_frag[8]), "=r"(b_frag[9]), "=r"(b_frag[10]), "=r"(b_frag[11])
                        : "r"(B_stage_addr + mma_stage * 8192 + (unsigned int)((warp_n * 64 + 32 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)((lane >> 3 & 1) * 16 ^ (warp_n * 64 + 32 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_frag[12]), "=r"(b_frag[13]), "=r"(b_frag[14]), "=r"(b_frag[15])
                        : "r"(B_stage_addr + mma_stage * 8192 + (unsigned int)((warp_n * 64 + 48 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)((lane >> 3 & 1) * 16 ^ (warp_n * 64 + 48 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    unsigned int _SFA_slot_reg_0[1];
                    {
                        const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(SFA_slot);
                        #pragma unroll
                        for (int _lr = 0; _lr < 1; _lr++)
                            _SFA_slot_reg_0[_lr] = _smem_ptr[((kp_1 * 2 >> 1 & 1) * 1024 + (kp_1 * 2 & 1) * 2 + (warp_m * 64 + (lane & 1) * 8 + (lane >> 2)) * 4) + _lr];
                    }
                    unsigned int _SFA_slot_reg_1[1];
                    {
                        const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(SFA_slot);
                        #pragma unroll
                        for (int _lr = 0; _lr < 1; _lr++)
                            _SFA_slot_reg_1[_lr] = _smem_ptr[((kp_1 * 2 >> 1 & 1) * 1024 + (kp_1 * 2 & 1) * 2 + (warp_m * 64 + 16 + (lane & 1) * 8 + (lane >> 2)) * 4) + _lr];
                    }
                    unsigned int _SFA_slot_reg_2[1];
                    {
                        const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(SFA_slot);
                        #pragma unroll
                        for (int _lr = 0; _lr < 1; _lr++)
                            _SFA_slot_reg_2[_lr] = _smem_ptr[((kp_1 * 2 >> 1 & 1) * 1024 + (kp_1 * 2 & 1) * 2 + (warp_m * 64 + 32 + (lane & 1) * 8 + (lane >> 2)) * 4) + _lr];
                    }
                    unsigned int _SFA_slot_reg_3[1];
                    {
                        const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(SFA_slot);
                        #pragma unroll
                        for (int _lr = 0; _lr < 1; _lr++)
                            _SFA_slot_reg_3[_lr] = _smem_ptr[((kp_1 * 2 >> 1 & 1) * 1024 + (kp_1 * 2 & 1) * 2 + (warp_m * 64 + 48 + (lane & 1) * 8 + (lane >> 2)) * 4) + _lr];
                    }
                    unsigned int _SFB_stage_reg_0[1];
                    {
                        const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(SFB_stage);
                        #pragma unroll
                        for (int _lr = 0; _lr < 1; _lr++)
                            _SFB_stage_reg_0[_lr] = _smem_ptr[(mma_stage * 256 + (unsigned int)((lane >> 2) * 4) + (unsigned int)(warp_n * 2)) + _lr];
                    }
                    unsigned int _SFB_stage_reg_1[1];
                    {
                        const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(SFB_stage);
                        #pragma unroll
                        for (int _lr = 0; _lr < 1; _lr++)
                            _SFB_stage_reg_1[_lr] = _smem_ptr[(mma_stage * 256 + (unsigned int)((8 + (lane >> 2)) * 4) + (unsigned int)(warp_n * 2)) + _lr];
                    }
                    unsigned int _SFB_stage_reg_2[1];
                    {
                        const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(SFB_stage);
                        #pragma unroll
                        for (int _lr = 0; _lr < 1; _lr++)
                            _SFB_stage_reg_2[_lr] = _smem_ptr[(mma_stage * 256 + (unsigned int)((16 + (lane >> 2)) * 4) + (unsigned int)(warp_n * 2)) + _lr];
                    }
                    unsigned int _SFB_stage_reg_3[1];
                    {
                        const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(SFB_stage);
                        #pragma unroll
                        for (int _lr = 0; _lr < 1; _lr++)
                            _SFB_stage_reg_3[_lr] = _smem_ptr[(mma_stage * 256 + (unsigned int)((24 + (lane >> 2)) * 4) + (unsigned int)(warp_n * 2)) + _lr];
                    }
                    unsigned int _SFB_stage_reg_4[1];
                    {
                        const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(SFB_stage);
                        #pragma unroll
                        for (int _lr = 0; _lr < 1; _lr++)
                            _SFB_stage_reg_4[_lr] = _smem_ptr[(mma_stage * 256 + (unsigned int)((lane >> 2) * 4) + (unsigned int)(warp_n * 2 + 1)) + _lr];
                    }
                    unsigned int _SFB_stage_reg_5[1];
                    {
                        const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(SFB_stage);
                        #pragma unroll
                        for (int _lr = 0; _lr < 1; _lr++)
                            _SFB_stage_reg_5[_lr] = _smem_ptr[(mma_stage * 256 + (unsigned int)((8 + (lane >> 2)) * 4) + (unsigned int)(warp_n * 2 + 1)) + _lr];
                    }
                    unsigned int _SFB_stage_reg_6[1];
                    {
                        const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(SFB_stage);
                        #pragma unroll
                        for (int _lr = 0; _lr < 1; _lr++)
                            _SFB_stage_reg_6[_lr] = _smem_ptr[(mma_stage * 256 + (unsigned int)((16 + (lane >> 2)) * 4) + (unsigned int)(warp_n * 2 + 1)) + _lr];
                    }
                    unsigned int _SFB_stage_reg_7[1];
                    {
                        const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(SFB_stage);
                        #pragma unroll
                        for (int _lr = 0; _lr < 1; _lr++)
                            _SFB_stage_reg_7[_lr] = _smem_ptr[(mma_stage * 256 + (unsigned int)((24 + (lane >> 2)) * 4) + (unsigned int)(warp_n * 2 + 1)) + _lr];
                    }
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[0]), "+f"(accum[1]), "+f"(accum[2]), "+f"(accum[3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]), "r"((uint32_t)(_SFA_slot_reg_0[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_0[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[4]), "+f"(accum[(4) + 1]), "+f"(accum[(4) + 2]), "+f"(accum[(4) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]), "r"((uint32_t)(_SFA_slot_reg_0[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_1[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[8]), "+f"(accum[(8) + 1]), "+f"(accum[(8) + 2]), "+f"(accum[(8) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[4]), "r"(b_frag[(4) + 1]), "r"((uint32_t)(_SFA_slot_reg_0[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_2[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[12]), "+f"(accum[(12) + 1]), "+f"(accum[(12) + 2]), "+f"(accum[(12) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[6]), "r"(b_frag[(6) + 1]), "r"((uint32_t)(_SFA_slot_reg_0[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_3[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[16]), "+f"(accum[(16) + 1]), "+f"(accum[(16) + 2]), "+f"(accum[(16) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[8]), "r"(b_frag[(8) + 1]), "r"((uint32_t)(_SFA_slot_reg_0[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_4[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[20]), "+f"(accum[(20) + 1]), "+f"(accum[(20) + 2]), "+f"(accum[(20) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[10]), "r"(b_frag[(10) + 1]), "r"((uint32_t)(_SFA_slot_reg_0[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_5[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[24]), "+f"(accum[(24) + 1]), "+f"(accum[(24) + 2]), "+f"(accum[(24) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[12]), "r"(b_frag[(12) + 1]), "r"((uint32_t)(_SFA_slot_reg_0[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_6[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[28]), "+f"(accum[(28) + 1]), "+f"(accum[(28) + 2]), "+f"(accum[(28) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[14]), "r"(b_frag[(14) + 1]), "r"((uint32_t)(_SFA_slot_reg_0[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_7[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[32]), "+f"(accum[(32) + 1]), "+f"(accum[(32) + 2]), "+f"(accum[(32) + 3])
                        : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[0]), "r"(b_frag[1]), "r"((uint32_t)(_SFA_slot_reg_1[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_0[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[36]), "+f"(accum[(36) + 1]), "+f"(accum[(36) + 2]), "+f"(accum[(36) + 3])
                        : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]), "r"((uint32_t)(_SFA_slot_reg_1[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_1[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[40]), "+f"(accum[(40) + 1]), "+f"(accum[(40) + 2]), "+f"(accum[(40) + 3])
                        : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[4]), "r"(b_frag[(4) + 1]), "r"((uint32_t)(_SFA_slot_reg_1[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_2[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[44]), "+f"(accum[(44) + 1]), "+f"(accum[(44) + 2]), "+f"(accum[(44) + 3])
                        : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[6]), "r"(b_frag[(6) + 1]), "r"((uint32_t)(_SFA_slot_reg_1[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_3[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[48]), "+f"(accum[(48) + 1]), "+f"(accum[(48) + 2]), "+f"(accum[(48) + 3])
                        : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[8]), "r"(b_frag[(8) + 1]), "r"((uint32_t)(_SFA_slot_reg_1[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_4[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[52]), "+f"(accum[(52) + 1]), "+f"(accum[(52) + 2]), "+f"(accum[(52) + 3])
                        : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[10]), "r"(b_frag[(10) + 1]), "r"((uint32_t)(_SFA_slot_reg_1[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_5[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[56]), "+f"(accum[(56) + 1]), "+f"(accum[(56) + 2]), "+f"(accum[(56) + 3])
                        : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[12]), "r"(b_frag[(12) + 1]), "r"((uint32_t)(_SFA_slot_reg_1[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_6[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[60]), "+f"(accum[(60) + 1]), "+f"(accum[(60) + 2]), "+f"(accum[(60) + 3])
                        : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[14]), "r"(b_frag[(14) + 1]), "r"((uint32_t)(_SFA_slot_reg_1[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_7[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[64]), "+f"(accum[(64) + 1]), "+f"(accum[(64) + 2]), "+f"(accum[(64) + 3])
                        : "r"(a_frag[8]), "r"(a_frag[(8) + 1]), "r"(a_frag[(8) + 2]), "r"(a_frag[(8) + 3]), "r"(b_frag[0]), "r"(b_frag[1]), "r"((uint32_t)(_SFA_slot_reg_2[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_0[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[68]), "+f"(accum[(68) + 1]), "+f"(accum[(68) + 2]), "+f"(accum[(68) + 3])
                        : "r"(a_frag[8]), "r"(a_frag[(8) + 1]), "r"(a_frag[(8) + 2]), "r"(a_frag[(8) + 3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]), "r"((uint32_t)(_SFA_slot_reg_2[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_1[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[72]), "+f"(accum[(72) + 1]), "+f"(accum[(72) + 2]), "+f"(accum[(72) + 3])
                        : "r"(a_frag[8]), "r"(a_frag[(8) + 1]), "r"(a_frag[(8) + 2]), "r"(a_frag[(8) + 3]), "r"(b_frag[4]), "r"(b_frag[(4) + 1]), "r"((uint32_t)(_SFA_slot_reg_2[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_2[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[76]), "+f"(accum[(76) + 1]), "+f"(accum[(76) + 2]), "+f"(accum[(76) + 3])
                        : "r"(a_frag[8]), "r"(a_frag[(8) + 1]), "r"(a_frag[(8) + 2]), "r"(a_frag[(8) + 3]), "r"(b_frag[6]), "r"(b_frag[(6) + 1]), "r"((uint32_t)(_SFA_slot_reg_2[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_3[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[80]), "+f"(accum[(80) + 1]), "+f"(accum[(80) + 2]), "+f"(accum[(80) + 3])
                        : "r"(a_frag[8]), "r"(a_frag[(8) + 1]), "r"(a_frag[(8) + 2]), "r"(a_frag[(8) + 3]), "r"(b_frag[8]), "r"(b_frag[(8) + 1]), "r"((uint32_t)(_SFA_slot_reg_2[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_4[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[84]), "+f"(accum[(84) + 1]), "+f"(accum[(84) + 2]), "+f"(accum[(84) + 3])
                        : "r"(a_frag[8]), "r"(a_frag[(8) + 1]), "r"(a_frag[(8) + 2]), "r"(a_frag[(8) + 3]), "r"(b_frag[10]), "r"(b_frag[(10) + 1]), "r"((uint32_t)(_SFA_slot_reg_2[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_5[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[88]), "+f"(accum[(88) + 1]), "+f"(accum[(88) + 2]), "+f"(accum[(88) + 3])
                        : "r"(a_frag[8]), "r"(a_frag[(8) + 1]), "r"(a_frag[(8) + 2]), "r"(a_frag[(8) + 3]), "r"(b_frag[12]), "r"(b_frag[(12) + 1]), "r"((uint32_t)(_SFA_slot_reg_2[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_6[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[92]), "+f"(accum[(92) + 1]), "+f"(accum[(92) + 2]), "+f"(accum[(92) + 3])
                        : "r"(a_frag[8]), "r"(a_frag[(8) + 1]), "r"(a_frag[(8) + 2]), "r"(a_frag[(8) + 3]), "r"(b_frag[14]), "r"(b_frag[(14) + 1]), "r"((uint32_t)(_SFA_slot_reg_2[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_7[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[96]), "+f"(accum[(96) + 1]), "+f"(accum[(96) + 2]), "+f"(accum[(96) + 3])
                        : "r"(a_frag[12]), "r"(a_frag[(12) + 1]), "r"(a_frag[(12) + 2]), "r"(a_frag[(12) + 3]), "r"(b_frag[0]), "r"(b_frag[1]), "r"((uint32_t)(_SFA_slot_reg_3[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_0[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[100]), "+f"(accum[(100) + 1]), "+f"(accum[(100) + 2]), "+f"(accum[(100) + 3])
                        : "r"(a_frag[12]), "r"(a_frag[(12) + 1]), "r"(a_frag[(12) + 2]), "r"(a_frag[(12) + 3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]), "r"((uint32_t)(_SFA_slot_reg_3[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_1[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[104]), "+f"(accum[(104) + 1]), "+f"(accum[(104) + 2]), "+f"(accum[(104) + 3])
                        : "r"(a_frag[12]), "r"(a_frag[(12) + 1]), "r"(a_frag[(12) + 2]), "r"(a_frag[(12) + 3]), "r"(b_frag[4]), "r"(b_frag[(4) + 1]), "r"((uint32_t)(_SFA_slot_reg_3[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_2[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[108]), "+f"(accum[(108) + 1]), "+f"(accum[(108) + 2]), "+f"(accum[(108) + 3])
                        : "r"(a_frag[12]), "r"(a_frag[(12) + 1]), "r"(a_frag[(12) + 2]), "r"(a_frag[(12) + 3]), "r"(b_frag[6]), "r"(b_frag[(6) + 1]), "r"((uint32_t)(_SFA_slot_reg_3[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_3[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[112]), "+f"(accum[(112) + 1]), "+f"(accum[(112) + 2]), "+f"(accum[(112) + 3])
                        : "r"(a_frag[12]), "r"(a_frag[(12) + 1]), "r"(a_frag[(12) + 2]), "r"(a_frag[(12) + 3]), "r"(b_frag[8]), "r"(b_frag[(8) + 1]), "r"((uint32_t)(_SFA_slot_reg_3[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_4[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[116]), "+f"(accum[(116) + 1]), "+f"(accum[(116) + 2]), "+f"(accum[(116) + 3])
                        : "r"(a_frag[12]), "r"(a_frag[(12) + 1]), "r"(a_frag[(12) + 2]), "r"(a_frag[(12) + 3]), "r"(b_frag[10]), "r"(b_frag[(10) + 1]), "r"((uint32_t)(_SFA_slot_reg_3[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_5[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[120]), "+f"(accum[(120) + 1]), "+f"(accum[(120) + 2]), "+f"(accum[(120) + 3])
                        : "r"(a_frag[12]), "r"(a_frag[(12) + 1]), "r"(a_frag[(12) + 2]), "r"(a_frag[(12) + 3]), "r"(b_frag[12]), "r"(b_frag[(12) + 1]), "r"((uint32_t)(_SFA_slot_reg_3[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_6[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[124]), "+f"(accum[(124) + 1]), "+f"(accum[(124) + 2]), "+f"(accum[(124) + 3])
                        : "r"(a_frag[12]), "r"(a_frag[(12) + 1]), "r"(a_frag[(12) + 2]), "r"(a_frag[(12) + 3]), "r"(b_frag[14]), "r"(b_frag[(14) + 1]), "r"((uint32_t)(_SFA_slot_reg_3[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_7[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                        : "r"(A_stage_addr + mma_stage * 16384 + (unsigned int)((warp_m * 64 + (lane >> 3 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(32 + (lane >> 3 >> 1) * 16 ^ (warp_m * 64 + (lane >> 3 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_frag[4]), "=r"(a_frag[5]), "=r"(a_frag[6]), "=r"(a_frag[7])
                        : "r"(A_stage_addr + mma_stage * 16384 + (unsigned int)((warp_m * 64 + 16 + (lane >> 3 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(32 + (lane >> 3 >> 1) * 16 ^ (warp_m * 64 + 16 + (lane >> 3 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_frag[8]), "=r"(a_frag[9]), "=r"(a_frag[10]), "=r"(a_frag[11])
                        : "r"(A_stage_addr + mma_stage * 16384 + (unsigned int)((warp_m * 64 + 32 + (lane >> 3 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(32 + (lane >> 3 >> 1) * 16 ^ (warp_m * 64 + 32 + (lane >> 3 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_frag[12]), "=r"(a_frag[13]), "=r"(a_frag[14]), "=r"(a_frag[15])
                        : "r"(A_stage_addr + mma_stage * 16384 + (unsigned int)((warp_m * 64 + 48 + (lane >> 3 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(32 + (lane >> 3 >> 1) * 16 ^ (warp_m * 64 + 48 + (lane >> 3 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                        : "r"(B_stage_addr + mma_stage * 8192 + (unsigned int)((warp_n * 64 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(32 + (lane >> 3 & 1) * 16 ^ (warp_n * 64 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_frag[4]), "=r"(b_frag[5]), "=r"(b_frag[6]), "=r"(b_frag[7])
                        : "r"(B_stage_addr + mma_stage * 8192 + (unsigned int)((warp_n * 64 + 16 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(32 + (lane >> 3 & 1) * 16 ^ (warp_n * 64 + 16 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_frag[8]), "=r"(b_frag[9]), "=r"(b_frag[10]), "=r"(b_frag[11])
                        : "r"(B_stage_addr + mma_stage * 8192 + (unsigned int)((warp_n * 64 + 32 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(32 + (lane >> 3 & 1) * 16 ^ (warp_n * 64 + 32 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_frag[12]), "=r"(b_frag[13]), "=r"(b_frag[14]), "=r"(b_frag[15])
                        : "r"(B_stage_addr + mma_stage * 8192 + (unsigned int)((warp_n * 64 + 48 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(32 + (lane >> 3 & 1) * 16 ^ (warp_n * 64 + 48 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    unsigned int _SFA_slot_reg_4[1];
                    {
                        const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(SFA_slot);
                        #pragma unroll
                        for (int _lr = 0; _lr < 1; _lr++)
                            _SFA_slot_reg_4[_lr] = _smem_ptr[((kp_1 * 2 >> 1 & 1) * 1024 + (kp_1 * 2 & 1) * 2 + 1 + (warp_m * 64 + (lane & 1) * 8 + (lane >> 2)) * 4) + _lr];
                    }
                    unsigned int _SFA_slot_reg_5[1];
                    {
                        const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(SFA_slot);
                        #pragma unroll
                        for (int _lr = 0; _lr < 1; _lr++)
                            _SFA_slot_reg_5[_lr] = _smem_ptr[((kp_1 * 2 >> 1 & 1) * 1024 + (kp_1 * 2 & 1) * 2 + 1 + (warp_m * 64 + 16 + (lane & 1) * 8 + (lane >> 2)) * 4) + _lr];
                    }
                    unsigned int _SFA_slot_reg_6[1];
                    {
                        const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(SFA_slot);
                        #pragma unroll
                        for (int _lr = 0; _lr < 1; _lr++)
                            _SFA_slot_reg_6[_lr] = _smem_ptr[((kp_1 * 2 >> 1 & 1) * 1024 + (kp_1 * 2 & 1) * 2 + 1 + (warp_m * 64 + 32 + (lane & 1) * 8 + (lane >> 2)) * 4) + _lr];
                    }
                    unsigned int _SFA_slot_reg_7[1];
                    {
                        const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(SFA_slot);
                        #pragma unroll
                        for (int _lr = 0; _lr < 1; _lr++)
                            _SFA_slot_reg_7[_lr] = _smem_ptr[((kp_1 * 2 >> 1 & 1) * 1024 + (kp_1 * 2 & 1) * 2 + 1 + (warp_m * 64 + 48 + (lane & 1) * 8 + (lane >> 2)) * 4) + _lr];
                    }
                    unsigned int _SFB_stage_reg_8[1];
                    {
                        const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(SFB_stage);
                        #pragma unroll
                        for (int _lr = 0; _lr < 1; _lr++)
                            _SFB_stage_reg_8[_lr] = _smem_ptr[(mma_stage * 256 + 128 + (unsigned int)((lane >> 2) * 4) + (unsigned int)(warp_n * 2)) + _lr];
                    }
                    unsigned int _SFB_stage_reg_9[1];
                    {
                        const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(SFB_stage);
                        #pragma unroll
                        for (int _lr = 0; _lr < 1; _lr++)
                            _SFB_stage_reg_9[_lr] = _smem_ptr[(mma_stage * 256 + 128 + (unsigned int)((8 + (lane >> 2)) * 4) + (unsigned int)(warp_n * 2)) + _lr];
                    }
                    unsigned int _SFB_stage_reg_10[1];
                    {
                        const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(SFB_stage);
                        #pragma unroll
                        for (int _lr = 0; _lr < 1; _lr++)
                            _SFB_stage_reg_10[_lr] = _smem_ptr[(mma_stage * 256 + 128 + (unsigned int)((16 + (lane >> 2)) * 4) + (unsigned int)(warp_n * 2)) + _lr];
                    }
                    unsigned int _SFB_stage_reg_11[1];
                    {
                        const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(SFB_stage);
                        #pragma unroll
                        for (int _lr = 0; _lr < 1; _lr++)
                            _SFB_stage_reg_11[_lr] = _smem_ptr[(mma_stage * 256 + 128 + (unsigned int)((24 + (lane >> 2)) * 4) + (unsigned int)(warp_n * 2)) + _lr];
                    }
                    unsigned int _SFB_stage_reg_12[1];
                    {
                        const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(SFB_stage);
                        #pragma unroll
                        for (int _lr = 0; _lr < 1; _lr++)
                            _SFB_stage_reg_12[_lr] = _smem_ptr[(mma_stage * 256 + 128 + (unsigned int)((lane >> 2) * 4) + (unsigned int)(warp_n * 2 + 1)) + _lr];
                    }
                    unsigned int _SFB_stage_reg_13[1];
                    {
                        const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(SFB_stage);
                        #pragma unroll
                        for (int _lr = 0; _lr < 1; _lr++)
                            _SFB_stage_reg_13[_lr] = _smem_ptr[(mma_stage * 256 + 128 + (unsigned int)((8 + (lane >> 2)) * 4) + (unsigned int)(warp_n * 2 + 1)) + _lr];
                    }
                    unsigned int _SFB_stage_reg_14[1];
                    {
                        const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(SFB_stage);
                        #pragma unroll
                        for (int _lr = 0; _lr < 1; _lr++)
                            _SFB_stage_reg_14[_lr] = _smem_ptr[(mma_stage * 256 + 128 + (unsigned int)((16 + (lane >> 2)) * 4) + (unsigned int)(warp_n * 2 + 1)) + _lr];
                    }
                    unsigned int _SFB_stage_reg_15[1];
                    {
                        const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(SFB_stage);
                        #pragma unroll
                        for (int _lr = 0; _lr < 1; _lr++)
                            _SFB_stage_reg_15[_lr] = _smem_ptr[(mma_stage * 256 + 128 + (unsigned int)((24 + (lane >> 2)) * 4) + (unsigned int)(warp_n * 2 + 1)) + _lr];
                    }
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[0]), "+f"(accum[1]), "+f"(accum[2]), "+f"(accum[3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]), "r"((uint32_t)(_SFA_slot_reg_4[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_8[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[4]), "+f"(accum[(4) + 1]), "+f"(accum[(4) + 2]), "+f"(accum[(4) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]), "r"((uint32_t)(_SFA_slot_reg_4[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_9[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[8]), "+f"(accum[(8) + 1]), "+f"(accum[(8) + 2]), "+f"(accum[(8) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[4]), "r"(b_frag[(4) + 1]), "r"((uint32_t)(_SFA_slot_reg_4[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_10[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[12]), "+f"(accum[(12) + 1]), "+f"(accum[(12) + 2]), "+f"(accum[(12) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[6]), "r"(b_frag[(6) + 1]), "r"((uint32_t)(_SFA_slot_reg_4[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_11[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[16]), "+f"(accum[(16) + 1]), "+f"(accum[(16) + 2]), "+f"(accum[(16) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[8]), "r"(b_frag[(8) + 1]), "r"((uint32_t)(_SFA_slot_reg_4[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_12[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[20]), "+f"(accum[(20) + 1]), "+f"(accum[(20) + 2]), "+f"(accum[(20) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[10]), "r"(b_frag[(10) + 1]), "r"((uint32_t)(_SFA_slot_reg_4[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_13[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[24]), "+f"(accum[(24) + 1]), "+f"(accum[(24) + 2]), "+f"(accum[(24) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[12]), "r"(b_frag[(12) + 1]), "r"((uint32_t)(_SFA_slot_reg_4[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_14[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[28]), "+f"(accum[(28) + 1]), "+f"(accum[(28) + 2]), "+f"(accum[(28) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[14]), "r"(b_frag[(14) + 1]), "r"((uint32_t)(_SFA_slot_reg_4[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_15[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[32]), "+f"(accum[(32) + 1]), "+f"(accum[(32) + 2]), "+f"(accum[(32) + 3])
                        : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[0]), "r"(b_frag[1]), "r"((uint32_t)(_SFA_slot_reg_5[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_8[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[36]), "+f"(accum[(36) + 1]), "+f"(accum[(36) + 2]), "+f"(accum[(36) + 3])
                        : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]), "r"((uint32_t)(_SFA_slot_reg_5[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_9[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[40]), "+f"(accum[(40) + 1]), "+f"(accum[(40) + 2]), "+f"(accum[(40) + 3])
                        : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[4]), "r"(b_frag[(4) + 1]), "r"((uint32_t)(_SFA_slot_reg_5[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_10[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[44]), "+f"(accum[(44) + 1]), "+f"(accum[(44) + 2]), "+f"(accum[(44) + 3])
                        : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[6]), "r"(b_frag[(6) + 1]), "r"((uint32_t)(_SFA_slot_reg_5[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_11[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[48]), "+f"(accum[(48) + 1]), "+f"(accum[(48) + 2]), "+f"(accum[(48) + 3])
                        : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[8]), "r"(b_frag[(8) + 1]), "r"((uint32_t)(_SFA_slot_reg_5[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_12[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[52]), "+f"(accum[(52) + 1]), "+f"(accum[(52) + 2]), "+f"(accum[(52) + 3])
                        : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[10]), "r"(b_frag[(10) + 1]), "r"((uint32_t)(_SFA_slot_reg_5[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_13[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[56]), "+f"(accum[(56) + 1]), "+f"(accum[(56) + 2]), "+f"(accum[(56) + 3])
                        : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[12]), "r"(b_frag[(12) + 1]), "r"((uint32_t)(_SFA_slot_reg_5[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_14[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[60]), "+f"(accum[(60) + 1]), "+f"(accum[(60) + 2]), "+f"(accum[(60) + 3])
                        : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[14]), "r"(b_frag[(14) + 1]), "r"((uint32_t)(_SFA_slot_reg_5[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_15[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[64]), "+f"(accum[(64) + 1]), "+f"(accum[(64) + 2]), "+f"(accum[(64) + 3])
                        : "r"(a_frag[8]), "r"(a_frag[(8) + 1]), "r"(a_frag[(8) + 2]), "r"(a_frag[(8) + 3]), "r"(b_frag[0]), "r"(b_frag[1]), "r"((uint32_t)(_SFA_slot_reg_6[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_8[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[68]), "+f"(accum[(68) + 1]), "+f"(accum[(68) + 2]), "+f"(accum[(68) + 3])
                        : "r"(a_frag[8]), "r"(a_frag[(8) + 1]), "r"(a_frag[(8) + 2]), "r"(a_frag[(8) + 3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]), "r"((uint32_t)(_SFA_slot_reg_6[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_9[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[72]), "+f"(accum[(72) + 1]), "+f"(accum[(72) + 2]), "+f"(accum[(72) + 3])
                        : "r"(a_frag[8]), "r"(a_frag[(8) + 1]), "r"(a_frag[(8) + 2]), "r"(a_frag[(8) + 3]), "r"(b_frag[4]), "r"(b_frag[(4) + 1]), "r"((uint32_t)(_SFA_slot_reg_6[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_10[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[76]), "+f"(accum[(76) + 1]), "+f"(accum[(76) + 2]), "+f"(accum[(76) + 3])
                        : "r"(a_frag[8]), "r"(a_frag[(8) + 1]), "r"(a_frag[(8) + 2]), "r"(a_frag[(8) + 3]), "r"(b_frag[6]), "r"(b_frag[(6) + 1]), "r"((uint32_t)(_SFA_slot_reg_6[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_11[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[80]), "+f"(accum[(80) + 1]), "+f"(accum[(80) + 2]), "+f"(accum[(80) + 3])
                        : "r"(a_frag[8]), "r"(a_frag[(8) + 1]), "r"(a_frag[(8) + 2]), "r"(a_frag[(8) + 3]), "r"(b_frag[8]), "r"(b_frag[(8) + 1]), "r"((uint32_t)(_SFA_slot_reg_6[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_12[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[84]), "+f"(accum[(84) + 1]), "+f"(accum[(84) + 2]), "+f"(accum[(84) + 3])
                        : "r"(a_frag[8]), "r"(a_frag[(8) + 1]), "r"(a_frag[(8) + 2]), "r"(a_frag[(8) + 3]), "r"(b_frag[10]), "r"(b_frag[(10) + 1]), "r"((uint32_t)(_SFA_slot_reg_6[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_13[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[88]), "+f"(accum[(88) + 1]), "+f"(accum[(88) + 2]), "+f"(accum[(88) + 3])
                        : "r"(a_frag[8]), "r"(a_frag[(8) + 1]), "r"(a_frag[(8) + 2]), "r"(a_frag[(8) + 3]), "r"(b_frag[12]), "r"(b_frag[(12) + 1]), "r"((uint32_t)(_SFA_slot_reg_6[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_14[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[92]), "+f"(accum[(92) + 1]), "+f"(accum[(92) + 2]), "+f"(accum[(92) + 3])
                        : "r"(a_frag[8]), "r"(a_frag[(8) + 1]), "r"(a_frag[(8) + 2]), "r"(a_frag[(8) + 3]), "r"(b_frag[14]), "r"(b_frag[(14) + 1]), "r"((uint32_t)(_SFA_slot_reg_6[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_15[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[96]), "+f"(accum[(96) + 1]), "+f"(accum[(96) + 2]), "+f"(accum[(96) + 3])
                        : "r"(a_frag[12]), "r"(a_frag[(12) + 1]), "r"(a_frag[(12) + 2]), "r"(a_frag[(12) + 3]), "r"(b_frag[0]), "r"(b_frag[1]), "r"((uint32_t)(_SFA_slot_reg_7[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_8[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[100]), "+f"(accum[(100) + 1]), "+f"(accum[(100) + 2]), "+f"(accum[(100) + 3])
                        : "r"(a_frag[12]), "r"(a_frag[(12) + 1]), "r"(a_frag[(12) + 2]), "r"(a_frag[(12) + 3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]), "r"((uint32_t)(_SFA_slot_reg_7[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_9[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[104]), "+f"(accum[(104) + 1]), "+f"(accum[(104) + 2]), "+f"(accum[(104) + 3])
                        : "r"(a_frag[12]), "r"(a_frag[(12) + 1]), "r"(a_frag[(12) + 2]), "r"(a_frag[(12) + 3]), "r"(b_frag[4]), "r"(b_frag[(4) + 1]), "r"((uint32_t)(_SFA_slot_reg_7[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_10[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[108]), "+f"(accum[(108) + 1]), "+f"(accum[(108) + 2]), "+f"(accum[(108) + 3])
                        : "r"(a_frag[12]), "r"(a_frag[(12) + 1]), "r"(a_frag[(12) + 2]), "r"(a_frag[(12) + 3]), "r"(b_frag[6]), "r"(b_frag[(6) + 1]), "r"((uint32_t)(_SFA_slot_reg_7[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_11[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[112]), "+f"(accum[(112) + 1]), "+f"(accum[(112) + 2]), "+f"(accum[(112) + 3])
                        : "r"(a_frag[12]), "r"(a_frag[(12) + 1]), "r"(a_frag[(12) + 2]), "r"(a_frag[(12) + 3]), "r"(b_frag[8]), "r"(b_frag[(8) + 1]), "r"((uint32_t)(_SFA_slot_reg_7[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_12[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[116]), "+f"(accum[(116) + 1]), "+f"(accum[(116) + 2]), "+f"(accum[(116) + 3])
                        : "r"(a_frag[12]), "r"(a_frag[(12) + 1]), "r"(a_frag[(12) + 2]), "r"(a_frag[(12) + 3]), "r"(b_frag[10]), "r"(b_frag[(10) + 1]), "r"((uint32_t)(_SFA_slot_reg_7[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_13[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[120]), "+f"(accum[(120) + 1]), "+f"(accum[(120) + 2]), "+f"(accum[(120) + 3])
                        : "r"(a_frag[12]), "r"(a_frag[(12) + 1]), "r"(a_frag[(12) + 2]), "r"(a_frag[(12) + 3]), "r"(b_frag[12]), "r"(b_frag[(12) + 1]), "r"((uint32_t)(_SFA_slot_reg_7[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_14[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[124]), "+f"(accum[(124) + 1]), "+f"(accum[(124) + 2]), "+f"(accum[(124) + 3])
                        : "r"(a_frag[12]), "r"(a_frag[(12) + 1]), "r"(a_frag[(12) + 2]), "r"(a_frag[(12) + 3]), "r"(b_frag[14]), "r"(b_frag[(14) + 1]), "r"((uint32_t)(_SFA_slot_reg_7[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_15[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    __syncwarp();
                    if (elect_sync()) {
                        mbarrier_arrive(ab_empty_addr + (mma_stage) * 8);
                    }
                    mma_stage += 1;
                    if (mma_stage == 3) { mma_stage = 0; _phase_ab_full ^= 1; }
                    mbarrier_wait(ab_full_addr + (mma_stage) * 8, _phase_ab_full);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                        : "r"(A_stage_addr + mma_stage * 16384 + (unsigned int)((warp_m * 64 + (lane >> 3 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)((lane >> 3 >> 1) * 16 ^ (warp_m * 64 + (lane >> 3 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_frag[4]), "=r"(a_frag[5]), "=r"(a_frag[6]), "=r"(a_frag[7])
                        : "r"(A_stage_addr + mma_stage * 16384 + (unsigned int)((warp_m * 64 + 16 + (lane >> 3 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)((lane >> 3 >> 1) * 16 ^ (warp_m * 64 + 16 + (lane >> 3 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_frag[8]), "=r"(a_frag[9]), "=r"(a_frag[10]), "=r"(a_frag[11])
                        : "r"(A_stage_addr + mma_stage * 16384 + (unsigned int)((warp_m * 64 + 32 + (lane >> 3 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)((lane >> 3 >> 1) * 16 ^ (warp_m * 64 + 32 + (lane >> 3 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_frag[12]), "=r"(a_frag[13]), "=r"(a_frag[14]), "=r"(a_frag[15])
                        : "r"(A_stage_addr + mma_stage * 16384 + (unsigned int)((warp_m * 64 + 48 + (lane >> 3 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)((lane >> 3 >> 1) * 16 ^ (warp_m * 64 + 48 + (lane >> 3 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                        : "r"(B_stage_addr + mma_stage * 8192 + (unsigned int)((warp_n * 64 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)((lane >> 3 & 1) * 16 ^ (warp_n * 64 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_frag[4]), "=r"(b_frag[5]), "=r"(b_frag[6]), "=r"(b_frag[7])
                        : "r"(B_stage_addr + mma_stage * 8192 + (unsigned int)((warp_n * 64 + 16 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)((lane >> 3 & 1) * 16 ^ (warp_n * 64 + 16 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_frag[8]), "=r"(b_frag[9]), "=r"(b_frag[10]), "=r"(b_frag[11])
                        : "r"(B_stage_addr + mma_stage * 8192 + (unsigned int)((warp_n * 64 + 32 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)((lane >> 3 & 1) * 16 ^ (warp_n * 64 + 32 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_frag[12]), "=r"(b_frag[13]), "=r"(b_frag[14]), "=r"(b_frag[15])
                        : "r"(B_stage_addr + mma_stage * 8192 + (unsigned int)((warp_n * 64 + 48 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)((lane >> 3 & 1) * 16 ^ (warp_n * 64 + 48 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    unsigned int _SFA_slot_reg_8[1];
                    {
                        const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(SFA_slot);
                        #pragma unroll
                        for (int _lr = 0; _lr < 1; _lr++)
                            _SFA_slot_reg_8[_lr] = _smem_ptr[((kp_1 * 2 + 1 >> 1 & 1) * 1024 + (kp_1 * 2 + 1 & 1) * 2 + (warp_m * 64 + (lane & 1) * 8 + (lane >> 2)) * 4) + _lr];
                    }
                    unsigned int _SFA_slot_reg_9[1];
                    {
                        const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(SFA_slot);
                        #pragma unroll
                        for (int _lr = 0; _lr < 1; _lr++)
                            _SFA_slot_reg_9[_lr] = _smem_ptr[((kp_1 * 2 + 1 >> 1 & 1) * 1024 + (kp_1 * 2 + 1 & 1) * 2 + (warp_m * 64 + 16 + (lane & 1) * 8 + (lane >> 2)) * 4) + _lr];
                    }
                    unsigned int _SFA_slot_reg_10[1];
                    {
                        const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(SFA_slot);
                        #pragma unroll
                        for (int _lr = 0; _lr < 1; _lr++)
                            _SFA_slot_reg_10[_lr] = _smem_ptr[((kp_1 * 2 + 1 >> 1 & 1) * 1024 + (kp_1 * 2 + 1 & 1) * 2 + (warp_m * 64 + 32 + (lane & 1) * 8 + (lane >> 2)) * 4) + _lr];
                    }
                    unsigned int _SFA_slot_reg_11[1];
                    {
                        const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(SFA_slot);
                        #pragma unroll
                        for (int _lr = 0; _lr < 1; _lr++)
                            _SFA_slot_reg_11[_lr] = _smem_ptr[((kp_1 * 2 + 1 >> 1 & 1) * 1024 + (kp_1 * 2 + 1 & 1) * 2 + (warp_m * 64 + 48 + (lane & 1) * 8 + (lane >> 2)) * 4) + _lr];
                    }
                    unsigned int _SFB_stage_reg_16[1];
                    {
                        const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(SFB_stage);
                        #pragma unroll
                        for (int _lr = 0; _lr < 1; _lr++)
                            _SFB_stage_reg_16[_lr] = _smem_ptr[(mma_stage * 256 + (unsigned int)((lane >> 2) * 4) + (unsigned int)(warp_n * 2)) + _lr];
                    }
                    unsigned int _SFB_stage_reg_17[1];
                    {
                        const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(SFB_stage);
                        #pragma unroll
                        for (int _lr = 0; _lr < 1; _lr++)
                            _SFB_stage_reg_17[_lr] = _smem_ptr[(mma_stage * 256 + (unsigned int)((8 + (lane >> 2)) * 4) + (unsigned int)(warp_n * 2)) + _lr];
                    }
                    unsigned int _SFB_stage_reg_18[1];
                    {
                        const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(SFB_stage);
                        #pragma unroll
                        for (int _lr = 0; _lr < 1; _lr++)
                            _SFB_stage_reg_18[_lr] = _smem_ptr[(mma_stage * 256 + (unsigned int)((16 + (lane >> 2)) * 4) + (unsigned int)(warp_n * 2)) + _lr];
                    }
                    unsigned int _SFB_stage_reg_19[1];
                    {
                        const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(SFB_stage);
                        #pragma unroll
                        for (int _lr = 0; _lr < 1; _lr++)
                            _SFB_stage_reg_19[_lr] = _smem_ptr[(mma_stage * 256 + (unsigned int)((24 + (lane >> 2)) * 4) + (unsigned int)(warp_n * 2)) + _lr];
                    }
                    unsigned int _SFB_stage_reg_20[1];
                    {
                        const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(SFB_stage);
                        #pragma unroll
                        for (int _lr = 0; _lr < 1; _lr++)
                            _SFB_stage_reg_20[_lr] = _smem_ptr[(mma_stage * 256 + (unsigned int)((lane >> 2) * 4) + (unsigned int)(warp_n * 2 + 1)) + _lr];
                    }
                    unsigned int _SFB_stage_reg_21[1];
                    {
                        const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(SFB_stage);
                        #pragma unroll
                        for (int _lr = 0; _lr < 1; _lr++)
                            _SFB_stage_reg_21[_lr] = _smem_ptr[(mma_stage * 256 + (unsigned int)((8 + (lane >> 2)) * 4) + (unsigned int)(warp_n * 2 + 1)) + _lr];
                    }
                    unsigned int _SFB_stage_reg_22[1];
                    {
                        const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(SFB_stage);
                        #pragma unroll
                        for (int _lr = 0; _lr < 1; _lr++)
                            _SFB_stage_reg_22[_lr] = _smem_ptr[(mma_stage * 256 + (unsigned int)((16 + (lane >> 2)) * 4) + (unsigned int)(warp_n * 2 + 1)) + _lr];
                    }
                    unsigned int _SFB_stage_reg_23[1];
                    {
                        const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(SFB_stage);
                        #pragma unroll
                        for (int _lr = 0; _lr < 1; _lr++)
                            _SFB_stage_reg_23[_lr] = _smem_ptr[(mma_stage * 256 + (unsigned int)((24 + (lane >> 2)) * 4) + (unsigned int)(warp_n * 2 + 1)) + _lr];
                    }
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[0]), "+f"(accum[1]), "+f"(accum[2]), "+f"(accum[3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]), "r"((uint32_t)(_SFA_slot_reg_8[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_16[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[4]), "+f"(accum[(4) + 1]), "+f"(accum[(4) + 2]), "+f"(accum[(4) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]), "r"((uint32_t)(_SFA_slot_reg_8[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_17[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[8]), "+f"(accum[(8) + 1]), "+f"(accum[(8) + 2]), "+f"(accum[(8) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[4]), "r"(b_frag[(4) + 1]), "r"((uint32_t)(_SFA_slot_reg_8[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_18[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[12]), "+f"(accum[(12) + 1]), "+f"(accum[(12) + 2]), "+f"(accum[(12) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[6]), "r"(b_frag[(6) + 1]), "r"((uint32_t)(_SFA_slot_reg_8[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_19[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[16]), "+f"(accum[(16) + 1]), "+f"(accum[(16) + 2]), "+f"(accum[(16) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[8]), "r"(b_frag[(8) + 1]), "r"((uint32_t)(_SFA_slot_reg_8[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_20[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[20]), "+f"(accum[(20) + 1]), "+f"(accum[(20) + 2]), "+f"(accum[(20) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[10]), "r"(b_frag[(10) + 1]), "r"((uint32_t)(_SFA_slot_reg_8[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_21[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[24]), "+f"(accum[(24) + 1]), "+f"(accum[(24) + 2]), "+f"(accum[(24) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[12]), "r"(b_frag[(12) + 1]), "r"((uint32_t)(_SFA_slot_reg_8[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_22[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[28]), "+f"(accum[(28) + 1]), "+f"(accum[(28) + 2]), "+f"(accum[(28) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[14]), "r"(b_frag[(14) + 1]), "r"((uint32_t)(_SFA_slot_reg_8[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_23[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[32]), "+f"(accum[(32) + 1]), "+f"(accum[(32) + 2]), "+f"(accum[(32) + 3])
                        : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[0]), "r"(b_frag[1]), "r"((uint32_t)(_SFA_slot_reg_9[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_16[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[36]), "+f"(accum[(36) + 1]), "+f"(accum[(36) + 2]), "+f"(accum[(36) + 3])
                        : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]), "r"((uint32_t)(_SFA_slot_reg_9[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_17[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[40]), "+f"(accum[(40) + 1]), "+f"(accum[(40) + 2]), "+f"(accum[(40) + 3])
                        : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[4]), "r"(b_frag[(4) + 1]), "r"((uint32_t)(_SFA_slot_reg_9[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_18[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[44]), "+f"(accum[(44) + 1]), "+f"(accum[(44) + 2]), "+f"(accum[(44) + 3])
                        : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[6]), "r"(b_frag[(6) + 1]), "r"((uint32_t)(_SFA_slot_reg_9[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_19[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[48]), "+f"(accum[(48) + 1]), "+f"(accum[(48) + 2]), "+f"(accum[(48) + 3])
                        : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[8]), "r"(b_frag[(8) + 1]), "r"((uint32_t)(_SFA_slot_reg_9[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_20[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[52]), "+f"(accum[(52) + 1]), "+f"(accum[(52) + 2]), "+f"(accum[(52) + 3])
                        : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[10]), "r"(b_frag[(10) + 1]), "r"((uint32_t)(_SFA_slot_reg_9[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_21[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[56]), "+f"(accum[(56) + 1]), "+f"(accum[(56) + 2]), "+f"(accum[(56) + 3])
                        : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[12]), "r"(b_frag[(12) + 1]), "r"((uint32_t)(_SFA_slot_reg_9[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_22[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[60]), "+f"(accum[(60) + 1]), "+f"(accum[(60) + 2]), "+f"(accum[(60) + 3])
                        : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[14]), "r"(b_frag[(14) + 1]), "r"((uint32_t)(_SFA_slot_reg_9[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_23[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[64]), "+f"(accum[(64) + 1]), "+f"(accum[(64) + 2]), "+f"(accum[(64) + 3])
                        : "r"(a_frag[8]), "r"(a_frag[(8) + 1]), "r"(a_frag[(8) + 2]), "r"(a_frag[(8) + 3]), "r"(b_frag[0]), "r"(b_frag[1]), "r"((uint32_t)(_SFA_slot_reg_10[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_16[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[68]), "+f"(accum[(68) + 1]), "+f"(accum[(68) + 2]), "+f"(accum[(68) + 3])
                        : "r"(a_frag[8]), "r"(a_frag[(8) + 1]), "r"(a_frag[(8) + 2]), "r"(a_frag[(8) + 3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]), "r"((uint32_t)(_SFA_slot_reg_10[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_17[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[72]), "+f"(accum[(72) + 1]), "+f"(accum[(72) + 2]), "+f"(accum[(72) + 3])
                        : "r"(a_frag[8]), "r"(a_frag[(8) + 1]), "r"(a_frag[(8) + 2]), "r"(a_frag[(8) + 3]), "r"(b_frag[4]), "r"(b_frag[(4) + 1]), "r"((uint32_t)(_SFA_slot_reg_10[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_18[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[76]), "+f"(accum[(76) + 1]), "+f"(accum[(76) + 2]), "+f"(accum[(76) + 3])
                        : "r"(a_frag[8]), "r"(a_frag[(8) + 1]), "r"(a_frag[(8) + 2]), "r"(a_frag[(8) + 3]), "r"(b_frag[6]), "r"(b_frag[(6) + 1]), "r"((uint32_t)(_SFA_slot_reg_10[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_19[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[80]), "+f"(accum[(80) + 1]), "+f"(accum[(80) + 2]), "+f"(accum[(80) + 3])
                        : "r"(a_frag[8]), "r"(a_frag[(8) + 1]), "r"(a_frag[(8) + 2]), "r"(a_frag[(8) + 3]), "r"(b_frag[8]), "r"(b_frag[(8) + 1]), "r"((uint32_t)(_SFA_slot_reg_10[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_20[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[84]), "+f"(accum[(84) + 1]), "+f"(accum[(84) + 2]), "+f"(accum[(84) + 3])
                        : "r"(a_frag[8]), "r"(a_frag[(8) + 1]), "r"(a_frag[(8) + 2]), "r"(a_frag[(8) + 3]), "r"(b_frag[10]), "r"(b_frag[(10) + 1]), "r"((uint32_t)(_SFA_slot_reg_10[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_21[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[88]), "+f"(accum[(88) + 1]), "+f"(accum[(88) + 2]), "+f"(accum[(88) + 3])
                        : "r"(a_frag[8]), "r"(a_frag[(8) + 1]), "r"(a_frag[(8) + 2]), "r"(a_frag[(8) + 3]), "r"(b_frag[12]), "r"(b_frag[(12) + 1]), "r"((uint32_t)(_SFA_slot_reg_10[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_22[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[92]), "+f"(accum[(92) + 1]), "+f"(accum[(92) + 2]), "+f"(accum[(92) + 3])
                        : "r"(a_frag[8]), "r"(a_frag[(8) + 1]), "r"(a_frag[(8) + 2]), "r"(a_frag[(8) + 3]), "r"(b_frag[14]), "r"(b_frag[(14) + 1]), "r"((uint32_t)(_SFA_slot_reg_10[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_23[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[96]), "+f"(accum[(96) + 1]), "+f"(accum[(96) + 2]), "+f"(accum[(96) + 3])
                        : "r"(a_frag[12]), "r"(a_frag[(12) + 1]), "r"(a_frag[(12) + 2]), "r"(a_frag[(12) + 3]), "r"(b_frag[0]), "r"(b_frag[1]), "r"((uint32_t)(_SFA_slot_reg_11[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_16[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[100]), "+f"(accum[(100) + 1]), "+f"(accum[(100) + 2]), "+f"(accum[(100) + 3])
                        : "r"(a_frag[12]), "r"(a_frag[(12) + 1]), "r"(a_frag[(12) + 2]), "r"(a_frag[(12) + 3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]), "r"((uint32_t)(_SFA_slot_reg_11[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_17[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[104]), "+f"(accum[(104) + 1]), "+f"(accum[(104) + 2]), "+f"(accum[(104) + 3])
                        : "r"(a_frag[12]), "r"(a_frag[(12) + 1]), "r"(a_frag[(12) + 2]), "r"(a_frag[(12) + 3]), "r"(b_frag[4]), "r"(b_frag[(4) + 1]), "r"((uint32_t)(_SFA_slot_reg_11[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_18[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[108]), "+f"(accum[(108) + 1]), "+f"(accum[(108) + 2]), "+f"(accum[(108) + 3])
                        : "r"(a_frag[12]), "r"(a_frag[(12) + 1]), "r"(a_frag[(12) + 2]), "r"(a_frag[(12) + 3]), "r"(b_frag[6]), "r"(b_frag[(6) + 1]), "r"((uint32_t)(_SFA_slot_reg_11[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_19[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[112]), "+f"(accum[(112) + 1]), "+f"(accum[(112) + 2]), "+f"(accum[(112) + 3])
                        : "r"(a_frag[12]), "r"(a_frag[(12) + 1]), "r"(a_frag[(12) + 2]), "r"(a_frag[(12) + 3]), "r"(b_frag[8]), "r"(b_frag[(8) + 1]), "r"((uint32_t)(_SFA_slot_reg_11[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_20[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[116]), "+f"(accum[(116) + 1]), "+f"(accum[(116) + 2]), "+f"(accum[(116) + 3])
                        : "r"(a_frag[12]), "r"(a_frag[(12) + 1]), "r"(a_frag[(12) + 2]), "r"(a_frag[(12) + 3]), "r"(b_frag[10]), "r"(b_frag[(10) + 1]), "r"((uint32_t)(_SFA_slot_reg_11[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_21[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[120]), "+f"(accum[(120) + 1]), "+f"(accum[(120) + 2]), "+f"(accum[(120) + 3])
                        : "r"(a_frag[12]), "r"(a_frag[(12) + 1]), "r"(a_frag[(12) + 2]), "r"(a_frag[(12) + 3]), "r"(b_frag[12]), "r"(b_frag[(12) + 1]), "r"((uint32_t)(_SFA_slot_reg_11[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_22[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[124]), "+f"(accum[(124) + 1]), "+f"(accum[(124) + 2]), "+f"(accum[(124) + 3])
                        : "r"(a_frag[12]), "r"(a_frag[(12) + 1]), "r"(a_frag[(12) + 2]), "r"(a_frag[(12) + 3]), "r"(b_frag[14]), "r"(b_frag[(14) + 1]), "r"((uint32_t)(_SFA_slot_reg_11[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_23[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                        : "r"(A_stage_addr + mma_stage * 16384 + (unsigned int)((warp_m * 64 + (lane >> 3 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(32 + (lane >> 3 >> 1) * 16 ^ (warp_m * 64 + (lane >> 3 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_frag[4]), "=r"(a_frag[5]), "=r"(a_frag[6]), "=r"(a_frag[7])
                        : "r"(A_stage_addr + mma_stage * 16384 + (unsigned int)((warp_m * 64 + 16 + (lane >> 3 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(32 + (lane >> 3 >> 1) * 16 ^ (warp_m * 64 + 16 + (lane >> 3 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_frag[8]), "=r"(a_frag[9]), "=r"(a_frag[10]), "=r"(a_frag[11])
                        : "r"(A_stage_addr + mma_stage * 16384 + (unsigned int)((warp_m * 64 + 32 + (lane >> 3 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(32 + (lane >> 3 >> 1) * 16 ^ (warp_m * 64 + 32 + (lane >> 3 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_frag[12]), "=r"(a_frag[13]), "=r"(a_frag[14]), "=r"(a_frag[15])
                        : "r"(A_stage_addr + mma_stage * 16384 + (unsigned int)((warp_m * 64 + 48 + (lane >> 3 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(32 + (lane >> 3 >> 1) * 16 ^ (warp_m * 64 + 48 + (lane >> 3 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                        : "r"(B_stage_addr + mma_stage * 8192 + (unsigned int)((warp_n * 64 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(32 + (lane >> 3 & 1) * 16 ^ (warp_n * 64 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_frag[4]), "=r"(b_frag[5]), "=r"(b_frag[6]), "=r"(b_frag[7])
                        : "r"(B_stage_addr + mma_stage * 8192 + (unsigned int)((warp_n * 64 + 16 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(32 + (lane >> 3 & 1) * 16 ^ (warp_n * 64 + 16 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_frag[8]), "=r"(b_frag[9]), "=r"(b_frag[10]), "=r"(b_frag[11])
                        : "r"(B_stage_addr + mma_stage * 8192 + (unsigned int)((warp_n * 64 + 32 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(32 + (lane >> 3 & 1) * 16 ^ (warp_n * 64 + 32 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_frag[12]), "=r"(b_frag[13]), "=r"(b_frag[14]), "=r"(b_frag[15])
                        : "r"(B_stage_addr + mma_stage * 8192 + (unsigned int)((warp_n * 64 + 48 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(32 + (lane >> 3 & 1) * 16 ^ (warp_n * 64 + 48 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    unsigned int _SFA_slot_reg_12[1];
                    {
                        const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(SFA_slot);
                        #pragma unroll
                        for (int _lr = 0; _lr < 1; _lr++)
                            _SFA_slot_reg_12[_lr] = _smem_ptr[((kp_1 * 2 + 1 >> 1 & 1) * 1024 + (kp_1 * 2 + 1 & 1) * 2 + 1 + (warp_m * 64 + (lane & 1) * 8 + (lane >> 2)) * 4) + _lr];
                    }
                    unsigned int _SFA_slot_reg_13[1];
                    {
                        const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(SFA_slot);
                        #pragma unroll
                        for (int _lr = 0; _lr < 1; _lr++)
                            _SFA_slot_reg_13[_lr] = _smem_ptr[((kp_1 * 2 + 1 >> 1 & 1) * 1024 + (kp_1 * 2 + 1 & 1) * 2 + 1 + (warp_m * 64 + 16 + (lane & 1) * 8 + (lane >> 2)) * 4) + _lr];
                    }
                    unsigned int _SFA_slot_reg_14[1];
                    {
                        const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(SFA_slot);
                        #pragma unroll
                        for (int _lr = 0; _lr < 1; _lr++)
                            _SFA_slot_reg_14[_lr] = _smem_ptr[((kp_1 * 2 + 1 >> 1 & 1) * 1024 + (kp_1 * 2 + 1 & 1) * 2 + 1 + (warp_m * 64 + 32 + (lane & 1) * 8 + (lane >> 2)) * 4) + _lr];
                    }
                    unsigned int _SFA_slot_reg_15[1];
                    {
                        const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(SFA_slot);
                        #pragma unroll
                        for (int _lr = 0; _lr < 1; _lr++)
                            _SFA_slot_reg_15[_lr] = _smem_ptr[((kp_1 * 2 + 1 >> 1 & 1) * 1024 + (kp_1 * 2 + 1 & 1) * 2 + 1 + (warp_m * 64 + 48 + (lane & 1) * 8 + (lane >> 2)) * 4) + _lr];
                    }
                    unsigned int _SFB_stage_reg_24[1];
                    {
                        const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(SFB_stage);
                        #pragma unroll
                        for (int _lr = 0; _lr < 1; _lr++)
                            _SFB_stage_reg_24[_lr] = _smem_ptr[(mma_stage * 256 + 128 + (unsigned int)((lane >> 2) * 4) + (unsigned int)(warp_n * 2)) + _lr];
                    }
                    unsigned int _SFB_stage_reg_25[1];
                    {
                        const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(SFB_stage);
                        #pragma unroll
                        for (int _lr = 0; _lr < 1; _lr++)
                            _SFB_stage_reg_25[_lr] = _smem_ptr[(mma_stage * 256 + 128 + (unsigned int)((8 + (lane >> 2)) * 4) + (unsigned int)(warp_n * 2)) + _lr];
                    }
                    unsigned int _SFB_stage_reg_26[1];
                    {
                        const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(SFB_stage);
                        #pragma unroll
                        for (int _lr = 0; _lr < 1; _lr++)
                            _SFB_stage_reg_26[_lr] = _smem_ptr[(mma_stage * 256 + 128 + (unsigned int)((16 + (lane >> 2)) * 4) + (unsigned int)(warp_n * 2)) + _lr];
                    }
                    unsigned int _SFB_stage_reg_27[1];
                    {
                        const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(SFB_stage);
                        #pragma unroll
                        for (int _lr = 0; _lr < 1; _lr++)
                            _SFB_stage_reg_27[_lr] = _smem_ptr[(mma_stage * 256 + 128 + (unsigned int)((24 + (lane >> 2)) * 4) + (unsigned int)(warp_n * 2)) + _lr];
                    }
                    unsigned int _SFB_stage_reg_28[1];
                    {
                        const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(SFB_stage);
                        #pragma unroll
                        for (int _lr = 0; _lr < 1; _lr++)
                            _SFB_stage_reg_28[_lr] = _smem_ptr[(mma_stage * 256 + 128 + (unsigned int)((lane >> 2) * 4) + (unsigned int)(warp_n * 2 + 1)) + _lr];
                    }
                    unsigned int _SFB_stage_reg_29[1];
                    {
                        const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(SFB_stage);
                        #pragma unroll
                        for (int _lr = 0; _lr < 1; _lr++)
                            _SFB_stage_reg_29[_lr] = _smem_ptr[(mma_stage * 256 + 128 + (unsigned int)((8 + (lane >> 2)) * 4) + (unsigned int)(warp_n * 2 + 1)) + _lr];
                    }
                    unsigned int _SFB_stage_reg_30[1];
                    {
                        const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(SFB_stage);
                        #pragma unroll
                        for (int _lr = 0; _lr < 1; _lr++)
                            _SFB_stage_reg_30[_lr] = _smem_ptr[(mma_stage * 256 + 128 + (unsigned int)((16 + (lane >> 2)) * 4) + (unsigned int)(warp_n * 2 + 1)) + _lr];
                    }
                    unsigned int _SFB_stage_reg_31[1];
                    {
                        const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(SFB_stage);
                        #pragma unroll
                        for (int _lr = 0; _lr < 1; _lr++)
                            _SFB_stage_reg_31[_lr] = _smem_ptr[(mma_stage * 256 + 128 + (unsigned int)((24 + (lane >> 2)) * 4) + (unsigned int)(warp_n * 2 + 1)) + _lr];
                    }
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[0]), "+f"(accum[1]), "+f"(accum[2]), "+f"(accum[3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]), "r"((uint32_t)(_SFA_slot_reg_12[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_24[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[4]), "+f"(accum[(4) + 1]), "+f"(accum[(4) + 2]), "+f"(accum[(4) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]), "r"((uint32_t)(_SFA_slot_reg_12[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_25[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[8]), "+f"(accum[(8) + 1]), "+f"(accum[(8) + 2]), "+f"(accum[(8) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[4]), "r"(b_frag[(4) + 1]), "r"((uint32_t)(_SFA_slot_reg_12[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_26[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[12]), "+f"(accum[(12) + 1]), "+f"(accum[(12) + 2]), "+f"(accum[(12) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[6]), "r"(b_frag[(6) + 1]), "r"((uint32_t)(_SFA_slot_reg_12[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_27[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[16]), "+f"(accum[(16) + 1]), "+f"(accum[(16) + 2]), "+f"(accum[(16) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[8]), "r"(b_frag[(8) + 1]), "r"((uint32_t)(_SFA_slot_reg_12[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_28[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[20]), "+f"(accum[(20) + 1]), "+f"(accum[(20) + 2]), "+f"(accum[(20) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[10]), "r"(b_frag[(10) + 1]), "r"((uint32_t)(_SFA_slot_reg_12[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_29[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[24]), "+f"(accum[(24) + 1]), "+f"(accum[(24) + 2]), "+f"(accum[(24) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[12]), "r"(b_frag[(12) + 1]), "r"((uint32_t)(_SFA_slot_reg_12[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_30[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[28]), "+f"(accum[(28) + 1]), "+f"(accum[(28) + 2]), "+f"(accum[(28) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[14]), "r"(b_frag[(14) + 1]), "r"((uint32_t)(_SFA_slot_reg_12[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_31[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[32]), "+f"(accum[(32) + 1]), "+f"(accum[(32) + 2]), "+f"(accum[(32) + 3])
                        : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[0]), "r"(b_frag[1]), "r"((uint32_t)(_SFA_slot_reg_13[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_24[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[36]), "+f"(accum[(36) + 1]), "+f"(accum[(36) + 2]), "+f"(accum[(36) + 3])
                        : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]), "r"((uint32_t)(_SFA_slot_reg_13[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_25[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[40]), "+f"(accum[(40) + 1]), "+f"(accum[(40) + 2]), "+f"(accum[(40) + 3])
                        : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[4]), "r"(b_frag[(4) + 1]), "r"((uint32_t)(_SFA_slot_reg_13[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_26[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[44]), "+f"(accum[(44) + 1]), "+f"(accum[(44) + 2]), "+f"(accum[(44) + 3])
                        : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[6]), "r"(b_frag[(6) + 1]), "r"((uint32_t)(_SFA_slot_reg_13[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_27[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[48]), "+f"(accum[(48) + 1]), "+f"(accum[(48) + 2]), "+f"(accum[(48) + 3])
                        : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[8]), "r"(b_frag[(8) + 1]), "r"((uint32_t)(_SFA_slot_reg_13[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_28[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[52]), "+f"(accum[(52) + 1]), "+f"(accum[(52) + 2]), "+f"(accum[(52) + 3])
                        : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[10]), "r"(b_frag[(10) + 1]), "r"((uint32_t)(_SFA_slot_reg_13[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_29[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[56]), "+f"(accum[(56) + 1]), "+f"(accum[(56) + 2]), "+f"(accum[(56) + 3])
                        : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[12]), "r"(b_frag[(12) + 1]), "r"((uint32_t)(_SFA_slot_reg_13[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_30[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[60]), "+f"(accum[(60) + 1]), "+f"(accum[(60) + 2]), "+f"(accum[(60) + 3])
                        : "r"(a_frag[4]), "r"(a_frag[(4) + 1]), "r"(a_frag[(4) + 2]), "r"(a_frag[(4) + 3]), "r"(b_frag[14]), "r"(b_frag[(14) + 1]), "r"((uint32_t)(_SFA_slot_reg_13[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_31[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[64]), "+f"(accum[(64) + 1]), "+f"(accum[(64) + 2]), "+f"(accum[(64) + 3])
                        : "r"(a_frag[8]), "r"(a_frag[(8) + 1]), "r"(a_frag[(8) + 2]), "r"(a_frag[(8) + 3]), "r"(b_frag[0]), "r"(b_frag[1]), "r"((uint32_t)(_SFA_slot_reg_14[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_24[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[68]), "+f"(accum[(68) + 1]), "+f"(accum[(68) + 2]), "+f"(accum[(68) + 3])
                        : "r"(a_frag[8]), "r"(a_frag[(8) + 1]), "r"(a_frag[(8) + 2]), "r"(a_frag[(8) + 3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]), "r"((uint32_t)(_SFA_slot_reg_14[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_25[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[72]), "+f"(accum[(72) + 1]), "+f"(accum[(72) + 2]), "+f"(accum[(72) + 3])
                        : "r"(a_frag[8]), "r"(a_frag[(8) + 1]), "r"(a_frag[(8) + 2]), "r"(a_frag[(8) + 3]), "r"(b_frag[4]), "r"(b_frag[(4) + 1]), "r"((uint32_t)(_SFA_slot_reg_14[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_26[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[76]), "+f"(accum[(76) + 1]), "+f"(accum[(76) + 2]), "+f"(accum[(76) + 3])
                        : "r"(a_frag[8]), "r"(a_frag[(8) + 1]), "r"(a_frag[(8) + 2]), "r"(a_frag[(8) + 3]), "r"(b_frag[6]), "r"(b_frag[(6) + 1]), "r"((uint32_t)(_SFA_slot_reg_14[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_27[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[80]), "+f"(accum[(80) + 1]), "+f"(accum[(80) + 2]), "+f"(accum[(80) + 3])
                        : "r"(a_frag[8]), "r"(a_frag[(8) + 1]), "r"(a_frag[(8) + 2]), "r"(a_frag[(8) + 3]), "r"(b_frag[8]), "r"(b_frag[(8) + 1]), "r"((uint32_t)(_SFA_slot_reg_14[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_28[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[84]), "+f"(accum[(84) + 1]), "+f"(accum[(84) + 2]), "+f"(accum[(84) + 3])
                        : "r"(a_frag[8]), "r"(a_frag[(8) + 1]), "r"(a_frag[(8) + 2]), "r"(a_frag[(8) + 3]), "r"(b_frag[10]), "r"(b_frag[(10) + 1]), "r"((uint32_t)(_SFA_slot_reg_14[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_29[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[88]), "+f"(accum[(88) + 1]), "+f"(accum[(88) + 2]), "+f"(accum[(88) + 3])
                        : "r"(a_frag[8]), "r"(a_frag[(8) + 1]), "r"(a_frag[(8) + 2]), "r"(a_frag[(8) + 3]), "r"(b_frag[12]), "r"(b_frag[(12) + 1]), "r"((uint32_t)(_SFA_slot_reg_14[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_30[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[92]), "+f"(accum[(92) + 1]), "+f"(accum[(92) + 2]), "+f"(accum[(92) + 3])
                        : "r"(a_frag[8]), "r"(a_frag[(8) + 1]), "r"(a_frag[(8) + 2]), "r"(a_frag[(8) + 3]), "r"(b_frag[14]), "r"(b_frag[(14) + 1]), "r"((uint32_t)(_SFA_slot_reg_14[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_31[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[96]), "+f"(accum[(96) + 1]), "+f"(accum[(96) + 2]), "+f"(accum[(96) + 3])
                        : "r"(a_frag[12]), "r"(a_frag[(12) + 1]), "r"(a_frag[(12) + 2]), "r"(a_frag[(12) + 3]), "r"(b_frag[0]), "r"(b_frag[1]), "r"((uint32_t)(_SFA_slot_reg_15[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_24[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[100]), "+f"(accum[(100) + 1]), "+f"(accum[(100) + 2]), "+f"(accum[(100) + 3])
                        : "r"(a_frag[12]), "r"(a_frag[(12) + 1]), "r"(a_frag[(12) + 2]), "r"(a_frag[(12) + 3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]), "r"((uint32_t)(_SFA_slot_reg_15[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_25[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[104]), "+f"(accum[(104) + 1]), "+f"(accum[(104) + 2]), "+f"(accum[(104) + 3])
                        : "r"(a_frag[12]), "r"(a_frag[(12) + 1]), "r"(a_frag[(12) + 2]), "r"(a_frag[(12) + 3]), "r"(b_frag[4]), "r"(b_frag[(4) + 1]), "r"((uint32_t)(_SFA_slot_reg_15[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_26[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[108]), "+f"(accum[(108) + 1]), "+f"(accum[(108) + 2]), "+f"(accum[(108) + 3])
                        : "r"(a_frag[12]), "r"(a_frag[(12) + 1]), "r"(a_frag[(12) + 2]), "r"(a_frag[(12) + 3]), "r"(b_frag[6]), "r"(b_frag[(6) + 1]), "r"((uint32_t)(_SFA_slot_reg_15[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_27[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[112]), "+f"(accum[(112) + 1]), "+f"(accum[(112) + 2]), "+f"(accum[(112) + 3])
                        : "r"(a_frag[12]), "r"(a_frag[(12) + 1]), "r"(a_frag[(12) + 2]), "r"(a_frag[(12) + 3]), "r"(b_frag[8]), "r"(b_frag[(8) + 1]), "r"((uint32_t)(_SFA_slot_reg_15[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_28[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[116]), "+f"(accum[(116) + 1]), "+f"(accum[(116) + 2]), "+f"(accum[(116) + 3])
                        : "r"(a_frag[12]), "r"(a_frag[(12) + 1]), "r"(a_frag[(12) + 2]), "r"(a_frag[(12) + 3]), "r"(b_frag[10]), "r"(b_frag[(10) + 1]), "r"((uint32_t)(_SFA_slot_reg_15[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_29[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[120]), "+f"(accum[(120) + 1]), "+f"(accum[(120) + 2]), "+f"(accum[(120) + 3])
                        : "r"(a_frag[12]), "r"(a_frag[(12) + 1]), "r"(a_frag[(12) + 2]), "r"(a_frag[(12) + 3]), "r"(b_frag[12]), "r"(b_frag[(12) + 1]), "r"((uint32_t)(_SFA_slot_reg_15[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_30[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(accum[124]), "+f"(accum[(124) + 1]), "+f"(accum[(124) + 2]), "+f"(accum[(124) + 3])
                        : "r"(a_frag[12]), "r"(a_frag[(12) + 1]), "r"(a_frag[(12) + 2]), "r"(a_frag[(12) + 3]), "r"(b_frag[14]), "r"(b_frag[(14) + 1]), "r"((uint32_t)(_SFA_slot_reg_15[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(_SFB_stage_reg_31[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    __syncwarp();
                    if (elect_sync()) {
                        mbarrier_arrive(ab_empty_addr + (mma_stage) * 8);
                    }
                    mma_stage += 1;
                    if (mma_stage == 3) { mma_stage = 0; _phase_ab_full ^= 1; }
                }
                if (tile_m_1 > 2147483647) {
                    float chk = 0.0f;
                    chk = chk + accum[0];
                    chk = chk + accum[1];
                    chk = chk + accum[2];
                    chk = chk + accum[3];
                    chk = chk + accum[4];
                    chk = chk + accum[5];
                    chk = chk + accum[6];
                    chk = chk + accum[7];
                    chk = chk + accum[8];
                    chk = chk + accum[9];
                    chk = chk + accum[10];
                    chk = chk + accum[11];
                    chk = chk + accum[12];
                    chk = chk + accum[13];
                    chk = chk + accum[14];
                    chk = chk + accum[15];
                    chk = chk + accum[16];
                    chk = chk + accum[17];
                    chk = chk + accum[18];
                    chk = chk + accum[19];
                    chk = chk + accum[20];
                    chk = chk + accum[21];
                    chk = chk + accum[22];
                    chk = chk + accum[23];
                    chk = chk + accum[24];
                    chk = chk + accum[25];
                    chk = chk + accum[26];
                    chk = chk + accum[27];
                    chk = chk + accum[28];
                    chk = chk + accum[29];
                    chk = chk + accum[30];
                    chk = chk + accum[31];
                    chk = chk + accum[32];
                    chk = chk + accum[33];
                    chk = chk + accum[34];
                    chk = chk + accum[35];
                    chk = chk + accum[36];
                    chk = chk + accum[37];
                    chk = chk + accum[38];
                    chk = chk + accum[39];
                    chk = chk + accum[40];
                    chk = chk + accum[41];
                    chk = chk + accum[42];
                    chk = chk + accum[43];
                    chk = chk + accum[44];
                    chk = chk + accum[45];
                    chk = chk + accum[46];
                    chk = chk + accum[47];
                    chk = chk + accum[48];
                    chk = chk + accum[49];
                    chk = chk + accum[50];
                    chk = chk + accum[51];
                    chk = chk + accum[52];
                    chk = chk + accum[53];
                    chk = chk + accum[54];
                    chk = chk + accum[55];
                    chk = chk + accum[56];
                    chk = chk + accum[57];
                    chk = chk + accum[58];
                    chk = chk + accum[59];
                    chk = chk + accum[60];
                    chk = chk + accum[61];
                    chk = chk + accum[62];
                    chk = chk + accum[63];
                    chk = chk + accum[64];
                    chk = chk + accum[65];
                    chk = chk + accum[66];
                    chk = chk + accum[67];
                    chk = chk + accum[68];
                    chk = chk + accum[69];
                    chk = chk + accum[70];
                    chk = chk + accum[71];
                    chk = chk + accum[72];
                    chk = chk + accum[73];
                    chk = chk + accum[74];
                    chk = chk + accum[75];
                    chk = chk + accum[76];
                    chk = chk + accum[77];
                    chk = chk + accum[78];
                    chk = chk + accum[79];
                    chk = chk + accum[80];
                    chk = chk + accum[81];
                    chk = chk + accum[82];
                    chk = chk + accum[83];
                    chk = chk + accum[84];
                    chk = chk + accum[85];
                    chk = chk + accum[86];
                    chk = chk + accum[87];
                    chk = chk + accum[88];
                    chk = chk + accum[89];
                    chk = chk + accum[90];
                    chk = chk + accum[91];
                    chk = chk + accum[92];
                    chk = chk + accum[93];
                    chk = chk + accum[94];
                    chk = chk + accum[95];
                    chk = chk + accum[96];
                    chk = chk + accum[97];
                    chk = chk + accum[98];
                    chk = chk + accum[99];
                    chk = chk + accum[100];
                    chk = chk + accum[101];
                    chk = chk + accum[102];
                    chk = chk + accum[103];
                    chk = chk + accum[104];
                    chk = chk + accum[105];
                    chk = chk + accum[106];
                    chk = chk + accum[107];
                    chk = chk + accum[108];
                    chk = chk + accum[109];
                    chk = chk + accum[110];
                    chk = chk + accum[111];
                    chk = chk + accum[112];
                    chk = chk + accum[113];
                    chk = chk + accum[114];
                    chk = chk + accum[115];
                    chk = chk + accum[116];
                    chk = chk + accum[117];
                    chk = chk + accum[118];
                    chk = chk + accum[119];
                    chk = chk + accum[120];
                    chk = chk + accum[121];
                    chk = chk + accum[122];
                    chk = chk + accum[123];
                    chk = chk + accum[124];
                    chk = chk + accum[125];
                    chk = chk + accum[126];
                    chk = chk + accum[127];
                    out[tile_1] = __as_u32(chk);
                }
                int srow_st = warp_m * 8 + (lane & 7);
                int smat = lane >> 3;
                int srow = role_tid >> 3;
                int c = role_tid & 7;
                int col = tile_n_1 * 128 + c * 16;
                int grow_base = tile_m_1 * 256 + (srow >> 3) * 64 + (srow & 7);
                float ops[32];
                int meta[2];
                meta[0] = gate_index[((grow_base < M) ? grow_base : 0)];
                meta[1] = ((meta[0] >= 0 && meta[0] < 9) ? 1 : 0);
                {
                    const uint4* _vptr_0 = reinterpret_cast<const uint4*>(gate + (((meta[1] == 1) ? meta[0] : 0) * 5376 + col) + 0);
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
                                : "=f"((&ops[0 + _blk * 8 + _pair * 2])[0]), "=f"((&ops[0 + _blk * 8 + _pair * 2])[1])
                                : "r"(_vpairs_0[_pair]));
                        }
                    }
                }
                {
                    const uint4* _vptr_1 = reinterpret_cast<const uint4*>(gate + (((meta[1] == 1) ? meta[0] : 0) * 5376 + col + 8) + 0);
                    uint4 _vld_1[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_1[_blk] = _vptr_1[_blk];
                        uint32_t* _vpairs_1 = reinterpret_cast<uint32_t*>(&_vld_1[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            asm volatile(
                                "{\n\t"
                                "shl.b32 %0, %2, 16;\n\t"
                                "and.b32 %1, %2, 0xffff0000;\n\t"
                                "}\n"
                                : "=f"((&ops[8 + _blk * 8 + _pair * 2])[0]), "=f"((&ops[8 + _blk * 8 + _pair * 2])[1])
                                : "r"(_vpairs_1[_pair]));
                        }
                    }
                }
                {
                    const uint4* _vptr_2 = reinterpret_cast<const uint4*>(residual + (((grow_base < M) ? grow_base : 0) * 5376 + col) + 0);
                    uint4 _vld_2[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_2[_blk] = _vptr_2[_blk];
                        uint32_t* _vpairs_2 = reinterpret_cast<uint32_t*>(&_vld_2[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            asm volatile(
                                "{\n\t"
                                "shl.b32 %0, %2, 16;\n\t"
                                "and.b32 %1, %2, 0xffff0000;\n\t"
                                "}\n"
                                : "=f"((&ops[16 + _blk * 8 + _pair * 2])[0]), "=f"((&ops[16 + _blk * 8 + _pair * 2])[1])
                                : "r"(_vpairs_2[_pair]));
                        }
                    }
                }
                {
                    const uint4* _vptr_3 = reinterpret_cast<const uint4*>(residual + (((grow_base < M) ? grow_base : 0) * 5376 + col + 8) + 0);
                    uint4 _vld_3[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_3[_blk] = _vptr_3[_blk];
                        uint32_t* _vpairs_3 = reinterpret_cast<uint32_t*>(&_vld_3[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            asm volatile(
                                "{\n\t"
                                "shl.b32 %0, %2, 16;\n\t"
                                "and.b32 %1, %2, 0xffff0000;\n\t"
                                "}\n"
                                : "=f"((&ops[24 + _blk * 8 + _pair * 2])[0]), "=f"((&ops[24 + _blk * 8 + _pair * 2])[1])
                                : "r"(_vpairs_3[_pair]));
                        }
                    }
                }
                float scaled[16];
                scaled[0] = accum[0] * sa[0] * sw[0];
                scaled[1] = accum[1] * sa[0] * sw[1];
                scaled[2] = accum[4] * sa[0] * sw[2];
                scaled[3] = accum[5] * sa[0] * sw[3];
                scaled[4] = accum[8] * sa[0] * sw[4];
                scaled[5] = accum[9] * sa[0] * sw[5];
                scaled[6] = accum[12] * sa[0] * sw[6];
                scaled[7] = accum[13] * sa[0] * sw[7];
                scaled[8] = accum[16] * sa[0] * sw[8];
                scaled[9] = accum[17] * sa[0] * sw[9];
                scaled[10] = accum[20] * sa[0] * sw[10];
                scaled[11] = accum[21] * sa[0] * sw[11];
                scaled[12] = accum[24] * sa[0] * sw[12];
                scaled[13] = accum[25] * sa[0] * sw[13];
                scaled[14] = accum[28] * sa[0] * sw[14];
                scaled[15] = accum[29] * sa[0] * sw[15];
                uint32_t scaled_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled[_lp*2 + 0], scaled[_lp*2+1 + 0]));
                    scaled_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t _stmatrix_addr_4 = static_cast<uint32_t>(staging_addr + (unsigned int)(srow_st * 256) + (unsigned int)((warp_n * 64 + smat * 8) * 2 ^ (srow_st & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_4), "r"(*reinterpret_cast<const uint32_t*>(&scaled_bf16[0])), "r"(*reinterpret_cast<const uint32_t*>(&scaled_bf16[1])), "r"(*reinterpret_cast<const uint32_t*>(&scaled_bf16[2])), "r"(*reinterpret_cast<const uint32_t*>(&scaled_bf16[3]))
                    : "memory");
                uint32_t _stmatrix_addr_5 = static_cast<uint32_t>(staging_addr + (unsigned int)(srow_st * 256) + (unsigned int)((warp_n * 64 + (4 + smat) * 8) * 2 ^ (srow_st & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_5), "r"(*reinterpret_cast<const uint32_t*>(&scaled_bf16[4])), "r"(*reinterpret_cast<const uint32_t*>(&scaled_bf16[5])), "r"(*reinterpret_cast<const uint32_t*>(&scaled_bf16[6])), "r"(*reinterpret_cast<const uint32_t*>(&scaled_bf16[7]))
                    : "memory");
                asm volatile("barrier.sync 8, 256;" ::: "memory");
                int grow = grow_base;
                if (grow < M) {
                    unsigned int o_words[8];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&o_words[0])), "=r"(*reinterpret_cast<uint32_t*>(&o_words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&o_words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&o_words[(0) + 3]))
                        : "r"(staging_addr + (unsigned int)(srow * 256) + (unsigned int)(c * 32 ^ (srow & 7) << 4)));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&o_words[4])), "=r"(*reinterpret_cast<uint32_t*>(&o_words[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&o_words[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&o_words[(4) + 3]))
                        : "r"(staging_addr + (unsigned int)(srow * 256) + (unsigned int)(c * 32 + 16 ^ (srow & 7) << 4)));
                    float o_words_f32[16];
                    #pragma unroll
                    for (int _pair = 0; _pair < 8; _pair++) {
                        asm volatile(
                            "{\n\t"
                            "shl.b32 %0, %2, 16;\n\t"
                            "and.b32 %1, %2, 0xffff0000;\n\t"
                            "}\n"
                            : "=f"((&o_words_f32[_pair * 2])[0]), "=f"((&o_words_f32[_pair * 2])[1])
                            : "r"(o_words[_pair]));
                    }
                    int valid = meta[1];
                    float p_raw[16];
                    for (int e = 0; e < 8; e++) {
                        float gv0 = ops[e];
                        float gv1 = ops[8 + e];
                        float gs0 = ((valid == 1) ? gv0 : 0.0f);
                        float gs1 = ((valid == 1) ? gv1 : 0.0f);
                        p_raw[e] = gs0 * o_words_f32[e];
                        p_raw[8 + e] = gs1 * o_words_f32[8 + e];
                    }
                    uint32_t p_raw_bf16[8];
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(p_raw[_lp*2 + 0], p_raw[_lp*2+1 + 0]));
                        p_raw_bf16[_lp] = *(uint32_t*)&_bf2;
                    }
                    float p_raw_bf16_f32[16];
                    #pragma unroll
                    for (int _pair = 0; _pair < 8; _pair++) {
                        asm volatile(
                            "{\n\t"
                            "shl.b32 %0, %2, 16;\n\t"
                            "and.b32 %1, %2, 0xffff0000;\n\t"
                            "}\n"
                            : "=f"((&p_raw_bf16_f32[_pair * 2])[0]), "=f"((&p_raw_bf16_f32[_pair * 2])[1])
                            : "r"(p_raw_bf16[_pair]));
                    }
                    float out_raw[16];
                    for (int e_1 = 0; e_1 < 8; e_1++) {
                        float rv0 = ops[16 + e_1];
                        float rv1 = ops[24 + e_1];
                        out_raw[e_1] = rv0 + p_raw_bf16_f32[e_1];
                        out_raw[8 + e_1] = rv1 + p_raw_bf16_f32[8 + e_1];
                    }
                    uint32_t out_raw_bf16[8];
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_raw[_lp*2 + 0], out_raw[_lp*2+1 + 0]));
                        out_raw_bf16[_lp] = *(uint32_t*)&_bf2;
                    }
                    if (grow < M) {
                        reinterpret_cast<int4*>(out + ((grow * 5376 + col) / 2))[0] = reinterpret_cast<int4*>(out_raw_bf16 + 0)[0];
                        reinterpret_cast<int4*>(out + ((grow * 5376 + col) / 2 + 4))[0] = reinterpret_cast<int4*>(out_raw_bf16 + 4)[0];
                    }
                }
                asm volatile("barrier.sync 8, 256;" ::: "memory");
                meta[0] = gate_index[((grow_base + 8 < M) ? grow_base + 8 : 0)];
                meta[1] = ((meta[0] >= 0 && meta[0] < 9) ? 1 : 0);
                {
                    const uint4* _vptr_6 = reinterpret_cast<const uint4*>(gate + (((meta[1] == 1) ? meta[0] : 0) * 5376 + col) + 0);
                    uint4 _vld_6[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_6[_blk] = _vptr_6[_blk];
                        uint32_t* _vpairs_6 = reinterpret_cast<uint32_t*>(&_vld_6[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            asm volatile(
                                "{\n\t"
                                "shl.b32 %0, %2, 16;\n\t"
                                "and.b32 %1, %2, 0xffff0000;\n\t"
                                "}\n"
                                : "=f"((&ops[0 + _blk * 8 + _pair * 2])[0]), "=f"((&ops[0 + _blk * 8 + _pair * 2])[1])
                                : "r"(_vpairs_6[_pair]));
                        }
                    }
                }
                {
                    const uint4* _vptr_7 = reinterpret_cast<const uint4*>(gate + (((meta[1] == 1) ? meta[0] : 0) * 5376 + col + 8) + 0);
                    uint4 _vld_7[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_7[_blk] = _vptr_7[_blk];
                        uint32_t* _vpairs_7 = reinterpret_cast<uint32_t*>(&_vld_7[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            asm volatile(
                                "{\n\t"
                                "shl.b32 %0, %2, 16;\n\t"
                                "and.b32 %1, %2, 0xffff0000;\n\t"
                                "}\n"
                                : "=f"((&ops[8 + _blk * 8 + _pair * 2])[0]), "=f"((&ops[8 + _blk * 8 + _pair * 2])[1])
                                : "r"(_vpairs_7[_pair]));
                        }
                    }
                }
                {
                    const uint4* _vptr_8 = reinterpret_cast<const uint4*>(residual + (((grow_base + 8 < M) ? grow_base + 8 : 0) * 5376 + col) + 0);
                    uint4 _vld_8[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_8[_blk] = _vptr_8[_blk];
                        uint32_t* _vpairs_8 = reinterpret_cast<uint32_t*>(&_vld_8[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            asm volatile(
                                "{\n\t"
                                "shl.b32 %0, %2, 16;\n\t"
                                "and.b32 %1, %2, 0xffff0000;\n\t"
                                "}\n"
                                : "=f"((&ops[16 + _blk * 8 + _pair * 2])[0]), "=f"((&ops[16 + _blk * 8 + _pair * 2])[1])
                                : "r"(_vpairs_8[_pair]));
                        }
                    }
                }
                {
                    const uint4* _vptr_9 = reinterpret_cast<const uint4*>(residual + (((grow_base + 8 < M) ? grow_base + 8 : 0) * 5376 + col + 8) + 0);
                    uint4 _vld_9[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_9[_blk] = _vptr_9[_blk];
                        uint32_t* _vpairs_9 = reinterpret_cast<uint32_t*>(&_vld_9[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            asm volatile(
                                "{\n\t"
                                "shl.b32 %0, %2, 16;\n\t"
                                "and.b32 %1, %2, 0xffff0000;\n\t"
                                "}\n"
                                : "=f"((&ops[24 + _blk * 8 + _pair * 2])[0]), "=f"((&ops[24 + _blk * 8 + _pair * 2])[1])
                                : "r"(_vpairs_9[_pair]));
                        }
                    }
                }
                float scaled_0[16];
                scaled_0[0] = accum[2] * sa[1] * sw[0];
                scaled_0[1] = accum[3] * sa[1] * sw[1];
                scaled_0[2] = accum[6] * sa[1] * sw[2];
                scaled_0[3] = accum[7] * sa[1] * sw[3];
                scaled_0[4] = accum[10] * sa[1] * sw[4];
                scaled_0[5] = accum[11] * sa[1] * sw[5];
                scaled_0[6] = accum[14] * sa[1] * sw[6];
                scaled_0[7] = accum[15] * sa[1] * sw[7];
                scaled_0[8] = accum[18] * sa[1] * sw[8];
                scaled_0[9] = accum[19] * sa[1] * sw[9];
                scaled_0[10] = accum[22] * sa[1] * sw[10];
                scaled_0[11] = accum[23] * sa[1] * sw[11];
                scaled_0[12] = accum[26] * sa[1] * sw[12];
                scaled_0[13] = accum[27] * sa[1] * sw[13];
                scaled_0[14] = accum[30] * sa[1] * sw[14];
                scaled_0[15] = accum[31] * sa[1] * sw[15];
                uint32_t scaled_0_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_0[_lp*2 + 0], scaled_0[_lp*2+1 + 0]));
                    scaled_0_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t _stmatrix_addr_10 = static_cast<uint32_t>(staging_addr + (unsigned int)(srow_st * 256) + (unsigned int)((warp_n * 64 + smat * 8) * 2 ^ (srow_st & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_10), "r"(*reinterpret_cast<const uint32_t*>(&scaled_0_bf16[0])), "r"(*reinterpret_cast<const uint32_t*>(&scaled_0_bf16[1])), "r"(*reinterpret_cast<const uint32_t*>(&scaled_0_bf16[2])), "r"(*reinterpret_cast<const uint32_t*>(&scaled_0_bf16[3]))
                    : "memory");
                uint32_t _stmatrix_addr_11 = static_cast<uint32_t>(staging_addr + (unsigned int)(srow_st * 256) + (unsigned int)((warp_n * 64 + (4 + smat) * 8) * 2 ^ (srow_st & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_11), "r"(*reinterpret_cast<const uint32_t*>(&scaled_0_bf16[4])), "r"(*reinterpret_cast<const uint32_t*>(&scaled_0_bf16[5])), "r"(*reinterpret_cast<const uint32_t*>(&scaled_0_bf16[6])), "r"(*reinterpret_cast<const uint32_t*>(&scaled_0_bf16[7]))
                    : "memory");
                asm volatile("barrier.sync 8, 256;" ::: "memory");
                int grow_1 = grow_base + 8;
                if (grow_1 < M) {
                    unsigned int o_words_1[8];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&o_words_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&o_words_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&o_words_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&o_words_1[(0) + 3]))
                        : "r"(staging_addr + (unsigned int)(srow * 256) + (unsigned int)(c * 32 ^ (srow & 7) << 4)));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&o_words_1[4])), "=r"(*reinterpret_cast<uint32_t*>(&o_words_1[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&o_words_1[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&o_words_1[(4) + 3]))
                        : "r"(staging_addr + (unsigned int)(srow * 256) + (unsigned int)(c * 32 + 16 ^ (srow & 7) << 4)));
                    float o_words_f32_1[16];
                    #pragma unroll
                    for (int _pair = 0; _pair < 8; _pair++) {
                        asm volatile(
                            "{\n\t"
                            "shl.b32 %0, %2, 16;\n\t"
                            "and.b32 %1, %2, 0xffff0000;\n\t"
                            "}\n"
                            : "=f"((&o_words_f32_1[_pair * 2])[0]), "=f"((&o_words_f32_1[_pair * 2])[1])
                            : "r"(o_words_1[_pair]));
                    }
                    int valid_1 = meta[1];
                    float p_raw_1[16];
                    for (int e_2 = 0; e_2 < 8; e_2++) {
                        float gv0_1 = ops[e_2];
                        float gv1_1 = ops[8 + e_2];
                        float gs0_1 = ((valid_1 == 1) ? gv0_1 : 0.0f);
                        float gs1_1 = ((valid_1 == 1) ? gv1_1 : 0.0f);
                        p_raw_1[e_2] = gs0_1 * o_words_f32_1[e_2];
                        p_raw_1[8 + e_2] = gs1_1 * o_words_f32_1[8 + e_2];
                    }
                    uint32_t p_raw_bf16_1[8];
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(p_raw_1[_lp*2 + 0], p_raw_1[_lp*2+1 + 0]));
                        p_raw_bf16_1[_lp] = *(uint32_t*)&_bf2;
                    }
                    float p_raw_bf16_f32_1[16];
                    #pragma unroll
                    for (int _pair = 0; _pair < 8; _pair++) {
                        asm volatile(
                            "{\n\t"
                            "shl.b32 %0, %2, 16;\n\t"
                            "and.b32 %1, %2, 0xffff0000;\n\t"
                            "}\n"
                            : "=f"((&p_raw_bf16_f32_1[_pair * 2])[0]), "=f"((&p_raw_bf16_f32_1[_pair * 2])[1])
                            : "r"(p_raw_bf16_1[_pair]));
                    }
                    float out_raw_1[16];
                    for (int e_3 = 0; e_3 < 8; e_3++) {
                        float rv0_1 = ops[16 + e_3];
                        float rv1_1 = ops[24 + e_3];
                        out_raw_1[e_3] = rv0_1 + p_raw_bf16_f32_1[e_3];
                        out_raw_1[8 + e_3] = rv1_1 + p_raw_bf16_f32_1[8 + e_3];
                    }
                    uint32_t out_raw_bf16_1[8];
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_raw_1[_lp*2 + 0], out_raw_1[_lp*2+1 + 0]));
                        out_raw_bf16_1[_lp] = *(uint32_t*)&_bf2;
                    }
                    if (grow_1 < M) {
                        reinterpret_cast<int4*>(out + ((grow_1 * 5376 + col) / 2))[0] = reinterpret_cast<int4*>(out_raw_bf16_1 + 0)[0];
                        reinterpret_cast<int4*>(out + ((grow_1 * 5376 + col) / 2 + 4))[0] = reinterpret_cast<int4*>(out_raw_bf16_1 + 4)[0];
                    }
                }
                asm volatile("barrier.sync 8, 256;" ::: "memory");
                meta[0] = gate_index[((grow_base + 16 < M) ? grow_base + 16 : 0)];
                meta[1] = ((meta[0] >= 0 && meta[0] < 9) ? 1 : 0);
                {
                    const uint4* _vptr_12 = reinterpret_cast<const uint4*>(gate + (((meta[1] == 1) ? meta[0] : 0) * 5376 + col) + 0);
                    uint4 _vld_12[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_12[_blk] = _vptr_12[_blk];
                        uint32_t* _vpairs_12 = reinterpret_cast<uint32_t*>(&_vld_12[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            asm volatile(
                                "{\n\t"
                                "shl.b32 %0, %2, 16;\n\t"
                                "and.b32 %1, %2, 0xffff0000;\n\t"
                                "}\n"
                                : "=f"((&ops[0 + _blk * 8 + _pair * 2])[0]), "=f"((&ops[0 + _blk * 8 + _pair * 2])[1])
                                : "r"(_vpairs_12[_pair]));
                        }
                    }
                }
                {
                    const uint4* _vptr_13 = reinterpret_cast<const uint4*>(gate + (((meta[1] == 1) ? meta[0] : 0) * 5376 + col + 8) + 0);
                    uint4 _vld_13[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_13[_blk] = _vptr_13[_blk];
                        uint32_t* _vpairs_13 = reinterpret_cast<uint32_t*>(&_vld_13[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            asm volatile(
                                "{\n\t"
                                "shl.b32 %0, %2, 16;\n\t"
                                "and.b32 %1, %2, 0xffff0000;\n\t"
                                "}\n"
                                : "=f"((&ops[8 + _blk * 8 + _pair * 2])[0]), "=f"((&ops[8 + _blk * 8 + _pair * 2])[1])
                                : "r"(_vpairs_13[_pair]));
                        }
                    }
                }
                {
                    const uint4* _vptr_14 = reinterpret_cast<const uint4*>(residual + (((grow_base + 16 < M) ? grow_base + 16 : 0) * 5376 + col) + 0);
                    uint4 _vld_14[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_14[_blk] = _vptr_14[_blk];
                        uint32_t* _vpairs_14 = reinterpret_cast<uint32_t*>(&_vld_14[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            asm volatile(
                                "{\n\t"
                                "shl.b32 %0, %2, 16;\n\t"
                                "and.b32 %1, %2, 0xffff0000;\n\t"
                                "}\n"
                                : "=f"((&ops[16 + _blk * 8 + _pair * 2])[0]), "=f"((&ops[16 + _blk * 8 + _pair * 2])[1])
                                : "r"(_vpairs_14[_pair]));
                        }
                    }
                }
                {
                    const uint4* _vptr_15 = reinterpret_cast<const uint4*>(residual + (((grow_base + 16 < M) ? grow_base + 16 : 0) * 5376 + col + 8) + 0);
                    uint4 _vld_15[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_15[_blk] = _vptr_15[_blk];
                        uint32_t* _vpairs_15 = reinterpret_cast<uint32_t*>(&_vld_15[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            asm volatile(
                                "{\n\t"
                                "shl.b32 %0, %2, 16;\n\t"
                                "and.b32 %1, %2, 0xffff0000;\n\t"
                                "}\n"
                                : "=f"((&ops[24 + _blk * 8 + _pair * 2])[0]), "=f"((&ops[24 + _blk * 8 + _pair * 2])[1])
                                : "r"(_vpairs_15[_pair]));
                        }
                    }
                }
                float scaled_2[16];
                scaled_2[0] = accum[32] * sa[2] * sw[0];
                scaled_2[1] = accum[33] * sa[2] * sw[1];
                scaled_2[2] = accum[36] * sa[2] * sw[2];
                scaled_2[3] = accum[37] * sa[2] * sw[3];
                scaled_2[4] = accum[40] * sa[2] * sw[4];
                scaled_2[5] = accum[41] * sa[2] * sw[5];
                scaled_2[6] = accum[44] * sa[2] * sw[6];
                scaled_2[7] = accum[45] * sa[2] * sw[7];
                scaled_2[8] = accum[48] * sa[2] * sw[8];
                scaled_2[9] = accum[49] * sa[2] * sw[9];
                scaled_2[10] = accum[52] * sa[2] * sw[10];
                scaled_2[11] = accum[53] * sa[2] * sw[11];
                scaled_2[12] = accum[56] * sa[2] * sw[12];
                scaled_2[13] = accum[57] * sa[2] * sw[13];
                scaled_2[14] = accum[60] * sa[2] * sw[14];
                scaled_2[15] = accum[61] * sa[2] * sw[15];
                uint32_t scaled_2_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_2[_lp*2 + 0], scaled_2[_lp*2+1 + 0]));
                    scaled_2_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t _stmatrix_addr_16 = static_cast<uint32_t>(staging_addr + (unsigned int)(srow_st * 256) + (unsigned int)((warp_n * 64 + smat * 8) * 2 ^ (srow_st & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_16), "r"(*reinterpret_cast<const uint32_t*>(&scaled_2_bf16[0])), "r"(*reinterpret_cast<const uint32_t*>(&scaled_2_bf16[1])), "r"(*reinterpret_cast<const uint32_t*>(&scaled_2_bf16[2])), "r"(*reinterpret_cast<const uint32_t*>(&scaled_2_bf16[3]))
                    : "memory");
                uint32_t _stmatrix_addr_17 = static_cast<uint32_t>(staging_addr + (unsigned int)(srow_st * 256) + (unsigned int)((warp_n * 64 + (4 + smat) * 8) * 2 ^ (srow_st & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_17), "r"(*reinterpret_cast<const uint32_t*>(&scaled_2_bf16[4])), "r"(*reinterpret_cast<const uint32_t*>(&scaled_2_bf16[5])), "r"(*reinterpret_cast<const uint32_t*>(&scaled_2_bf16[6])), "r"(*reinterpret_cast<const uint32_t*>(&scaled_2_bf16[7]))
                    : "memory");
                asm volatile("barrier.sync 8, 256;" ::: "memory");
                int grow_3 = grow_base + 16;
                if (grow_3 < M) {
                    unsigned int o_words_2[8];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&o_words_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&o_words_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&o_words_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&o_words_2[(0) + 3]))
                        : "r"(staging_addr + (unsigned int)(srow * 256) + (unsigned int)(c * 32 ^ (srow & 7) << 4)));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&o_words_2[4])), "=r"(*reinterpret_cast<uint32_t*>(&o_words_2[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&o_words_2[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&o_words_2[(4) + 3]))
                        : "r"(staging_addr + (unsigned int)(srow * 256) + (unsigned int)(c * 32 + 16 ^ (srow & 7) << 4)));
                    float o_words_f32_2[16];
                    #pragma unroll
                    for (int _pair = 0; _pair < 8; _pair++) {
                        asm volatile(
                            "{\n\t"
                            "shl.b32 %0, %2, 16;\n\t"
                            "and.b32 %1, %2, 0xffff0000;\n\t"
                            "}\n"
                            : "=f"((&o_words_f32_2[_pair * 2])[0]), "=f"((&o_words_f32_2[_pair * 2])[1])
                            : "r"(o_words_2[_pair]));
                    }
                    int valid_2 = meta[1];
                    float p_raw_2[16];
                    for (int e_4 = 0; e_4 < 8; e_4++) {
                        float gv0_2 = ops[e_4];
                        float gv1_2 = ops[8 + e_4];
                        float gs0_2 = ((valid_2 == 1) ? gv0_2 : 0.0f);
                        float gs1_2 = ((valid_2 == 1) ? gv1_2 : 0.0f);
                        p_raw_2[e_4] = gs0_2 * o_words_f32_2[e_4];
                        p_raw_2[8 + e_4] = gs1_2 * o_words_f32_2[8 + e_4];
                    }
                    uint32_t p_raw_bf16_2[8];
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(p_raw_2[_lp*2 + 0], p_raw_2[_lp*2+1 + 0]));
                        p_raw_bf16_2[_lp] = *(uint32_t*)&_bf2;
                    }
                    float p_raw_bf16_f32_2[16];
                    #pragma unroll
                    for (int _pair = 0; _pair < 8; _pair++) {
                        asm volatile(
                            "{\n\t"
                            "shl.b32 %0, %2, 16;\n\t"
                            "and.b32 %1, %2, 0xffff0000;\n\t"
                            "}\n"
                            : "=f"((&p_raw_bf16_f32_2[_pair * 2])[0]), "=f"((&p_raw_bf16_f32_2[_pair * 2])[1])
                            : "r"(p_raw_bf16_2[_pair]));
                    }
                    float out_raw_2[16];
                    for (int e_5 = 0; e_5 < 8; e_5++) {
                        float rv0_2 = ops[16 + e_5];
                        float rv1_2 = ops[24 + e_5];
                        out_raw_2[e_5] = rv0_2 + p_raw_bf16_f32_2[e_5];
                        out_raw_2[8 + e_5] = rv1_2 + p_raw_bf16_f32_2[8 + e_5];
                    }
                    uint32_t out_raw_bf16_2[8];
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_raw_2[_lp*2 + 0], out_raw_2[_lp*2+1 + 0]));
                        out_raw_bf16_2[_lp] = *(uint32_t*)&_bf2;
                    }
                    if (grow_3 < M) {
                        reinterpret_cast<int4*>(out + ((grow_3 * 5376 + col) / 2))[0] = reinterpret_cast<int4*>(out_raw_bf16_2 + 0)[0];
                        reinterpret_cast<int4*>(out + ((grow_3 * 5376 + col) / 2 + 4))[0] = reinterpret_cast<int4*>(out_raw_bf16_2 + 4)[0];
                    }
                }
                asm volatile("barrier.sync 8, 256;" ::: "memory");
                meta[0] = gate_index[((grow_base + 24 < M) ? grow_base + 24 : 0)];
                meta[1] = ((meta[0] >= 0 && meta[0] < 9) ? 1 : 0);
                {
                    const uint4* _vptr_18 = reinterpret_cast<const uint4*>(gate + (((meta[1] == 1) ? meta[0] : 0) * 5376 + col) + 0);
                    uint4 _vld_18[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_18[_blk] = _vptr_18[_blk];
                        uint32_t* _vpairs_18 = reinterpret_cast<uint32_t*>(&_vld_18[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            asm volatile(
                                "{\n\t"
                                "shl.b32 %0, %2, 16;\n\t"
                                "and.b32 %1, %2, 0xffff0000;\n\t"
                                "}\n"
                                : "=f"((&ops[0 + _blk * 8 + _pair * 2])[0]), "=f"((&ops[0 + _blk * 8 + _pair * 2])[1])
                                : "r"(_vpairs_18[_pair]));
                        }
                    }
                }
                {
                    const uint4* _vptr_19 = reinterpret_cast<const uint4*>(gate + (((meta[1] == 1) ? meta[0] : 0) * 5376 + col + 8) + 0);
                    uint4 _vld_19[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_19[_blk] = _vptr_19[_blk];
                        uint32_t* _vpairs_19 = reinterpret_cast<uint32_t*>(&_vld_19[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            asm volatile(
                                "{\n\t"
                                "shl.b32 %0, %2, 16;\n\t"
                                "and.b32 %1, %2, 0xffff0000;\n\t"
                                "}\n"
                                : "=f"((&ops[8 + _blk * 8 + _pair * 2])[0]), "=f"((&ops[8 + _blk * 8 + _pair * 2])[1])
                                : "r"(_vpairs_19[_pair]));
                        }
                    }
                }
                {
                    const uint4* _vptr_20 = reinterpret_cast<const uint4*>(residual + (((grow_base + 24 < M) ? grow_base + 24 : 0) * 5376 + col) + 0);
                    uint4 _vld_20[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_20[_blk] = _vptr_20[_blk];
                        uint32_t* _vpairs_20 = reinterpret_cast<uint32_t*>(&_vld_20[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            asm volatile(
                                "{\n\t"
                                "shl.b32 %0, %2, 16;\n\t"
                                "and.b32 %1, %2, 0xffff0000;\n\t"
                                "}\n"
                                : "=f"((&ops[16 + _blk * 8 + _pair * 2])[0]), "=f"((&ops[16 + _blk * 8 + _pair * 2])[1])
                                : "r"(_vpairs_20[_pair]));
                        }
                    }
                }
                {
                    const uint4* _vptr_21 = reinterpret_cast<const uint4*>(residual + (((grow_base + 24 < M) ? grow_base + 24 : 0) * 5376 + col + 8) + 0);
                    uint4 _vld_21[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_21[_blk] = _vptr_21[_blk];
                        uint32_t* _vpairs_21 = reinterpret_cast<uint32_t*>(&_vld_21[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            asm volatile(
                                "{\n\t"
                                "shl.b32 %0, %2, 16;\n\t"
                                "and.b32 %1, %2, 0xffff0000;\n\t"
                                "}\n"
                                : "=f"((&ops[24 + _blk * 8 + _pair * 2])[0]), "=f"((&ops[24 + _blk * 8 + _pair * 2])[1])
                                : "r"(_vpairs_21[_pair]));
                        }
                    }
                }
                float scaled_4[16];
                scaled_4[0] = accum[34] * sa[3] * sw[0];
                scaled_4[1] = accum[35] * sa[3] * sw[1];
                scaled_4[2] = accum[38] * sa[3] * sw[2];
                scaled_4[3] = accum[39] * sa[3] * sw[3];
                scaled_4[4] = accum[42] * sa[3] * sw[4];
                scaled_4[5] = accum[43] * sa[3] * sw[5];
                scaled_4[6] = accum[46] * sa[3] * sw[6];
                scaled_4[7] = accum[47] * sa[3] * sw[7];
                scaled_4[8] = accum[50] * sa[3] * sw[8];
                scaled_4[9] = accum[51] * sa[3] * sw[9];
                scaled_4[10] = accum[54] * sa[3] * sw[10];
                scaled_4[11] = accum[55] * sa[3] * sw[11];
                scaled_4[12] = accum[58] * sa[3] * sw[12];
                scaled_4[13] = accum[59] * sa[3] * sw[13];
                scaled_4[14] = accum[62] * sa[3] * sw[14];
                scaled_4[15] = accum[63] * sa[3] * sw[15];
                uint32_t scaled_4_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_4[_lp*2 + 0], scaled_4[_lp*2+1 + 0]));
                    scaled_4_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t _stmatrix_addr_22 = static_cast<uint32_t>(staging_addr + (unsigned int)(srow_st * 256) + (unsigned int)((warp_n * 64 + smat * 8) * 2 ^ (srow_st & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_22), "r"(*reinterpret_cast<const uint32_t*>(&scaled_4_bf16[0])), "r"(*reinterpret_cast<const uint32_t*>(&scaled_4_bf16[1])), "r"(*reinterpret_cast<const uint32_t*>(&scaled_4_bf16[2])), "r"(*reinterpret_cast<const uint32_t*>(&scaled_4_bf16[3]))
                    : "memory");
                uint32_t _stmatrix_addr_23 = static_cast<uint32_t>(staging_addr + (unsigned int)(srow_st * 256) + (unsigned int)((warp_n * 64 + (4 + smat) * 8) * 2 ^ (srow_st & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_23), "r"(*reinterpret_cast<const uint32_t*>(&scaled_4_bf16[4])), "r"(*reinterpret_cast<const uint32_t*>(&scaled_4_bf16[5])), "r"(*reinterpret_cast<const uint32_t*>(&scaled_4_bf16[6])), "r"(*reinterpret_cast<const uint32_t*>(&scaled_4_bf16[7]))
                    : "memory");
                asm volatile("barrier.sync 8, 256;" ::: "memory");
                int grow_5 = grow_base + 24;
                if (grow_5 < M) {
                    unsigned int o_words_3[8];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&o_words_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&o_words_3[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&o_words_3[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&o_words_3[(0) + 3]))
                        : "r"(staging_addr + (unsigned int)(srow * 256) + (unsigned int)(c * 32 ^ (srow & 7) << 4)));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&o_words_3[4])), "=r"(*reinterpret_cast<uint32_t*>(&o_words_3[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&o_words_3[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&o_words_3[(4) + 3]))
                        : "r"(staging_addr + (unsigned int)(srow * 256) + (unsigned int)(c * 32 + 16 ^ (srow & 7) << 4)));
                    float o_words_f32_3[16];
                    #pragma unroll
                    for (int _pair = 0; _pair < 8; _pair++) {
                        asm volatile(
                            "{\n\t"
                            "shl.b32 %0, %2, 16;\n\t"
                            "and.b32 %1, %2, 0xffff0000;\n\t"
                            "}\n"
                            : "=f"((&o_words_f32_3[_pair * 2])[0]), "=f"((&o_words_f32_3[_pair * 2])[1])
                            : "r"(o_words_3[_pair]));
                    }
                    int valid_3 = meta[1];
                    float p_raw_3[16];
                    for (int e_6 = 0; e_6 < 8; e_6++) {
                        float gv0_3 = ops[e_6];
                        float gv1_3 = ops[8 + e_6];
                        float gs0_3 = ((valid_3 == 1) ? gv0_3 : 0.0f);
                        float gs1_3 = ((valid_3 == 1) ? gv1_3 : 0.0f);
                        p_raw_3[e_6] = gs0_3 * o_words_f32_3[e_6];
                        p_raw_3[8 + e_6] = gs1_3 * o_words_f32_3[8 + e_6];
                    }
                    uint32_t p_raw_bf16_3[8];
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(p_raw_3[_lp*2 + 0], p_raw_3[_lp*2+1 + 0]));
                        p_raw_bf16_3[_lp] = *(uint32_t*)&_bf2;
                    }
                    float p_raw_bf16_f32_3[16];
                    #pragma unroll
                    for (int _pair = 0; _pair < 8; _pair++) {
                        asm volatile(
                            "{\n\t"
                            "shl.b32 %0, %2, 16;\n\t"
                            "and.b32 %1, %2, 0xffff0000;\n\t"
                            "}\n"
                            : "=f"((&p_raw_bf16_f32_3[_pair * 2])[0]), "=f"((&p_raw_bf16_f32_3[_pair * 2])[1])
                            : "r"(p_raw_bf16_3[_pair]));
                    }
                    float out_raw_3[16];
                    for (int e_7 = 0; e_7 < 8; e_7++) {
                        float rv0_3 = ops[16 + e_7];
                        float rv1_3 = ops[24 + e_7];
                        out_raw_3[e_7] = rv0_3 + p_raw_bf16_f32_3[e_7];
                        out_raw_3[8 + e_7] = rv1_3 + p_raw_bf16_f32_3[8 + e_7];
                    }
                    uint32_t out_raw_bf16_3[8];
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_raw_3[_lp*2 + 0], out_raw_3[_lp*2+1 + 0]));
                        out_raw_bf16_3[_lp] = *(uint32_t*)&_bf2;
                    }
                    if (grow_5 < M) {
                        reinterpret_cast<int4*>(out + ((grow_5 * 5376 + col) / 2))[0] = reinterpret_cast<int4*>(out_raw_bf16_3 + 0)[0];
                        reinterpret_cast<int4*>(out + ((grow_5 * 5376 + col) / 2 + 4))[0] = reinterpret_cast<int4*>(out_raw_bf16_3 + 4)[0];
                    }
                }
                asm volatile("barrier.sync 8, 256;" ::: "memory");
                meta[0] = gate_index[((grow_base + 32 < M) ? grow_base + 32 : 0)];
                meta[1] = ((meta[0] >= 0 && meta[0] < 9) ? 1 : 0);
                {
                    const uint4* _vptr_24 = reinterpret_cast<const uint4*>(gate + (((meta[1] == 1) ? meta[0] : 0) * 5376 + col) + 0);
                    uint4 _vld_24[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_24[_blk] = _vptr_24[_blk];
                        uint32_t* _vpairs_24 = reinterpret_cast<uint32_t*>(&_vld_24[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            asm volatile(
                                "{\n\t"
                                "shl.b32 %0, %2, 16;\n\t"
                                "and.b32 %1, %2, 0xffff0000;\n\t"
                                "}\n"
                                : "=f"((&ops[0 + _blk * 8 + _pair * 2])[0]), "=f"((&ops[0 + _blk * 8 + _pair * 2])[1])
                                : "r"(_vpairs_24[_pair]));
                        }
                    }
                }
                {
                    const uint4* _vptr_25 = reinterpret_cast<const uint4*>(gate + (((meta[1] == 1) ? meta[0] : 0) * 5376 + col + 8) + 0);
                    uint4 _vld_25[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_25[_blk] = _vptr_25[_blk];
                        uint32_t* _vpairs_25 = reinterpret_cast<uint32_t*>(&_vld_25[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            asm volatile(
                                "{\n\t"
                                "shl.b32 %0, %2, 16;\n\t"
                                "and.b32 %1, %2, 0xffff0000;\n\t"
                                "}\n"
                                : "=f"((&ops[8 + _blk * 8 + _pair * 2])[0]), "=f"((&ops[8 + _blk * 8 + _pair * 2])[1])
                                : "r"(_vpairs_25[_pair]));
                        }
                    }
                }
                {
                    const uint4* _vptr_26 = reinterpret_cast<const uint4*>(residual + (((grow_base + 32 < M) ? grow_base + 32 : 0) * 5376 + col) + 0);
                    uint4 _vld_26[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_26[_blk] = _vptr_26[_blk];
                        uint32_t* _vpairs_26 = reinterpret_cast<uint32_t*>(&_vld_26[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            asm volatile(
                                "{\n\t"
                                "shl.b32 %0, %2, 16;\n\t"
                                "and.b32 %1, %2, 0xffff0000;\n\t"
                                "}\n"
                                : "=f"((&ops[16 + _blk * 8 + _pair * 2])[0]), "=f"((&ops[16 + _blk * 8 + _pair * 2])[1])
                                : "r"(_vpairs_26[_pair]));
                        }
                    }
                }
                {
                    const uint4* _vptr_27 = reinterpret_cast<const uint4*>(residual + (((grow_base + 32 < M) ? grow_base + 32 : 0) * 5376 + col + 8) + 0);
                    uint4 _vld_27[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_27[_blk] = _vptr_27[_blk];
                        uint32_t* _vpairs_27 = reinterpret_cast<uint32_t*>(&_vld_27[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            asm volatile(
                                "{\n\t"
                                "shl.b32 %0, %2, 16;\n\t"
                                "and.b32 %1, %2, 0xffff0000;\n\t"
                                "}\n"
                                : "=f"((&ops[24 + _blk * 8 + _pair * 2])[0]), "=f"((&ops[24 + _blk * 8 + _pair * 2])[1])
                                : "r"(_vpairs_27[_pair]));
                        }
                    }
                }
                float scaled_6[16];
                scaled_6[0] = accum[64] * sa[4] * sw[0];
                scaled_6[1] = accum[65] * sa[4] * sw[1];
                scaled_6[2] = accum[68] * sa[4] * sw[2];
                scaled_6[3] = accum[69] * sa[4] * sw[3];
                scaled_6[4] = accum[72] * sa[4] * sw[4];
                scaled_6[5] = accum[73] * sa[4] * sw[5];
                scaled_6[6] = accum[76] * sa[4] * sw[6];
                scaled_6[7] = accum[77] * sa[4] * sw[7];
                scaled_6[8] = accum[80] * sa[4] * sw[8];
                scaled_6[9] = accum[81] * sa[4] * sw[9];
                scaled_6[10] = accum[84] * sa[4] * sw[10];
                scaled_6[11] = accum[85] * sa[4] * sw[11];
                scaled_6[12] = accum[88] * sa[4] * sw[12];
                scaled_6[13] = accum[89] * sa[4] * sw[13];
                scaled_6[14] = accum[92] * sa[4] * sw[14];
                scaled_6[15] = accum[93] * sa[4] * sw[15];
                uint32_t scaled_6_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_6[_lp*2 + 0], scaled_6[_lp*2+1 + 0]));
                    scaled_6_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t _stmatrix_addr_28 = static_cast<uint32_t>(staging_addr + (unsigned int)(srow_st * 256) + (unsigned int)((warp_n * 64 + smat * 8) * 2 ^ (srow_st & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_28), "r"(*reinterpret_cast<const uint32_t*>(&scaled_6_bf16[0])), "r"(*reinterpret_cast<const uint32_t*>(&scaled_6_bf16[1])), "r"(*reinterpret_cast<const uint32_t*>(&scaled_6_bf16[2])), "r"(*reinterpret_cast<const uint32_t*>(&scaled_6_bf16[3]))
                    : "memory");
                uint32_t _stmatrix_addr_29 = static_cast<uint32_t>(staging_addr + (unsigned int)(srow_st * 256) + (unsigned int)((warp_n * 64 + (4 + smat) * 8) * 2 ^ (srow_st & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_29), "r"(*reinterpret_cast<const uint32_t*>(&scaled_6_bf16[4])), "r"(*reinterpret_cast<const uint32_t*>(&scaled_6_bf16[5])), "r"(*reinterpret_cast<const uint32_t*>(&scaled_6_bf16[6])), "r"(*reinterpret_cast<const uint32_t*>(&scaled_6_bf16[7]))
                    : "memory");
                asm volatile("barrier.sync 8, 256;" ::: "memory");
                int grow_7 = grow_base + 32;
                if (grow_7 < M) {
                    unsigned int o_words_4[8];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&o_words_4[0])), "=r"(*reinterpret_cast<uint32_t*>(&o_words_4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&o_words_4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&o_words_4[(0) + 3]))
                        : "r"(staging_addr + (unsigned int)(srow * 256) + (unsigned int)(c * 32 ^ (srow & 7) << 4)));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&o_words_4[4])), "=r"(*reinterpret_cast<uint32_t*>(&o_words_4[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&o_words_4[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&o_words_4[(4) + 3]))
                        : "r"(staging_addr + (unsigned int)(srow * 256) + (unsigned int)(c * 32 + 16 ^ (srow & 7) << 4)));
                    float o_words_f32_4[16];
                    #pragma unroll
                    for (int _pair = 0; _pair < 8; _pair++) {
                        asm volatile(
                            "{\n\t"
                            "shl.b32 %0, %2, 16;\n\t"
                            "and.b32 %1, %2, 0xffff0000;\n\t"
                            "}\n"
                            : "=f"((&o_words_f32_4[_pair * 2])[0]), "=f"((&o_words_f32_4[_pair * 2])[1])
                            : "r"(o_words_4[_pair]));
                    }
                    int valid_4 = meta[1];
                    float p_raw_4[16];
                    for (int e_8 = 0; e_8 < 8; e_8++) {
                        float gv0_4 = ops[e_8];
                        float gv1_4 = ops[8 + e_8];
                        float gs0_4 = ((valid_4 == 1) ? gv0_4 : 0.0f);
                        float gs1_4 = ((valid_4 == 1) ? gv1_4 : 0.0f);
                        p_raw_4[e_8] = gs0_4 * o_words_f32_4[e_8];
                        p_raw_4[8 + e_8] = gs1_4 * o_words_f32_4[8 + e_8];
                    }
                    uint32_t p_raw_bf16_4[8];
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(p_raw_4[_lp*2 + 0], p_raw_4[_lp*2+1 + 0]));
                        p_raw_bf16_4[_lp] = *(uint32_t*)&_bf2;
                    }
                    float p_raw_bf16_f32_4[16];
                    #pragma unroll
                    for (int _pair = 0; _pair < 8; _pair++) {
                        asm volatile(
                            "{\n\t"
                            "shl.b32 %0, %2, 16;\n\t"
                            "and.b32 %1, %2, 0xffff0000;\n\t"
                            "}\n"
                            : "=f"((&p_raw_bf16_f32_4[_pair * 2])[0]), "=f"((&p_raw_bf16_f32_4[_pair * 2])[1])
                            : "r"(p_raw_bf16_4[_pair]));
                    }
                    float out_raw_4[16];
                    for (int e_9 = 0; e_9 < 8; e_9++) {
                        float rv0_4 = ops[16 + e_9];
                        float rv1_4 = ops[24 + e_9];
                        out_raw_4[e_9] = rv0_4 + p_raw_bf16_f32_4[e_9];
                        out_raw_4[8 + e_9] = rv1_4 + p_raw_bf16_f32_4[8 + e_9];
                    }
                    uint32_t out_raw_bf16_4[8];
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_raw_4[_lp*2 + 0], out_raw_4[_lp*2+1 + 0]));
                        out_raw_bf16_4[_lp] = *(uint32_t*)&_bf2;
                    }
                    if (grow_7 < M) {
                        reinterpret_cast<int4*>(out + ((grow_7 * 5376 + col) / 2))[0] = reinterpret_cast<int4*>(out_raw_bf16_4 + 0)[0];
                        reinterpret_cast<int4*>(out + ((grow_7 * 5376 + col) / 2 + 4))[0] = reinterpret_cast<int4*>(out_raw_bf16_4 + 4)[0];
                    }
                }
                asm volatile("barrier.sync 8, 256;" ::: "memory");
                meta[0] = gate_index[((grow_base + 40 < M) ? grow_base + 40 : 0)];
                meta[1] = ((meta[0] >= 0 && meta[0] < 9) ? 1 : 0);
                {
                    const uint4* _vptr_30 = reinterpret_cast<const uint4*>(gate + (((meta[1] == 1) ? meta[0] : 0) * 5376 + col) + 0);
                    uint4 _vld_30[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_30[_blk] = _vptr_30[_blk];
                        uint32_t* _vpairs_30 = reinterpret_cast<uint32_t*>(&_vld_30[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            asm volatile(
                                "{\n\t"
                                "shl.b32 %0, %2, 16;\n\t"
                                "and.b32 %1, %2, 0xffff0000;\n\t"
                                "}\n"
                                : "=f"((&ops[0 + _blk * 8 + _pair * 2])[0]), "=f"((&ops[0 + _blk * 8 + _pair * 2])[1])
                                : "r"(_vpairs_30[_pair]));
                        }
                    }
                }
                {
                    const uint4* _vptr_31 = reinterpret_cast<const uint4*>(gate + (((meta[1] == 1) ? meta[0] : 0) * 5376 + col + 8) + 0);
                    uint4 _vld_31[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_31[_blk] = _vptr_31[_blk];
                        uint32_t* _vpairs_31 = reinterpret_cast<uint32_t*>(&_vld_31[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            asm volatile(
                                "{\n\t"
                                "shl.b32 %0, %2, 16;\n\t"
                                "and.b32 %1, %2, 0xffff0000;\n\t"
                                "}\n"
                                : "=f"((&ops[8 + _blk * 8 + _pair * 2])[0]), "=f"((&ops[8 + _blk * 8 + _pair * 2])[1])
                                : "r"(_vpairs_31[_pair]));
                        }
                    }
                }
                {
                    const uint4* _vptr_32 = reinterpret_cast<const uint4*>(residual + (((grow_base + 40 < M) ? grow_base + 40 : 0) * 5376 + col) + 0);
                    uint4 _vld_32[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_32[_blk] = _vptr_32[_blk];
                        uint32_t* _vpairs_32 = reinterpret_cast<uint32_t*>(&_vld_32[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            asm volatile(
                                "{\n\t"
                                "shl.b32 %0, %2, 16;\n\t"
                                "and.b32 %1, %2, 0xffff0000;\n\t"
                                "}\n"
                                : "=f"((&ops[16 + _blk * 8 + _pair * 2])[0]), "=f"((&ops[16 + _blk * 8 + _pair * 2])[1])
                                : "r"(_vpairs_32[_pair]));
                        }
                    }
                }
                {
                    const uint4* _vptr_33 = reinterpret_cast<const uint4*>(residual + (((grow_base + 40 < M) ? grow_base + 40 : 0) * 5376 + col + 8) + 0);
                    uint4 _vld_33[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_33[_blk] = _vptr_33[_blk];
                        uint32_t* _vpairs_33 = reinterpret_cast<uint32_t*>(&_vld_33[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            asm volatile(
                                "{\n\t"
                                "shl.b32 %0, %2, 16;\n\t"
                                "and.b32 %1, %2, 0xffff0000;\n\t"
                                "}\n"
                                : "=f"((&ops[24 + _blk * 8 + _pair * 2])[0]), "=f"((&ops[24 + _blk * 8 + _pair * 2])[1])
                                : "r"(_vpairs_33[_pair]));
                        }
                    }
                }
                float scaled_8[16];
                scaled_8[0] = accum[66] * sa[5] * sw[0];
                scaled_8[1] = accum[67] * sa[5] * sw[1];
                scaled_8[2] = accum[70] * sa[5] * sw[2];
                scaled_8[3] = accum[71] * sa[5] * sw[3];
                scaled_8[4] = accum[74] * sa[5] * sw[4];
                scaled_8[5] = accum[75] * sa[5] * sw[5];
                scaled_8[6] = accum[78] * sa[5] * sw[6];
                scaled_8[7] = accum[79] * sa[5] * sw[7];
                scaled_8[8] = accum[82] * sa[5] * sw[8];
                scaled_8[9] = accum[83] * sa[5] * sw[9];
                scaled_8[10] = accum[86] * sa[5] * sw[10];
                scaled_8[11] = accum[87] * sa[5] * sw[11];
                scaled_8[12] = accum[90] * sa[5] * sw[12];
                scaled_8[13] = accum[91] * sa[5] * sw[13];
                scaled_8[14] = accum[94] * sa[5] * sw[14];
                scaled_8[15] = accum[95] * sa[5] * sw[15];
                uint32_t scaled_8_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_8[_lp*2 + 0], scaled_8[_lp*2+1 + 0]));
                    scaled_8_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t _stmatrix_addr_34 = static_cast<uint32_t>(staging_addr + (unsigned int)(srow_st * 256) + (unsigned int)((warp_n * 64 + smat * 8) * 2 ^ (srow_st & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_34), "r"(*reinterpret_cast<const uint32_t*>(&scaled_8_bf16[0])), "r"(*reinterpret_cast<const uint32_t*>(&scaled_8_bf16[1])), "r"(*reinterpret_cast<const uint32_t*>(&scaled_8_bf16[2])), "r"(*reinterpret_cast<const uint32_t*>(&scaled_8_bf16[3]))
                    : "memory");
                uint32_t _stmatrix_addr_35 = static_cast<uint32_t>(staging_addr + (unsigned int)(srow_st * 256) + (unsigned int)((warp_n * 64 + (4 + smat) * 8) * 2 ^ (srow_st & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_35), "r"(*reinterpret_cast<const uint32_t*>(&scaled_8_bf16[4])), "r"(*reinterpret_cast<const uint32_t*>(&scaled_8_bf16[5])), "r"(*reinterpret_cast<const uint32_t*>(&scaled_8_bf16[6])), "r"(*reinterpret_cast<const uint32_t*>(&scaled_8_bf16[7]))
                    : "memory");
                asm volatile("barrier.sync 8, 256;" ::: "memory");
                int grow_9 = grow_base + 40;
                if (grow_9 < M) {
                    unsigned int o_words_5[8];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&o_words_5[0])), "=r"(*reinterpret_cast<uint32_t*>(&o_words_5[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&o_words_5[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&o_words_5[(0) + 3]))
                        : "r"(staging_addr + (unsigned int)(srow * 256) + (unsigned int)(c * 32 ^ (srow & 7) << 4)));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&o_words_5[4])), "=r"(*reinterpret_cast<uint32_t*>(&o_words_5[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&o_words_5[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&o_words_5[(4) + 3]))
                        : "r"(staging_addr + (unsigned int)(srow * 256) + (unsigned int)(c * 32 + 16 ^ (srow & 7) << 4)));
                    float o_words_f32_5[16];
                    #pragma unroll
                    for (int _pair = 0; _pair < 8; _pair++) {
                        asm volatile(
                            "{\n\t"
                            "shl.b32 %0, %2, 16;\n\t"
                            "and.b32 %1, %2, 0xffff0000;\n\t"
                            "}\n"
                            : "=f"((&o_words_f32_5[_pair * 2])[0]), "=f"((&o_words_f32_5[_pair * 2])[1])
                            : "r"(o_words_5[_pair]));
                    }
                    int valid_5 = meta[1];
                    float p_raw_5[16];
                    for (int e_10 = 0; e_10 < 8; e_10++) {
                        float gv0_5 = ops[e_10];
                        float gv1_5 = ops[8 + e_10];
                        float gs0_5 = ((valid_5 == 1) ? gv0_5 : 0.0f);
                        float gs1_5 = ((valid_5 == 1) ? gv1_5 : 0.0f);
                        p_raw_5[e_10] = gs0_5 * o_words_f32_5[e_10];
                        p_raw_5[8 + e_10] = gs1_5 * o_words_f32_5[8 + e_10];
                    }
                    uint32_t p_raw_bf16_5[8];
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(p_raw_5[_lp*2 + 0], p_raw_5[_lp*2+1 + 0]));
                        p_raw_bf16_5[_lp] = *(uint32_t*)&_bf2;
                    }
                    float p_raw_bf16_f32_5[16];
                    #pragma unroll
                    for (int _pair = 0; _pair < 8; _pair++) {
                        asm volatile(
                            "{\n\t"
                            "shl.b32 %0, %2, 16;\n\t"
                            "and.b32 %1, %2, 0xffff0000;\n\t"
                            "}\n"
                            : "=f"((&p_raw_bf16_f32_5[_pair * 2])[0]), "=f"((&p_raw_bf16_f32_5[_pair * 2])[1])
                            : "r"(p_raw_bf16_5[_pair]));
                    }
                    float out_raw_5[16];
                    for (int e_11 = 0; e_11 < 8; e_11++) {
                        float rv0_5 = ops[16 + e_11];
                        float rv1_5 = ops[24 + e_11];
                        out_raw_5[e_11] = rv0_5 + p_raw_bf16_f32_5[e_11];
                        out_raw_5[8 + e_11] = rv1_5 + p_raw_bf16_f32_5[8 + e_11];
                    }
                    uint32_t out_raw_bf16_5[8];
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_raw_5[_lp*2 + 0], out_raw_5[_lp*2+1 + 0]));
                        out_raw_bf16_5[_lp] = *(uint32_t*)&_bf2;
                    }
                    if (grow_9 < M) {
                        reinterpret_cast<int4*>(out + ((grow_9 * 5376 + col) / 2))[0] = reinterpret_cast<int4*>(out_raw_bf16_5 + 0)[0];
                        reinterpret_cast<int4*>(out + ((grow_9 * 5376 + col) / 2 + 4))[0] = reinterpret_cast<int4*>(out_raw_bf16_5 + 4)[0];
                    }
                }
                asm volatile("barrier.sync 8, 256;" ::: "memory");
                meta[0] = gate_index[((grow_base + 48 < M) ? grow_base + 48 : 0)];
                meta[1] = ((meta[0] >= 0 && meta[0] < 9) ? 1 : 0);
                {
                    const uint4* _vptr_36 = reinterpret_cast<const uint4*>(gate + (((meta[1] == 1) ? meta[0] : 0) * 5376 + col) + 0);
                    uint4 _vld_36[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_36[_blk] = _vptr_36[_blk];
                        uint32_t* _vpairs_36 = reinterpret_cast<uint32_t*>(&_vld_36[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            asm volatile(
                                "{\n\t"
                                "shl.b32 %0, %2, 16;\n\t"
                                "and.b32 %1, %2, 0xffff0000;\n\t"
                                "}\n"
                                : "=f"((&ops[0 + _blk * 8 + _pair * 2])[0]), "=f"((&ops[0 + _blk * 8 + _pair * 2])[1])
                                : "r"(_vpairs_36[_pair]));
                        }
                    }
                }
                {
                    const uint4* _vptr_37 = reinterpret_cast<const uint4*>(gate + (((meta[1] == 1) ? meta[0] : 0) * 5376 + col + 8) + 0);
                    uint4 _vld_37[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_37[_blk] = _vptr_37[_blk];
                        uint32_t* _vpairs_37 = reinterpret_cast<uint32_t*>(&_vld_37[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            asm volatile(
                                "{\n\t"
                                "shl.b32 %0, %2, 16;\n\t"
                                "and.b32 %1, %2, 0xffff0000;\n\t"
                                "}\n"
                                : "=f"((&ops[8 + _blk * 8 + _pair * 2])[0]), "=f"((&ops[8 + _blk * 8 + _pair * 2])[1])
                                : "r"(_vpairs_37[_pair]));
                        }
                    }
                }
                {
                    const uint4* _vptr_38 = reinterpret_cast<const uint4*>(residual + (((grow_base + 48 < M) ? grow_base + 48 : 0) * 5376 + col) + 0);
                    uint4 _vld_38[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_38[_blk] = _vptr_38[_blk];
                        uint32_t* _vpairs_38 = reinterpret_cast<uint32_t*>(&_vld_38[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            asm volatile(
                                "{\n\t"
                                "shl.b32 %0, %2, 16;\n\t"
                                "and.b32 %1, %2, 0xffff0000;\n\t"
                                "}\n"
                                : "=f"((&ops[16 + _blk * 8 + _pair * 2])[0]), "=f"((&ops[16 + _blk * 8 + _pair * 2])[1])
                                : "r"(_vpairs_38[_pair]));
                        }
                    }
                }
                {
                    const uint4* _vptr_39 = reinterpret_cast<const uint4*>(residual + (((grow_base + 48 < M) ? grow_base + 48 : 0) * 5376 + col + 8) + 0);
                    uint4 _vld_39[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_39[_blk] = _vptr_39[_blk];
                        uint32_t* _vpairs_39 = reinterpret_cast<uint32_t*>(&_vld_39[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            asm volatile(
                                "{\n\t"
                                "shl.b32 %0, %2, 16;\n\t"
                                "and.b32 %1, %2, 0xffff0000;\n\t"
                                "}\n"
                                : "=f"((&ops[24 + _blk * 8 + _pair * 2])[0]), "=f"((&ops[24 + _blk * 8 + _pair * 2])[1])
                                : "r"(_vpairs_39[_pair]));
                        }
                    }
                }
                float scaled_10[16];
                scaled_10[0] = accum[96] * sa[6] * sw[0];
                scaled_10[1] = accum[97] * sa[6] * sw[1];
                scaled_10[2] = accum[100] * sa[6] * sw[2];
                scaled_10[3] = accum[101] * sa[6] * sw[3];
                scaled_10[4] = accum[104] * sa[6] * sw[4];
                scaled_10[5] = accum[105] * sa[6] * sw[5];
                scaled_10[6] = accum[108] * sa[6] * sw[6];
                scaled_10[7] = accum[109] * sa[6] * sw[7];
                scaled_10[8] = accum[112] * sa[6] * sw[8];
                scaled_10[9] = accum[113] * sa[6] * sw[9];
                scaled_10[10] = accum[116] * sa[6] * sw[10];
                scaled_10[11] = accum[117] * sa[6] * sw[11];
                scaled_10[12] = accum[120] * sa[6] * sw[12];
                scaled_10[13] = accum[121] * sa[6] * sw[13];
                scaled_10[14] = accum[124] * sa[6] * sw[14];
                scaled_10[15] = accum[125] * sa[6] * sw[15];
                uint32_t scaled_10_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_10[_lp*2 + 0], scaled_10[_lp*2+1 + 0]));
                    scaled_10_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t _stmatrix_addr_40 = static_cast<uint32_t>(staging_addr + (unsigned int)(srow_st * 256) + (unsigned int)((warp_n * 64 + smat * 8) * 2 ^ (srow_st & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_40), "r"(*reinterpret_cast<const uint32_t*>(&scaled_10_bf16[0])), "r"(*reinterpret_cast<const uint32_t*>(&scaled_10_bf16[1])), "r"(*reinterpret_cast<const uint32_t*>(&scaled_10_bf16[2])), "r"(*reinterpret_cast<const uint32_t*>(&scaled_10_bf16[3]))
                    : "memory");
                uint32_t _stmatrix_addr_41 = static_cast<uint32_t>(staging_addr + (unsigned int)(srow_st * 256) + (unsigned int)((warp_n * 64 + (4 + smat) * 8) * 2 ^ (srow_st & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_41), "r"(*reinterpret_cast<const uint32_t*>(&scaled_10_bf16[4])), "r"(*reinterpret_cast<const uint32_t*>(&scaled_10_bf16[5])), "r"(*reinterpret_cast<const uint32_t*>(&scaled_10_bf16[6])), "r"(*reinterpret_cast<const uint32_t*>(&scaled_10_bf16[7]))
                    : "memory");
                asm volatile("barrier.sync 8, 256;" ::: "memory");
                int grow_11 = grow_base + 48;
                if (grow_11 < M) {
                    unsigned int o_words_6[8];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&o_words_6[0])), "=r"(*reinterpret_cast<uint32_t*>(&o_words_6[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&o_words_6[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&o_words_6[(0) + 3]))
                        : "r"(staging_addr + (unsigned int)(srow * 256) + (unsigned int)(c * 32 ^ (srow & 7) << 4)));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&o_words_6[4])), "=r"(*reinterpret_cast<uint32_t*>(&o_words_6[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&o_words_6[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&o_words_6[(4) + 3]))
                        : "r"(staging_addr + (unsigned int)(srow * 256) + (unsigned int)(c * 32 + 16 ^ (srow & 7) << 4)));
                    float o_words_f32_6[16];
                    #pragma unroll
                    for (int _pair = 0; _pair < 8; _pair++) {
                        asm volatile(
                            "{\n\t"
                            "shl.b32 %0, %2, 16;\n\t"
                            "and.b32 %1, %2, 0xffff0000;\n\t"
                            "}\n"
                            : "=f"((&o_words_f32_6[_pair * 2])[0]), "=f"((&o_words_f32_6[_pair * 2])[1])
                            : "r"(o_words_6[_pair]));
                    }
                    int valid_6 = meta[1];
                    float p_raw_6[16];
                    for (int e_12 = 0; e_12 < 8; e_12++) {
                        float gv0_6 = ops[e_12];
                        float gv1_6 = ops[8 + e_12];
                        float gs0_6 = ((valid_6 == 1) ? gv0_6 : 0.0f);
                        float gs1_6 = ((valid_6 == 1) ? gv1_6 : 0.0f);
                        p_raw_6[e_12] = gs0_6 * o_words_f32_6[e_12];
                        p_raw_6[8 + e_12] = gs1_6 * o_words_f32_6[8 + e_12];
                    }
                    uint32_t p_raw_bf16_6[8];
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(p_raw_6[_lp*2 + 0], p_raw_6[_lp*2+1 + 0]));
                        p_raw_bf16_6[_lp] = *(uint32_t*)&_bf2;
                    }
                    float p_raw_bf16_f32_6[16];
                    #pragma unroll
                    for (int _pair = 0; _pair < 8; _pair++) {
                        asm volatile(
                            "{\n\t"
                            "shl.b32 %0, %2, 16;\n\t"
                            "and.b32 %1, %2, 0xffff0000;\n\t"
                            "}\n"
                            : "=f"((&p_raw_bf16_f32_6[_pair * 2])[0]), "=f"((&p_raw_bf16_f32_6[_pair * 2])[1])
                            : "r"(p_raw_bf16_6[_pair]));
                    }
                    float out_raw_6[16];
                    for (int e_13 = 0; e_13 < 8; e_13++) {
                        float rv0_6 = ops[16 + e_13];
                        float rv1_6 = ops[24 + e_13];
                        out_raw_6[e_13] = rv0_6 + p_raw_bf16_f32_6[e_13];
                        out_raw_6[8 + e_13] = rv1_6 + p_raw_bf16_f32_6[8 + e_13];
                    }
                    uint32_t out_raw_bf16_6[8];
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_raw_6[_lp*2 + 0], out_raw_6[_lp*2+1 + 0]));
                        out_raw_bf16_6[_lp] = *(uint32_t*)&_bf2;
                    }
                    if (grow_11 < M) {
                        reinterpret_cast<int4*>(out + ((grow_11 * 5376 + col) / 2))[0] = reinterpret_cast<int4*>(out_raw_bf16_6 + 0)[0];
                        reinterpret_cast<int4*>(out + ((grow_11 * 5376 + col) / 2 + 4))[0] = reinterpret_cast<int4*>(out_raw_bf16_6 + 4)[0];
                    }
                }
                asm volatile("barrier.sync 8, 256;" ::: "memory");
                meta[0] = gate_index[((grow_base + 56 < M) ? grow_base + 56 : 0)];
                meta[1] = ((meta[0] >= 0 && meta[0] < 9) ? 1 : 0);
                {
                    const uint4* _vptr_42 = reinterpret_cast<const uint4*>(gate + (((meta[1] == 1) ? meta[0] : 0) * 5376 + col) + 0);
                    uint4 _vld_42[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_42[_blk] = _vptr_42[_blk];
                        uint32_t* _vpairs_42 = reinterpret_cast<uint32_t*>(&_vld_42[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            asm volatile(
                                "{\n\t"
                                "shl.b32 %0, %2, 16;\n\t"
                                "and.b32 %1, %2, 0xffff0000;\n\t"
                                "}\n"
                                : "=f"((&ops[0 + _blk * 8 + _pair * 2])[0]), "=f"((&ops[0 + _blk * 8 + _pair * 2])[1])
                                : "r"(_vpairs_42[_pair]));
                        }
                    }
                }
                {
                    const uint4* _vptr_43 = reinterpret_cast<const uint4*>(gate + (((meta[1] == 1) ? meta[0] : 0) * 5376 + col + 8) + 0);
                    uint4 _vld_43[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_43[_blk] = _vptr_43[_blk];
                        uint32_t* _vpairs_43 = reinterpret_cast<uint32_t*>(&_vld_43[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            asm volatile(
                                "{\n\t"
                                "shl.b32 %0, %2, 16;\n\t"
                                "and.b32 %1, %2, 0xffff0000;\n\t"
                                "}\n"
                                : "=f"((&ops[8 + _blk * 8 + _pair * 2])[0]), "=f"((&ops[8 + _blk * 8 + _pair * 2])[1])
                                : "r"(_vpairs_43[_pair]));
                        }
                    }
                }
                {
                    const uint4* _vptr_44 = reinterpret_cast<const uint4*>(residual + (((grow_base + 56 < M) ? grow_base + 56 : 0) * 5376 + col) + 0);
                    uint4 _vld_44[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_44[_blk] = _vptr_44[_blk];
                        uint32_t* _vpairs_44 = reinterpret_cast<uint32_t*>(&_vld_44[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            asm volatile(
                                "{\n\t"
                                "shl.b32 %0, %2, 16;\n\t"
                                "and.b32 %1, %2, 0xffff0000;\n\t"
                                "}\n"
                                : "=f"((&ops[16 + _blk * 8 + _pair * 2])[0]), "=f"((&ops[16 + _blk * 8 + _pair * 2])[1])
                                : "r"(_vpairs_44[_pair]));
                        }
                    }
                }
                {
                    const uint4* _vptr_45 = reinterpret_cast<const uint4*>(residual + (((grow_base + 56 < M) ? grow_base + 56 : 0) * 5376 + col + 8) + 0);
                    uint4 _vld_45[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_45[_blk] = _vptr_45[_blk];
                        uint32_t* _vpairs_45 = reinterpret_cast<uint32_t*>(&_vld_45[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            asm volatile(
                                "{\n\t"
                                "shl.b32 %0, %2, 16;\n\t"
                                "and.b32 %1, %2, 0xffff0000;\n\t"
                                "}\n"
                                : "=f"((&ops[24 + _blk * 8 + _pair * 2])[0]), "=f"((&ops[24 + _blk * 8 + _pair * 2])[1])
                                : "r"(_vpairs_45[_pair]));
                        }
                    }
                }
                float scaled_12[16];
                scaled_12[0] = accum[98] * sa[7] * sw[0];
                scaled_12[1] = accum[99] * sa[7] * sw[1];
                scaled_12[2] = accum[102] * sa[7] * sw[2];
                scaled_12[3] = accum[103] * sa[7] * sw[3];
                scaled_12[4] = accum[106] * sa[7] * sw[4];
                scaled_12[5] = accum[107] * sa[7] * sw[5];
                scaled_12[6] = accum[110] * sa[7] * sw[6];
                scaled_12[7] = accum[111] * sa[7] * sw[7];
                scaled_12[8] = accum[114] * sa[7] * sw[8];
                scaled_12[9] = accum[115] * sa[7] * sw[9];
                scaled_12[10] = accum[118] * sa[7] * sw[10];
                scaled_12[11] = accum[119] * sa[7] * sw[11];
                scaled_12[12] = accum[122] * sa[7] * sw[12];
                scaled_12[13] = accum[123] * sa[7] * sw[13];
                scaled_12[14] = accum[126] * sa[7] * sw[14];
                scaled_12[15] = accum[127] * sa[7] * sw[15];
                uint32_t scaled_12_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_12[_lp*2 + 0], scaled_12[_lp*2+1 + 0]));
                    scaled_12_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t _stmatrix_addr_46 = static_cast<uint32_t>(staging_addr + (unsigned int)(srow_st * 256) + (unsigned int)((warp_n * 64 + smat * 8) * 2 ^ (srow_st & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_46), "r"(*reinterpret_cast<const uint32_t*>(&scaled_12_bf16[0])), "r"(*reinterpret_cast<const uint32_t*>(&scaled_12_bf16[1])), "r"(*reinterpret_cast<const uint32_t*>(&scaled_12_bf16[2])), "r"(*reinterpret_cast<const uint32_t*>(&scaled_12_bf16[3]))
                    : "memory");
                uint32_t _stmatrix_addr_47 = static_cast<uint32_t>(staging_addr + (unsigned int)(srow_st * 256) + (unsigned int)((warp_n * 64 + (4 + smat) * 8) * 2 ^ (srow_st & 7) << 4));
                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_47), "r"(*reinterpret_cast<const uint32_t*>(&scaled_12_bf16[4])), "r"(*reinterpret_cast<const uint32_t*>(&scaled_12_bf16[5])), "r"(*reinterpret_cast<const uint32_t*>(&scaled_12_bf16[6])), "r"(*reinterpret_cast<const uint32_t*>(&scaled_12_bf16[7]))
                    : "memory");
                asm volatile("barrier.sync 8, 256;" ::: "memory");
                int grow_13 = grow_base + 56;
                if (grow_13 < M) {
                    unsigned int o_words_7[8];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&o_words_7[0])), "=r"(*reinterpret_cast<uint32_t*>(&o_words_7[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&o_words_7[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&o_words_7[(0) + 3]))
                        : "r"(staging_addr + (unsigned int)(srow * 256) + (unsigned int)(c * 32 ^ (srow & 7) << 4)));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&o_words_7[4])), "=r"(*reinterpret_cast<uint32_t*>(&o_words_7[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&o_words_7[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&o_words_7[(4) + 3]))
                        : "r"(staging_addr + (unsigned int)(srow * 256) + (unsigned int)(c * 32 + 16 ^ (srow & 7) << 4)));
                    float o_words_f32_7[16];
                    #pragma unroll
                    for (int _pair = 0; _pair < 8; _pair++) {
                        asm volatile(
                            "{\n\t"
                            "shl.b32 %0, %2, 16;\n\t"
                            "and.b32 %1, %2, 0xffff0000;\n\t"
                            "}\n"
                            : "=f"((&o_words_f32_7[_pair * 2])[0]), "=f"((&o_words_f32_7[_pair * 2])[1])
                            : "r"(o_words_7[_pair]));
                    }
                    int valid_7 = meta[1];
                    float p_raw_7[16];
                    for (int e_14 = 0; e_14 < 8; e_14++) {
                        float gv0_7 = ops[e_14];
                        float gv1_7 = ops[8 + e_14];
                        float gs0_7 = ((valid_7 == 1) ? gv0_7 : 0.0f);
                        float gs1_7 = ((valid_7 == 1) ? gv1_7 : 0.0f);
                        p_raw_7[e_14] = gs0_7 * o_words_f32_7[e_14];
                        p_raw_7[8 + e_14] = gs1_7 * o_words_f32_7[8 + e_14];
                    }
                    uint32_t p_raw_bf16_7[8];
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(p_raw_7[_lp*2 + 0], p_raw_7[_lp*2+1 + 0]));
                        p_raw_bf16_7[_lp] = *(uint32_t*)&_bf2;
                    }
                    float p_raw_bf16_f32_7[16];
                    #pragma unroll
                    for (int _pair = 0; _pair < 8; _pair++) {
                        asm volatile(
                            "{\n\t"
                            "shl.b32 %0, %2, 16;\n\t"
                            "and.b32 %1, %2, 0xffff0000;\n\t"
                            "}\n"
                            : "=f"((&p_raw_bf16_f32_7[_pair * 2])[0]), "=f"((&p_raw_bf16_f32_7[_pair * 2])[1])
                            : "r"(p_raw_bf16_7[_pair]));
                    }
                    float out_raw_7[16];
                    for (int e_15 = 0; e_15 < 8; e_15++) {
                        float rv0_7 = ops[16 + e_15];
                        float rv1_7 = ops[24 + e_15];
                        out_raw_7[e_15] = rv0_7 + p_raw_bf16_f32_7[e_15];
                        out_raw_7[8 + e_15] = rv1_7 + p_raw_bf16_f32_7[8 + e_15];
                    }
                    uint32_t out_raw_bf16_7[8];
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_raw_7[_lp*2 + 0], out_raw_7[_lp*2+1 + 0]));
                        out_raw_bf16_7[_lp] = *(uint32_t*)&_bf2;
                    }
                    if (grow_13 < M) {
                        reinterpret_cast<int4*>(out + ((grow_13 * 5376 + col) / 2))[0] = reinterpret_cast<int4*>(out_raw_bf16_7 + 0)[0];
                        reinterpret_cast<int4*>(out + ((grow_13 * 5376 + col) / 2 + 4))[0] = reinterpret_cast<int4*>(out_raw_bf16_7 + 4)[0];
                    }
                }
                asm volatile("barrier.sync 8, 256;" ::: "memory");
            }
        }
    }

    // Cleanup
}

}  // namespace h3_out_proj_gemm_nvfp4_sm120a
#undef GROUP_M
#undef H3_QOP_INF
#undef NUM_AB_PIPE_STAGES
#undef SMEM_A_STAGE_OFF
#undef SMEM_A_STAGE_STAGE_BYTES
#undef SMEM_A_STAGE_STRIDE
#undef SMEM_B_STAGE_OFF
#undef SMEM_B_STAGE_STAGE_BYTES
#undef SMEM_B_STAGE_STRIDE
#undef SMEM_SFA_SLOT_OFF
#undef SMEM_SFA_SLOT_STAGE_BYTES
#undef SMEM_SFA_SLOT_STRIDE
#undef SMEM_SFB_STAGE_OFF
#undef SMEM_SFB_STAGE_STAGE_BYTES
#undef SMEM_SFB_STAGE_STRIDE
#undef SMEM_STAGING_OFF
#undef SMEM_STAGING_STAGE_BYTES
#undef SMEM_STAGING_STRIDE
#undef SMEM_TOTAL
#undef THREADS
#undef ab_empty_addr
#undef ab_full_addr

#include <cuda_runtime.h>

#include <algorithm>
#include <mutex>
#include <vector>

#include "tvm_ffi_utils.h"

namespace {

constexpr int64_t kHidden = 5376;   // output width N
constexpr int64_t kAttnDim = 7168;  // reduction K
constexpr int64_t kSfBlock = 16;
constexpr int64_t kAttnSf = kAttnDim / kSfBlock;  // 448 UE4M3 scales per activation row
// TMA coordinates are 32-bit; the persistent tile counter is int32.
constexpr int64_t kMaxRows = int64_t{1} << 24;
constexpr int kBlockM = 256;
constexpr int kBlockN = 128;
constexpr int kBlockK = 64;
constexpr int kNTiles = 42;
constexpr int kSfbRowsPerNTile = 224;
constexpr int kSfbRowsPerKTile = 4;
constexpr uint32_t kSfaBoxBytes = 16;  // one k256 pair of ring stages per activation-scale box
constexpr CUtensorMapSwizzle kOperandSwizzle = CU_TENSOR_MAP_SWIZZLE_64B;  // 64-byte operand rows per stage
constexpr int kGemmThreads = 384;

using GemmKernel = void (*)(CUtensorMap, CUtensorMap, CUtensorMap, CUtensorMap, unsigned int*, unsigned int*, uint8_t*,
                            float*, float*, float*, __nv_bfloat16*, int*, __nv_bfloat16*, unsigned int*, unsigned int*,
                            int, int, int, float);

struct GemmVariant {
  GemmKernel kernel;
  int dynamic_smem_bytes;
};

// [quant (0 = fp8, 1 = nvfp4)]
const GemmVariant kGemmVariants[2] = {
    {h3_out_proj_gemm_fp8_sm120a::kernel_h3_out_proj_gemm_fused, 94208},
    {h3_out_proj_gemm_nvfp4_sm120a::kernel_h3_out_proj_gemm_fused, 94208},
};

void CheckTensor(const TensorView& tensor, const char* name, DLDevice device, DLDataType dtype,
                 std::initializer_list<int64_t> shape) {
  TVM_FFI_CHECK(tensor.device().device_type == kDLCUDA, ValueError) << name << " must be a CUDA tensor";
  TVM_FFI_CHECK(tensor.device().device_id == device.device_id, ValueError)
      << name << " must be on the same CUDA device as attn_out";
  TVM_FFI_CHECK(encode_dlpack_dtype(tensor.dtype()) == encode_dlpack_dtype(dtype), ValueError)
      << name << " has the wrong dtype";
  TVM_FFI_CHECK(tensor.ndim() == static_cast<int>(shape.size()), ValueError)
      << name << " must have " << shape.size() << " dimensions";
  int64_t expected_stride = 1;
  int dim = tensor.ndim() - 1;
  for (auto it = std::rbegin(shape); it != std::rend(shape); ++it, --dim) {
    TVM_FFI_CHECK(tensor.size(dim) == *it, ValueError) << name << " has the wrong shape (dimension " << dim << ")";
    TVM_FFI_CHECK(tensor.size(dim) == 1 || tensor.stride(dim) == expected_stride, ValueError)
        << name << " must be contiguous";
    expected_stride *= *it;
  }
  TVM_FFI_CHECK(reinterpret_cast<uintptr_t>(tensor.data_ptr()) % 16 == 0, ValueError)
      << name << " must be 16-byte aligned";
}

// 2-D byte tile map: rows of ``inner_bytes`` contiguous bytes, box = ``box_rows`` x ``box_inner`` bytes.
// Rows beyond the tensor are zero-filled by TMA (partial M tail tiles).
CUtensorMap EncodeByteTile(const void* base, int64_t inner_bytes, int64_t rows, uint32_t box_inner,
                           uint32_t box_rows, CUtensorMapSwizzle swizzle, const char* name) {
  uint64_t global_dim[2] = {static_cast<uint64_t>(inner_bytes), static_cast<uint64_t>(rows)};
  uint64_t global_strides[1] = {static_cast<uint64_t>(inner_bytes)};
  uint32_t box_dim[2] = {box_inner, box_rows};
  uint32_t element_strides[2] = {1, 1};
  CUtensorMap descriptor{};
  CUresult result = cuTensorMapEncodeTiled(
      &descriptor, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, const_cast<void*>(base), global_dim, global_strides, box_dim,
      element_strides, CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle, CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_CHECK(result == CUDA_SUCCESS, RuntimeError)
      << "failed to encode the " << name << " tensor map: CUresult=" << static_cast<int>(result);
  return descriptor;
}

int ConfigureKernels() {
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
  TVM_FFI_CHECK(properties.major == 12, RuntimeError)
      << "MiniMax-H3 SM120 quantized output projection requires compute capability 12.x (GB202); got "
      << properties.major << "." << properties.minor;
  for (const auto& variant : kGemmVariants) {
    status = cudaFuncSetAttribute(variant.kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  variant.dynamic_smem_bytes);
    TVM_FFI_CHECK(status == cudaSuccess, RuntimeError)
        << "failed to opt in to dynamic shared memory: " << cudaGetErrorString(status);
  }
  configured_devices.emplace_back(device, properties.multiProcessorCount);
  return properties.multiProcessorCount;
}

struct LaunchPlan {
  int gemm_grid;
  int num_m_tiles;
  int total_tiles;
};

LaunchPlan MakeLaunchPlan(int64_t rows, int num_sms) {
  // Mirrors the Python launch_plan(): one persistent CTA per SM (ctas_per_sm = 1).  The in-kernel
  // quantization / TMA flag protocol needs every CTA resident, which a grid of at most one CTA per
  // SM (384 threads, ~92 KB dynamic SMEM) guarantees.
  const int64_t num_m_tiles = (rows + kBlockM - 1) / kBlockM;
  const int64_t total_tiles = num_m_tiles * kNTiles;
  LaunchPlan plan{};
  plan.gemm_grid = static_cast<int>(std::max<int64_t>(1, std::min<int64_t>(total_tiles, num_sms)));
  plan.num_m_tiles = static_cast<int>(num_m_tiles);
  plan.total_tiles = static_cast<int>(total_tiles);
  return plan;
}

struct CommonArgs {
  int64_t rows;
  DLDevice device;
  cudaStream_t stream;
  int num_sms;
  LaunchPlan plan;
  __nv_bfloat16* gate;
  int* gate_index;
  __nv_bfloat16* residual;
  unsigned int* out;
};

// Shared validation of the BF16 operands and the caller-owned output.
CommonArgs CheckCommon(const TensorView& attn_out, const TensorView& gate, const TensorView& gate_index,
                       const TensorView& residual, const TensorView& out) {
  TVM_FFI_CHECK(attn_out.ndim() == 2 && attn_out.size(0) >= 1 && attn_out.size(0) <= kMaxRows &&
                    attn_out.size(1) == kAttnDim, ValueError)
      << "attn_out must be [M, 7168] with 1 <= M <= " << kMaxRows;
  CommonArgs args{};
  args.rows = attn_out.size(0);
  args.device = attn_out.device();
  CheckTensor(attn_out, "attn_out", args.device, dl_bfloat16, {args.rows, kAttnDim});
  TVM_FFI_CHECK(gate.ndim() == 2 && gate.size(0) >= 1 && gate.size(1) == kHidden, ValueError)
      << "gate must be [rows, 5376]";
  CheckTensor(gate, "gate", args.device, dl_bfloat16, {gate.size(0), kHidden});
  CheckTensor(gate_index, "gate_index", args.device, dl_int32, {args.rows});
  CheckTensor(residual, "residual", args.device, dl_bfloat16, {args.rows, kHidden});
  CheckTensor(out, "out", args.device, dl_bfloat16, {args.rows, kHidden});
  TVM_FFI_CHECK(gate.size(0) == 9, ValueError)
      << "gate must have exactly 9 rows (the MiniMax-H3 AdaLN plan); indices outside the table contribute zero";
  args.gate = static_cast<__nv_bfloat16*>(gate.data_ptr());
  args.gate_index = static_cast<int*>(gate_index.data_ptr());
  args.residual = static_cast<__nv_bfloat16*>(residual.data_ptr());
  args.out = static_cast<unsigned int*>(out.data_ptr());
  args.num_sms = ConfigureKernels();
  args.stream = get_stream(args.device);
  args.plan = MakeLaunchPlan(args.rows, args.num_sms);
  return args;
}

struct QuantOutputs {
  unsigned int* attn_out;  // the BF16 [M, 7168] rows as packed 32-bit words
  unsigned int* act_q;
  uint8_t* act_sf;          // NVFP4 only (nullptr for FP8: never dereferenced)
  float* act_scale;         // FP8 only (a valid dummy for NVFP4: read for the epilogue row scale = 1.0)
  float* act_global_scale;  // NVFP4 only (a valid dummy for FP8: read once)
};

void LaunchGemm(int quant, const CommonArgs& args, const CUtensorMap& a_map, const CUtensorMap& b_map,
                const CUtensorMap& sfa_map, const CUtensorMap& sfb_map, const QuantOutputs& q, float* w_scale,
                double alpha, const char* what) {
  const GemmVariant variant = kGemmVariants[quant];
  // Per-M-tile ready counters (rows quantized so far), zeroed per call: a stream-ordered scratch
  // allocation so the public API needs no caller workspace.
  unsigned int* flags = nullptr;
  const size_t flag_bytes = static_cast<size_t>(args.plan.num_m_tiles) * sizeof(unsigned int);
  cudaError_t status = cudaMallocAsync(reinterpret_cast<void**>(&flags), flag_bytes, args.stream);
  TVM_FFI_CHECK(status == cudaSuccess, RuntimeError) << what << " flag workspace allocation failed: " << cudaGetErrorString(status);
  status = cudaMemsetAsync(flags, 0, flag_bytes, args.stream);
  TVM_FFI_CHECK(status == cudaSuccess, RuntimeError) << what << " flag workspace memset failed: " << cudaGetErrorString(status);
  variant.kernel<<<dim3(args.plan.gemm_grid), dim3(kGemmThreads), variant.dynamic_smem_bytes, args.stream>>>(
      a_map, b_map, sfa_map, sfb_map, q.attn_out, q.act_q, q.act_sf, q.act_scale, q.act_global_scale, w_scale,
      args.gate, args.gate_index, args.residual, args.out, flags, static_cast<int>(args.rows), args.plan.num_m_tiles,
      args.plan.total_tiles, static_cast<float>(alpha));
  status = cudaGetLastError();
  TVM_FFI_CHECK(status == cudaSuccess, RuntimeError) << what << " GEMM launch failed: " << cudaGetErrorString(status);
  status = cudaFreeAsync(flags, args.stream);
  TVM_FFI_CHECK(status == cudaSuccess, RuntimeError) << what << " flag workspace release failed: " << cudaGetErrorString(status);
}

// Single-element FP32 constant 1.0 per device (dummy row scale / global scale of the other route).
float* OnesScalar(DLDevice device, cudaStream_t stream) {
  static std::mutex mutex;
  static std::vector<std::pair<int, float*>> per_device;
  std::lock_guard<std::mutex> lock(mutex);
  for (const auto& entry : per_device) {
    if (entry.first == device.device_id) return entry.second;
  }
  float* ptr = nullptr;
  cudaError_t status = cudaMalloc(reinterpret_cast<void**>(&ptr), 16 * sizeof(float));
  TVM_FFI_CHECK(status == cudaSuccess, RuntimeError) << "constant allocation failed: " << cudaGetErrorString(status);
  const float ones[16] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
  status = cudaMemcpyAsync(ptr, ones, sizeof(ones), cudaMemcpyHostToDevice, stream);
  TVM_FFI_CHECK(status == cudaSuccess, RuntimeError) << "constant upload failed: " << cudaGetErrorString(status);
  status = cudaStreamSynchronize(stream);
  TVM_FFI_CHECK(status == cudaSuccess, RuntimeError) << "constant upload sync failed: " << cudaGetErrorString(status);
  per_device.emplace_back(device.device_id, ptr);
  return ptr;
}

}  // namespace

// FP8 W8A8 route.  attn_out [M, 7168] BF16 -> act_q [M, 7168] E4M3 (per-token scale act_scale [M] FP32 =
// RN(amax / 448)) -> o = BF16(act_q @ o_weight_q^T * act_scale[m] * o_weight_scale[n])
// -> out [M, 5376] BF16 = BF16(residual + BF16(gate[gate_index[m]] * o)).
// o_weight_q: E4M3 [5376, 7168] (per-output-channel), o_weight_scale: FP32 [5376].
void minimax_h3_sm120_fp8_out_proj(TensorView attn_out, TensorView o_weight_q, TensorView o_weight_scale,
                                   TensorView gate, TensorView gate_index, TensorView residual, TensorView act_q,
                                   TensorView act_scale, TensorView out) {
  const CommonArgs args = CheckCommon(attn_out, gate, gate_index, residual, out);
  CheckTensor(o_weight_q, "o_weight_q", args.device, dl_float8_e4m3fn, {kHidden, kAttnDim});
  CheckTensor(o_weight_scale, "o_weight_scale", args.device, dl_float32, {kHidden});
  CheckTensor(act_q, "act_q", args.device, dl_float8_e4m3fn, {args.rows, kAttnDim});
  CheckTensor(act_scale, "act_scale", args.device, dl_float32, {args.rows});
  ffi::CUDADeviceGuard device_guard(args.device.device_id);

  const CUtensorMap a_map = EncodeByteTile(act_q.data_ptr(), kAttnDim, args.rows, kBlockK, kBlockM,
                                           kOperandSwizzle, "act_q");
  const CUtensorMap b_map = EncodeByteTile(o_weight_q.data_ptr(), kAttnDim, kHidden, kBlockK, kBlockN,
                                           kOperandSwizzle, "o_weight_q");
  // The FP8 kernel never issues scale-tile loads; its descriptors only need valid parameter storage.
  const CUtensorMap unused_map{};
  QuantOutputs q{};
  q.attn_out = static_cast<unsigned int*>(attn_out.data_ptr());
  q.act_q = static_cast<unsigned int*>(act_q.data_ptr());
  q.act_sf = nullptr;
  q.act_scale = static_cast<float*>(act_scale.data_ptr());
  q.act_global_scale = OnesScalar(args.device, args.stream);
  LaunchGemm(0, args, a_map, b_map, unused_map, unused_map, q, static_cast<float*>(o_weight_scale.data_ptr()), 1.0,
             "MiniMax-H3 FP8");
}

// NVFP4 route (FlashInfer conventions).  attn_out -> act_q [M, 3584] u8 (E2M1x2) + act_sf [M, 448] UE4M3 u8
// (row-major, block 16) with the caller's activation global scale act_global_scale [1] FP32
// (448 * 6 / amax).  o_weight_q: u8 [5376, 3584], o_weight_sf: u8 with 5376 * 448 entries in the
// FlashInfer 128x4 swizzled layout (fp4_quantize(..., is_sf_swizzled_layout=True)).
// alpha = 1 / (act_global_scale * weight_global_scale) rescales the block-scaled accumulator.
void minimax_h3_sm120_nvfp4_out_proj(TensorView attn_out, TensorView o_weight_q, TensorView o_weight_sf,
                                     TensorView act_global_scale, TensorView gate, TensorView gate_index,
                                     TensorView residual, TensorView act_q, TensorView act_sf, TensorView out,
                                     double alpha) {
  const CommonArgs args = CheckCommon(attn_out, gate, gate_index, residual, out);
  CheckTensor(o_weight_q, "o_weight_q", args.device, dl_uint8, {kHidden, kAttnDim / 2});
  TVM_FFI_CHECK(o_weight_sf.device().device_type == kDLCUDA &&
                    o_weight_sf.device().device_id == args.device.device_id, ValueError)
      << "o_weight_sf must be a CUDA tensor on the same device as attn_out";
  TVM_FFI_CHECK(encode_dlpack_dtype(o_weight_sf.dtype()) == encode_dlpack_dtype(dl_uint8), ValueError)
      << "o_weight_sf must be uint8";
  int64_t sf_numel = 1;
  for (int dim = 0; dim < o_weight_sf.ndim(); ++dim) sf_numel *= o_weight_sf.size(dim);
  TVM_FFI_CHECK(sf_numel == kHidden * kAttnSf && o_weight_sf.IsContiguous(), ValueError)
      << "o_weight_sf must be a contiguous uint8 tensor with 5376 * 448 entries (128x4 swizzled layout)";
  CheckTensor(act_global_scale, "act_global_scale", args.device, dl_float32, {1});
  CheckTensor(act_q, "act_q", args.device, dl_uint8, {args.rows, kAttnDim / 2});
  CheckTensor(act_sf, "act_sf", args.device, dl_uint8, {args.rows, kAttnSf});
  ffi::CUDADeviceGuard device_guard(args.device.device_id);

  const CUtensorMap a_map = EncodeByteTile(act_q.data_ptr(), kAttnDim / 2, args.rows, kBlockK, kBlockM,
                                           kOperandSwizzle, "act_q");
  const CUtensorMap b_map = EncodeByteTile(o_weight_q.data_ptr(), kAttnDim / 2, kHidden, kBlockK, kBlockN,
                                           kOperandSwizzle, "o_weight_q");
  const CUtensorMap sfa_map = EncodeByteTile(act_sf.data_ptr(), kAttnSf, args.rows, kSfaBoxBytes,
                                             kBlockM, CU_TENSOR_MAP_SWIZZLE_NONE, "act_sf");
  // The 128x4 swizzled weight scales are addressed as [42 N tiles x 224 rows, 256 bytes].
  const CUtensorMap sfb_map = EncodeByteTile(o_weight_sf.data_ptr(), 256, int64_t{kNTiles} * kSfbRowsPerNTile,
                                             256, kSfbRowsPerKTile, CU_TENSOR_MAP_SWIZZLE_NONE, "o_weight_sf");
  QuantOutputs q{};
  q.attn_out = static_cast<unsigned int*>(attn_out.data_ptr());
  q.act_q = static_cast<unsigned int*>(act_q.data_ptr());
  q.act_sf = static_cast<uint8_t*>(act_sf.data_ptr());
  q.act_scale = OnesScalar(args.device, args.stream);
  q.act_global_scale = static_cast<float*>(act_global_scale.data_ptr());
  LaunchGemm(1, args, a_map, b_map, sfa_map, sfb_map, q, OnesScalar(args.device, args.stream), alpha,
             "MiniMax-H3 NVFP4");
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(minimax_h3_sm120_fp8_out_proj, minimax_h3_sm120_fp8_out_proj);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(minimax_h3_sm120_nvfp4_out_proj, minimax_h3_sm120_nvfp4_out_proj);
// clang-format on
