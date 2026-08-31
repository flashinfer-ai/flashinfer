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

// Generated Loom device kernels merged into a single translation unit
// following the csrc/tinygemm2.cu convention (one TU = kernel family +
// TVM-FFI binding).
//
// Provenance: six generated Loom schedules, each an exact port of the
// TensorRT-LLM tinygemm2 kernel (csrc/tinygemm2.cu) with bit-identical
// outputs:
//   stage4      — schedule 'flashinfer_tinygemm2' STAGES=4  (USE_PDL=0)
//   stage4_pdl  — schedule 'flashinfer_tinygemm2' STAGES=4  (USE_PDL=1)
//   stage8      — schedule 'flashinfer_tinygemm2' STAGES=8  (USE_PDL=0)
//   stage8_pdl  — schedule 'flashinfer_tinygemm2' STAGES=8  (USE_PDL=1)
//   stage16     — schedule 'flashinfer_tinygemm2' STAGES=16 (USE_PDL=0)
//   stage16_pdl — schedule 'flashinfer_tinygemm2' STAGES=16 (USE_PDL=1)
//
// All six use the trailing by-value __grid_constant__ tensor-map pack ABI
// (LoomTensorMapPack<2>: maps[0] = weight, maps[1] = activation).
//
// The generated TUs are concatenated here by a mechanical transform that
// does not touch kernel code:
//   1. the extern "C" symbol kernel_flashinfer_tinygemm2 is renamed
//      kernel_tinygemm2_sm100_<variant> (one rename per section);
//   2. each section keeps its generated #define block verbatim and #undef's
//      every section-local macro at the section end; SMEM_TOTAL is captured
//      into a per-variant constexpr for the launcher first;
//   3. the fixed-width typedefs and the opaque CUtensorMap typedef of the
//      generated prelude are dropped (the host headers below supply them),
//      and the LoomTensorMap/LoomTensorMapPack structs + __device__ helpers
//      — byte-identical across the six generated TUs — are deduplicated to
//      one copy.
// Kernel bodies are byte-identical to the generated output modulo the symbol
// rename.

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math_constants.h>

#include <cstdint>
#include <cstring>
#include <limits>
#include <mutex>
#include <vector>

#include "tvm_ffi_utils.h"

// clang-format off

struct __align__(128) LoomTensorMap { uint64_t opaque[16]; };
template <int N>
struct __align__(128) LoomTensorMapPack { LoomTensorMap maps[N]; };

__device__ __forceinline__ int make_warp_uniform(int x) {
    int result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1F, 0xFFFFFFFF;"
                 : "=r"(result) : "r"(x));
    return result;
}

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
        :: "r"(mbar_addr), "r"(count));
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


__device__ __forceinline__ void mbarrier_init_pred(int mbar_addr, uint32_t count, uint32_t pred) {
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, %2, 0;\n\t"
        "@p mbarrier.init.shared::cta.b64 [%0], %1;\n\t"
        "}\n" :: "r"(mbar_addr), "r"(count), "r"(pred));
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
__device__ __forceinline__ unsigned int __as_u32(unsigned int v) { return v; }
__device__ __forceinline__ unsigned int __as_u32(int v) {
    unsigned int u;
    asm("mov.b32 %0, %1;" : "=r"(u) : "r"(v));
    return u;
}

// ============================================================================
// Section stage4 — generated Loom schedule 'flashinfer_tinygemm2' (STAGES=4, USE_PDL=0).
// ============================================================================

#define LOOM_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define SMEM_SMEM_WT_OFF 1024
#define SMEM_SMEM_WT_STAGE_BYTES 2048
#define SMEM_SMEM_WT_STRIDE 2048
#define SMEM_SMEM_ACT_OFF 33792
#define SMEM_SMEM_ACT_STAGE_BYTES 1024
#define SMEM_SMEM_ACT_STRIDE 1024
#define SMEM_SMEM_RED_OFF 50176
#define SMEM_SMEM_RED_STAGE_BYTES 2048
#define SMEM_SMEM_RED_STRIDE 2048
#define SMEM_SMEM_BIAS_OFF 52224
#define SMEM_SMEM_BIAS_STAGE_BYTES 32
#define SMEM_SMEM_BIAS_STRIDE 32
#define SMEM_TOTAL 52352
#define THREADS 384
#define USE_PDL 0

extern "C" {

__global__ __launch_bounds__(384, 1) void
kernel_tinygemm2_sm100_stage4(__nv_bfloat16* __restrict__ output, __nv_bfloat16* __restrict__ bias, int M, int N, int K, __grid_constant__ LoomTensorMapPack<2> const _loom_tma_params)
{
    uint64_t _loom_tma_param_base;
    asm volatile("mov.b64 %0, %1;" : "=l"(_loom_tma_param_base) : "l"((uint64_t)(&_loom_tma_params)));

    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // Kernel setup ops
    __nv_bfloat16* smem_wt = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int smem_wt_addr = smem + 1024;
    __nv_bfloat16* smem_act = reinterpret_cast<__nv_bfloat16*>(smem_raw + 33792);
    const int smem_act_addr = smem + 33792;
    float* smem_red = reinterpret_cast<float*>(smem_raw + 50176);
    const int smem_red_addr = smem + 50176;
    __nv_bfloat16* smem_bias = reinterpret_cast<__nv_bfloat16*>(smem_raw + 52224);
    const int smem_bias_addr = smem + 52224;
    if (tid == 0) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)(((const void*)(_loom_tma_param_base + 0)))) : "memory"); }
    if (tid == 0) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)(((const void*)(_loom_tma_param_base + 128)))) : "memory"); }

    // Mbarrier init (3 groups, 12 barriers)
    // Mbarriers at smem_raw[0..96)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        // wt_ready: 4 barriers, init_count=1
        mbarrier_init_pred(smem + 0, 1, leader);
        mbarrier_init_pred(smem + 8, 1, leader);
        mbarrier_init_pred(smem + 16, 1, leader);
        mbarrier_init_pred(smem + 24, 1, leader);
        // act_ready: 4 barriers, init_count=1
        mbarrier_init_pred(smem + 32, 1, leader);
        mbarrier_init_pred(smem + 40, 1, leader);
        mbarrier_init_pred(smem + 48, 1, leader);
        mbarrier_init_pred(smem + 56, 1, leader);
        // data_consumed: 4 barriers, init_count=32
        mbarrier_init_pred(smem + 64, 32, leader);
        mbarrier_init_pred(smem + 72, 32, leader);
        mbarrier_init_pred(smem + 80, 32, leader);
        mbarrier_init_pred(smem + 88, 32, leader);
        asm volatile("fence.mbarrier_init.release.cluster;");
    }

    __syncthreads();

    __syncthreads();

    const int mbar_base = smem;
    #define wt_ready_addr (mbar_base + 0)
    #define act_ready_addr (mbar_base + 32)
    #define data_consumed_addr (mbar_base + 64)

    // ---- Role: compute ----
    if (warp <= 3) {
        { // compute_main
            int k_loops_c = (K + 1024 - 1) / 1024;
            int mib_c = blockIdx.x * 16;
            int ni_c = blockIdx.y * 8;
            if (tid < 16) {
                smem_bias[tid] = bias[mib_c + tid];
            }
            float accum[4];
            #pragma unroll
            for (int z = 0; z < 4; z++) {
                accum[z] = 0.0f;
            }
            unsigned int lane_div8 = lane / 8;
            unsigned int lane_mod8 = lane % 8;
            unsigned int row_wt = lane_mod8 + lane_div8 % 2 * 8;
            unsigned int col_off_wt = lane_div8 / 2;
            unsigned int row_act = lane_mod8;
            #pragma unroll 2
            for (unsigned int ki = 0; ki < k_loops_c; ki++) {
                unsigned int stage_c = (unsigned int)warp;
                unsigned int phase_c = ki & 1;
                mbarrier_wait(wt_ready_addr + (stage_c) * 8, phase_c);
                mbarrier_wait(act_ready_addr + (stage_c) * 8, phase_c);
                #pragma unroll
                for (int su = 0; su < 4; su++) {
                    unsigned int base_wt = smem_wt_addr + (stage_c * 4 + (unsigned int)su) * 2048;
                    unsigned int base_act = smem_act_addr + (stage_c * 4 + (unsigned int)su) * 1024;
                    #pragma unroll
                    for (int kii = 0; kii < 4; kii++) {
                        unsigned int a_frag[4];
                        unsigned int b_frag[2];
                        unsigned int col_w = (unsigned int)(2 * kii) + col_off_wt;
                        unsigned int col_sw_w = row_wt % 8 ^ col_w;
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                            : "r"(base_wt + row_wt * 128 + col_sw_w * 16)
                            : "memory");
                        unsigned int col_a = (unsigned int)(2 * kii) + lane_div8;
                        unsigned int col_sw_a = row_act % 8 ^ col_a;
                        asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                            : "=r"(b_frag[0]), "=r"(b_frag[1])
                            : "r"(base_act + row_act * 128 + col_sw_a * 16)
                            : "memory");
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(accum[0]), "+f"(accum[1]), "+f"(accum[2]), "+f"(accum[3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
                    }
                }
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_arrive(data_consumed_addr + (stage_c) * 8);
            }
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_red_addr + (unsigned int)(tid * 16)), "r"(__as_u32(accum[0])), "r"(__as_u32(accum[1])), "r"(__as_u32(accum[2])), "r"(__as_u32(accum[3])) : "memory");
            asm volatile("barrier.sync 2, 384;" ::: "memory");
            if (warp == 0) {
                float part[12];
                #pragma unroll
                for (int w = 0; w < 3; w++) {
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&part[w * 4])), "=r"(*reinterpret_cast<uint32_t*>(&part[(w * 4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&part[(w * 4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&part[(w * 4) + 3]))
                        : "r"(smem_red_addr + (unsigned int)((32 + w * 32 + tid) * 16)));
                }
                #pragma unroll
                for (int z_1 = 0; z_1 < 4; z_1++) {
                    accum[z_1] = accum[z_1] + part[z_1] + part[4 + z_1] + part[8 + z_1];
                }
                int tm = mib_c + lane / 4;
                int tn = ni_c + 2 * (lane % 4);
                float bias_lo = smem_bias[lane / 4];
                float bias_hi = smem_bias[lane / 4 + 8];
                float o00 = accum[0] + bias_lo;
                float o01 = accum[1] + bias_lo;
                float o10 = accum[2] + bias_hi;
                float o11 = accum[3] + bias_hi;
                if (tn < N) {
                    if (tm < M) {
                        *(reinterpret_cast<__nv_bfloat16*>(output + (tn * M + tm)) + (0)) = __float2bfloat16_rn(o00);
                    }
                }
                if (tn + 1 < N) {
                    if (tm < M) {
                        *(reinterpret_cast<__nv_bfloat16*>(output + ((tn + 1) * M + tm)) + (0)) = __float2bfloat16_rn(o01);
                    }
                }
                if (tn < N) {
                    if (tm + 8 < M) {
                        *(reinterpret_cast<__nv_bfloat16*>(output + (tn * M + tm + 8)) + (0)) = __float2bfloat16_rn(o10);
                    }
                }
                if (tn + 1 < N) {
                    if (tm + 8 < M) {
                        *(reinterpret_cast<__nv_bfloat16*>(output + ((tn + 1) * M + tm + 8)) + (0)) = __float2bfloat16_rn(o11);
                    }
                }
            }
        }
    // ---- Role: load_wt ----
    } else if (warp >= 4 && warp <= 7) {
        { // load_wt_main
            int k_loops = (K + 1024 - 1) / 1024;
            int mib = blockIdx.x * 16;
            unsigned int wslot = warp % 4;
            if (elect_sync()) {
                #pragma unroll 1
                for (unsigned int ki_1 = 0; ki_1 < k_loops; ki_1++) {
                    unsigned int stage = wslot;
                    unsigned int phase = ki_1 & 1;
                    int k_base = (ki_1 * 4 + wslot) * 256;
                    mbarrier_wait(data_consumed_addr + (stage) * 8, phase ^ 1);
                    mbarrier_arrive_expect_tx(wt_ready_addr + (stage) * 8, 8192);
                    #pragma unroll
                    for (int i = 0; i < 4; i++) {
                        int dst_wt = smem_wt_addr + (stage * 4 + (unsigned int)i) * 2048;
                        tma_2d_gmem2smem(dst_wt, ((const void*)(_loom_tma_param_base + 0)), k_base + i * 64, mib, wt_ready_addr + (stage) * 8);
                    }
                }
                #pragma unroll
                for (int di = 0; di < 1; di++) {
                    if (di + 1 < 1) {
                        unsigned int dki = k_loops + di;
                        unsigned int dstage = wslot;
                        unsigned int dphase = dki & 1;
                        mbarrier_wait(data_consumed_addr + (dstage) * 8, dphase ^ 1);
                    }
                }
            }
            asm volatile("barrier.sync 2, 384;" ::: "memory");
        }
    // ---- Role: load_act ----
    } else if (warp >= 8 && warp <= 11) {
        { // load_act_main
            int k_loops_a = (K + 1024 - 1) / 1024;
            int ni = blockIdx.y * 8;
            unsigned int aslot = warp % 4;
            if (elect_sync()) {
                #pragma unroll 1
                for (unsigned int ki_2 = 0; ki_2 < k_loops_a; ki_2++) {
                    unsigned int stage_a = aslot;
                    unsigned int phase_a = ki_2 & 1;
                    int k_base_a = (ki_2 * 4 + aslot) * 256;
                    mbarrier_wait(data_consumed_addr + (stage_a) * 8, phase_a ^ 1);
                    mbarrier_arrive_expect_tx(act_ready_addr + (stage_a) * 8, 4096);
                    #pragma unroll
                    for (int i_1 = 0; i_1 < 4; i_1++) {
                        int dst_act = smem_act_addr + (stage_a * 4 + (unsigned int)i_1) * 1024;
                        tma_2d_gmem2smem(dst_act, ((const void*)(_loom_tma_param_base + 128)), k_base_a + i_1 * 64, ni, act_ready_addr + (stage_a) * 8);
                    }
                }
                #pragma unroll
                for (int di_1 = 0; di_1 < 1; di_1++) {
                    if (di_1 + 1 < 1) {
                        unsigned int dki_a = k_loops_a + di_1;
                        unsigned int dstage_a = aslot;
                        unsigned int dphase_a = dki_a & 1;
                        mbarrier_wait(data_consumed_addr + (dstage_a) * 8, dphase_a ^ 1);
                    }
                }
            }
            asm volatile("barrier.sync 2, 384;" ::: "memory");
        }
    }

    // Cleanup
}

} // extern "C"

constexpr int kSmemBytesStage4 = SMEM_TOTAL;
static_assert(kSmemBytesStage4 == 52352,
              "generated SMEM footprint for stage4 changed; update the launcher expectations");

#undef LOOM_INF
#undef NUM_MAIN_STAGES
#undef SMEM_SMEM_WT_OFF
#undef SMEM_SMEM_WT_STAGE_BYTES
#undef SMEM_SMEM_WT_STRIDE
#undef SMEM_SMEM_ACT_OFF
#undef SMEM_SMEM_ACT_STAGE_BYTES
#undef SMEM_SMEM_ACT_STRIDE
#undef SMEM_SMEM_RED_OFF
#undef SMEM_SMEM_RED_STAGE_BYTES
#undef SMEM_SMEM_RED_STRIDE
#undef SMEM_SMEM_BIAS_OFF
#undef SMEM_SMEM_BIAS_STAGE_BYTES
#undef SMEM_SMEM_BIAS_STRIDE
#undef SMEM_TOTAL
#undef THREADS
#undef USE_PDL

// ============================================================================
// Section stage4_pdl — generated Loom schedule 'flashinfer_tinygemm2' (STAGES=4, USE_PDL=1).
// ============================================================================

#define LOOM_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define SMEM_SMEM_WT_OFF 1024
#define SMEM_SMEM_WT_STAGE_BYTES 2048
#define SMEM_SMEM_WT_STRIDE 2048
#define SMEM_SMEM_ACT_OFF 33792
#define SMEM_SMEM_ACT_STAGE_BYTES 1024
#define SMEM_SMEM_ACT_STRIDE 1024
#define SMEM_SMEM_RED_OFF 50176
#define SMEM_SMEM_RED_STAGE_BYTES 2048
#define SMEM_SMEM_RED_STRIDE 2048
#define SMEM_SMEM_BIAS_OFF 52224
#define SMEM_SMEM_BIAS_STAGE_BYTES 32
#define SMEM_SMEM_BIAS_STRIDE 32
#define SMEM_TOTAL 52352
#define THREADS 384
#define USE_PDL 1

extern "C" {

__global__ __launch_bounds__(384, 1) void
kernel_tinygemm2_sm100_stage4_pdl(__nv_bfloat16* __restrict__ output, __nv_bfloat16* __restrict__ bias, int M, int N, int K, __grid_constant__ LoomTensorMapPack<2> const _loom_tma_params)
{
    uint64_t _loom_tma_param_base;
    asm volatile("mov.b64 %0, %1;" : "=l"(_loom_tma_param_base) : "l"((uint64_t)(&_loom_tma_params)));

    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // Kernel setup ops
    __nv_bfloat16* smem_wt = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int smem_wt_addr = smem + 1024;
    __nv_bfloat16* smem_act = reinterpret_cast<__nv_bfloat16*>(smem_raw + 33792);
    const int smem_act_addr = smem + 33792;
    float* smem_red = reinterpret_cast<float*>(smem_raw + 50176);
    const int smem_red_addr = smem + 50176;
    __nv_bfloat16* smem_bias = reinterpret_cast<__nv_bfloat16*>(smem_raw + 52224);
    const int smem_bias_addr = smem + 52224;
    if (tid == 0) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)(((const void*)(_loom_tma_param_base + 0)))) : "memory"); }
    if (tid == 0) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)(((const void*)(_loom_tma_param_base + 128)))) : "memory"); }

    // Mbarrier init (3 groups, 12 barriers)
    // Mbarriers at smem_raw[0..96)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        // wt_ready: 4 barriers, init_count=1
        mbarrier_init_pred(smem + 0, 1, leader);
        mbarrier_init_pred(smem + 8, 1, leader);
        mbarrier_init_pred(smem + 16, 1, leader);
        mbarrier_init_pred(smem + 24, 1, leader);
        // act_ready: 4 barriers, init_count=1
        mbarrier_init_pred(smem + 32, 1, leader);
        mbarrier_init_pred(smem + 40, 1, leader);
        mbarrier_init_pred(smem + 48, 1, leader);
        mbarrier_init_pred(smem + 56, 1, leader);
        // data_consumed: 4 barriers, init_count=32
        mbarrier_init_pred(smem + 64, 32, leader);
        mbarrier_init_pred(smem + 72, 32, leader);
        mbarrier_init_pred(smem + 80, 32, leader);
        mbarrier_init_pred(smem + 88, 32, leader);
        asm volatile("fence.mbarrier_init.release.cluster;");
    }

    __syncthreads();

    __syncthreads();

    const int mbar_base = smem;
    #define wt_ready_addr (mbar_base + 0)
    #define act_ready_addr (mbar_base + 32)
    #define data_consumed_addr (mbar_base + 64)

    // ---- Role: compute ----
    if (warp <= 3) {
        { // compute_main
            int k_loops_c = (K + 1024 - 1) / 1024;
            int mib_c = blockIdx.x * 16;
            int ni_c = blockIdx.y * 8;
            if (tid < 16) {
                smem_bias[tid] = bias[mib_c + tid];
            }
            float accum[4];
            #pragma unroll
            for (int z = 0; z < 4; z++) {
                accum[z] = 0.0f;
            }
            unsigned int lane_div8 = lane / 8;
            unsigned int lane_mod8 = lane % 8;
            unsigned int row_wt = lane_mod8 + lane_div8 % 2 * 8;
            unsigned int col_off_wt = lane_div8 / 2;
            unsigned int row_act = lane_mod8;
            #pragma unroll 2
            for (unsigned int ki = 0; ki < k_loops_c; ki++) {
                unsigned int stage_c = (unsigned int)warp;
                unsigned int phase_c = ki & 1;
                mbarrier_wait(wt_ready_addr + (stage_c) * 8, phase_c);
                mbarrier_wait(act_ready_addr + (stage_c) * 8, phase_c);
                #pragma unroll
                for (int su = 0; su < 4; su++) {
                    unsigned int base_wt = smem_wt_addr + (stage_c * 4 + (unsigned int)su) * 2048;
                    unsigned int base_act = smem_act_addr + (stage_c * 4 + (unsigned int)su) * 1024;
                    #pragma unroll
                    for (int kii = 0; kii < 4; kii++) {
                        unsigned int a_frag[4];
                        unsigned int b_frag[2];
                        unsigned int col_w = (unsigned int)(2 * kii) + col_off_wt;
                        unsigned int col_sw_w = row_wt % 8 ^ col_w;
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                            : "r"(base_wt + row_wt * 128 + col_sw_w * 16)
                            : "memory");
                        unsigned int col_a = (unsigned int)(2 * kii) + lane_div8;
                        unsigned int col_sw_a = row_act % 8 ^ col_a;
                        asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                            : "=r"(b_frag[0]), "=r"(b_frag[1])
                            : "r"(base_act + row_act * 128 + col_sw_a * 16)
                            : "memory");
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(accum[0]), "+f"(accum[1]), "+f"(accum[2]), "+f"(accum[3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
                    }
                }
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_arrive(data_consumed_addr + (stage_c) * 8);
            }
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_red_addr + (unsigned int)(tid * 16)), "r"(__as_u32(accum[0])), "r"(__as_u32(accum[1])), "r"(__as_u32(accum[2])), "r"(__as_u32(accum[3])) : "memory");
            asm volatile("barrier.sync 2, 384;" ::: "memory");
            if (warp == 0) {
                float part[12];
                #pragma unroll
                for (int w = 0; w < 3; w++) {
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&part[w * 4])), "=r"(*reinterpret_cast<uint32_t*>(&part[(w * 4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&part[(w * 4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&part[(w * 4) + 3]))
                        : "r"(smem_red_addr + (unsigned int)((32 + w * 32 + tid) * 16)));
                }
                #pragma unroll
                for (int z_1 = 0; z_1 < 4; z_1++) {
                    accum[z_1] = accum[z_1] + part[z_1] + part[4 + z_1] + part[8 + z_1];
                }
                int tm = mib_c + lane / 4;
                int tn = ni_c + 2 * (lane % 4);
                float bias_lo = smem_bias[lane / 4];
                float bias_hi = smem_bias[lane / 4 + 8];
                float o00 = accum[0] + bias_lo;
                float o01 = accum[1] + bias_lo;
                float o10 = accum[2] + bias_hi;
                float o11 = accum[3] + bias_hi;
                if (tn < N) {
                    if (tm < M) {
                        *(reinterpret_cast<__nv_bfloat16*>(output + (tn * M + tm)) + (0)) = __float2bfloat16_rn(o00);
                    }
                }
                if (tn + 1 < N) {
                    if (tm < M) {
                        *(reinterpret_cast<__nv_bfloat16*>(output + ((tn + 1) * M + tm)) + (0)) = __float2bfloat16_rn(o01);
                    }
                }
                if (tn < N) {
                    if (tm + 8 < M) {
                        *(reinterpret_cast<__nv_bfloat16*>(output + (tn * M + tm + 8)) + (0)) = __float2bfloat16_rn(o10);
                    }
                }
                if (tn + 1 < N) {
                    if (tm + 8 < M) {
                        *(reinterpret_cast<__nv_bfloat16*>(output + ((tn + 1) * M + tm + 8)) + (0)) = __float2bfloat16_rn(o11);
                    }
                }
            }
        }
    // ---- Role: load_wt ----
    } else if (warp >= 4 && warp <= 7) {
        { // load_wt_main
            int k_loops = (K + 1024 - 1) / 1024;
            int mib = blockIdx.x * 16;
            unsigned int wslot = warp % 4;
            if (elect_sync()) {
                #pragma unroll 1
                for (unsigned int ki_1 = 0; ki_1 < k_loops; ki_1++) {
                    unsigned int stage = wslot;
                    unsigned int phase = ki_1 & 1;
                    int k_base = (ki_1 * 4 + wslot) * 256;
                    mbarrier_wait(data_consumed_addr + (stage) * 8, phase ^ 1);
                    mbarrier_arrive_expect_tx(wt_ready_addr + (stage) * 8, 8192);
                    #pragma unroll
                    for (int i = 0; i < 4; i++) {
                        int dst_wt = smem_wt_addr + (stage * 4 + (unsigned int)i) * 2048;
                        tma_2d_gmem2smem(dst_wt, ((const void*)(_loom_tma_param_base + 0)), k_base + i * 64, mib, wt_ready_addr + (stage) * 8);
                    }
                }
                #pragma unroll
                for (int di = 0; di < 1; di++) {
                    if (di + 1 < 1) {
                        unsigned int dki = k_loops + di;
                        unsigned int dstage = wslot;
                        unsigned int dphase = dki & 1;
                        mbarrier_wait(data_consumed_addr + (dstage) * 8, dphase ^ 1);
                    }
                }
            }
            asm volatile("barrier.sync 2, 384;" ::: "memory");
        }
    // ---- Role: load_act ----
    } else if (warp >= 8 && warp <= 11) {
        { // load_act_main
            int k_loops_a = (K + 1024 - 1) / 1024;
            int ni = blockIdx.y * 8;
            unsigned int aslot = warp % 4;
            if (elect_sync()) {
                {
                    asm volatile("griddepcontrol.wait;" ::: "memory");
                    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
                }
                #pragma unroll 1
                for (unsigned int ki_2 = 0; ki_2 < k_loops_a; ki_2++) {
                    unsigned int stage_a = aslot;
                    unsigned int phase_a = ki_2 & 1;
                    int k_base_a = (ki_2 * 4 + aslot) * 256;
                    mbarrier_wait(data_consumed_addr + (stage_a) * 8, phase_a ^ 1);
                    mbarrier_arrive_expect_tx(act_ready_addr + (stage_a) * 8, 4096);
                    #pragma unroll
                    for (int i_1 = 0; i_1 < 4; i_1++) {
                        int dst_act = smem_act_addr + (stage_a * 4 + (unsigned int)i_1) * 1024;
                        tma_2d_gmem2smem(dst_act, ((const void*)(_loom_tma_param_base + 128)), k_base_a + i_1 * 64, ni, act_ready_addr + (stage_a) * 8);
                    }
                }
                #pragma unroll
                for (int di_1 = 0; di_1 < 1; di_1++) {
                    if (di_1 + 1 < 1) {
                        unsigned int dki_a = k_loops_a + di_1;
                        unsigned int dstage_a = aslot;
                        unsigned int dphase_a = dki_a & 1;
                        mbarrier_wait(data_consumed_addr + (dstage_a) * 8, dphase_a ^ 1);
                    }
                }
            }
            asm volatile("barrier.sync 2, 384;" ::: "memory");
        }
    }

    // Cleanup
}

} // extern "C"

constexpr int kSmemBytesStage4Pdl = SMEM_TOTAL;
static_assert(kSmemBytesStage4Pdl == 52352,
              "generated SMEM footprint for stage4_pdl changed; update the launcher expectations");

#undef LOOM_INF
#undef NUM_MAIN_STAGES
#undef SMEM_SMEM_WT_OFF
#undef SMEM_SMEM_WT_STAGE_BYTES
#undef SMEM_SMEM_WT_STRIDE
#undef SMEM_SMEM_ACT_OFF
#undef SMEM_SMEM_ACT_STAGE_BYTES
#undef SMEM_SMEM_ACT_STRIDE
#undef SMEM_SMEM_RED_OFF
#undef SMEM_SMEM_RED_STAGE_BYTES
#undef SMEM_SMEM_RED_STRIDE
#undef SMEM_SMEM_BIAS_OFF
#undef SMEM_SMEM_BIAS_STAGE_BYTES
#undef SMEM_SMEM_BIAS_STRIDE
#undef SMEM_TOTAL
#undef THREADS
#undef USE_PDL

// ============================================================================
// Section stage8 — generated Loom schedule 'flashinfer_tinygemm2' (STAGES=8, USE_PDL=0).
// ============================================================================

#define LOOM_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define SMEM_SMEM_WT_OFF 1024
#define SMEM_SMEM_WT_STAGE_BYTES 2048
#define SMEM_SMEM_WT_STRIDE 2048
#define SMEM_SMEM_ACT_OFF 66560
#define SMEM_SMEM_ACT_STAGE_BYTES 1024
#define SMEM_SMEM_ACT_STRIDE 1024
#define SMEM_SMEM_RED_OFF 99328
#define SMEM_SMEM_RED_STAGE_BYTES 2048
#define SMEM_SMEM_RED_STRIDE 2048
#define SMEM_SMEM_BIAS_OFF 101376
#define SMEM_SMEM_BIAS_STAGE_BYTES 32
#define SMEM_SMEM_BIAS_STRIDE 32
#define SMEM_TOTAL 101504
#define THREADS 384
#define USE_PDL 0

extern "C" {

__global__ __launch_bounds__(384, 1) void
kernel_tinygemm2_sm100_stage8(__nv_bfloat16* __restrict__ output, __nv_bfloat16* __restrict__ bias, int M, int N, int K, __grid_constant__ LoomTensorMapPack<2> const _loom_tma_params)
{
    uint64_t _loom_tma_param_base;
    asm volatile("mov.b64 %0, %1;" : "=l"(_loom_tma_param_base) : "l"((uint64_t)(&_loom_tma_params)));

    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // Kernel setup ops
    __nv_bfloat16* smem_wt = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int smem_wt_addr = smem + 1024;
    __nv_bfloat16* smem_act = reinterpret_cast<__nv_bfloat16*>(smem_raw + 66560);
    const int smem_act_addr = smem + 66560;
    float* smem_red = reinterpret_cast<float*>(smem_raw + 99328);
    const int smem_red_addr = smem + 99328;
    __nv_bfloat16* smem_bias = reinterpret_cast<__nv_bfloat16*>(smem_raw + 101376);
    const int smem_bias_addr = smem + 101376;
    if (tid == 0) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)(((const void*)(_loom_tma_param_base + 0)))) : "memory"); }
    if (tid == 0) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)(((const void*)(_loom_tma_param_base + 128)))) : "memory"); }

    // Mbarrier init (3 groups, 24 barriers)
    // Mbarriers at smem_raw[0..192)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        // wt_ready: 8 barriers, init_count=1
        mbarrier_init_pred(smem + 0, 1, leader);
        mbarrier_init_pred(smem + 8, 1, leader);
        mbarrier_init_pred(smem + 16, 1, leader);
        mbarrier_init_pred(smem + 24, 1, leader);
        mbarrier_init_pred(smem + 32, 1, leader);
        mbarrier_init_pred(smem + 40, 1, leader);
        mbarrier_init_pred(smem + 48, 1, leader);
        mbarrier_init_pred(smem + 56, 1, leader);
        // act_ready: 8 barriers, init_count=1
        mbarrier_init_pred(smem + 64, 1, leader);
        mbarrier_init_pred(smem + 72, 1, leader);
        mbarrier_init_pred(smem + 80, 1, leader);
        mbarrier_init_pred(smem + 88, 1, leader);
        mbarrier_init_pred(smem + 96, 1, leader);
        mbarrier_init_pred(smem + 104, 1, leader);
        mbarrier_init_pred(smem + 112, 1, leader);
        mbarrier_init_pred(smem + 120, 1, leader);
        // data_consumed: 8 barriers, init_count=32
        mbarrier_init_pred(smem + 128, 32, leader);
        mbarrier_init_pred(smem + 136, 32, leader);
        mbarrier_init_pred(smem + 144, 32, leader);
        mbarrier_init_pred(smem + 152, 32, leader);
        mbarrier_init_pred(smem + 160, 32, leader);
        mbarrier_init_pred(smem + 168, 32, leader);
        mbarrier_init_pred(smem + 176, 32, leader);
        mbarrier_init_pred(smem + 184, 32, leader);
        asm volatile("fence.mbarrier_init.release.cluster;");
    }

    __syncthreads();

    __syncthreads();

    const int mbar_base = smem;
    #define wt_ready_addr (mbar_base + 0)
    #define act_ready_addr (mbar_base + 64)
    #define data_consumed_addr (mbar_base + 128)

    // ---- Role: compute ----
    if (warp <= 3) {
        { // compute_main
            int k_loops_c = (K + 1024 - 1) / 1024;
            int mib_c = blockIdx.x * 16;
            int ni_c = blockIdx.y * 8;
            if (tid < 16) {
                smem_bias[tid] = bias[mib_c + tid];
            }
            float accum[4];
            #pragma unroll
            for (int z = 0; z < 4; z++) {
                accum[z] = 0.0f;
            }
            unsigned int lane_div8 = lane / 8;
            unsigned int lane_mod8 = lane % 8;
            unsigned int row_wt = lane_mod8 + lane_div8 % 2 * 8;
            unsigned int col_off_wt = lane_div8 / 2;
            unsigned int row_act = lane_mod8;
            #pragma unroll 2
            for (unsigned int ki = 0; ki < k_loops_c; ki++) {
                unsigned int stage_c = (unsigned int)warp + 4 * (ki % 2);
                unsigned int phase_c = ki / 2 & 1;
                mbarrier_wait(wt_ready_addr + (stage_c) * 8, phase_c);
                mbarrier_wait(act_ready_addr + (stage_c) * 8, phase_c);
                #pragma unroll
                for (int su = 0; su < 4; su++) {
                    unsigned int base_wt = smem_wt_addr + (stage_c * 4 + (unsigned int)su) * 2048;
                    unsigned int base_act = smem_act_addr + (stage_c * 4 + (unsigned int)su) * 1024;
                    #pragma unroll
                    for (int kii = 0; kii < 4; kii++) {
                        unsigned int a_frag[4];
                        unsigned int b_frag[2];
                        unsigned int col_w = (unsigned int)(2 * kii) + col_off_wt;
                        unsigned int col_sw_w = row_wt % 8 ^ col_w;
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                            : "r"(base_wt + row_wt * 128 + col_sw_w * 16)
                            : "memory");
                        unsigned int col_a = (unsigned int)(2 * kii) + lane_div8;
                        unsigned int col_sw_a = row_act % 8 ^ col_a;
                        asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                            : "=r"(b_frag[0]), "=r"(b_frag[1])
                            : "r"(base_act + row_act * 128 + col_sw_a * 16)
                            : "memory");
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(accum[0]), "+f"(accum[1]), "+f"(accum[2]), "+f"(accum[3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
                    }
                }
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_arrive(data_consumed_addr + (stage_c) * 8);
            }
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_red_addr + (unsigned int)(tid * 16)), "r"(__as_u32(accum[0])), "r"(__as_u32(accum[1])), "r"(__as_u32(accum[2])), "r"(__as_u32(accum[3])) : "memory");
            asm volatile("barrier.sync 2, 384;" ::: "memory");
            if (warp == 0) {
                float part[12];
                #pragma unroll
                for (int w = 0; w < 3; w++) {
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&part[w * 4])), "=r"(*reinterpret_cast<uint32_t*>(&part[(w * 4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&part[(w * 4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&part[(w * 4) + 3]))
                        : "r"(smem_red_addr + (unsigned int)((32 + w * 32 + tid) * 16)));
                }
                #pragma unroll
                for (int z_1 = 0; z_1 < 4; z_1++) {
                    accum[z_1] = accum[z_1] + part[z_1] + part[4 + z_1] + part[8 + z_1];
                }
                int tm = mib_c + lane / 4;
                int tn = ni_c + 2 * (lane % 4);
                float bias_lo = smem_bias[lane / 4];
                float bias_hi = smem_bias[lane / 4 + 8];
                float o00 = accum[0] + bias_lo;
                float o01 = accum[1] + bias_lo;
                float o10 = accum[2] + bias_hi;
                float o11 = accum[3] + bias_hi;
                if (tn < N) {
                    if (tm < M) {
                        *(reinterpret_cast<__nv_bfloat16*>(output + (tn * M + tm)) + (0)) = __float2bfloat16_rn(o00);
                    }
                }
                if (tn + 1 < N) {
                    if (tm < M) {
                        *(reinterpret_cast<__nv_bfloat16*>(output + ((tn + 1) * M + tm)) + (0)) = __float2bfloat16_rn(o01);
                    }
                }
                if (tn < N) {
                    if (tm + 8 < M) {
                        *(reinterpret_cast<__nv_bfloat16*>(output + (tn * M + tm + 8)) + (0)) = __float2bfloat16_rn(o10);
                    }
                }
                if (tn + 1 < N) {
                    if (tm + 8 < M) {
                        *(reinterpret_cast<__nv_bfloat16*>(output + ((tn + 1) * M + tm + 8)) + (0)) = __float2bfloat16_rn(o11);
                    }
                }
            }
        }
    // ---- Role: load_wt ----
    } else if (warp >= 4 && warp <= 7) {
        { // load_wt_main
            int k_loops = (K + 1024 - 1) / 1024;
            int mib = blockIdx.x * 16;
            unsigned int wslot = warp % 4;
            if (elect_sync()) {
                #pragma unroll 1
                for (unsigned int ki_1 = 0; ki_1 < k_loops; ki_1++) {
                    unsigned int stage = wslot + 4 * (ki_1 % 2);
                    unsigned int phase = ki_1 / 2 & 1;
                    int k_base = (ki_1 * 4 + wslot) * 256;
                    mbarrier_wait(data_consumed_addr + (stage) * 8, phase ^ 1);
                    mbarrier_arrive_expect_tx(wt_ready_addr + (stage) * 8, 8192);
                    #pragma unroll
                    for (int i = 0; i < 4; i++) {
                        int dst_wt = smem_wt_addr + (stage * 4 + (unsigned int)i) * 2048;
                        tma_2d_gmem2smem(dst_wt, ((const void*)(_loom_tma_param_base + 0)), k_base + i * 64, mib, wt_ready_addr + (stage) * 8);
                    }
                }
                #pragma unroll
                for (int di = 0; di < 2; di++) {
                    if (di + 1 < 2) {
                        unsigned int dki = k_loops + di;
                        unsigned int dstage = wslot + 4 * (dki % 2);
                        unsigned int dphase = dki / 2 & 1;
                        mbarrier_wait(data_consumed_addr + (dstage) * 8, dphase ^ 1);
                    }
                }
            }
            asm volatile("barrier.sync 2, 384;" ::: "memory");
        }
    // ---- Role: load_act ----
    } else if (warp >= 8 && warp <= 11) {
        { // load_act_main
            int k_loops_a = (K + 1024 - 1) / 1024;
            int ni = blockIdx.y * 8;
            unsigned int aslot = warp % 4;
            if (elect_sync()) {
                #pragma unroll 1
                for (unsigned int ki_2 = 0; ki_2 < k_loops_a; ki_2++) {
                    unsigned int stage_a = aslot + 4 * (ki_2 % 2);
                    unsigned int phase_a = ki_2 / 2 & 1;
                    int k_base_a = (ki_2 * 4 + aslot) * 256;
                    mbarrier_wait(data_consumed_addr + (stage_a) * 8, phase_a ^ 1);
                    mbarrier_arrive_expect_tx(act_ready_addr + (stage_a) * 8, 4096);
                    #pragma unroll
                    for (int i_1 = 0; i_1 < 4; i_1++) {
                        int dst_act = smem_act_addr + (stage_a * 4 + (unsigned int)i_1) * 1024;
                        tma_2d_gmem2smem(dst_act, ((const void*)(_loom_tma_param_base + 128)), k_base_a + i_1 * 64, ni, act_ready_addr + (stage_a) * 8);
                    }
                }
                #pragma unroll
                for (int di_1 = 0; di_1 < 2; di_1++) {
                    if (di_1 + 1 < 2) {
                        unsigned int dki_a = k_loops_a + di_1;
                        unsigned int dstage_a = aslot + 4 * (dki_a % 2);
                        unsigned int dphase_a = dki_a / 2 & 1;
                        mbarrier_wait(data_consumed_addr + (dstage_a) * 8, dphase_a ^ 1);
                    }
                }
            }
            asm volatile("barrier.sync 2, 384;" ::: "memory");
        }
    }

    // Cleanup
}

} // extern "C"

constexpr int kSmemBytesStage8 = SMEM_TOTAL;
static_assert(kSmemBytesStage8 == 101504,
              "generated SMEM footprint for stage8 changed; update the launcher expectations");

#undef LOOM_INF
#undef NUM_MAIN_STAGES
#undef SMEM_SMEM_WT_OFF
#undef SMEM_SMEM_WT_STAGE_BYTES
#undef SMEM_SMEM_WT_STRIDE
#undef SMEM_SMEM_ACT_OFF
#undef SMEM_SMEM_ACT_STAGE_BYTES
#undef SMEM_SMEM_ACT_STRIDE
#undef SMEM_SMEM_RED_OFF
#undef SMEM_SMEM_RED_STAGE_BYTES
#undef SMEM_SMEM_RED_STRIDE
#undef SMEM_SMEM_BIAS_OFF
#undef SMEM_SMEM_BIAS_STAGE_BYTES
#undef SMEM_SMEM_BIAS_STRIDE
#undef SMEM_TOTAL
#undef THREADS
#undef USE_PDL

// ============================================================================
// Section stage8_pdl — generated Loom schedule 'flashinfer_tinygemm2' (STAGES=8, USE_PDL=1).
// ============================================================================

#define LOOM_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define SMEM_SMEM_WT_OFF 1024
#define SMEM_SMEM_WT_STAGE_BYTES 2048
#define SMEM_SMEM_WT_STRIDE 2048
#define SMEM_SMEM_ACT_OFF 66560
#define SMEM_SMEM_ACT_STAGE_BYTES 1024
#define SMEM_SMEM_ACT_STRIDE 1024
#define SMEM_SMEM_RED_OFF 99328
#define SMEM_SMEM_RED_STAGE_BYTES 2048
#define SMEM_SMEM_RED_STRIDE 2048
#define SMEM_SMEM_BIAS_OFF 101376
#define SMEM_SMEM_BIAS_STAGE_BYTES 32
#define SMEM_SMEM_BIAS_STRIDE 32
#define SMEM_TOTAL 101504
#define THREADS 384
#define USE_PDL 1

extern "C" {

__global__ __launch_bounds__(384, 1) void
kernel_tinygemm2_sm100_stage8_pdl(__nv_bfloat16* __restrict__ output, __nv_bfloat16* __restrict__ bias, int M, int N, int K, __grid_constant__ LoomTensorMapPack<2> const _loom_tma_params)
{
    uint64_t _loom_tma_param_base;
    asm volatile("mov.b64 %0, %1;" : "=l"(_loom_tma_param_base) : "l"((uint64_t)(&_loom_tma_params)));

    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // Kernel setup ops
    __nv_bfloat16* smem_wt = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int smem_wt_addr = smem + 1024;
    __nv_bfloat16* smem_act = reinterpret_cast<__nv_bfloat16*>(smem_raw + 66560);
    const int smem_act_addr = smem + 66560;
    float* smem_red = reinterpret_cast<float*>(smem_raw + 99328);
    const int smem_red_addr = smem + 99328;
    __nv_bfloat16* smem_bias = reinterpret_cast<__nv_bfloat16*>(smem_raw + 101376);
    const int smem_bias_addr = smem + 101376;
    if (tid == 0) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)(((const void*)(_loom_tma_param_base + 0)))) : "memory"); }
    if (tid == 0) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)(((const void*)(_loom_tma_param_base + 128)))) : "memory"); }

    // Mbarrier init (3 groups, 24 barriers)
    // Mbarriers at smem_raw[0..192)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        // wt_ready: 8 barriers, init_count=1
        mbarrier_init_pred(smem + 0, 1, leader);
        mbarrier_init_pred(smem + 8, 1, leader);
        mbarrier_init_pred(smem + 16, 1, leader);
        mbarrier_init_pred(smem + 24, 1, leader);
        mbarrier_init_pred(smem + 32, 1, leader);
        mbarrier_init_pred(smem + 40, 1, leader);
        mbarrier_init_pred(smem + 48, 1, leader);
        mbarrier_init_pred(smem + 56, 1, leader);
        // act_ready: 8 barriers, init_count=1
        mbarrier_init_pred(smem + 64, 1, leader);
        mbarrier_init_pred(smem + 72, 1, leader);
        mbarrier_init_pred(smem + 80, 1, leader);
        mbarrier_init_pred(smem + 88, 1, leader);
        mbarrier_init_pred(smem + 96, 1, leader);
        mbarrier_init_pred(smem + 104, 1, leader);
        mbarrier_init_pred(smem + 112, 1, leader);
        mbarrier_init_pred(smem + 120, 1, leader);
        // data_consumed: 8 barriers, init_count=32
        mbarrier_init_pred(smem + 128, 32, leader);
        mbarrier_init_pred(smem + 136, 32, leader);
        mbarrier_init_pred(smem + 144, 32, leader);
        mbarrier_init_pred(smem + 152, 32, leader);
        mbarrier_init_pred(smem + 160, 32, leader);
        mbarrier_init_pred(smem + 168, 32, leader);
        mbarrier_init_pred(smem + 176, 32, leader);
        mbarrier_init_pred(smem + 184, 32, leader);
        asm volatile("fence.mbarrier_init.release.cluster;");
    }

    __syncthreads();

    __syncthreads();

    const int mbar_base = smem;
    #define wt_ready_addr (mbar_base + 0)
    #define act_ready_addr (mbar_base + 64)
    #define data_consumed_addr (mbar_base + 128)

    // ---- Role: compute ----
    if (warp <= 3) {
        { // compute_main
            int k_loops_c = (K + 1024 - 1) / 1024;
            int mib_c = blockIdx.x * 16;
            int ni_c = blockIdx.y * 8;
            if (tid < 16) {
                smem_bias[tid] = bias[mib_c + tid];
            }
            float accum[4];
            #pragma unroll
            for (int z = 0; z < 4; z++) {
                accum[z] = 0.0f;
            }
            unsigned int lane_div8 = lane / 8;
            unsigned int lane_mod8 = lane % 8;
            unsigned int row_wt = lane_mod8 + lane_div8 % 2 * 8;
            unsigned int col_off_wt = lane_div8 / 2;
            unsigned int row_act = lane_mod8;
            #pragma unroll 2
            for (unsigned int ki = 0; ki < k_loops_c; ki++) {
                unsigned int stage_c = (unsigned int)warp + 4 * (ki % 2);
                unsigned int phase_c = ki / 2 & 1;
                mbarrier_wait(wt_ready_addr + (stage_c) * 8, phase_c);
                mbarrier_wait(act_ready_addr + (stage_c) * 8, phase_c);
                #pragma unroll
                for (int su = 0; su < 4; su++) {
                    unsigned int base_wt = smem_wt_addr + (stage_c * 4 + (unsigned int)su) * 2048;
                    unsigned int base_act = smem_act_addr + (stage_c * 4 + (unsigned int)su) * 1024;
                    #pragma unroll
                    for (int kii = 0; kii < 4; kii++) {
                        unsigned int a_frag[4];
                        unsigned int b_frag[2];
                        unsigned int col_w = (unsigned int)(2 * kii) + col_off_wt;
                        unsigned int col_sw_w = row_wt % 8 ^ col_w;
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                            : "r"(base_wt + row_wt * 128 + col_sw_w * 16)
                            : "memory");
                        unsigned int col_a = (unsigned int)(2 * kii) + lane_div8;
                        unsigned int col_sw_a = row_act % 8 ^ col_a;
                        asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                            : "=r"(b_frag[0]), "=r"(b_frag[1])
                            : "r"(base_act + row_act * 128 + col_sw_a * 16)
                            : "memory");
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(accum[0]), "+f"(accum[1]), "+f"(accum[2]), "+f"(accum[3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
                    }
                }
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_arrive(data_consumed_addr + (stage_c) * 8);
            }
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_red_addr + (unsigned int)(tid * 16)), "r"(__as_u32(accum[0])), "r"(__as_u32(accum[1])), "r"(__as_u32(accum[2])), "r"(__as_u32(accum[3])) : "memory");
            asm volatile("barrier.sync 2, 384;" ::: "memory");
            if (warp == 0) {
                float part[12];
                #pragma unroll
                for (int w = 0; w < 3; w++) {
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&part[w * 4])), "=r"(*reinterpret_cast<uint32_t*>(&part[(w * 4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&part[(w * 4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&part[(w * 4) + 3]))
                        : "r"(smem_red_addr + (unsigned int)((32 + w * 32 + tid) * 16)));
                }
                #pragma unroll
                for (int z_1 = 0; z_1 < 4; z_1++) {
                    accum[z_1] = accum[z_1] + part[z_1] + part[4 + z_1] + part[8 + z_1];
                }
                int tm = mib_c + lane / 4;
                int tn = ni_c + 2 * (lane % 4);
                float bias_lo = smem_bias[lane / 4];
                float bias_hi = smem_bias[lane / 4 + 8];
                float o00 = accum[0] + bias_lo;
                float o01 = accum[1] + bias_lo;
                float o10 = accum[2] + bias_hi;
                float o11 = accum[3] + bias_hi;
                if (tn < N) {
                    if (tm < M) {
                        *(reinterpret_cast<__nv_bfloat16*>(output + (tn * M + tm)) + (0)) = __float2bfloat16_rn(o00);
                    }
                }
                if (tn + 1 < N) {
                    if (tm < M) {
                        *(reinterpret_cast<__nv_bfloat16*>(output + ((tn + 1) * M + tm)) + (0)) = __float2bfloat16_rn(o01);
                    }
                }
                if (tn < N) {
                    if (tm + 8 < M) {
                        *(reinterpret_cast<__nv_bfloat16*>(output + (tn * M + tm + 8)) + (0)) = __float2bfloat16_rn(o10);
                    }
                }
                if (tn + 1 < N) {
                    if (tm + 8 < M) {
                        *(reinterpret_cast<__nv_bfloat16*>(output + ((tn + 1) * M + tm + 8)) + (0)) = __float2bfloat16_rn(o11);
                    }
                }
            }
        }
    // ---- Role: load_wt ----
    } else if (warp >= 4 && warp <= 7) {
        { // load_wt_main
            int k_loops = (K + 1024 - 1) / 1024;
            int mib = blockIdx.x * 16;
            unsigned int wslot = warp % 4;
            if (elect_sync()) {
                #pragma unroll 1
                for (unsigned int ki_1 = 0; ki_1 < k_loops; ki_1++) {
                    unsigned int stage = wslot + 4 * (ki_1 % 2);
                    unsigned int phase = ki_1 / 2 & 1;
                    int k_base = (ki_1 * 4 + wslot) * 256;
                    mbarrier_wait(data_consumed_addr + (stage) * 8, phase ^ 1);
                    mbarrier_arrive_expect_tx(wt_ready_addr + (stage) * 8, 8192);
                    #pragma unroll
                    for (int i = 0; i < 4; i++) {
                        int dst_wt = smem_wt_addr + (stage * 4 + (unsigned int)i) * 2048;
                        tma_2d_gmem2smem(dst_wt, ((const void*)(_loom_tma_param_base + 0)), k_base + i * 64, mib, wt_ready_addr + (stage) * 8);
                    }
                }
                #pragma unroll
                for (int di = 0; di < 2; di++) {
                    if (di + 1 < 2) {
                        unsigned int dki = k_loops + di;
                        unsigned int dstage = wslot + 4 * (dki % 2);
                        unsigned int dphase = dki / 2 & 1;
                        mbarrier_wait(data_consumed_addr + (dstage) * 8, dphase ^ 1);
                    }
                }
            }
            asm volatile("barrier.sync 2, 384;" ::: "memory");
        }
    // ---- Role: load_act ----
    } else if (warp >= 8 && warp <= 11) {
        { // load_act_main
            int k_loops_a = (K + 1024 - 1) / 1024;
            int ni = blockIdx.y * 8;
            unsigned int aslot = warp % 4;
            if (elect_sync()) {
                {
                    asm volatile("griddepcontrol.wait;" ::: "memory");
                    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
                }
                #pragma unroll 1
                for (unsigned int ki_2 = 0; ki_2 < k_loops_a; ki_2++) {
                    unsigned int stage_a = aslot + 4 * (ki_2 % 2);
                    unsigned int phase_a = ki_2 / 2 & 1;
                    int k_base_a = (ki_2 * 4 + aslot) * 256;
                    mbarrier_wait(data_consumed_addr + (stage_a) * 8, phase_a ^ 1);
                    mbarrier_arrive_expect_tx(act_ready_addr + (stage_a) * 8, 4096);
                    #pragma unroll
                    for (int i_1 = 0; i_1 < 4; i_1++) {
                        int dst_act = smem_act_addr + (stage_a * 4 + (unsigned int)i_1) * 1024;
                        tma_2d_gmem2smem(dst_act, ((const void*)(_loom_tma_param_base + 128)), k_base_a + i_1 * 64, ni, act_ready_addr + (stage_a) * 8);
                    }
                }
                #pragma unroll
                for (int di_1 = 0; di_1 < 2; di_1++) {
                    if (di_1 + 1 < 2) {
                        unsigned int dki_a = k_loops_a + di_1;
                        unsigned int dstage_a = aslot + 4 * (dki_a % 2);
                        unsigned int dphase_a = dki_a / 2 & 1;
                        mbarrier_wait(data_consumed_addr + (dstage_a) * 8, dphase_a ^ 1);
                    }
                }
            }
            asm volatile("barrier.sync 2, 384;" ::: "memory");
        }
    }

    // Cleanup
}

} // extern "C"

constexpr int kSmemBytesStage8Pdl = SMEM_TOTAL;
static_assert(kSmemBytesStage8Pdl == 101504,
              "generated SMEM footprint for stage8_pdl changed; update the launcher expectations");

#undef LOOM_INF
#undef NUM_MAIN_STAGES
#undef SMEM_SMEM_WT_OFF
#undef SMEM_SMEM_WT_STAGE_BYTES
#undef SMEM_SMEM_WT_STRIDE
#undef SMEM_SMEM_ACT_OFF
#undef SMEM_SMEM_ACT_STAGE_BYTES
#undef SMEM_SMEM_ACT_STRIDE
#undef SMEM_SMEM_RED_OFF
#undef SMEM_SMEM_RED_STAGE_BYTES
#undef SMEM_SMEM_RED_STRIDE
#undef SMEM_SMEM_BIAS_OFF
#undef SMEM_SMEM_BIAS_STAGE_BYTES
#undef SMEM_SMEM_BIAS_STRIDE
#undef SMEM_TOTAL
#undef THREADS
#undef USE_PDL

// ============================================================================
// Section stage16 — generated Loom schedule 'flashinfer_tinygemm2' (STAGES=16, USE_PDL=0).
// ============================================================================

#define LOOM_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define SMEM_SMEM_WT_OFF 1024
#define SMEM_SMEM_WT_STAGE_BYTES 2048
#define SMEM_SMEM_WT_STRIDE 2048
#define SMEM_SMEM_ACT_OFF 132096
#define SMEM_SMEM_ACT_STAGE_BYTES 1024
#define SMEM_SMEM_ACT_STRIDE 1024
#define SMEM_SMEM_RED_OFF 197632
#define SMEM_SMEM_RED_STAGE_BYTES 2048
#define SMEM_SMEM_RED_STRIDE 2048
#define SMEM_SMEM_BIAS_OFF 199680
#define SMEM_SMEM_BIAS_STAGE_BYTES 32
#define SMEM_SMEM_BIAS_STRIDE 32
#define SMEM_TOTAL 199808
#define THREADS 384
#define USE_PDL 0

extern "C" {

__global__ __launch_bounds__(384, 1) void
kernel_tinygemm2_sm100_stage16(__nv_bfloat16* __restrict__ output, __nv_bfloat16* __restrict__ bias, int M, int N, int K, __grid_constant__ LoomTensorMapPack<2> const _loom_tma_params)
{
    uint64_t _loom_tma_param_base;
    asm volatile("mov.b64 %0, %1;" : "=l"(_loom_tma_param_base) : "l"((uint64_t)(&_loom_tma_params)));

    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // Kernel setup ops
    __nv_bfloat16* smem_wt = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int smem_wt_addr = smem + 1024;
    __nv_bfloat16* smem_act = reinterpret_cast<__nv_bfloat16*>(smem_raw + 132096);
    const int smem_act_addr = smem + 132096;
    float* smem_red = reinterpret_cast<float*>(smem_raw + 197632);
    const int smem_red_addr = smem + 197632;
    __nv_bfloat16* smem_bias = reinterpret_cast<__nv_bfloat16*>(smem_raw + 199680);
    const int smem_bias_addr = smem + 199680;
    if (tid == 0) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)(((const void*)(_loom_tma_param_base + 0)))) : "memory"); }
    if (tid == 0) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)(((const void*)(_loom_tma_param_base + 128)))) : "memory"); }

    // Mbarrier init (3 groups, 48 barriers)
    // Mbarriers at smem_raw[0..384)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        // wt_ready: 16 barriers, init_count=1
        mbarrier_init_pred(smem + 0, 1, leader);
        mbarrier_init_pred(smem + 8, 1, leader);
        mbarrier_init_pred(smem + 16, 1, leader);
        mbarrier_init_pred(smem + 24, 1, leader);
        mbarrier_init_pred(smem + 32, 1, leader);
        mbarrier_init_pred(smem + 40, 1, leader);
        mbarrier_init_pred(smem + 48, 1, leader);
        mbarrier_init_pred(smem + 56, 1, leader);
        mbarrier_init_pred(smem + 64, 1, leader);
        mbarrier_init_pred(smem + 72, 1, leader);
        mbarrier_init_pred(smem + 80, 1, leader);
        mbarrier_init_pred(smem + 88, 1, leader);
        mbarrier_init_pred(smem + 96, 1, leader);
        mbarrier_init_pred(smem + 104, 1, leader);
        mbarrier_init_pred(smem + 112, 1, leader);
        mbarrier_init_pred(smem + 120, 1, leader);
        // act_ready: 16 barriers, init_count=1
        mbarrier_init_pred(smem + 128, 1, leader);
        mbarrier_init_pred(smem + 136, 1, leader);
        mbarrier_init_pred(smem + 144, 1, leader);
        mbarrier_init_pred(smem + 152, 1, leader);
        mbarrier_init_pred(smem + 160, 1, leader);
        mbarrier_init_pred(smem + 168, 1, leader);
        mbarrier_init_pred(smem + 176, 1, leader);
        mbarrier_init_pred(smem + 184, 1, leader);
        mbarrier_init_pred(smem + 192, 1, leader);
        mbarrier_init_pred(smem + 200, 1, leader);
        mbarrier_init_pred(smem + 208, 1, leader);
        mbarrier_init_pred(smem + 216, 1, leader);
        mbarrier_init_pred(smem + 224, 1, leader);
        mbarrier_init_pred(smem + 232, 1, leader);
        mbarrier_init_pred(smem + 240, 1, leader);
        mbarrier_init_pred(smem + 248, 1, leader);
        // data_consumed: 16 barriers, init_count=32
        mbarrier_init_pred(smem + 256, 32, leader);
        mbarrier_init_pred(smem + 264, 32, leader);
        mbarrier_init_pred(smem + 272, 32, leader);
        mbarrier_init_pred(smem + 280, 32, leader);
        mbarrier_init_pred(smem + 288, 32, leader);
        mbarrier_init_pred(smem + 296, 32, leader);
        mbarrier_init_pred(smem + 304, 32, leader);
        mbarrier_init_pred(smem + 312, 32, leader);
        mbarrier_init_pred(smem + 320, 32, leader);
        mbarrier_init_pred(smem + 328, 32, leader);
        mbarrier_init_pred(smem + 336, 32, leader);
        mbarrier_init_pred(smem + 344, 32, leader);
        mbarrier_init_pred(smem + 352, 32, leader);
        mbarrier_init_pred(smem + 360, 32, leader);
        mbarrier_init_pred(smem + 368, 32, leader);
        mbarrier_init_pred(smem + 376, 32, leader);
        asm volatile("fence.mbarrier_init.release.cluster;");
    }

    __syncthreads();

    __syncthreads();

    const int mbar_base = smem;
    #define wt_ready_addr (mbar_base + 0)
    #define act_ready_addr (mbar_base + 128)
    #define data_consumed_addr (mbar_base + 256)

    // ---- Role: compute ----
    if (warp <= 3) {
        { // compute_main
            int k_loops_c = (K + 1024 - 1) / 1024;
            int mib_c = blockIdx.x * 16;
            int ni_c = blockIdx.y * 8;
            if (tid < 16) {
                smem_bias[tid] = bias[mib_c + tid];
            }
            float accum[4];
            #pragma unroll
            for (int z = 0; z < 4; z++) {
                accum[z] = 0.0f;
            }
            unsigned int lane_div8 = lane / 8;
            unsigned int lane_mod8 = lane % 8;
            unsigned int row_wt = lane_mod8 + lane_div8 % 2 * 8;
            unsigned int col_off_wt = lane_div8 / 2;
            unsigned int row_act = lane_mod8;
            #pragma unroll 2
            for (unsigned int ki = 0; ki < k_loops_c; ki++) {
                unsigned int stage_c = (unsigned int)warp + 4 * (ki % 4);
                unsigned int phase_c = ki / 4 & 1;
                mbarrier_wait(wt_ready_addr + (stage_c) * 8, phase_c);
                mbarrier_wait(act_ready_addr + (stage_c) * 8, phase_c);
                #pragma unroll
                for (int su = 0; su < 4; su++) {
                    unsigned int base_wt = smem_wt_addr + (stage_c * 4 + (unsigned int)su) * 2048;
                    unsigned int base_act = smem_act_addr + (stage_c * 4 + (unsigned int)su) * 1024;
                    #pragma unroll
                    for (int kii = 0; kii < 4; kii++) {
                        unsigned int a_frag[4];
                        unsigned int b_frag[2];
                        unsigned int col_w = (unsigned int)(2 * kii) + col_off_wt;
                        unsigned int col_sw_w = row_wt % 8 ^ col_w;
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                            : "r"(base_wt + row_wt * 128 + col_sw_w * 16)
                            : "memory");
                        unsigned int col_a = (unsigned int)(2 * kii) + lane_div8;
                        unsigned int col_sw_a = row_act % 8 ^ col_a;
                        asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                            : "=r"(b_frag[0]), "=r"(b_frag[1])
                            : "r"(base_act + row_act * 128 + col_sw_a * 16)
                            : "memory");
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(accum[0]), "+f"(accum[1]), "+f"(accum[2]), "+f"(accum[3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
                    }
                }
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_arrive(data_consumed_addr + (stage_c) * 8);
            }
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_red_addr + (unsigned int)(tid * 16)), "r"(__as_u32(accum[0])), "r"(__as_u32(accum[1])), "r"(__as_u32(accum[2])), "r"(__as_u32(accum[3])) : "memory");
            asm volatile("barrier.sync 2, 384;" ::: "memory");
            if (warp == 0) {
                float part[12];
                #pragma unroll
                for (int w = 0; w < 3; w++) {
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&part[w * 4])), "=r"(*reinterpret_cast<uint32_t*>(&part[(w * 4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&part[(w * 4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&part[(w * 4) + 3]))
                        : "r"(smem_red_addr + (unsigned int)((32 + w * 32 + tid) * 16)));
                }
                #pragma unroll
                for (int z_1 = 0; z_1 < 4; z_1++) {
                    accum[z_1] = accum[z_1] + part[z_1] + part[4 + z_1] + part[8 + z_1];
                }
                int tm = mib_c + lane / 4;
                int tn = ni_c + 2 * (lane % 4);
                float bias_lo = smem_bias[lane / 4];
                float bias_hi = smem_bias[lane / 4 + 8];
                float o00 = accum[0] + bias_lo;
                float o01 = accum[1] + bias_lo;
                float o10 = accum[2] + bias_hi;
                float o11 = accum[3] + bias_hi;
                if (tn < N) {
                    if (tm < M) {
                        *(reinterpret_cast<__nv_bfloat16*>(output + (tn * M + tm)) + (0)) = __float2bfloat16_rn(o00);
                    }
                }
                if (tn + 1 < N) {
                    if (tm < M) {
                        *(reinterpret_cast<__nv_bfloat16*>(output + ((tn + 1) * M + tm)) + (0)) = __float2bfloat16_rn(o01);
                    }
                }
                if (tn < N) {
                    if (tm + 8 < M) {
                        *(reinterpret_cast<__nv_bfloat16*>(output + (tn * M + tm + 8)) + (0)) = __float2bfloat16_rn(o10);
                    }
                }
                if (tn + 1 < N) {
                    if (tm + 8 < M) {
                        *(reinterpret_cast<__nv_bfloat16*>(output + ((tn + 1) * M + tm + 8)) + (0)) = __float2bfloat16_rn(o11);
                    }
                }
            }
        }
    // ---- Role: load_wt ----
    } else if (warp >= 4 && warp <= 7) {
        { // load_wt_main
            int k_loops = (K + 1024 - 1) / 1024;
            int mib = blockIdx.x * 16;
            unsigned int wslot = warp % 4;
            if (elect_sync()) {
                #pragma unroll 1
                for (unsigned int ki_1 = 0; ki_1 < k_loops; ki_1++) {
                    unsigned int stage = wslot + 4 * (ki_1 % 4);
                    unsigned int phase = ki_1 / 4 & 1;
                    int k_base = (ki_1 * 4 + wslot) * 256;
                    mbarrier_wait(data_consumed_addr + (stage) * 8, phase ^ 1);
                    mbarrier_arrive_expect_tx(wt_ready_addr + (stage) * 8, 8192);
                    #pragma unroll
                    for (int i = 0; i < 4; i++) {
                        int dst_wt = smem_wt_addr + (stage * 4 + (unsigned int)i) * 2048;
                        tma_2d_gmem2smem(dst_wt, ((const void*)(_loom_tma_param_base + 0)), k_base + i * 64, mib, wt_ready_addr + (stage) * 8);
                    }
                }
                #pragma unroll
                for (int di = 0; di < 4; di++) {
                    if (di + 1 < 4) {
                        unsigned int dki = k_loops + di;
                        unsigned int dstage = wslot + 4 * (dki % 4);
                        unsigned int dphase = dki / 4 & 1;
                        mbarrier_wait(data_consumed_addr + (dstage) * 8, dphase ^ 1);
                    }
                }
            }
            asm volatile("barrier.sync 2, 384;" ::: "memory");
        }
    // ---- Role: load_act ----
    } else if (warp >= 8 && warp <= 11) {
        { // load_act_main
            int k_loops_a = (K + 1024 - 1) / 1024;
            int ni = blockIdx.y * 8;
            unsigned int aslot = warp % 4;
            if (elect_sync()) {
                #pragma unroll 1
                for (unsigned int ki_2 = 0; ki_2 < k_loops_a; ki_2++) {
                    unsigned int stage_a = aslot + 4 * (ki_2 % 4);
                    unsigned int phase_a = ki_2 / 4 & 1;
                    int k_base_a = (ki_2 * 4 + aslot) * 256;
                    mbarrier_wait(data_consumed_addr + (stage_a) * 8, phase_a ^ 1);
                    mbarrier_arrive_expect_tx(act_ready_addr + (stage_a) * 8, 4096);
                    #pragma unroll
                    for (int i_1 = 0; i_1 < 4; i_1++) {
                        int dst_act = smem_act_addr + (stage_a * 4 + (unsigned int)i_1) * 1024;
                        tma_2d_gmem2smem(dst_act, ((const void*)(_loom_tma_param_base + 128)), k_base_a + i_1 * 64, ni, act_ready_addr + (stage_a) * 8);
                    }
                }
                #pragma unroll
                for (int di_1 = 0; di_1 < 4; di_1++) {
                    if (di_1 + 1 < 4) {
                        unsigned int dki_a = k_loops_a + di_1;
                        unsigned int dstage_a = aslot + 4 * (dki_a % 4);
                        unsigned int dphase_a = dki_a / 4 & 1;
                        mbarrier_wait(data_consumed_addr + (dstage_a) * 8, dphase_a ^ 1);
                    }
                }
            }
            asm volatile("barrier.sync 2, 384;" ::: "memory");
        }
    }

    // Cleanup
}

} // extern "C"

constexpr int kSmemBytesStage16 = SMEM_TOTAL;
static_assert(kSmemBytesStage16 == 199808,
              "generated SMEM footprint for stage16 changed; update the launcher expectations");

#undef LOOM_INF
#undef NUM_MAIN_STAGES
#undef SMEM_SMEM_WT_OFF
#undef SMEM_SMEM_WT_STAGE_BYTES
#undef SMEM_SMEM_WT_STRIDE
#undef SMEM_SMEM_ACT_OFF
#undef SMEM_SMEM_ACT_STAGE_BYTES
#undef SMEM_SMEM_ACT_STRIDE
#undef SMEM_SMEM_RED_OFF
#undef SMEM_SMEM_RED_STAGE_BYTES
#undef SMEM_SMEM_RED_STRIDE
#undef SMEM_SMEM_BIAS_OFF
#undef SMEM_SMEM_BIAS_STAGE_BYTES
#undef SMEM_SMEM_BIAS_STRIDE
#undef SMEM_TOTAL
#undef THREADS
#undef USE_PDL

// ============================================================================
// Section stage16_pdl — generated Loom schedule 'flashinfer_tinygemm2' (STAGES=16, USE_PDL=1).
// ============================================================================

#define LOOM_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define SMEM_SMEM_WT_OFF 1024
#define SMEM_SMEM_WT_STAGE_BYTES 2048
#define SMEM_SMEM_WT_STRIDE 2048
#define SMEM_SMEM_ACT_OFF 132096
#define SMEM_SMEM_ACT_STAGE_BYTES 1024
#define SMEM_SMEM_ACT_STRIDE 1024
#define SMEM_SMEM_RED_OFF 197632
#define SMEM_SMEM_RED_STAGE_BYTES 2048
#define SMEM_SMEM_RED_STRIDE 2048
#define SMEM_SMEM_BIAS_OFF 199680
#define SMEM_SMEM_BIAS_STAGE_BYTES 32
#define SMEM_SMEM_BIAS_STRIDE 32
#define SMEM_TOTAL 199808
#define THREADS 384
#define USE_PDL 1

extern "C" {

__global__ __launch_bounds__(384, 1) void
kernel_tinygemm2_sm100_stage16_pdl(__nv_bfloat16* __restrict__ output, __nv_bfloat16* __restrict__ bias, int M, int N, int K, __grid_constant__ LoomTensorMapPack<2> const _loom_tma_params)
{
    uint64_t _loom_tma_param_base;
    asm volatile("mov.b64 %0, %1;" : "=l"(_loom_tma_param_base) : "l"((uint64_t)(&_loom_tma_params)));

    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // Kernel setup ops
    __nv_bfloat16* smem_wt = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int smem_wt_addr = smem + 1024;
    __nv_bfloat16* smem_act = reinterpret_cast<__nv_bfloat16*>(smem_raw + 132096);
    const int smem_act_addr = smem + 132096;
    float* smem_red = reinterpret_cast<float*>(smem_raw + 197632);
    const int smem_red_addr = smem + 197632;
    __nv_bfloat16* smem_bias = reinterpret_cast<__nv_bfloat16*>(smem_raw + 199680);
    const int smem_bias_addr = smem + 199680;
    if (tid == 0) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)(((const void*)(_loom_tma_param_base + 0)))) : "memory"); }
    if (tid == 0) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)(((const void*)(_loom_tma_param_base + 128)))) : "memory"); }

    // Mbarrier init (3 groups, 48 barriers)
    // Mbarriers at smem_raw[0..384)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        // wt_ready: 16 barriers, init_count=1
        mbarrier_init_pred(smem + 0, 1, leader);
        mbarrier_init_pred(smem + 8, 1, leader);
        mbarrier_init_pred(smem + 16, 1, leader);
        mbarrier_init_pred(smem + 24, 1, leader);
        mbarrier_init_pred(smem + 32, 1, leader);
        mbarrier_init_pred(smem + 40, 1, leader);
        mbarrier_init_pred(smem + 48, 1, leader);
        mbarrier_init_pred(smem + 56, 1, leader);
        mbarrier_init_pred(smem + 64, 1, leader);
        mbarrier_init_pred(smem + 72, 1, leader);
        mbarrier_init_pred(smem + 80, 1, leader);
        mbarrier_init_pred(smem + 88, 1, leader);
        mbarrier_init_pred(smem + 96, 1, leader);
        mbarrier_init_pred(smem + 104, 1, leader);
        mbarrier_init_pred(smem + 112, 1, leader);
        mbarrier_init_pred(smem + 120, 1, leader);
        // act_ready: 16 barriers, init_count=1
        mbarrier_init_pred(smem + 128, 1, leader);
        mbarrier_init_pred(smem + 136, 1, leader);
        mbarrier_init_pred(smem + 144, 1, leader);
        mbarrier_init_pred(smem + 152, 1, leader);
        mbarrier_init_pred(smem + 160, 1, leader);
        mbarrier_init_pred(smem + 168, 1, leader);
        mbarrier_init_pred(smem + 176, 1, leader);
        mbarrier_init_pred(smem + 184, 1, leader);
        mbarrier_init_pred(smem + 192, 1, leader);
        mbarrier_init_pred(smem + 200, 1, leader);
        mbarrier_init_pred(smem + 208, 1, leader);
        mbarrier_init_pred(smem + 216, 1, leader);
        mbarrier_init_pred(smem + 224, 1, leader);
        mbarrier_init_pred(smem + 232, 1, leader);
        mbarrier_init_pred(smem + 240, 1, leader);
        mbarrier_init_pred(smem + 248, 1, leader);
        // data_consumed: 16 barriers, init_count=32
        mbarrier_init_pred(smem + 256, 32, leader);
        mbarrier_init_pred(smem + 264, 32, leader);
        mbarrier_init_pred(smem + 272, 32, leader);
        mbarrier_init_pred(smem + 280, 32, leader);
        mbarrier_init_pred(smem + 288, 32, leader);
        mbarrier_init_pred(smem + 296, 32, leader);
        mbarrier_init_pred(smem + 304, 32, leader);
        mbarrier_init_pred(smem + 312, 32, leader);
        mbarrier_init_pred(smem + 320, 32, leader);
        mbarrier_init_pred(smem + 328, 32, leader);
        mbarrier_init_pred(smem + 336, 32, leader);
        mbarrier_init_pred(smem + 344, 32, leader);
        mbarrier_init_pred(smem + 352, 32, leader);
        mbarrier_init_pred(smem + 360, 32, leader);
        mbarrier_init_pred(smem + 368, 32, leader);
        mbarrier_init_pred(smem + 376, 32, leader);
        asm volatile("fence.mbarrier_init.release.cluster;");
    }

    __syncthreads();

    __syncthreads();

    const int mbar_base = smem;
    #define wt_ready_addr (mbar_base + 0)
    #define act_ready_addr (mbar_base + 128)
    #define data_consumed_addr (mbar_base + 256)

    // ---- Role: compute ----
    if (warp <= 3) {
        { // compute_main
            int k_loops_c = (K + 1024 - 1) / 1024;
            int mib_c = blockIdx.x * 16;
            int ni_c = blockIdx.y * 8;
            if (tid < 16) {
                smem_bias[tid] = bias[mib_c + tid];
            }
            float accum[4];
            #pragma unroll
            for (int z = 0; z < 4; z++) {
                accum[z] = 0.0f;
            }
            unsigned int lane_div8 = lane / 8;
            unsigned int lane_mod8 = lane % 8;
            unsigned int row_wt = lane_mod8 + lane_div8 % 2 * 8;
            unsigned int col_off_wt = lane_div8 / 2;
            unsigned int row_act = lane_mod8;
            #pragma unroll 2
            for (unsigned int ki = 0; ki < k_loops_c; ki++) {
                unsigned int stage_c = (unsigned int)warp + 4 * (ki % 4);
                unsigned int phase_c = ki / 4 & 1;
                mbarrier_wait(wt_ready_addr + (stage_c) * 8, phase_c);
                mbarrier_wait(act_ready_addr + (stage_c) * 8, phase_c);
                #pragma unroll
                for (int su = 0; su < 4; su++) {
                    unsigned int base_wt = smem_wt_addr + (stage_c * 4 + (unsigned int)su) * 2048;
                    unsigned int base_act = smem_act_addr + (stage_c * 4 + (unsigned int)su) * 1024;
                    #pragma unroll
                    for (int kii = 0; kii < 4; kii++) {
                        unsigned int a_frag[4];
                        unsigned int b_frag[2];
                        unsigned int col_w = (unsigned int)(2 * kii) + col_off_wt;
                        unsigned int col_sw_w = row_wt % 8 ^ col_w;
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                            : "r"(base_wt + row_wt * 128 + col_sw_w * 16)
                            : "memory");
                        unsigned int col_a = (unsigned int)(2 * kii) + lane_div8;
                        unsigned int col_sw_a = row_act % 8 ^ col_a;
                        asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                            : "=r"(b_frag[0]), "=r"(b_frag[1])
                            : "r"(base_act + row_act * 128 + col_sw_a * 16)
                            : "memory");
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(accum[0]), "+f"(accum[1]), "+f"(accum[2]), "+f"(accum[3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
                    }
                }
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_arrive(data_consumed_addr + (stage_c) * 8);
            }
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_red_addr + (unsigned int)(tid * 16)), "r"(__as_u32(accum[0])), "r"(__as_u32(accum[1])), "r"(__as_u32(accum[2])), "r"(__as_u32(accum[3])) : "memory");
            asm volatile("barrier.sync 2, 384;" ::: "memory");
            if (warp == 0) {
                float part[12];
                #pragma unroll
                for (int w = 0; w < 3; w++) {
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&part[w * 4])), "=r"(*reinterpret_cast<uint32_t*>(&part[(w * 4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&part[(w * 4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&part[(w * 4) + 3]))
                        : "r"(smem_red_addr + (unsigned int)((32 + w * 32 + tid) * 16)));
                }
                #pragma unroll
                for (int z_1 = 0; z_1 < 4; z_1++) {
                    accum[z_1] = accum[z_1] + part[z_1] + part[4 + z_1] + part[8 + z_1];
                }
                int tm = mib_c + lane / 4;
                int tn = ni_c + 2 * (lane % 4);
                float bias_lo = smem_bias[lane / 4];
                float bias_hi = smem_bias[lane / 4 + 8];
                float o00 = accum[0] + bias_lo;
                float o01 = accum[1] + bias_lo;
                float o10 = accum[2] + bias_hi;
                float o11 = accum[3] + bias_hi;
                if (tn < N) {
                    if (tm < M) {
                        *(reinterpret_cast<__nv_bfloat16*>(output + (tn * M + tm)) + (0)) = __float2bfloat16_rn(o00);
                    }
                }
                if (tn + 1 < N) {
                    if (tm < M) {
                        *(reinterpret_cast<__nv_bfloat16*>(output + ((tn + 1) * M + tm)) + (0)) = __float2bfloat16_rn(o01);
                    }
                }
                if (tn < N) {
                    if (tm + 8 < M) {
                        *(reinterpret_cast<__nv_bfloat16*>(output + (tn * M + tm + 8)) + (0)) = __float2bfloat16_rn(o10);
                    }
                }
                if (tn + 1 < N) {
                    if (tm + 8 < M) {
                        *(reinterpret_cast<__nv_bfloat16*>(output + ((tn + 1) * M + tm + 8)) + (0)) = __float2bfloat16_rn(o11);
                    }
                }
            }
        }
    // ---- Role: load_wt ----
    } else if (warp >= 4 && warp <= 7) {
        { // load_wt_main
            int k_loops = (K + 1024 - 1) / 1024;
            int mib = blockIdx.x * 16;
            unsigned int wslot = warp % 4;
            if (elect_sync()) {
                #pragma unroll 1
                for (unsigned int ki_1 = 0; ki_1 < k_loops; ki_1++) {
                    unsigned int stage = wslot + 4 * (ki_1 % 4);
                    unsigned int phase = ki_1 / 4 & 1;
                    int k_base = (ki_1 * 4 + wslot) * 256;
                    mbarrier_wait(data_consumed_addr + (stage) * 8, phase ^ 1);
                    mbarrier_arrive_expect_tx(wt_ready_addr + (stage) * 8, 8192);
                    #pragma unroll
                    for (int i = 0; i < 4; i++) {
                        int dst_wt = smem_wt_addr + (stage * 4 + (unsigned int)i) * 2048;
                        tma_2d_gmem2smem(dst_wt, ((const void*)(_loom_tma_param_base + 0)), k_base + i * 64, mib, wt_ready_addr + (stage) * 8);
                    }
                }
                #pragma unroll
                for (int di = 0; di < 4; di++) {
                    if (di + 1 < 4) {
                        unsigned int dki = k_loops + di;
                        unsigned int dstage = wslot + 4 * (dki % 4);
                        unsigned int dphase = dki / 4 & 1;
                        mbarrier_wait(data_consumed_addr + (dstage) * 8, dphase ^ 1);
                    }
                }
            }
            asm volatile("barrier.sync 2, 384;" ::: "memory");
        }
    // ---- Role: load_act ----
    } else if (warp >= 8 && warp <= 11) {
        { // load_act_main
            int k_loops_a = (K + 1024 - 1) / 1024;
            int ni = blockIdx.y * 8;
            unsigned int aslot = warp % 4;
            if (elect_sync()) {
                {
                    asm volatile("griddepcontrol.wait;" ::: "memory");
                    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
                }
                #pragma unroll 1
                for (unsigned int ki_2 = 0; ki_2 < k_loops_a; ki_2++) {
                    unsigned int stage_a = aslot + 4 * (ki_2 % 4);
                    unsigned int phase_a = ki_2 / 4 & 1;
                    int k_base_a = (ki_2 * 4 + aslot) * 256;
                    mbarrier_wait(data_consumed_addr + (stage_a) * 8, phase_a ^ 1);
                    mbarrier_arrive_expect_tx(act_ready_addr + (stage_a) * 8, 4096);
                    #pragma unroll
                    for (int i_1 = 0; i_1 < 4; i_1++) {
                        int dst_act = smem_act_addr + (stage_a * 4 + (unsigned int)i_1) * 1024;
                        tma_2d_gmem2smem(dst_act, ((const void*)(_loom_tma_param_base + 128)), k_base_a + i_1 * 64, ni, act_ready_addr + (stage_a) * 8);
                    }
                }
                #pragma unroll
                for (int di_1 = 0; di_1 < 4; di_1++) {
                    if (di_1 + 1 < 4) {
                        unsigned int dki_a = k_loops_a + di_1;
                        unsigned int dstage_a = aslot + 4 * (dki_a % 4);
                        unsigned int dphase_a = dki_a / 4 & 1;
                        mbarrier_wait(data_consumed_addr + (dstage_a) * 8, dphase_a ^ 1);
                    }
                }
            }
            asm volatile("barrier.sync 2, 384;" ::: "memory");
        }
    }

    // Cleanup
}

} // extern "C"

constexpr int kSmemBytesStage16Pdl = SMEM_TOTAL;
static_assert(kSmemBytesStage16Pdl == 199808,
              "generated SMEM footprint for stage16_pdl changed; update the launcher expectations");

#undef LOOM_INF
#undef NUM_MAIN_STAGES
#undef SMEM_SMEM_WT_OFF
#undef SMEM_SMEM_WT_STAGE_BYTES
#undef SMEM_SMEM_WT_STRIDE
#undef SMEM_SMEM_ACT_OFF
#undef SMEM_SMEM_ACT_STAGE_BYTES
#undef SMEM_SMEM_ACT_STRIDE
#undef SMEM_SMEM_RED_OFF
#undef SMEM_SMEM_RED_STAGE_BYTES
#undef SMEM_SMEM_RED_STRIDE
#undef SMEM_SMEM_BIAS_OFF
#undef SMEM_SMEM_BIAS_STAGE_BYTES
#undef SMEM_SMEM_BIAS_STRIDE
#undef SMEM_TOTAL
#undef THREADS
#undef USE_PDL

// clang-format on

namespace flashinfer {
namespace tinygemm2_sm100 {

using tvm::ffi::TensorView;

// Fixed tile geometry shared by every generated variant. These mirror the
// TensorRT-LLM tinygemm2 template constants (WARP_TILE_M=16, TILE_N=8,
// TILE_K=64) that the generated schedules were ported from.
constexpr int kTileM = 16;  // output-features tile
constexpr int kTileN = 8;   // batch tile
constexpr int kTileK = 64;  // reduction tile (one TMA box)
constexpr int kThreads = 384;
// One TMA K loop of the generated schedules covers 1024 reduction elements;
// mirrors the (former) Python-side selection constant.
constexpr int kKPerLoop = 1024;
// Measured stage8→stage16 crossover (GB300, CUPTI, flushed-L2 and warm-L2,
// grids of 8/64/128 CTAs): the deep ring's flushed-state gain exceeds its
// ~0.2us warm-state cost from K=4608 upward and grows with K; at K<=4096 the
// two effects cancel or favor stage8. Multi-wave grids stay on stage8 — the
// doubled SMEM footprint halves CTA residency and loses 2-6us there.
constexpr int kStage16MinK = 4608;

struct ProblemDims {
  int batch;
  int in_features;
  int out_features;
};

inline void CheckCuda(cudaError_t status, const char* operation) {
  TVM_FFI_ICHECK(status == cudaSuccess) << operation << " failed: " << cudaGetErrorString(status);
}

inline void CheckSm100Family(int device_id) {
  // The verdict is a device property; cache it so steady-state launches skip
  // the two cudaDeviceGetAttribute calls.
  static std::mutex fam_mu;
  static std::vector<int> fam_ok;
  {
    std::lock_guard<std::mutex> lock(fam_mu);
    for (int cached : fam_ok) {
      if (cached == device_id) return;
    }
  }
  int major = 0;
  int minor = 0;
  CheckCuda(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device_id),
            "cudaDeviceGetAttribute(compute capability major)");
  CheckCuda(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device_id),
            "cudaDeviceGetAttribute(compute capability minor)");
  TVM_FFI_ICHECK(major == 10 && (minor == 0 || minor == 3 || minor == 7))
      << "tinygemm2_sm100 requires an SM100/SM103/SM107 (B200/B300/Rubin class) device, got sm_"
      << major << minor;
  {
    std::lock_guard<std::mutex> lock(fam_mu);
    fam_ok.push_back(device_id);
  }
}

inline void CheckBf16(const TensorView& t, const char* name) {
  const DLDataType d = t.dtype();
  TVM_FFI_ICHECK(d.code == kDLBfloat && d.bits == 16 && d.lanes == 1)
      << name << " must be bfloat16, got (code=" << int(d.code) << ", bits=" << int(d.bits)
      << ", lanes=" << int(d.lanes) << ")";
}

inline void CheckCudaBf16Contiguous(const TensorView& t, int ndim, const char* name) {
  TVM_FFI_ICHECK(t.device().device_type == kDLCUDA) << name << " must be a CUDA tensor";
  TVM_FFI_ICHECK(t.ndim() == ndim) << name << " must be " << ndim << "D, got ndim=" << t.ndim();
  TVM_FFI_ICHECK(t.IsContiguous()) << name << " must be contiguous";
  CheckBf16(t, name);
}

// Validate the public `out = input @ weight.T + bias` contract plus the
// coverage guards of the generated kernels (mirroring the Loom host shim):
// in_features must fit one TMA box; out_features must be a positive multiple
// of the kTileM output tile. The batch axis has NO lower guard — the
// activation descriptor deliberately allows an out-of-bounds box on that axis
// and TMA zero-fills rows past the end, so batch 1..7 inputs are valid.
inline ProblemDims CheckInputs(const TensorView& input, const TensorView& weight,
                               const TensorView& bias, const TensorView& out) {
  CheckCudaBf16Contiguous(input, 2, "input");
  CheckCudaBf16Contiguous(weight, 2, "weight");
  CheckCudaBf16Contiguous(bias, 1, "bias");
  CheckCudaBf16Contiguous(out, 2, "out");
  const int device_id = input.device().device_id;
  TVM_FFI_ICHECK(weight.device().device_id == device_id && bias.device().device_id == device_id &&
                 out.device().device_id == device_id)
      << "input/weight/bias/out must live on the same CUDA device";
  CheckSm100Family(device_id);

  const int64_t batch = input.size(0);
  const int64_t in_features = input.size(1);
  const int64_t out_features = weight.size(0);
  TVM_FFI_ICHECK(weight.size(1) == in_features)
      << "weight.shape[1] (" << weight.size(1) << ") must equal input.shape[1] (" << in_features
      << ")";
  TVM_FFI_ICHECK(bias.size(0) == out_features)
      << "bias.shape[0] (" << bias.size(0) << ") must equal weight.shape[0] (" << out_features
      << ")";
  TVM_FFI_ICHECK(out.size(0) == batch && out.size(1) == out_features)
      << "out must have shape (" << batch << ", " << out_features << "), got (" << out.size(0)
      << ", " << out.size(1) << ")";

  TVM_FFI_ICHECK(batch > 0) << "batch must be positive, got " << batch;
  TVM_FFI_ICHECK(in_features >= kTileK)
      << "in_features (" << in_features << ") must be at least " << kTileK << " (one TMA box)";
  TVM_FFI_ICHECK(out_features >= kTileM && out_features % kTileM == 0)
      << "out_features (" << out_features << ") must be a positive multiple of " << kTileM;
  TVM_FFI_ICHECK(batch <= std::numeric_limits<int>::max() &&
                 in_features <= std::numeric_limits<int>::max() &&
                 out_features <= std::numeric_limits<int>::max())
      << "problem dimensions exceed the kernel's i32 scalar range";

  return ProblemDims{static_cast<int>(batch), static_cast<int>(in_features),
                     static_cast<int>(out_features)};
}

// 2D TMA descriptor for the weight matrix — field-for-field the descriptor
// the Loom host shim encodes for 'tmap_wt': box (kTileK, kTileM), 128B
// swizzle, no L2 promotion, no OOB fill. Both boxed axes stay in bounds
// (CheckInputs guarantees in_features >= kTileK and out_features >= kTileM).
inline CUtensorMap EncodeWeightTma(const TensorView& weight) {
  const uint64_t global_dim[2] = {static_cast<uint64_t>(weight.size(1)),
                                  static_cast<uint64_t>(weight.size(0))};
  const uint64_t global_strides[1] = {static_cast<uint64_t>(weight.stride(0)) *
                                      sizeof(__nv_bfloat16)};
  const uint32_t box_dim[2] = {static_cast<uint32_t>(kTileK), static_cast<uint32_t>(kTileM)};
  const uint32_t elem_strides[2] = {1u, 1u};
  CUtensorMap tm;
  const CUresult r = cuTensorMapEncodeTiled(
      &tm, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, weight.data_ptr(), global_dim, global_strides,
      box_dim, elem_strides, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
      CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_ICHECK(r == CUDA_SUCCESS)
      << "cuTensorMapEncodeTiled failed for the weight descriptor: CUresult=" << int(r);
  return tm;
}

// 2D TMA descriptor for the activation matrix — the Loom host shim's
// 'tmap_act' descriptor. The batch axis opts into an out-of-bounds box
// (box kTileN may exceed batch for batch 1..7); TMA zero-fills those rows.
inline CUtensorMap EncodeActivationTma(const TensorView& input) {
  const uint64_t global_dim[2] = {static_cast<uint64_t>(input.size(1)),
                                  static_cast<uint64_t>(input.size(0))};
  const uint64_t global_strides[1] = {static_cast<uint64_t>(input.stride(0)) *
                                      sizeof(__nv_bfloat16)};
  const uint32_t box_dim[2] = {static_cast<uint32_t>(kTileK), static_cast<uint32_t>(kTileN)};
  const uint32_t elem_strides[2] = {1u, 1u};
  CUtensorMap tm;
  const CUresult r = cuTensorMapEncodeTiled(
      &tm, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, input.data_ptr(), global_dim, global_strides,
      box_dim, elem_strides, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
      CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_ICHECK(r == CUDA_SUCCESS)
      << "cuTensorMapEncodeTiled failed for the activation descriptor: CUresult=" << int(r);
  return tm;
}

// Generated kernel signature (pack ABI): ordinary arguments keep the authored
// order and both 128-byte tensor maps ride in one trailing by-value
// __grid_constant__ pack (maps[0] = weight, maps[1] = activation).
using GeneratedKernel = void (*)(__nv_bfloat16*, __nv_bfloat16*, int, int, int,
                                 LoomTensorMapPack<2>);

// Set the dynamic-SMEM opt-in once per (kernel, current device): the
// attribute is sticky, and setting it on every launch costs a host API call
// on the critical path of a ~5us kernel.
inline void EnsureKernelSmemAttr(GeneratedKernel kernel, int smem_bytes) {
  static std::mutex attr_mu;
  static std::vector<std::pair<const void*, int>> attr_done;
  int device_id = -1;
  CheckCuda(cudaGetDevice(&device_id), "cudaGetDevice(tinygemm2_sm100 smem attr)");
  {
    std::lock_guard<std::mutex> lock(attr_mu);
    for (const auto& entry : attr_done) {
      if (entry.first == reinterpret_cast<const void*>(kernel) && entry.second == device_id) {
        return;
      }
    }
  }
  CheckCuda(cudaFuncSetAttribute(reinterpret_cast<const void*>(kernel),
                                 cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes),
            "cudaFuncSetAttribute(tinygemm2_sm100 dynamic smem)");
  std::lock_guard<std::mutex> lock(attr_mu);
  attr_done.emplace_back(reinterpret_cast<const void*>(kernel), device_id);
}

// Launch one generated variant. LoomTensorMap is layout-compatible with
// CUtensorMap; both descriptors ride in the trailing by-value pack. PDL
// variants launch through cudaLaunchKernelEx with programmatic stream
// serialization, matching the in-kernel griddepcontrol pair compiled into
// those sections.
inline void LaunchVariant(GeneratedKernel kernel, int smem_bytes, bool pdl,
                          const CUtensorMap& weight_map, const CUtensorMap& activation_map,
                          __nv_bfloat16* out, __nv_bfloat16* bias, const ProblemDims& dims,
                          cudaStream_t stream) {
  static_assert(sizeof(LoomTensorMap) == sizeof(CUtensorMap),
                "generated tensor-map parameter must be layout-compatible with CUtensorMap");
  LoomTensorMapPack<2> pack;
  std::memcpy(&pack.maps[0], &weight_map, sizeof(LoomTensorMap));
  std::memcpy(&pack.maps[1], &activation_map, sizeof(LoomTensorMap));

  EnsureKernelSmemAttr(kernel, smem_bytes);

  const dim3 grid((dims.out_features + kTileM - 1) / kTileM, (dims.batch + kTileN - 1) / kTileN);
  const dim3 block(kThreads);

  if (pdl) {
    cudaLaunchConfig_t config;
    cudaLaunchAttribute attrs[1];
    config.gridDim = grid;
    config.blockDim = block;
    config.dynamicSmemBytes = smem_bytes;
    config.stream = stream;
    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[0].val.programmaticStreamSerializationAllowed = 1;
    config.attrs = attrs;
    config.numAttrs = 1;
    CheckCuda(cudaLaunchKernelEx(&config, kernel, out, bias, dims.out_features, dims.batch,
                                 dims.in_features, pack),
              "cudaLaunchKernelEx(tinygemm2_sm100)");
  } else {
    kernel<<<grid, block, smem_bytes, stream>>>(out, bias, dims.out_features, dims.batch,
                                                dims.in_features, pack);
    CheckCuda(cudaGetLastError(), "tinygemm2_sm100 kernel launch");
  }
}

inline int NumSms() {
  // The SM count is a device constant; cache it per device.
  static std::mutex sm_mu;
  static std::vector<std::pair<int, int>> sm_counts;
  int device_id = -1;
  CheckCuda(cudaGetDevice(&device_id), "cudaGetDevice(tinygemm2_sm100 stage select)");
  {
    std::lock_guard<std::mutex> lock(sm_mu);
    for (const auto& entry : sm_counts) {
      if (entry.first == device_id) return entry.second;
    }
  }
  int num_sms = -1;
  CheckCuda(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, device_id),
            "cudaDeviceGetAttribute(multiprocessor count)");
  std::lock_guard<std::mutex> lock(sm_mu);
  sm_counts.emplace_back(device_id, num_sms);
  return num_sms;
}

// Ring-depth selection, evaluated in the binding like the reference launcher
// selects STAGES. Three tiers:
//   stage4  — K fits one loader iteration, or the grid runs multiple waves
//             (the halved SMEM footprint doubles CTA residency);
//   stage16 — single-wave long-K shapes, where the deep ring hides the
//             elevated cold-miss latency that the 8-deep ring exposes;
//   stage8  — everything between.
inline int SelectStages(const ProblemDims& dims) {
  const int num_sms = NumSms();
  const int tiles_m = (dims.out_features + kTileM - 1) / kTileM;
  const int tiles_n = (dims.batch + kTileN - 1) / kTileN;
  const int total_ctas = tiles_m * tiles_n;
  if (dims.in_features <= kKPerLoop || total_ctas > 2 * num_sms) return 4;
  if (dims.in_features >= kStage16MinK && total_ctas <= num_sms) return 16;
  return 8;
}

// out = input @ weight.T + bias (bf16, fp32 accumulation), column-major
// epilogue identical to csrc/tinygemm2.cu.
void Run(TensorView input, TensorView weight, TensorView bias, TensorView out, bool use_pdl) {
  const ProblemDims dims = CheckInputs(input, weight, bias, out);
  const CUtensorMap weight_map = EncodeWeightTma(weight);
  const CUtensorMap activation_map = EncodeActivationTma(input);
  const cudaStream_t stream = get_stream(input.device());
  GeneratedKernel kernel;
  int smem_bytes;
  switch (SelectStages(dims)) {
    case 4:
      kernel = use_pdl ? &kernel_tinygemm2_sm100_stage4_pdl : &kernel_tinygemm2_sm100_stage4;
      smem_bytes = use_pdl ? kSmemBytesStage4Pdl : kSmemBytesStage4;
      break;
    case 16:
      kernel = use_pdl ? &kernel_tinygemm2_sm100_stage16_pdl : &kernel_tinygemm2_sm100_stage16;
      smem_bytes = use_pdl ? kSmemBytesStage16Pdl : kSmemBytesStage16;
      break;
    default:
      kernel = use_pdl ? &kernel_tinygemm2_sm100_stage8_pdl : &kernel_tinygemm2_sm100_stage8;
      smem_bytes = use_pdl ? kSmemBytesStage8Pdl : kSmemBytesStage8;
      break;
  }
  LaunchVariant(kernel, smem_bytes, use_pdl, weight_map, activation_map,
                reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
                reinterpret_cast<__nv_bfloat16*>(bias.data_ptr()), dims, stream);
}

// Per-variant entries for the direct-launch tests.
#define TINYGEMM2_SM100_DEFINE_RUN(Name, Kernel, SmemBytes, Pdl)                    \
  void Name(TensorView input, TensorView weight, TensorView bias, TensorView out) { \
    const ProblemDims dims = CheckInputs(input, weight, bias, out);                 \
    const CUtensorMap weight_map = EncodeWeightTma(weight);                         \
    const CUtensorMap activation_map = EncodeActivationTma(input);                  \
    const cudaStream_t stream = get_stream(input.device());                         \
    LaunchVariant(&Kernel, SmemBytes, Pdl, weight_map, activation_map,              \
                  reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),                 \
                  reinterpret_cast<__nv_bfloat16*>(bias.data_ptr()), dims, stream); \
  }

TINYGEMM2_SM100_DEFINE_RUN(RunStage4, kernel_tinygemm2_sm100_stage4, kSmemBytesStage4, false)
TINYGEMM2_SM100_DEFINE_RUN(RunStage4Pdl, kernel_tinygemm2_sm100_stage4_pdl, kSmemBytesStage4Pdl,
                           true)
TINYGEMM2_SM100_DEFINE_RUN(RunStage8, kernel_tinygemm2_sm100_stage8, kSmemBytesStage8, false)
TINYGEMM2_SM100_DEFINE_RUN(RunStage8Pdl, kernel_tinygemm2_sm100_stage8_pdl, kSmemBytesStage8Pdl,
                           true)
TINYGEMM2_SM100_DEFINE_RUN(RunStage16, kernel_tinygemm2_sm100_stage16, kSmemBytesStage16, false)
TINYGEMM2_SM100_DEFINE_RUN(RunStage16Pdl, kernel_tinygemm2_sm100_stage16_pdl, kSmemBytesStage16Pdl,
                           true)

#undef TINYGEMM2_SM100_DEFINE_RUN

}  // namespace tinygemm2_sm100
}  // namespace flashinfer

TVM_FFI_DLL_EXPORT_TYPED_FUNC(tinygemm2_sm100_op, flashinfer::tinygemm2_sm100::Run);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(stage4_op, flashinfer::tinygemm2_sm100::RunStage4);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(stage4_pdl_op, flashinfer::tinygemm2_sm100::RunStage4Pdl);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(stage8_op, flashinfer::tinygemm2_sm100::RunStage8);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(stage8_pdl_op, flashinfer::tinygemm2_sm100::RunStage8Pdl);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(stage16_op, flashinfer::tinygemm2_sm100::RunStage16);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(stage16_pdl_op, flashinfer::tinygemm2_sm100::RunStage16Pdl);
