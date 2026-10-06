/*
 * Copyright 2026 Cursor Research
 * Copyright (c) 2026 by FlashInfer team.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Derived from the Apache-2.0 Mixture of Kittens BF16 training kernels.
 * Modified: generated CUDA implementation and standalone TVM-FFI bindings.
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
#define NUM_MAIN_STAGES 1
#define SMEM_VECTORS_OFF 1024
#define SMEM_VECTORS_STAGE_BYTES 12288
#define SMEM_VECTORS_STRIDE 12288
#define SMEM_WEIGHTS_OFF 13312
#define SMEM_WEIGHTS_STAGE_BYTES 16
#define SMEM_WEIGHTS_STRIDE 16
#define SMEM_TOTAL 13440
#define THREADS 256

#include <math_constants.h>


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

// Source-faithful relaxed CTA wait used only by a typed protocol that does
// not attach the PTX acquire qualifier, such as FA4's interior P-ready edge.
// Exact source ports may request the PTX suspendTimeHint operand explicitly.
// The hint is expressed in nanoseconds and is kept separate from the canonical
// no-hint CTA helper so unrelated schedules retain their existing retry path.
// Exact unqualified CTA wait used by source schedules whose PTX intentionally
// omits the acquire qualifier while retaining a typed suspendTimeHint operand.


__device__ __forceinline__ void mbarrier_arrive_expect_tx(int mbar_addr, uint32_t bytes) {
    asm volatile(
        "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
        :: "r"(mbar_addr), "r"(bytes) : "memory");
}






__device__ __forceinline__ void tma_store_4d(
    const void *tmap, int x, int y, int z, int w, unsigned smem_addr) {
    asm volatile(
        "cp.async.bulk.tensor.4d.global.shared::cta.tile.bulk_group"
        " [%0, {%1, %2, %3, %4}], [%5];"
        :: "l"(tmap), "r"(x), "r"(y), "r"(z), "r"(w), "r"(smem_addr) : "memory");
}

extern "C" {

__global__ __launch_bounds__(256) void
kernel_cake_mok_epilogue_forward_2(const __grid_constant__ CUtensorMap shared, const __grid_constant__ CUtensorMap routed, float* __restrict__ scores, const __grid_constant__ CUtensorMap output, int hidden)
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
    #define inputs_addr (mbar_base + 0)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    __nv_bfloat16* vectors = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int vectors_addr = smem + 1024;
    float* weights = reinterpret_cast<float*>(smem_raw + 13312);
    const int weights_addr = smem + 13312;

    // Mbarrier init (1 pipeline groups, 0 ordered-sequence groups, 2 barriers)
    // Mbarriers at smem_raw[0..16)


    __syncthreads();

    // === Task calls (dependency order) ===
    int col_blocks = (hidden + 1023) / 1024;
    int col = bid % col_blocks * 1024;
    int first_token = bid / col_blocks * 2;
    if (tid == 0) {
        #pragma unroll
        for (int stage = 0; stage < 2; stage++) {
            mbarrier_init(inputs_addr + (stage) * 8, 1);
            mbarrier_arrive_expect_tx(inputs_addr + (stage) * 8, 6144);
        }
    }
    for (int idx = tid; idx < 4; idx += 256) {
        weights[idx] = scores[first_token * 2 + idx];
    }
    __syncthreads();
    #pragma unroll
    for (int stage_1 = 0; stage_1 < 2; stage_1++) {
        #pragma unroll
        for (int chunk = 0; chunk < 4; chunk++) {
            if (tid == 0) {
                asm volatile(
                    "cp.async.bulk.tensor.4d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
                    " [%0], [%1, {%2, %3, %4, %5}], [%6];"
                    :: "r"(vectors_addr + (unsigned int)(stage_1 * 3 * 2048) + (unsigned int)(chunk * 512)), "l"((&shared)), "r"(col + chunk * 256), "r"(first_token + stage_1), "r"(0), "r"(0), "r"(inputs_addr + (stage_1) * 8) : "memory");
            } else if (tid < 3) {
                asm volatile(
                    "cp.async.bulk.tensor.4d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
                    " [%0], [%1, {%2, %3, %4, %5}], [%6];"
                    :: "r"(vectors_addr + (unsigned int)((stage_1 * 3 + tid) * 2048) + (unsigned int)(chunk * 512)), "l"((&routed)), "r"(col + chunk * 256), "r"((first_token + stage_1) * 2 + tid - 1), "r"(0), "r"(0), "r"(inputs_addr + (stage_1) * 8) : "memory");
            }
        }
    }
    int lane_col = tid / 32 * 128 + tid % 32;
    #pragma unroll
    for (int stage_2 = 0; stage_2 < 2; stage_2++) {
        mbarrier_wait(inputs_addr + (stage_2) * 8, 0);
        float accumulator[4];
        #pragma unroll
        for (int elem = 0; elem < 4; elem++) {
            accumulator[elem] = (float)vectors[stage_2 * 3 * 1024 + lane_col + elem * 32];
        }
        #pragma unroll 1
        for (int k = 0; k < 2; k++) {
            float weight = weights[stage_2 * 2 + k];
            #pragma unroll
            for (int elem_1 = 0; elem_1 < 4; elem_1++) {
                float term = (float)vectors[(stage_2 * 3 + 1 + k) * 1024 + lane_col + elem_1 * 32];
                term = term * weight;
                accumulator[elem_1] = accumulator[elem_1] + term;
            }
        }
        #pragma unroll
        for (int elem_2 = 0; elem_2 < 4; elem_2++) {
            vectors[stage_2 * 3 * 1024 + lane_col + elem_2 * 32] = (__nv_bfloat16)accumulator[elem_2];
        }
        __syncthreads();
        if (tid == 0) {
            #pragma unroll
            for (int chunk_1 = 0; chunk_1 < 4; chunk_1++) {
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                tma_store_4d((&output), col + chunk_1 * 256, first_token + stage_2, 0, 0, vectors_addr + (unsigned int)(stage_2 * 3 * 2048) + (unsigned int)(chunk_1 * 512));
            }
            asm volatile("cp.async.bulk.commit_group;");
        }
    }

    // Cleanup
    __syncthreads();
}

} // extern "C"
