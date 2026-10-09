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
#define SMEM_VECTORS_STAGE_BYTES 36864
#define SMEM_VECTORS_STRIDE 36864
#define SMEM_WORDS_OFF 1024
#define SMEM_WORDS_STAGE_BYTES 36864
#define SMEM_WORDS_STRIDE 36864
#define SMEM_TOTAL 37888
#define THREADS 256

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



__device__ __forceinline__ void cp_async_bulk_gmem2smem(
    unsigned smem_addr, const void* gmem_ptr, unsigned bytes, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes"
        " [%0], [%1], %2, [%3];"
        :: "r"(smem_addr), "l"(gmem_ptr), "r"(bytes), "r"(mbar_addr)
        : "memory");
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

__device__ __forceinline__ __nv_bfloat162 __as_bf16x2(unsigned int v) {
    __nv_bfloat162_raw raw;
    raw.x = static_cast<unsigned short>(v);
    raw.y = static_cast<unsigned short>(v >> 16);
    return __nv_bfloat162(raw);
}

extern "C" {

__global__ __launch_bounds__(256) void
kernel_cake_mok_epilogue_forward_2(__nv_bfloat16* __restrict__ shared, __nv_bfloat16* __restrict__ routed, float* __restrict__ scores, unsigned int* __restrict__ output, int tokens, int hidden, int ctas)
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
    unsigned int* words = reinterpret_cast<unsigned int*>(smem_raw + 1024);
    const int words_addr = smem + 1024;

    // Mbarrier init (1 pipeline groups, 0 ordered-sequence groups, 6 barriers)
    // Mbarriers at smem_raw[0..48)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // inputs: 6 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            mbarrier_init(smem + 8, 1);
            mbarrier_init(smem + 16, 1);
            mbarrier_init(smem + 24, 1);
            mbarrier_init(smem + 32, 1);
            mbarrier_init(smem + 40, 1);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncthreads();

    // === Task calls (dependency order) ===
    int col_blocks = (hidden + 1024 - 1) / 1024;
    int units = tokens * col_blocks;
    int count = 0;
    if (units > bid) {
        count = (units - bid + ctas - 1) / ctas;
    }
    int _min_0 = ((count) < (6) ? (count) : (6));
    #pragma unroll 1
    for (int first = 0; first < _min_0; first++) {
        int stage = first % 6;
        int token = (bid + first * ctas) / col_blocks;
        int col = (bid + first * ctas) % col_blocks * 1024;
        int _min_1 = ((1024) < (hidden - col) ? (1024) : (hidden - col));
        int cols = _min_1;
        if (tid == 0) {
            mbarrier_arrive_expect_tx(inputs_addr + (stage) * 8, (unsigned int)(3 * cols * 2));
        }
        if (tid == 0) {
            cp_async_bulk_gmem2smem(vectors_addr + (unsigned int)(stage * 3 * 1024 * 2), reinterpret_cast<const void*>(reinterpret_cast<const uint8_t*>(shared) + ((unsigned long long)((unsigned long long)token * (unsigned long long)hidden + (unsigned long long)col) * (unsigned long long)2)), cols * 2, inputs_addr + (stage) * 8);
        } else if (tid < 3) {
            cp_async_bulk_gmem2smem(vectors_addr + (unsigned int)((stage * 3 + tid) * 1024 * 2), reinterpret_cast<const void*>(reinterpret_cast<const uint8_t*>(routed) + ((unsigned long long)(((unsigned long long)token * 2 + (unsigned long long)tid - 1) * (unsigned long long)hidden + (unsigned long long)col) * (unsigned long long)2)), cols * 2, inputs_addr + (stage) * 8);
        }
    }
    #pragma unroll 1
    for (int j = 0; j < count; j++) {
        int stage_1 = j % 6;
        int unit = bid + j * ctas;
        int token_1 = unit / col_blocks;
        int col_1 = unit % col_blocks * 1024;
        int _min_2 = ((1024) < (hidden - col_1) ? (1024) : (hidden - col_1));
        int cols_1 = _min_2;
        mbarrier_wait(inputs_addr + (stage_1) * 8, j / 6 & 1);
        if (cols_1 > tid * 4) {
            int base = stage_1 * 3 * 512 + tid * 2;
            float accumulator[4];
            #pragma unroll
            for (int pair = 0; pair < 2; pair++) {
                float2 _cvt_f32_0 = __bfloat1622float2(__as_bf16x2(words[base + pair]));
                accumulator[pair * 2] = _cvt_f32_0.x;
                accumulator[pair * 2 + 1] = _cvt_f32_0.y;
            }
            #pragma unroll 1
            for (int k = 0; k < 2; k++) {
                float weight = 1.0f;
                weight = scores[token_1 * 2 + k];
                #pragma unroll
                for (int pair_1 = 0; pair_1 < 2; pair_1++) {
                    float2 _cvt_f32_1 = __bfloat1622float2(__as_bf16x2(words[base + (1 + k) * 512 + pair_1]));
                    float term_x = _cvt_f32_1.x * weight;
                    float term_y = _cvt_f32_1.y * weight;
                    accumulator[pair_1 * 2] = accumulator[pair_1 * 2] + term_x;
                    accumulator[pair_1 * 2 + 1] = accumulator[pair_1 * 2 + 1] + term_y;
                }
            }
            long long out = ((long long)token_1 * (long long)hidden + (long long)col_1) / 2 + (long long)(tid * 2);
            #pragma unroll
            for (int pair_2 = 0; pair_2 < 2; pair_2++) {
                __nv_bfloat162 _bf16x2_0 = __float22bfloat162_rn(make_float2(accumulator[pair_2 * 2], accumulator[pair_2 * 2 + 1]));
                output[out + (long long)pair_2] = __as_u32(_bf16x2_0);
            }
        }
        __syncthreads();
        if (count > j + 6) {
            int stage_0 = (j + 6) % 6;
            int token_1_1 = (bid + (j + 6) * ctas) / col_blocks;
            int col_2 = (bid + (j + 6) * ctas) % col_blocks * 1024;
            int _min_3 = ((1024) < (hidden - col_2) ? (1024) : (hidden - col_2));
            int cols_3 = _min_3;
            if (tid == 0) {
                mbarrier_arrive_expect_tx(inputs_addr + (stage_0) * 8, (unsigned int)(3 * cols_3 * 2));
            }
            if (tid == 0) {
                cp_async_bulk_gmem2smem(vectors_addr + (unsigned int)(stage_0 * 3 * 1024 * 2), reinterpret_cast<const void*>(reinterpret_cast<const uint8_t*>(shared) + ((unsigned long long)((unsigned long long)token_1_1 * (unsigned long long)hidden + (unsigned long long)col_2) * (unsigned long long)2)), cols_3 * 2, inputs_addr + (stage_0) * 8);
            } else if (tid < 3) {
                cp_async_bulk_gmem2smem(vectors_addr + (unsigned int)((stage_0 * 3 + tid) * 1024 * 2), reinterpret_cast<const void*>(reinterpret_cast<const uint8_t*>(routed) + ((unsigned long long)(((unsigned long long)token_1_1 * 2 + (unsigned long long)tid - 1) * (unsigned long long)hidden + (unsigned long long)col_2) * (unsigned long long)2)), cols_3 * 2, inputs_addr + (stage_0) * 8);
            }
        }
    }

    // Cleanup
    __syncthreads();
}

} // extern "C"
