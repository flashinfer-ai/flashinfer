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
#define SMEM_X_TILE_OFF 1024
#define SMEM_X_TILE_STAGE_BYTES 32768
#define SMEM_X_TILE_STRIDE 32768
#define SMEM_X_HALVES_OFF 1024
#define SMEM_X_HALVES_STAGE_BYTES 32768
#define SMEM_X_HALVES_STRIDE 32768
#define SMEM_X_WORDS_OFF 1024
#define SMEM_X_WORDS_STAGE_BYTES 32768
#define SMEM_X_WORDS_STRIDE 32768
#define SMEM_T_WORDS_OFF 33792
#define SMEM_T_WORDS_STAGE_BYTES 16384
#define SMEM_T_WORDS_STRIDE 16384
#define SMEM_SC_WORDS_OFF 50176
#define SMEM_SC_WORDS_STAGE_BYTES 512
#define SMEM_SC_WORDS_STRIDE 512
#define SMEM_SC_T_WORDS_OFF 50688
#define SMEM_SC_T_WORDS_STAGE_BYTES 512
#define SMEM_SC_T_WORDS_STRIDE 512
#define SMEM_TOTAL 51200
#define THREADS 128

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


__device__ __forceinline__ float max_noftz(float a, float b) {
    float c;
    asm("max.f32 %0, %1, %2;" : "=f"(c) : "f"(a), "f"(b));
    return c;
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


__device__ __forceinline__ void tma_store_3d(
    const void *tmap, int x, int y, int z, unsigned smem_addr) {
    asm volatile(
        "cp.async.bulk.tensor.3d.global.shared::cta.tile.bulk_group"
        " [%0, {%1, %2, %3}], [%4];"
        :: "l"(tmap), "r"(x), "r"(y), "r"(z), "r"(smem_addr) : "memory");
}


__device__ __forceinline__ void tma_store_4d(
    const void *tmap, int x, int y, int z, int w, unsigned smem_addr) {
    asm volatile(
        "cp.async.bulk.tensor.4d.global.shared::cta.tile.bulk_group"
        " [%0, {%1, %2, %3, %4}], [%5];"
        :: "l"(tmap), "r"(x), "r"(y), "r"(z), "r"(w), "r"(smem_addr) : "memory");
}

extern "C" {

__global__ __launch_bounds__(128) void
kernel_cake_mok_quantize_transposed(const __grid_constant__ CUtensorMap x_bf16, const __grid_constant__ CUtensorMap x_fp8, const __grid_constant__ CUtensorMap x_fp8_t, const __grid_constant__ CUtensorMap x_sc, const __grid_constant__ CUtensorMap x_sc_t, int col_blocks, int row_blocks)
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
    __nv_bfloat16* x_tile = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int x_tile_addr = smem + 1024;
    uint16_t* x_halves = reinterpret_cast<uint16_t*>(smem_raw + 1024);
    const int x_halves_addr = smem + 1024;
    unsigned int* x_words = reinterpret_cast<unsigned int*>(smem_raw + 1024);
    const int x_words_addr = smem + 1024;
    unsigned int* t_words = reinterpret_cast<unsigned int*>(smem_raw + 33792);
    const int t_words_addr = smem + 33792;
    unsigned int* sc_words = reinterpret_cast<unsigned int*>(smem_raw + 50176);
    const int sc_words_addr = smem + 50176;
    unsigned int* sc_t_words = reinterpret_cast<unsigned int*>(smem_raw + 50688);
    const int sc_t_words_addr = smem + 50688;

    // Mbarrier init (1 pipeline groups, 0 ordered-sequence groups, 1 barriers)
    // Mbarriers at smem_raw[0..8)


    __syncthreads();

    // === Task calls (dependency order) ===
    int col = bid % col_blocks;
    int row = bid / col_blocks % row_blocks;
    int expert = bid / (col_blocks * row_blocks);
    if (tid == 0) {
        mbarrier_init(inputs_addr, 1);
        mbarrier_arrive_expect_tx(inputs_addr, 32768);
    }
    __syncthreads();
    if (tid == 0) {
        tma_4d_gmem2smem(x_tile_addr, (&x_bf16), 0, row * 128, col, expert, inputs_addr);
    }
    mbarrier_wait(inputs_addr, 0);
    float inv_e4m3_max = 0.002232142857f;
    float scale_floor = 1e-12f;
    int t_row = tid % 64 * 2 + tid / 64;
    int rotation = tid / 8;
    unsigned int t_scale_word = 0;
    #pragma unroll 1
    for (int j = 0; j < 4; j++) {
        int k_block = (j + rotation) % 4;
        unsigned int t_words_0[16];
        #pragma unroll
        for (int k = 0; k < 16; k++) {
            int src_row = k_block * 32 + (tid * 4 + k * 2) % 32;
            unsigned int lo = (unsigned int)x_halves[src_row * 128 + t_row];
            unsigned int hi = (unsigned int)x_halves[(src_row + 1) * 128 + t_row];
            uint32_t _prmt_b32_0;
            asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_b32_0) : "r"(lo), "r"(hi));
            t_words_0[k] = _prmt_b32_0;
        }
        unsigned int t_packed[8];
        uint32_t _bf16x2_abs_0;
        asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_0) : "r"(t_words_0[0]));
        unsigned int amax2 = _bf16x2_abs_0;
        #pragma unroll
        for (int k_1 = 1; k_1 < 16; k_1++) {
            uint32_t _bf16x2_abs_1;
            asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_1) : "r"(t_words_0[k_1]));
            uint32_t _bf16x2_max_0;
            asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_0) : "r"(amax2), "r"(_bf16x2_abs_1));
            amax2 = _bf16x2_max_0;
        }
        uint16_t _bf16_max_0;
        asm("max.bf16 %0, %1, %2;" : "=h"(_bf16_max_0) : "h"((uint16_t)(amax2 & 65535)), "h"((uint16_t)(amax2 >> 16)));
        float _cvt_f32_bf16_0;
        asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_0) : "h"((uint16_t)(_bf16_max_0)));
        float amax = _cvt_f32_bf16_0;
        float _max_0 = max_noftz(amax * inv_e4m3_max, scale_floor);
        float scale = _max_0;
        uint16_t _ue8m0x2_f32_0;
        asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_ue8m0x2_f32_0) : "f"(scale), "f"(scale));
        uint16_t codes = _ue8m0x2_f32_0;
        unsigned int scale_byte = (unsigned int)codes & 255;
        unsigned int inv_bits = 254 - scale_byte << 23;
        float inv = 0.0f;
        inv = __uint_as_float(inv_bits);
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            unsigned int w0 = t_words_0[2 * i];
            unsigned int w1 = t_words_0[2 * i + 1];
            float _cvt_f32_bf16_1;
            asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_1) : "h"((uint16_t)(w0 & 65535)));
            float v0 = _cvt_f32_bf16_1;
            float _cvt_f32_bf16_2;
            asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_2) : "h"((uint16_t)(w0 >> 16)));
            float v1 = _cvt_f32_bf16_2;
            float _cvt_f32_bf16_3;
            asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_3) : "h"((uint16_t)(w1 & 65535)));
            float v2 = _cvt_f32_bf16_3;
            float _cvt_f32_bf16_4;
            asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_4) : "h"((uint16_t)(w1 >> 16)));
            float v3 = _cvt_f32_bf16_4;
            uint16_t _e4m3x2_f32_0;
            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_0) : "f"(v1 * inv), "f"(v0 * inv));
            uint16_t lo_1 = _e4m3x2_f32_0;
            uint16_t _e4m3x2_f32_1;
            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_1) : "f"(v3 * inv), "f"(v2 * inv));
            uint16_t hi_1 = _e4m3x2_f32_1;
            t_packed[i] = (unsigned int)lo_1 | (unsigned int)hi_1 << 16;
        }
        unsigned int t_scale_byte = scale_byte;
        #pragma unroll
        for (int i_1 = 0; i_1 < 8; i_1++) {
            int t_col = k_block * 32 + (tid * 4 + i_1 * 4) % 32;
            t_words[t_row * 32 + t_col / 4] = t_packed[i_1];
        }
        t_scale_word = t_scale_word | t_scale_byte << (unsigned int)(k_block * 8);
    }
    sc_t_words[t_row % 32 * 4 + t_row / 32] = t_scale_word;
    __syncthreads();
    if (tid == 0) {
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        int t_tile = (expert * col_blocks + col) * row_blocks + row;
        tma_store_4d((&x_fp8_t), 0, col * 128, row, expert, t_words_addr);
        tma_store_3d((&x_sc_t), 0, t_tile * 32, 0, sc_t_words_addr);
        asm volatile("cp.async.bulk.commit_group;");
        asm volatile("cp.async.bulk.wait_group 0;");
    }

    // Cleanup
    __syncthreads();
}

} // extern "C"
