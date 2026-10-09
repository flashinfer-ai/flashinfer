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
#define TMEM_NCOLS 512
#define TMEM_TMEM_SFA_OFFSET 256
#define TMEM_TMEM_SFB_OFFSET 268
#define TMEM_ACCUMULATOR_OFFSET 0
#define NUM_SCHEDULE_PIPE_STAGES 1
#define NUM_DRAIN_PIPE_0_STAGES 1
#define NUM_DRAIN_PIPE_1_STAGES 1
#define NUM_DRAIN_PIPE_2_STAGES 1
#define NUM_DRAIN_PIPE_3_STAGES 1
#define NUM_DRAIN_PIPE_4_STAGES 1
#define NUM_DRAIN_PIPE_5_STAGES 1
#define NUM_DRAIN_PIPE_6_STAGES 1
#define NUM_DRAIN_PIPE_7_STAGES 1
#define SMEM_A_NT_OFF 1024
#define SMEM_A_NT_STAGE_BYTES 16384
#define SMEM_A_NT_STRIDE 16384
#define SMEM_A_ATB_OFF 1024
#define SMEM_A_ATB_STAGE_BYTES 16384
#define SMEM_A_ATB_STRIDE 16384
#define SMEM_B_NT_OFF 66560
#define SMEM_B_NT_STAGE_BYTES 16384
#define SMEM_B_NT_STRIDE 16384
#define SMEM_B_AB_OFF 66560
#define SMEM_B_AB_STAGE_BYTES 16384
#define SMEM_B_AB_STRIDE 16384
#define SMEM_B_NT_HI_OFF 132096
#define SMEM_B_NT_HI_STAGE_BYTES 16384
#define SMEM_B_NT_HI_STRIDE 16384
#define SMEM_B_AB_HI_OFF 132096
#define SMEM_B_AB_HI_STAGE_BYTES 16384
#define SMEM_B_AB_HI_STRIDE 16384
#define SMEM_D_SMEM_OFF 207872
#define SMEM_D_SMEM_STAGE_BYTES 8192
#define SMEM_D_SMEM_STRIDE 8192
#define SMEM_D_WORDS_OFF 207872
#define SMEM_D_WORDS_STAGE_BYTES 24576
#define SMEM_D_WORDS_STRIDE 24576
#define SMEM_SW_DH_OFF 1024
#define SMEM_SW_DH_STAGE_BYTES 32768
#define SMEM_SW_DH_STRIDE 32768
#define SMEM_SW_GATE_OFF 66560
#define SMEM_SW_GATE_STAGE_BYTES 32768
#define SMEM_SW_GATE_STRIDE 32768
#define SMEM_SW_UP_OFF 132096
#define SMEM_SW_UP_STAGE_BYTES 32768
#define SMEM_SW_UP_STRIDE 32768
#define SMEM_SW_ROUTER_OFF 197632
#define SMEM_SW_ROUTER_STAGE_BYTES 1024
#define SMEM_SW_ROUTER_STRIDE 1024
#define SMEM_SW_DH_WORDS_OFF 1024
#define SMEM_SW_DH_WORDS_STAGE_BYTES 65536
#define SMEM_SW_DH_WORDS_STRIDE 65536
#define SMEM_SW_GATE_WORDS_OFF 66560
#define SMEM_SW_GATE_WORDS_STAGE_BYTES 65536
#define SMEM_SW_GATE_WORDS_STRIDE 65536
#define SMEM_SW_UP_WORDS_OFF 132096
#define SMEM_SW_UP_WORDS_STAGE_BYTES 65536
#define SMEM_SW_UP_WORDS_STRIDE 65536
#define SMEM_REPLAY_GATE_OFF 1024
#define SMEM_REPLAY_GATE_STAGE_BYTES 32768
#define SMEM_REPLAY_GATE_STRIDE 32768
#define SMEM_REPLAY_UP_OFF 99328
#define SMEM_REPLAY_UP_STAGE_BYTES 32768
#define SMEM_REPLAY_UP_STRIDE 32768
#define SMEM_HIDDEN_STAGING_OFF 197632
#define SMEM_HIDDEN_STAGING_STAGE_BYTES 34816
#define SMEM_HIDDEN_STAGING_STRIDE 34816
#define SMEM_HIDDEN_WORDS_OFF 197632
#define SMEM_HIDDEN_WORDS_STAGE_BYTES 34816
#define SMEM_HIDDEN_WORDS_STRIDE 34816
#define SMEM_SMEM_V19_OFF 1024
#define SMEM_SMEM_V19_STAGE_BYTES 16384
#define SMEM_SMEM_V19_STRIDE 16384
#define SMEM_SMEM_V20_OFF 17408
#define SMEM_SMEM_V20_STAGE_BYTES 512
#define SMEM_SMEM_V20_STRIDE 512
#define SMEM_SMEM_V21_OFF 99328
#define SMEM_SMEM_V21_STAGE_BYTES 16384
#define SMEM_SMEM_V21_STRIDE 16384
#define SMEM_SMEM_V22_OFF 115712
#define SMEM_SMEM_V22_STAGE_BYTES 512
#define SMEM_SMEM_V22_STRIDE 512
#define SMEM_SMEM_V23_OFF 33792
#define SMEM_SMEM_V23_STAGE_BYTES 16384
#define SMEM_SMEM_V23_STRIDE 16384
#define SMEM_SMEM_V24_OFF 50176
#define SMEM_SMEM_V24_STAGE_BYTES 512
#define SMEM_SMEM_V24_STRIDE 512
#define SMEM_SMEM_V25_OFF 132096
#define SMEM_SMEM_V25_STAGE_BYTES 16384
#define SMEM_SMEM_V25_STRIDE 16384
#define SMEM_SMEM_V26_OFF 148480
#define SMEM_SMEM_V26_STAGE_BYTES 512
#define SMEM_SMEM_V26_STRIDE 512
#define SMEM_SMEM_V27_OFF 66560
#define SMEM_SMEM_V27_STAGE_BYTES 16384
#define SMEM_SMEM_V27_STRIDE 16384
#define SMEM_SMEM_V28_OFF 82944
#define SMEM_SMEM_V28_STAGE_BYTES 512
#define SMEM_SMEM_V28_STRIDE 512
#define SMEM_SMEM_V29_OFF 164864
#define SMEM_SMEM_V29_STAGE_BYTES 16384
#define SMEM_SMEM_V29_STRIDE 16384
#define SMEM_SMEM_V30_OFF 181248
#define SMEM_SMEM_V30_STAGE_BYTES 512
#define SMEM_SMEM_V30_STRIDE 512
#define SMEM_DISPATCH_WORDS_OFF 1024
#define SMEM_DISPATCH_WORDS_STAGE_BYTES 131072
#define SMEM_DISPATCH_WORDS_STRIDE 131072
#define SMEM_DISPATCH_WEIGHTS_OFF 199680
#define SMEM_DISPATCH_WEIGHTS_STAGE_BYTES 512
#define SMEM_DISPATCH_WEIGHTS_STRIDE 512
#define SMEM_SMEM_V33_OFF 1024
#define SMEM_SMEM_V33_STAGE_BYTES 131072
#define SMEM_SMEM_V33_STRIDE 131072
#define SMEM_SMEM_V34_OFF 1280
#define SMEM_SMEM_V34_STAGE_BYTES 130816
#define SMEM_SMEM_V34_STRIDE 130816
#define SMEM_SMEM_V35_OFF 1536
#define SMEM_SMEM_V35_STAGE_BYTES 130560
#define SMEM_SMEM_V35_STRIDE 130560
#define SMEM_SMEM_V36_OFF 1792
#define SMEM_SMEM_V36_STAGE_BYTES 130304
#define SMEM_SMEM_V36_STRIDE 130304
#define SMEM_SMEM_V37_OFF 132096
#define SMEM_SMEM_V37_STAGE_BYTES 16384
#define SMEM_SMEM_V37_STRIDE 16384
#define SMEM_SMEM_V38_OFF 148480
#define SMEM_SMEM_V38_STAGE_BYTES 16384
#define SMEM_SMEM_V38_STRIDE 16384
#define SMEM_SMEM_V39_OFF 164864
#define SMEM_SMEM_V39_STAGE_BYTES 512
#define SMEM_SMEM_V39_STRIDE 512
#define SMEM_SMEM_V40_OFF 165376
#define SMEM_SMEM_V40_STAGE_BYTES 512
#define SMEM_SMEM_V40_STRIDE 512
#define SMEM_SMEM_V41_OFF 165888
#define SMEM_SMEM_V41_STAGE_BYTES 16384
#define SMEM_SMEM_V41_STRIDE 16384
#define SMEM_SMEM_V42_OFF 182272
#define SMEM_SMEM_V42_STAGE_BYTES 16384
#define SMEM_SMEM_V42_STRIDE 16384
#define SMEM_SMEM_V43_OFF 198656
#define SMEM_SMEM_V43_STAGE_BYTES 512
#define SMEM_SMEM_V43_STRIDE 512
#define SMEM_SMEM_V44_OFF 199168
#define SMEM_SMEM_V44_STAGE_BYTES 512
#define SMEM_SMEM_V44_STRIDE 512
#define SMEM_COMBINE_SMEM_OFF 1024
#define SMEM_COMBINE_SMEM_STAGE_BYTES 229376
#define SMEM_COMBINE_SMEM_STRIDE 229376
#define SMEM_SMEM_V46_OFF 1024
#define SMEM_SMEM_V46_STAGE_BYTES 16384
#define SMEM_SMEM_V46_STRIDE 16384
#define SMEM_SMEM_V47_OFF 66560
#define SMEM_SMEM_V47_STAGE_BYTES 16384
#define SMEM_SMEM_V47_STRIDE 16384
#define SMEM_SMEM_V48_OFF 132096
#define SMEM_SMEM_V48_STAGE_BYTES 16384
#define SMEM_SMEM_V48_STRIDE 16384
#define SMEM_SMEM_V49_OFF 197632
#define SMEM_SMEM_V49_STAGE_BYTES 512
#define SMEM_SMEM_V49_STRIDE 512
#define SMEM_SMEM_V50_OFF 199680
#define SMEM_SMEM_V50_STAGE_BYTES 1024
#define SMEM_SMEM_V50_STRIDE 1024
#define SMEM_SMEM_V51_OFF 203776
#define SMEM_SMEM_V51_STAGE_BYTES 1024
#define SMEM_SMEM_V51_STRIDE 1024
#define SMEM_SMEM_V52_OFF 1024
#define SMEM_SMEM_V52_STAGE_BYTES 16384
#define SMEM_SMEM_V52_STRIDE 16384
#define SMEM_SMEM_V53_OFF 99328
#define SMEM_SMEM_V53_STAGE_BYTES 16384
#define SMEM_SMEM_V53_STRIDE 16384
#define SMEM_SMEM_V54_OFF 197632
#define SMEM_SMEM_V54_STAGE_BYTES 512
#define SMEM_SMEM_V54_STRIDE 512
#define SMEM_SMEM_V55_OFF 200704
#define SMEM_SMEM_V55_STAGE_BYTES 1024
#define SMEM_SMEM_V55_STRIDE 1024
#define SMEM_SMEM_V56_OFF 207872
#define SMEM_SMEM_V56_STAGE_BYTES 16384
#define SMEM_SMEM_V56_STRIDE 16384
#define SMEM_SMEM_V57_OFF 224256
#define SMEM_SMEM_V57_STAGE_BYTES 4096
#define SMEM_SMEM_V57_STRIDE 4096
#define SMEM_SMEM_V58_OFF 228352
#define SMEM_SMEM_V58_STAGE_BYTES 512
#define SMEM_SMEM_V58_STRIDE 512
#define SMEM_SMEM_V59_OFF 228864
#define SMEM_SMEM_V59_STAGE_BYTES 512
#define SMEM_SMEM_V59_STRIDE 512
#define SMEM_SMEM_V60_OFF 1024
#define SMEM_SMEM_V60_STAGE_BYTES 32768
#define SMEM_SMEM_V60_STRIDE 32768
#define SMEM_SMEM_V61_OFF 33792
#define SMEM_SMEM_V61_STAGE_BYTES 32768
#define SMEM_SMEM_V61_STRIDE 32768
#define SMEM_SMEM_V62_OFF 1024
#define SMEM_SMEM_V62_STAGE_BYTES 32768
#define SMEM_SMEM_V62_STRIDE 32768
#define SMEM_SMEM_V63_OFF 33792
#define SMEM_SMEM_V63_STAGE_BYTES 32768
#define SMEM_SMEM_V63_STRIDE 32768
#define SMEM_SMEM_V64_OFF 66560
#define SMEM_SMEM_V64_STAGE_BYTES 16384
#define SMEM_SMEM_V64_STRIDE 16384
#define SMEM_SMEM_V65_OFF 82944
#define SMEM_SMEM_V65_STAGE_BYTES 16384
#define SMEM_SMEM_V65_STRIDE 16384
#define SMEM_SMEM_V66_OFF 99328
#define SMEM_SMEM_V66_STAGE_BYTES 16384
#define SMEM_SMEM_V66_STRIDE 16384
#define SMEM_SMEM_V67_OFF 115712
#define SMEM_SMEM_V67_STAGE_BYTES 16384
#define SMEM_SMEM_V67_STRIDE 16384
#define SMEM_SMEM_V68_OFF 132096
#define SMEM_SMEM_V68_STAGE_BYTES 512
#define SMEM_SMEM_V68_STRIDE 512
#define SMEM_SMEM_V69_OFF 132608
#define SMEM_SMEM_V69_STAGE_BYTES 512
#define SMEM_SMEM_V69_STRIDE 512
#define SMEM_SMEM_V70_OFF 133120
#define SMEM_SMEM_V70_STAGE_BYTES 512
#define SMEM_SMEM_V70_STRIDE 512
#define SMEM_SMEM_V71_OFF 133632
#define SMEM_SMEM_V71_STAGE_BYTES 512
#define SMEM_SMEM_V71_STRIDE 512
#define SMEM_SMEM_V72_OFF 132096
#define SMEM_SMEM_V72_STAGE_BYTES 512
#define SMEM_SMEM_V72_STRIDE 512
#define SMEM_SMEM_V73_OFF 132608
#define SMEM_SMEM_V73_STAGE_BYTES 512
#define SMEM_SMEM_V73_STRIDE 512
#define SMEM_SMEM_V74_OFF 133120
#define SMEM_SMEM_V74_STAGE_BYTES 512
#define SMEM_SMEM_V74_STRIDE 512
#define SMEM_SMEM_V75_OFF 133632
#define SMEM_SMEM_V75_STAGE_BYTES 512
#define SMEM_SMEM_V75_STRIDE 512
#define SMEM_SMEM_V76_OFF 134144
#define SMEM_SMEM_V76_STAGE_BYTES 32768
#define SMEM_SMEM_V76_STRIDE 32768
#define SMEM_SMEM_V77_OFF 134144
#define SMEM_SMEM_V77_STAGE_BYTES 32768
#define SMEM_SMEM_V77_STRIDE 32768
#define SMEM_SMEM_V78_OFF 166912
#define SMEM_SMEM_V78_STAGE_BYTES 16384
#define SMEM_SMEM_V78_STRIDE 16384
#define SMEM_SMEM_V79_OFF 183296
#define SMEM_SMEM_V79_STAGE_BYTES 16384
#define SMEM_SMEM_V79_STRIDE 16384
#define SMEM_SMEM_V80_OFF 199680
#define SMEM_SMEM_V80_STAGE_BYTES 512
#define SMEM_SMEM_V80_STRIDE 512
#define SMEM_SMEM_V81_OFF 200192
#define SMEM_SMEM_V81_STAGE_BYTES 512
#define SMEM_SMEM_V81_STRIDE 512
#define SMEM_SMEM_V82_OFF 200704
#define SMEM_SMEM_V82_STAGE_BYTES 1024
#define SMEM_SMEM_V82_STRIDE 1024
#define SMEM_TOTAL 232448
#define THREADS 256
#define CAKE_TMEM_HOLD_OFFSET 456

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

__device__ __forceinline__ void tcgen05_mma_mxf8_bs_cta2(
    int taddr, uint64_t a_desc, uint64_t b_desc, uint32_t i_desc,
    int sfa_taddr, int sfb_taddr, int enable_input_d) {
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, %6, 0;\n\t"
        "tcgen05.mma.cta_group::2.kind::mxf8f6f4.block_scale"
        " [%0], %1, %2, %3, [%4], [%5], p;\n\t"
        "}\n"
        :: "r"(taddr), "l"(a_desc), "l"(b_desc),
           "r"(i_desc), "r"(sfa_taddr), "r"(sfb_taddr),
           "r"(enable_input_d));
}



__device__ __forceinline__ uint64_t desc_encode(uint64_t x) {
    return (x & 0x3FFFFULL) >> 4ULL;
}


union MmaSmemDesc {
    uint64_t u64;
    uint32_t u32[2];
};




__device__ __forceinline__ void mbarrier_arrive_expect_tx(int mbar_addr, uint32_t bytes) {
    asm volatile(
        "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
        :: "r"(mbar_addr), "r"(bytes) : "memory");
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



__device__ __forceinline__ uint64_t make_sf_cp_desc_sbo128(int addr) {
    const int SBO = 128;
    return desc_encode(addr)
         | (desc_encode(SBO) << 32ULL)
         | (1ULL << 46ULL);
}



__device__ __forceinline__ void tcgen05_cp_32x128b_warpx4_cta2(
    int taddr, uint64_t s_desc) {
    asm volatile(
        "tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;"
        :: "r"(taddr), "l"(s_desc));
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




__device__ __forceinline__ void tma_store_2d(
    const void *tmap, int x, int y, unsigned smem_addr) {
    asm volatile(
        "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group"
        " [%0, {%1, %2}], [%3];"
        :: "l"(tmap), "r"(x), "r"(y), "r"(smem_addr) : "memory");
}


__device__ __forceinline__ void tma_store_3d(
    const void *tmap, int x, int y, int z, unsigned smem_addr) {
    asm volatile(
        "cp.async.bulk.tensor.3d.global.shared::cta.tile.bulk_group"
        " [%0, {%1, %2, %3}], [%4];"
        :: "l"(tmap), "r"(x), "r"(y), "r"(z), "r"(smem_addr) : "memory");
}


__device__ __forceinline__ void tma_store_5d(
    const void *tmap, int x, int y, int z, int w, int v, unsigned smem_addr) {
    asm volatile(
        "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group"
        " [%0, {%1, %2, %3, %4, %5}], [%6];"
        :: "l"(tmap), "r"(x), "r"(y), "r"(z), "r"(w), "r"(v), "r"(smem_addr) : "memory");
}


__device__ __forceinline__ void cp_async_bulk_gmem2smem(
    unsigned smem_addr, const void* gmem_ptr, unsigned bytes, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes"
        " [%0], [%1], %2, [%3];"
        :: "r"(smem_addr), "l"(gmem_ptr), "r"(bytes), "r"(mbar_addr)
        : "memory");
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

__global__ __launch_bounds__(256, 1) __cluster_dims__(2,1,1) void
kernel_cake_mok_backward_mxfp8_fp32(const __grid_constant__ CUtensorMap dy_s, const __grid_constant__ CUtensorMap dg_s, const __grid_constant__ CUtensorMap du_s, const __grid_constant__ CUtensorMap dy_atb_s, const __grid_constant__ CUtensorMap dg_atb_s, const __grid_constant__ CUtensorMap du_atb_s, const __grid_constant__ CUtensorMap x_atb_s, const __grid_constant__ CUtensorMap h_atb_s, const __grid_constant__ CUtensorMap wg_s, const __grid_constant__ CUtensorMap wu_s, const __grid_constant__ CUtensorMap wd_s, const __grid_constant__ CUtensorMap dh_s, const __grid_constant__ CUtensorMap dh_r, const __grid_constant__ CUtensorMap dx_s, const __grid_constant__ CUtensorMap dx_r, const __grid_constant__ CUtensorMap gate_out_r, const __grid_constant__ CUtensorMap up_out_r, const __grid_constant__ CUtensorMap dwg_s, const __grid_constant__ CUtensorMap dwu_s, const __grid_constant__ CUtensorMap dwd_s, const __grid_constant__ CUtensorMap dwg_r, const __grid_constant__ CUtensorMap dwu_r, const __grid_constant__ CUtensorMap dwd_r, const __grid_constant__ CUtensorMap dh_sw_s, const __grid_constant__ CUtensorMap gate_sw_s, const __grid_constant__ CUtensorMap up_sw_s, const __grid_constant__ CUtensorMap dg_sw_s, const __grid_constant__ CUtensorMap du_sw_s, const __grid_constant__ CUtensorMap gate_sw_r, const __grid_constant__ CUtensorMap up_sw_r, const __grid_constant__ CUtensorMap dy_q_store, const __grid_constant__ CUtensorMap dy_sc_store, const __grid_constant__ CUtensorMap dy_t_store, const __grid_constant__ CUtensorMap dy_sc_t_store, const __grid_constant__ CUtensorMap x_q_store, const __grid_constant__ CUtensorMap x_sc_store, const __grid_constant__ CUtensorMap x_t_store, const __grid_constant__ CUtensorMap x_sc_t_store, const __grid_constant__ CUtensorMap dy_q, const __grid_constant__ CUtensorMap dy_sc, const __grid_constant__ CUtensorMap dy_t, const __grid_constant__ CUtensorMap dy_sc_t, const __grid_constant__ CUtensorMap x_q, const __grid_constant__ CUtensorMap x_sc, const __grid_constant__ CUtensorMap x_t, const __grid_constant__ CUtensorMap x_sc_t, const __grid_constant__ CUtensorMap wg_q, const __grid_constant__ CUtensorMap wg_sc, const __grid_constant__ CUtensorMap wu_q, const __grid_constant__ CUtensorMap wu_sc, const __grid_constant__ CUtensorMap wd_t_q, const __grid_constant__ CUtensorMap wd_t_sc, const __grid_constant__ CUtensorMap wg_t_q, const __grid_constant__ CUtensorMap wg_t_sc, const __grid_constant__ CUtensorMap wu_t_q, const __grid_constant__ CUtensorMap wu_t_sc, const __grid_constant__ CUtensorMap gate_q_store, const __grid_constant__ CUtensorMap gate_sc_store, const __grid_constant__ CUtensorMap up_q_store, const __grid_constant__ CUtensorMap up_sc_store, const __grid_constant__ CUtensorMap h_q_store, const __grid_constant__ CUtensorMap h_sc_store, const __grid_constant__ CUtensorMap h_t_store, const __grid_constant__ CUtensorMap h_sc_t_store, const __grid_constant__ CUtensorMap h_t, const __grid_constant__ CUtensorMap h_sc_t, const __grid_constant__ CUtensorMap dh_tile, const __grid_constant__ CUtensorMap gate_tile, const __grid_constant__ CUtensorMap up_tile, const __grid_constant__ CUtensorMap gate_sc, const __grid_constant__ CUtensorMap up_sc, const __grid_constant__ CUtensorMap dg_store, const __grid_constant__ CUtensorMap dg_sc_store, const __grid_constant__ CUtensorMap du_store, const __grid_constant__ CUtensorMap du_sc_store, const __grid_constant__ CUtensorMap dg_t_store, const __grid_constant__ CUtensorMap dg_sc_t_store, const __grid_constant__ CUtensorMap du_t_store, const __grid_constant__ CUtensorMap du_sc_t_store, const __grid_constant__ CUtensorMap dg_q, const __grid_constant__ CUtensorMap dg_sc, const __grid_constant__ CUtensorMap du_q, const __grid_constant__ CUtensorMap du_sc, const __grid_constant__ CUtensorMap dg_t, const __grid_constant__ CUtensorMap dg_sc_t, const __grid_constant__ CUtensorMap du_t, const __grid_constant__ CUtensorMap du_sc_t, __nv_bfloat16* __restrict__ dx_routed_ptr, float* __restrict__ weights, float* __restrict__ partials, unsigned long long* __restrict__ x_peers, unsigned long long* __restrict__ dy_peers, unsigned long long* __restrict__ dx_peers, unsigned long long* __restrict__ weight_peers, unsigned long long* __restrict__ dweight_peers, int* __restrict__ schedule_rank, int* __restrict__ schedule_token, int* __restrict__ num_tokens, int* __restrict__ counts, int* __restrict__ dh_ready, int* __restrict__ dg_ready, int* __restrict__ dy_ready, int* __restrict__ dx_ready, int* __restrict__ replay_x, int* __restrict__ replay_gu, int* __restrict__ replay_h, int* __restrict__ buffers_done, int* __restrict__ weight_ready, int local_tokens, int hidden, int intermediate, int experts, int topk, int comm_sms, int macro_size, int mini_size, float swiglu_limit)
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
    #define replay_arrived_addr (mbar_base + 0)
    #define swiglu_arrived_addr (mbar_base + 24)
    #define gemm_arrived_addr (mbar_base + 40)
    #define scales_arrived_addr (mbar_base + 88)
    #define gemm_finished_addr (mbar_base + 136)
    #define scales_finished_addr (mbar_base + 184)
    #define output_arrived_addr (mbar_base + 232)
    #define output_finished_addr (mbar_base + 240)
    #define schedule_arrived_addr (mbar_base + 248)
    #define schedule_finished_addr (mbar_base + 256)
    #define drain_arrived_0_addr (mbar_base + 264)
    #define drain_arrived_1_addr (mbar_base + 272)
    #define drain_arrived_2_addr (mbar_base + 280)
    #define drain_arrived_3_addr (mbar_base + 288)
    #define drain_arrived_4_addr (mbar_base + 296)
    #define drain_arrived_5_addr (mbar_base + 304)
    #define drain_arrived_6_addr (mbar_base + 312)
    #define drain_arrived_7_addr (mbar_base + 320)
    #define drain_finished_0_addr (mbar_base + 328)
    #define drain_finished_1_addr (mbar_base + 336)
    #define drain_finished_2_addr (mbar_base + 344)
    #define drain_finished_3_addr (mbar_base + 352)
    #define drain_finished_4_addr (mbar_base + 360)
    #define drain_finished_5_addr (mbar_base + 368)
    #define drain_finished_6_addr (mbar_base + 376)
    #define drain_finished_7_addr (mbar_base + 384)
    #define dispatch_arrived_addr (mbar_base + 392)
    #define combine_arrived_addr (mbar_base + 400)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    const unsigned int clusters_x = gridDim.x / 2;
    const unsigned int cluster_id = ((blockIdx.z * gridDim.y + blockIdx.y) * clusters_x) + blockIdx.x / 2;
    const unsigned int num_clusters = clusters_x * gridDim.y * gridDim.z;

    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));

    // Kernel setup ops
    __nv_bfloat16* a_nt = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int a_nt_addr = smem + 1024;
    __nv_bfloat16* a_atb = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int a_atb_addr = smem + 1024;
    __nv_bfloat16* b_nt = reinterpret_cast<__nv_bfloat16*>(smem_raw + 66560);
    const int b_nt_addr = smem + 66560;
    __nv_bfloat16* b_ab = reinterpret_cast<__nv_bfloat16*>(smem_raw + 66560);
    const int b_ab_addr = smem + 66560;
    __nv_bfloat16* b_nt_hi = reinterpret_cast<__nv_bfloat16*>(smem_raw + 132096);
    const int b_nt_hi_addr = smem + 132096;
    __nv_bfloat16* b_ab_hi = reinterpret_cast<__nv_bfloat16*>(smem_raw + 132096);
    const int b_ab_hi_addr = smem + 132096;
    __nv_bfloat16* d_smem = reinterpret_cast<__nv_bfloat16*>(smem_raw + 207872);
    const int d_smem_addr = smem + 207872;
    unsigned int* d_words = reinterpret_cast<unsigned int*>(smem_raw + 207872);
    const int d_words_addr = smem + 207872;
    __nv_bfloat16* sw_dh = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int sw_dh_addr = smem + 1024;
    __nv_bfloat16* sw_gate = reinterpret_cast<__nv_bfloat16*>(smem_raw + 66560);
    const int sw_gate_addr = smem + 66560;
    __nv_bfloat16* sw_up = reinterpret_cast<__nv_bfloat16*>(smem_raw + 132096);
    const int sw_up_addr = smem + 132096;
    float* sw_router = reinterpret_cast<float*>(smem_raw + 197632);
    const int sw_router_addr = smem + 197632;
    unsigned int* sw_dh_words = reinterpret_cast<unsigned int*>(smem_raw + 1024);
    const int sw_dh_words_addr = smem + 1024;
    unsigned int* sw_gate_words = reinterpret_cast<unsigned int*>(smem_raw + 66560);
    const int sw_gate_words_addr = smem + 66560;
    unsigned int* sw_up_words = reinterpret_cast<unsigned int*>(smem_raw + 132096);
    const int sw_up_words_addr = smem + 132096;
    __nv_bfloat16* replay_gate = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int replay_gate_addr = smem + 1024;
    __nv_bfloat16* replay_up = reinterpret_cast<__nv_bfloat16*>(smem_raw + 99328);
    const int replay_up_addr = smem + 99328;
    __nv_bfloat16* hidden_staging = reinterpret_cast<__nv_bfloat16*>(smem_raw + 197632);
    const int hidden_staging_addr = smem + 197632;
    unsigned int* hidden_words = reinterpret_cast<unsigned int*>(smem_raw + 197632);
    const int hidden_words_addr = smem + 197632;
    unsigned int* smem_v19 = reinterpret_cast<unsigned int*>(smem_raw + 1024);
    const int smem_v19_addr = smem + 1024;
    unsigned int* smem_v20 = reinterpret_cast<unsigned int*>(smem_raw + 17408);
    const int smem_v20_addr = smem + 17408;
    unsigned int* smem_v21 = reinterpret_cast<unsigned int*>(smem_raw + 99328);
    const int smem_v21_addr = smem + 99328;
    unsigned int* smem_v22 = reinterpret_cast<unsigned int*>(smem_raw + 115712);
    const int smem_v22_addr = smem + 115712;
    unsigned int* smem_v23 = reinterpret_cast<unsigned int*>(smem_raw + 33792);
    const int smem_v23_addr = smem + 33792;
    unsigned int* smem_v24 = reinterpret_cast<unsigned int*>(smem_raw + 50176);
    const int smem_v24_addr = smem + 50176;
    unsigned int* smem_v25 = reinterpret_cast<unsigned int*>(smem_raw + 132096);
    const int smem_v25_addr = smem + 132096;
    unsigned int* smem_v26 = reinterpret_cast<unsigned int*>(smem_raw + 148480);
    const int smem_v26_addr = smem + 148480;
    unsigned int* smem_v27 = reinterpret_cast<unsigned int*>(smem_raw + 66560);
    const int smem_v27_addr = smem + 66560;
    unsigned int* smem_v28 = reinterpret_cast<unsigned int*>(smem_raw + 82944);
    const int smem_v28_addr = smem + 82944;
    unsigned int* smem_v29 = reinterpret_cast<unsigned int*>(smem_raw + 164864);
    const int smem_v29_addr = smem + 164864;
    unsigned int* smem_v30 = reinterpret_cast<unsigned int*>(smem_raw + 181248);
    const int smem_v30_addr = smem + 181248;
    unsigned int* dispatch_words = reinterpret_cast<unsigned int*>(smem_raw + 1024);
    const int dispatch_words_addr = smem + 1024;
    float* dispatch_weights = reinterpret_cast<float*>(smem_raw + 199680);
    const int dispatch_weights_addr = smem + 199680;
    __nv_bfloat16* smem_v33 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int smem_v33_addr = smem + 1024;
    __nv_bfloat16* smem_v34 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1280);
    const int smem_v34_addr = smem + 1280;
    __nv_bfloat16* smem_v35 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1536);
    const int smem_v35_addr = smem + 1536;
    __nv_bfloat16* smem_v36 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1792);
    const int smem_v36_addr = smem + 1792;
    unsigned int* smem_v37 = reinterpret_cast<unsigned int*>(smem_raw + 132096);
    const int smem_v37_addr = smem + 132096;
    unsigned int* smem_v38 = reinterpret_cast<unsigned int*>(smem_raw + 148480);
    const int smem_v38_addr = smem + 148480;
    unsigned int* smem_v39 = reinterpret_cast<unsigned int*>(smem_raw + 164864);
    const int smem_v39_addr = smem + 164864;
    unsigned int* smem_v40 = reinterpret_cast<unsigned int*>(smem_raw + 165376);
    const int smem_v40_addr = smem + 165376;
    unsigned int* smem_v41 = reinterpret_cast<unsigned int*>(smem_raw + 165888);
    const int smem_v41_addr = smem + 165888;
    unsigned int* smem_v42 = reinterpret_cast<unsigned int*>(smem_raw + 182272);
    const int smem_v42_addr = smem + 182272;
    unsigned int* smem_v43 = reinterpret_cast<unsigned int*>(smem_raw + 198656);
    const int smem_v43_addr = smem + 198656;
    unsigned int* smem_v44 = reinterpret_cast<unsigned int*>(smem_raw + 199168);
    const int smem_v44_addr = smem + 199168;
    __nv_bfloat16* combine_smem = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int combine_smem_addr = smem + 1024;
    uint8_t* smem_v46 = reinterpret_cast<uint8_t*>(smem_raw + 1024);
    const int smem_v46_addr = smem + 1024;
    uint8_t* smem_v47 = reinterpret_cast<uint8_t*>(smem_raw + 66560);
    const int smem_v47_addr = smem + 66560;
    uint8_t* smem_v48 = reinterpret_cast<uint8_t*>(smem_raw + 132096);
    const int smem_v48_addr = smem + 132096;
    uint8_t* smem_v49 = reinterpret_cast<uint8_t*>(smem_raw + 197632);
    const int smem_v49_addr = smem + 197632;
    uint8_t* smem_v50 = reinterpret_cast<uint8_t*>(smem_raw + 199680);
    const int smem_v50_addr = smem + 199680;
    uint8_t* smem_v51 = reinterpret_cast<uint8_t*>(smem_raw + 203776);
    const int smem_v51_addr = smem + 203776;
    uint8_t* smem_v52 = reinterpret_cast<uint8_t*>(smem_raw + 1024);
    const int smem_v52_addr = smem + 1024;
    uint8_t* smem_v53 = reinterpret_cast<uint8_t*>(smem_raw + 99328);
    const int smem_v53_addr = smem + 99328;
    uint8_t* smem_v54 = reinterpret_cast<uint8_t*>(smem_raw + 197632);
    const int smem_v54_addr = smem + 197632;
    uint8_t* smem_v55 = reinterpret_cast<uint8_t*>(smem_raw + 200704);
    const int smem_v55_addr = smem + 200704;
    unsigned int* smem_v56 = reinterpret_cast<unsigned int*>(smem_raw + 207872);
    const int smem_v56_addr = smem + 207872;
    unsigned int* smem_v57 = reinterpret_cast<unsigned int*>(smem_raw + 224256);
    const int smem_v57_addr = smem + 224256;
    unsigned int* smem_v58 = reinterpret_cast<unsigned int*>(smem_raw + 228352);
    const int smem_v58_addr = smem + 228352;
    unsigned int* smem_v59 = reinterpret_cast<unsigned int*>(smem_raw + 228864);
    const int smem_v59_addr = smem + 228864;
    __nv_bfloat16* smem_v60 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int smem_v60_addr = smem + 1024;
    __nv_bfloat16* smem_v61 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 33792);
    const int smem_v61_addr = smem + 33792;
    unsigned int* smem_v62 = reinterpret_cast<unsigned int*>(smem_raw + 1024);
    const int smem_v62_addr = smem + 1024;
    unsigned int* smem_v63 = reinterpret_cast<unsigned int*>(smem_raw + 33792);
    const int smem_v63_addr = smem + 33792;
    unsigned int* smem_v64 = reinterpret_cast<unsigned int*>(smem_raw + 66560);
    const int smem_v64_addr = smem + 66560;
    unsigned int* smem_v65 = reinterpret_cast<unsigned int*>(smem_raw + 82944);
    const int smem_v65_addr = smem + 82944;
    unsigned int* smem_v66 = reinterpret_cast<unsigned int*>(smem_raw + 99328);
    const int smem_v66_addr = smem + 99328;
    unsigned int* smem_v67 = reinterpret_cast<unsigned int*>(smem_raw + 115712);
    const int smem_v67_addr = smem + 115712;
    unsigned int* smem_v68 = reinterpret_cast<unsigned int*>(smem_raw + 132096);
    const int smem_v68_addr = smem + 132096;
    unsigned int* smem_v69 = reinterpret_cast<unsigned int*>(smem_raw + 132608);
    const int smem_v69_addr = smem + 132608;
    unsigned int* smem_v70 = reinterpret_cast<unsigned int*>(smem_raw + 133120);
    const int smem_v70_addr = smem + 133120;
    unsigned int* smem_v71 = reinterpret_cast<unsigned int*>(smem_raw + 133632);
    const int smem_v71_addr = smem + 133632;
    uint16_t* smem_v72 = reinterpret_cast<uint16_t*>(smem_raw + 132096);
    const int smem_v72_addr = smem + 132096;
    uint16_t* smem_v73 = reinterpret_cast<uint16_t*>(smem_raw + 132608);
    const int smem_v73_addr = smem + 132608;
    uint16_t* smem_v74 = reinterpret_cast<uint16_t*>(smem_raw + 133120);
    const int smem_v74_addr = smem + 133120;
    uint16_t* smem_v75 = reinterpret_cast<uint16_t*>(smem_raw + 133632);
    const int smem_v75_addr = smem + 133632;
    __nv_bfloat16* smem_v76 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 134144);
    const int smem_v76_addr = smem + 134144;
    unsigned int* smem_v77 = reinterpret_cast<unsigned int*>(smem_raw + 134144);
    const int smem_v77_addr = smem + 134144;
    unsigned int* smem_v78 = reinterpret_cast<unsigned int*>(smem_raw + 166912);
    const int smem_v78_addr = smem + 166912;
    unsigned int* smem_v79 = reinterpret_cast<unsigned int*>(smem_raw + 183296);
    const int smem_v79_addr = smem + 183296;
    unsigned int* smem_v80 = reinterpret_cast<unsigned int*>(smem_raw + 199680);
    const int smem_v80_addr = smem + 199680;
    unsigned int* smem_v81 = reinterpret_cast<unsigned int*>(smem_raw + 200192);
    const int smem_v81_addr = smem + 200192;
    float* smem_v82 = reinterpret_cast<float*>(smem_raw + 200704);
    const int smem_v82_addr = smem + 200704;
    int tokens = num_tokens[0];
    int i_tiles = intermediate / 128;
    int shared_down = local_tokens / 256 * ((intermediate + 511) / 512);
    int shared_swiglu = (local_tokens / 128 * (intermediate / 128) + 3) / 4;
    int shared_dx = local_tokens / 256 * ((hidden + 511) / 512);
    int _max_0 = ((intermediate / 256 * ((hidden + 512 - 1) / 512)) > (hidden / 256 * ((intermediate + 512 - 1) / 512)) ? (intermediate / 256 * ((hidden + 512 - 1) / 512)) : (hidden / 256 * ((intermediate + 512 - 1) / 512)));
    int shared_wgrad = _max_0;
    int _max_1 = ((intermediate / 256 * ((hidden + 256 - 1) / 256)) > (hidden / 256 * ((intermediate + 256 - 1) / 256)) ? (intermediate / 256 * ((hidden + 256 - 1) / 256)) : (hidden / 256 * ((intermediate + 256 - 1) / 256)));
    int routed_wgrad = _max_1;
    int shared_tasks = shared_down + shared_swiglu + shared_dx + 3 * shared_wgrad;
    int mini_down = mini_size / 256 * ((intermediate + 256 - 1) / 256);
    int mini_swiglu = (mini_size / 128 * (intermediate / 128) + 3) / 4;
    int mini_dx = mini_size / 256 * ((hidden + 256 - 1) / 256);
    int mini_replay_swiglu = (mini_size / 128 * (intermediate / 128) + 5) / 6;
    int mini_replay_gate = mini_size / 256 * (intermediate / 256);
    int mini_bwd = mini_down + mini_swiglu + mini_dx;
    int mini_replay = 2 * mini_replay_gate + mini_replay_swiglu;
    int wgrad_tasks = 3 * experts * routed_wgrad;
    int macros = (tokens + macro_size - 1) / macro_size;
    int minis = (tokens + mini_size - 1) / mini_size;
    int _min_0 = ((tokens) < (macro_size) ? (tokens) : (macro_size));
    int saved_minis = (_min_0 + mini_size - 1) / mini_size;
    int true_compute = shared_tasks + minis * mini_bwd + (minis - saved_minis) * mini_replay + macros * wgrad_tasks;
    int comm_clusters = comm_sms / 2;
    int true_clusters = comm_clusters + true_compute;
    if (true_clusters <= bid / 2) return;
    asm volatile("setmaxnreg.inc.sync.aligned.u32 256;");

    // Mbarrier init (28 pipeline groups, 0 ordered-sequence groups, 57 barriers)
    // Mbarriers at smem_raw[0..456)

    if (threadIdx.x == 0) {
        // replay_arrived: 3 barriers, init_count=1
        mbarrier_init(smem + 0, 1);
        mbarrier_init(smem + 8, 1);
        mbarrier_init(smem + 16, 1);
        // swiglu_arrived: 2 barriers, init_count=1
        mbarrier_init(smem + 24, 1);
        mbarrier_init(smem + 32, 1);
        // gemm_arrived: 6 barriers, init_count=1
        mbarrier_init(smem + 40, 1);
        mbarrier_init(smem + 48, 1);
        mbarrier_init(smem + 56, 1);
        mbarrier_init(smem + 64, 1);
        mbarrier_init(smem + 72, 1);
        mbarrier_init(smem + 80, 1);
        // scales_arrived: 6 barriers, init_count=1
        mbarrier_init(smem + 88, 1);
        mbarrier_init(smem + 96, 1);
        mbarrier_init(smem + 104, 1);
        mbarrier_init(smem + 112, 1);
        mbarrier_init(smem + 120, 1);
        mbarrier_init(smem + 128, 1);
        // gemm_finished: 6 barriers, init_count=1
        mbarrier_init(smem + 136, 1);
        mbarrier_init(smem + 144, 1);
        mbarrier_init(smem + 152, 1);
        mbarrier_init(smem + 160, 1);
        mbarrier_init(smem + 168, 1);
        mbarrier_init(smem + 176, 1);
        // scales_finished: 6 barriers, init_count=1
        mbarrier_init(smem + 184, 1);
        mbarrier_init(smem + 192, 1);
        mbarrier_init(smem + 200, 1);
        mbarrier_init(smem + 208, 1);
        mbarrier_init(smem + 216, 1);
        mbarrier_init(smem + 224, 1);
        // output_arrived: 1 barriers, init_count=1
        mbarrier_init(smem + 232, 1);
        // output_finished: 1 barriers, init_count=2
        mbarrier_init(smem + 240, 2);
        // --- pipeline 'schedule_pipe' ---
        // schedule_arrived: 1 barriers, init_count=1
        mbarrier_init(smem + 248, 1);
        // schedule_finished: 1 barriers, init_count=16
        mbarrier_init(smem + 256, 16);
        // --- pipeline 'drain_pipe_0' ---
        // drain_arrived_0: 1 barriers, init_count=1
        mbarrier_init(smem + 264, 1);
        // --- pipeline 'drain_pipe_1' ---
        // drain_arrived_1: 1 barriers, init_count=1
        mbarrier_init(smem + 272, 1);
        // --- pipeline 'drain_pipe_2' ---
        // drain_arrived_2: 1 barriers, init_count=1
        mbarrier_init(smem + 280, 1);
        // --- pipeline 'drain_pipe_3' ---
        // drain_arrived_3: 1 barriers, init_count=1
        mbarrier_init(smem + 288, 1);
        // --- pipeline 'drain_pipe_4' ---
        // drain_arrived_4: 1 barriers, init_count=1
        mbarrier_init(smem + 296, 1);
        // --- pipeline 'drain_pipe_5' ---
        // drain_arrived_5: 1 barriers, init_count=1
        mbarrier_init(smem + 304, 1);
        // --- pipeline 'drain_pipe_6' ---
        // drain_arrived_6: 1 barriers, init_count=1
        mbarrier_init(smem + 312, 1);
        // --- pipeline 'drain_pipe_7' ---
        // drain_arrived_7: 1 barriers, init_count=1
        mbarrier_init(smem + 320, 1);
        // --- pipeline 'drain_pipe_0' ---
        // drain_finished_0: 1 barriers, init_count=2
        mbarrier_init(smem + 328, 2);
        // --- pipeline 'drain_pipe_1' ---
        // drain_finished_1: 1 barriers, init_count=2
        mbarrier_init(smem + 336, 2);
        // --- pipeline 'drain_pipe_2' ---
        // drain_finished_2: 1 barriers, init_count=2
        mbarrier_init(smem + 344, 2);
        // --- pipeline 'drain_pipe_3' ---
        // drain_finished_3: 1 barriers, init_count=2
        mbarrier_init(smem + 352, 2);
        // --- pipeline 'drain_pipe_4' ---
        // drain_finished_4: 1 barriers, init_count=2
        mbarrier_init(smem + 360, 2);
        // --- pipeline 'drain_pipe_5' ---
        // drain_finished_5: 1 barriers, init_count=2
        mbarrier_init(smem + 368, 2);
        // --- pipeline 'drain_pipe_6' ---
        // drain_finished_6: 1 barriers, init_count=2
        mbarrier_init(smem + 376, 2);
        // --- pipeline 'drain_pipe_7' ---
        // drain_finished_7: 1 barriers, init_count=2
        mbarrier_init(smem + 384, 2);
        // dispatch_arrived: 1 barriers, init_count=1
        mbarrier_init(smem + 392, 1);
        // combine_arrived: 7 barriers, init_count=1
        mbarrier_init(smem + 400, 1);
        mbarrier_init(smem + 408, 1);
        mbarrier_init(smem + 416, 1);
        mbarrier_init(smem + 424, 1);
        mbarrier_init(smem + 432, 1);
        mbarrier_init(smem + 440, 1);
        mbarrier_init(smem + 448, 1);
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }

    __syncwarp();

    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");

    // TMEM alloc (512 columns, 512 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 456);
    if (warp == 0) {
        int _tmem_hold = smem + 456;
        asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(512) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;");
    }

    __syncthreads();
    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_tmem_sfa = taddr + 256;
    const int tmem_tmem_sfb = taddr + 268;
    const int tmem_accumulator = taddr;
    unsigned int taddr_1 = reinterpret_cast<const volatile unsigned int*>(reinterpret_cast<uint8_t*>(smem_raw) + CAKE_TMEM_HOLD_OFFSET)[0];
    int cta_rank_0 = cta_rank;
    int cluster = bid / 2;
    unsigned int gemm_bits = 4294901760;
    unsigned int swiglu_bits = 4294901760;
    unsigned int replay_bits = 4294901760;
    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    if (cluster < comm_clusters) {
        unsigned int dispatch_bits = 4294901760;
        unsigned int combine_bits = 4294901760;
        if (tid == 0) {
        }
        __syncthreads();
        int offset = 0;
        int _min_1 = ((macro_size) < (tokens - offset) ? (macro_size) : (tokens - offset));
        int rows = _min_1;
        #pragma unroll 1
        for (int row = (cluster * 2 + cta_rank_0) * 256 + tid; row < rows; row += comm_sms * 256) {
            int peer = schedule_rank[offset + row];
            int token = schedule_token[offset + row];
            float value = 0.0f;
            if (peer >= 0) {
                value = reinterpret_cast<float*>(weight_peers[peer])[(int)token];
            }
            weights[row] = value;
        }
        __syncthreads();
        if (tid == 0) {
            asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(weight_ready)) + (0))), "r"(static_cast<unsigned int>(1)) : "memory");
            int32_t _relaxed_ld_0;
            asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_0) : "l"(weight_ready + 0) : "memory");
            int value_1 = _relaxed_ld_0;
            while (value_1 < comm_sms) {
                asm volatile("nanosleep.u32 %0;" :: "r"(16));
                int32_t _relaxed_ld_1;
                asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_1) : "l"(weight_ready + 0) : "memory");
                value_1 = _relaxed_ld_1;
            }
            asm volatile("fence.acquire.gpu;" ::: "memory");
        }
        __syncthreads();
        int _min_2 = ((tokens) < (macro_size) ? (tokens) : (macro_size));
        int first_rows = _min_2;
        #pragma unroll 1
        for (int task = cluster * 2 + cta_rank_0; task < first_rows / 128 * ((hidden + 511) / 512); task += comm_sms) {
            unsigned int phase_bits = dispatch_bits;
            int col_blocks = (hidden + 511) / 512;
            int macro_offset = 0;
            int _min_3 = ((macro_size) < (tokens - macro_offset) ? (macro_size) : (tokens - macro_offset));
            int macro_tokens = _min_3;
            if (task < macro_tokens / 128 * col_blocks) {
                int row_1 = task / col_blocks * 128;
                int col_block = task % col_blocks;
                int _min_4 = ((512) < (hidden - col_block * 512) ? (512) : (hidden - col_block * 512));
                int chunk_cols = _min_4;
                unsigned int chunk_bytes = (unsigned int)(chunk_cols * 2);
                int peer_1 = -1;
                int peer_token = -1;
                if (tid < 128) {
                    peer_1 = schedule_rank[macro_offset + row_1 + tid];
                    peer_token = schedule_token[macro_offset + row_1 + tid];
                }
                uint32_t _cta_count_0 = __syncthreads_count(peer_1 >= 0);
                if (tid == 0) {
                    int previous_offset = -1 * macro_size;
                    int _min_5 = ((macro_size) < (tokens - previous_offset) ? (macro_size) : (tokens - previous_offset));
                    int previous_tokens = _min_5;
                    if (row_1 < previous_tokens) {
                        int previous_mini = (previous_offset + row_1) / mini_size;
                        int _min_6 = ((mini_size) < (tokens - previous_mini * mini_size) ? (mini_size) : (tokens - previous_mini * mini_size));
                        int mini_rows = _min_6;
                        int required = (mini_rows + 255) / 256 * (hidden / 256) * 2;
                    }
                    mbarrier_arrive_expect_tx(dispatch_arrived_addr, _cta_count_0 * chunk_bytes);
                }
                __syncthreads();
                if (tid < 128) {
                    dispatch_weights[tid] = ((peer_1 >= 0) ? weights[row_1 + tid] : 0.0f);
                }
                if (peer_1 >= 0) {
                    cp_async_bulk_gmem2smem(smem_v33_addr + (unsigned int)(tid * 512 * 2), reinterpret_cast<const void*>(reinterpret_cast<const uint8_t*>(reinterpret_cast<__nv_bfloat16*>(dy_peers[peer_1])) + ((unsigned long long)((unsigned long long)(peer_token / topk) * (unsigned long long)hidden + (unsigned long long)(col_block * 512)) * (unsigned long long)2)), chunk_cols * 2, dispatch_arrived_addr);
                } else if (tid < 128) {
                    #pragma unroll
                    for (int vec = 0; vec < 64; vec++) {
                        asm volatile("st.v4.u32 [%0], {%1, %2, %3, %4};" :: "l"((uint64_t)(reinterpret_cast<uint8_t*>(dispatch_words) + (tid * 1024 + vec * 16))), "r"(static_cast<uint32_t>((unsigned int)0)), "r"(static_cast<uint32_t>((unsigned int)0)), "r"(static_cast<uint32_t>((unsigned int)0)), "r"(static_cast<uint32_t>((unsigned int)0)) : "memory");
                    }
                }
                mbarrier_wait(dispatch_arrived_addr, phase_bits & 1);
                phase_bits = phase_bits ^ 1;
                int row_tile = row_1 / 128;
                int k_tiles = hidden / 128;
                int macro_tiles = macro_size / 128;
                if (chunk_cols / 128 > 0) {
                    if (tid == 0) {
                        asm volatile("cp.async.bulk.wait_group.read 1;");
                    }
                    __syncthreads();
                    if (tid < 128) {
                        int row_0 = tid;
                        row_0 = tid % 64 * 2 + tid / 64;
                        unsigned int scale_word = 0;
                        #pragma unroll 1
                        for (int j = 0; j < 4; j++) {
                            int k_block = (j + tid / 8) % 4;
                            unsigned int pairs[16];
                            #pragma unroll
                            for (int k = 0; k < 16; k++) {
                                int col = k_block * 32 + (tid * 4 + k * 2) % 32;
                                float x0 = 0.0f;
                                float x1 = 0.0f;
                                x0 = (float)smem_v33[col * 512 + row_0];
                                x1 = (float)smem_v33[(col + 1) * 512 + row_0];
                                x0 = x0 * dispatch_weights[col];
                                x1 = x1 * dispatch_weights[col + 1];
                                __nv_bfloat162 _bf16x2_0 = __float22bfloat162_rn(make_float2(x0, x1));
                                pairs[k] = __as_u32(_bf16x2_0);
                            }
                            uint32_t _bf16x2_abs_0;
                            asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_0) : "r"(pairs[0]));
                            unsigned int amax_pair = _bf16x2_abs_0;
                            #pragma unroll
                            for (int i = 1; i < 16; i++) {
                                uint32_t _bf16x2_abs_1;
                                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_1) : "r"(pairs[i]));
                                uint32_t _bf16x2_max_0;
                                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_0) : "r"(amax_pair), "r"(_bf16x2_abs_1));
                                amax_pair = _bf16x2_max_0;
                            }
                            uint16_t _bf16_max_0;
                            asm("max.bf16 %0, %1, %2;" : "=h"(_bf16_max_0) : "h"((uint16_t)(amax_pair & 65535)), "h"((uint16_t)(amax_pair >> 16)));
                            float _cvt_f32_bf16_0;
                            asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_0) : "h"((uint16_t)(_bf16_max_0)));
                            float amax = _cvt_f32_bf16_0;
                            float _fmax_0 = fmaxf(amax * 0.002232142857f, 1e-12f);
                            float scale = _fmax_0;
                            uint16_t _ue8m0x2_f32_0;
                            asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_ue8m0x2_f32_0) : "f"(scale), "f"(scale));
                            unsigned int scale_byte = (unsigned int)_ue8m0x2_f32_0 & 255;
                            unsigned int inverse_lane = 254 - scale_byte << 7;
                            unsigned int inverse = inverse_lane | inverse_lane << 16;
                            unsigned int words[8];
                            #pragma unroll
                            for (int i_1 = 0; i_1 < 8; i_1++) {
                                uint32_t _bf16x2_mul_0;
                                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_0) : "r"(pairs[i_1 * 2]), "r"(inverse));
                                uint16_t _e4m3x2_0;
                                asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_0) : "r"(_bf16x2_mul_0));
                                uint32_t _bf16x2_mul_1;
                                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_1) : "r"(pairs[i_1 * 2 + 1]), "r"(inverse));
                                uint16_t _e4m3x2_1;
                                asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_1) : "r"(_bf16x2_mul_1));
                                words[i_1] = (unsigned int)_e4m3x2_0 | (unsigned int)_e4m3x2_1 << 16;
                            }
                            scale_word = scale_word | scale_byte << (unsigned int)(k_block * 8);
                            #pragma unroll
                            for (int k_1 = 0; k_1 < 8; k_1++) {
                                int col_1 = k_block * 32 + (tid * 4 + k_1 * 4) % 32;
                                smem_v41[(row_0 * 128 + col_1) / 4] = words[k_1];
                            }
                        }
                        smem_v43[row_0 % 32 * 4 + row_0 / 32] = scale_word;
                    } else {
                        int row_0_1 = tid - 128;
                        unsigned int scale_word_1 = 0;
                        #pragma unroll 1
                        for (int j_1 = 0; j_1 < 4; j_1++) {
                            int k_block_1 = (j_1 + (tid - 128) / 8) % 4;
                            unsigned int pairs_1[16];
                            #pragma unroll
                            for (int k_2 = 0; k_2 < 16; k_2++) {
                                int col_2 = k_block_1 * 32 + ((tid - 128) * 4 + k_2 * 2) % 32;
                                float x0_1 = 0.0f;
                                float x1_1 = 0.0f;
                                float2 _cvt_f32_0 = __bfloat1622float2(__as_bf16x2(dispatch_words[(row_0_1 * 512 + col_2) / 2]));
                                __nv_bfloat162 _bf16x2_1 = __float22bfloat162_rn(make_float2(_cvt_f32_0.x * dispatch_weights[row_0_1], _cvt_f32_0.y * dispatch_weights[row_0_1]));
                                pairs_1[k_2] = __as_u32(_bf16x2_1);
                            }
                            uint32_t _bf16x2_abs_2;
                            asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_2) : "r"(pairs_1[0]));
                            unsigned int amax_pair_1 = _bf16x2_abs_2;
                            #pragma unroll
                            for (int i_2 = 1; i_2 < 16; i_2++) {
                                uint32_t _bf16x2_abs_3;
                                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_3) : "r"(pairs_1[i_2]));
                                uint32_t _bf16x2_max_1;
                                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_1) : "r"(amax_pair_1), "r"(_bf16x2_abs_3));
                                amax_pair_1 = _bf16x2_max_1;
                            }
                            uint16_t _bf16_max_1;
                            asm("max.bf16 %0, %1, %2;" : "=h"(_bf16_max_1) : "h"((uint16_t)(amax_pair_1 & 65535)), "h"((uint16_t)(amax_pair_1 >> 16)));
                            float _cvt_f32_bf16_1;
                            asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_1) : "h"((uint16_t)(_bf16_max_1)));
                            float amax_1 = _cvt_f32_bf16_1;
                            float _fmax_1 = fmaxf(amax_1 * 0.002232142857f, 1e-12f);
                            float scale_1 = _fmax_1;
                            uint16_t _ue8m0x2_f32_1;
                            asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_ue8m0x2_f32_1) : "f"(scale_1), "f"(scale_1));
                            unsigned int scale_byte_1 = (unsigned int)_ue8m0x2_f32_1 & 255;
                            unsigned int inverse_lane_1 = 254 - scale_byte_1 << 7;
                            unsigned int inverse_1 = inverse_lane_1 | inverse_lane_1 << 16;
                            unsigned int words_1[8];
                            #pragma unroll
                            for (int i_3 = 0; i_3 < 8; i_3++) {
                                uint32_t _bf16x2_mul_2;
                                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_2) : "r"(pairs_1[i_3 * 2]), "r"(inverse_1));
                                uint16_t _e4m3x2_2;
                                asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_2) : "r"(_bf16x2_mul_2));
                                uint32_t _bf16x2_mul_3;
                                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_3) : "r"(pairs_1[i_3 * 2 + 1]), "r"(inverse_1));
                                uint16_t _e4m3x2_3;
                                asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_3) : "r"(_bf16x2_mul_3));
                                words_1[i_3] = (unsigned int)_e4m3x2_2 | (unsigned int)_e4m3x2_3 << 16;
                            }
                            scale_word_1 = scale_word_1 | scale_byte_1 << (unsigned int)(k_block_1 * 8);
                            #pragma unroll
                            for (int k_3 = 0; k_3 < 8; k_3++) {
                                int col_3 = k_block_1 * 32 + ((tid - 128) * 4 + k_3 * 4) % 32;
                                smem_v37[(row_0_1 * 128 + col_3) / 4] = words_1[k_3];
                            }
                        }
                        smem_v39[row_0_1 % 32 * 4 + row_0_1 / 32] = scale_word_1;
                    }
                    __syncthreads();
                    if (tid == 0) {
                        int col_tile = col_block * 4;
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        tma_store_2d((&dy_q_store), col_tile * 128, row_1, smem_v37_addr);
                        tma_store_3d((&dy_sc_store), 0, 0, row_tile * k_tiles + col_tile, smem_v39_addr);
                        tma_store_2d((&dy_t_store), row_1, col_tile * 128, smem_v41_addr);
                        tma_store_3d((&dy_sc_t_store), 0, 0, col_tile * macro_tiles + row_tile, smem_v43_addr);
                        asm volatile("cp.async.bulk.commit_group;");
                    }
                }
                if (chunk_cols / 128 > 1) {
                    if (tid == 0) {
                        asm volatile("cp.async.bulk.wait_group.read 1;");
                    }
                    __syncthreads();
                    if (tid < 128) {
                        int row_0_2 = tid;
                        row_0_2 = tid % 64 * 2 + tid / 64;
                        unsigned int scale_word_2 = 0;
                        #pragma unroll 1
                        for (int j_2 = 0; j_2 < 4; j_2++) {
                            int k_block_2 = (j_2 + tid / 8) % 4;
                            unsigned int pairs_2[16];
                            #pragma unroll
                            for (int k_4 = 0; k_4 < 16; k_4++) {
                                int col_4 = k_block_2 * 32 + (tid * 4 + k_4 * 2) % 32;
                                float x0_2 = 0.0f;
                                float x1_2 = 0.0f;
                                x0_2 = (float)smem_v34[col_4 * 512 + row_0_2];
                                x1_2 = (float)smem_v34[(col_4 + 1) * 512 + row_0_2];
                                x0_2 = x0_2 * dispatch_weights[col_4];
                                x1_2 = x1_2 * dispatch_weights[col_4 + 1];
                                __nv_bfloat162 _bf16x2_2 = __float22bfloat162_rn(make_float2(x0_2, x1_2));
                                pairs_2[k_4] = __as_u32(_bf16x2_2);
                            }
                            uint32_t _bf16x2_abs_4;
                            asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_4) : "r"(pairs_2[0]));
                            unsigned int amax_pair_2 = _bf16x2_abs_4;
                            #pragma unroll
                            for (int i_4 = 1; i_4 < 16; i_4++) {
                                uint32_t _bf16x2_abs_5;
                                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_5) : "r"(pairs_2[i_4]));
                                uint32_t _bf16x2_max_2;
                                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_2) : "r"(amax_pair_2), "r"(_bf16x2_abs_5));
                                amax_pair_2 = _bf16x2_max_2;
                            }
                            uint16_t _bf16_max_2;
                            asm("max.bf16 %0, %1, %2;" : "=h"(_bf16_max_2) : "h"((uint16_t)(amax_pair_2 & 65535)), "h"((uint16_t)(amax_pair_2 >> 16)));
                            float _cvt_f32_bf16_2;
                            asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_2) : "h"((uint16_t)(_bf16_max_2)));
                            float amax_2 = _cvt_f32_bf16_2;
                            float _fmax_2 = fmaxf(amax_2 * 0.002232142857f, 1e-12f);
                            float scale_2 = _fmax_2;
                            uint16_t _ue8m0x2_f32_2;
                            asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_ue8m0x2_f32_2) : "f"(scale_2), "f"(scale_2));
                            unsigned int scale_byte_2 = (unsigned int)_ue8m0x2_f32_2 & 255;
                            unsigned int inverse_lane_2 = 254 - scale_byte_2 << 7;
                            unsigned int inverse_2 = inverse_lane_2 | inverse_lane_2 << 16;
                            unsigned int words_2[8];
                            #pragma unroll
                            for (int i_5 = 0; i_5 < 8; i_5++) {
                                uint32_t _bf16x2_mul_4;
                                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_4) : "r"(pairs_2[i_5 * 2]), "r"(inverse_2));
                                uint16_t _e4m3x2_4;
                                asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_4) : "r"(_bf16x2_mul_4));
                                uint32_t _bf16x2_mul_5;
                                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_5) : "r"(pairs_2[i_5 * 2 + 1]), "r"(inverse_2));
                                uint16_t _e4m3x2_5;
                                asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_5) : "r"(_bf16x2_mul_5));
                                words_2[i_5] = (unsigned int)_e4m3x2_4 | (unsigned int)_e4m3x2_5 << 16;
                            }
                            scale_word_2 = scale_word_2 | scale_byte_2 << (unsigned int)(k_block_2 * 8);
                            #pragma unroll
                            for (int k_5 = 0; k_5 < 8; k_5++) {
                                int col_5 = k_block_2 * 32 + (tid * 4 + k_5 * 4) % 32;
                                smem_v42[(row_0_2 * 128 + col_5) / 4] = words_2[k_5];
                            }
                        }
                        smem_v44[row_0_2 % 32 * 4 + row_0_2 / 32] = scale_word_2;
                    } else {
                        int row_0_3 = tid - 128;
                        unsigned int scale_word_3 = 0;
                        #pragma unroll 1
                        for (int j_3 = 0; j_3 < 4; j_3++) {
                            int k_block_3 = (j_3 + (tid - 128) / 8) % 4;
                            unsigned int pairs_3[16];
                            #pragma unroll
                            for (int k_6 = 0; k_6 < 16; k_6++) {
                                int col_6 = k_block_3 * 32 + ((tid - 128) * 4 + k_6 * 2) % 32;
                                float x0_3 = 0.0f;
                                float x1_3 = 0.0f;
                                float2 _cvt_f32_1 = __bfloat1622float2(__as_bf16x2(dispatch_words[64 + (row_0_3 * 512 + col_6) / 2]));
                                __nv_bfloat162 _bf16x2_3 = __float22bfloat162_rn(make_float2(_cvt_f32_1.x * dispatch_weights[row_0_3], _cvt_f32_1.y * dispatch_weights[row_0_3]));
                                pairs_3[k_6] = __as_u32(_bf16x2_3);
                            }
                            uint32_t _bf16x2_abs_6;
                            asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_6) : "r"(pairs_3[0]));
                            unsigned int amax_pair_3 = _bf16x2_abs_6;
                            #pragma unroll
                            for (int i_6 = 1; i_6 < 16; i_6++) {
                                uint32_t _bf16x2_abs_7;
                                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_7) : "r"(pairs_3[i_6]));
                                uint32_t _bf16x2_max_3;
                                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_3) : "r"(amax_pair_3), "r"(_bf16x2_abs_7));
                                amax_pair_3 = _bf16x2_max_3;
                            }
                            uint16_t _bf16_max_3;
                            asm("max.bf16 %0, %1, %2;" : "=h"(_bf16_max_3) : "h"((uint16_t)(amax_pair_3 & 65535)), "h"((uint16_t)(amax_pair_3 >> 16)));
                            float _cvt_f32_bf16_3;
                            asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_3) : "h"((uint16_t)(_bf16_max_3)));
                            float amax_3 = _cvt_f32_bf16_3;
                            float _fmax_3 = fmaxf(amax_3 * 0.002232142857f, 1e-12f);
                            float scale_3 = _fmax_3;
                            uint16_t _ue8m0x2_f32_3;
                            asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_ue8m0x2_f32_3) : "f"(scale_3), "f"(scale_3));
                            unsigned int scale_byte_3 = (unsigned int)_ue8m0x2_f32_3 & 255;
                            unsigned int inverse_lane_3 = 254 - scale_byte_3 << 7;
                            unsigned int inverse_3 = inverse_lane_3 | inverse_lane_3 << 16;
                            unsigned int words_3[8];
                            #pragma unroll
                            for (int i_7 = 0; i_7 < 8; i_7++) {
                                uint32_t _bf16x2_mul_6;
                                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_6) : "r"(pairs_3[i_7 * 2]), "r"(inverse_3));
                                uint16_t _e4m3x2_6;
                                asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_6) : "r"(_bf16x2_mul_6));
                                uint32_t _bf16x2_mul_7;
                                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_7) : "r"(pairs_3[i_7 * 2 + 1]), "r"(inverse_3));
                                uint16_t _e4m3x2_7;
                                asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_7) : "r"(_bf16x2_mul_7));
                                words_3[i_7] = (unsigned int)_e4m3x2_6 | (unsigned int)_e4m3x2_7 << 16;
                            }
                            scale_word_3 = scale_word_3 | scale_byte_3 << (unsigned int)(k_block_3 * 8);
                            #pragma unroll
                            for (int k_7 = 0; k_7 < 8; k_7++) {
                                int col_7 = k_block_3 * 32 + ((tid - 128) * 4 + k_7 * 4) % 32;
                                smem_v38[(row_0_3 * 128 + col_7) / 4] = words_3[k_7];
                            }
                        }
                        smem_v40[row_0_3 % 32 * 4 + row_0_3 / 32] = scale_word_3;
                    }
                    __syncthreads();
                    if (tid == 0) {
                        int col_tile_1 = col_block * 4 + 1;
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        tma_store_2d((&dy_q_store), col_tile_1 * 128, row_1, smem_v38_addr);
                        tma_store_3d((&dy_sc_store), 0, 0, row_tile * k_tiles + col_tile_1, smem_v40_addr);
                        tma_store_2d((&dy_t_store), row_1, col_tile_1 * 128, smem_v42_addr);
                        tma_store_3d((&dy_sc_t_store), 0, 0, col_tile_1 * macro_tiles + row_tile, smem_v44_addr);
                        asm volatile("cp.async.bulk.commit_group;");
                    }
                }
                if (chunk_cols / 128 > 2) {
                    if (tid == 0) {
                        asm volatile("cp.async.bulk.wait_group.read 1;");
                    }
                    __syncthreads();
                    if (tid < 128) {
                        int row_0_4 = tid;
                        row_0_4 = tid % 64 * 2 + tid / 64;
                        unsigned int scale_word_4 = 0;
                        #pragma unroll 1
                        for (int j_4 = 0; j_4 < 4; j_4++) {
                            int k_block_4 = (j_4 + tid / 8) % 4;
                            unsigned int pairs_4[16];
                            #pragma unroll
                            for (int k_8 = 0; k_8 < 16; k_8++) {
                                int col_8 = k_block_4 * 32 + (tid * 4 + k_8 * 2) % 32;
                                float x0_4 = 0.0f;
                                float x1_4 = 0.0f;
                                x0_4 = (float)smem_v35[col_8 * 512 + row_0_4];
                                x1_4 = (float)smem_v35[(col_8 + 1) * 512 + row_0_4];
                                x0_4 = x0_4 * dispatch_weights[col_8];
                                x1_4 = x1_4 * dispatch_weights[col_8 + 1];
                                __nv_bfloat162 _bf16x2_4 = __float22bfloat162_rn(make_float2(x0_4, x1_4));
                                pairs_4[k_8] = __as_u32(_bf16x2_4);
                            }
                            uint32_t _bf16x2_abs_8;
                            asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_8) : "r"(pairs_4[0]));
                            unsigned int amax_pair_4 = _bf16x2_abs_8;
                            #pragma unroll
                            for (int i_8 = 1; i_8 < 16; i_8++) {
                                uint32_t _bf16x2_abs_9;
                                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_9) : "r"(pairs_4[i_8]));
                                uint32_t _bf16x2_max_4;
                                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_4) : "r"(amax_pair_4), "r"(_bf16x2_abs_9));
                                amax_pair_4 = _bf16x2_max_4;
                            }
                            uint16_t _bf16_max_4;
                            asm("max.bf16 %0, %1, %2;" : "=h"(_bf16_max_4) : "h"((uint16_t)(amax_pair_4 & 65535)), "h"((uint16_t)(amax_pair_4 >> 16)));
                            float _cvt_f32_bf16_4;
                            asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_4) : "h"((uint16_t)(_bf16_max_4)));
                            float amax_4 = _cvt_f32_bf16_4;
                            float _fmax_4 = fmaxf(amax_4 * 0.002232142857f, 1e-12f);
                            float scale_4 = _fmax_4;
                            uint16_t _ue8m0x2_f32_4;
                            asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_ue8m0x2_f32_4) : "f"(scale_4), "f"(scale_4));
                            unsigned int scale_byte_4 = (unsigned int)_ue8m0x2_f32_4 & 255;
                            unsigned int inverse_lane_4 = 254 - scale_byte_4 << 7;
                            unsigned int inverse_4 = inverse_lane_4 | inverse_lane_4 << 16;
                            unsigned int words_4[8];
                            #pragma unroll
                            for (int i_9 = 0; i_9 < 8; i_9++) {
                                uint32_t _bf16x2_mul_8;
                                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_8) : "r"(pairs_4[i_9 * 2]), "r"(inverse_4));
                                uint16_t _e4m3x2_8;
                                asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_8) : "r"(_bf16x2_mul_8));
                                uint32_t _bf16x2_mul_9;
                                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_9) : "r"(pairs_4[i_9 * 2 + 1]), "r"(inverse_4));
                                uint16_t _e4m3x2_9;
                                asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_9) : "r"(_bf16x2_mul_9));
                                words_4[i_9] = (unsigned int)_e4m3x2_8 | (unsigned int)_e4m3x2_9 << 16;
                            }
                            scale_word_4 = scale_word_4 | scale_byte_4 << (unsigned int)(k_block_4 * 8);
                            #pragma unroll
                            for (int k_9 = 0; k_9 < 8; k_9++) {
                                int col_9 = k_block_4 * 32 + (tid * 4 + k_9 * 4) % 32;
                                smem_v41[(row_0_4 * 128 + col_9) / 4] = words_4[k_9];
                            }
                        }
                        smem_v43[row_0_4 % 32 * 4 + row_0_4 / 32] = scale_word_4;
                    } else {
                        int row_0_5 = tid - 128;
                        unsigned int scale_word_5 = 0;
                        #pragma unroll 1
                        for (int j_5 = 0; j_5 < 4; j_5++) {
                            int k_block_5 = (j_5 + (tid - 128) / 8) % 4;
                            unsigned int pairs_5[16];
                            #pragma unroll
                            for (int k_10 = 0; k_10 < 16; k_10++) {
                                int col_10 = k_block_5 * 32 + ((tid - 128) * 4 + k_10 * 2) % 32;
                                float x0_5 = 0.0f;
                                float x1_5 = 0.0f;
                                float2 _cvt_f32_2 = __bfloat1622float2(__as_bf16x2(dispatch_words[128 + (row_0_5 * 512 + col_10) / 2]));
                                __nv_bfloat162 _bf16x2_5 = __float22bfloat162_rn(make_float2(_cvt_f32_2.x * dispatch_weights[row_0_5], _cvt_f32_2.y * dispatch_weights[row_0_5]));
                                pairs_5[k_10] = __as_u32(_bf16x2_5);
                            }
                            uint32_t _bf16x2_abs_10;
                            asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_10) : "r"(pairs_5[0]));
                            unsigned int amax_pair_5 = _bf16x2_abs_10;
                            #pragma unroll
                            for (int i_10 = 1; i_10 < 16; i_10++) {
                                uint32_t _bf16x2_abs_11;
                                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_11) : "r"(pairs_5[i_10]));
                                uint32_t _bf16x2_max_5;
                                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_5) : "r"(amax_pair_5), "r"(_bf16x2_abs_11));
                                amax_pair_5 = _bf16x2_max_5;
                            }
                            uint16_t _bf16_max_5;
                            asm("max.bf16 %0, %1, %2;" : "=h"(_bf16_max_5) : "h"((uint16_t)(amax_pair_5 & 65535)), "h"((uint16_t)(amax_pair_5 >> 16)));
                            float _cvt_f32_bf16_5;
                            asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_5) : "h"((uint16_t)(_bf16_max_5)));
                            float amax_5 = _cvt_f32_bf16_5;
                            float _fmax_5 = fmaxf(amax_5 * 0.002232142857f, 1e-12f);
                            float scale_5 = _fmax_5;
                            uint16_t _ue8m0x2_f32_5;
                            asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_ue8m0x2_f32_5) : "f"(scale_5), "f"(scale_5));
                            unsigned int scale_byte_5 = (unsigned int)_ue8m0x2_f32_5 & 255;
                            unsigned int inverse_lane_5 = 254 - scale_byte_5 << 7;
                            unsigned int inverse_5 = inverse_lane_5 | inverse_lane_5 << 16;
                            unsigned int words_5[8];
                            #pragma unroll
                            for (int i_11 = 0; i_11 < 8; i_11++) {
                                uint32_t _bf16x2_mul_10;
                                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_10) : "r"(pairs_5[i_11 * 2]), "r"(inverse_5));
                                uint16_t _e4m3x2_10;
                                asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_10) : "r"(_bf16x2_mul_10));
                                uint32_t _bf16x2_mul_11;
                                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_11) : "r"(pairs_5[i_11 * 2 + 1]), "r"(inverse_5));
                                uint16_t _e4m3x2_11;
                                asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_11) : "r"(_bf16x2_mul_11));
                                words_5[i_11] = (unsigned int)_e4m3x2_10 | (unsigned int)_e4m3x2_11 << 16;
                            }
                            scale_word_5 = scale_word_5 | scale_byte_5 << (unsigned int)(k_block_5 * 8);
                            #pragma unroll
                            for (int k_11 = 0; k_11 < 8; k_11++) {
                                int col_11 = k_block_5 * 32 + ((tid - 128) * 4 + k_11 * 4) % 32;
                                smem_v37[(row_0_5 * 128 + col_11) / 4] = words_5[k_11];
                            }
                        }
                        smem_v39[row_0_5 % 32 * 4 + row_0_5 / 32] = scale_word_5;
                    }
                    __syncthreads();
                    if (tid == 0) {
                        int col_tile_2 = col_block * 4 + 2;
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        tma_store_2d((&dy_q_store), col_tile_2 * 128, row_1, smem_v37_addr);
                        tma_store_3d((&dy_sc_store), 0, 0, row_tile * k_tiles + col_tile_2, smem_v39_addr);
                        tma_store_2d((&dy_t_store), row_1, col_tile_2 * 128, smem_v41_addr);
                        tma_store_3d((&dy_sc_t_store), 0, 0, col_tile_2 * macro_tiles + row_tile, smem_v43_addr);
                        asm volatile("cp.async.bulk.commit_group;");
                    }
                }
                if (chunk_cols / 128 > 3) {
                    if (tid == 0) {
                        asm volatile("cp.async.bulk.wait_group.read 1;");
                    }
                    __syncthreads();
                    if (tid < 128) {
                        int row_0_6 = tid;
                        row_0_6 = tid % 64 * 2 + tid / 64;
                        unsigned int scale_word_6 = 0;
                        #pragma unroll 1
                        for (int j_6 = 0; j_6 < 4; j_6++) {
                            int k_block_6 = (j_6 + tid / 8) % 4;
                            unsigned int pairs_6[16];
                            #pragma unroll
                            for (int k_12 = 0; k_12 < 16; k_12++) {
                                int col_12 = k_block_6 * 32 + (tid * 4 + k_12 * 2) % 32;
                                float x0_6 = 0.0f;
                                float x1_6 = 0.0f;
                                x0_6 = (float)smem_v36[col_12 * 512 + row_0_6];
                                x1_6 = (float)smem_v36[(col_12 + 1) * 512 + row_0_6];
                                x0_6 = x0_6 * dispatch_weights[col_12];
                                x1_6 = x1_6 * dispatch_weights[col_12 + 1];
                                __nv_bfloat162 _bf16x2_6 = __float22bfloat162_rn(make_float2(x0_6, x1_6));
                                pairs_6[k_12] = __as_u32(_bf16x2_6);
                            }
                            uint32_t _bf16x2_abs_12;
                            asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_12) : "r"(pairs_6[0]));
                            unsigned int amax_pair_6 = _bf16x2_abs_12;
                            #pragma unroll
                            for (int i_12 = 1; i_12 < 16; i_12++) {
                                uint32_t _bf16x2_abs_13;
                                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_13) : "r"(pairs_6[i_12]));
                                uint32_t _bf16x2_max_6;
                                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_6) : "r"(amax_pair_6), "r"(_bf16x2_abs_13));
                                amax_pair_6 = _bf16x2_max_6;
                            }
                            uint16_t _bf16_max_6;
                            asm("max.bf16 %0, %1, %2;" : "=h"(_bf16_max_6) : "h"((uint16_t)(amax_pair_6 & 65535)), "h"((uint16_t)(amax_pair_6 >> 16)));
                            float _cvt_f32_bf16_6;
                            asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_6) : "h"((uint16_t)(_bf16_max_6)));
                            float amax_6 = _cvt_f32_bf16_6;
                            float _fmax_6 = fmaxf(amax_6 * 0.002232142857f, 1e-12f);
                            float scale_6 = _fmax_6;
                            uint16_t _ue8m0x2_f32_6;
                            asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_ue8m0x2_f32_6) : "f"(scale_6), "f"(scale_6));
                            unsigned int scale_byte_6 = (unsigned int)_ue8m0x2_f32_6 & 255;
                            unsigned int inverse_lane_6 = 254 - scale_byte_6 << 7;
                            unsigned int inverse_6 = inverse_lane_6 | inverse_lane_6 << 16;
                            unsigned int words_6[8];
                            #pragma unroll
                            for (int i_13 = 0; i_13 < 8; i_13++) {
                                uint32_t _bf16x2_mul_12;
                                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_12) : "r"(pairs_6[i_13 * 2]), "r"(inverse_6));
                                uint16_t _e4m3x2_12;
                                asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_12) : "r"(_bf16x2_mul_12));
                                uint32_t _bf16x2_mul_13;
                                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_13) : "r"(pairs_6[i_13 * 2 + 1]), "r"(inverse_6));
                                uint16_t _e4m3x2_13;
                                asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_13) : "r"(_bf16x2_mul_13));
                                words_6[i_13] = (unsigned int)_e4m3x2_12 | (unsigned int)_e4m3x2_13 << 16;
                            }
                            scale_word_6 = scale_word_6 | scale_byte_6 << (unsigned int)(k_block_6 * 8);
                            #pragma unroll
                            for (int k_13 = 0; k_13 < 8; k_13++) {
                                int col_13 = k_block_6 * 32 + (tid * 4 + k_13 * 4) % 32;
                                smem_v42[(row_0_6 * 128 + col_13) / 4] = words_6[k_13];
                            }
                        }
                        smem_v44[row_0_6 % 32 * 4 + row_0_6 / 32] = scale_word_6;
                    } else {
                        int row_0_7 = tid - 128;
                        unsigned int scale_word_7 = 0;
                        #pragma unroll 1
                        for (int j_7 = 0; j_7 < 4; j_7++) {
                            int k_block_7 = (j_7 + (tid - 128) / 8) % 4;
                            unsigned int pairs_7[16];
                            #pragma unroll
                            for (int k_14 = 0; k_14 < 16; k_14++) {
                                int col_14 = k_block_7 * 32 + ((tid - 128) * 4 + k_14 * 2) % 32;
                                float x0_7 = 0.0f;
                                float x1_7 = 0.0f;
                                float2 _cvt_f32_3 = __bfloat1622float2(__as_bf16x2(dispatch_words[192 + (row_0_7 * 512 + col_14) / 2]));
                                __nv_bfloat162 _bf16x2_7 = __float22bfloat162_rn(make_float2(_cvt_f32_3.x * dispatch_weights[row_0_7], _cvt_f32_3.y * dispatch_weights[row_0_7]));
                                pairs_7[k_14] = __as_u32(_bf16x2_7);
                            }
                            uint32_t _bf16x2_abs_14;
                            asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_14) : "r"(pairs_7[0]));
                            unsigned int amax_pair_7 = _bf16x2_abs_14;
                            #pragma unroll
                            for (int i_14 = 1; i_14 < 16; i_14++) {
                                uint32_t _bf16x2_abs_15;
                                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_15) : "r"(pairs_7[i_14]));
                                uint32_t _bf16x2_max_7;
                                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_7) : "r"(amax_pair_7), "r"(_bf16x2_abs_15));
                                amax_pair_7 = _bf16x2_max_7;
                            }
                            uint16_t _bf16_max_7;
                            asm("max.bf16 %0, %1, %2;" : "=h"(_bf16_max_7) : "h"((uint16_t)(amax_pair_7 & 65535)), "h"((uint16_t)(amax_pair_7 >> 16)));
                            float _cvt_f32_bf16_7;
                            asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_7) : "h"((uint16_t)(_bf16_max_7)));
                            float amax_7 = _cvt_f32_bf16_7;
                            float _fmax_7 = fmaxf(amax_7 * 0.002232142857f, 1e-12f);
                            float scale_7 = _fmax_7;
                            uint16_t _ue8m0x2_f32_7;
                            asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_ue8m0x2_f32_7) : "f"(scale_7), "f"(scale_7));
                            unsigned int scale_byte_7 = (unsigned int)_ue8m0x2_f32_7 & 255;
                            unsigned int inverse_lane_7 = 254 - scale_byte_7 << 7;
                            unsigned int inverse_7 = inverse_lane_7 | inverse_lane_7 << 16;
                            unsigned int words_7[8];
                            #pragma unroll
                            for (int i_15 = 0; i_15 < 8; i_15++) {
                                uint32_t _bf16x2_mul_14;
                                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_14) : "r"(pairs_7[i_15 * 2]), "r"(inverse_7));
                                uint16_t _e4m3x2_14;
                                asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_14) : "r"(_bf16x2_mul_14));
                                uint32_t _bf16x2_mul_15;
                                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_15) : "r"(pairs_7[i_15 * 2 + 1]), "r"(inverse_7));
                                uint16_t _e4m3x2_15;
                                asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_15) : "r"(_bf16x2_mul_15));
                                words_7[i_15] = (unsigned int)_e4m3x2_14 | (unsigned int)_e4m3x2_15 << 16;
                            }
                            scale_word_7 = scale_word_7 | scale_byte_7 << (unsigned int)(k_block_7 * 8);
                            #pragma unroll
                            for (int k_15 = 0; k_15 < 8; k_15++) {
                                int col_15 = k_block_7 * 32 + ((tid - 128) * 4 + k_15 * 4) % 32;
                                smem_v38[(row_0_7 * 128 + col_15) / 4] = words_7[k_15];
                            }
                        }
                        smem_v40[row_0_7 % 32 * 4 + row_0_7 / 32] = scale_word_7;
                    }
                    __syncthreads();
                    if (tid == 0) {
                        int col_tile_3 = col_block * 4 + 3;
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        tma_store_2d((&dy_q_store), col_tile_3 * 128, row_1, smem_v38_addr);
                        tma_store_3d((&dy_sc_store), 0, 0, row_tile * k_tiles + col_tile_3, smem_v40_addr);
                        tma_store_2d((&dy_t_store), row_1, col_tile_3 * 128, smem_v42_addr);
                        tma_store_3d((&dy_sc_t_store), 0, 0, col_tile_3 * macro_tiles + row_tile, smem_v44_addr);
                        asm volatile("cp.async.bulk.commit_group;");
                    }
                }
                if (tid == 0) {
                    asm volatile("cp.async.bulk.wait_group 0;");
                    asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(dy_ready)) + ((macro_offset + row_1) / mini_size))), "r"(static_cast<unsigned int>(1)) : "memory");
                }
                __syncthreads();
            }
            dispatch_bits = phase_bits;
        }
        #pragma unroll 1
        for (int macro = 0; macro < macros; macro++) {
            int _min_7 = ((tokens - macro * macro_size) < (macro_size) ? (tokens - macro * macro_size) : (macro_size));
            int rows_0 = _min_7;
            int combine_tasks = (rows_0 / 16 * ((hidden + 1023) / 1024) + 6) / 7;
            #pragma unroll 1
            for (int task_1 = cluster * 2 + cta_rank_0; task_1 < combine_tasks; task_1 += comm_sms) {
                unsigned int phase_bits_1 = combine_bits;
                int col_blocks_1 = (hidden + 1023) / 1024;
                int first_tile = task_1 * 7;
                int macro_offset_1 = macro * macro_size;
                int _min_8 = ((macro_size) < (tokens - macro_offset_1) ? (macro_size) : (tokens - macro_offset_1));
                int macro_tokens_1 = _min_8;
                int _min_9 = ((7) < (macro_tokens_1 / 16 * col_blocks_1 - first_tile) ? (7) : (macro_tokens_1 / 16 * col_blocks_1 - first_tile));
                int valid_tiles = _min_9;
                if (valid_tiles > 0) {
                    int first_row = first_tile / col_blocks_1 * 16 + tid;
                    int first_col = first_tile % col_blocks_1;
                    int rows_1[7];
                    int columns[7];
                    int peers[7];
                    int tokens_2[7];
                    unsigned int counts_3[7];
                    int row_2 = first_row;
                    int column = first_col;
                    #pragma unroll
                    for (int stage = 0; stage < 7; stage++) {
                        rows_1[stage] = row_2;
                        columns[stage] = column;
                        peers[stage] = -1;
                        tokens_2[stage] = -1;
                        if (valid_tiles > stage && tid < 16) {
                            peers[stage] = schedule_rank[macro_offset_1 + row_2];
                            tokens_2[stage] = schedule_token[macro_offset_1 + row_2];
                        }
                        counts_3[stage] = 0;
                        if (valid_tiles > stage) {
                            if (stage == 0 || column == 0) {
                                uint32_t _cta_count_1 = __syncthreads_count(peers[stage] >= 0);
                                counts_3[stage] = _cta_count_1;
                            } else {
                                counts_3[stage] = counts_3[stage - 1];
                            }
                        }
                        column = column + 1;
                        if (column == col_blocks_1) {
                            column = 0;
                            row_2 = row_2 + 16;
                        }
                    }
                    if (tid == 0) {
                        int first_mini = (macro_offset_1 + first_row) / mini_size;
                        int last_mini = (macro_offset_1 + (first_tile + valid_tiles - 1) / col_blocks_1 * 16) / mini_size;
                        #pragma unroll 1
                        for (int mini = first_mini; mini < last_mini + 1; mini++) {
                            int _min_10 = ((mini_size) < (tokens - mini * mini_size) ? (mini_size) : (tokens - mini * mini_size));
                            int mini_rows_1 = _min_10;
                            int required_1 = (mini_rows_1 + 255) / 256 * (hidden / 256) * 2;
                            int32_t _relaxed_ld_2;
                            asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_2) : "l"(dx_ready + mini) : "memory");
                            int value_2 = _relaxed_ld_2;
                            while (value_2 < required_1) {
                                asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                int32_t _relaxed_ld_3;
                                asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_3) : "l"(dx_ready + mini) : "memory");
                                value_2 = _relaxed_ld_3;
                            }
                            asm volatile("fence.acquire.gpu;" ::: "memory");
                        }
                        #pragma unroll
                        for (int stage_1 = 0; stage_1 < 7; stage_1++) {
                            if (valid_tiles > stage_1) {
                                int _min_11 = ((1024) < (hidden - columns[stage_1] * 1024) ? (1024) : (hidden - columns[stage_1] * 1024));
                                unsigned int chunk_bytes_1 = (unsigned int)(_min_11 * 2);
                                mbarrier_arrive_expect_tx(combine_arrived_addr + (stage_1) * 8, counts_3[stage_1] * chunk_bytes_1);
                            }
                        }
                    }
                    __syncthreads();
                    #pragma unroll
                    for (int stage_2 = 0; stage_2 < 7; stage_2++) {
                        if (peers[stage_2] >= 0) {
                            int _min_12 = ((1024) < (hidden - columns[stage_2] * 1024) ? (1024) : (hidden - columns[stage_2] * 1024));
                            int chunk_cols_1 = _min_12;
                            cp_async_bulk_gmem2smem(combine_smem_addr + (unsigned int)((stage_2 * 16 + tid) * 1024 * 2), reinterpret_cast<const void*>(reinterpret_cast<const uint8_t*>(dx_routed_ptr) + ((unsigned long long)((unsigned long long)rows_1[stage_2] * (unsigned long long)hidden + (unsigned long long)(columns[stage_2] * 1024)) * (unsigned long long)2)), chunk_cols_1 * 2, combine_arrived_addr + (stage_2) * 8);
                        }
                    }
                    #pragma unroll
                    for (int stage_3 = 0; stage_3 < 7; stage_3++) {
                        if (valid_tiles > stage_3) {
                            mbarrier_wait(combine_arrived_addr + (stage_3) * 8, phase_bits_1 >> (unsigned int)stage_3 & 1);
                            phase_bits_1 = phase_bits_1 ^ (unsigned int)(1 << stage_3);
                            if (peers[stage_3] >= 0) {
                                int _min_13 = ((1024) < (hidden - columns[stage_3] * 1024) ? (1024) : (hidden - columns[stage_3] * 1024));
                                unsigned int chunk_bytes_2 = (unsigned int)(_min_13 * 2);
                                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                {
                                    void* _cpbulk_dst_0 = reinterpret_cast<void*>(reinterpret_cast<__nv_bfloat16*>(dx_peers[peers[stage_3]]) + ((unsigned long long)tokens_2[stage_3] * (unsigned long long)hidden + (unsigned long long)(columns[stage_3] * 1024)));
                                    asm volatile(
                                        "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                        :: "l"(_cpbulk_dst_0), "r"(combine_smem_addr + (unsigned int)((stage_3 * 16 + tid) * 2048)), "r"((uint32_t)(chunk_bytes_2))
                                        : "memory");
                                }
                                asm volatile("cp.async.bulk.commit_group;");
                                if (columns[stage_3] == 0) {
                                    float gradient = 0.0f;
                                    #pragma unroll 1
                                    for (int col_16 = 0; col_16 < intermediate / 128; col_16++) {
                                        gradient = gradient + partials[rows_1[stage_3] * (intermediate / 128) + col_16];
                                    }
                                    reinterpret_cast<float*>(dweight_peers[peers[stage_3]])[tokens_2[stage_3]] = gradient;
                                }
                            }
                        }
                    }
                    asm volatile("cp.async.bulk.wait_group.read 0;");
                    __syncthreads();
                    if (tid == 0 && macros > 1) {
                        bool enabled_value = 1;
                        if (enabled_value != 0) {
                            asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(buffers_done)) + (macro))), "r"(static_cast<unsigned int>(1)) : "memory");
                        }
                    }
                }
                combine_bits = phase_bits_1;
            }
            if (macros > macro + 1) {
                int minis_0 = (rows_0 + mini_size - 1) / mini_size;
                int required_2 = 2 * (minis_0 * mini_bwd + wgrad_tasks) + combine_tasks;
                if (tid == 0) {
                    bool enabled_value_1 = 1;
                    if (enabled_value_1 != 0) {
                        int32_t _relaxed_ld_4;
                        asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_4) : "l"(buffers_done + macro) : "memory");
                        int value_3 = _relaxed_ld_4;
                        while (value_3 < required_2) {
                            asm volatile("nanosleep.u32 %0;" :: "r"(16));
                            int32_t _relaxed_ld_5;
                            asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_5) : "l"(buffers_done + macro) : "memory");
                            value_3 = _relaxed_ld_5;
                        }
                        asm volatile("fence.acquire.gpu;" ::: "memory");
                    }
                }
                __syncthreads();
                int offset_1 = (macro + 1) * macro_size;
                int _min_14 = ((macro_size) < (tokens - offset_1) ? (macro_size) : (tokens - offset_1));
                int rows_2 = _min_14;
                #pragma unroll 1
                for (int row_3 = (cluster * 2 + cta_rank_0) * 256 + tid; row_3 < rows_2; row_3 += comm_sms * 256) {
                    int peer_2 = schedule_rank[offset_1 + row_3];
                    int token_1 = schedule_token[offset_1 + row_3];
                    float value_4 = 0.0f;
                    if (peer_2 >= 0) {
                        value_4 = reinterpret_cast<float*>(weight_peers[peer_2])[(int)token_1];
                    }
                    weights[row_3] = value_4;
                }
                __syncthreads();
                if (tid == 0) {
                    asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(weight_ready)) + (macro + 1))), "r"(static_cast<unsigned int>(1)) : "memory");
                    int32_t _relaxed_ld_6;
                    asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_6) : "l"(weight_ready + (macro + 1)) : "memory");
                    int value_5 = _relaxed_ld_6;
                    while (value_5 < comm_sms) {
                        asm volatile("nanosleep.u32 %0;" :: "r"(16));
                        int32_t _relaxed_ld_7;
                        asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_7) : "l"(weight_ready + (macro + 1)) : "memory");
                        value_5 = _relaxed_ld_7;
                    }
                    asm volatile("fence.acquire.gpu;" ::: "memory");
                }
                __syncthreads();
                int _min_15 = ((tokens - (macro + 1) * macro_size) < (macro_size) ? (tokens - (macro + 1) * macro_size) : (macro_size));
                int next_rows = _min_15;
                #pragma unroll 1
                for (int task_2 = cluster * 2 + cta_rank_0; task_2 < next_rows / 128 * ((hidden + 511) / 512); task_2 += comm_sms) {
                    unsigned int phase_bits_2 = dispatch_bits;
                    int col_blocks_2 = (hidden + 511) / 512;
                    int macro_offset_2 = (macro + 1) * macro_size;
                    int _min_16 = ((macro_size) < (tokens - macro_offset_2) ? (macro_size) : (tokens - macro_offset_2));
                    int macro_tokens_2 = _min_16;
                    if (task_2 < macro_tokens_2 / 128 * col_blocks_2) {
                        int row_4 = task_2 / col_blocks_2 * 128;
                        int col_block_1 = task_2 % col_blocks_2;
                        int _min_17 = ((512) < (hidden - col_block_1 * 512) ? (512) : (hidden - col_block_1 * 512));
                        int chunk_cols_2 = _min_17;
                        unsigned int chunk_bytes_3 = (unsigned int)(chunk_cols_2 * 2);
                        int peer_3 = -1;
                        int peer_token_1 = -1;
                        if (tid < 128) {
                            peer_3 = schedule_rank[macro_offset_2 + row_4 + tid];
                            peer_token_1 = schedule_token[macro_offset_2 + row_4 + tid];
                        }
                        uint32_t _cta_count_2 = __syncthreads_count(peer_3 >= 0);
                        if (tid == 0) {
                            int previous_offset_1 = -1 * macro_size;
                            int _min_18 = ((macro_size) < (tokens - previous_offset_1) ? (macro_size) : (tokens - previous_offset_1));
                            int previous_tokens_1 = _min_18;
                            if (row_4 < previous_tokens_1) {
                                int previous_mini_1 = (previous_offset_1 + row_4) / mini_size;
                                int _min_19 = ((mini_size) < (tokens - previous_mini_1 * mini_size) ? (mini_size) : (tokens - previous_mini_1 * mini_size));
                                int mini_rows_2 = _min_19;
                                int required_0 = (mini_rows_2 + 255) / 256 * (hidden / 256) * 2;
                            }
                            mbarrier_arrive_expect_tx(dispatch_arrived_addr, _cta_count_2 * chunk_bytes_3);
                        }
                        __syncthreads();
                        if (tid < 128) {
                            dispatch_weights[tid] = ((peer_3 >= 0) ? weights[row_4 + tid] : 0.0f);
                        }
                        if (peer_3 >= 0) {
                            cp_async_bulk_gmem2smem(smem_v33_addr + (unsigned int)(tid * 512 * 2), reinterpret_cast<const void*>(reinterpret_cast<const uint8_t*>(reinterpret_cast<__nv_bfloat16*>(dy_peers[peer_3])) + ((unsigned long long)((unsigned long long)(peer_token_1 / topk) * (unsigned long long)hidden + (unsigned long long)(col_block_1 * 512)) * (unsigned long long)2)), chunk_cols_2 * 2, dispatch_arrived_addr);
                        } else if (tid < 128) {
                            #pragma unroll
                            for (int vec_1 = 0; vec_1 < 64; vec_1++) {
                                asm volatile("st.v4.u32 [%0], {%1, %2, %3, %4};" :: "l"((uint64_t)(reinterpret_cast<uint8_t*>(dispatch_words) + (tid * 1024 + vec_1 * 16))), "r"(static_cast<uint32_t>((unsigned int)0)), "r"(static_cast<uint32_t>((unsigned int)0)), "r"(static_cast<uint32_t>((unsigned int)0)), "r"(static_cast<uint32_t>((unsigned int)0)) : "memory");
                            }
                        }
                        mbarrier_wait(dispatch_arrived_addr, phase_bits_2 & 1);
                        phase_bits_2 = phase_bits_2 ^ 1;
                        int row_tile_1 = row_4 / 128;
                        int k_tiles_1 = hidden / 128;
                        int macro_tiles_1 = macro_size / 128;
                        if (chunk_cols_2 / 128 > 0) {
                            if (tid == 0) {
                                asm volatile("cp.async.bulk.wait_group.read 1;");
                            }
                            __syncthreads();
                            if (tid < 128) {
                                int row_0_8 = tid;
                                row_0_8 = tid % 64 * 2 + tid / 64;
                                unsigned int scale_word_8 = 0;
                                #pragma unroll 1
                                for (int j_8 = 0; j_8 < 4; j_8++) {
                                    int k_block_8 = (j_8 + tid / 8) % 4;
                                    unsigned int pairs_8[16];
                                    #pragma unroll
                                    for (int k_16 = 0; k_16 < 16; k_16++) {
                                        int col_17 = k_block_8 * 32 + (tid * 4 + k_16 * 2) % 32;
                                        float x0_8 = 0.0f;
                                        float x1_8 = 0.0f;
                                        x0_8 = (float)smem_v33[col_17 * 512 + row_0_8];
                                        x1_8 = (float)smem_v33[(col_17 + 1) * 512 + row_0_8];
                                        x0_8 = x0_8 * dispatch_weights[col_17];
                                        x1_8 = x1_8 * dispatch_weights[col_17 + 1];
                                        __nv_bfloat162 _bf16x2_8 = __float22bfloat162_rn(make_float2(x0_8, x1_8));
                                        pairs_8[k_16] = __as_u32(_bf16x2_8);
                                    }
                                    uint32_t _bf16x2_abs_16;
                                    asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_16) : "r"(pairs_8[0]));
                                    unsigned int amax_pair_8 = _bf16x2_abs_16;
                                    #pragma unroll
                                    for (int i_16 = 1; i_16 < 16; i_16++) {
                                        uint32_t _bf16x2_abs_17;
                                        asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_17) : "r"(pairs_8[i_16]));
                                        uint32_t _bf16x2_max_8;
                                        asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_8) : "r"(amax_pair_8), "r"(_bf16x2_abs_17));
                                        amax_pair_8 = _bf16x2_max_8;
                                    }
                                    uint16_t _bf16_max_8;
                                    asm("max.bf16 %0, %1, %2;" : "=h"(_bf16_max_8) : "h"((uint16_t)(amax_pair_8 & 65535)), "h"((uint16_t)(amax_pair_8 >> 16)));
                                    float _cvt_f32_bf16_8;
                                    asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_8) : "h"((uint16_t)(_bf16_max_8)));
                                    float amax_8 = _cvt_f32_bf16_8;
                                    float _fmax_8 = fmaxf(amax_8 * 0.002232142857f, 1e-12f);
                                    float scale_8 = _fmax_8;
                                    uint16_t _ue8m0x2_f32_8;
                                    asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_ue8m0x2_f32_8) : "f"(scale_8), "f"(scale_8));
                                    unsigned int scale_byte_8 = (unsigned int)_ue8m0x2_f32_8 & 255;
                                    unsigned int inverse_lane_8 = 254 - scale_byte_8 << 7;
                                    unsigned int inverse_8 = inverse_lane_8 | inverse_lane_8 << 16;
                                    unsigned int words_8[8];
                                    #pragma unroll
                                    for (int i_17 = 0; i_17 < 8; i_17++) {
                                        uint32_t _bf16x2_mul_16;
                                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_16) : "r"(pairs_8[i_17 * 2]), "r"(inverse_8));
                                        uint16_t _e4m3x2_16;
                                        asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_16) : "r"(_bf16x2_mul_16));
                                        uint32_t _bf16x2_mul_17;
                                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_17) : "r"(pairs_8[i_17 * 2 + 1]), "r"(inverse_8));
                                        uint16_t _e4m3x2_17;
                                        asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_17) : "r"(_bf16x2_mul_17));
                                        words_8[i_17] = (unsigned int)_e4m3x2_16 | (unsigned int)_e4m3x2_17 << 16;
                                    }
                                    scale_word_8 = scale_word_8 | scale_byte_8 << (unsigned int)(k_block_8 * 8);
                                    #pragma unroll
                                    for (int k_17 = 0; k_17 < 8; k_17++) {
                                        int col_18 = k_block_8 * 32 + (tid * 4 + k_17 * 4) % 32;
                                        smem_v41[(row_0_8 * 128 + col_18) / 4] = words_8[k_17];
                                    }
                                }
                                smem_v43[row_0_8 % 32 * 4 + row_0_8 / 32] = scale_word_8;
                            } else {
                                int row_0_9 = tid - 128;
                                unsigned int scale_word_9 = 0;
                                #pragma unroll 1
                                for (int j_9 = 0; j_9 < 4; j_9++) {
                                    int k_block_9 = (j_9 + (tid - 128) / 8) % 4;
                                    unsigned int pairs_9[16];
                                    #pragma unroll
                                    for (int k_18 = 0; k_18 < 16; k_18++) {
                                        int col_19 = k_block_9 * 32 + ((tid - 128) * 4 + k_18 * 2) % 32;
                                        float x0_9 = 0.0f;
                                        float x1_9 = 0.0f;
                                        float2 _cvt_f32_4 = __bfloat1622float2(__as_bf16x2(dispatch_words[(row_0_9 * 512 + col_19) / 2]));
                                        __nv_bfloat162 _bf16x2_9 = __float22bfloat162_rn(make_float2(_cvt_f32_4.x * dispatch_weights[row_0_9], _cvt_f32_4.y * dispatch_weights[row_0_9]));
                                        pairs_9[k_18] = __as_u32(_bf16x2_9);
                                    }
                                    uint32_t _bf16x2_abs_18;
                                    asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_18) : "r"(pairs_9[0]));
                                    unsigned int amax_pair_9 = _bf16x2_abs_18;
                                    #pragma unroll
                                    for (int i_18 = 1; i_18 < 16; i_18++) {
                                        uint32_t _bf16x2_abs_19;
                                        asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_19) : "r"(pairs_9[i_18]));
                                        uint32_t _bf16x2_max_9;
                                        asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_9) : "r"(amax_pair_9), "r"(_bf16x2_abs_19));
                                        amax_pair_9 = _bf16x2_max_9;
                                    }
                                    uint16_t _bf16_max_9;
                                    asm("max.bf16 %0, %1, %2;" : "=h"(_bf16_max_9) : "h"((uint16_t)(amax_pair_9 & 65535)), "h"((uint16_t)(amax_pair_9 >> 16)));
                                    float _cvt_f32_bf16_9;
                                    asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_9) : "h"((uint16_t)(_bf16_max_9)));
                                    float amax_9 = _cvt_f32_bf16_9;
                                    float _fmax_9 = fmaxf(amax_9 * 0.002232142857f, 1e-12f);
                                    float scale_9 = _fmax_9;
                                    uint16_t _ue8m0x2_f32_9;
                                    asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_ue8m0x2_f32_9) : "f"(scale_9), "f"(scale_9));
                                    unsigned int scale_byte_9 = (unsigned int)_ue8m0x2_f32_9 & 255;
                                    unsigned int inverse_lane_9 = 254 - scale_byte_9 << 7;
                                    unsigned int inverse_9 = inverse_lane_9 | inverse_lane_9 << 16;
                                    unsigned int words_9[8];
                                    #pragma unroll
                                    for (int i_19 = 0; i_19 < 8; i_19++) {
                                        uint32_t _bf16x2_mul_18;
                                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_18) : "r"(pairs_9[i_19 * 2]), "r"(inverse_9));
                                        uint16_t _e4m3x2_18;
                                        asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_18) : "r"(_bf16x2_mul_18));
                                        uint32_t _bf16x2_mul_19;
                                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_19) : "r"(pairs_9[i_19 * 2 + 1]), "r"(inverse_9));
                                        uint16_t _e4m3x2_19;
                                        asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_19) : "r"(_bf16x2_mul_19));
                                        words_9[i_19] = (unsigned int)_e4m3x2_18 | (unsigned int)_e4m3x2_19 << 16;
                                    }
                                    scale_word_9 = scale_word_9 | scale_byte_9 << (unsigned int)(k_block_9 * 8);
                                    #pragma unroll
                                    for (int k_19 = 0; k_19 < 8; k_19++) {
                                        int col_20 = k_block_9 * 32 + ((tid - 128) * 4 + k_19 * 4) % 32;
                                        smem_v37[(row_0_9 * 128 + col_20) / 4] = words_9[k_19];
                                    }
                                }
                                smem_v39[row_0_9 % 32 * 4 + row_0_9 / 32] = scale_word_9;
                            }
                            __syncthreads();
                            if (tid == 0) {
                                int col_tile_4 = col_block_1 * 4;
                                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                tma_store_2d((&dy_q_store), col_tile_4 * 128, row_4, smem_v37_addr);
                                tma_store_3d((&dy_sc_store), 0, 0, row_tile_1 * k_tiles_1 + col_tile_4, smem_v39_addr);
                                tma_store_2d((&dy_t_store), row_4, col_tile_4 * 128, smem_v41_addr);
                                tma_store_3d((&dy_sc_t_store), 0, 0, col_tile_4 * macro_tiles_1 + row_tile_1, smem_v43_addr);
                                asm volatile("cp.async.bulk.commit_group;");
                            }
                        }
                        if (chunk_cols_2 / 128 > 1) {
                            if (tid == 0) {
                                asm volatile("cp.async.bulk.wait_group.read 1;");
                            }
                            __syncthreads();
                            if (tid < 128) {
                                int row_0_10 = tid;
                                row_0_10 = tid % 64 * 2 + tid / 64;
                                unsigned int scale_word_10 = 0;
                                #pragma unroll 1
                                for (int j_10 = 0; j_10 < 4; j_10++) {
                                    int k_block_10 = (j_10 + tid / 8) % 4;
                                    unsigned int pairs_10[16];
                                    #pragma unroll
                                    for (int k_20 = 0; k_20 < 16; k_20++) {
                                        int col_21 = k_block_10 * 32 + (tid * 4 + k_20 * 2) % 32;
                                        float x0_10 = 0.0f;
                                        float x1_10 = 0.0f;
                                        x0_10 = (float)smem_v34[col_21 * 512 + row_0_10];
                                        x1_10 = (float)smem_v34[(col_21 + 1) * 512 + row_0_10];
                                        x0_10 = x0_10 * dispatch_weights[col_21];
                                        x1_10 = x1_10 * dispatch_weights[col_21 + 1];
                                        __nv_bfloat162 _bf16x2_10 = __float22bfloat162_rn(make_float2(x0_10, x1_10));
                                        pairs_10[k_20] = __as_u32(_bf16x2_10);
                                    }
                                    uint32_t _bf16x2_abs_20;
                                    asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_20) : "r"(pairs_10[0]));
                                    unsigned int amax_pair_10 = _bf16x2_abs_20;
                                    #pragma unroll
                                    for (int i_20 = 1; i_20 < 16; i_20++) {
                                        uint32_t _bf16x2_abs_21;
                                        asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_21) : "r"(pairs_10[i_20]));
                                        uint32_t _bf16x2_max_10;
                                        asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_10) : "r"(amax_pair_10), "r"(_bf16x2_abs_21));
                                        amax_pair_10 = _bf16x2_max_10;
                                    }
                                    uint16_t _bf16_max_10;
                                    asm("max.bf16 %0, %1, %2;" : "=h"(_bf16_max_10) : "h"((uint16_t)(amax_pair_10 & 65535)), "h"((uint16_t)(amax_pair_10 >> 16)));
                                    float _cvt_f32_bf16_10;
                                    asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_10) : "h"((uint16_t)(_bf16_max_10)));
                                    float amax_10 = _cvt_f32_bf16_10;
                                    float _fmax_10 = fmaxf(amax_10 * 0.002232142857f, 1e-12f);
                                    float scale_10 = _fmax_10;
                                    uint16_t _ue8m0x2_f32_10;
                                    asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_ue8m0x2_f32_10) : "f"(scale_10), "f"(scale_10));
                                    unsigned int scale_byte_10 = (unsigned int)_ue8m0x2_f32_10 & 255;
                                    unsigned int inverse_lane_10 = 254 - scale_byte_10 << 7;
                                    unsigned int inverse_10 = inverse_lane_10 | inverse_lane_10 << 16;
                                    unsigned int words_10[8];
                                    #pragma unroll
                                    for (int i_21 = 0; i_21 < 8; i_21++) {
                                        uint32_t _bf16x2_mul_20;
                                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_20) : "r"(pairs_10[i_21 * 2]), "r"(inverse_10));
                                        uint16_t _e4m3x2_20;
                                        asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_20) : "r"(_bf16x2_mul_20));
                                        uint32_t _bf16x2_mul_21;
                                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_21) : "r"(pairs_10[i_21 * 2 + 1]), "r"(inverse_10));
                                        uint16_t _e4m3x2_21;
                                        asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_21) : "r"(_bf16x2_mul_21));
                                        words_10[i_21] = (unsigned int)_e4m3x2_20 | (unsigned int)_e4m3x2_21 << 16;
                                    }
                                    scale_word_10 = scale_word_10 | scale_byte_10 << (unsigned int)(k_block_10 * 8);
                                    #pragma unroll
                                    for (int k_21 = 0; k_21 < 8; k_21++) {
                                        int col_22 = k_block_10 * 32 + (tid * 4 + k_21 * 4) % 32;
                                        smem_v42[(row_0_10 * 128 + col_22) / 4] = words_10[k_21];
                                    }
                                }
                                smem_v44[row_0_10 % 32 * 4 + row_0_10 / 32] = scale_word_10;
                            } else {
                                int row_0_11 = tid - 128;
                                unsigned int scale_word_11 = 0;
                                #pragma unroll 1
                                for (int j_11 = 0; j_11 < 4; j_11++) {
                                    int k_block_11 = (j_11 + (tid - 128) / 8) % 4;
                                    unsigned int pairs_11[16];
                                    #pragma unroll
                                    for (int k_22 = 0; k_22 < 16; k_22++) {
                                        int col_23 = k_block_11 * 32 + ((tid - 128) * 4 + k_22 * 2) % 32;
                                        float x0_11 = 0.0f;
                                        float x1_11 = 0.0f;
                                        float2 _cvt_f32_5 = __bfloat1622float2(__as_bf16x2(dispatch_words[64 + (row_0_11 * 512 + col_23) / 2]));
                                        __nv_bfloat162 _bf16x2_11 = __float22bfloat162_rn(make_float2(_cvt_f32_5.x * dispatch_weights[row_0_11], _cvt_f32_5.y * dispatch_weights[row_0_11]));
                                        pairs_11[k_22] = __as_u32(_bf16x2_11);
                                    }
                                    uint32_t _bf16x2_abs_22;
                                    asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_22) : "r"(pairs_11[0]));
                                    unsigned int amax_pair_11 = _bf16x2_abs_22;
                                    #pragma unroll
                                    for (int i_22 = 1; i_22 < 16; i_22++) {
                                        uint32_t _bf16x2_abs_23;
                                        asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_23) : "r"(pairs_11[i_22]));
                                        uint32_t _bf16x2_max_11;
                                        asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_11) : "r"(amax_pair_11), "r"(_bf16x2_abs_23));
                                        amax_pair_11 = _bf16x2_max_11;
                                    }
                                    uint16_t _bf16_max_11;
                                    asm("max.bf16 %0, %1, %2;" : "=h"(_bf16_max_11) : "h"((uint16_t)(amax_pair_11 & 65535)), "h"((uint16_t)(amax_pair_11 >> 16)));
                                    float _cvt_f32_bf16_11;
                                    asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_11) : "h"((uint16_t)(_bf16_max_11)));
                                    float amax_11 = _cvt_f32_bf16_11;
                                    float _fmax_11 = fmaxf(amax_11 * 0.002232142857f, 1e-12f);
                                    float scale_11 = _fmax_11;
                                    uint16_t _ue8m0x2_f32_11;
                                    asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_ue8m0x2_f32_11) : "f"(scale_11), "f"(scale_11));
                                    unsigned int scale_byte_11 = (unsigned int)_ue8m0x2_f32_11 & 255;
                                    unsigned int inverse_lane_11 = 254 - scale_byte_11 << 7;
                                    unsigned int inverse_11 = inverse_lane_11 | inverse_lane_11 << 16;
                                    unsigned int words_11[8];
                                    #pragma unroll
                                    for (int i_23 = 0; i_23 < 8; i_23++) {
                                        uint32_t _bf16x2_mul_22;
                                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_22) : "r"(pairs_11[i_23 * 2]), "r"(inverse_11));
                                        uint16_t _e4m3x2_22;
                                        asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_22) : "r"(_bf16x2_mul_22));
                                        uint32_t _bf16x2_mul_23;
                                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_23) : "r"(pairs_11[i_23 * 2 + 1]), "r"(inverse_11));
                                        uint16_t _e4m3x2_23;
                                        asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_23) : "r"(_bf16x2_mul_23));
                                        words_11[i_23] = (unsigned int)_e4m3x2_22 | (unsigned int)_e4m3x2_23 << 16;
                                    }
                                    scale_word_11 = scale_word_11 | scale_byte_11 << (unsigned int)(k_block_11 * 8);
                                    #pragma unroll
                                    for (int k_23 = 0; k_23 < 8; k_23++) {
                                        int col_24 = k_block_11 * 32 + ((tid - 128) * 4 + k_23 * 4) % 32;
                                        smem_v38[(row_0_11 * 128 + col_24) / 4] = words_11[k_23];
                                    }
                                }
                                smem_v40[row_0_11 % 32 * 4 + row_0_11 / 32] = scale_word_11;
                            }
                            __syncthreads();
                            if (tid == 0) {
                                int col_tile_5 = col_block_1 * 4 + 1;
                                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                tma_store_2d((&dy_q_store), col_tile_5 * 128, row_4, smem_v38_addr);
                                tma_store_3d((&dy_sc_store), 0, 0, row_tile_1 * k_tiles_1 + col_tile_5, smem_v40_addr);
                                tma_store_2d((&dy_t_store), row_4, col_tile_5 * 128, smem_v42_addr);
                                tma_store_3d((&dy_sc_t_store), 0, 0, col_tile_5 * macro_tiles_1 + row_tile_1, smem_v44_addr);
                                asm volatile("cp.async.bulk.commit_group;");
                            }
                        }
                        if (chunk_cols_2 / 128 > 2) {
                            if (tid == 0) {
                                asm volatile("cp.async.bulk.wait_group.read 1;");
                            }
                            __syncthreads();
                            if (tid < 128) {
                                int row_0_12 = tid;
                                row_0_12 = tid % 64 * 2 + tid / 64;
                                unsigned int scale_word_12 = 0;
                                #pragma unroll 1
                                for (int j_12 = 0; j_12 < 4; j_12++) {
                                    int k_block_12 = (j_12 + tid / 8) % 4;
                                    unsigned int pairs_12[16];
                                    #pragma unroll
                                    for (int k_24 = 0; k_24 < 16; k_24++) {
                                        int col_25 = k_block_12 * 32 + (tid * 4 + k_24 * 2) % 32;
                                        float x0_12 = 0.0f;
                                        float x1_12 = 0.0f;
                                        x0_12 = (float)smem_v35[col_25 * 512 + row_0_12];
                                        x1_12 = (float)smem_v35[(col_25 + 1) * 512 + row_0_12];
                                        x0_12 = x0_12 * dispatch_weights[col_25];
                                        x1_12 = x1_12 * dispatch_weights[col_25 + 1];
                                        __nv_bfloat162 _bf16x2_12 = __float22bfloat162_rn(make_float2(x0_12, x1_12));
                                        pairs_12[k_24] = __as_u32(_bf16x2_12);
                                    }
                                    uint32_t _bf16x2_abs_24;
                                    asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_24) : "r"(pairs_12[0]));
                                    unsigned int amax_pair_12 = _bf16x2_abs_24;
                                    #pragma unroll
                                    for (int i_24 = 1; i_24 < 16; i_24++) {
                                        uint32_t _bf16x2_abs_25;
                                        asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_25) : "r"(pairs_12[i_24]));
                                        uint32_t _bf16x2_max_12;
                                        asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_12) : "r"(amax_pair_12), "r"(_bf16x2_abs_25));
                                        amax_pair_12 = _bf16x2_max_12;
                                    }
                                    uint16_t _bf16_max_12;
                                    asm("max.bf16 %0, %1, %2;" : "=h"(_bf16_max_12) : "h"((uint16_t)(amax_pair_12 & 65535)), "h"((uint16_t)(amax_pair_12 >> 16)));
                                    float _cvt_f32_bf16_12;
                                    asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_12) : "h"((uint16_t)(_bf16_max_12)));
                                    float amax_12 = _cvt_f32_bf16_12;
                                    float _fmax_12 = fmaxf(amax_12 * 0.002232142857f, 1e-12f);
                                    float scale_12 = _fmax_12;
                                    uint16_t _ue8m0x2_f32_12;
                                    asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_ue8m0x2_f32_12) : "f"(scale_12), "f"(scale_12));
                                    unsigned int scale_byte_12 = (unsigned int)_ue8m0x2_f32_12 & 255;
                                    unsigned int inverse_lane_12 = 254 - scale_byte_12 << 7;
                                    unsigned int inverse_12 = inverse_lane_12 | inverse_lane_12 << 16;
                                    unsigned int words_12[8];
                                    #pragma unroll
                                    for (int i_25 = 0; i_25 < 8; i_25++) {
                                        uint32_t _bf16x2_mul_24;
                                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_24) : "r"(pairs_12[i_25 * 2]), "r"(inverse_12));
                                        uint16_t _e4m3x2_24;
                                        asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_24) : "r"(_bf16x2_mul_24));
                                        uint32_t _bf16x2_mul_25;
                                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_25) : "r"(pairs_12[i_25 * 2 + 1]), "r"(inverse_12));
                                        uint16_t _e4m3x2_25;
                                        asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_25) : "r"(_bf16x2_mul_25));
                                        words_12[i_25] = (unsigned int)_e4m3x2_24 | (unsigned int)_e4m3x2_25 << 16;
                                    }
                                    scale_word_12 = scale_word_12 | scale_byte_12 << (unsigned int)(k_block_12 * 8);
                                    #pragma unroll
                                    for (int k_25 = 0; k_25 < 8; k_25++) {
                                        int col_26 = k_block_12 * 32 + (tid * 4 + k_25 * 4) % 32;
                                        smem_v41[(row_0_12 * 128 + col_26) / 4] = words_12[k_25];
                                    }
                                }
                                smem_v43[row_0_12 % 32 * 4 + row_0_12 / 32] = scale_word_12;
                            } else {
                                int row_0_13 = tid - 128;
                                unsigned int scale_word_13 = 0;
                                #pragma unroll 1
                                for (int j_13 = 0; j_13 < 4; j_13++) {
                                    int k_block_13 = (j_13 + (tid - 128) / 8) % 4;
                                    unsigned int pairs_13[16];
                                    #pragma unroll
                                    for (int k_26 = 0; k_26 < 16; k_26++) {
                                        int col_27 = k_block_13 * 32 + ((tid - 128) * 4 + k_26 * 2) % 32;
                                        float x0_13 = 0.0f;
                                        float x1_13 = 0.0f;
                                        float2 _cvt_f32_6 = __bfloat1622float2(__as_bf16x2(dispatch_words[128 + (row_0_13 * 512 + col_27) / 2]));
                                        __nv_bfloat162 _bf16x2_13 = __float22bfloat162_rn(make_float2(_cvt_f32_6.x * dispatch_weights[row_0_13], _cvt_f32_6.y * dispatch_weights[row_0_13]));
                                        pairs_13[k_26] = __as_u32(_bf16x2_13);
                                    }
                                    uint32_t _bf16x2_abs_26;
                                    asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_26) : "r"(pairs_13[0]));
                                    unsigned int amax_pair_13 = _bf16x2_abs_26;
                                    #pragma unroll
                                    for (int i_26 = 1; i_26 < 16; i_26++) {
                                        uint32_t _bf16x2_abs_27;
                                        asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_27) : "r"(pairs_13[i_26]));
                                        uint32_t _bf16x2_max_13;
                                        asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_13) : "r"(amax_pair_13), "r"(_bf16x2_abs_27));
                                        amax_pair_13 = _bf16x2_max_13;
                                    }
                                    uint16_t _bf16_max_13;
                                    asm("max.bf16 %0, %1, %2;" : "=h"(_bf16_max_13) : "h"((uint16_t)(amax_pair_13 & 65535)), "h"((uint16_t)(amax_pair_13 >> 16)));
                                    float _cvt_f32_bf16_13;
                                    asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_13) : "h"((uint16_t)(_bf16_max_13)));
                                    float amax_13 = _cvt_f32_bf16_13;
                                    float _fmax_13 = fmaxf(amax_13 * 0.002232142857f, 1e-12f);
                                    float scale_13 = _fmax_13;
                                    uint16_t _ue8m0x2_f32_13;
                                    asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_ue8m0x2_f32_13) : "f"(scale_13), "f"(scale_13));
                                    unsigned int scale_byte_13 = (unsigned int)_ue8m0x2_f32_13 & 255;
                                    unsigned int inverse_lane_13 = 254 - scale_byte_13 << 7;
                                    unsigned int inverse_13 = inverse_lane_13 | inverse_lane_13 << 16;
                                    unsigned int words_13[8];
                                    #pragma unroll
                                    for (int i_27 = 0; i_27 < 8; i_27++) {
                                        uint32_t _bf16x2_mul_26;
                                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_26) : "r"(pairs_13[i_27 * 2]), "r"(inverse_13));
                                        uint16_t _e4m3x2_26;
                                        asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_26) : "r"(_bf16x2_mul_26));
                                        uint32_t _bf16x2_mul_27;
                                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_27) : "r"(pairs_13[i_27 * 2 + 1]), "r"(inverse_13));
                                        uint16_t _e4m3x2_27;
                                        asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_27) : "r"(_bf16x2_mul_27));
                                        words_13[i_27] = (unsigned int)_e4m3x2_26 | (unsigned int)_e4m3x2_27 << 16;
                                    }
                                    scale_word_13 = scale_word_13 | scale_byte_13 << (unsigned int)(k_block_13 * 8);
                                    #pragma unroll
                                    for (int k_27 = 0; k_27 < 8; k_27++) {
                                        int col_28 = k_block_13 * 32 + ((tid - 128) * 4 + k_27 * 4) % 32;
                                        smem_v37[(row_0_13 * 128 + col_28) / 4] = words_13[k_27];
                                    }
                                }
                                smem_v39[row_0_13 % 32 * 4 + row_0_13 / 32] = scale_word_13;
                            }
                            __syncthreads();
                            if (tid == 0) {
                                int col_tile_6 = col_block_1 * 4 + 2;
                                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                tma_store_2d((&dy_q_store), col_tile_6 * 128, row_4, smem_v37_addr);
                                tma_store_3d((&dy_sc_store), 0, 0, row_tile_1 * k_tiles_1 + col_tile_6, smem_v39_addr);
                                tma_store_2d((&dy_t_store), row_4, col_tile_6 * 128, smem_v41_addr);
                                tma_store_3d((&dy_sc_t_store), 0, 0, col_tile_6 * macro_tiles_1 + row_tile_1, smem_v43_addr);
                                asm volatile("cp.async.bulk.commit_group;");
                            }
                        }
                        if (chunk_cols_2 / 128 > 3) {
                            if (tid == 0) {
                                asm volatile("cp.async.bulk.wait_group.read 1;");
                            }
                            __syncthreads();
                            if (tid < 128) {
                                int row_0_14 = tid;
                                row_0_14 = tid % 64 * 2 + tid / 64;
                                unsigned int scale_word_14 = 0;
                                #pragma unroll 1
                                for (int j_14 = 0; j_14 < 4; j_14++) {
                                    int k_block_14 = (j_14 + tid / 8) % 4;
                                    unsigned int pairs_14[16];
                                    #pragma unroll
                                    for (int k_28 = 0; k_28 < 16; k_28++) {
                                        int col_29 = k_block_14 * 32 + (tid * 4 + k_28 * 2) % 32;
                                        float x0_14 = 0.0f;
                                        float x1_14 = 0.0f;
                                        x0_14 = (float)smem_v36[col_29 * 512 + row_0_14];
                                        x1_14 = (float)smem_v36[(col_29 + 1) * 512 + row_0_14];
                                        x0_14 = x0_14 * dispatch_weights[col_29];
                                        x1_14 = x1_14 * dispatch_weights[col_29 + 1];
                                        __nv_bfloat162 _bf16x2_14 = __float22bfloat162_rn(make_float2(x0_14, x1_14));
                                        pairs_14[k_28] = __as_u32(_bf16x2_14);
                                    }
                                    uint32_t _bf16x2_abs_28;
                                    asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_28) : "r"(pairs_14[0]));
                                    unsigned int amax_pair_14 = _bf16x2_abs_28;
                                    #pragma unroll
                                    for (int i_28 = 1; i_28 < 16; i_28++) {
                                        uint32_t _bf16x2_abs_29;
                                        asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_29) : "r"(pairs_14[i_28]));
                                        uint32_t _bf16x2_max_14;
                                        asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_14) : "r"(amax_pair_14), "r"(_bf16x2_abs_29));
                                        amax_pair_14 = _bf16x2_max_14;
                                    }
                                    uint16_t _bf16_max_14;
                                    asm("max.bf16 %0, %1, %2;" : "=h"(_bf16_max_14) : "h"((uint16_t)(amax_pair_14 & 65535)), "h"((uint16_t)(amax_pair_14 >> 16)));
                                    float _cvt_f32_bf16_14;
                                    asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_14) : "h"((uint16_t)(_bf16_max_14)));
                                    float amax_14 = _cvt_f32_bf16_14;
                                    float _fmax_14 = fmaxf(amax_14 * 0.002232142857f, 1e-12f);
                                    float scale_14 = _fmax_14;
                                    uint16_t _ue8m0x2_f32_14;
                                    asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_ue8m0x2_f32_14) : "f"(scale_14), "f"(scale_14));
                                    unsigned int scale_byte_14 = (unsigned int)_ue8m0x2_f32_14 & 255;
                                    unsigned int inverse_lane_14 = 254 - scale_byte_14 << 7;
                                    unsigned int inverse_14 = inverse_lane_14 | inverse_lane_14 << 16;
                                    unsigned int words_14[8];
                                    #pragma unroll
                                    for (int i_29 = 0; i_29 < 8; i_29++) {
                                        uint32_t _bf16x2_mul_28;
                                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_28) : "r"(pairs_14[i_29 * 2]), "r"(inverse_14));
                                        uint16_t _e4m3x2_28;
                                        asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_28) : "r"(_bf16x2_mul_28));
                                        uint32_t _bf16x2_mul_29;
                                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_29) : "r"(pairs_14[i_29 * 2 + 1]), "r"(inverse_14));
                                        uint16_t _e4m3x2_29;
                                        asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_29) : "r"(_bf16x2_mul_29));
                                        words_14[i_29] = (unsigned int)_e4m3x2_28 | (unsigned int)_e4m3x2_29 << 16;
                                    }
                                    scale_word_14 = scale_word_14 | scale_byte_14 << (unsigned int)(k_block_14 * 8);
                                    #pragma unroll
                                    for (int k_29 = 0; k_29 < 8; k_29++) {
                                        int col_30 = k_block_14 * 32 + (tid * 4 + k_29 * 4) % 32;
                                        smem_v42[(row_0_14 * 128 + col_30) / 4] = words_14[k_29];
                                    }
                                }
                                smem_v44[row_0_14 % 32 * 4 + row_0_14 / 32] = scale_word_14;
                            } else {
                                int row_0_15 = tid - 128;
                                unsigned int scale_word_15 = 0;
                                #pragma unroll 1
                                for (int j_15 = 0; j_15 < 4; j_15++) {
                                    int k_block_15 = (j_15 + (tid - 128) / 8) % 4;
                                    unsigned int pairs_15[16];
                                    #pragma unroll
                                    for (int k_30 = 0; k_30 < 16; k_30++) {
                                        int col_31 = k_block_15 * 32 + ((tid - 128) * 4 + k_30 * 2) % 32;
                                        float x0_15 = 0.0f;
                                        float x1_15 = 0.0f;
                                        float2 _cvt_f32_7 = __bfloat1622float2(__as_bf16x2(dispatch_words[192 + (row_0_15 * 512 + col_31) / 2]));
                                        __nv_bfloat162 _bf16x2_15 = __float22bfloat162_rn(make_float2(_cvt_f32_7.x * dispatch_weights[row_0_15], _cvt_f32_7.y * dispatch_weights[row_0_15]));
                                        pairs_15[k_30] = __as_u32(_bf16x2_15);
                                    }
                                    uint32_t _bf16x2_abs_30;
                                    asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_30) : "r"(pairs_15[0]));
                                    unsigned int amax_pair_15 = _bf16x2_abs_30;
                                    #pragma unroll
                                    for (int i_30 = 1; i_30 < 16; i_30++) {
                                        uint32_t _bf16x2_abs_31;
                                        asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_31) : "r"(pairs_15[i_30]));
                                        uint32_t _bf16x2_max_15;
                                        asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_15) : "r"(amax_pair_15), "r"(_bf16x2_abs_31));
                                        amax_pair_15 = _bf16x2_max_15;
                                    }
                                    uint16_t _bf16_max_15;
                                    asm("max.bf16 %0, %1, %2;" : "=h"(_bf16_max_15) : "h"((uint16_t)(amax_pair_15 & 65535)), "h"((uint16_t)(amax_pair_15 >> 16)));
                                    float _cvt_f32_bf16_15;
                                    asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_15) : "h"((uint16_t)(_bf16_max_15)));
                                    float amax_15 = _cvt_f32_bf16_15;
                                    float _fmax_15 = fmaxf(amax_15 * 0.002232142857f, 1e-12f);
                                    float scale_15 = _fmax_15;
                                    uint16_t _ue8m0x2_f32_15;
                                    asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_ue8m0x2_f32_15) : "f"(scale_15), "f"(scale_15));
                                    unsigned int scale_byte_15 = (unsigned int)_ue8m0x2_f32_15 & 255;
                                    unsigned int inverse_lane_15 = 254 - scale_byte_15 << 7;
                                    unsigned int inverse_15 = inverse_lane_15 | inverse_lane_15 << 16;
                                    unsigned int words_15[8];
                                    #pragma unroll
                                    for (int i_31 = 0; i_31 < 8; i_31++) {
                                        uint32_t _bf16x2_mul_30;
                                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_30) : "r"(pairs_15[i_31 * 2]), "r"(inverse_15));
                                        uint16_t _e4m3x2_30;
                                        asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_30) : "r"(_bf16x2_mul_30));
                                        uint32_t _bf16x2_mul_31;
                                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_31) : "r"(pairs_15[i_31 * 2 + 1]), "r"(inverse_15));
                                        uint16_t _e4m3x2_31;
                                        asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_31) : "r"(_bf16x2_mul_31));
                                        words_15[i_31] = (unsigned int)_e4m3x2_30 | (unsigned int)_e4m3x2_31 << 16;
                                    }
                                    scale_word_15 = scale_word_15 | scale_byte_15 << (unsigned int)(k_block_15 * 8);
                                    #pragma unroll
                                    for (int k_31 = 0; k_31 < 8; k_31++) {
                                        int col_32 = k_block_15 * 32 + ((tid - 128) * 4 + k_31 * 4) % 32;
                                        smem_v38[(row_0_15 * 128 + col_32) / 4] = words_15[k_31];
                                    }
                                }
                                smem_v40[row_0_15 % 32 * 4 + row_0_15 / 32] = scale_word_15;
                            }
                            __syncthreads();
                            if (tid == 0) {
                                int col_tile_7 = col_block_1 * 4 + 3;
                                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                tma_store_2d((&dy_q_store), col_tile_7 * 128, row_4, smem_v38_addr);
                                tma_store_3d((&dy_sc_store), 0, 0, row_tile_1 * k_tiles_1 + col_tile_7, smem_v40_addr);
                                tma_store_2d((&dy_t_store), row_4, col_tile_7 * 128, smem_v42_addr);
                                tma_store_3d((&dy_sc_t_store), 0, 0, col_tile_7 * macro_tiles_1 + row_tile_1, smem_v44_addr);
                                asm volatile("cp.async.bulk.commit_group;");
                            }
                        }
                        if (tid == 0) {
                            asm volatile("cp.async.bulk.wait_group 0;");
                            asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(dy_ready)) + ((macro_offset_2 + row_4) / mini_size))), "r"(static_cast<unsigned int>(1)) : "memory");
                        }
                        __syncthreads();
                    }
                    dispatch_bits = phase_bits_2;
                    unsigned int phase_bits_0 = dispatch_bits;
                    int col_blocks_1_1 = (hidden + 511) / 512;
                    int macro_offset_2_1 = (macro + 1) * macro_size;
                    int _min_20 = ((macro_size) < (tokens - macro_offset_2_1) ? (macro_size) : (tokens - macro_offset_2_1));
                    int macro_tokens_3 = _min_20;
                    if (task_2 < macro_tokens_3 / 128 * col_blocks_1_1) {
                        int row_5 = task_2 / col_blocks_1_1 * 128;
                        int col_block_2 = task_2 % col_blocks_1_1;
                        int _min_21 = ((512) < (hidden - col_block_2 * 512) ? (512) : (hidden - col_block_2 * 512));
                        int chunk_cols_3 = _min_21;
                        unsigned int chunk_bytes_4 = (unsigned int)(chunk_cols_3 * 2);
                        int peer_4 = -1;
                        int peer_token_2 = -1;
                        if (tid < 128) {
                            peer_4 = schedule_rank[macro_offset_2_1 + row_5 + tid];
                            peer_token_2 = schedule_token[macro_offset_2_1 + row_5 + tid];
                        }
                        uint32_t _cta_count_3 = __syncthreads_count(peer_4 >= 0);
                        if (tid == 0) {
                            int previous_offset_2 = -1 * macro_size;
                            int _min_22 = ((macro_size) < (tokens - previous_offset_2) ? (macro_size) : (tokens - previous_offset_2));
                            int previous_tokens_2 = _min_22;
                            if (row_5 < previous_tokens_2) {
                                int previous_mini_2 = (previous_offset_2 + row_5) / mini_size;
                                int _min_23 = ((mini_size) < (tokens - previous_mini_2 * mini_size) ? (mini_size) : (tokens - previous_mini_2 * mini_size));
                                int mini_rows_3 = _min_23;
                                int required_0_1 = (mini_rows_3 + 255) / 256 * (hidden / 256) * 2;
                            }
                            mbarrier_arrive_expect_tx(dispatch_arrived_addr, _cta_count_3 * chunk_bytes_4);
                        }
                        __syncthreads();
                        if (peer_4 >= 0) {
                            cp_async_bulk_gmem2smem(smem_v33_addr + (unsigned int)(tid * 512 * 2), reinterpret_cast<const void*>(reinterpret_cast<const uint8_t*>(reinterpret_cast<__nv_bfloat16*>(x_peers[peer_4])) + ((unsigned long long)((unsigned long long)(peer_token_2 / topk) * (unsigned long long)hidden + (unsigned long long)(col_block_2 * 512)) * (unsigned long long)2)), chunk_cols_3 * 2, dispatch_arrived_addr);
                        } else if (tid < 128) {
                            #pragma unroll
                            for (int vec_2 = 0; vec_2 < 64; vec_2++) {
                                asm volatile("st.v4.u32 [%0], {%1, %2, %3, %4};" :: "l"((uint64_t)(reinterpret_cast<uint8_t*>(dispatch_words) + (tid * 1024 + vec_2 * 16))), "r"(static_cast<uint32_t>((unsigned int)0)), "r"(static_cast<uint32_t>((unsigned int)0)), "r"(static_cast<uint32_t>((unsigned int)0)), "r"(static_cast<uint32_t>((unsigned int)0)) : "memory");
                            }
                        }
                        mbarrier_wait(dispatch_arrived_addr, phase_bits_0 & 1);
                        phase_bits_0 = phase_bits_0 ^ 1;
                        int row_tile_2 = row_5 / 128;
                        int k_tiles_2 = hidden / 128;
                        int macro_tiles_2 = macro_size / 128;
                        if (chunk_cols_3 / 128 > 0) {
                            if (tid == 0) {
                                asm volatile("cp.async.bulk.wait_group.read 1;");
                            }
                            __syncthreads();
                            if (tid < 128) {
                                int row_0_16 = tid;
                                row_0_16 = tid % 64 * 2 + tid / 64;
                                unsigned int scale_word_16 = 0;
                                #pragma unroll 1
                                for (int j_16 = 0; j_16 < 4; j_16++) {
                                    int k_block_16 = (j_16 + tid / 8) % 4;
                                    unsigned int pairs_16[16];
                                    #pragma unroll
                                    for (int k_32 = 0; k_32 < 16; k_32++) {
                                        int col_33 = k_block_16 * 32 + (tid * 4 + k_32 * 2) % 32;
                                        float x0_16 = 0.0f;
                                        float x1_16 = 0.0f;
                                        x0_16 = (float)smem_v33[col_33 * 512 + row_0_16];
                                        x1_16 = (float)smem_v33[(col_33 + 1) * 512 + row_0_16];
                                        __nv_bfloat162 _bf16x2_16 = __float22bfloat162_rn(make_float2(x0_16, x1_16));
                                        pairs_16[k_32] = __as_u32(_bf16x2_16);
                                    }
                                    uint32_t _bf16x2_abs_32;
                                    asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_32) : "r"(pairs_16[0]));
                                    unsigned int amax_pair_16 = _bf16x2_abs_32;
                                    #pragma unroll
                                    for (int i_32 = 1; i_32 < 16; i_32++) {
                                        uint32_t _bf16x2_abs_33;
                                        asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_33) : "r"(pairs_16[i_32]));
                                        uint32_t _bf16x2_max_16;
                                        asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_16) : "r"(amax_pair_16), "r"(_bf16x2_abs_33));
                                        amax_pair_16 = _bf16x2_max_16;
                                    }
                                    uint16_t _bf16_max_16;
                                    asm("max.bf16 %0, %1, %2;" : "=h"(_bf16_max_16) : "h"((uint16_t)(amax_pair_16 & 65535)), "h"((uint16_t)(amax_pair_16 >> 16)));
                                    float _cvt_f32_bf16_16;
                                    asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_16) : "h"((uint16_t)(_bf16_max_16)));
                                    float amax_16 = _cvt_f32_bf16_16;
                                    float _fmax_16 = fmaxf(amax_16 * 0.002232142857f, 1e-12f);
                                    float scale_16 = _fmax_16;
                                    uint16_t _ue8m0x2_f32_16;
                                    asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_ue8m0x2_f32_16) : "f"(scale_16), "f"(scale_16));
                                    unsigned int scale_byte_16 = (unsigned int)_ue8m0x2_f32_16 & 255;
                                    unsigned int inverse_lane_16 = 254 - scale_byte_16 << 7;
                                    unsigned int inverse_16 = inverse_lane_16 | inverse_lane_16 << 16;
                                    unsigned int words_16[8];
                                    #pragma unroll
                                    for (int i_33 = 0; i_33 < 8; i_33++) {
                                        uint32_t _bf16x2_mul_32;
                                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_32) : "r"(pairs_16[i_33 * 2]), "r"(inverse_16));
                                        uint16_t _e4m3x2_32;
                                        asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_32) : "r"(_bf16x2_mul_32));
                                        uint32_t _bf16x2_mul_33;
                                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_33) : "r"(pairs_16[i_33 * 2 + 1]), "r"(inverse_16));
                                        uint16_t _e4m3x2_33;
                                        asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_33) : "r"(_bf16x2_mul_33));
                                        words_16[i_33] = (unsigned int)_e4m3x2_32 | (unsigned int)_e4m3x2_33 << 16;
                                    }
                                    scale_word_16 = scale_word_16 | scale_byte_16 << (unsigned int)(k_block_16 * 8);
                                    #pragma unroll
                                    for (int k_33 = 0; k_33 < 8; k_33++) {
                                        int col_34 = k_block_16 * 32 + (tid * 4 + k_33 * 4) % 32;
                                        smem_v41[(row_0_16 * 128 + col_34) / 4] = words_16[k_33];
                                    }
                                }
                                smem_v43[row_0_16 % 32 * 4 + row_0_16 / 32] = scale_word_16;
                            } else {
                                int row_0_17 = tid - 128;
                                unsigned int scale_word_17 = 0;
                                #pragma unroll 1
                                for (int j_17 = 0; j_17 < 4; j_17++) {
                                    int k_block_17 = (j_17 + (tid - 128) / 8) % 4;
                                    unsigned int pairs_17[16];
                                    #pragma unroll
                                    for (int k_34 = 0; k_34 < 16; k_34++) {
                                        int col_35 = k_block_17 * 32 + ((tid - 128) * 4 + k_34 * 2) % 32;
                                        float x0_17 = 0.0f;
                                        float x1_17 = 0.0f;
                                        pairs_17[k_34] = dispatch_words[(row_0_17 * 512 + col_35) / 2];
                                    }
                                    uint32_t _bf16x2_abs_34;
                                    asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_34) : "r"(pairs_17[0]));
                                    unsigned int amax_pair_17 = _bf16x2_abs_34;
                                    #pragma unroll
                                    for (int i_34 = 1; i_34 < 16; i_34++) {
                                        uint32_t _bf16x2_abs_35;
                                        asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_35) : "r"(pairs_17[i_34]));
                                        uint32_t _bf16x2_max_17;
                                        asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_17) : "r"(amax_pair_17), "r"(_bf16x2_abs_35));
                                        amax_pair_17 = _bf16x2_max_17;
                                    }
                                    uint16_t _bf16_max_17;
                                    asm("max.bf16 %0, %1, %2;" : "=h"(_bf16_max_17) : "h"((uint16_t)(amax_pair_17 & 65535)), "h"((uint16_t)(amax_pair_17 >> 16)));
                                    float _cvt_f32_bf16_17;
                                    asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_17) : "h"((uint16_t)(_bf16_max_17)));
                                    float amax_17 = _cvt_f32_bf16_17;
                                    float _fmax_17 = fmaxf(amax_17 * 0.002232142857f, 1e-12f);
                                    float scale_17 = _fmax_17;
                                    uint16_t _ue8m0x2_f32_17;
                                    asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_ue8m0x2_f32_17) : "f"(scale_17), "f"(scale_17));
                                    unsigned int scale_byte_17 = (unsigned int)_ue8m0x2_f32_17 & 255;
                                    unsigned int inverse_lane_17 = 254 - scale_byte_17 << 7;
                                    unsigned int inverse_17 = inverse_lane_17 | inverse_lane_17 << 16;
                                    unsigned int words_17[8];
                                    #pragma unroll
                                    for (int i_35 = 0; i_35 < 8; i_35++) {
                                        uint32_t _bf16x2_mul_34;
                                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_34) : "r"(pairs_17[i_35 * 2]), "r"(inverse_17));
                                        uint16_t _e4m3x2_34;
                                        asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_34) : "r"(_bf16x2_mul_34));
                                        uint32_t _bf16x2_mul_35;
                                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_35) : "r"(pairs_17[i_35 * 2 + 1]), "r"(inverse_17));
                                        uint16_t _e4m3x2_35;
                                        asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_35) : "r"(_bf16x2_mul_35));
                                        words_17[i_35] = (unsigned int)_e4m3x2_34 | (unsigned int)_e4m3x2_35 << 16;
                                    }
                                    scale_word_17 = scale_word_17 | scale_byte_17 << (unsigned int)(k_block_17 * 8);
                                    #pragma unroll
                                    for (int k_35 = 0; k_35 < 8; k_35++) {
                                        int col_36 = k_block_17 * 32 + ((tid - 128) * 4 + k_35 * 4) % 32;
                                        smem_v37[(row_0_17 * 128 + col_36) / 4] = words_17[k_35];
                                    }
                                }
                                smem_v39[row_0_17 % 32 * 4 + row_0_17 / 32] = scale_word_17;
                            }
                            __syncthreads();
                            if (tid == 0) {
                                int col_tile_8 = col_block_2 * 4;
                                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                tma_store_2d((&x_q_store), col_tile_8 * 128, row_5, smem_v37_addr);
                                tma_store_3d((&x_sc_store), 0, 0, row_tile_2 * k_tiles_2 + col_tile_8, smem_v39_addr);
                                tma_store_2d((&x_t_store), row_5, col_tile_8 * 128, smem_v41_addr);
                                tma_store_3d((&x_sc_t_store), 0, 0, col_tile_8 * macro_tiles_2 + row_tile_2, smem_v43_addr);
                                asm volatile("cp.async.bulk.commit_group;");
                            }
                        }
                        if (chunk_cols_3 / 128 > 1) {
                            if (tid == 0) {
                                asm volatile("cp.async.bulk.wait_group.read 1;");
                            }
                            __syncthreads();
                            if (tid < 128) {
                                int row_0_18 = tid;
                                row_0_18 = tid % 64 * 2 + tid / 64;
                                unsigned int scale_word_18 = 0;
                                #pragma unroll 1
                                for (int j_18 = 0; j_18 < 4; j_18++) {
                                    int k_block_18 = (j_18 + tid / 8) % 4;
                                    unsigned int pairs_18[16];
                                    #pragma unroll
                                    for (int k_36 = 0; k_36 < 16; k_36++) {
                                        int col_37 = k_block_18 * 32 + (tid * 4 + k_36 * 2) % 32;
                                        float x0_18 = 0.0f;
                                        float x1_18 = 0.0f;
                                        x0_18 = (float)smem_v34[col_37 * 512 + row_0_18];
                                        x1_18 = (float)smem_v34[(col_37 + 1) * 512 + row_0_18];
                                        __nv_bfloat162 _bf16x2_17 = __float22bfloat162_rn(make_float2(x0_18, x1_18));
                                        pairs_18[k_36] = __as_u32(_bf16x2_17);
                                    }
                                    uint32_t _bf16x2_abs_36;
                                    asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_36) : "r"(pairs_18[0]));
                                    unsigned int amax_pair_18 = _bf16x2_abs_36;
                                    #pragma unroll
                                    for (int i_36 = 1; i_36 < 16; i_36++) {
                                        uint32_t _bf16x2_abs_37;
                                        asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_37) : "r"(pairs_18[i_36]));
                                        uint32_t _bf16x2_max_18;
                                        asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_18) : "r"(amax_pair_18), "r"(_bf16x2_abs_37));
                                        amax_pair_18 = _bf16x2_max_18;
                                    }
                                    uint16_t _bf16_max_18;
                                    asm("max.bf16 %0, %1, %2;" : "=h"(_bf16_max_18) : "h"((uint16_t)(amax_pair_18 & 65535)), "h"((uint16_t)(amax_pair_18 >> 16)));
                                    float _cvt_f32_bf16_18;
                                    asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_18) : "h"((uint16_t)(_bf16_max_18)));
                                    float amax_18 = _cvt_f32_bf16_18;
                                    float _fmax_18 = fmaxf(amax_18 * 0.002232142857f, 1e-12f);
                                    float scale_18 = _fmax_18;
                                    uint16_t _ue8m0x2_f32_18;
                                    asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_ue8m0x2_f32_18) : "f"(scale_18), "f"(scale_18));
                                    unsigned int scale_byte_18 = (unsigned int)_ue8m0x2_f32_18 & 255;
                                    unsigned int inverse_lane_18 = 254 - scale_byte_18 << 7;
                                    unsigned int inverse_18 = inverse_lane_18 | inverse_lane_18 << 16;
                                    unsigned int words_18[8];
                                    #pragma unroll
                                    for (int i_37 = 0; i_37 < 8; i_37++) {
                                        uint32_t _bf16x2_mul_36;
                                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_36) : "r"(pairs_18[i_37 * 2]), "r"(inverse_18));
                                        uint16_t _e4m3x2_36;
                                        asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_36) : "r"(_bf16x2_mul_36));
                                        uint32_t _bf16x2_mul_37;
                                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_37) : "r"(pairs_18[i_37 * 2 + 1]), "r"(inverse_18));
                                        uint16_t _e4m3x2_37;
                                        asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_37) : "r"(_bf16x2_mul_37));
                                        words_18[i_37] = (unsigned int)_e4m3x2_36 | (unsigned int)_e4m3x2_37 << 16;
                                    }
                                    scale_word_18 = scale_word_18 | scale_byte_18 << (unsigned int)(k_block_18 * 8);
                                    #pragma unroll
                                    for (int k_37 = 0; k_37 < 8; k_37++) {
                                        int col_38 = k_block_18 * 32 + (tid * 4 + k_37 * 4) % 32;
                                        smem_v42[(row_0_18 * 128 + col_38) / 4] = words_18[k_37];
                                    }
                                }
                                smem_v44[row_0_18 % 32 * 4 + row_0_18 / 32] = scale_word_18;
                            } else {
                                int row_0_19 = tid - 128;
                                unsigned int scale_word_19 = 0;
                                #pragma unroll 1
                                for (int j_19 = 0; j_19 < 4; j_19++) {
                                    int k_block_19 = (j_19 + (tid - 128) / 8) % 4;
                                    unsigned int pairs_19[16];
                                    #pragma unroll
                                    for (int k_38 = 0; k_38 < 16; k_38++) {
                                        int col_39 = k_block_19 * 32 + ((tid - 128) * 4 + k_38 * 2) % 32;
                                        float x0_19 = 0.0f;
                                        float x1_19 = 0.0f;
                                        pairs_19[k_38] = dispatch_words[64 + (row_0_19 * 512 + col_39) / 2];
                                    }
                                    uint32_t _bf16x2_abs_38;
                                    asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_38) : "r"(pairs_19[0]));
                                    unsigned int amax_pair_19 = _bf16x2_abs_38;
                                    #pragma unroll
                                    for (int i_38 = 1; i_38 < 16; i_38++) {
                                        uint32_t _bf16x2_abs_39;
                                        asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_39) : "r"(pairs_19[i_38]));
                                        uint32_t _bf16x2_max_19;
                                        asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_19) : "r"(amax_pair_19), "r"(_bf16x2_abs_39));
                                        amax_pair_19 = _bf16x2_max_19;
                                    }
                                    uint16_t _bf16_max_19;
                                    asm("max.bf16 %0, %1, %2;" : "=h"(_bf16_max_19) : "h"((uint16_t)(amax_pair_19 & 65535)), "h"((uint16_t)(amax_pair_19 >> 16)));
                                    float _cvt_f32_bf16_19;
                                    asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_19) : "h"((uint16_t)(_bf16_max_19)));
                                    float amax_19 = _cvt_f32_bf16_19;
                                    float _fmax_19 = fmaxf(amax_19 * 0.002232142857f, 1e-12f);
                                    float scale_19 = _fmax_19;
                                    uint16_t _ue8m0x2_f32_19;
                                    asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_ue8m0x2_f32_19) : "f"(scale_19), "f"(scale_19));
                                    unsigned int scale_byte_19 = (unsigned int)_ue8m0x2_f32_19 & 255;
                                    unsigned int inverse_lane_19 = 254 - scale_byte_19 << 7;
                                    unsigned int inverse_19 = inverse_lane_19 | inverse_lane_19 << 16;
                                    unsigned int words_19[8];
                                    #pragma unroll
                                    for (int i_39 = 0; i_39 < 8; i_39++) {
                                        uint32_t _bf16x2_mul_38;
                                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_38) : "r"(pairs_19[i_39 * 2]), "r"(inverse_19));
                                        uint16_t _e4m3x2_38;
                                        asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_38) : "r"(_bf16x2_mul_38));
                                        uint32_t _bf16x2_mul_39;
                                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_39) : "r"(pairs_19[i_39 * 2 + 1]), "r"(inverse_19));
                                        uint16_t _e4m3x2_39;
                                        asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_39) : "r"(_bf16x2_mul_39));
                                        words_19[i_39] = (unsigned int)_e4m3x2_38 | (unsigned int)_e4m3x2_39 << 16;
                                    }
                                    scale_word_19 = scale_word_19 | scale_byte_19 << (unsigned int)(k_block_19 * 8);
                                    #pragma unroll
                                    for (int k_39 = 0; k_39 < 8; k_39++) {
                                        int col_40 = k_block_19 * 32 + ((tid - 128) * 4 + k_39 * 4) % 32;
                                        smem_v38[(row_0_19 * 128 + col_40) / 4] = words_19[k_39];
                                    }
                                }
                                smem_v40[row_0_19 % 32 * 4 + row_0_19 / 32] = scale_word_19;
                            }
                            __syncthreads();
                            if (tid == 0) {
                                int col_tile_9 = col_block_2 * 4 + 1;
                                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                tma_store_2d((&x_q_store), col_tile_9 * 128, row_5, smem_v38_addr);
                                tma_store_3d((&x_sc_store), 0, 0, row_tile_2 * k_tiles_2 + col_tile_9, smem_v40_addr);
                                tma_store_2d((&x_t_store), row_5, col_tile_9 * 128, smem_v42_addr);
                                tma_store_3d((&x_sc_t_store), 0, 0, col_tile_9 * macro_tiles_2 + row_tile_2, smem_v44_addr);
                                asm volatile("cp.async.bulk.commit_group;");
                            }
                        }
                        if (chunk_cols_3 / 128 > 2) {
                            if (tid == 0) {
                                asm volatile("cp.async.bulk.wait_group.read 1;");
                            }
                            __syncthreads();
                            if (tid < 128) {
                                int row_0_20 = tid;
                                row_0_20 = tid % 64 * 2 + tid / 64;
                                unsigned int scale_word_20 = 0;
                                #pragma unroll 1
                                for (int j_20 = 0; j_20 < 4; j_20++) {
                                    int k_block_20 = (j_20 + tid / 8) % 4;
                                    unsigned int pairs_20[16];
                                    #pragma unroll
                                    for (int k_40 = 0; k_40 < 16; k_40++) {
                                        int col_41 = k_block_20 * 32 + (tid * 4 + k_40 * 2) % 32;
                                        float x0_20 = 0.0f;
                                        float x1_20 = 0.0f;
                                        x0_20 = (float)smem_v35[col_41 * 512 + row_0_20];
                                        x1_20 = (float)smem_v35[(col_41 + 1) * 512 + row_0_20];
                                        __nv_bfloat162 _bf16x2_18 = __float22bfloat162_rn(make_float2(x0_20, x1_20));
                                        pairs_20[k_40] = __as_u32(_bf16x2_18);
                                    }
                                    uint32_t _bf16x2_abs_40;
                                    asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_40) : "r"(pairs_20[0]));
                                    unsigned int amax_pair_20 = _bf16x2_abs_40;
                                    #pragma unroll
                                    for (int i_40 = 1; i_40 < 16; i_40++) {
                                        uint32_t _bf16x2_abs_41;
                                        asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_41) : "r"(pairs_20[i_40]));
                                        uint32_t _bf16x2_max_20;
                                        asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_20) : "r"(amax_pair_20), "r"(_bf16x2_abs_41));
                                        amax_pair_20 = _bf16x2_max_20;
                                    }
                                    uint16_t _bf16_max_20;
                                    asm("max.bf16 %0, %1, %2;" : "=h"(_bf16_max_20) : "h"((uint16_t)(amax_pair_20 & 65535)), "h"((uint16_t)(amax_pair_20 >> 16)));
                                    float _cvt_f32_bf16_20;
                                    asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_20) : "h"((uint16_t)(_bf16_max_20)));
                                    float amax_20 = _cvt_f32_bf16_20;
                                    float _fmax_20 = fmaxf(amax_20 * 0.002232142857f, 1e-12f);
                                    float scale_20 = _fmax_20;
                                    uint16_t _ue8m0x2_f32_20;
                                    asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_ue8m0x2_f32_20) : "f"(scale_20), "f"(scale_20));
                                    unsigned int scale_byte_20 = (unsigned int)_ue8m0x2_f32_20 & 255;
                                    unsigned int inverse_lane_20 = 254 - scale_byte_20 << 7;
                                    unsigned int inverse_20 = inverse_lane_20 | inverse_lane_20 << 16;
                                    unsigned int words_20[8];
                                    #pragma unroll
                                    for (int i_41 = 0; i_41 < 8; i_41++) {
                                        uint32_t _bf16x2_mul_40;
                                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_40) : "r"(pairs_20[i_41 * 2]), "r"(inverse_20));
                                        uint16_t _e4m3x2_40;
                                        asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_40) : "r"(_bf16x2_mul_40));
                                        uint32_t _bf16x2_mul_41;
                                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_41) : "r"(pairs_20[i_41 * 2 + 1]), "r"(inverse_20));
                                        uint16_t _e4m3x2_41;
                                        asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_41) : "r"(_bf16x2_mul_41));
                                        words_20[i_41] = (unsigned int)_e4m3x2_40 | (unsigned int)_e4m3x2_41 << 16;
                                    }
                                    scale_word_20 = scale_word_20 | scale_byte_20 << (unsigned int)(k_block_20 * 8);
                                    #pragma unroll
                                    for (int k_41 = 0; k_41 < 8; k_41++) {
                                        int col_42 = k_block_20 * 32 + (tid * 4 + k_41 * 4) % 32;
                                        smem_v41[(row_0_20 * 128 + col_42) / 4] = words_20[k_41];
                                    }
                                }
                                smem_v43[row_0_20 % 32 * 4 + row_0_20 / 32] = scale_word_20;
                            } else {
                                int row_0_21 = tid - 128;
                                unsigned int scale_word_21 = 0;
                                #pragma unroll 1
                                for (int j_21 = 0; j_21 < 4; j_21++) {
                                    int k_block_21 = (j_21 + (tid - 128) / 8) % 4;
                                    unsigned int pairs_21[16];
                                    #pragma unroll
                                    for (int k_42 = 0; k_42 < 16; k_42++) {
                                        int col_43 = k_block_21 * 32 + ((tid - 128) * 4 + k_42 * 2) % 32;
                                        float x0_21 = 0.0f;
                                        float x1_21 = 0.0f;
                                        pairs_21[k_42] = dispatch_words[128 + (row_0_21 * 512 + col_43) / 2];
                                    }
                                    uint32_t _bf16x2_abs_42;
                                    asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_42) : "r"(pairs_21[0]));
                                    unsigned int amax_pair_21 = _bf16x2_abs_42;
                                    #pragma unroll
                                    for (int i_42 = 1; i_42 < 16; i_42++) {
                                        uint32_t _bf16x2_abs_43;
                                        asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_43) : "r"(pairs_21[i_42]));
                                        uint32_t _bf16x2_max_21;
                                        asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_21) : "r"(amax_pair_21), "r"(_bf16x2_abs_43));
                                        amax_pair_21 = _bf16x2_max_21;
                                    }
                                    uint16_t _bf16_max_21;
                                    asm("max.bf16 %0, %1, %2;" : "=h"(_bf16_max_21) : "h"((uint16_t)(amax_pair_21 & 65535)), "h"((uint16_t)(amax_pair_21 >> 16)));
                                    float _cvt_f32_bf16_21;
                                    asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_21) : "h"((uint16_t)(_bf16_max_21)));
                                    float amax_21 = _cvt_f32_bf16_21;
                                    float _fmax_21 = fmaxf(amax_21 * 0.002232142857f, 1e-12f);
                                    float scale_21 = _fmax_21;
                                    uint16_t _ue8m0x2_f32_21;
                                    asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_ue8m0x2_f32_21) : "f"(scale_21), "f"(scale_21));
                                    unsigned int scale_byte_21 = (unsigned int)_ue8m0x2_f32_21 & 255;
                                    unsigned int inverse_lane_21 = 254 - scale_byte_21 << 7;
                                    unsigned int inverse_21 = inverse_lane_21 | inverse_lane_21 << 16;
                                    unsigned int words_21[8];
                                    #pragma unroll
                                    for (int i_43 = 0; i_43 < 8; i_43++) {
                                        uint32_t _bf16x2_mul_42;
                                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_42) : "r"(pairs_21[i_43 * 2]), "r"(inverse_21));
                                        uint16_t _e4m3x2_42;
                                        asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_42) : "r"(_bf16x2_mul_42));
                                        uint32_t _bf16x2_mul_43;
                                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_43) : "r"(pairs_21[i_43 * 2 + 1]), "r"(inverse_21));
                                        uint16_t _e4m3x2_43;
                                        asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_43) : "r"(_bf16x2_mul_43));
                                        words_21[i_43] = (unsigned int)_e4m3x2_42 | (unsigned int)_e4m3x2_43 << 16;
                                    }
                                    scale_word_21 = scale_word_21 | scale_byte_21 << (unsigned int)(k_block_21 * 8);
                                    #pragma unroll
                                    for (int k_43 = 0; k_43 < 8; k_43++) {
                                        int col_44 = k_block_21 * 32 + ((tid - 128) * 4 + k_43 * 4) % 32;
                                        smem_v37[(row_0_21 * 128 + col_44) / 4] = words_21[k_43];
                                    }
                                }
                                smem_v39[row_0_21 % 32 * 4 + row_0_21 / 32] = scale_word_21;
                            }
                            __syncthreads();
                            if (tid == 0) {
                                int col_tile_10 = col_block_2 * 4 + 2;
                                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                tma_store_2d((&x_q_store), col_tile_10 * 128, row_5, smem_v37_addr);
                                tma_store_3d((&x_sc_store), 0, 0, row_tile_2 * k_tiles_2 + col_tile_10, smem_v39_addr);
                                tma_store_2d((&x_t_store), row_5, col_tile_10 * 128, smem_v41_addr);
                                tma_store_3d((&x_sc_t_store), 0, 0, col_tile_10 * macro_tiles_2 + row_tile_2, smem_v43_addr);
                                asm volatile("cp.async.bulk.commit_group;");
                            }
                        }
                        if (chunk_cols_3 / 128 > 3) {
                            if (tid == 0) {
                                asm volatile("cp.async.bulk.wait_group.read 1;");
                            }
                            __syncthreads();
                            if (tid < 128) {
                                int row_0_22 = tid;
                                row_0_22 = tid % 64 * 2 + tid / 64;
                                unsigned int scale_word_22 = 0;
                                #pragma unroll 1
                                for (int j_22 = 0; j_22 < 4; j_22++) {
                                    int k_block_22 = (j_22 + tid / 8) % 4;
                                    unsigned int pairs_22[16];
                                    #pragma unroll
                                    for (int k_44 = 0; k_44 < 16; k_44++) {
                                        int col_45 = k_block_22 * 32 + (tid * 4 + k_44 * 2) % 32;
                                        float x0_22 = 0.0f;
                                        float x1_22 = 0.0f;
                                        x0_22 = (float)smem_v36[col_45 * 512 + row_0_22];
                                        x1_22 = (float)smem_v36[(col_45 + 1) * 512 + row_0_22];
                                        __nv_bfloat162 _bf16x2_19 = __float22bfloat162_rn(make_float2(x0_22, x1_22));
                                        pairs_22[k_44] = __as_u32(_bf16x2_19);
                                    }
                                    uint32_t _bf16x2_abs_44;
                                    asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_44) : "r"(pairs_22[0]));
                                    unsigned int amax_pair_22 = _bf16x2_abs_44;
                                    #pragma unroll
                                    for (int i_44 = 1; i_44 < 16; i_44++) {
                                        uint32_t _bf16x2_abs_45;
                                        asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_45) : "r"(pairs_22[i_44]));
                                        uint32_t _bf16x2_max_22;
                                        asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_22) : "r"(amax_pair_22), "r"(_bf16x2_abs_45));
                                        amax_pair_22 = _bf16x2_max_22;
                                    }
                                    uint16_t _bf16_max_22;
                                    asm("max.bf16 %0, %1, %2;" : "=h"(_bf16_max_22) : "h"((uint16_t)(amax_pair_22 & 65535)), "h"((uint16_t)(amax_pair_22 >> 16)));
                                    float _cvt_f32_bf16_22;
                                    asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_22) : "h"((uint16_t)(_bf16_max_22)));
                                    float amax_22 = _cvt_f32_bf16_22;
                                    float _fmax_22 = fmaxf(amax_22 * 0.002232142857f, 1e-12f);
                                    float scale_22 = _fmax_22;
                                    uint16_t _ue8m0x2_f32_22;
                                    asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_ue8m0x2_f32_22) : "f"(scale_22), "f"(scale_22));
                                    unsigned int scale_byte_22 = (unsigned int)_ue8m0x2_f32_22 & 255;
                                    unsigned int inverse_lane_22 = 254 - scale_byte_22 << 7;
                                    unsigned int inverse_22 = inverse_lane_22 | inverse_lane_22 << 16;
                                    unsigned int words_22[8];
                                    #pragma unroll
                                    for (int i_45 = 0; i_45 < 8; i_45++) {
                                        uint32_t _bf16x2_mul_44;
                                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_44) : "r"(pairs_22[i_45 * 2]), "r"(inverse_22));
                                        uint16_t _e4m3x2_44;
                                        asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_44) : "r"(_bf16x2_mul_44));
                                        uint32_t _bf16x2_mul_45;
                                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_45) : "r"(pairs_22[i_45 * 2 + 1]), "r"(inverse_22));
                                        uint16_t _e4m3x2_45;
                                        asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_45) : "r"(_bf16x2_mul_45));
                                        words_22[i_45] = (unsigned int)_e4m3x2_44 | (unsigned int)_e4m3x2_45 << 16;
                                    }
                                    scale_word_22 = scale_word_22 | scale_byte_22 << (unsigned int)(k_block_22 * 8);
                                    #pragma unroll
                                    for (int k_45 = 0; k_45 < 8; k_45++) {
                                        int col_46 = k_block_22 * 32 + (tid * 4 + k_45 * 4) % 32;
                                        smem_v42[(row_0_22 * 128 + col_46) / 4] = words_22[k_45];
                                    }
                                }
                                smem_v44[row_0_22 % 32 * 4 + row_0_22 / 32] = scale_word_22;
                            } else {
                                int row_0_23 = tid - 128;
                                unsigned int scale_word_23 = 0;
                                #pragma unroll 1
                                for (int j_23 = 0; j_23 < 4; j_23++) {
                                    int k_block_23 = (j_23 + (tid - 128) / 8) % 4;
                                    unsigned int pairs_23[16];
                                    #pragma unroll
                                    for (int k_46 = 0; k_46 < 16; k_46++) {
                                        int col_47 = k_block_23 * 32 + ((tid - 128) * 4 + k_46 * 2) % 32;
                                        float x0_23 = 0.0f;
                                        float x1_23 = 0.0f;
                                        pairs_23[k_46] = dispatch_words[192 + (row_0_23 * 512 + col_47) / 2];
                                    }
                                    uint32_t _bf16x2_abs_46;
                                    asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_46) : "r"(pairs_23[0]));
                                    unsigned int amax_pair_23 = _bf16x2_abs_46;
                                    #pragma unroll
                                    for (int i_46 = 1; i_46 < 16; i_46++) {
                                        uint32_t _bf16x2_abs_47;
                                        asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_47) : "r"(pairs_23[i_46]));
                                        uint32_t _bf16x2_max_23;
                                        asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_23) : "r"(amax_pair_23), "r"(_bf16x2_abs_47));
                                        amax_pair_23 = _bf16x2_max_23;
                                    }
                                    uint16_t _bf16_max_23;
                                    asm("max.bf16 %0, %1, %2;" : "=h"(_bf16_max_23) : "h"((uint16_t)(amax_pair_23 & 65535)), "h"((uint16_t)(amax_pair_23 >> 16)));
                                    float _cvt_f32_bf16_23;
                                    asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_23) : "h"((uint16_t)(_bf16_max_23)));
                                    float amax_23 = _cvt_f32_bf16_23;
                                    float _fmax_23 = fmaxf(amax_23 * 0.002232142857f, 1e-12f);
                                    float scale_23 = _fmax_23;
                                    uint16_t _ue8m0x2_f32_23;
                                    asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_ue8m0x2_f32_23) : "f"(scale_23), "f"(scale_23));
                                    unsigned int scale_byte_23 = (unsigned int)_ue8m0x2_f32_23 & 255;
                                    unsigned int inverse_lane_23 = 254 - scale_byte_23 << 7;
                                    unsigned int inverse_23 = inverse_lane_23 | inverse_lane_23 << 16;
                                    unsigned int words_23[8];
                                    #pragma unroll
                                    for (int i_47 = 0; i_47 < 8; i_47++) {
                                        uint32_t _bf16x2_mul_46;
                                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_46) : "r"(pairs_23[i_47 * 2]), "r"(inverse_23));
                                        uint16_t _e4m3x2_46;
                                        asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_46) : "r"(_bf16x2_mul_46));
                                        uint32_t _bf16x2_mul_47;
                                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_47) : "r"(pairs_23[i_47 * 2 + 1]), "r"(inverse_23));
                                        uint16_t _e4m3x2_47;
                                        asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_47) : "r"(_bf16x2_mul_47));
                                        words_23[i_47] = (unsigned int)_e4m3x2_46 | (unsigned int)_e4m3x2_47 << 16;
                                    }
                                    scale_word_23 = scale_word_23 | scale_byte_23 << (unsigned int)(k_block_23 * 8);
                                    #pragma unroll
                                    for (int k_47 = 0; k_47 < 8; k_47++) {
                                        int col_48 = k_block_23 * 32 + ((tid - 128) * 4 + k_47 * 4) % 32;
                                        smem_v38[(row_0_23 * 128 + col_48) / 4] = words_23[k_47];
                                    }
                                }
                                smem_v40[row_0_23 % 32 * 4 + row_0_23 / 32] = scale_word_23;
                            }
                            __syncthreads();
                            if (tid == 0) {
                                int col_tile_11 = col_block_2 * 4 + 3;
                                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                tma_store_2d((&x_q_store), col_tile_11 * 128, row_5, smem_v38_addr);
                                tma_store_3d((&x_sc_store), 0, 0, row_tile_2 * k_tiles_2 + col_tile_11, smem_v40_addr);
                                tma_store_2d((&x_t_store), row_5, col_tile_11 * 128, smem_v42_addr);
                                tma_store_3d((&x_sc_t_store), 0, 0, col_tile_11 * macro_tiles_2 + row_tile_2, smem_v44_addr);
                                asm volatile("cp.async.bulk.commit_group;");
                            }
                        }
                        if (tid == 0) {
                            asm volatile("cp.async.bulk.wait_group 0;");
                            asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(replay_x)) + ((macro_offset_2_1 + row_5) / mini_size))), "r"(static_cast<unsigned int>(1)) : "memory");
                        }
                        __syncthreads();
                    }
                    dispatch_bits = phase_bits_0;
                }
            }
        }
    } else {
        int iteration = 0;
        while (cluster >= 0 && cluster < true_clusters) {
            if (tid / 32 == 5) {
                if (warp == 5) {
                    if (elect_sync()) {
                        if (cta_rank_0 == 0) {
                            mbarrier_wait(schedule_finished_addr, (iteration + 1) % 2);
                            asm volatile(
                                "clusterlaunchcontrol.try_cancel.async.shared::cta"
                                    ".mbarrier::complete_tx::bytes.multicast::cluster::all.b128"
                                    " [%0], [%1];"
                                :: "r"(smem + 512 + 0 * 16 + 0 * 16), "r"(schedule_arrived_addr + 0 * 8)
                                : "memory");
                        }
                        mbarrier_arrive_expect_tx(schedule_arrived_addr, 16);
                    }
                }
            }
            int shared_tasks_0 = shared_down + shared_swiglu + shared_dx + 3 * shared_wgrad;
            int mini_bwd_1 = mini_down + mini_swiglu + mini_dx;
            int mini_replay_2 = 2 * mini_replay_gate + mini_replay_swiglu;
            int weight_tasks = experts * routed_wgrad;
            int _min_24 = ((tokens) < (macro_size) ? (tokens) : (macro_size));
            int saved_minis_3 = (_min_24 + mini_size - 1) / mini_size;
            int saved_tasks = saved_minis_3 * mini_bwd_1 + 3 * weight_tasks;
            int replay_macro_tasks = macro_size / mini_size * (mini_replay_2 + mini_bwd_1) + 3 * weight_tasks;
            int kind = -1;
            int task_3 = 0;
            int macro_1 = 0;
            int mini_1 = 0;
            int shared = 0;
            if (cluster - comm_clusters >= 0 && true_compute > cluster - comm_clusters) {
                if (shared_tasks_0 > cluster - comm_clusters) {
                    shared = 1;
                    if (shared_down > cluster - comm_clusters) {
                        kind = 0;
                        task_3 = cluster - comm_clusters;
                    } else if (cluster - comm_clusters < shared_down + shared_swiglu) {
                        kind = 1;
                        task_3 = cluster - comm_clusters - shared_down;
                    } else {
                        if (cluster - comm_clusters < shared_down + shared_swiglu + shared_dx) {
                            kind = 2;
                            task_3 = cluster - comm_clusters - shared_down - shared_swiglu;
                        } else {
                            int weight_task = cluster - comm_clusters - shared_down - shared_swiglu - shared_dx;
                            kind = 3 + weight_task / shared_wgrad;
                            task_3 = weight_task % shared_wgrad;
                        }
                    }
                } else {
                    int routed = cluster - comm_clusters - shared_tasks_0;
                    int macro_task = routed;
                    int replay_tasks = 0;
                    if (routed >= saved_tasks) {
                        macro_1 = 1 + (routed - saved_tasks) / replay_macro_tasks;
                        macro_task = (routed - saved_tasks) % replay_macro_tasks;
                        int _min_25 = ((tokens - macro_1 * macro_size) < (macro_size) ? (tokens - macro_1 * macro_size) : (macro_size));
                        int macro_minis = (_min_25 + mini_size - 1) / mini_size;
                        replay_tasks = macro_minis * mini_replay_2;
                    }
                    int _min_26 = ((tokens - macro_1 * macro_size) < (macro_size) ? (tokens - macro_1 * macro_size) : (macro_size));
                    int macro_minis_1 = (_min_26 + mini_size - 1) / mini_size;
                    if (macro_task < replay_tasks) {
                        mini_1 = macro_task / mini_replay_2;
                        int mini_task = macro_task % mini_replay_2;
                        if (mini_task < mini_replay_gate) {
                            kind = 6;
                            task_3 = mini_task;
                        } else if (mini_task < 2 * mini_replay_gate) {
                            kind = 7;
                            task_3 = mini_task - mini_replay_gate;
                        } else {
                            kind = 8;
                            task_3 = mini_task - 2 * mini_replay_gate;
                        }
                    } else {
                        int bwd_task = macro_task - replay_tasks;
                        if (bwd_task < macro_minis_1 * mini_bwd_1) {
                            mini_1 = bwd_task / mini_bwd_1;
                            int mini_task_1 = bwd_task % mini_bwd_1;
                            if (mini_task_1 < mini_down) {
                                kind = 0;
                                task_3 = mini_task_1;
                            } else if (mini_task_1 < mini_down + mini_swiglu) {
                                kind = 1;
                                task_3 = mini_task_1 - mini_down;
                            } else {
                                kind = 2;
                                task_3 = mini_task_1 - mini_down - mini_swiglu;
                            }
                        } else {
                            int weight_task_1 = bwd_task - macro_minis_1 * mini_bwd_1;
                            kind = 3 + weight_task_1 / weight_tasks;
                            task_3 = weight_task_1 % weight_tasks;
                        }
                    }
                }
            }
            unsigned int gemm_phase = gemm_bits;
            unsigned int swiglu_phase = swiglu_bits;
            unsigned int replay_phase = replay_bits;
            int row_count = 2 * (intermediate / 128);
            int shared_rows = local_tokens / 256;
            int shared_down_4 = shared_rows * (intermediate / 256);
            int i_tiles_5 = intermediate / 128;
            int h_tiles = hidden / 128;
            if (shared != 0) {
                if (kind == 0) {
                    int col_blocks_3 = (hidden + 512 - 1) / 512;
                    {
                        col_blocks_3 = (intermediate + 512 - 1) / 512;
                    }
                    int x = -1;
                    int y = -1;
                    int expert = -1;
                    int k_start = 0;
                    int k_end = 0;
                    int first = 0;
                    int row_blocks = local_tokens / 256;
                    if (task_3 < row_blocks * col_blocks_3) {
                        int supergroup = task_3 / (row_blocks * 8);
                        int full_cols = col_blocks_3 / 8 * 8;
                        int row_6 = 0;
                        int col_49 = 0;
                        if (task_3 < row_blocks * full_cols) {
                            row_6 = task_3 % (row_blocks * 8) / 8;
                            col_49 = supergroup * 8 + task_3 % 8;
                        } else {
                            row_6 = (task_3 - row_blocks * full_cols) / (col_blocks_3 - full_cols);
                            col_49 = full_cols + (task_3 - row_blocks * full_cols) % (col_blocks_3 - full_cols);
                        }
                        if ((supergroup & 1) != 0) {
                            row_6 = row_blocks - row_6 - 1;
                        }
                        x = row_6;
                        y = col_49;
                        expert = 0;
                    }
                    unsigned int phase_bits_3 = gemm_phase;
                    int has_hi = 0;
                    has_hi = (int)((y * 2 + 1) * 256 < intermediate);
                    int global_mini = 0;
                    int macro_rows = 0;
                    int iterations = hidden / 64;
                    if (expert < 0) {
                        if (tid == 0) {
                        }
                    } else if (tid / 32 == 7) {
                        if (warp == 7) {
                            if (elect_sync()) {
                                {
                                    int _min_27 = ((mini_size) < (tokens - global_mini * mini_size) ? (mini_size) : (tokens - global_mini * mini_size));
                                    int _max_2 = ((0) > (_min_27) ? (0) : (_min_27));
                                    int mini_rows_4 = _max_2;
                                    int required_3 = (mini_rows_4 + 127) / 128 * ((hidden + 511) / 512);
                                }
                                unsigned int previous = phase_bits_3 >> 7 & 1;
                                unsigned int bits = phase_bits_3;
                                if (previous != 0) {
                                    mbarrier_wait(gemm_finished_addr, bits >> 16 & 1);
                                    mbarrier_wait(gemm_finished_addr + 8, bits >> 17 & 1);
                                    mbarrier_wait(gemm_finished_addr + 16, bits >> 18 & 1);
                                    mbarrier_wait(gemm_finished_addr + 24, bits >> 19 & 1);
                                    mbarrier_wait(gemm_finished_addr + 32, bits >> 20 & 1);
                                    mbarrier_wait(gemm_finished_addr + 40, bits >> 21 & 1);
                                    bits = bits ^ 128;
                                }
                                phase_bits_3 = bits;
                                int ring = 0;
                                #pragma unroll 1
                                for (int idx = 0; idx < iterations; idx++) {
                                    mbarrier_wait(gemm_finished_addr + (ring) * 8, phase_bits_3 >> (unsigned int)(16 + ring) & 1);
                                    asm volatile(
                                        "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                        " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                        :: "r"(a_nt_addr + (unsigned int)(ring * 16384)), "l"((&dy_s)), "r"(0), "r"(x * 256 + cta_rank_0 * 128), "r"(idx), "r"(0), "r"(0),
                                           "r"(((gemm_arrived_addr + (ring) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                    asm volatile(
                                        "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                        " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                        :: "r"(b_ab_addr + (unsigned int)(ring * 16384)), "l"((&wd_s)), "r"(0), "r"(idx * 64), "r"(y * 2 * 4 + cta_rank_0 * 2), "r"(expert), "r"(0),
                                           "r"(((gemm_arrived_addr + (ring) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                    asm volatile(
                                        "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                        " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                        :: "r"(b_ab_hi_addr + (unsigned int)(ring * 16384)), "l"((&wd_s)), "r"(0), "r"(idx * 64), "r"((y * 2 + 1) * 4 + cta_rank_0 * 2), "r"(expert), "r"(0),
                                           "r"(((gemm_arrived_addr + (ring) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                    phase_bits_3 = phase_bits_3 ^ (unsigned int)(1 << 16 + ring);
                                    ring = (ring + 1) % 4;
                                }
                            }
                        }
                    } else {
                        if (tid / 32 == 4 && cta_rank_0 == 0) {
                            if (warp == 4) {
                                if (elect_sync()) {
                                    int ring_1 = 0;
                                    mbarrier_wait(output_finished_addr, phase_bits_3 >> 22 & 1);
                                    phase_bits_3 = phase_bits_3 ^ 4194304;
                                    asm volatile("tcgen05.fence::after_thread_sync;");
                                    #pragma unroll 1
                                    for (int idx_1 = 0; idx_1 < iterations; idx_1++) {
                                        mbarrier_arrive_expect_tx(gemm_arrived_addr + (ring_1) * 8, 98304);
                                        mbarrier_wait(gemm_arrived_addr + (ring_1) * 8, phase_bits_3 >> (unsigned int)ring_1 & 1);
                                        int _mma_a_lo_0 = (((a_nt_addr) >> 4) & 0x3FFF) + (ring_1) * 1024;
                                        int _mma_b_lo_0 = ((((b_ab_addr) >> 4) & 0x3FFF) | 0x2000000) + (ring_1) * 1024;
                                        asm volatile(
            "{\n\t"
            ".reg .pred p0, p1;\n\t"
            ".reg .b32 adhi, bdhi, alo, blo, id, m0, m1, m2, m3, m4, m5, m6, m7;\n\t"
            ".reg .b64 da, db;\n\t"
            ""
            "setp.ne.b32 p0, %3, 0;\n\t"
            "setp.ne.b32 p1, 1, 0;\n\t"
            "mov.b32 m0, 0; mov.b32 m1, 0; mov.b32 m2, 0; mov.b32 m3, 0;\n\tmov.b32 m4, 0; mov.b32 m5, 0; mov.b32 m6, 0; mov.b32 m7, 0;\n\t"
            "mov.b32 adhi, 0x40004040;\n\t"
            "mov.b32 bdhi, 0x40004040;\n\t"
            "mov.b32 id, 272696464;\n\t"
            "mov.b32 alo, %0;\n\t"
            "mov.b32 blo, %1;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
            "add.u32 alo, alo, 2;\n\t"
            "add.u32 blo, blo, 128;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
            "add.u32 alo, alo, 2;\n\t"
            "add.u32 blo, blo, 128;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
            "add.u32 alo, alo, 2;\n\t"
            "add.u32 blo, blo, 128;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
            "}\n"
            :: "r"(_mma_a_lo_0), "r"(_mma_b_lo_0), "r"(tmem_accumulator), "r"(((idx_1 == 0) ? 0 : 1)));
                                        int _mma_a_lo_1 = (((a_nt_addr) >> 4) & 0x3FFF) + (ring_1) * 1024;
                                        int _mma_b_lo_1 = ((((b_ab_hi_addr) >> 4) & 0x3FFF) | 0x2000000) + (ring_1) * 1024;
                                        asm volatile(
            "{\n\t"
            ".reg .pred p0, p1;\n\t"
            ".reg .b32 adhi, bdhi, alo, blo, id, m0, m1, m2, m3, m4, m5, m6, m7;\n\t"
            ".reg .b64 da, db;\n\t"
            ""
            "setp.ne.b32 p0, %3, 0;\n\t"
            "setp.ne.b32 p1, 1, 0;\n\t"
            "mov.b32 m0, 0; mov.b32 m1, 0; mov.b32 m2, 0; mov.b32 m3, 0;\n\tmov.b32 m4, 0; mov.b32 m5, 0; mov.b32 m6, 0; mov.b32 m7, 0;\n\t"
            "mov.b32 adhi, 0x40004040;\n\t"
            "mov.b32 bdhi, 0x40004040;\n\t"
            "mov.b32 id, 272696464;\n\t"
            "mov.b32 alo, %0;\n\t"
            "mov.b32 blo, %1;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
            "add.u32 alo, alo, 2;\n\t"
            "add.u32 blo, blo, 128;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
            "add.u32 alo, alo, 2;\n\t"
            "add.u32 blo, blo, 128;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
            "add.u32 alo, alo, 2;\n\t"
            "add.u32 blo, blo, 128;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
            "}\n"
            :: "r"(_mma_a_lo_1), "r"(_mma_b_lo_1), "r"((tmem_accumulator + (256))), "r"(((idx_1 == 0) ? 0 : 1)));
                                        tcgen05_commit_cg2_multicast(gemm_finished_addr + (ring_1) * 8, (uint16_t)(3));
                                        phase_bits_3 = phase_bits_3 ^ (unsigned int)(1 << ring_1);
                                        ring_1 = (ring_1 + 1) % 4;
                                    }
                                    tcgen05_commit_cg2_multicast(output_arrived_addr, (uint16_t)(3));
                                }
                            }
                        } else if (tid < 128) {
                            mbarrier_wait(output_arrived_addr, phase_bits_3 >> 6 & 1);
                            phase_bits_3 = phase_bits_3 ^ 64;
                            unsigned int packed[128];
                            #pragma unroll
                            for (int chunk = 0; chunk < 8; chunk++) {
                                #pragma unroll
                                for (int sub = 0; sub < 2; sub++) {
                                    unsigned int address = taddr_1 + (unsigned int)(tid / 32 * 32 + sub * 16 << 16) + (unsigned int)(chunk * 32);
                                    float _tmem_load_0[16];
                                    asm volatile(
                                        "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                        : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[15]))
                                        : "r"(address));
                                    #pragma unroll
                                    for (int pair = 0; pair < 8; pair++) {
                                        __nv_bfloat162 _bf16x2_20 = __float22bfloat162_rn(make_float2(_tmem_load_0[pair * 2], _tmem_load_0[pair * 2 + 1]));
                                        packed[chunk * 16 + sub * 8 + pair] = __as_u32(_bf16x2_20);
                                    }
                                }
                            }
                            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                            int last = 1;
                            last = 1 - has_hi;
                            if (last != 0) {
                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                if (tid == 0) {
                                    asm volatile(
                                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                                        :: "r"((output_finished_addr) & 0xFEFFFFFF) : "memory");
                                }
                            }
                            if (tid == 0) {
                                int previous_offset_3 = macro_size;
                                int output_row = x * 256 + cta_rank_0 * 128;
                                int _min_28 = ((macro_size) < (tokens - previous_offset_3) ? (macro_size) : (tokens - previous_offset_3));
                                if (output_row < _min_28) {
                                }
                            }
                            #pragma unroll
                            for (int chunk_1 = 0; chunk_1 < 8; chunk_1++) {
                                if (tid == 0) {
                                    asm volatile("cp.async.bulk.wait_group.read 2;");
                                }
                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                int warp_0 = tid / 32;
                                int lane_1 = tid % 32;
                                #pragma unroll
                                for (int half = 0; half < 2; half++) {
                                    #pragma unroll
                                    for (int col_tile_12 = 0; col_tile_12 < 2; col_tile_12++) {
                                        int row_7 = warp_0 * 32 + half * 16 + lane_1 % 16;
                                        int col_50 = col_tile_12 * 16 + lane_1 / 16 * 8;
                                        unsigned int address_1 = d_smem_addr + (unsigned int)(chunk_1 % 3 * 8192) + (unsigned int)((row_7 * 32 + col_50) * 2);
                                        address_1 = address_1 ^ (address_1 & 511) >> 7 << 4;
                                        int offset_2 = chunk_1 * 16 + half * 8 + col_tile_12 * 4;
                                        uint32_t _stmatrix_addr_1 = static_cast<uint32_t>(address_1);
                                        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                            :: "r"(_stmatrix_addr_1), "r"(*reinterpret_cast<const uint32_t*>(&packed[offset_2])), "r"(*reinterpret_cast<const uint32_t*>(&packed[offset_2 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&packed[offset_2 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&packed[offset_2 + 3]))
                                            : "memory");
                                    }
                                }
                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                if (tid == 0) {
                                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                    asm volatile(
                                        "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                        " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                        :: "l"((&dh_s)), "r"(0), "r"(x * 256 + cta_rank_0 * 128), "r"(y * 2 * 8 + chunk_1), "r"(0), "r"(0), "r"(d_smem_addr + (unsigned int)(chunk_1 % 3 * 8192)), "l"(0x12F0000000000000ULL) : "memory");
                                    asm volatile("cp.async.bulk.commit_group;");
                                }
                            }
                            if (has_hi != 0) {
                                unsigned int packed_0[128];
                                #pragma unroll
                                for (int chunk_2 = 0; chunk_2 < 8; chunk_2++) {
                                    #pragma unroll
                                    for (int sub_1 = 0; sub_1 < 2; sub_1++) {
                                        unsigned int address_2 = taddr_1 + (unsigned int)(tid / 32 * 32 + sub_1 * 16 << 16) + 256 + (unsigned int)(chunk_2 * 32);
                                        float _tmem_load_1[16];
                                        asm volatile(
                                            "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                            : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[15]))
                                            : "r"(address_2));
                                        #pragma unroll
                                        for (int pair_1 = 0; pair_1 < 8; pair_1++) {
                                            __nv_bfloat162 _bf16x2_21 = __float22bfloat162_rn(make_float2(_tmem_load_1[pair_1 * 2], _tmem_load_1[pair_1 * 2 + 1]));
                                            packed_0[chunk_2 * 16 + sub_1 * 8 + pair_1] = __as_u32(_bf16x2_21);
                                        }
                                    }
                                }
                                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                                int last_1 = 1;
                                if (last_1 != 0) {
                                    asm volatile("barrier.sync 1, 128;" ::: "memory");
                                    if (tid == 0) {
                                        asm volatile(
                                            "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                                            :: "r"((output_finished_addr) & 0xFEFFFFFF) : "memory");
                                    }
                                }
                                #pragma unroll
                                for (int chunk_3 = 0; chunk_3 < 8; chunk_3++) {
                                    if (tid == 0) {
                                        asm volatile("cp.async.bulk.wait_group.read 2;");
                                    }
                                    asm volatile("barrier.sync 1, 128;" ::: "memory");
                                    int warp_0_1 = tid / 32;
                                    int lane_2 = tid % 32;
                                    #pragma unroll
                                    for (int half_1 = 0; half_1 < 2; half_1++) {
                                        #pragma unroll
                                        for (int col_tile_13 = 0; col_tile_13 < 2; col_tile_13++) {
                                            int row_8 = warp_0_1 * 32 + half_1 * 16 + lane_2 % 16;
                                            int col_51 = col_tile_13 * 16 + lane_2 / 16 * 8;
                                            unsigned int address_3 = d_smem_addr + (unsigned int)((8 + chunk_3) % 3 * 8192) + (unsigned int)((row_8 * 32 + col_51) * 2);
                                            address_3 = address_3 ^ (address_3 & 511) >> 7 << 4;
                                            int offset_3 = chunk_3 * 16 + half_1 * 8 + col_tile_13 * 4;
                                            uint32_t _stmatrix_addr_2 = static_cast<uint32_t>(address_3);
                                            asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                                :: "r"(_stmatrix_addr_2), "r"(*reinterpret_cast<const uint32_t*>(&packed_0[offset_3])), "r"(*reinterpret_cast<const uint32_t*>(&packed_0[offset_3 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&packed_0[offset_3 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&packed_0[offset_3 + 3]))
                                                : "memory");
                                        }
                                    }
                                    asm volatile("barrier.sync 1, 128;" ::: "memory");
                                    if (tid == 0) {
                                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                        asm volatile(
                                            "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                            " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                            :: "l"((&dh_s)), "r"(0), "r"(x * 256 + cta_rank_0 * 128), "r"((y * 2 + 1) * 8 + chunk_3), "r"(0), "r"(0), "r"(d_smem_addr + (unsigned int)((8 + chunk_3) % 3 * 8192)), "l"(0x12F0000000000000ULL) : "memory");
                                        asm volatile("cp.async.bulk.commit_group;");
                                    }
                                }
                            }
                            if (tid == 0) {
                                asm volatile("cp.async.bulk.wait_group.read 0;");
                            }
                            asm volatile("barrier.sync 4, 128;" ::: "memory");
                            if (tid / 32 == 0) {
                                if (warp == 0) {
                                    if (elect_sync()) {
                                        asm volatile("cp.async.bulk.wait_group 0;");
                                        bool enabled_value_2 = 1;
                                        if (enabled_value_2 != 0) {
                                            asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(dh_ready)) + ((macro_rows + x) * (intermediate / 256) + y * 2))), "r"(static_cast<unsigned int>(1)) : "memory");
                                        }
                                        if (has_hi != 0) {
                                            asm volatile("cp.async.bulk.wait_group 0;");
                                            bool enabled_value_0 = 1;
                                            if (enabled_value_0 != 0) {
                                                asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(dh_ready)) + ((macro_rows + x) * (intermediate / 256) + y * 2 + 1))), "r"(static_cast<unsigned int>(1)) : "memory");
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    gemm_phase = phase_bits_3;
                } else if (kind == 1) {
                    unsigned int phase_bits_4 = swiglu_phase;
                    int col_blocks_4 = intermediate / 128;
                    int num_tiles = local_tokens / 128 * col_blocks_4;
                    int macro_row_offset = 0;
                    int first_tile_1 = task_3 * 16 + cta_rank_0 * 8;
                    int tile_end = num_tiles;
                    int _min_30 = ((8) < (tile_end - first_tile_1) ? (8) : (tile_end - first_tile_1));
                    int _max_3 = ((0) > (_min_30) ? (0) : (_min_30));
                    int tiles = _max_3;
                    if (tiles > 0) {
                        if (tid == 0) {
                            int _min_31 = ((tiles) < (2) ? (tiles) : (2));
                            #pragma unroll 1
                            for (int stage_4 = 0; stage_4 < _min_31; stage_4++) {
                                int col_blocks_0 = intermediate / 128;
                                int row_9 = (first_tile_1 + stage_4) / col_blocks_0;
                                int col_52 = (first_tile_1 + stage_4) % col_blocks_0;
                                mbarrier_arrive_expect_tx(swiglu_arrived_addr + (stage_4) * 8, 98304);
                                int parent = row_9 / 2 * (intermediate / 256) + col_52 / 2;
                                int32_t _relaxed_ld_8;
                                asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_8) : "l"(dh_ready + parent) : "memory");
                                int value_6 = _relaxed_ld_8;
                                while (value_6 < 2) {
                                    asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                    int32_t _relaxed_ld_9;
                                    asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_9) : "l"(dh_ready + parent) : "memory");
                                    value_6 = _relaxed_ld_9;
                                }
                                asm volatile("fence.acquire.gpu;" ::: "memory");
                                asm volatile(
                                    "cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
                                    " [%0], [%1, {%2, %3, %4, %5, %6}], [%7];"
                                    :: "r"(sw_dh_addr + (unsigned int)(stage_4 * 32768)), "l"((&dh_sw_s)), "r"(0), "r"((row_9 - macro_row_offset) * 128), "r"(col_52 * 2), "r"(0), "r"(0), "r"(swiglu_arrived_addr + (stage_4) * 8) : "memory");
                                asm volatile(
                                    "cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
                                    " [%0], [%1, {%2, %3, %4, %5, %6}], [%7];"
                                    :: "r"(sw_gate_addr + (unsigned int)(stage_4 * 32768)), "l"((&gate_sw_s)), "r"(0), "r"((row_9 - macro_row_offset) * 128), "r"(col_52 * 2), "r"(0), "r"(0), "r"(swiglu_arrived_addr + (stage_4) * 8) : "memory");
                                asm volatile(
                                    "cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
                                    " [%0], [%1, {%2, %3, %4, %5, %6}], [%7];"
                                    :: "r"(sw_up_addr + (unsigned int)(stage_4 * 32768)), "l"((&up_sw_s)), "r"(0), "r"((row_9 - macro_row_offset) * 128), "r"(col_52 * 2), "r"(0), "r"(0), "r"(swiglu_arrived_addr + (stage_4) * 8) : "memory");
                            }
                        }
                        #pragma unroll 1
                        for (int index = 0; index < tiles; index++) {
                            int stage_5 = index % 2;
                            mbarrier_wait(swiglu_arrived_addr + (stage_5) * 8, phase_bits_4 >> (unsigned int)stage_5 & 1);
                            phase_bits_4 = phase_bits_4 ^ (unsigned int)(1 << stage_5);
                            int row_10 = (first_tile_1 + index) / col_blocks_4;
                            int col_53 = (first_tile_1 + index) % col_blocks_4;
                            float gate[64];
                            float up[64];
                            float dhidden[64];
                            int warp_0_2 = tid / 32;
                            int local_warp = warp_0_2 / 4 + warp_0_2 % 4 * 2;
                            int lane_3 = tid % 32;
                            #pragma unroll
                            for (int tile_col = 0; tile_col < 8; tile_col++) {
                                unsigned int packed_1[4];
                                unsigned int address_4 = sw_gate_addr + (unsigned int)(stage_5 * 32768) + (unsigned int)(((tile_col * 16 + lane_3 / 16 * 8) / 64 * 128 * 64 + (local_warp * 16 + lane_3 % 16) * 64 + (tile_col * 16 + lane_3 / 16 * 8) % 64) * 2);
                                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                                    : "=r"(packed_1[0]), "=r"(packed_1[1]), "=r"(packed_1[2]), "=r"(packed_1[3])
                                    : "r"(address_4 ^ (address_4 & 1023) >> 7 << 4)
                                    : "memory");
                                #pragma unroll
                                for (int pair_2 = 0; pair_2 < 4; pair_2++) {
                                    float2 _cvt_f32_8 = __bfloat1622float2(__as_bf16x2(packed_1[pair_2]));
                                    gate[tile_col * 8 + pair_2 * 2] = _cvt_f32_8.x;
                                    gate[tile_col * 8 + pair_2 * 2 + 1] = _cvt_f32_8.y;
                                }
                            }
                            #pragma unroll
                            for (int elem = 0; elem < 64; elem++) {
                                dhidden[elem] = gate[elem] * -1.0f;
                            }
                            #pragma unroll
                            for (int elem_1 = 0; elem_1 < 64; elem_1++) {
                                float _exp_0 = expf(dhidden[elem_1]);
                                dhidden[elem_1] = _exp_0;
                            }
                            #pragma unroll
                            for (int elem_2 = 0; elem_2 < 64; elem_2++) {
                                dhidden[elem_2] = dhidden[elem_2] + 1.0f;
                            }
                            #pragma unroll
                            for (int elem_3 = 0; elem_3 < 64; elem_3++) {
                                gate[elem_3] = gate[elem_3] / dhidden[elem_3];
                            }
                            #pragma unroll
                            for (int elem_4 = 0; elem_4 < 64; elem_4++) {
                                up[elem_4] = gate[elem_4] * -1.0f;
                            }
                            #pragma unroll
                            for (int elem_5 = 0; elem_5 < 64; elem_5++) {
                                up[elem_5] = up[elem_5] + 1.0f;
                            }
                            #pragma unroll
                            for (int elem_6 = 0; elem_6 < 64; elem_6++) {
                                up[elem_6] = up[elem_6] / dhidden[elem_6];
                            }
                            #pragma unroll
                            for (int elem_7 = 0; elem_7 < 64; elem_7++) {
                                up[elem_7] = up[elem_7] + gate[elem_7];
                            }
                            int warp_1 = tid / 32;
                            int local_warp_2 = warp_1 / 4 + warp_1 % 4 * 2;
                            int lane_3_1 = tid % 32;
                            #pragma unroll
                            for (int tile_col_1 = 0; tile_col_1 < 8; tile_col_1++) {
                                unsigned int packed_2[4];
                                unsigned int address_5 = sw_dh_addr + (unsigned int)(stage_5 * 32768) + (unsigned int)(((tile_col_1 * 16 + lane_3_1 / 16 * 8) / 64 * 128 * 64 + (local_warp_2 * 16 + lane_3_1 % 16) * 64 + (tile_col_1 * 16 + lane_3_1 / 16 * 8) % 64) * 2);
                                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                                    : "=r"(packed_2[0]), "=r"(packed_2[1]), "=r"(packed_2[2]), "=r"(packed_2[3])
                                    : "r"(address_5 ^ (address_5 & 1023) >> 7 << 4)
                                    : "memory");
                                #pragma unroll
                                for (int pair_3 = 0; pair_3 < 4; pair_3++) {
                                    float2 _cvt_f32_9 = __bfloat1622float2(__as_bf16x2(packed_2[pair_3]));
                                    dhidden[tile_col_1 * 8 + pair_3 * 2] = _cvt_f32_9.x;
                                    dhidden[tile_col_1 * 8 + pair_3 * 2 + 1] = _cvt_f32_9.y;
                                }
                            }
                            #pragma unroll
                            for (int elem_8 = 0; elem_8 < 64; elem_8++) {
                                gate[elem_8] = gate[elem_8] * dhidden[elem_8];
                            }
                            #pragma unroll
                            for (int elem_9 = 0; elem_9 < 64; elem_9++) {
                                dhidden[elem_9] = dhidden[elem_9] * up[elem_9];
                            }
                            int warp_4 = tid / 32;
                            int local_warp_5 = warp_4 / 4 + warp_4 % 4 * 2;
                            int lane_6 = tid % 32;
                            #pragma unroll
                            for (int tile_col_2 = 0; tile_col_2 < 8; tile_col_2++) {
                                unsigned int packed_3[4];
                                unsigned int address_6 = sw_up_addr + (unsigned int)(stage_5 * 32768) + (unsigned int)(((tile_col_2 * 16 + lane_6 / 16 * 8) / 64 * 128 * 64 + (local_warp_5 * 16 + lane_6 % 16) * 64 + (tile_col_2 * 16 + lane_6 / 16 * 8) % 64) * 2);
                                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                                    : "=r"(packed_3[0]), "=r"(packed_3[1]), "=r"(packed_3[2]), "=r"(packed_3[3])
                                    : "r"(address_6 ^ (address_6 & 1023) >> 7 << 4)
                                    : "memory");
                                #pragma unroll
                                for (int pair_4 = 0; pair_4 < 4; pair_4++) {
                                    float2 _cvt_f32_10 = __bfloat1622float2(__as_bf16x2(packed_3[pair_4]));
                                    up[tile_col_2 * 8 + pair_4 * 2] = _cvt_f32_10.x;
                                    up[tile_col_2 * 8 + pair_4 * 2 + 1] = _cvt_f32_10.y;
                                }
                            }
                            #pragma unroll
                            for (int elem_10 = 0; elem_10 < 64; elem_10++) {
                                dhidden[elem_10] = dhidden[elem_10] * up[elem_10];
                            }
                            int warp_7 = tid / 32;
                            int local_warp_8 = warp_7 / 4 + warp_7 % 4 * 2;
                            int lane_9 = tid % 32;
                            #pragma unroll
                            for (int tile_col_3 = 0; tile_col_3 < 8; tile_col_3++) {
                                unsigned int packed_4[4];
                                #pragma unroll
                                for (int pair_5 = 0; pair_5 < 4; pair_5++) {
                                    __nv_bfloat162 _bf16x2_22 = __float22bfloat162_rn(make_float2(dhidden[tile_col_3 * 8 + pair_5 * 2], dhidden[tile_col_3 * 8 + pair_5 * 2 + 1]));
                                    packed_4[pair_5] = __as_u32(_bf16x2_22);
                                }
                                unsigned int address_7 = sw_gate_addr + (unsigned int)(stage_5 * 32768) + (unsigned int)(((tile_col_3 * 16 + lane_9 / 16 * 8) / 64 * 128 * 64 + (local_warp_8 * 16 + lane_9 % 16) * 64 + (tile_col_3 * 16 + lane_9 / 16 * 8) % 64) * 2);
                                uint32_t _stmatrix_addr_3 = static_cast<uint32_t>(address_7 ^ (address_7 & 1023) >> 7 << 4);
                                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                    :: "r"(_stmatrix_addr_3), "r"(*reinterpret_cast<const uint32_t*>(&packed_4[0])), "r"(*reinterpret_cast<const uint32_t*>(&packed_4[1])), "r"(*reinterpret_cast<const uint32_t*>(&packed_4[2])), "r"(*reinterpret_cast<const uint32_t*>(&packed_4[3]))
                                    : "memory");
                            }
                            int warp_10 = tid / 32;
                            int local_warp_11 = warp_10 / 4 + warp_10 % 4 * 2;
                            int lane_12 = tid % 32;
                            #pragma unroll
                            for (int tile_col_4 = 0; tile_col_4 < 8; tile_col_4++) {
                                unsigned int packed_5[4];
                                #pragma unroll
                                for (int pair_6 = 0; pair_6 < 4; pair_6++) {
                                    __nv_bfloat162 _bf16x2_23 = __float22bfloat162_rn(make_float2(gate[tile_col_4 * 8 + pair_6 * 2], gate[tile_col_4 * 8 + pair_6 * 2 + 1]));
                                    packed_5[pair_6] = __as_u32(_bf16x2_23);
                                }
                                unsigned int address_8 = sw_up_addr + (unsigned int)(stage_5 * 32768) + (unsigned int)(((tile_col_4 * 16 + lane_12 / 16 * 8) / 64 * 128 * 64 + (local_warp_11 * 16 + lane_12 % 16) * 64 + (tile_col_4 * 16 + lane_12 / 16 * 8) % 64) * 2);
                                uint32_t _stmatrix_addr_4 = static_cast<uint32_t>(address_8 ^ (address_8 & 1023) >> 7 << 4);
                                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                    :: "r"(_stmatrix_addr_4), "r"(*reinterpret_cast<const uint32_t*>(&packed_5[0])), "r"(*reinterpret_cast<const uint32_t*>(&packed_5[1])), "r"(*reinterpret_cast<const uint32_t*>(&packed_5[2])), "r"(*reinterpret_cast<const uint32_t*>(&packed_5[3]))
                                    : "memory");
                            }
                            __syncthreads();
                            if (tid == 0) {
                                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                tma_store_5d((&dg_sw_s), 0, (row_10 - macro_row_offset) * 128, col_53 * 2, 0, 0, sw_gate_addr + (unsigned int)(stage_5 * 32768));
                                asm volatile("cp.async.bulk.commit_group;");
                                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                tma_store_5d((&du_sw_s), 0, (row_10 - macro_row_offset) * 128, col_53 * 2, 0, 0, sw_up_addr + (unsigned int)(stage_5 * 32768));
                                asm volatile("cp.async.bulk.commit_group;");
                                if (tiles > index + 2) {
                                    asm volatile("cp.async.bulk.wait_group.read 0;");
                                    int col_blocks_0_1 = intermediate / 128;
                                    int row_1_1 = (first_tile_1 + index + 2) / col_blocks_0_1;
                                    int col_2_1 = (first_tile_1 + index + 2) % col_blocks_0_1;
                                    mbarrier_arrive_expect_tx(swiglu_arrived_addr + (stage_5) * 8, 98304);
                                    int parent_1 = row_1_1 / 2 * (intermediate / 256) + col_2_1 / 2;
                                    int32_t _relaxed_ld_10;
                                    asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_10) : "l"(dh_ready + parent_1) : "memory");
                                    int value_7 = _relaxed_ld_10;
                                    while (value_7 < 2) {
                                        asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                        int32_t _relaxed_ld_11;
                                        asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_11) : "l"(dh_ready + parent_1) : "memory");
                                        value_7 = _relaxed_ld_11;
                                    }
                                    asm volatile("fence.acquire.gpu;" ::: "memory");
                                    asm volatile(
                                        "cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
                                        " [%0], [%1, {%2, %3, %4, %5, %6}], [%7];"
                                        :: "r"(sw_dh_addr + (unsigned int)(stage_5 * 32768)), "l"((&dh_sw_s)), "r"(0), "r"((row_1_1 - macro_row_offset) * 128), "r"(col_2_1 * 2), "r"(0), "r"(0), "r"(swiglu_arrived_addr + (stage_5) * 8) : "memory");
                                    asm volatile(
                                        "cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
                                        " [%0], [%1, {%2, %3, %4, %5, %6}], [%7];"
                                        :: "r"(sw_gate_addr + (unsigned int)(stage_5 * 32768)), "l"((&gate_sw_s)), "r"(0), "r"((row_1_1 - macro_row_offset) * 128), "r"(col_2_1 * 2), "r"(0), "r"(0), "r"(swiglu_arrived_addr + (stage_5) * 8) : "memory");
                                    asm volatile(
                                        "cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
                                        " [%0], [%1, {%2, %3, %4, %5, %6}], [%7];"
                                        :: "r"(sw_up_addr + (unsigned int)(stage_5 * 32768)), "l"((&up_sw_s)), "r"(0), "r"((row_1_1 - macro_row_offset) * 128), "r"(col_2_1 * 2), "r"(0), "r"(0), "r"(swiglu_arrived_addr + (stage_5) * 8) : "memory");
                                }
                            }
                            __syncthreads();
                        }
                        if (tid == 0) {
                            asm volatile("cp.async.bulk.wait_group 0;");
                            #pragma unroll 1
                            for (int index_1 = 0; index_1 < tiles; index_1++) {
                                asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(dg_ready)) + ((first_tile_1 + index_1) / col_blocks_4 / 2))), "r"(static_cast<unsigned int>(1)) : "memory");
                            }
                        }
                    }
                    if (tid == 0) {
                    }
                    swiglu_phase = phase_bits_4;
                } else {
                    if (kind == 2) {
                        int col_blocks_5 = (intermediate + 512 - 1) / 512;
                        {
                            col_blocks_5 = (hidden + 512 - 1) / 512;
                        }
                        int x_1 = -1;
                        int y_1 = -1;
                        int expert_1 = -1;
                        int k_start_1 = 0;
                        int k_end_1 = 0;
                        int first_1 = 0;
                        int row_blocks_1 = local_tokens / 256;
                        if (task_3 < row_blocks_1 * col_blocks_5) {
                            int supergroup_1 = task_3 / (row_blocks_1 * 8);
                            int full_cols_1 = col_blocks_5 / 8 * 8;
                            int row_11 = 0;
                            int col_54 = 0;
                            if (task_3 < row_blocks_1 * full_cols_1) {
                                row_11 = task_3 % (row_blocks_1 * 8) / 8;
                                col_54 = supergroup_1 * 8 + task_3 % 8;
                            } else {
                                row_11 = (task_3 - row_blocks_1 * full_cols_1) / (col_blocks_5 - full_cols_1);
                                col_54 = full_cols_1 + (task_3 - row_blocks_1 * full_cols_1) % (col_blocks_5 - full_cols_1);
                            }
                            if ((supergroup_1 & 1) != 0) {
                                row_11 = row_blocks_1 - row_11 - 1;
                            }
                            x_1 = row_11;
                            y_1 = col_54;
                            expert_1 = 0;
                        }
                        unsigned int phase_bits_5 = gemm_phase;
                        int has_hi_1 = 0;
                        has_hi_1 = (int)((y_1 * 2 + 1) * 256 < hidden);
                        int global_mini_1 = 0;
                        int macro_rows_1 = 0;
                        int iterations_1 = intermediate / 64 + intermediate / 64;
                        if (expert_1 < 0) {
                            if (tid == 0) {
                            }
                        } else if (tid / 32 == 7) {
                            if (warp == 7) {
                                if (elect_sync()) {
                                    {
                                        bool enabled_value_3 = 1;
                                        if (enabled_value_3 != 0) {
                                            int32_t _relaxed_ld_12;
                                            asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_12) : "l"(dg_ready + (macro_rows_1 + x_1)) : "memory");
                                            int value_8 = _relaxed_ld_12;
                                            while (value_8 < row_count) {
                                                asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                                int32_t _relaxed_ld_13;
                                                asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_13) : "l"(dg_ready + (macro_rows_1 + x_1)) : "memory");
                                                value_8 = _relaxed_ld_13;
                                            }
                                            asm volatile("fence.acquire.gpu;" ::: "memory");
                                        }
                                        int _min_32 = ((mini_size) < (tokens - global_mini_1 * mini_size) ? (mini_size) : (tokens - global_mini_1 * mini_size));
                                        int _max_4 = ((0) > (_min_32) ? (0) : (_min_32));
                                        int mini_rows_5 = _max_4;
                                        int required_4 = (mini_rows_5 + 127) / 128 * ((intermediate + 511) / 512);
                                    }
                                    unsigned int previous_1 = phase_bits_5 >> 7 & 1;
                                    unsigned int bits_1 = phase_bits_5;
                                    if (previous_1 != 0) {
                                        mbarrier_wait(gemm_finished_addr, bits_1 >> 16 & 1);
                                        mbarrier_wait(gemm_finished_addr + 8, bits_1 >> 17 & 1);
                                        mbarrier_wait(gemm_finished_addr + 16, bits_1 >> 18 & 1);
                                        mbarrier_wait(gemm_finished_addr + 24, bits_1 >> 19 & 1);
                                        mbarrier_wait(gemm_finished_addr + 32, bits_1 >> 20 & 1);
                                        mbarrier_wait(gemm_finished_addr + 40, bits_1 >> 21 & 1);
                                        bits_1 = bits_1 ^ 128;
                                    }
                                    phase_bits_5 = bits_1;
                                    int ring_2 = 0;
                                    #pragma unroll 1
                                    for (int idx_2 = 0; idx_2 < iterations_1; idx_2++) {
                                        mbarrier_wait(gemm_finished_addr + (ring_2) * 8, phase_bits_5 >> (unsigned int)(16 + ring_2) & 1);
                                        if (idx_2 < intermediate / 64) {
                                            asm volatile(
                                                "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                                " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                                :: "r"(a_nt_addr + (unsigned int)(ring_2 * 16384)), "l"((&dg_s)), "r"(0), "r"(x_1 * 256 + cta_rank_0 * 128), "r"(idx_2), "r"(0), "r"(0),
                                                   "r"(((gemm_arrived_addr + (ring_2) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                            asm volatile(
                                                "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                                " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                                :: "r"(b_ab_addr + (unsigned int)(ring_2 * 16384)), "l"((&wg_s)), "r"(0), "r"(idx_2 * 64), "r"(y_1 * 2 * 4 + cta_rank_0 * 2), "r"(expert_1), "r"(0),
                                                   "r"(((gemm_arrived_addr + (ring_2) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                            asm volatile(
                                                "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                                " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                                :: "r"(b_ab_hi_addr + (unsigned int)(ring_2 * 16384)), "l"((&wg_s)), "r"(0), "r"(idx_2 * 64), "r"((y_1 * 2 + 1) * 4 + cta_rank_0 * 2), "r"(expert_1), "r"(0),
                                                   "r"(((gemm_arrived_addr + (ring_2) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                        } else {
                                            asm volatile(
                                                "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                                " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                                :: "r"(a_nt_addr + (unsigned int)(ring_2 * 16384)), "l"((&du_s)), "r"(0), "r"(x_1 * 256 + cta_rank_0 * 128), "r"(idx_2 - intermediate / 64), "r"(0), "r"(0),
                                                   "r"(((gemm_arrived_addr + (ring_2) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                            asm volatile(
                                                "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                                " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                                :: "r"(b_ab_addr + (unsigned int)(ring_2 * 16384)), "l"((&wu_s)), "r"(0), "r"((idx_2 - intermediate / 64) * 64), "r"(y_1 * 2 * 4 + cta_rank_0 * 2), "r"(expert_1), "r"(0),
                                                   "r"(((gemm_arrived_addr + (ring_2) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                            asm volatile(
                                                "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                                " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                                :: "r"(b_ab_hi_addr + (unsigned int)(ring_2 * 16384)), "l"((&wu_s)), "r"(0), "r"((idx_2 - intermediate / 64) * 64), "r"((y_1 * 2 + 1) * 4 + cta_rank_0 * 2), "r"(expert_1), "r"(0),
                                                   "r"(((gemm_arrived_addr + (ring_2) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                        }
                                        phase_bits_5 = phase_bits_5 ^ (unsigned int)(1 << 16 + ring_2);
                                        ring_2 = (ring_2 + 1) % 4;
                                    }
                                }
                            }
                        } else {
                            if (tid / 32 == 4 && cta_rank_0 == 0) {
                                if (warp == 4) {
                                    if (elect_sync()) {
                                        int ring_3 = 0;
                                        mbarrier_wait(output_finished_addr, phase_bits_5 >> 22 & 1);
                                        phase_bits_5 = phase_bits_5 ^ 4194304;
                                        asm volatile("tcgen05.fence::after_thread_sync;");
                                        #pragma unroll 1
                                        for (int idx_3 = 0; idx_3 < iterations_1; idx_3++) {
                                            mbarrier_arrive_expect_tx(gemm_arrived_addr + (ring_3) * 8, 98304);
                                            mbarrier_wait(gemm_arrived_addr + (ring_3) * 8, phase_bits_5 >> (unsigned int)ring_3 & 1);
                                            int _mma_a_lo_2 = (((a_nt_addr) >> 4) & 0x3FFF) + (ring_3) * 1024;
                                            int _mma_b_lo_2 = ((((b_ab_addr) >> 4) & 0x3FFF) | 0x2000000) + (ring_3) * 1024;
                                            asm volatile(
            "{\n\t"
            ".reg .pred p0, p1;\n\t"
            ".reg .b32 adhi, bdhi, alo, blo, id, m0, m1, m2, m3, m4, m5, m6, m7;\n\t"
            ".reg .b64 da, db;\n\t"
            ""
            "setp.ne.b32 p0, %3, 0;\n\t"
            "setp.ne.b32 p1, 1, 0;\n\t"
            "mov.b32 m0, 0; mov.b32 m1, 0; mov.b32 m2, 0; mov.b32 m3, 0;\n\tmov.b32 m4, 0; mov.b32 m5, 0; mov.b32 m6, 0; mov.b32 m7, 0;\n\t"
            "mov.b32 adhi, 0x40004040;\n\t"
            "mov.b32 bdhi, 0x40004040;\n\t"
            "mov.b32 id, 272696464;\n\t"
            "mov.b32 alo, %0;\n\t"
            "mov.b32 blo, %1;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
            "add.u32 alo, alo, 2;\n\t"
            "add.u32 blo, blo, 128;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
            "add.u32 alo, alo, 2;\n\t"
            "add.u32 blo, blo, 128;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
            "add.u32 alo, alo, 2;\n\t"
            "add.u32 blo, blo, 128;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
            "}\n"
            :: "r"(_mma_a_lo_2), "r"(_mma_b_lo_2), "r"(tmem_accumulator), "r"(((idx_3 == 0) ? 0 : 1)));
                                            int _mma_a_lo_3 = (((a_nt_addr) >> 4) & 0x3FFF) + (ring_3) * 1024;
                                            int _mma_b_lo_3 = ((((b_ab_hi_addr) >> 4) & 0x3FFF) | 0x2000000) + (ring_3) * 1024;
                                            asm volatile(
            "{\n\t"
            ".reg .pred p0, p1;\n\t"
            ".reg .b32 adhi, bdhi, alo, blo, id, m0, m1, m2, m3, m4, m5, m6, m7;\n\t"
            ".reg .b64 da, db;\n\t"
            ""
            "setp.ne.b32 p0, %3, 0;\n\t"
            "setp.ne.b32 p1, 1, 0;\n\t"
            "mov.b32 m0, 0; mov.b32 m1, 0; mov.b32 m2, 0; mov.b32 m3, 0;\n\tmov.b32 m4, 0; mov.b32 m5, 0; mov.b32 m6, 0; mov.b32 m7, 0;\n\t"
            "mov.b32 adhi, 0x40004040;\n\t"
            "mov.b32 bdhi, 0x40004040;\n\t"
            "mov.b32 id, 272696464;\n\t"
            "mov.b32 alo, %0;\n\t"
            "mov.b32 blo, %1;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
            "add.u32 alo, alo, 2;\n\t"
            "add.u32 blo, blo, 128;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
            "add.u32 alo, alo, 2;\n\t"
            "add.u32 blo, blo, 128;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
            "add.u32 alo, alo, 2;\n\t"
            "add.u32 blo, blo, 128;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
            "}\n"
            :: "r"(_mma_a_lo_3), "r"(_mma_b_lo_3), "r"((tmem_accumulator + (256))), "r"(((idx_3 == 0) ? 0 : 1)));
                                            tcgen05_commit_cg2_multicast(gemm_finished_addr + (ring_3) * 8, (uint16_t)(3));
                                            phase_bits_5 = phase_bits_5 ^ (unsigned int)(1 << ring_3);
                                            ring_3 = (ring_3 + 1) % 4;
                                        }
                                        tcgen05_commit_cg2_multicast(output_arrived_addr, (uint16_t)(3));
                                    }
                                }
                            } else if (tid < 128) {
                                mbarrier_wait(output_arrived_addr, phase_bits_5 >> 6 & 1);
                                phase_bits_5 = phase_bits_5 ^ 64;
                                unsigned int packed_6[128];
                                #pragma unroll
                                for (int chunk_4 = 0; chunk_4 < 8; chunk_4++) {
                                    #pragma unroll
                                    for (int sub_2 = 0; sub_2 < 2; sub_2++) {
                                        unsigned int address_9 = taddr_1 + (unsigned int)(tid / 32 * 32 + sub_2 * 16 << 16) + (unsigned int)(chunk_4 * 32);
                                        float _tmem_load_2[16];
                                        asm volatile(
                                            "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                            : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[15]))
                                            : "r"(address_9));
                                        #pragma unroll
                                        for (int pair_7 = 0; pair_7 < 8; pair_7++) {
                                            __nv_bfloat162 _bf16x2_24 = __float22bfloat162_rn(make_float2(_tmem_load_2[pair_7 * 2], _tmem_load_2[pair_7 * 2 + 1]));
                                            packed_6[chunk_4 * 16 + sub_2 * 8 + pair_7] = __as_u32(_bf16x2_24);
                                        }
                                    }
                                }
                                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                                int last_2 = 1;
                                last_2 = 1 - has_hi_1;
                                if (last_2 != 0) {
                                    asm volatile("barrier.sync 1, 128;" ::: "memory");
                                    if (tid == 0) {
                                        asm volatile(
                                            "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                                            :: "r"((output_finished_addr) & 0xFEFFFFFF) : "memory");
                                    }
                                }
                                if (tid == 0) {
                                    int previous_offset_4 = macro_size;
                                    int output_row_1 = x_1 * 256 + cta_rank_0 * 128;
                                    int _min_33 = ((macro_size) < (tokens - previous_offset_4) ? (macro_size) : (tokens - previous_offset_4));
                                    if (output_row_1 < _min_33) {
                                    }
                                }
                                #pragma unroll
                                for (int chunk_5 = 0; chunk_5 < 8; chunk_5++) {
                                    if (tid == 0) {
                                        asm volatile("cp.async.bulk.wait_group.read 2;");
                                    }
                                    asm volatile("barrier.sync 1, 128;" ::: "memory");
                                    int warp_0_3 = tid / 32;
                                    int lane_4 = tid % 32;
                                    #pragma unroll
                                    for (int half_2 = 0; half_2 < 2; half_2++) {
                                        #pragma unroll
                                        for (int col_tile_14 = 0; col_tile_14 < 2; col_tile_14++) {
                                            int row_12 = warp_0_3 * 32 + half_2 * 16 + lane_4 % 16;
                                            int col_55 = col_tile_14 * 16 + lane_4 / 16 * 8;
                                            unsigned int address_10 = d_smem_addr + (unsigned int)(chunk_5 % 3 * 8192) + (unsigned int)((row_12 * 32 + col_55) * 2);
                                            address_10 = address_10 ^ (address_10 & 511) >> 7 << 4;
                                            int offset_4 = chunk_5 * 16 + half_2 * 8 + col_tile_14 * 4;
                                            uint32_t _stmatrix_addr_5 = static_cast<uint32_t>(address_10);
                                            asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                                :: "r"(_stmatrix_addr_5), "r"(*reinterpret_cast<const uint32_t*>(&packed_6[offset_4])), "r"(*reinterpret_cast<const uint32_t*>(&packed_6[offset_4 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&packed_6[offset_4 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&packed_6[offset_4 + 3]))
                                                : "memory");
                                        }
                                    }
                                    asm volatile("barrier.sync 1, 128;" ::: "memory");
                                    if (tid == 0) {
                                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                        asm volatile(
                                            "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                            " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                            :: "l"((&dx_s)), "r"(0), "r"(x_1 * 256 + cta_rank_0 * 128), "r"(y_1 * 2 * 8 + chunk_5), "r"(0), "r"(0), "r"(d_smem_addr + (unsigned int)(chunk_5 % 3 * 8192)), "l"(0x12F0000000000000ULL) : "memory");
                                        asm volatile("cp.async.bulk.commit_group;");
                                    }
                                }
                                if (has_hi_1 != 0) {
                                    unsigned int packed_0_1[128];
                                    #pragma unroll
                                    for (int chunk_6 = 0; chunk_6 < 8; chunk_6++) {
                                        #pragma unroll
                                        for (int sub_3 = 0; sub_3 < 2; sub_3++) {
                                            unsigned int address_11 = taddr_1 + (unsigned int)(tid / 32 * 32 + sub_3 * 16 << 16) + 256 + (unsigned int)(chunk_6 * 32);
                                            float _tmem_load_3[16];
                                            asm volatile(
                                                "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[15]))
                                                : "r"(address_11));
                                            #pragma unroll
                                            for (int pair_8 = 0; pair_8 < 8; pair_8++) {
                                                __nv_bfloat162 _bf16x2_25 = __float22bfloat162_rn(make_float2(_tmem_load_3[pair_8 * 2], _tmem_load_3[pair_8 * 2 + 1]));
                                                packed_0_1[chunk_6 * 16 + sub_3 * 8 + pair_8] = __as_u32(_bf16x2_25);
                                            }
                                        }
                                    }
                                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                                    int last_1_1 = 1;
                                    if (last_1_1 != 0) {
                                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                                        if (tid == 0) {
                                            asm volatile(
                                                "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                                                :: "r"((output_finished_addr) & 0xFEFFFFFF) : "memory");
                                        }
                                    }
                                    #pragma unroll
                                    for (int chunk_7 = 0; chunk_7 < 8; chunk_7++) {
                                        if (tid == 0) {
                                            asm volatile("cp.async.bulk.wait_group.read 2;");
                                        }
                                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                                        int warp_0_4 = tid / 32;
                                        int lane_5 = tid % 32;
                                        #pragma unroll
                                        for (int half_3 = 0; half_3 < 2; half_3++) {
                                            #pragma unroll
                                            for (int col_tile_15 = 0; col_tile_15 < 2; col_tile_15++) {
                                                int row_13 = warp_0_4 * 32 + half_3 * 16 + lane_5 % 16;
                                                int col_56 = col_tile_15 * 16 + lane_5 / 16 * 8;
                                                unsigned int address_12 = d_smem_addr + (unsigned int)((8 + chunk_7) % 3 * 8192) + (unsigned int)((row_13 * 32 + col_56) * 2);
                                                address_12 = address_12 ^ (address_12 & 511) >> 7 << 4;
                                                int offset_5 = chunk_7 * 16 + half_3 * 8 + col_tile_15 * 4;
                                                uint32_t _stmatrix_addr_6 = static_cast<uint32_t>(address_12);
                                                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                                    :: "r"(_stmatrix_addr_6), "r"(*reinterpret_cast<const uint32_t*>(&packed_0_1[offset_5])), "r"(*reinterpret_cast<const uint32_t*>(&packed_0_1[offset_5 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&packed_0_1[offset_5 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&packed_0_1[offset_5 + 3]))
                                                    : "memory");
                                            }
                                        }
                                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                                        if (tid == 0) {
                                            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                            asm volatile(
                                                "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                                " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                                :: "l"((&dx_s)), "r"(0), "r"(x_1 * 256 + cta_rank_0 * 128), "r"((y_1 * 2 + 1) * 8 + chunk_7), "r"(0), "r"(0), "r"(d_smem_addr + (unsigned int)((8 + chunk_7) % 3 * 8192)), "l"(0x12F0000000000000ULL) : "memory");
                                            asm volatile("cp.async.bulk.commit_group;");
                                        }
                                    }
                                }
                                if (tid == 0) {
                                    asm volatile("cp.async.bulk.wait_group.read 0;");
                                }
                                asm volatile("barrier.sync 4, 128;" ::: "memory");
                                if (tid / 32 == 0) {
                                    if (warp == 0) {
                                        if (elect_sync()) {
                                            if (has_hi_1 != 0) {
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        gemm_phase = phase_bits_5;
                    } else if (kind == 3) {
                        int col_blocks_6 = (local_tokens + 512 - 1) / 512;
                        {
                            col_blocks_6 = (intermediate + 512 - 1) / 512;
                        }
                        int x_2 = -1;
                        int y_2 = -1;
                        int expert_2 = -1;
                        int k_start_2 = 0;
                        int k_end_2 = 0;
                        int first_2 = 0;
                        int row_blocks_2 = hidden / 256;
                        int expert_idx = 0;
                        int local_task = task_3;
                        int _max_5 = ((row_blocks_2 * col_blocks_6) > (intermediate / 256 * ((hidden + 512 - 1) / 512)) ? (row_blocks_2 * col_blocks_6) : (intermediate / 256 * ((hidden + 512 - 1) / 512)));
                        int stride = _max_5;
                        k_end_2 = local_tokens;
                        first_2 = 1;
                        if (k_start_2 < k_end_2 && local_task < row_blocks_2 * col_blocks_6) {
                            int supergroup_2 = local_task / (row_blocks_2 * 8);
                            int full_cols_2 = col_blocks_6 / 8 * 8;
                            int row_14 = 0;
                            int col_57 = 0;
                            if (local_task < row_blocks_2 * full_cols_2) {
                                row_14 = local_task % (row_blocks_2 * 8) / 8;
                                col_57 = supergroup_2 * 8 + local_task % 8;
                            } else {
                                row_14 = (local_task - row_blocks_2 * full_cols_2) / (col_blocks_6 - full_cols_2);
                                col_57 = full_cols_2 + (local_task - row_blocks_2 * full_cols_2) % (col_blocks_6 - full_cols_2);
                            }
                            if ((supergroup_2 & 1) != 0) {
                                row_14 = row_blocks_2 - row_14 - 1;
                            }
                            x_2 = row_14;
                            y_2 = col_57;
                            expert_2 = expert_idx;
                        }
                        unsigned int phase_bits_6 = gemm_phase;
                        int has_hi_2 = 0;
                        has_hi_2 = (int)((y_2 * 2 + 1) * 256 < intermediate);
                        int global_mini_2 = 0;
                        int macro_rows_2 = 0;
                        int iterations_2 = hidden / 64;
                        iterations_2 = (k_end_2 - k_start_2) / 64;
                        if (expert_2 < 0) {
                            if (tid == 0) {
                            }
                        } else if (tid / 32 == 7) {
                            if (warp == 7) {
                                if (elect_sync()) {
                                    unsigned int previous_2 = phase_bits_6 >> 7 & 1;
                                    unsigned int bits_2 = phase_bits_6;
                                    if (previous_2 != 0) {
                                        mbarrier_wait(gemm_finished_addr, bits_2 >> 16 & 1);
                                        mbarrier_wait(gemm_finished_addr + 8, bits_2 >> 17 & 1);
                                        mbarrier_wait(gemm_finished_addr + 16, bits_2 >> 18 & 1);
                                        mbarrier_wait(gemm_finished_addr + 24, bits_2 >> 19 & 1);
                                        mbarrier_wait(gemm_finished_addr + 32, bits_2 >> 20 & 1);
                                        mbarrier_wait(gemm_finished_addr + 40, bits_2 >> 21 & 1);
                                        bits_2 = bits_2 ^ 128;
                                    }
                                    phase_bits_6 = bits_2;
                                    int ring_4 = 0;
                                    #pragma unroll 1
                                    for (int idx_4 = 0; idx_4 < iterations_2; idx_4++) {
                                        int token_row = k_start_2 + idx_4 * 64;
                                        if (idx_4 == 0 || token_row % 256 == 0) {
                                        }
                                        if (idx_4 == 0 || token_row % mini_size == 0) {
                                            int input_mini = token_row / mini_size;
                                            int _min_35 = ((mini_size) < (tokens - input_mini * mini_size) ? (mini_size) : (tokens - input_mini * mini_size));
                                            int input_rows = _min_35;
                                            int input_count = (input_rows + 127) / 128 * ((hidden + 511) / 512);
                                        }
                                        mbarrier_wait(gemm_finished_addr + (ring_4) * 8, phase_bits_6 >> (unsigned int)(16 + ring_4) & 1);
                                        int local_row = k_start_2 + idx_4 * 64;
                                        asm volatile(
                                            "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                            " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                            :: "r"(a_atb_addr + (unsigned int)(ring_4 * 16384)), "l"((&dy_atb_s)), "r"(0), "r"(local_row), "r"(x_2 * 4 + cta_rank_0 * 2), "r"(0), "r"(0),
                                               "r"(((gemm_arrived_addr + (ring_4) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                        asm volatile(
                                            "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                            " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                            :: "r"(b_ab_addr + (unsigned int)(ring_4 * 16384)), "l"((&h_atb_s)), "r"(0), "r"(local_row), "r"(y_2 * 2 * 4 + cta_rank_0 * 2), "r"(0), "r"(0),
                                               "r"(((gemm_arrived_addr + (ring_4) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                        asm volatile(
                                            "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                            " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                            :: "r"(b_ab_hi_addr + (unsigned int)(ring_4 * 16384)), "l"((&h_atb_s)), "r"(0), "r"(local_row), "r"((y_2 * 2 + 1) * 4 + cta_rank_0 * 2), "r"(0), "r"(0),
                                               "r"(((gemm_arrived_addr + (ring_4) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                        phase_bits_6 = phase_bits_6 ^ (unsigned int)(1 << 16 + ring_4);
                                        ring_4 = (ring_4 + 1) % 4;
                                    }
                                }
                            }
                        } else {
                            if (tid / 32 == 4 && cta_rank_0 == 0) {
                                if (warp == 4) {
                                    if (elect_sync()) {
                                        int ring_5 = 0;
                                        mbarrier_wait(output_finished_addr, phase_bits_6 >> 22 & 1);
                                        phase_bits_6 = phase_bits_6 ^ 4194304;
                                        asm volatile("tcgen05.fence::after_thread_sync;");
                                        #pragma unroll 1
                                        for (int idx_5 = 0; idx_5 < iterations_2; idx_5++) {
                                            mbarrier_arrive_expect_tx(gemm_arrived_addr + (ring_5) * 8, 98304);
                                            mbarrier_wait(gemm_arrived_addr + (ring_5) * 8, phase_bits_6 >> (unsigned int)ring_5 & 1);
                                            int _mma_a_lo_4 = ((((a_atb_addr) >> 4) & 0x3FFF) | 0x2000000) + (ring_5) * 1024;
                                            int _mma_b_lo_4 = ((((b_ab_addr) >> 4) & 0x3FFF) | 0x2000000) + (ring_5) * 1024;
                                            asm volatile(
            "{\n\t"
            ".reg .pred p0, p1;\n\t"
            ".reg .b32 adhi, bdhi, alo, blo, id, m0, m1, m2, m3, m4, m5, m6, m7;\n\t"
            ".reg .b64 da, db;\n\t"
            ""
            "setp.ne.b32 p0, %3, 0;\n\t"
            "setp.ne.b32 p1, 1, 0;\n\t"
            "mov.b32 m0, 0; mov.b32 m1, 0; mov.b32 m2, 0; mov.b32 m3, 0;\n\tmov.b32 m4, 0; mov.b32 m5, 0; mov.b32 m6, 0; mov.b32 m7, 0;\n\t"
            "mov.b32 adhi, 0x40004040;\n\t"
            "mov.b32 bdhi, 0x40004040;\n\t"
            "mov.b32 id, 272729232;\n\t"
            "mov.b32 alo, %0;\n\t"
            "mov.b32 blo, %1;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
            "add.u32 alo, alo, 128;\n\t"
            "add.u32 blo, blo, 128;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
            "add.u32 alo, alo, 128;\n\t"
            "add.u32 blo, blo, 128;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
            "add.u32 alo, alo, 128;\n\t"
            "add.u32 blo, blo, 128;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
            "}\n"
            :: "r"(_mma_a_lo_4), "r"(_mma_b_lo_4), "r"(tmem_accumulator), "r"(((idx_5 == 0) ? 0 : 1)));
                                            int _mma_a_lo_5 = ((((a_atb_addr) >> 4) & 0x3FFF) | 0x2000000) + (ring_5) * 1024;
                                            int _mma_b_lo_5 = ((((b_ab_hi_addr) >> 4) & 0x3FFF) | 0x2000000) + (ring_5) * 1024;
                                            asm volatile(
            "{\n\t"
            ".reg .pred p0, p1;\n\t"
            ".reg .b32 adhi, bdhi, alo, blo, id, m0, m1, m2, m3, m4, m5, m6, m7;\n\t"
            ".reg .b64 da, db;\n\t"
            ""
            "setp.ne.b32 p0, %3, 0;\n\t"
            "setp.ne.b32 p1, 1, 0;\n\t"
            "mov.b32 m0, 0; mov.b32 m1, 0; mov.b32 m2, 0; mov.b32 m3, 0;\n\tmov.b32 m4, 0; mov.b32 m5, 0; mov.b32 m6, 0; mov.b32 m7, 0;\n\t"
            "mov.b32 adhi, 0x40004040;\n\t"
            "mov.b32 bdhi, 0x40004040;\n\t"
            "mov.b32 id, 272729232;\n\t"
            "mov.b32 alo, %0;\n\t"
            "mov.b32 blo, %1;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
            "add.u32 alo, alo, 128;\n\t"
            "add.u32 blo, blo, 128;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
            "add.u32 alo, alo, 128;\n\t"
            "add.u32 blo, blo, 128;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
            "add.u32 alo, alo, 128;\n\t"
            "add.u32 blo, blo, 128;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
            "}\n"
            :: "r"(_mma_a_lo_5), "r"(_mma_b_lo_5), "r"((tmem_accumulator + (256))), "r"(((idx_5 == 0) ? 0 : 1)));
                                            tcgen05_commit_cg2_multicast(gemm_finished_addr + (ring_5) * 8, (uint16_t)(3));
                                            phase_bits_6 = phase_bits_6 ^ (unsigned int)(1 << ring_5);
                                            ring_5 = (ring_5 + 1) % 4;
                                        }
                                        tcgen05_commit_cg2_multicast(output_arrived_addr, (uint16_t)(3));
                                    }
                                }
                            } else if (tid < 128) {
                                mbarrier_wait(output_arrived_addr, phase_bits_6 >> 6 & 1);
                                phase_bits_6 = phase_bits_6 ^ 64;
                                int warp_row = tid / 32 * 32;
                                #pragma unroll
                                for (int chunk_8 = 0; chunk_8 < 16; chunk_8++) {
                                    float _tmem_load_4[16];
                                    tmem_ld_x16(&_tmem_load_4[0], taddr_1 + (unsigned int)(warp_row << 16) + (unsigned int)(chunk_8 * 16));
                                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                                    if (tid == 0) {
                                        asm volatile("cp.async.bulk.wait_group.read 2;");
                                    }
                                    asm volatile("barrier.sync 1, 128;" ::: "memory");
                                    #pragma unroll
                                    for (int vec_3 = 0; vec_3 < 4; vec_3++) {
                                        unsigned int address_13 = d_smem_addr + (unsigned int)(chunk_8 % 3 * 8192) + (unsigned int)(tid * 64) + (unsigned int)(vec_3 * 16);
                                        address_13 = address_13 ^ (address_13 & 511) >> 7 << 4;
                                        asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(d_words_addr + (address_13 - d_words_addr)), "f"(_tmem_load_4[vec_3 * 4]), "f"(_tmem_load_4[vec_3 * 4 + 1]), "f"(_tmem_load_4[vec_3 * 4 + 2]), "f"(_tmem_load_4[vec_3 * 4 + 3]) : "memory");
                                    }
                                    asm volatile("barrier.sync 1, 128;" ::: "memory");
                                    if (tid == 0) {
                                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 900)
                                        #error "TmaReduceAdd5d requires SM90 or newer"
                                        #endif
                                        asm volatile(
                                            "cp.reduce.async.bulk.tensor.5d.global.shared::cta.add.tile.bulk_group.L2::cache_hint"
                                            " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                            :: "l"((&dwd_s)), "r"(0), "r"(x_2 * 256 + cta_rank_0 * 128), "r"(y_2 * 2 * 16 + chunk_8), "r"(expert_2), "r"(0), "r"(d_smem_addr + (unsigned int)(chunk_8 % 3 * 8192)), "l"(0x12F0000000000000ULL) : "memory");
                                        asm volatile("cp.async.bulk.commit_group;");
                                    }
                                }
                                if (has_hi_2 != 0) {
                                    #pragma unroll
                                    for (int chunk_9 = 0; chunk_9 < 16; chunk_9++) {
                                        float _tmem_load_5[16];
                                        tmem_ld_x16(&_tmem_load_5[0], taddr_1 + (unsigned int)(warp_row << 16) + 256 + (unsigned int)(chunk_9 * 16));
                                        asm volatile("tcgen05.wait::ld.sync.aligned;");
                                        if (tid == 0) {
                                            asm volatile("cp.async.bulk.wait_group.read 2;");
                                        }
                                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                                        #pragma unroll
                                        for (int vec_4 = 0; vec_4 < 4; vec_4++) {
                                            unsigned int address_14 = d_smem_addr + (unsigned int)((16 + chunk_9) % 3 * 8192) + (unsigned int)(tid * 64) + (unsigned int)(vec_4 * 16);
                                            address_14 = address_14 ^ (address_14 & 511) >> 7 << 4;
                                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(d_words_addr + (address_14 - d_words_addr)), "f"(_tmem_load_5[vec_4 * 4]), "f"(_tmem_load_5[vec_4 * 4 + 1]), "f"(_tmem_load_5[vec_4 * 4 + 2]), "f"(_tmem_load_5[vec_4 * 4 + 3]) : "memory");
                                        }
                                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                                        if (tid == 0) {
                                            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 900)
                                            #error "TmaReduceAdd5d requires SM90 or newer"
                                            #endif
                                            asm volatile(
                                                "cp.reduce.async.bulk.tensor.5d.global.shared::cta.add.tile.bulk_group.L2::cache_hint"
                                                " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                                :: "l"((&dwd_s)), "r"(0), "r"(x_2 * 256 + cta_rank_0 * 128), "r"((y_2 * 2 + 1) * 16 + chunk_9), "r"(expert_2), "r"(0), "r"(d_smem_addr + (unsigned int)((16 + chunk_9) % 3 * 8192)), "l"(0x12F0000000000000ULL) : "memory");
                                            asm volatile("cp.async.bulk.commit_group;");
                                        }
                                    }
                                }
                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                if (tid == 0) {
                                    asm volatile(
                                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                                        :: "r"((output_finished_addr) & 0xFEFFFFFF) : "memory");
                                    asm volatile("cp.async.bulk.wait_group.read 0;");
                                }
                                asm volatile("barrier.sync 4, 128;" ::: "memory");
                                if (tid / 32 == 0) {
                                    if (warp == 0) {
                                        if (elect_sync()) {
                                            if (has_hi_2 != 0) {
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        gemm_phase = phase_bits_6;
                    } else {
                        if (kind == 4) {
                            int col_blocks_7 = (local_tokens + 512 - 1) / 512;
                            {
                                col_blocks_7 = (hidden + 512 - 1) / 512;
                            }
                            int x_3 = -1;
                            int y_3 = -1;
                            int expert_3 = -1;
                            int k_start_3 = 0;
                            int k_end_3 = 0;
                            int first_3 = 0;
                            int row_blocks_3 = intermediate / 256;
                            int expert_idx_1 = 0;
                            int local_task_1 = task_3;
                            int _max_7 = ((row_blocks_3 * col_blocks_7) > (hidden / 256 * ((intermediate + 512 - 1) / 512)) ? (row_blocks_3 * col_blocks_7) : (hidden / 256 * ((intermediate + 512 - 1) / 512)));
                            int stride_1 = _max_7;
                            k_end_3 = local_tokens;
                            first_3 = 1;
                            if (k_start_3 < k_end_3 && local_task_1 < row_blocks_3 * col_blocks_7) {
                                int supergroup_3 = local_task_1 / (row_blocks_3 * 8);
                                int full_cols_3 = col_blocks_7 / 8 * 8;
                                int row_15 = 0;
                                int col_58 = 0;
                                if (local_task_1 < row_blocks_3 * full_cols_3) {
                                    row_15 = local_task_1 % (row_blocks_3 * 8) / 8;
                                    col_58 = supergroup_3 * 8 + local_task_1 % 8;
                                } else {
                                    row_15 = (local_task_1 - row_blocks_3 * full_cols_3) / (col_blocks_7 - full_cols_3);
                                    col_58 = full_cols_3 + (local_task_1 - row_blocks_3 * full_cols_3) % (col_blocks_7 - full_cols_3);
                                }
                                if ((supergroup_3 & 1) != 0) {
                                    row_15 = row_blocks_3 - row_15 - 1;
                                }
                                x_3 = row_15;
                                y_3 = col_58;
                                expert_3 = expert_idx_1;
                            }
                            unsigned int phase_bits_7 = gemm_phase;
                            int has_hi_3 = 0;
                            has_hi_3 = (int)((y_3 * 2 + 1) * 256 < hidden);
                            int global_mini_3 = 0;
                            int macro_rows_3 = 0;
                            int iterations_3 = intermediate / 64;
                            iterations_3 = (k_end_3 - k_start_3) / 64;
                            if (expert_3 < 0) {
                                if (tid == 0) {
                                }
                            } else if (tid / 32 == 7) {
                                if (warp == 7) {
                                    if (elect_sync()) {
                                        unsigned int previous_3 = phase_bits_7 >> 7 & 1;
                                        unsigned int bits_3 = phase_bits_7;
                                        if (previous_3 != 0) {
                                            mbarrier_wait(gemm_finished_addr, bits_3 >> 16 & 1);
                                            mbarrier_wait(gemm_finished_addr + 8, bits_3 >> 17 & 1);
                                            mbarrier_wait(gemm_finished_addr + 16, bits_3 >> 18 & 1);
                                            mbarrier_wait(gemm_finished_addr + 24, bits_3 >> 19 & 1);
                                            mbarrier_wait(gemm_finished_addr + 32, bits_3 >> 20 & 1);
                                            mbarrier_wait(gemm_finished_addr + 40, bits_3 >> 21 & 1);
                                            bits_3 = bits_3 ^ 128;
                                        }
                                        phase_bits_7 = bits_3;
                                        int ring_6 = 0;
                                        #pragma unroll 1
                                        for (int idx_6 = 0; idx_6 < iterations_3; idx_6++) {
                                            int token_row_1 = k_start_3 + idx_6 * 64;
                                            if (idx_6 == 0 || token_row_1 % 256 == 0) {
                                                bool enabled_value_4 = 1;
                                                if (enabled_value_4 != 0) {
                                                    int32_t _relaxed_ld_16;
                                                    asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_16) : "l"(dg_ready + (token_row_1 / 256)) : "memory");
                                                    int value_9 = _relaxed_ld_16;
                                                    while (value_9 < row_count) {
                                                        asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                                        int32_t _relaxed_ld_17;
                                                        asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_17) : "l"(dg_ready + (token_row_1 / 256)) : "memory");
                                                        value_9 = _relaxed_ld_17;
                                                    }
                                                    asm volatile("fence.acquire.gpu;" ::: "memory");
                                                }
                                            }
                                            if (idx_6 == 0 || token_row_1 % mini_size == 0) {
                                                int input_mini_1 = token_row_1 / mini_size;
                                                int _min_37 = ((mini_size) < (tokens - input_mini_1 * mini_size) ? (mini_size) : (tokens - input_mini_1 * mini_size));
                                                int input_rows_1 = _min_37;
                                                int input_count_1 = (input_rows_1 + 127) / 128 * ((hidden + 511) / 512);
                                            }
                                            mbarrier_wait(gemm_finished_addr + (ring_6) * 8, phase_bits_7 >> (unsigned int)(16 + ring_6) & 1);
                                            int local_row_1 = k_start_3 + idx_6 * 64;
                                            asm volatile(
                                                "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                                " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                                :: "r"(a_atb_addr + (unsigned int)(ring_6 * 16384)), "l"((&dg_atb_s)), "r"(0), "r"(local_row_1), "r"(x_3 * 4 + cta_rank_0 * 2), "r"(0), "r"(0),
                                                   "r"(((gemm_arrived_addr + (ring_6) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                            asm volatile(
                                                "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                                " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                                :: "r"(b_ab_addr + (unsigned int)(ring_6 * 16384)), "l"((&x_atb_s)), "r"(0), "r"(local_row_1), "r"(y_3 * 2 * 4 + cta_rank_0 * 2), "r"(0), "r"(0),
                                                   "r"(((gemm_arrived_addr + (ring_6) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                            asm volatile(
                                                "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                                " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                                :: "r"(b_ab_hi_addr + (unsigned int)(ring_6 * 16384)), "l"((&x_atb_s)), "r"(0), "r"(local_row_1), "r"((y_3 * 2 + 1) * 4 + cta_rank_0 * 2), "r"(0), "r"(0),
                                                   "r"(((gemm_arrived_addr + (ring_6) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                            phase_bits_7 = phase_bits_7 ^ (unsigned int)(1 << 16 + ring_6);
                                            ring_6 = (ring_6 + 1) % 4;
                                        }
                                    }
                                }
                            } else {
                                if (tid / 32 == 4 && cta_rank_0 == 0) {
                                    if (warp == 4) {
                                        if (elect_sync()) {
                                            int ring_7 = 0;
                                            mbarrier_wait(output_finished_addr, phase_bits_7 >> 22 & 1);
                                            phase_bits_7 = phase_bits_7 ^ 4194304;
                                            asm volatile("tcgen05.fence::after_thread_sync;");
                                            #pragma unroll 1
                                            for (int idx_7 = 0; idx_7 < iterations_3; idx_7++) {
                                                mbarrier_arrive_expect_tx(gemm_arrived_addr + (ring_7) * 8, 98304);
                                                mbarrier_wait(gemm_arrived_addr + (ring_7) * 8, phase_bits_7 >> (unsigned int)ring_7 & 1);
                                                int _mma_a_lo_6 = ((((a_atb_addr) >> 4) & 0x3FFF) | 0x2000000) + (ring_7) * 1024;
                                                int _mma_b_lo_6 = ((((b_ab_addr) >> 4) & 0x3FFF) | 0x2000000) + (ring_7) * 1024;
                                                asm volatile(
            "{\n\t"
            ".reg .pred p0, p1;\n\t"
            ".reg .b32 adhi, bdhi, alo, blo, id, m0, m1, m2, m3, m4, m5, m6, m7;\n\t"
            ".reg .b64 da, db;\n\t"
            ""
            "setp.ne.b32 p0, %3, 0;\n\t"
            "setp.ne.b32 p1, 1, 0;\n\t"
            "mov.b32 m0, 0; mov.b32 m1, 0; mov.b32 m2, 0; mov.b32 m3, 0;\n\tmov.b32 m4, 0; mov.b32 m5, 0; mov.b32 m6, 0; mov.b32 m7, 0;\n\t"
            "mov.b32 adhi, 0x40004040;\n\t"
            "mov.b32 bdhi, 0x40004040;\n\t"
            "mov.b32 id, 272729232;\n\t"
            "mov.b32 alo, %0;\n\t"
            "mov.b32 blo, %1;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
            "add.u32 alo, alo, 128;\n\t"
            "add.u32 blo, blo, 128;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
            "add.u32 alo, alo, 128;\n\t"
            "add.u32 blo, blo, 128;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
            "add.u32 alo, alo, 128;\n\t"
            "add.u32 blo, blo, 128;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
            "}\n"
            :: "r"(_mma_a_lo_6), "r"(_mma_b_lo_6), "r"(tmem_accumulator), "r"(((idx_7 == 0) ? 0 : 1)));
                                                int _mma_a_lo_7 = ((((a_atb_addr) >> 4) & 0x3FFF) | 0x2000000) + (ring_7) * 1024;
                                                int _mma_b_lo_7 = ((((b_ab_hi_addr) >> 4) & 0x3FFF) | 0x2000000) + (ring_7) * 1024;
                                                asm volatile(
            "{\n\t"
            ".reg .pred p0, p1;\n\t"
            ".reg .b32 adhi, bdhi, alo, blo, id, m0, m1, m2, m3, m4, m5, m6, m7;\n\t"
            ".reg .b64 da, db;\n\t"
            ""
            "setp.ne.b32 p0, %3, 0;\n\t"
            "setp.ne.b32 p1, 1, 0;\n\t"
            "mov.b32 m0, 0; mov.b32 m1, 0; mov.b32 m2, 0; mov.b32 m3, 0;\n\tmov.b32 m4, 0; mov.b32 m5, 0; mov.b32 m6, 0; mov.b32 m7, 0;\n\t"
            "mov.b32 adhi, 0x40004040;\n\t"
            "mov.b32 bdhi, 0x40004040;\n\t"
            "mov.b32 id, 272729232;\n\t"
            "mov.b32 alo, %0;\n\t"
            "mov.b32 blo, %1;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
            "add.u32 alo, alo, 128;\n\t"
            "add.u32 blo, blo, 128;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
            "add.u32 alo, alo, 128;\n\t"
            "add.u32 blo, blo, 128;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
            "add.u32 alo, alo, 128;\n\t"
            "add.u32 blo, blo, 128;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
            "}\n"
            :: "r"(_mma_a_lo_7), "r"(_mma_b_lo_7), "r"((tmem_accumulator + (256))), "r"(((idx_7 == 0) ? 0 : 1)));
                                                tcgen05_commit_cg2_multicast(gemm_finished_addr + (ring_7) * 8, (uint16_t)(3));
                                                phase_bits_7 = phase_bits_7 ^ (unsigned int)(1 << ring_7);
                                                ring_7 = (ring_7 + 1) % 4;
                                            }
                                            tcgen05_commit_cg2_multicast(output_arrived_addr, (uint16_t)(3));
                                        }
                                    }
                                } else if (tid < 128) {
                                    mbarrier_wait(output_arrived_addr, phase_bits_7 >> 6 & 1);
                                    phase_bits_7 = phase_bits_7 ^ 64;
                                    int warp_row_1 = tid / 32 * 32;
                                    #pragma unroll
                                    for (int chunk_10 = 0; chunk_10 < 16; chunk_10++) {
                                        float _tmem_load_6[16];
                                        tmem_ld_x16(&_tmem_load_6[0], taddr_1 + (unsigned int)(warp_row_1 << 16) + (unsigned int)(chunk_10 * 16));
                                        asm volatile("tcgen05.wait::ld.sync.aligned;");
                                        if (tid == 0) {
                                            asm volatile("cp.async.bulk.wait_group.read 2;");
                                        }
                                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                                        #pragma unroll
                                        for (int vec_5 = 0; vec_5 < 4; vec_5++) {
                                            unsigned int address_15 = d_smem_addr + (unsigned int)(chunk_10 % 3 * 8192) + (unsigned int)(tid * 64) + (unsigned int)(vec_5 * 16);
                                            address_15 = address_15 ^ (address_15 & 511) >> 7 << 4;
                                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(d_words_addr + (address_15 - d_words_addr)), "f"(_tmem_load_6[vec_5 * 4]), "f"(_tmem_load_6[vec_5 * 4 + 1]), "f"(_tmem_load_6[vec_5 * 4 + 2]), "f"(_tmem_load_6[vec_5 * 4 + 3]) : "memory");
                                        }
                                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                                        if (tid == 0) {
                                            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 900)
                                            #error "TmaReduceAdd5d requires SM90 or newer"
                                            #endif
                                            asm volatile(
                                                "cp.reduce.async.bulk.tensor.5d.global.shared::cta.add.tile.bulk_group.L2::cache_hint"
                                                " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                                :: "l"((&dwg_s)), "r"(0), "r"(x_3 * 256 + cta_rank_0 * 128), "r"(y_3 * 2 * 16 + chunk_10), "r"(expert_3), "r"(0), "r"(d_smem_addr + (unsigned int)(chunk_10 % 3 * 8192)), "l"(0x12F0000000000000ULL) : "memory");
                                            asm volatile("cp.async.bulk.commit_group;");
                                        }
                                    }
                                    if (has_hi_3 != 0) {
                                        #pragma unroll
                                        for (int chunk_11 = 0; chunk_11 < 16; chunk_11++) {
                                            float _tmem_load_7[16];
                                            tmem_ld_x16(&_tmem_load_7[0], taddr_1 + (unsigned int)(warp_row_1 << 16) + 256 + (unsigned int)(chunk_11 * 16));
                                            asm volatile("tcgen05.wait::ld.sync.aligned;");
                                            if (tid == 0) {
                                                asm volatile("cp.async.bulk.wait_group.read 2;");
                                            }
                                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                                            #pragma unroll
                                            for (int vec_6 = 0; vec_6 < 4; vec_6++) {
                                                unsigned int address_16 = d_smem_addr + (unsigned int)((16 + chunk_11) % 3 * 8192) + (unsigned int)(tid * 64) + (unsigned int)(vec_6 * 16);
                                                address_16 = address_16 ^ (address_16 & 511) >> 7 << 4;
                                                asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(d_words_addr + (address_16 - d_words_addr)), "f"(_tmem_load_7[vec_6 * 4]), "f"(_tmem_load_7[vec_6 * 4 + 1]), "f"(_tmem_load_7[vec_6 * 4 + 2]), "f"(_tmem_load_7[vec_6 * 4 + 3]) : "memory");
                                            }
                                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                                            if (tid == 0) {
                                                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 900)
                                                #error "TmaReduceAdd5d requires SM90 or newer"
                                                #endif
                                                asm volatile(
                                                    "cp.reduce.async.bulk.tensor.5d.global.shared::cta.add.tile.bulk_group.L2::cache_hint"
                                                    " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                                    :: "l"((&dwg_s)), "r"(0), "r"(x_3 * 256 + cta_rank_0 * 128), "r"((y_3 * 2 + 1) * 16 + chunk_11), "r"(expert_3), "r"(0), "r"(d_smem_addr + (unsigned int)((16 + chunk_11) % 3 * 8192)), "l"(0x12F0000000000000ULL) : "memory");
                                                asm volatile("cp.async.bulk.commit_group;");
                                            }
                                        }
                                    }
                                    asm volatile("barrier.sync 1, 128;" ::: "memory");
                                    if (tid == 0) {
                                        asm volatile(
                                            "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                                            :: "r"((output_finished_addr) & 0xFEFFFFFF) : "memory");
                                        asm volatile("cp.async.bulk.wait_group.read 0;");
                                    }
                                    asm volatile("barrier.sync 4, 128;" ::: "memory");
                                    if (tid / 32 == 0) {
                                        if (warp == 0) {
                                            if (elect_sync()) {
                                                if (has_hi_3 != 0) {
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                            gemm_phase = phase_bits_7;
                        } else if (kind == 5) {
                            int col_blocks_8 = (local_tokens + 512 - 1) / 512;
                            {
                                col_blocks_8 = (hidden + 512 - 1) / 512;
                            }
                            int x_4 = -1;
                            int y_4 = -1;
                            int expert_4 = -1;
                            int k_start_4 = 0;
                            int k_end_4 = 0;
                            int first_4 = 0;
                            int row_blocks_4 = intermediate / 256;
                            int expert_idx_2 = 0;
                            int local_task_2 = task_3;
                            int _max_9 = ((row_blocks_4 * col_blocks_8) > (hidden / 256 * ((intermediate + 512 - 1) / 512)) ? (row_blocks_4 * col_blocks_8) : (hidden / 256 * ((intermediate + 512 - 1) / 512)));
                            int stride_2 = _max_9;
                            k_end_4 = local_tokens;
                            first_4 = 1;
                            if (k_start_4 < k_end_4 && local_task_2 < row_blocks_4 * col_blocks_8) {
                                int supergroup_4 = local_task_2 / (row_blocks_4 * 8);
                                int full_cols_4 = col_blocks_8 / 8 * 8;
                                int row_16 = 0;
                                int col_59 = 0;
                                if (local_task_2 < row_blocks_4 * full_cols_4) {
                                    row_16 = local_task_2 % (row_blocks_4 * 8) / 8;
                                    col_59 = supergroup_4 * 8 + local_task_2 % 8;
                                } else {
                                    row_16 = (local_task_2 - row_blocks_4 * full_cols_4) / (col_blocks_8 - full_cols_4);
                                    col_59 = full_cols_4 + (local_task_2 - row_blocks_4 * full_cols_4) % (col_blocks_8 - full_cols_4);
                                }
                                if ((supergroup_4 & 1) != 0) {
                                    row_16 = row_blocks_4 - row_16 - 1;
                                }
                                x_4 = row_16;
                                y_4 = col_59;
                                expert_4 = expert_idx_2;
                            }
                            unsigned int phase_bits_8 = gemm_phase;
                            int has_hi_4 = 0;
                            has_hi_4 = (int)((y_4 * 2 + 1) * 256 < hidden);
                            int global_mini_4 = 0;
                            int macro_rows_4 = 0;
                            int iterations_4 = intermediate / 64;
                            iterations_4 = (k_end_4 - k_start_4) / 64;
                            if (expert_4 < 0) {
                                if (tid == 0) {
                                }
                            } else if (tid / 32 == 7) {
                                if (warp == 7) {
                                    if (elect_sync()) {
                                        unsigned int previous_4 = phase_bits_8 >> 7 & 1;
                                        unsigned int bits_4 = phase_bits_8;
                                        if (previous_4 != 0) {
                                            mbarrier_wait(gemm_finished_addr, bits_4 >> 16 & 1);
                                            mbarrier_wait(gemm_finished_addr + 8, bits_4 >> 17 & 1);
                                            mbarrier_wait(gemm_finished_addr + 16, bits_4 >> 18 & 1);
                                            mbarrier_wait(gemm_finished_addr + 24, bits_4 >> 19 & 1);
                                            mbarrier_wait(gemm_finished_addr + 32, bits_4 >> 20 & 1);
                                            mbarrier_wait(gemm_finished_addr + 40, bits_4 >> 21 & 1);
                                            bits_4 = bits_4 ^ 128;
                                        }
                                        phase_bits_8 = bits_4;
                                        int ring_8 = 0;
                                        #pragma unroll 1
                                        for (int idx_8 = 0; idx_8 < iterations_4; idx_8++) {
                                            int token_row_2 = k_start_4 + idx_8 * 64;
                                            if (idx_8 == 0 || token_row_2 % 256 == 0) {
                                                bool enabled_value_5 = 1;
                                                if (enabled_value_5 != 0) {
                                                    int32_t _relaxed_ld_20;
                                                    asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_20) : "l"(dg_ready + (token_row_2 / 256)) : "memory");
                                                    int value_10 = _relaxed_ld_20;
                                                    while (value_10 < row_count) {
                                                        asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                                        int32_t _relaxed_ld_21;
                                                        asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_21) : "l"(dg_ready + (token_row_2 / 256)) : "memory");
                                                        value_10 = _relaxed_ld_21;
                                                    }
                                                    asm volatile("fence.acquire.gpu;" ::: "memory");
                                                }
                                            }
                                            if (idx_8 == 0 || token_row_2 % mini_size == 0) {
                                                int input_mini_2 = token_row_2 / mini_size;
                                                int _min_39 = ((mini_size) < (tokens - input_mini_2 * mini_size) ? (mini_size) : (tokens - input_mini_2 * mini_size));
                                                int input_rows_2 = _min_39;
                                                int input_count_2 = (input_rows_2 + 127) / 128 * ((hidden + 511) / 512);
                                            }
                                            mbarrier_wait(gemm_finished_addr + (ring_8) * 8, phase_bits_8 >> (unsigned int)(16 + ring_8) & 1);
                                            int local_row_2 = k_start_4 + idx_8 * 64;
                                            asm volatile(
                                                "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                                " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                                :: "r"(a_atb_addr + (unsigned int)(ring_8 * 16384)), "l"((&du_atb_s)), "r"(0), "r"(local_row_2), "r"(x_4 * 4 + cta_rank_0 * 2), "r"(0), "r"(0),
                                                   "r"(((gemm_arrived_addr + (ring_8) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                            asm volatile(
                                                "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                                " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                                :: "r"(b_ab_addr + (unsigned int)(ring_8 * 16384)), "l"((&x_atb_s)), "r"(0), "r"(local_row_2), "r"(y_4 * 2 * 4 + cta_rank_0 * 2), "r"(0), "r"(0),
                                                   "r"(((gemm_arrived_addr + (ring_8) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                            asm volatile(
                                                "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                                " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                                :: "r"(b_ab_hi_addr + (unsigned int)(ring_8 * 16384)), "l"((&x_atb_s)), "r"(0), "r"(local_row_2), "r"((y_4 * 2 + 1) * 4 + cta_rank_0 * 2), "r"(0), "r"(0),
                                                   "r"(((gemm_arrived_addr + (ring_8) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                            phase_bits_8 = phase_bits_8 ^ (unsigned int)(1 << 16 + ring_8);
                                            ring_8 = (ring_8 + 1) % 4;
                                        }
                                    }
                                }
                            } else {
                                if (tid / 32 == 4 && cta_rank_0 == 0) {
                                    if (warp == 4) {
                                        if (elect_sync()) {
                                            int ring_9 = 0;
                                            mbarrier_wait(output_finished_addr, phase_bits_8 >> 22 & 1);
                                            phase_bits_8 = phase_bits_8 ^ 4194304;
                                            asm volatile("tcgen05.fence::after_thread_sync;");
                                            #pragma unroll 1
                                            for (int idx_9 = 0; idx_9 < iterations_4; idx_9++) {
                                                mbarrier_arrive_expect_tx(gemm_arrived_addr + (ring_9) * 8, 98304);
                                                mbarrier_wait(gemm_arrived_addr + (ring_9) * 8, phase_bits_8 >> (unsigned int)ring_9 & 1);
                                                int _mma_a_lo_8 = ((((a_atb_addr) >> 4) & 0x3FFF) | 0x2000000) + (ring_9) * 1024;
                                                int _mma_b_lo_8 = ((((b_ab_addr) >> 4) & 0x3FFF) | 0x2000000) + (ring_9) * 1024;
                                                asm volatile(
            "{\n\t"
            ".reg .pred p0, p1;\n\t"
            ".reg .b32 adhi, bdhi, alo, blo, id, m0, m1, m2, m3, m4, m5, m6, m7;\n\t"
            ".reg .b64 da, db;\n\t"
            ""
            "setp.ne.b32 p0, %3, 0;\n\t"
            "setp.ne.b32 p1, 1, 0;\n\t"
            "mov.b32 m0, 0; mov.b32 m1, 0; mov.b32 m2, 0; mov.b32 m3, 0;\n\tmov.b32 m4, 0; mov.b32 m5, 0; mov.b32 m6, 0; mov.b32 m7, 0;\n\t"
            "mov.b32 adhi, 0x40004040;\n\t"
            "mov.b32 bdhi, 0x40004040;\n\t"
            "mov.b32 id, 272729232;\n\t"
            "mov.b32 alo, %0;\n\t"
            "mov.b32 blo, %1;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
            "add.u32 alo, alo, 128;\n\t"
            "add.u32 blo, blo, 128;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
            "add.u32 alo, alo, 128;\n\t"
            "add.u32 blo, blo, 128;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
            "add.u32 alo, alo, 128;\n\t"
            "add.u32 blo, blo, 128;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
            "}\n"
            :: "r"(_mma_a_lo_8), "r"(_mma_b_lo_8), "r"(tmem_accumulator), "r"(((idx_9 == 0) ? 0 : 1)));
                                                int _mma_a_lo_9 = ((((a_atb_addr) >> 4) & 0x3FFF) | 0x2000000) + (ring_9) * 1024;
                                                int _mma_b_lo_9 = ((((b_ab_hi_addr) >> 4) & 0x3FFF) | 0x2000000) + (ring_9) * 1024;
                                                asm volatile(
            "{\n\t"
            ".reg .pred p0, p1;\n\t"
            ".reg .b32 adhi, bdhi, alo, blo, id, m0, m1, m2, m3, m4, m5, m6, m7;\n\t"
            ".reg .b64 da, db;\n\t"
            ""
            "setp.ne.b32 p0, %3, 0;\n\t"
            "setp.ne.b32 p1, 1, 0;\n\t"
            "mov.b32 m0, 0; mov.b32 m1, 0; mov.b32 m2, 0; mov.b32 m3, 0;\n\tmov.b32 m4, 0; mov.b32 m5, 0; mov.b32 m6, 0; mov.b32 m7, 0;\n\t"
            "mov.b32 adhi, 0x40004040;\n\t"
            "mov.b32 bdhi, 0x40004040;\n\t"
            "mov.b32 id, 272729232;\n\t"
            "mov.b32 alo, %0;\n\t"
            "mov.b32 blo, %1;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
            "add.u32 alo, alo, 128;\n\t"
            "add.u32 blo, blo, 128;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
            "add.u32 alo, alo, 128;\n\t"
            "add.u32 blo, blo, 128;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
            "add.u32 alo, alo, 128;\n\t"
            "add.u32 blo, blo, 128;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
            "}\n"
            :: "r"(_mma_a_lo_9), "r"(_mma_b_lo_9), "r"((tmem_accumulator + (256))), "r"(((idx_9 == 0) ? 0 : 1)));
                                                tcgen05_commit_cg2_multicast(gemm_finished_addr + (ring_9) * 8, (uint16_t)(3));
                                                phase_bits_8 = phase_bits_8 ^ (unsigned int)(1 << ring_9);
                                                ring_9 = (ring_9 + 1) % 4;
                                            }
                                            tcgen05_commit_cg2_multicast(output_arrived_addr, (uint16_t)(3));
                                        }
                                    }
                                } else if (tid < 128) {
                                    mbarrier_wait(output_arrived_addr, phase_bits_8 >> 6 & 1);
                                    phase_bits_8 = phase_bits_8 ^ 64;
                                    int warp_row_2 = tid / 32 * 32;
                                    #pragma unroll
                                    for (int chunk_12 = 0; chunk_12 < 16; chunk_12++) {
                                        float _tmem_load_8[16];
                                        tmem_ld_x16(&_tmem_load_8[0], taddr_1 + (unsigned int)(warp_row_2 << 16) + (unsigned int)(chunk_12 * 16));
                                        asm volatile("tcgen05.wait::ld.sync.aligned;");
                                        if (tid == 0) {
                                            asm volatile("cp.async.bulk.wait_group.read 2;");
                                        }
                                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                                        #pragma unroll
                                        for (int vec_7 = 0; vec_7 < 4; vec_7++) {
                                            unsigned int address_17 = d_smem_addr + (unsigned int)(chunk_12 % 3 * 8192) + (unsigned int)(tid * 64) + (unsigned int)(vec_7 * 16);
                                            address_17 = address_17 ^ (address_17 & 511) >> 7 << 4;
                                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(d_words_addr + (address_17 - d_words_addr)), "f"(_tmem_load_8[vec_7 * 4]), "f"(_tmem_load_8[vec_7 * 4 + 1]), "f"(_tmem_load_8[vec_7 * 4 + 2]), "f"(_tmem_load_8[vec_7 * 4 + 3]) : "memory");
                                        }
                                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                                        if (tid == 0) {
                                            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 900)
                                            #error "TmaReduceAdd5d requires SM90 or newer"
                                            #endif
                                            asm volatile(
                                                "cp.reduce.async.bulk.tensor.5d.global.shared::cta.add.tile.bulk_group.L2::cache_hint"
                                                " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                                :: "l"((&dwu_s)), "r"(0), "r"(x_4 * 256 + cta_rank_0 * 128), "r"(y_4 * 2 * 16 + chunk_12), "r"(expert_4), "r"(0), "r"(d_smem_addr + (unsigned int)(chunk_12 % 3 * 8192)), "l"(0x12F0000000000000ULL) : "memory");
                                            asm volatile("cp.async.bulk.commit_group;");
                                        }
                                    }
                                    if (has_hi_4 != 0) {
                                        #pragma unroll
                                        for (int chunk_13 = 0; chunk_13 < 16; chunk_13++) {
                                            float _tmem_load_9[16];
                                            tmem_ld_x16(&_tmem_load_9[0], taddr_1 + (unsigned int)(warp_row_2 << 16) + 256 + (unsigned int)(chunk_13 * 16));
                                            asm volatile("tcgen05.wait::ld.sync.aligned;");
                                            if (tid == 0) {
                                                asm volatile("cp.async.bulk.wait_group.read 2;");
                                            }
                                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                                            #pragma unroll
                                            for (int vec_8 = 0; vec_8 < 4; vec_8++) {
                                                unsigned int address_18 = d_smem_addr + (unsigned int)((16 + chunk_13) % 3 * 8192) + (unsigned int)(tid * 64) + (unsigned int)(vec_8 * 16);
                                                address_18 = address_18 ^ (address_18 & 511) >> 7 << 4;
                                                asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(d_words_addr + (address_18 - d_words_addr)), "f"(_tmem_load_9[vec_8 * 4]), "f"(_tmem_load_9[vec_8 * 4 + 1]), "f"(_tmem_load_9[vec_8 * 4 + 2]), "f"(_tmem_load_9[vec_8 * 4 + 3]) : "memory");
                                            }
                                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                                            if (tid == 0) {
                                                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 900)
                                                #error "TmaReduceAdd5d requires SM90 or newer"
                                                #endif
                                                asm volatile(
                                                    "cp.reduce.async.bulk.tensor.5d.global.shared::cta.add.tile.bulk_group.L2::cache_hint"
                                                    " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                                    :: "l"((&dwu_s)), "r"(0), "r"(x_4 * 256 + cta_rank_0 * 128), "r"((y_4 * 2 + 1) * 16 + chunk_13), "r"(expert_4), "r"(0), "r"(d_smem_addr + (unsigned int)((16 + chunk_13) % 3 * 8192)), "l"(0x12F0000000000000ULL) : "memory");
                                                asm volatile("cp.async.bulk.commit_group;");
                                            }
                                        }
                                    }
                                    asm volatile("barrier.sync 1, 128;" ::: "memory");
                                    if (tid == 0) {
                                        asm volatile(
                                            "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                                            :: "r"((output_finished_addr) & 0xFEFFFFFF) : "memory");
                                        asm volatile("cp.async.bulk.wait_group.read 0;");
                                    }
                                    asm volatile("barrier.sync 4, 128;" ::: "memory");
                                    if (tid / 32 == 0) {
                                        if (warp == 0) {
                                            if (elect_sync()) {
                                                if (has_hi_4 != 0) {
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                            gemm_phase = phase_bits_8;
                        }
                    }
                }
            } else if (kind == 6) {
                int col_blocks_9 = (intermediate + 256 - 1) / 256;
                int x_5 = -1;
                int y_5 = -1;
                int expert_5 = -1;
                int k_start_5 = 0;
                int k_end_5 = 0;
                int first_5 = 0;
                int first_block = (macro_1 * (macro_size / mini_size) + mini_1) * (mini_size / 256);
                int _min_40 = ((first_block + mini_size / 256) < (tokens / 256) ? (first_block + mini_size / 256) : (tokens / 256));
                int end_block = _min_40;
                int block = first_block + task_3 / col_blocks_9;
                if (block < end_block) {
                    int index_2 = counts[3 * experts + block];
                    int offset_6 = counts[experts + index_2] / 256;
                    int _max_11 = ((first_block) > (offset_6) ? (first_block) : (offset_6));
                    int first_row_1 = _max_11;
                    int _min_41 = ((end_block) < (offset_6 + counts[index_2] / 256) ? (end_block) : (offset_6 + counts[index_2] / 256));
                    int rows_3 = _min_41 - first_row_1;
                    int supergroup_5 = (task_3 - (first_row_1 - first_block) * col_blocks_9) / (rows_3 * 8);
                    int full_cols_5 = col_blocks_9 / 8 * 8;
                    int row_17 = 0;
                    int col_60 = 0;
                    if (task_3 - (first_row_1 - first_block) * col_blocks_9 < rows_3 * full_cols_5) {
                        row_17 = (task_3 - (first_row_1 - first_block) * col_blocks_9) % (rows_3 * 8) / 8;
                        col_60 = supergroup_5 * 8 + (task_3 - (first_row_1 - first_block) * col_blocks_9) % 8;
                    } else {
                        row_17 = (task_3 - (first_row_1 - first_block) * col_blocks_9 - rows_3 * full_cols_5) / (col_blocks_9 - full_cols_5);
                        col_60 = full_cols_5 + (task_3 - (first_row_1 - first_block) * col_blocks_9 - rows_3 * full_cols_5) % (col_blocks_9 - full_cols_5);
                    }
                    if ((supergroup_5 & 1) != 0) {
                        row_17 = rows_3 - row_17 - 1;
                    }
                    x_5 = first_row_1 + row_17 - macro_1 * (macro_size / 256);
                    y_5 = col_60;
                    expert_5 = index_2;
                }
                unsigned int phase_bits_9 = gemm_phase;
                int has_hi_5 = 0;
                int global_mini_5 = macro_1 * (macro_size / mini_size) + mini_1;
                int macro_rows_5 = macro_1 * (macro_size / 256);
                int iterations_5 = hidden / 128;
                int macro_k = macro_1 * (macro_size / 128);
                if (expert_5 < 0) {
                    if (tid == 0) {
                    }
                } else if (tid / 32 == 7) {
                    if (warp == 7) {
                        if (elect_sync()) {
                            {
                                int _min_42 = ((mini_size) < (tokens - global_mini_5 * mini_size) ? (mini_size) : (tokens - global_mini_5 * mini_size));
                                int _max_12 = ((0) > (_min_42) ? (0) : (_min_42));
                                int mini_rows_6 = _max_12;
                                int required_5 = (mini_rows_6 + 127) / 128 * ((hidden + 511) / 512);
                                bool enabled_value_6 = 1;
                                if (enabled_value_6 != 0) {
                                    int32_t _relaxed_ld_22;
                                    asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_22) : "l"(replay_x + global_mini_5) : "memory");
                                    int value_11 = _relaxed_ld_22;
                                    while (value_11 < required_5) {
                                        asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                        int32_t _relaxed_ld_23;
                                        asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_23) : "l"(replay_x + global_mini_5) : "memory");
                                        value_11 = _relaxed_ld_23;
                                    }
                                    asm volatile("fence.acquire.gpu;" ::: "memory");
                                }
                            }
                            unsigned int previous_5 = phase_bits_9 >> 7 & 1;
                            unsigned int bits_5 = phase_bits_9;
                            if (previous_5 != 1) {
                                mbarrier_wait(gemm_finished_addr, bits_5 >> 16 & 1);
                                mbarrier_wait(gemm_finished_addr + 8, bits_5 >> 17 & 1);
                                mbarrier_wait(gemm_finished_addr + 16, bits_5 >> 18 & 1);
                                mbarrier_wait(gemm_finished_addr + 24, bits_5 >> 19 & 1);
                                mbarrier_wait(gemm_finished_addr + 32, bits_5 >> 20 & 1);
                                mbarrier_wait(gemm_finished_addr + 40, bits_5 >> 21 & 1);
                                bits_5 = bits_5 ^ 128;
                            }
                            phase_bits_9 = bits_5;
                            int ring_10 = 0;
                            #pragma unroll 1
                            for (int idx_10 = 0; idx_10 < iterations_5; idx_10++) {
                                mbarrier_wait(gemm_finished_addr + (ring_10) * 8, phase_bits_9 >> (unsigned int)(16 + ring_10) & 1);
                                asm volatile(
                                    "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                    " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                    :: "r"(smem_v52_addr + (unsigned int)(ring_10 * 16384)), "l"((&x_q)), "r"(0), "r"(x_5 * 256 + cta_rank_0 * 128), "r"(idx_10), "r"(0), "r"(0),
                                       "r"(((gemm_arrived_addr + (ring_10) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                asm volatile(
                                    "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                    " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                    :: "r"(smem_v53_addr + (unsigned int)(ring_10 * 16384)), "l"((&wg_q)), "r"(0), "r"(y_5 * 256 + cta_rank_0 * 128), "r"(idx_10), "r"(expert_5), "r"(0),
                                       "r"(((gemm_arrived_addr + (ring_10) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                phase_bits_9 = phase_bits_9 ^ (unsigned int)(1 << 16 + ring_10);
                                ring_10 = (ring_10 + 1) % 6;
                            }
                        }
                    }
                } else {
                    if (tid / 32 == 6) {
                        if (warp == 6) {
                            if (elect_sync()) {
                                {
                                    int _min_43 = ((mini_size) < (tokens - global_mini_5 * mini_size) ? (mini_size) : (tokens - global_mini_5 * mini_size));
                                    int _max_13 = ((0) > (_min_43) ? (0) : (_min_43));
                                    int mini_rows_7 = _max_13;
                                    int required_6 = (mini_rows_7 + 127) / 128 * ((hidden + 511) / 512);
                                    bool enabled_value_7 = 1;
                                    if (enabled_value_7 != 0) {
                                        int32_t _relaxed_ld_24;
                                        asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_24) : "l"(replay_x + global_mini_5) : "memory");
                                        int value_12 = _relaxed_ld_24;
                                        while (value_12 < required_6) {
                                            asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                            int32_t _relaxed_ld_25;
                                            asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_25) : "l"(replay_x + global_mini_5) : "memory");
                                            value_12 = _relaxed_ld_25;
                                        }
                                        asm volatile("fence.acquire.gpu;" ::: "memory");
                                    }
                                }
                                unsigned int previous_6 = phase_bits_9 >> 7 & 1;
                                unsigned int bits_6 = phase_bits_9;
                                if (previous_6 != 1) {
                                    mbarrier_wait(scales_finished_addr, bits_6 >> 16 & 1);
                                    mbarrier_wait(scales_finished_addr + 8, bits_6 >> 17 & 1);
                                    mbarrier_wait(scales_finished_addr + 16, bits_6 >> 18 & 1);
                                    mbarrier_wait(scales_finished_addr + 24, bits_6 >> 19 & 1);
                                    mbarrier_wait(scales_finished_addr + 32, bits_6 >> 20 & 1);
                                    mbarrier_wait(scales_finished_addr + 40, bits_6 >> 21 & 1);
                                    bits_6 = bits_6 ^ 128;
                                }
                                phase_bits_9 = bits_6;
                                int ring_11 = 0;
                                #pragma unroll 1
                                for (int idx_11 = 0; idx_11 < iterations_5; idx_11++) {
                                    mbarrier_wait(scales_finished_addr + (ring_11) * 8, phase_bits_9 >> (unsigned int)(16 + ring_11) & 1);
                                    asm volatile(
                                        "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                        " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                        :: "r"(smem_v54_addr + (unsigned int)(ring_11 * 512)), "l"((&x_sc)), "r"(0), "r"(0), "r"((x_5 * 2 + cta_rank_0) * (hidden / 128) + idx_11),
                                           "r"(((scales_arrived_addr + (ring_11) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                    asm volatile(
                                        "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                        " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                        :: "r"(smem_v55_addr + (unsigned int)(ring_11 * 1024) + (unsigned int)(cta_rank_0 * 512)), "l"((&wg_sc)), "r"(0), "r"(0), "r"((expert_5 * i_tiles_5 + y_5 * 2 + cta_rank_0) * (hidden / 128) + idx_11),
                                           "r"(((scales_arrived_addr + (ring_11) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(3)) : "memory");
                                    phase_bits_9 = phase_bits_9 ^ (unsigned int)(1 << 16 + ring_11);
                                    ring_11 = (ring_11 + 1) % 6;
                                }
                            }
                        }
                    } else if (tid / 32 == 4 && cta_rank_0 == 0) {
                        if (warp == 4) {
                            if (elect_sync()) {
                                int ring_12 = 0;
                                mbarrier_wait(output_finished_addr, phase_bits_9 >> 22 & 1);
                                phase_bits_9 = phase_bits_9 ^ 4194304;
                                asm volatile("tcgen05.fence::after_thread_sync;");
                                #pragma unroll 1
                                for (int idx_12 = 0; idx_12 < iterations_5; idx_12++) {
                                    mbarrier_arrive_expect_tx(scales_arrived_addr + (ring_12) * 8, 3072);
                                    mbarrier_wait(scales_arrived_addr + (ring_12) * 8, phase_bits_9 >> (unsigned int)(8 + ring_12) & 1);
                                    int buffer = idx_12 % 3;
                                    tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfa + buffer * 4, make_sf_cp_desc_sbo128(smem_v54_addr + (unsigned int)(ring_12 * 512)));
                                    tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfb + buffer * 8, make_sf_cp_desc_sbo128(smem_v55_addr + (unsigned int)(ring_12 * 1024)));
                                    tcgen05_cp_32x128b_warpx4_cta2((tmem_tmem_sfb + buffer * 8 + 4), make_sf_cp_desc_sbo128((smem_v55_addr + (unsigned int)(ring_12 * 1024) + 512)));
                                    tcgen05_commit_cg2_multicast(scales_finished_addr + (ring_12) * 8, (uint16_t)(3));
                                    mbarrier_arrive_expect_tx(gemm_arrived_addr + (ring_12) * 8, 65536);
                                    mbarrier_wait(gemm_arrived_addr + (ring_12) * 8, phase_bits_9 >> (unsigned int)ring_12 & 1);
                                    int _mma_a_lo_10 = (((smem_v52_addr) >> 4) & 0x3FFF) + (ring_12) * 1024;
                                    int _mma_b_lo_10 = (((smem_v53_addr) >> 4) & 0x3FFF) + (ring_12) * 1024;
                                    {
                                        uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_10) | ((uint64_t)0x40004040 << 32);
                                        uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_10) | ((uint64_t)0x40004040 << 32);

                                        tcgen05_mma_mxf8_bs_cta2(tmem_accumulator, a_desc + 0, b_desc + 0,
                                            0x10c00000U, tmem_tmem_sfa + buffer * 4, tmem_tmem_sfb + buffer * 8, ((idx_12 == 0) ? 0 : 1));
                                        tcgen05_mma_mxf8_bs_cta2(tmem_accumulator, a_desc + 2, b_desc + 2,
                                            0x30c00010U, tmem_tmem_sfa + buffer * 4, tmem_tmem_sfb + buffer * 8, 1);
                                        tcgen05_mma_mxf8_bs_cta2(tmem_accumulator, a_desc + 4, b_desc + 4,
                                            0x50c00020U, tmem_tmem_sfa + buffer * 4, tmem_tmem_sfb + buffer * 8, 1);
                                        tcgen05_mma_mxf8_bs_cta2(tmem_accumulator, a_desc + 6, b_desc + 6,
                                            0x70c00030U, tmem_tmem_sfa + buffer * 4, tmem_tmem_sfb + buffer * 8, 1);
                                    }
                                    tcgen05_commit_cg2_multicast(gemm_finished_addr + (ring_12) * 8, (uint16_t)(3));
                                    phase_bits_9 = phase_bits_9 ^ (unsigned int)(1 << ring_12) ^ (unsigned int)(1 << 8 + ring_12);
                                    ring_12 = (ring_12 + 1) % 6;
                                }
                                tcgen05_commit_cg2_multicast(output_arrived_addr, (uint16_t)(3));
                            }
                        }
                    } else {
                        if (tid < 128) {
                            mbarrier_wait(output_arrived_addr, phase_bits_9 >> 6 & 1);
                            int warp_row_3 = tid / 32 * 32;
                            unsigned int packed_7[128];
                            #pragma unroll
                            for (int i_48 = 0; i_48 < 8; i_48++) {
                                float _tmem_load_10[32];
                                asm volatile(
                                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                                    : "=f"(_tmem_load_10[0]), "=f"(_tmem_load_10[1]), "=f"(_tmem_load_10[2]), "=f"(_tmem_load_10[3]), "=f"(_tmem_load_10[4]), "=f"(_tmem_load_10[5]), "=f"(_tmem_load_10[6]), "=f"(_tmem_load_10[7]), "=f"(_tmem_load_10[8]), "=f"(_tmem_load_10[9]), "=f"(_tmem_load_10[10]), "=f"(_tmem_load_10[11]), "=f"(_tmem_load_10[12]), "=f"(_tmem_load_10[13]), "=f"(_tmem_load_10[14]), "=f"(_tmem_load_10[15]), "=f"(_tmem_load_10[16]), "=f"(_tmem_load_10[17]), "=f"(_tmem_load_10[18]), "=f"(_tmem_load_10[19]), "=f"(_tmem_load_10[20]), "=f"(_tmem_load_10[21]), "=f"(_tmem_load_10[22]), "=f"(_tmem_load_10[23]), "=f"(_tmem_load_10[24]), "=f"(_tmem_load_10[25]), "=f"(_tmem_load_10[26]), "=f"(_tmem_load_10[27]), "=f"(_tmem_load_10[28]), "=f"(_tmem_load_10[29]), "=f"(_tmem_load_10[30]), "=f"(_tmem_load_10[31])
                                    : "r"(taddr_1 + (unsigned int)(warp_row_3 << 16) + (unsigned int)(i_48 * 32)));
                                #pragma unroll
                                for (int j_24 = 0; j_24 < 16; j_24++) {
                                    __nv_bfloat162 _bf16x2_26 = __float22bfloat162_rn(make_float2(_tmem_load_10[2 * j_24], _tmem_load_10[2 * j_24 + 1]));
                                    packed_7[i_48 * 16 + j_24] = __as_u32(_bf16x2_26);
                                }
                            }
                            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                            if (tid == 0) {
                                asm volatile(
                                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                                    :: "r"((output_finished_addr) & 0xFEFFFFFF) : "memory");
                            }
                            unsigned int scale_word_24 = 0;
                            unsigned int block_0[16];
                            #pragma unroll
                            for (int j_25 = 0; j_25 < 16; j_25++) {
                                block_0[j_25] = packed_7[j_25];
                            }
                            #pragma unroll
                            for (int j_26 = 0; j_26 < 4; j_26++) {
                                unsigned int address_19 = d_smem_addr + (unsigned int)(tid * 64) + (unsigned int)(j_26 * 16);
                                address_19 = address_19 ^ (address_19 & 511) >> 7 << 4;
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v56_addr + (address_19 - smem_v56_addr)), "r"(packed_7[4 * j_26]), "r"(packed_7[4 * j_26 + 1]), "r"(packed_7[4 * j_26 + 2]), "r"(packed_7[4 * j_26 + 3]) : "memory");
                            }
                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                            if (tid == 0) {
                                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                asm volatile(
                                    "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                    " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                    :: "l"((&gate_out_r)), "r"(0), "r"(x_5 * 256 + cta_rank_0 * 128), "r"(y_5 * 8), "r"(0), "r"(0), "r"(d_smem_addr), "l"(0x12F0000000000000ULL) : "memory");
                                asm volatile("cp.async.bulk.commit_group;");
                                asm volatile("cp.async.bulk.wait_group.read 1;");
                            }
                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                            uint32_t _bf16x2_abs_48;
                            asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_48) : "r"(block_0[0]));
                            unsigned int amax_pair_24 = _bf16x2_abs_48;
                            #pragma unroll
                            for (int i_49 = 1; i_49 < 16; i_49++) {
                                uint32_t _bf16x2_abs_49;
                                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_49) : "r"(block_0[i_49]));
                                uint32_t _bf16x2_max_24;
                                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_24) : "r"(amax_pair_24), "r"(_bf16x2_abs_49));
                                amax_pair_24 = _bf16x2_max_24;
                            }
                            uint16_t _bf16_max_24;
                            asm("max.bf16 %0, %1, %2;" : "=h"(_bf16_max_24) : "h"((uint16_t)(amax_pair_24 & 65535)), "h"((uint16_t)(amax_pair_24 >> 16)));
                            float _cvt_f32_bf16_24;
                            asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_24) : "h"((uint16_t)(_bf16_max_24)));
                            float amax_24 = _cvt_f32_bf16_24;
                            float _fmax_24 = fmaxf(amax_24 * 0.002232142857f, 1e-12f);
                            float scale_24 = _fmax_24;
                            uint16_t _ue8m0x2_f32_24;
                            asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_ue8m0x2_f32_24) : "f"(scale_24), "f"(scale_24));
                            unsigned int scale_byte_24 = (unsigned int)_ue8m0x2_f32_24 & 255;
                            unsigned int inverse_lane_24 = 254 - scale_byte_24 << 7;
                            unsigned int inverse_24 = inverse_lane_24 | inverse_lane_24 << 16;
                            unsigned int words_24[8];
                            #pragma unroll
                            for (int i_50 = 0; i_50 < 8; i_50++) {
                                uint32_t _bf16x2_mul_48;
                                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_48) : "r"(block_0[i_50 * 2]), "r"(inverse_24));
                                uint16_t _e4m3x2_48;
                                asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_48) : "r"(_bf16x2_mul_48));
                                uint32_t _bf16x2_mul_49;
                                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_49) : "r"(block_0[i_50 * 2 + 1]), "r"(inverse_24));
                                uint16_t _e4m3x2_49;
                                asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_49) : "r"(_bf16x2_mul_49));
                                words_24[i_50] = (unsigned int)_e4m3x2_48 | (unsigned int)_e4m3x2_49 << 16;
                            }
                            scale_word_24 = scale_word_24 | scale_byte_24;
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v57_addr + (unsigned int)(tid * 32)), "r"(words_24[0]), "r"(words_24[1]), "r"(words_24[2]), "r"(words_24[3]) : "memory");
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v57_addr + (unsigned int)(tid * 32 + 16)), "r"(words_24[4]), "r"(words_24[5]), "r"(words_24[6]), "r"(words_24[7]) : "memory");
                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                            if (tid == 0) {
                                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                asm volatile(
                                    "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                    " [%0, {%1, %2}], [%3], %4;"
                                    :: "l"((&gate_q_store)), "r"(y_5 * 256), "r"(x_5 * 256 + cta_rank_0 * 128), "r"(smem_v57_addr), "l"(0x12F0000000000000ULL) : "memory");
                                asm volatile("cp.async.bulk.commit_group;");
                            }
                            unsigned int block_1[16];
                            #pragma unroll
                            for (int j_27 = 0; j_27 < 16; j_27++) {
                                block_1[j_27] = packed_7[16 + j_27];
                            }
                            #pragma unroll
                            for (int j_28 = 0; j_28 < 4; j_28++) {
                                unsigned int address_20 = d_smem_addr + 8192 + (unsigned int)(tid * 64) + (unsigned int)(j_28 * 16);
                                address_20 = address_20 ^ (address_20 & 511) >> 7 << 4;
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v56_addr + (address_20 - smem_v56_addr)), "r"(packed_7[16 + 4 * j_28]), "r"(packed_7[16 + 4 * j_28 + 1]), "r"(packed_7[16 + 4 * j_28 + 2]), "r"(packed_7[16 + 4 * j_28 + 3]) : "memory");
                            }
                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                            if (tid == 0) {
                                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                asm volatile(
                                    "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                    " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                    :: "l"((&gate_out_r)), "r"(0), "r"(x_5 * 256 + cta_rank_0 * 128), "r"(y_5 * 8 + 1), "r"(0), "r"(0), "r"(d_smem_addr + 8192), "l"(0x12F0000000000000ULL) : "memory");
                                asm volatile("cp.async.bulk.commit_group;");
                                asm volatile("cp.async.bulk.wait_group.read 1;");
                            }
                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                            uint32_t _bf16x2_abs_50;
                            asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_50) : "r"(block_1[0]));
                            unsigned int amax_pair_2_1 = _bf16x2_abs_50;
                            #pragma unroll
                            for (int i_51 = 1; i_51 < 16; i_51++) {
                                uint32_t _bf16x2_abs_51;
                                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_51) : "r"(block_1[i_51]));
                                uint32_t _bf16x2_max_25;
                                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_25) : "r"(amax_pair_2_1), "r"(_bf16x2_abs_51));
                                amax_pair_2_1 = _bf16x2_max_25;
                            }
                            uint16_t _bf16_max_25;
                            asm("max.bf16 %0, %1, %2;" : "=h"(_bf16_max_25) : "h"((uint16_t)(amax_pair_2_1 & 65535)), "h"((uint16_t)(amax_pair_2_1 >> 16)));
                            float _cvt_f32_bf16_25;
                            asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_25) : "h"((uint16_t)(_bf16_max_25)));
                            float amax_3_1 = _cvt_f32_bf16_25;
                            float _fmax_25 = fmaxf(amax_3_1 * 0.002232142857f, 1e-12f);
                            float scale_4_1 = _fmax_25;
                            uint16_t _ue8m0x2_f32_25;
                            asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_ue8m0x2_f32_25) : "f"(scale_4_1), "f"(scale_4_1));
                            unsigned int scale_byte_5_1 = (unsigned int)_ue8m0x2_f32_25 & 255;
                            unsigned int inverse_lane_6_1 = 254 - scale_byte_5_1 << 7;
                            unsigned int inverse_7_1 = inverse_lane_6_1 | inverse_lane_6_1 << 16;
                            unsigned int words_8_1[8];
                            #pragma unroll
                            for (int i_52 = 0; i_52 < 8; i_52++) {
                                uint32_t _bf16x2_mul_50;
                                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_50) : "r"(block_1[i_52 * 2]), "r"(inverse_7_1));
                                uint16_t _e4m3x2_50;
                                asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_50) : "r"(_bf16x2_mul_50));
                                uint32_t _bf16x2_mul_51;
                                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_51) : "r"(block_1[i_52 * 2 + 1]), "r"(inverse_7_1));
                                uint16_t _e4m3x2_51;
                                asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_51) : "r"(_bf16x2_mul_51));
                                words_8_1[i_52] = (unsigned int)_e4m3x2_50 | (unsigned int)_e4m3x2_51 << 16;
                            }
                            scale_word_24 = scale_word_24 | scale_byte_5_1 << 8;
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v57_addr + (unsigned int)(tid * 32)), "r"(words_8_1[0]), "r"(words_8_1[1]), "r"(words_8_1[2]), "r"(words_8_1[3]) : "memory");
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v57_addr + (unsigned int)(tid * 32 + 16)), "r"(words_8_1[4]), "r"(words_8_1[5]), "r"(words_8_1[6]), "r"(words_8_1[7]) : "memory");
                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                            if (tid == 0) {
                                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                asm volatile(
                                    "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                    " [%0, {%1, %2}], [%3], %4;"
                                    :: "l"((&gate_q_store)), "r"(y_5 * 256 + 32), "r"(x_5 * 256 + cta_rank_0 * 128), "r"(smem_v57_addr), "l"(0x12F0000000000000ULL) : "memory");
                                asm volatile("cp.async.bulk.commit_group;");
                            }
                            unsigned int block_9[16];
                            #pragma unroll
                            for (int j_29 = 0; j_29 < 16; j_29++) {
                                block_9[j_29] = packed_7[32 + j_29];
                            }
                            #pragma unroll
                            for (int j_30 = 0; j_30 < 4; j_30++) {
                                unsigned int address_21 = d_smem_addr + (unsigned int)(tid * 64) + (unsigned int)(j_30 * 16);
                                address_21 = address_21 ^ (address_21 & 511) >> 7 << 4;
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v56_addr + (address_21 - smem_v56_addr)), "r"(packed_7[32 + 4 * j_30]), "r"(packed_7[32 + 4 * j_30 + 1]), "r"(packed_7[32 + 4 * j_30 + 2]), "r"(packed_7[32 + 4 * j_30 + 3]) : "memory");
                            }
                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                            if (tid == 0) {
                                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                asm volatile(
                                    "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                    " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                    :: "l"((&gate_out_r)), "r"(0), "r"(x_5 * 256 + cta_rank_0 * 128), "r"(y_5 * 8 + 2), "r"(0), "r"(0), "r"(d_smem_addr), "l"(0x12F0000000000000ULL) : "memory");
                                asm volatile("cp.async.bulk.commit_group;");
                                asm volatile("cp.async.bulk.wait_group.read 1;");
                            }
                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                            uint32_t _bf16x2_abs_52;
                            asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_52) : "r"(block_9[0]));
                            unsigned int amax_pair_10_1 = _bf16x2_abs_52;
                            #pragma unroll
                            for (int i_53 = 1; i_53 < 16; i_53++) {
                                uint32_t _bf16x2_abs_53;
                                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_53) : "r"(block_9[i_53]));
                                uint32_t _bf16x2_max_26;
                                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_26) : "r"(amax_pair_10_1), "r"(_bf16x2_abs_53));
                                amax_pair_10_1 = _bf16x2_max_26;
                            }
                            uint16_t _bf16_max_26;
                            asm("max.bf16 %0, %1, %2;" : "=h"(_bf16_max_26) : "h"((uint16_t)(amax_pair_10_1 & 65535)), "h"((uint16_t)(amax_pair_10_1 >> 16)));
                            float _cvt_f32_bf16_26;
                            asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_26) : "h"((uint16_t)(_bf16_max_26)));
                            float amax_11_1 = _cvt_f32_bf16_26;
                            float _fmax_26 = fmaxf(amax_11_1 * 0.002232142857f, 1e-12f);
                            float scale_12_1 = _fmax_26;
                            uint16_t _ue8m0x2_f32_26;
                            asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_ue8m0x2_f32_26) : "f"(scale_12_1), "f"(scale_12_1));
                            unsigned int scale_byte_13_1 = (unsigned int)_ue8m0x2_f32_26 & 255;
                            unsigned int inverse_lane_14_1 = 254 - scale_byte_13_1 << 7;
                            unsigned int inverse_15_1 = inverse_lane_14_1 | inverse_lane_14_1 << 16;
                            unsigned int words_16_1[8];
                            #pragma unroll
                            for (int i_54 = 0; i_54 < 8; i_54++) {
                                uint32_t _bf16x2_mul_52;
                                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_52) : "r"(block_9[i_54 * 2]), "r"(inverse_15_1));
                                uint16_t _e4m3x2_52;
                                asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_52) : "r"(_bf16x2_mul_52));
                                uint32_t _bf16x2_mul_53;
                                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_53) : "r"(block_9[i_54 * 2 + 1]), "r"(inverse_15_1));
                                uint16_t _e4m3x2_53;
                                asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_53) : "r"(_bf16x2_mul_53));
                                words_16_1[i_54] = (unsigned int)_e4m3x2_52 | (unsigned int)_e4m3x2_53 << 16;
                            }
                            scale_word_24 = scale_word_24 | scale_byte_13_1 << 16;
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v57_addr + (unsigned int)(tid * 32)), "r"(words_16_1[0]), "r"(words_16_1[1]), "r"(words_16_1[2]), "r"(words_16_1[3]) : "memory");
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v57_addr + (unsigned int)(tid * 32 + 16)), "r"(words_16_1[4]), "r"(words_16_1[5]), "r"(words_16_1[6]), "r"(words_16_1[7]) : "memory");
                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                            if (tid == 0) {
                                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                asm volatile(
                                    "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                    " [%0, {%1, %2}], [%3], %4;"
                                    :: "l"((&gate_q_store)), "r"(y_5 * 256 + 64), "r"(x_5 * 256 + cta_rank_0 * 128), "r"(smem_v57_addr), "l"(0x12F0000000000000ULL) : "memory");
                                asm volatile("cp.async.bulk.commit_group;");
                            }
                            unsigned int block_17[16];
                            #pragma unroll
                            for (int j_31 = 0; j_31 < 16; j_31++) {
                                block_17[j_31] = packed_7[48 + j_31];
                            }
                            #pragma unroll
                            for (int j_32 = 0; j_32 < 4; j_32++) {
                                unsigned int address_22 = d_smem_addr + 8192 + (unsigned int)(tid * 64) + (unsigned int)(j_32 * 16);
                                address_22 = address_22 ^ (address_22 & 511) >> 7 << 4;
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v56_addr + (address_22 - smem_v56_addr)), "r"(packed_7[48 + 4 * j_32]), "r"(packed_7[48 + 4 * j_32 + 1]), "r"(packed_7[48 + 4 * j_32 + 2]), "r"(packed_7[48 + 4 * j_32 + 3]) : "memory");
                            }
                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                            if (tid == 0) {
                                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                asm volatile(
                                    "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                    " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                    :: "l"((&gate_out_r)), "r"(0), "r"(x_5 * 256 + cta_rank_0 * 128), "r"(y_5 * 8 + 3), "r"(0), "r"(0), "r"(d_smem_addr + 8192), "l"(0x12F0000000000000ULL) : "memory");
                                asm volatile("cp.async.bulk.commit_group;");
                                asm volatile("cp.async.bulk.wait_group.read 1;");
                            }
                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                            uint32_t _bf16x2_abs_54;
                            asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_54) : "r"(block_17[0]));
                            unsigned int amax_pair_18_1 = _bf16x2_abs_54;
                            #pragma unroll
                            for (int i_55 = 1; i_55 < 16; i_55++) {
                                uint32_t _bf16x2_abs_55;
                                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_55) : "r"(block_17[i_55]));
                                uint32_t _bf16x2_max_27;
                                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_27) : "r"(amax_pair_18_1), "r"(_bf16x2_abs_55));
                                amax_pair_18_1 = _bf16x2_max_27;
                            }
                            uint16_t _bf16_max_27;
                            asm("max.bf16 %0, %1, %2;" : "=h"(_bf16_max_27) : "h"((uint16_t)(amax_pair_18_1 & 65535)), "h"((uint16_t)(amax_pair_18_1 >> 16)));
                            float _cvt_f32_bf16_27;
                            asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_27) : "h"((uint16_t)(_bf16_max_27)));
                            float amax_19_1 = _cvt_f32_bf16_27;
                            float _fmax_27 = fmaxf(amax_19_1 * 0.002232142857f, 1e-12f);
                            float scale_20_1 = _fmax_27;
                            uint16_t _ue8m0x2_f32_27;
                            asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_ue8m0x2_f32_27) : "f"(scale_20_1), "f"(scale_20_1));
                            unsigned int scale_byte_21_1 = (unsigned int)_ue8m0x2_f32_27 & 255;
                            unsigned int inverse_lane_22_1 = 254 - scale_byte_21_1 << 7;
                            unsigned int inverse_23_1 = inverse_lane_22_1 | inverse_lane_22_1 << 16;
                            unsigned int words_24_1[8];
                            #pragma unroll
                            for (int i_56 = 0; i_56 < 8; i_56++) {
                                uint32_t _bf16x2_mul_54;
                                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_54) : "r"(block_17[i_56 * 2]), "r"(inverse_23_1));
                                uint16_t _e4m3x2_54;
                                asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_54) : "r"(_bf16x2_mul_54));
                                uint32_t _bf16x2_mul_55;
                                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_55) : "r"(block_17[i_56 * 2 + 1]), "r"(inverse_23_1));
                                uint16_t _e4m3x2_55;
                                asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_55) : "r"(_bf16x2_mul_55));
                                words_24_1[i_56] = (unsigned int)_e4m3x2_54 | (unsigned int)_e4m3x2_55 << 16;
                            }
                            scale_word_24 = scale_word_24 | scale_byte_21_1 << 24;
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v57_addr + (unsigned int)(tid * 32)), "r"(words_24_1[0]), "r"(words_24_1[1]), "r"(words_24_1[2]), "r"(words_24_1[3]) : "memory");
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v57_addr + (unsigned int)(tid * 32 + 16)), "r"(words_24_1[4]), "r"(words_24_1[5]), "r"(words_24_1[6]), "r"(words_24_1[7]) : "memory");
                            smem_v58[tid % 32 * 4 + tid / 32] = scale_word_24;
                            scale_word_24 = 0;
                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                            if (tid == 0) {
                                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                asm volatile(
                                    "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                    " [%0, {%1, %2}], [%3], %4;"
                                    :: "l"((&gate_q_store)), "r"(y_5 * 256 + 96), "r"(x_5 * 256 + cta_rank_0 * 128), "r"(smem_v57_addr), "l"(0x12F0000000000000ULL) : "memory");
                                tma_store_3d((&gate_sc_store), 0, 0, (x_5 * 2 + cta_rank_0) * i_tiles + y_5 * 2, smem_v58_addr);
                                asm volatile("cp.async.bulk.commit_group;");
                            }
                            unsigned int block_25[16];
                            #pragma unroll
                            for (int j_33 = 0; j_33 < 16; j_33++) {
                                block_25[j_33] = packed_7[64 + j_33];
                            }
                            #pragma unroll
                            for (int j_34 = 0; j_34 < 4; j_34++) {
                                unsigned int address_23 = d_smem_addr + (unsigned int)(tid * 64) + (unsigned int)(j_34 * 16);
                                address_23 = address_23 ^ (address_23 & 511) >> 7 << 4;
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v56_addr + (address_23 - smem_v56_addr)), "r"(packed_7[64 + 4 * j_34]), "r"(packed_7[64 + 4 * j_34 + 1]), "r"(packed_7[64 + 4 * j_34 + 2]), "r"(packed_7[64 + 4 * j_34 + 3]) : "memory");
                            }
                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                            if (tid == 0) {
                                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                asm volatile(
                                    "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                    " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                    :: "l"((&gate_out_r)), "r"(0), "r"(x_5 * 256 + cta_rank_0 * 128), "r"(y_5 * 8 + 4), "r"(0), "r"(0), "r"(d_smem_addr), "l"(0x12F0000000000000ULL) : "memory");
                                asm volatile("cp.async.bulk.commit_group;");
                                asm volatile("cp.async.bulk.wait_group.read 1;");
                            }
                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                            uint32_t _bf16x2_abs_56;
                            asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_56) : "r"(block_25[0]));
                            unsigned int amax_pair_26 = _bf16x2_abs_56;
                            #pragma unroll
                            for (int i_57 = 1; i_57 < 16; i_57++) {
                                uint32_t _bf16x2_abs_57;
                                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_57) : "r"(block_25[i_57]));
                                uint32_t _bf16x2_max_28;
                                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_28) : "r"(amax_pair_26), "r"(_bf16x2_abs_57));
                                amax_pair_26 = _bf16x2_max_28;
                            }
                            uint16_t _bf16_max_28;
                            asm("max.bf16 %0, %1, %2;" : "=h"(_bf16_max_28) : "h"((uint16_t)(amax_pair_26 & 65535)), "h"((uint16_t)(amax_pair_26 >> 16)));
                            float _cvt_f32_bf16_28;
                            asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_28) : "h"((uint16_t)(_bf16_max_28)));
                            float amax_27 = _cvt_f32_bf16_28;
                            float _fmax_28 = fmaxf(amax_27 * 0.002232142857f, 1e-12f);
                            float scale_28 = _fmax_28;
                            uint16_t _ue8m0x2_f32_28;
                            asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_ue8m0x2_f32_28) : "f"(scale_28), "f"(scale_28));
                            unsigned int scale_byte_29 = (unsigned int)_ue8m0x2_f32_28 & 255;
                            unsigned int inverse_lane_30 = 254 - scale_byte_29 << 7;
                            unsigned int inverse_31 = inverse_lane_30 | inverse_lane_30 << 16;
                            unsigned int words_32[8];
                            #pragma unroll
                            for (int i_58 = 0; i_58 < 8; i_58++) {
                                uint32_t _bf16x2_mul_56;
                                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_56) : "r"(block_25[i_58 * 2]), "r"(inverse_31));
                                uint16_t _e4m3x2_56;
                                asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_56) : "r"(_bf16x2_mul_56));
                                uint32_t _bf16x2_mul_57;
                                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_57) : "r"(block_25[i_58 * 2 + 1]), "r"(inverse_31));
                                uint16_t _e4m3x2_57;
                                asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_57) : "r"(_bf16x2_mul_57));
                                words_32[i_58] = (unsigned int)_e4m3x2_56 | (unsigned int)_e4m3x2_57 << 16;
                            }
                            scale_word_24 = scale_word_24 | scale_byte_29;
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v57_addr + (unsigned int)(tid * 32)), "r"(words_32[0]), "r"(words_32[1]), "r"(words_32[2]), "r"(words_32[3]) : "memory");
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v57_addr + (unsigned int)(tid * 32 + 16)), "r"(words_32[4]), "r"(words_32[5]), "r"(words_32[6]), "r"(words_32[7]) : "memory");
                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                            if (tid == 0) {
                                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                asm volatile(
                                    "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                    " [%0, {%1, %2}], [%3], %4;"
                                    :: "l"((&gate_q_store)), "r"(y_5 * 256 + 128), "r"(x_5 * 256 + cta_rank_0 * 128), "r"(smem_v57_addr), "l"(0x12F0000000000000ULL) : "memory");
                                asm volatile("cp.async.bulk.commit_group;");
                            }
                            unsigned int block_33[16];
                            #pragma unroll
                            for (int j_35 = 0; j_35 < 16; j_35++) {
                                block_33[j_35] = packed_7[80 + j_35];
                            }
                            #pragma unroll
                            for (int j_36 = 0; j_36 < 4; j_36++) {
                                unsigned int address_24 = d_smem_addr + 8192 + (unsigned int)(tid * 64) + (unsigned int)(j_36 * 16);
                                address_24 = address_24 ^ (address_24 & 511) >> 7 << 4;
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v56_addr + (address_24 - smem_v56_addr)), "r"(packed_7[80 + 4 * j_36]), "r"(packed_7[80 + 4 * j_36 + 1]), "r"(packed_7[80 + 4 * j_36 + 2]), "r"(packed_7[80 + 4 * j_36 + 3]) : "memory");
                            }
                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                            if (tid == 0) {
                                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                asm volatile(
                                    "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                    " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                    :: "l"((&gate_out_r)), "r"(0), "r"(x_5 * 256 + cta_rank_0 * 128), "r"(y_5 * 8 + 5), "r"(0), "r"(0), "r"(d_smem_addr + 8192), "l"(0x12F0000000000000ULL) : "memory");
                                asm volatile("cp.async.bulk.commit_group;");
                                asm volatile("cp.async.bulk.wait_group.read 1;");
                            }
                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                            uint32_t _bf16x2_abs_58;
                            asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_58) : "r"(block_33[0]));
                            unsigned int amax_pair_34 = _bf16x2_abs_58;
                            #pragma unroll
                            for (int i_59 = 1; i_59 < 16; i_59++) {
                                uint32_t _bf16x2_abs_59;
                                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_59) : "r"(block_33[i_59]));
                                uint32_t _bf16x2_max_29;
                                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_29) : "r"(amax_pair_34), "r"(_bf16x2_abs_59));
                                amax_pair_34 = _bf16x2_max_29;
                            }
                            uint16_t _bf16_max_29;
                            asm("max.bf16 %0, %1, %2;" : "=h"(_bf16_max_29) : "h"((uint16_t)(amax_pair_34 & 65535)), "h"((uint16_t)(amax_pair_34 >> 16)));
                            float _cvt_f32_bf16_29;
                            asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_29) : "h"((uint16_t)(_bf16_max_29)));
                            float amax_35 = _cvt_f32_bf16_29;
                            float _fmax_29 = fmaxf(amax_35 * 0.002232142857f, 1e-12f);
                            float scale_36 = _fmax_29;
                            uint16_t _ue8m0x2_f32_29;
                            asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_ue8m0x2_f32_29) : "f"(scale_36), "f"(scale_36));
                            unsigned int scale_byte_37 = (unsigned int)_ue8m0x2_f32_29 & 255;
                            unsigned int inverse_lane_38 = 254 - scale_byte_37 << 7;
                            unsigned int inverse_39 = inverse_lane_38 | inverse_lane_38 << 16;
                            unsigned int words_40[8];
                            #pragma unroll
                            for (int i_60 = 0; i_60 < 8; i_60++) {
                                uint32_t _bf16x2_mul_58;
                                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_58) : "r"(block_33[i_60 * 2]), "r"(inverse_39));
                                uint16_t _e4m3x2_58;
                                asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_58) : "r"(_bf16x2_mul_58));
                                uint32_t _bf16x2_mul_59;
                                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_59) : "r"(block_33[i_60 * 2 + 1]), "r"(inverse_39));
                                uint16_t _e4m3x2_59;
                                asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_59) : "r"(_bf16x2_mul_59));
                                words_40[i_60] = (unsigned int)_e4m3x2_58 | (unsigned int)_e4m3x2_59 << 16;
                            }
                            scale_word_24 = scale_word_24 | scale_byte_37 << 8;
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v57_addr + (unsigned int)(tid * 32)), "r"(words_40[0]), "r"(words_40[1]), "r"(words_40[2]), "r"(words_40[3]) : "memory");
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v57_addr + (unsigned int)(tid * 32 + 16)), "r"(words_40[4]), "r"(words_40[5]), "r"(words_40[6]), "r"(words_40[7]) : "memory");
                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                            if (tid == 0) {
                                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                asm volatile(
                                    "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                    " [%0, {%1, %2}], [%3], %4;"
                                    :: "l"((&gate_q_store)), "r"(y_5 * 256 + 160), "r"(x_5 * 256 + cta_rank_0 * 128), "r"(smem_v57_addr), "l"(0x12F0000000000000ULL) : "memory");
                                asm volatile("cp.async.bulk.commit_group;");
                            }
                            unsigned int block_41[16];
                            #pragma unroll
                            for (int j_37 = 0; j_37 < 16; j_37++) {
                                block_41[j_37] = packed_7[96 + j_37];
                            }
                            #pragma unroll
                            for (int j_38 = 0; j_38 < 4; j_38++) {
                                unsigned int address_25 = d_smem_addr + (unsigned int)(tid * 64) + (unsigned int)(j_38 * 16);
                                address_25 = address_25 ^ (address_25 & 511) >> 7 << 4;
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v56_addr + (address_25 - smem_v56_addr)), "r"(packed_7[96 + 4 * j_38]), "r"(packed_7[96 + 4 * j_38 + 1]), "r"(packed_7[96 + 4 * j_38 + 2]), "r"(packed_7[96 + 4 * j_38 + 3]) : "memory");
                            }
                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                            if (tid == 0) {
                                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                asm volatile(
                                    "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                    " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                    :: "l"((&gate_out_r)), "r"(0), "r"(x_5 * 256 + cta_rank_0 * 128), "r"(y_5 * 8 + 6), "r"(0), "r"(0), "r"(d_smem_addr), "l"(0x12F0000000000000ULL) : "memory");
                                asm volatile("cp.async.bulk.commit_group;");
                                asm volatile("cp.async.bulk.wait_group.read 1;");
                            }
                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                            uint32_t _bf16x2_abs_60;
                            asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_60) : "r"(block_41[0]));
                            unsigned int amax_pair_42 = _bf16x2_abs_60;
                            #pragma unroll
                            for (int i_61 = 1; i_61 < 16; i_61++) {
                                uint32_t _bf16x2_abs_61;
                                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_61) : "r"(block_41[i_61]));
                                uint32_t _bf16x2_max_30;
                                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_30) : "r"(amax_pair_42), "r"(_bf16x2_abs_61));
                                amax_pair_42 = _bf16x2_max_30;
                            }
                            uint16_t _bf16_max_30;
                            asm("max.bf16 %0, %1, %2;" : "=h"(_bf16_max_30) : "h"((uint16_t)(amax_pair_42 & 65535)), "h"((uint16_t)(amax_pair_42 >> 16)));
                            float _cvt_f32_bf16_30;
                            asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_30) : "h"((uint16_t)(_bf16_max_30)));
                            float amax_43 = _cvt_f32_bf16_30;
                            float _fmax_30 = fmaxf(amax_43 * 0.002232142857f, 1e-12f);
                            float scale_44 = _fmax_30;
                            uint16_t _ue8m0x2_f32_30;
                            asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_ue8m0x2_f32_30) : "f"(scale_44), "f"(scale_44));
                            unsigned int scale_byte_45 = (unsigned int)_ue8m0x2_f32_30 & 255;
                            unsigned int inverse_lane_46 = 254 - scale_byte_45 << 7;
                            unsigned int inverse_47 = inverse_lane_46 | inverse_lane_46 << 16;
                            unsigned int words_48[8];
                            #pragma unroll
                            for (int i_62 = 0; i_62 < 8; i_62++) {
                                uint32_t _bf16x2_mul_60;
                                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_60) : "r"(block_41[i_62 * 2]), "r"(inverse_47));
                                uint16_t _e4m3x2_60;
                                asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_60) : "r"(_bf16x2_mul_60));
                                uint32_t _bf16x2_mul_61;
                                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_61) : "r"(block_41[i_62 * 2 + 1]), "r"(inverse_47));
                                uint16_t _e4m3x2_61;
                                asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_61) : "r"(_bf16x2_mul_61));
                                words_48[i_62] = (unsigned int)_e4m3x2_60 | (unsigned int)_e4m3x2_61 << 16;
                            }
                            scale_word_24 = scale_word_24 | scale_byte_45 << 16;
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v57_addr + (unsigned int)(tid * 32)), "r"(words_48[0]), "r"(words_48[1]), "r"(words_48[2]), "r"(words_48[3]) : "memory");
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v57_addr + (unsigned int)(tid * 32 + 16)), "r"(words_48[4]), "r"(words_48[5]), "r"(words_48[6]), "r"(words_48[7]) : "memory");
                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                            if (tid == 0) {
                                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                asm volatile(
                                    "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                    " [%0, {%1, %2}], [%3], %4;"
                                    :: "l"((&gate_q_store)), "r"(y_5 * 256 + 192), "r"(x_5 * 256 + cta_rank_0 * 128), "r"(smem_v57_addr), "l"(0x12F0000000000000ULL) : "memory");
                                asm volatile("cp.async.bulk.commit_group;");
                            }
                            unsigned int block_49[16];
                            #pragma unroll
                            for (int j_39 = 0; j_39 < 16; j_39++) {
                                block_49[j_39] = packed_7[112 + j_39];
                            }
                            #pragma unroll
                            for (int j_40 = 0; j_40 < 4; j_40++) {
                                unsigned int address_26 = d_smem_addr + 8192 + (unsigned int)(tid * 64) + (unsigned int)(j_40 * 16);
                                address_26 = address_26 ^ (address_26 & 511) >> 7 << 4;
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v56_addr + (address_26 - smem_v56_addr)), "r"(packed_7[112 + 4 * j_40]), "r"(packed_7[112 + 4 * j_40 + 1]), "r"(packed_7[112 + 4 * j_40 + 2]), "r"(packed_7[112 + 4 * j_40 + 3]) : "memory");
                            }
                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                            if (tid == 0) {
                                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                asm volatile(
                                    "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                    " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                    :: "l"((&gate_out_r)), "r"(0), "r"(x_5 * 256 + cta_rank_0 * 128), "r"(y_5 * 8 + 7), "r"(0), "r"(0), "r"(d_smem_addr + 8192), "l"(0x12F0000000000000ULL) : "memory");
                                asm volatile("cp.async.bulk.commit_group;");
                                asm volatile("cp.async.bulk.wait_group.read 1;");
                            }
                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                            uint32_t _bf16x2_abs_62;
                            asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_62) : "r"(block_49[0]));
                            unsigned int amax_pair_50 = _bf16x2_abs_62;
                            #pragma unroll
                            for (int i_63 = 1; i_63 < 16; i_63++) {
                                uint32_t _bf16x2_abs_63;
                                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_63) : "r"(block_49[i_63]));
                                uint32_t _bf16x2_max_31;
                                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_31) : "r"(amax_pair_50), "r"(_bf16x2_abs_63));
                                amax_pair_50 = _bf16x2_max_31;
                            }
                            uint16_t _bf16_max_31;
                            asm("max.bf16 %0, %1, %2;" : "=h"(_bf16_max_31) : "h"((uint16_t)(amax_pair_50 & 65535)), "h"((uint16_t)(amax_pair_50 >> 16)));
                            float _cvt_f32_bf16_31;
                            asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_31) : "h"((uint16_t)(_bf16_max_31)));
                            float amax_51 = _cvt_f32_bf16_31;
                            float _fmax_31 = fmaxf(amax_51 * 0.002232142857f, 1e-12f);
                            float scale_52 = _fmax_31;
                            uint16_t _ue8m0x2_f32_31;
                            asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_ue8m0x2_f32_31) : "f"(scale_52), "f"(scale_52));
                            unsigned int scale_byte_53 = (unsigned int)_ue8m0x2_f32_31 & 255;
                            unsigned int inverse_lane_54 = 254 - scale_byte_53 << 7;
                            unsigned int inverse_55 = inverse_lane_54 | inverse_lane_54 << 16;
                            unsigned int words_56[8];
                            #pragma unroll
                            for (int i_64 = 0; i_64 < 8; i_64++) {
                                uint32_t _bf16x2_mul_62;
                                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_62) : "r"(block_49[i_64 * 2]), "r"(inverse_55));
                                uint16_t _e4m3x2_62;
                                asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_62) : "r"(_bf16x2_mul_62));
                                uint32_t _bf16x2_mul_63;
                                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_63) : "r"(block_49[i_64 * 2 + 1]), "r"(inverse_55));
                                uint16_t _e4m3x2_63;
                                asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_63) : "r"(_bf16x2_mul_63));
                                words_56[i_64] = (unsigned int)_e4m3x2_62 | (unsigned int)_e4m3x2_63 << 16;
                            }
                            scale_word_24 = scale_word_24 | scale_byte_53 << 24;
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v57_addr + (unsigned int)(tid * 32)), "r"(words_56[0]), "r"(words_56[1]), "r"(words_56[2]), "r"(words_56[3]) : "memory");
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v57_addr + (unsigned int)(tid * 32 + 16)), "r"(words_56[4]), "r"(words_56[5]), "r"(words_56[6]), "r"(words_56[7]) : "memory");
                            smem_v59[tid % 32 * 4 + tid / 32] = scale_word_24;
                            scale_word_24 = 0;
                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                            if (tid == 0) {
                                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                asm volatile(
                                    "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                    " [%0, {%1, %2}], [%3], %4;"
                                    :: "l"((&gate_q_store)), "r"(y_5 * 256 + 224), "r"(x_5 * 256 + cta_rank_0 * 128), "r"(smem_v57_addr), "l"(0x12F0000000000000ULL) : "memory");
                                tma_store_3d((&gate_sc_store), 0, 0, (x_5 * 2 + cta_rank_0) * i_tiles + y_5 * 2 + 1, smem_v59_addr);
                                asm volatile("cp.async.bulk.commit_group;");
                            }
                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                            if (tid == 0) {
                                asm volatile("cp.async.bulk.wait_group.read 0;");
                            }
                            asm volatile("barrier.sync 4, 128;" ::: "memory");
                            phase_bits_9 = phase_bits_9 ^ 64;
                            if (tid / 32 == 0) {
                                if (warp == 0) {
                                    if (elect_sync()) {
                                        asm volatile("cp.async.bulk.wait_group 0;");
                                        bool enabled_value_8 = 1;
                                        if (enabled_value_8 != 0) {
                                            asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(replay_gu)) + ((macro_rows_5 + x_5) * (intermediate / 256) + y_5))), "r"(static_cast<unsigned int>(1)) : "memory");
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                gemm_phase = phase_bits_9;
            } else {
                if (kind == 7) {
                    int col_blocks_10 = (intermediate + 256 - 1) / 256;
                    int x_6 = -1;
                    int y_6 = -1;
                    int expert_6 = -1;
                    int k_start_6 = 0;
                    int k_end_6 = 0;
                    int first_6 = 0;
                    int first_block_1 = (macro_1 * (macro_size / mini_size) + mini_1) * (mini_size / 256);
                    int _min_44 = ((first_block_1 + mini_size / 256) < (tokens / 256) ? (first_block_1 + mini_size / 256) : (tokens / 256));
                    int end_block_1 = _min_44;
                    int block_2 = first_block_1 + task_3 / col_blocks_10;
                    if (block_2 < end_block_1) {
                        int index_3 = counts[3 * experts + block_2];
                        int offset_7 = counts[experts + index_3] / 256;
                        int _max_14 = ((first_block_1) > (offset_7) ? (first_block_1) : (offset_7));
                        int first_row_2 = _max_14;
                        int _min_45 = ((end_block_1) < (offset_7 + counts[index_3] / 256) ? (end_block_1) : (offset_7 + counts[index_3] / 256));
                        int rows_4 = _min_45 - first_row_2;
                        int supergroup_6 = (task_3 - (first_row_2 - first_block_1) * col_blocks_10) / (rows_4 * 8);
                        int full_cols_6 = col_blocks_10 / 8 * 8;
                        int row_18 = 0;
                        int col_61 = 0;
                        if (task_3 - (first_row_2 - first_block_1) * col_blocks_10 < rows_4 * full_cols_6) {
                            row_18 = (task_3 - (first_row_2 - first_block_1) * col_blocks_10) % (rows_4 * 8) / 8;
                            col_61 = supergroup_6 * 8 + (task_3 - (first_row_2 - first_block_1) * col_blocks_10) % 8;
                        } else {
                            row_18 = (task_3 - (first_row_2 - first_block_1) * col_blocks_10 - rows_4 * full_cols_6) / (col_blocks_10 - full_cols_6);
                            col_61 = full_cols_6 + (task_3 - (first_row_2 - first_block_1) * col_blocks_10 - rows_4 * full_cols_6) % (col_blocks_10 - full_cols_6);
                        }
                        if ((supergroup_6 & 1) != 0) {
                            row_18 = rows_4 - row_18 - 1;
                        }
                        x_6 = first_row_2 + row_18 - macro_1 * (macro_size / 256);
                        y_6 = col_61;
                        expert_6 = index_3;
                    }
                    unsigned int phase_bits_10 = gemm_phase;
                    int has_hi_6 = 0;
                    int global_mini_6 = macro_1 * (macro_size / mini_size) + mini_1;
                    int macro_rows_6 = macro_1 * (macro_size / 256);
                    int iterations_6 = hidden / 128;
                    int macro_k_1 = macro_1 * (macro_size / 128);
                    if (expert_6 < 0) {
                        if (tid == 0) {
                        }
                    } else if (tid / 32 == 7) {
                        if (warp == 7) {
                            if (elect_sync()) {
                                {
                                    int _min_46 = ((mini_size) < (tokens - global_mini_6 * mini_size) ? (mini_size) : (tokens - global_mini_6 * mini_size));
                                    int _max_15 = ((0) > (_min_46) ? (0) : (_min_46));
                                    int mini_rows_8 = _max_15;
                                    int required_7 = (mini_rows_8 + 127) / 128 * ((hidden + 511) / 512);
                                    bool enabled_value_9 = 1;
                                    if (enabled_value_9 != 0) {
                                        int32_t _relaxed_ld_26;
                                        asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_26) : "l"(replay_x + global_mini_6) : "memory");
                                        int value_13 = _relaxed_ld_26;
                                        while (value_13 < required_7) {
                                            asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                            int32_t _relaxed_ld_27;
                                            asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_27) : "l"(replay_x + global_mini_6) : "memory");
                                            value_13 = _relaxed_ld_27;
                                        }
                                        asm volatile("fence.acquire.gpu;" ::: "memory");
                                    }
                                }
                                unsigned int previous_7 = phase_bits_10 >> 7 & 1;
                                unsigned int bits_7 = phase_bits_10;
                                if (previous_7 != 1) {
                                    mbarrier_wait(gemm_finished_addr, bits_7 >> 16 & 1);
                                    mbarrier_wait(gemm_finished_addr + 8, bits_7 >> 17 & 1);
                                    mbarrier_wait(gemm_finished_addr + 16, bits_7 >> 18 & 1);
                                    mbarrier_wait(gemm_finished_addr + 24, bits_7 >> 19 & 1);
                                    mbarrier_wait(gemm_finished_addr + 32, bits_7 >> 20 & 1);
                                    mbarrier_wait(gemm_finished_addr + 40, bits_7 >> 21 & 1);
                                    bits_7 = bits_7 ^ 128;
                                }
                                phase_bits_10 = bits_7;
                                int ring_13 = 0;
                                #pragma unroll 1
                                for (int idx_13 = 0; idx_13 < iterations_6; idx_13++) {
                                    mbarrier_wait(gemm_finished_addr + (ring_13) * 8, phase_bits_10 >> (unsigned int)(16 + ring_13) & 1);
                                    asm volatile(
                                        "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                        " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                        :: "r"(smem_v52_addr + (unsigned int)(ring_13 * 16384)), "l"((&x_q)), "r"(0), "r"(x_6 * 256 + cta_rank_0 * 128), "r"(idx_13), "r"(0), "r"(0),
                                           "r"(((gemm_arrived_addr + (ring_13) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                    asm volatile(
                                        "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                        " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                        :: "r"(smem_v53_addr + (unsigned int)(ring_13 * 16384)), "l"((&wu_q)), "r"(0), "r"(y_6 * 256 + cta_rank_0 * 128), "r"(idx_13), "r"(expert_6), "r"(0),
                                           "r"(((gemm_arrived_addr + (ring_13) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                    phase_bits_10 = phase_bits_10 ^ (unsigned int)(1 << 16 + ring_13);
                                    ring_13 = (ring_13 + 1) % 6;
                                }
                            }
                        }
                    } else {
                        if (tid / 32 == 6) {
                            if (warp == 6) {
                                if (elect_sync()) {
                                    {
                                        int _min_47 = ((mini_size) < (tokens - global_mini_6 * mini_size) ? (mini_size) : (tokens - global_mini_6 * mini_size));
                                        int _max_16 = ((0) > (_min_47) ? (0) : (_min_47));
                                        int mini_rows_9 = _max_16;
                                        int required_8 = (mini_rows_9 + 127) / 128 * ((hidden + 511) / 512);
                                        bool enabled_value_10 = 1;
                                        if (enabled_value_10 != 0) {
                                            int32_t _relaxed_ld_28;
                                            asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_28) : "l"(replay_x + global_mini_6) : "memory");
                                            int value_14 = _relaxed_ld_28;
                                            while (value_14 < required_8) {
                                                asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                                int32_t _relaxed_ld_29;
                                                asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_29) : "l"(replay_x + global_mini_6) : "memory");
                                                value_14 = _relaxed_ld_29;
                                            }
                                            asm volatile("fence.acquire.gpu;" ::: "memory");
                                        }
                                    }
                                    unsigned int previous_8 = phase_bits_10 >> 7 & 1;
                                    unsigned int bits_8 = phase_bits_10;
                                    if (previous_8 != 1) {
                                        mbarrier_wait(scales_finished_addr, bits_8 >> 16 & 1);
                                        mbarrier_wait(scales_finished_addr + 8, bits_8 >> 17 & 1);
                                        mbarrier_wait(scales_finished_addr + 16, bits_8 >> 18 & 1);
                                        mbarrier_wait(scales_finished_addr + 24, bits_8 >> 19 & 1);
                                        mbarrier_wait(scales_finished_addr + 32, bits_8 >> 20 & 1);
                                        mbarrier_wait(scales_finished_addr + 40, bits_8 >> 21 & 1);
                                        bits_8 = bits_8 ^ 128;
                                    }
                                    phase_bits_10 = bits_8;
                                    int ring_14 = 0;
                                    #pragma unroll 1
                                    for (int idx_14 = 0; idx_14 < iterations_6; idx_14++) {
                                        mbarrier_wait(scales_finished_addr + (ring_14) * 8, phase_bits_10 >> (unsigned int)(16 + ring_14) & 1);
                                        asm volatile(
                                            "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                            " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                            :: "r"(smem_v54_addr + (unsigned int)(ring_14 * 512)), "l"((&x_sc)), "r"(0), "r"(0), "r"((x_6 * 2 + cta_rank_0) * (hidden / 128) + idx_14),
                                               "r"(((scales_arrived_addr + (ring_14) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                        asm volatile(
                                            "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                            " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                            :: "r"(smem_v55_addr + (unsigned int)(ring_14 * 1024) + (unsigned int)(cta_rank_0 * 512)), "l"((&wu_sc)), "r"(0), "r"(0), "r"((expert_6 * i_tiles_5 + y_6 * 2 + cta_rank_0) * (hidden / 128) + idx_14),
                                               "r"(((scales_arrived_addr + (ring_14) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(3)) : "memory");
                                        phase_bits_10 = phase_bits_10 ^ (unsigned int)(1 << 16 + ring_14);
                                        ring_14 = (ring_14 + 1) % 6;
                                    }
                                }
                            }
                        } else if (tid / 32 == 4 && cta_rank_0 == 0) {
                            if (warp == 4) {
                                if (elect_sync()) {
                                    int ring_15 = 0;
                                    mbarrier_wait(output_finished_addr, phase_bits_10 >> 22 & 1);
                                    phase_bits_10 = phase_bits_10 ^ 4194304;
                                    asm volatile("tcgen05.fence::after_thread_sync;");
                                    #pragma unroll 1
                                    for (int idx_15 = 0; idx_15 < iterations_6; idx_15++) {
                                        mbarrier_arrive_expect_tx(scales_arrived_addr + (ring_15) * 8, 3072);
                                        mbarrier_wait(scales_arrived_addr + (ring_15) * 8, phase_bits_10 >> (unsigned int)(8 + ring_15) & 1);
                                        int buffer_1 = idx_15 % 3;
                                        tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfa + buffer_1 * 4, make_sf_cp_desc_sbo128(smem_v54_addr + (unsigned int)(ring_15 * 512)));
                                        tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfb + buffer_1 * 8, make_sf_cp_desc_sbo128(smem_v55_addr + (unsigned int)(ring_15 * 1024)));
                                        tcgen05_cp_32x128b_warpx4_cta2((tmem_tmem_sfb + buffer_1 * 8 + 4), make_sf_cp_desc_sbo128((smem_v55_addr + (unsigned int)(ring_15 * 1024) + 512)));
                                        tcgen05_commit_cg2_multicast(scales_finished_addr + (ring_15) * 8, (uint16_t)(3));
                                        mbarrier_arrive_expect_tx(gemm_arrived_addr + (ring_15) * 8, 65536);
                                        mbarrier_wait(gemm_arrived_addr + (ring_15) * 8, phase_bits_10 >> (unsigned int)ring_15 & 1);
                                        int _mma_a_lo_11 = (((smem_v52_addr) >> 4) & 0x3FFF) + (ring_15) * 1024;
                                        int _mma_b_lo_11 = (((smem_v53_addr) >> 4) & 0x3FFF) + (ring_15) * 1024;
                                        {
                                            uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_11) | ((uint64_t)0x40004040 << 32);
                                            uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_11) | ((uint64_t)0x40004040 << 32);

                                            tcgen05_mma_mxf8_bs_cta2(tmem_accumulator, a_desc + 0, b_desc + 0,
                                                0x10c00000U, tmem_tmem_sfa + buffer_1 * 4, tmem_tmem_sfb + buffer_1 * 8, ((idx_15 == 0) ? 0 : 1));
                                            tcgen05_mma_mxf8_bs_cta2(tmem_accumulator, a_desc + 2, b_desc + 2,
                                                0x30c00010U, tmem_tmem_sfa + buffer_1 * 4, tmem_tmem_sfb + buffer_1 * 8, 1);
                                            tcgen05_mma_mxf8_bs_cta2(tmem_accumulator, a_desc + 4, b_desc + 4,
                                                0x50c00020U, tmem_tmem_sfa + buffer_1 * 4, tmem_tmem_sfb + buffer_1 * 8, 1);
                                            tcgen05_mma_mxf8_bs_cta2(tmem_accumulator, a_desc + 6, b_desc + 6,
                                                0x70c00030U, tmem_tmem_sfa + buffer_1 * 4, tmem_tmem_sfb + buffer_1 * 8, 1);
                                        }
                                        tcgen05_commit_cg2_multicast(gemm_finished_addr + (ring_15) * 8, (uint16_t)(3));
                                        phase_bits_10 = phase_bits_10 ^ (unsigned int)(1 << ring_15) ^ (unsigned int)(1 << 8 + ring_15);
                                        ring_15 = (ring_15 + 1) % 6;
                                    }
                                    tcgen05_commit_cg2_multicast(output_arrived_addr, (uint16_t)(3));
                                }
                            }
                        } else {
                            if (tid < 128) {
                                mbarrier_wait(output_arrived_addr, phase_bits_10 >> 6 & 1);
                                int warp_row_4 = tid / 32 * 32;
                                unsigned int packed_8[128];
                                #pragma unroll
                                for (int i_65 = 0; i_65 < 8; i_65++) {
                                    float _tmem_load_11[32];
                                    asm volatile(
                                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                                        : "=f"(_tmem_load_11[0]), "=f"(_tmem_load_11[1]), "=f"(_tmem_load_11[2]), "=f"(_tmem_load_11[3]), "=f"(_tmem_load_11[4]), "=f"(_tmem_load_11[5]), "=f"(_tmem_load_11[6]), "=f"(_tmem_load_11[7]), "=f"(_tmem_load_11[8]), "=f"(_tmem_load_11[9]), "=f"(_tmem_load_11[10]), "=f"(_tmem_load_11[11]), "=f"(_tmem_load_11[12]), "=f"(_tmem_load_11[13]), "=f"(_tmem_load_11[14]), "=f"(_tmem_load_11[15]), "=f"(_tmem_load_11[16]), "=f"(_tmem_load_11[17]), "=f"(_tmem_load_11[18]), "=f"(_tmem_load_11[19]), "=f"(_tmem_load_11[20]), "=f"(_tmem_load_11[21]), "=f"(_tmem_load_11[22]), "=f"(_tmem_load_11[23]), "=f"(_tmem_load_11[24]), "=f"(_tmem_load_11[25]), "=f"(_tmem_load_11[26]), "=f"(_tmem_load_11[27]), "=f"(_tmem_load_11[28]), "=f"(_tmem_load_11[29]), "=f"(_tmem_load_11[30]), "=f"(_tmem_load_11[31])
                                        : "r"(taddr_1 + (unsigned int)(warp_row_4 << 16) + (unsigned int)(i_65 * 32)));
                                    #pragma unroll
                                    for (int j_41 = 0; j_41 < 16; j_41++) {
                                        __nv_bfloat162 _bf16x2_27 = __float22bfloat162_rn(make_float2(_tmem_load_11[2 * j_41], _tmem_load_11[2 * j_41 + 1]));
                                        packed_8[i_65 * 16 + j_41] = __as_u32(_bf16x2_27);
                                    }
                                }
                                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                if (tid == 0) {
                                    asm volatile(
                                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                                        :: "r"((output_finished_addr) & 0xFEFFFFFF) : "memory");
                                }
                                unsigned int scale_word_25 = 0;
                                unsigned int block_0_1[16];
                                #pragma unroll
                                for (int j_42 = 0; j_42 < 16; j_42++) {
                                    block_0_1[j_42] = packed_8[j_42];
                                }
                                #pragma unroll
                                for (int j_43 = 0; j_43 < 4; j_43++) {
                                    unsigned int address_27 = d_smem_addr + (unsigned int)(tid * 64) + (unsigned int)(j_43 * 16);
                                    address_27 = address_27 ^ (address_27 & 511) >> 7 << 4;
                                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v56_addr + (address_27 - smem_v56_addr)), "r"(packed_8[4 * j_43]), "r"(packed_8[4 * j_43 + 1]), "r"(packed_8[4 * j_43 + 2]), "r"(packed_8[4 * j_43 + 3]) : "memory");
                                }
                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                if (tid == 0) {
                                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                    asm volatile(
                                        "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                        " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                        :: "l"((&up_out_r)), "r"(0), "r"(x_6 * 256 + cta_rank_0 * 128), "r"(y_6 * 8), "r"(0), "r"(0), "r"(d_smem_addr), "l"(0x12F0000000000000ULL) : "memory");
                                    asm volatile("cp.async.bulk.commit_group;");
                                    asm volatile("cp.async.bulk.wait_group.read 1;");
                                }
                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                uint32_t _bf16x2_abs_64;
                                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_64) : "r"(block_0_1[0]));
                                unsigned int amax_pair_25 = _bf16x2_abs_64;
                                #pragma unroll
                                for (int i_66 = 1; i_66 < 16; i_66++) {
                                    uint32_t _bf16x2_abs_65;
                                    asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_65) : "r"(block_0_1[i_66]));
                                    uint32_t _bf16x2_max_32;
                                    asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_32) : "r"(amax_pair_25), "r"(_bf16x2_abs_65));
                                    amax_pair_25 = _bf16x2_max_32;
                                }
                                uint16_t _bf16_max_32;
                                asm("max.bf16 %0, %1, %2;" : "=h"(_bf16_max_32) : "h"((uint16_t)(amax_pair_25 & 65535)), "h"((uint16_t)(amax_pair_25 >> 16)));
                                float _cvt_f32_bf16_32;
                                asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_32) : "h"((uint16_t)(_bf16_max_32)));
                                float amax_25 = _cvt_f32_bf16_32;
                                float _fmax_32 = fmaxf(amax_25 * 0.002232142857f, 1e-12f);
                                float scale_25 = _fmax_32;
                                uint16_t _ue8m0x2_f32_32;
                                asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_ue8m0x2_f32_32) : "f"(scale_25), "f"(scale_25));
                                unsigned int scale_byte_25 = (unsigned int)_ue8m0x2_f32_32 & 255;
                                unsigned int inverse_lane_25 = 254 - scale_byte_25 << 7;
                                unsigned int inverse_25 = inverse_lane_25 | inverse_lane_25 << 16;
                                unsigned int words_25[8];
                                #pragma unroll
                                for (int i_67 = 0; i_67 < 8; i_67++) {
                                    uint32_t _bf16x2_mul_64;
                                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_64) : "r"(block_0_1[i_67 * 2]), "r"(inverse_25));
                                    uint16_t _e4m3x2_64;
                                    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_64) : "r"(_bf16x2_mul_64));
                                    uint32_t _bf16x2_mul_65;
                                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_65) : "r"(block_0_1[i_67 * 2 + 1]), "r"(inverse_25));
                                    uint16_t _e4m3x2_65;
                                    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_65) : "r"(_bf16x2_mul_65));
                                    words_25[i_67] = (unsigned int)_e4m3x2_64 | (unsigned int)_e4m3x2_65 << 16;
                                }
                                scale_word_25 = scale_word_25 | scale_byte_25;
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v57_addr + (unsigned int)(tid * 32)), "r"(words_25[0]), "r"(words_25[1]), "r"(words_25[2]), "r"(words_25[3]) : "memory");
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v57_addr + (unsigned int)(tid * 32 + 16)), "r"(words_25[4]), "r"(words_25[5]), "r"(words_25[6]), "r"(words_25[7]) : "memory");
                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                if (tid == 0) {
                                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                    asm volatile(
                                        "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                        " [%0, {%1, %2}], [%3], %4;"
                                        :: "l"((&up_q_store)), "r"(y_6 * 256), "r"(x_6 * 256 + cta_rank_0 * 128), "r"(smem_v57_addr), "l"(0x12F0000000000000ULL) : "memory");
                                    asm volatile("cp.async.bulk.commit_group;");
                                }
                                unsigned int block_1_1[16];
                                #pragma unroll
                                for (int j_44 = 0; j_44 < 16; j_44++) {
                                    block_1_1[j_44] = packed_8[16 + j_44];
                                }
                                #pragma unroll
                                for (int j_45 = 0; j_45 < 4; j_45++) {
                                    unsigned int address_28 = d_smem_addr + 8192 + (unsigned int)(tid * 64) + (unsigned int)(j_45 * 16);
                                    address_28 = address_28 ^ (address_28 & 511) >> 7 << 4;
                                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v56_addr + (address_28 - smem_v56_addr)), "r"(packed_8[16 + 4 * j_45]), "r"(packed_8[16 + 4 * j_45 + 1]), "r"(packed_8[16 + 4 * j_45 + 2]), "r"(packed_8[16 + 4 * j_45 + 3]) : "memory");
                                }
                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                if (tid == 0) {
                                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                    asm volatile(
                                        "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                        " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                        :: "l"((&up_out_r)), "r"(0), "r"(x_6 * 256 + cta_rank_0 * 128), "r"(y_6 * 8 + 1), "r"(0), "r"(0), "r"(d_smem_addr + 8192), "l"(0x12F0000000000000ULL) : "memory");
                                    asm volatile("cp.async.bulk.commit_group;");
                                    asm volatile("cp.async.bulk.wait_group.read 1;");
                                }
                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                uint32_t _bf16x2_abs_66;
                                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_66) : "r"(block_1_1[0]));
                                unsigned int amax_pair_2_2 = _bf16x2_abs_66;
                                #pragma unroll
                                for (int i_68 = 1; i_68 < 16; i_68++) {
                                    uint32_t _bf16x2_abs_67;
                                    asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_67) : "r"(block_1_1[i_68]));
                                    uint32_t _bf16x2_max_33;
                                    asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_33) : "r"(amax_pair_2_2), "r"(_bf16x2_abs_67));
                                    amax_pair_2_2 = _bf16x2_max_33;
                                }
                                uint16_t _bf16_max_33;
                                asm("max.bf16 %0, %1, %2;" : "=h"(_bf16_max_33) : "h"((uint16_t)(amax_pair_2_2 & 65535)), "h"((uint16_t)(amax_pair_2_2 >> 16)));
                                float _cvt_f32_bf16_33;
                                asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_33) : "h"((uint16_t)(_bf16_max_33)));
                                float amax_3_2 = _cvt_f32_bf16_33;
                                float _fmax_33 = fmaxf(amax_3_2 * 0.002232142857f, 1e-12f);
                                float scale_4_2 = _fmax_33;
                                uint16_t _ue8m0x2_f32_33;
                                asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_ue8m0x2_f32_33) : "f"(scale_4_2), "f"(scale_4_2));
                                unsigned int scale_byte_5_2 = (unsigned int)_ue8m0x2_f32_33 & 255;
                                unsigned int inverse_lane_6_2 = 254 - scale_byte_5_2 << 7;
                                unsigned int inverse_7_2 = inverse_lane_6_2 | inverse_lane_6_2 << 16;
                                unsigned int words_8_2[8];
                                #pragma unroll
                                for (int i_69 = 0; i_69 < 8; i_69++) {
                                    uint32_t _bf16x2_mul_66;
                                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_66) : "r"(block_1_1[i_69 * 2]), "r"(inverse_7_2));
                                    uint16_t _e4m3x2_66;
                                    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_66) : "r"(_bf16x2_mul_66));
                                    uint32_t _bf16x2_mul_67;
                                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_67) : "r"(block_1_1[i_69 * 2 + 1]), "r"(inverse_7_2));
                                    uint16_t _e4m3x2_67;
                                    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_67) : "r"(_bf16x2_mul_67));
                                    words_8_2[i_69] = (unsigned int)_e4m3x2_66 | (unsigned int)_e4m3x2_67 << 16;
                                }
                                scale_word_25 = scale_word_25 | scale_byte_5_2 << 8;
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v57_addr + (unsigned int)(tid * 32)), "r"(words_8_2[0]), "r"(words_8_2[1]), "r"(words_8_2[2]), "r"(words_8_2[3]) : "memory");
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v57_addr + (unsigned int)(tid * 32 + 16)), "r"(words_8_2[4]), "r"(words_8_2[5]), "r"(words_8_2[6]), "r"(words_8_2[7]) : "memory");
                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                if (tid == 0) {
                                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                    asm volatile(
                                        "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                        " [%0, {%1, %2}], [%3], %4;"
                                        :: "l"((&up_q_store)), "r"(y_6 * 256 + 32), "r"(x_6 * 256 + cta_rank_0 * 128), "r"(smem_v57_addr), "l"(0x12F0000000000000ULL) : "memory");
                                    asm volatile("cp.async.bulk.commit_group;");
                                }
                                unsigned int block_9_1[16];
                                #pragma unroll
                                for (int j_46 = 0; j_46 < 16; j_46++) {
                                    block_9_1[j_46] = packed_8[32 + j_46];
                                }
                                #pragma unroll
                                for (int j_47 = 0; j_47 < 4; j_47++) {
                                    unsigned int address_29 = d_smem_addr + (unsigned int)(tid * 64) + (unsigned int)(j_47 * 16);
                                    address_29 = address_29 ^ (address_29 & 511) >> 7 << 4;
                                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v56_addr + (address_29 - smem_v56_addr)), "r"(packed_8[32 + 4 * j_47]), "r"(packed_8[32 + 4 * j_47 + 1]), "r"(packed_8[32 + 4 * j_47 + 2]), "r"(packed_8[32 + 4 * j_47 + 3]) : "memory");
                                }
                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                if (tid == 0) {
                                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                    asm volatile(
                                        "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                        " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                        :: "l"((&up_out_r)), "r"(0), "r"(x_6 * 256 + cta_rank_0 * 128), "r"(y_6 * 8 + 2), "r"(0), "r"(0), "r"(d_smem_addr), "l"(0x12F0000000000000ULL) : "memory");
                                    asm volatile("cp.async.bulk.commit_group;");
                                    asm volatile("cp.async.bulk.wait_group.read 1;");
                                }
                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                uint32_t _bf16x2_abs_68;
                                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_68) : "r"(block_9_1[0]));
                                unsigned int amax_pair_10_2 = _bf16x2_abs_68;
                                #pragma unroll
                                for (int i_70 = 1; i_70 < 16; i_70++) {
                                    uint32_t _bf16x2_abs_69;
                                    asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_69) : "r"(block_9_1[i_70]));
                                    uint32_t _bf16x2_max_34;
                                    asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_34) : "r"(amax_pair_10_2), "r"(_bf16x2_abs_69));
                                    amax_pair_10_2 = _bf16x2_max_34;
                                }
                                uint16_t _bf16_max_34;
                                asm("max.bf16 %0, %1, %2;" : "=h"(_bf16_max_34) : "h"((uint16_t)(amax_pair_10_2 & 65535)), "h"((uint16_t)(amax_pair_10_2 >> 16)));
                                float _cvt_f32_bf16_34;
                                asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_34) : "h"((uint16_t)(_bf16_max_34)));
                                float amax_11_2 = _cvt_f32_bf16_34;
                                float _fmax_34 = fmaxf(amax_11_2 * 0.002232142857f, 1e-12f);
                                float scale_12_2 = _fmax_34;
                                uint16_t _ue8m0x2_f32_34;
                                asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_ue8m0x2_f32_34) : "f"(scale_12_2), "f"(scale_12_2));
                                unsigned int scale_byte_13_2 = (unsigned int)_ue8m0x2_f32_34 & 255;
                                unsigned int inverse_lane_14_2 = 254 - scale_byte_13_2 << 7;
                                unsigned int inverse_15_2 = inverse_lane_14_2 | inverse_lane_14_2 << 16;
                                unsigned int words_16_2[8];
                                #pragma unroll
                                for (int i_71 = 0; i_71 < 8; i_71++) {
                                    uint32_t _bf16x2_mul_68;
                                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_68) : "r"(block_9_1[i_71 * 2]), "r"(inverse_15_2));
                                    uint16_t _e4m3x2_68;
                                    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_68) : "r"(_bf16x2_mul_68));
                                    uint32_t _bf16x2_mul_69;
                                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_69) : "r"(block_9_1[i_71 * 2 + 1]), "r"(inverse_15_2));
                                    uint16_t _e4m3x2_69;
                                    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_69) : "r"(_bf16x2_mul_69));
                                    words_16_2[i_71] = (unsigned int)_e4m3x2_68 | (unsigned int)_e4m3x2_69 << 16;
                                }
                                scale_word_25 = scale_word_25 | scale_byte_13_2 << 16;
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v57_addr + (unsigned int)(tid * 32)), "r"(words_16_2[0]), "r"(words_16_2[1]), "r"(words_16_2[2]), "r"(words_16_2[3]) : "memory");
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v57_addr + (unsigned int)(tid * 32 + 16)), "r"(words_16_2[4]), "r"(words_16_2[5]), "r"(words_16_2[6]), "r"(words_16_2[7]) : "memory");
                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                if (tid == 0) {
                                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                    asm volatile(
                                        "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                        " [%0, {%1, %2}], [%3], %4;"
                                        :: "l"((&up_q_store)), "r"(y_6 * 256 + 64), "r"(x_6 * 256 + cta_rank_0 * 128), "r"(smem_v57_addr), "l"(0x12F0000000000000ULL) : "memory");
                                    asm volatile("cp.async.bulk.commit_group;");
                                }
                                unsigned int block_17_1[16];
                                #pragma unroll
                                for (int j_48 = 0; j_48 < 16; j_48++) {
                                    block_17_1[j_48] = packed_8[48 + j_48];
                                }
                                #pragma unroll
                                for (int j_49 = 0; j_49 < 4; j_49++) {
                                    unsigned int address_30 = d_smem_addr + 8192 + (unsigned int)(tid * 64) + (unsigned int)(j_49 * 16);
                                    address_30 = address_30 ^ (address_30 & 511) >> 7 << 4;
                                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v56_addr + (address_30 - smem_v56_addr)), "r"(packed_8[48 + 4 * j_49]), "r"(packed_8[48 + 4 * j_49 + 1]), "r"(packed_8[48 + 4 * j_49 + 2]), "r"(packed_8[48 + 4 * j_49 + 3]) : "memory");
                                }
                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                if (tid == 0) {
                                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                    asm volatile(
                                        "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                        " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                        :: "l"((&up_out_r)), "r"(0), "r"(x_6 * 256 + cta_rank_0 * 128), "r"(y_6 * 8 + 3), "r"(0), "r"(0), "r"(d_smem_addr + 8192), "l"(0x12F0000000000000ULL) : "memory");
                                    asm volatile("cp.async.bulk.commit_group;");
                                    asm volatile("cp.async.bulk.wait_group.read 1;");
                                }
                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                uint32_t _bf16x2_abs_70;
                                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_70) : "r"(block_17_1[0]));
                                unsigned int amax_pair_18_2 = _bf16x2_abs_70;
                                #pragma unroll
                                for (int i_72 = 1; i_72 < 16; i_72++) {
                                    uint32_t _bf16x2_abs_71;
                                    asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_71) : "r"(block_17_1[i_72]));
                                    uint32_t _bf16x2_max_35;
                                    asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_35) : "r"(amax_pair_18_2), "r"(_bf16x2_abs_71));
                                    amax_pair_18_2 = _bf16x2_max_35;
                                }
                                uint16_t _bf16_max_35;
                                asm("max.bf16 %0, %1, %2;" : "=h"(_bf16_max_35) : "h"((uint16_t)(amax_pair_18_2 & 65535)), "h"((uint16_t)(amax_pair_18_2 >> 16)));
                                float _cvt_f32_bf16_35;
                                asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_35) : "h"((uint16_t)(_bf16_max_35)));
                                float amax_19_2 = _cvt_f32_bf16_35;
                                float _fmax_35 = fmaxf(amax_19_2 * 0.002232142857f, 1e-12f);
                                float scale_20_2 = _fmax_35;
                                uint16_t _ue8m0x2_f32_35;
                                asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_ue8m0x2_f32_35) : "f"(scale_20_2), "f"(scale_20_2));
                                unsigned int scale_byte_21_2 = (unsigned int)_ue8m0x2_f32_35 & 255;
                                unsigned int inverse_lane_22_2 = 254 - scale_byte_21_2 << 7;
                                unsigned int inverse_23_2 = inverse_lane_22_2 | inverse_lane_22_2 << 16;
                                unsigned int words_24_2[8];
                                #pragma unroll
                                for (int i_73 = 0; i_73 < 8; i_73++) {
                                    uint32_t _bf16x2_mul_70;
                                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_70) : "r"(block_17_1[i_73 * 2]), "r"(inverse_23_2));
                                    uint16_t _e4m3x2_70;
                                    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_70) : "r"(_bf16x2_mul_70));
                                    uint32_t _bf16x2_mul_71;
                                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_71) : "r"(block_17_1[i_73 * 2 + 1]), "r"(inverse_23_2));
                                    uint16_t _e4m3x2_71;
                                    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_71) : "r"(_bf16x2_mul_71));
                                    words_24_2[i_73] = (unsigned int)_e4m3x2_70 | (unsigned int)_e4m3x2_71 << 16;
                                }
                                scale_word_25 = scale_word_25 | scale_byte_21_2 << 24;
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v57_addr + (unsigned int)(tid * 32)), "r"(words_24_2[0]), "r"(words_24_2[1]), "r"(words_24_2[2]), "r"(words_24_2[3]) : "memory");
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v57_addr + (unsigned int)(tid * 32 + 16)), "r"(words_24_2[4]), "r"(words_24_2[5]), "r"(words_24_2[6]), "r"(words_24_2[7]) : "memory");
                                smem_v58[tid % 32 * 4 + tid / 32] = scale_word_25;
                                scale_word_25 = 0;
                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                if (tid == 0) {
                                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                    asm volatile(
                                        "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                        " [%0, {%1, %2}], [%3], %4;"
                                        :: "l"((&up_q_store)), "r"(y_6 * 256 + 96), "r"(x_6 * 256 + cta_rank_0 * 128), "r"(smem_v57_addr), "l"(0x12F0000000000000ULL) : "memory");
                                    tma_store_3d((&up_sc_store), 0, 0, (x_6 * 2 + cta_rank_0) * i_tiles + y_6 * 2, smem_v58_addr);
                                    asm volatile("cp.async.bulk.commit_group;");
                                }
                                unsigned int block_25_1[16];
                                #pragma unroll
                                for (int j_50 = 0; j_50 < 16; j_50++) {
                                    block_25_1[j_50] = packed_8[64 + j_50];
                                }
                                #pragma unroll
                                for (int j_51 = 0; j_51 < 4; j_51++) {
                                    unsigned int address_31 = d_smem_addr + (unsigned int)(tid * 64) + (unsigned int)(j_51 * 16);
                                    address_31 = address_31 ^ (address_31 & 511) >> 7 << 4;
                                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v56_addr + (address_31 - smem_v56_addr)), "r"(packed_8[64 + 4 * j_51]), "r"(packed_8[64 + 4 * j_51 + 1]), "r"(packed_8[64 + 4 * j_51 + 2]), "r"(packed_8[64 + 4 * j_51 + 3]) : "memory");
                                }
                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                if (tid == 0) {
                                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                    asm volatile(
                                        "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                        " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                        :: "l"((&up_out_r)), "r"(0), "r"(x_6 * 256 + cta_rank_0 * 128), "r"(y_6 * 8 + 4), "r"(0), "r"(0), "r"(d_smem_addr), "l"(0x12F0000000000000ULL) : "memory");
                                    asm volatile("cp.async.bulk.commit_group;");
                                    asm volatile("cp.async.bulk.wait_group.read 1;");
                                }
                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                uint32_t _bf16x2_abs_72;
                                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_72) : "r"(block_25_1[0]));
                                unsigned int amax_pair_26_1 = _bf16x2_abs_72;
                                #pragma unroll
                                for (int i_74 = 1; i_74 < 16; i_74++) {
                                    uint32_t _bf16x2_abs_73;
                                    asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_73) : "r"(block_25_1[i_74]));
                                    uint32_t _bf16x2_max_36;
                                    asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_36) : "r"(amax_pair_26_1), "r"(_bf16x2_abs_73));
                                    amax_pair_26_1 = _bf16x2_max_36;
                                }
                                uint16_t _bf16_max_36;
                                asm("max.bf16 %0, %1, %2;" : "=h"(_bf16_max_36) : "h"((uint16_t)(amax_pair_26_1 & 65535)), "h"((uint16_t)(amax_pair_26_1 >> 16)));
                                float _cvt_f32_bf16_36;
                                asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_36) : "h"((uint16_t)(_bf16_max_36)));
                                float amax_27_1 = _cvt_f32_bf16_36;
                                float _fmax_36 = fmaxf(amax_27_1 * 0.002232142857f, 1e-12f);
                                float scale_28_1 = _fmax_36;
                                uint16_t _ue8m0x2_f32_36;
                                asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_ue8m0x2_f32_36) : "f"(scale_28_1), "f"(scale_28_1));
                                unsigned int scale_byte_29_1 = (unsigned int)_ue8m0x2_f32_36 & 255;
                                unsigned int inverse_lane_30_1 = 254 - scale_byte_29_1 << 7;
                                unsigned int inverse_31_1 = inverse_lane_30_1 | inverse_lane_30_1 << 16;
                                unsigned int words_32_1[8];
                                #pragma unroll
                                for (int i_75 = 0; i_75 < 8; i_75++) {
                                    uint32_t _bf16x2_mul_72;
                                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_72) : "r"(block_25_1[i_75 * 2]), "r"(inverse_31_1));
                                    uint16_t _e4m3x2_72;
                                    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_72) : "r"(_bf16x2_mul_72));
                                    uint32_t _bf16x2_mul_73;
                                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_73) : "r"(block_25_1[i_75 * 2 + 1]), "r"(inverse_31_1));
                                    uint16_t _e4m3x2_73;
                                    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_73) : "r"(_bf16x2_mul_73));
                                    words_32_1[i_75] = (unsigned int)_e4m3x2_72 | (unsigned int)_e4m3x2_73 << 16;
                                }
                                scale_word_25 = scale_word_25 | scale_byte_29_1;
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v57_addr + (unsigned int)(tid * 32)), "r"(words_32_1[0]), "r"(words_32_1[1]), "r"(words_32_1[2]), "r"(words_32_1[3]) : "memory");
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v57_addr + (unsigned int)(tid * 32 + 16)), "r"(words_32_1[4]), "r"(words_32_1[5]), "r"(words_32_1[6]), "r"(words_32_1[7]) : "memory");
                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                if (tid == 0) {
                                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                    asm volatile(
                                        "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                        " [%0, {%1, %2}], [%3], %4;"
                                        :: "l"((&up_q_store)), "r"(y_6 * 256 + 128), "r"(x_6 * 256 + cta_rank_0 * 128), "r"(smem_v57_addr), "l"(0x12F0000000000000ULL) : "memory");
                                    asm volatile("cp.async.bulk.commit_group;");
                                }
                                unsigned int block_33_1[16];
                                #pragma unroll
                                for (int j_52 = 0; j_52 < 16; j_52++) {
                                    block_33_1[j_52] = packed_8[80 + j_52];
                                }
                                #pragma unroll
                                for (int j_53 = 0; j_53 < 4; j_53++) {
                                    unsigned int address_32 = d_smem_addr + 8192 + (unsigned int)(tid * 64) + (unsigned int)(j_53 * 16);
                                    address_32 = address_32 ^ (address_32 & 511) >> 7 << 4;
                                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v56_addr + (address_32 - smem_v56_addr)), "r"(packed_8[80 + 4 * j_53]), "r"(packed_8[80 + 4 * j_53 + 1]), "r"(packed_8[80 + 4 * j_53 + 2]), "r"(packed_8[80 + 4 * j_53 + 3]) : "memory");
                                }
                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                if (tid == 0) {
                                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                    asm volatile(
                                        "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                        " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                        :: "l"((&up_out_r)), "r"(0), "r"(x_6 * 256 + cta_rank_0 * 128), "r"(y_6 * 8 + 5), "r"(0), "r"(0), "r"(d_smem_addr + 8192), "l"(0x12F0000000000000ULL) : "memory");
                                    asm volatile("cp.async.bulk.commit_group;");
                                    asm volatile("cp.async.bulk.wait_group.read 1;");
                                }
                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                uint32_t _bf16x2_abs_74;
                                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_74) : "r"(block_33_1[0]));
                                unsigned int amax_pair_34_1 = _bf16x2_abs_74;
                                #pragma unroll
                                for (int i_76 = 1; i_76 < 16; i_76++) {
                                    uint32_t _bf16x2_abs_75;
                                    asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_75) : "r"(block_33_1[i_76]));
                                    uint32_t _bf16x2_max_37;
                                    asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_37) : "r"(amax_pair_34_1), "r"(_bf16x2_abs_75));
                                    amax_pair_34_1 = _bf16x2_max_37;
                                }
                                uint16_t _bf16_max_37;
                                asm("max.bf16 %0, %1, %2;" : "=h"(_bf16_max_37) : "h"((uint16_t)(amax_pair_34_1 & 65535)), "h"((uint16_t)(amax_pair_34_1 >> 16)));
                                float _cvt_f32_bf16_37;
                                asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_37) : "h"((uint16_t)(_bf16_max_37)));
                                float amax_35_1 = _cvt_f32_bf16_37;
                                float _fmax_37 = fmaxf(amax_35_1 * 0.002232142857f, 1e-12f);
                                float scale_36_1 = _fmax_37;
                                uint16_t _ue8m0x2_f32_37;
                                asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_ue8m0x2_f32_37) : "f"(scale_36_1), "f"(scale_36_1));
                                unsigned int scale_byte_37_1 = (unsigned int)_ue8m0x2_f32_37 & 255;
                                unsigned int inverse_lane_38_1 = 254 - scale_byte_37_1 << 7;
                                unsigned int inverse_39_1 = inverse_lane_38_1 | inverse_lane_38_1 << 16;
                                unsigned int words_40_1[8];
                                #pragma unroll
                                for (int i_77 = 0; i_77 < 8; i_77++) {
                                    uint32_t _bf16x2_mul_74;
                                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_74) : "r"(block_33_1[i_77 * 2]), "r"(inverse_39_1));
                                    uint16_t _e4m3x2_74;
                                    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_74) : "r"(_bf16x2_mul_74));
                                    uint32_t _bf16x2_mul_75;
                                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_75) : "r"(block_33_1[i_77 * 2 + 1]), "r"(inverse_39_1));
                                    uint16_t _e4m3x2_75;
                                    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_75) : "r"(_bf16x2_mul_75));
                                    words_40_1[i_77] = (unsigned int)_e4m3x2_74 | (unsigned int)_e4m3x2_75 << 16;
                                }
                                scale_word_25 = scale_word_25 | scale_byte_37_1 << 8;
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v57_addr + (unsigned int)(tid * 32)), "r"(words_40_1[0]), "r"(words_40_1[1]), "r"(words_40_1[2]), "r"(words_40_1[3]) : "memory");
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v57_addr + (unsigned int)(tid * 32 + 16)), "r"(words_40_1[4]), "r"(words_40_1[5]), "r"(words_40_1[6]), "r"(words_40_1[7]) : "memory");
                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                if (tid == 0) {
                                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                    asm volatile(
                                        "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                        " [%0, {%1, %2}], [%3], %4;"
                                        :: "l"((&up_q_store)), "r"(y_6 * 256 + 160), "r"(x_6 * 256 + cta_rank_0 * 128), "r"(smem_v57_addr), "l"(0x12F0000000000000ULL) : "memory");
                                    asm volatile("cp.async.bulk.commit_group;");
                                }
                                unsigned int block_41_1[16];
                                #pragma unroll
                                for (int j_54 = 0; j_54 < 16; j_54++) {
                                    block_41_1[j_54] = packed_8[96 + j_54];
                                }
                                #pragma unroll
                                for (int j_55 = 0; j_55 < 4; j_55++) {
                                    unsigned int address_33 = d_smem_addr + (unsigned int)(tid * 64) + (unsigned int)(j_55 * 16);
                                    address_33 = address_33 ^ (address_33 & 511) >> 7 << 4;
                                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v56_addr + (address_33 - smem_v56_addr)), "r"(packed_8[96 + 4 * j_55]), "r"(packed_8[96 + 4 * j_55 + 1]), "r"(packed_8[96 + 4 * j_55 + 2]), "r"(packed_8[96 + 4 * j_55 + 3]) : "memory");
                                }
                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                if (tid == 0) {
                                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                    asm volatile(
                                        "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                        " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                        :: "l"((&up_out_r)), "r"(0), "r"(x_6 * 256 + cta_rank_0 * 128), "r"(y_6 * 8 + 6), "r"(0), "r"(0), "r"(d_smem_addr), "l"(0x12F0000000000000ULL) : "memory");
                                    asm volatile("cp.async.bulk.commit_group;");
                                    asm volatile("cp.async.bulk.wait_group.read 1;");
                                }
                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                uint32_t _bf16x2_abs_76;
                                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_76) : "r"(block_41_1[0]));
                                unsigned int amax_pair_42_1 = _bf16x2_abs_76;
                                #pragma unroll
                                for (int i_78 = 1; i_78 < 16; i_78++) {
                                    uint32_t _bf16x2_abs_77;
                                    asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_77) : "r"(block_41_1[i_78]));
                                    uint32_t _bf16x2_max_38;
                                    asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_38) : "r"(amax_pair_42_1), "r"(_bf16x2_abs_77));
                                    amax_pair_42_1 = _bf16x2_max_38;
                                }
                                uint16_t _bf16_max_38;
                                asm("max.bf16 %0, %1, %2;" : "=h"(_bf16_max_38) : "h"((uint16_t)(amax_pair_42_1 & 65535)), "h"((uint16_t)(amax_pair_42_1 >> 16)));
                                float _cvt_f32_bf16_38;
                                asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_38) : "h"((uint16_t)(_bf16_max_38)));
                                float amax_43_1 = _cvt_f32_bf16_38;
                                float _fmax_38 = fmaxf(amax_43_1 * 0.002232142857f, 1e-12f);
                                float scale_44_1 = _fmax_38;
                                uint16_t _ue8m0x2_f32_38;
                                asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_ue8m0x2_f32_38) : "f"(scale_44_1), "f"(scale_44_1));
                                unsigned int scale_byte_45_1 = (unsigned int)_ue8m0x2_f32_38 & 255;
                                unsigned int inverse_lane_46_1 = 254 - scale_byte_45_1 << 7;
                                unsigned int inverse_47_1 = inverse_lane_46_1 | inverse_lane_46_1 << 16;
                                unsigned int words_48_1[8];
                                #pragma unroll
                                for (int i_79 = 0; i_79 < 8; i_79++) {
                                    uint32_t _bf16x2_mul_76;
                                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_76) : "r"(block_41_1[i_79 * 2]), "r"(inverse_47_1));
                                    uint16_t _e4m3x2_76;
                                    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_76) : "r"(_bf16x2_mul_76));
                                    uint32_t _bf16x2_mul_77;
                                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_77) : "r"(block_41_1[i_79 * 2 + 1]), "r"(inverse_47_1));
                                    uint16_t _e4m3x2_77;
                                    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_77) : "r"(_bf16x2_mul_77));
                                    words_48_1[i_79] = (unsigned int)_e4m3x2_76 | (unsigned int)_e4m3x2_77 << 16;
                                }
                                scale_word_25 = scale_word_25 | scale_byte_45_1 << 16;
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v57_addr + (unsigned int)(tid * 32)), "r"(words_48_1[0]), "r"(words_48_1[1]), "r"(words_48_1[2]), "r"(words_48_1[3]) : "memory");
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v57_addr + (unsigned int)(tid * 32 + 16)), "r"(words_48_1[4]), "r"(words_48_1[5]), "r"(words_48_1[6]), "r"(words_48_1[7]) : "memory");
                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                if (tid == 0) {
                                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                    asm volatile(
                                        "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                        " [%0, {%1, %2}], [%3], %4;"
                                        :: "l"((&up_q_store)), "r"(y_6 * 256 + 192), "r"(x_6 * 256 + cta_rank_0 * 128), "r"(smem_v57_addr), "l"(0x12F0000000000000ULL) : "memory");
                                    asm volatile("cp.async.bulk.commit_group;");
                                }
                                unsigned int block_49_1[16];
                                #pragma unroll
                                for (int j_56 = 0; j_56 < 16; j_56++) {
                                    block_49_1[j_56] = packed_8[112 + j_56];
                                }
                                #pragma unroll
                                for (int j_57 = 0; j_57 < 4; j_57++) {
                                    unsigned int address_34 = d_smem_addr + 8192 + (unsigned int)(tid * 64) + (unsigned int)(j_57 * 16);
                                    address_34 = address_34 ^ (address_34 & 511) >> 7 << 4;
                                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v56_addr + (address_34 - smem_v56_addr)), "r"(packed_8[112 + 4 * j_57]), "r"(packed_8[112 + 4 * j_57 + 1]), "r"(packed_8[112 + 4 * j_57 + 2]), "r"(packed_8[112 + 4 * j_57 + 3]) : "memory");
                                }
                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                if (tid == 0) {
                                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                    asm volatile(
                                        "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                        " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                        :: "l"((&up_out_r)), "r"(0), "r"(x_6 * 256 + cta_rank_0 * 128), "r"(y_6 * 8 + 7), "r"(0), "r"(0), "r"(d_smem_addr + 8192), "l"(0x12F0000000000000ULL) : "memory");
                                    asm volatile("cp.async.bulk.commit_group;");
                                    asm volatile("cp.async.bulk.wait_group.read 1;");
                                }
                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                uint32_t _bf16x2_abs_78;
                                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_78) : "r"(block_49_1[0]));
                                unsigned int amax_pair_50_1 = _bf16x2_abs_78;
                                #pragma unroll
                                for (int i_80 = 1; i_80 < 16; i_80++) {
                                    uint32_t _bf16x2_abs_79;
                                    asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_79) : "r"(block_49_1[i_80]));
                                    uint32_t _bf16x2_max_39;
                                    asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_39) : "r"(amax_pair_50_1), "r"(_bf16x2_abs_79));
                                    amax_pair_50_1 = _bf16x2_max_39;
                                }
                                uint16_t _bf16_max_39;
                                asm("max.bf16 %0, %1, %2;" : "=h"(_bf16_max_39) : "h"((uint16_t)(amax_pair_50_1 & 65535)), "h"((uint16_t)(amax_pair_50_1 >> 16)));
                                float _cvt_f32_bf16_39;
                                asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_39) : "h"((uint16_t)(_bf16_max_39)));
                                float amax_51_1 = _cvt_f32_bf16_39;
                                float _fmax_39 = fmaxf(amax_51_1 * 0.002232142857f, 1e-12f);
                                float scale_52_1 = _fmax_39;
                                uint16_t _ue8m0x2_f32_39;
                                asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_ue8m0x2_f32_39) : "f"(scale_52_1), "f"(scale_52_1));
                                unsigned int scale_byte_53_1 = (unsigned int)_ue8m0x2_f32_39 & 255;
                                unsigned int inverse_lane_54_1 = 254 - scale_byte_53_1 << 7;
                                unsigned int inverse_55_1 = inverse_lane_54_1 | inverse_lane_54_1 << 16;
                                unsigned int words_56_1[8];
                                #pragma unroll
                                for (int i_81 = 0; i_81 < 8; i_81++) {
                                    uint32_t _bf16x2_mul_78;
                                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_78) : "r"(block_49_1[i_81 * 2]), "r"(inverse_55_1));
                                    uint16_t _e4m3x2_78;
                                    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_78) : "r"(_bf16x2_mul_78));
                                    uint32_t _bf16x2_mul_79;
                                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_79) : "r"(block_49_1[i_81 * 2 + 1]), "r"(inverse_55_1));
                                    uint16_t _e4m3x2_79;
                                    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_79) : "r"(_bf16x2_mul_79));
                                    words_56_1[i_81] = (unsigned int)_e4m3x2_78 | (unsigned int)_e4m3x2_79 << 16;
                                }
                                scale_word_25 = scale_word_25 | scale_byte_53_1 << 24;
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v57_addr + (unsigned int)(tid * 32)), "r"(words_56_1[0]), "r"(words_56_1[1]), "r"(words_56_1[2]), "r"(words_56_1[3]) : "memory");
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v57_addr + (unsigned int)(tid * 32 + 16)), "r"(words_56_1[4]), "r"(words_56_1[5]), "r"(words_56_1[6]), "r"(words_56_1[7]) : "memory");
                                smem_v59[tid % 32 * 4 + tid / 32] = scale_word_25;
                                scale_word_25 = 0;
                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                if (tid == 0) {
                                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                    asm volatile(
                                        "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                        " [%0, {%1, %2}], [%3], %4;"
                                        :: "l"((&up_q_store)), "r"(y_6 * 256 + 224), "r"(x_6 * 256 + cta_rank_0 * 128), "r"(smem_v57_addr), "l"(0x12F0000000000000ULL) : "memory");
                                    tma_store_3d((&up_sc_store), 0, 0, (x_6 * 2 + cta_rank_0) * i_tiles + y_6 * 2 + 1, smem_v59_addr);
                                    asm volatile("cp.async.bulk.commit_group;");
                                }
                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                if (tid == 0) {
                                    asm volatile("cp.async.bulk.wait_group.read 0;");
                                }
                                asm volatile("barrier.sync 4, 128;" ::: "memory");
                                phase_bits_10 = phase_bits_10 ^ 64;
                                if (tid / 32 == 0) {
                                    if (warp == 0) {
                                        if (elect_sync()) {
                                            asm volatile("cp.async.bulk.wait_group 0;");
                                            bool enabled_value_11 = 1;
                                            if (enabled_value_11 != 0) {
                                                asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(replay_gu)) + ((macro_rows_6 + x_6) * (intermediate / 256) + y_6))), "r"(static_cast<unsigned int>(1)) : "memory");
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    gemm_phase = phase_bits_10;
                } else if (kind == 8) {
                    unsigned int phase_bits_11 = replay_phase;
                    int col_blocks_11 = intermediate / 128;
                    int num_tiles_1 = tokens / 128 * col_blocks_11;
                    int macro_row_offset_1 = macro_1 * (macro_size / 128);
                    int first_tile_2 = task_3 * 6 + cta_rank_0 * 3;
                    int global_mini_7 = macro_1 * (macro_size / mini_size) + mini_1;
                    int mini_tiles = mini_size / 128 * col_blocks_11;
                    first_tile_2 = first_tile_2 + global_mini_7 * mini_tiles;
                    int _min_48 = ((num_tiles_1) < ((global_mini_7 + 1) * mini_tiles) ? (num_tiles_1) : ((global_mini_7 + 1) * mini_tiles));
                    int tile_end_1 = _min_48;
                    int macro_tiles_3 = macro_size / 128;
                    if (first_tile_2 < tile_end_1) {
                        int first_row_3 = first_tile_2 / col_blocks_11;
                        int first_col_1 = first_tile_2 % col_blocks_11;
                        if (tid == 0) {
                            if (tile_end_1 > first_tile_2) {
                                int row_19 = first_row_3;
                                int col_62 = first_col_1;
                                if (col_62 >= col_blocks_11) {
                                    row_19 = row_19 + 1;
                                    col_62 = col_62 - col_blocks_11;
                                }
                                mbarrier_arrive_expect_tx(replay_arrived_addr, 65536);
                                int parent_2 = row_19 / 2 * (intermediate / 256) + col_62 / 2;
                                int32_t _relaxed_ld_30;
                                asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_30) : "l"(replay_gu + parent_2) : "memory");
                                int value_15 = _relaxed_ld_30;
                                while (value_15 < 4) {
                                    asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                    int32_t _relaxed_ld_31;
                                    asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_31) : "l"(replay_gu + parent_2) : "memory");
                                    value_15 = _relaxed_ld_31;
                                }
                                asm volatile("fence.acquire.gpu;" ::: "memory");
                                asm volatile(
                                    "cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
                                    " [%0], [%1, {%2, %3, %4, %5, %6}], [%7];"
                                    :: "r"(replay_gate_addr), "l"((&gate_sw_r)), "r"(0), "r"((row_19 - macro_row_offset_1) * 128), "r"(col_62 * 2), "r"(0), "r"(0), "r"(replay_arrived_addr) : "memory");
                                asm volatile(
                                    "cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
                                    " [%0], [%1, {%2, %3, %4, %5, %6}], [%7];"
                                    :: "r"(replay_up_addr), "l"((&up_sw_r)), "r"(0), "r"((row_19 - macro_row_offset_1) * 128), "r"(col_62 * 2), "r"(0), "r"(0), "r"(replay_arrived_addr) : "memory");
                            }
                            if (tile_end_1 > first_tile_2 + 1) {
                                int row_20 = first_row_3;
                                int col_63 = first_col_1 + 1;
                                if (col_63 >= col_blocks_11) {
                                    row_20 = row_20 + 1;
                                    col_63 = col_63 - col_blocks_11;
                                }
                                mbarrier_arrive_expect_tx(replay_arrived_addr + 8, 65536);
                                int parent_3 = row_20 / 2 * (intermediate / 256) + col_63 / 2;
                                int32_t _relaxed_ld_32;
                                asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_32) : "l"(replay_gu + parent_3) : "memory");
                                int value_16 = _relaxed_ld_32;
                                while (value_16 < 4) {
                                    asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                    int32_t _relaxed_ld_33;
                                    asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_33) : "l"(replay_gu + parent_3) : "memory");
                                    value_16 = _relaxed_ld_33;
                                }
                                asm volatile("fence.acquire.gpu;" ::: "memory");
                                asm volatile(
                                    "cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
                                    " [%0], [%1, {%2, %3, %4, %5, %6}], [%7];"
                                    :: "r"(replay_gate_addr + 32768), "l"((&gate_sw_r)), "r"(0), "r"((row_20 - macro_row_offset_1) * 128), "r"(col_63 * 2), "r"(0), "r"(0), "r"(replay_arrived_addr + 8) : "memory");
                                asm volatile(
                                    "cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
                                    " [%0], [%1, {%2, %3, %4, %5, %6}], [%7];"
                                    :: "r"(replay_up_addr + 32768), "l"((&up_sw_r)), "r"(0), "r"((row_20 - macro_row_offset_1) * 128), "r"(col_63 * 2), "r"(0), "r"(0), "r"(replay_arrived_addr + 8) : "memory");
                            }
                            if (tile_end_1 > first_tile_2 + 2) {
                                int row_21 = first_row_3;
                                int col_64 = first_col_1 + 2;
                                if (col_64 >= col_blocks_11) {
                                    row_21 = row_21 + 1;
                                    col_64 = col_64 - col_blocks_11;
                                }
                                mbarrier_arrive_expect_tx(replay_arrived_addr + 16, 65536);
                                int parent_4 = row_21 / 2 * (intermediate / 256) + col_64 / 2;
                                int32_t _relaxed_ld_34;
                                asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_34) : "l"(replay_gu + parent_4) : "memory");
                                int value_17 = _relaxed_ld_34;
                                while (value_17 < 4) {
                                    asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                    int32_t _relaxed_ld_35;
                                    asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_35) : "l"(replay_gu + parent_4) : "memory");
                                    value_17 = _relaxed_ld_35;
                                }
                                asm volatile("fence.acquire.gpu;" ::: "memory");
                                asm volatile(
                                    "cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
                                    " [%0], [%1, {%2, %3, %4, %5, %6}], [%7];"
                                    :: "r"(replay_gate_addr + 65536), "l"((&gate_sw_r)), "r"(0), "r"((row_21 - macro_row_offset_1) * 128), "r"(col_64 * 2), "r"(0), "r"(0), "r"(replay_arrived_addr + 16) : "memory");
                                asm volatile(
                                    "cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
                                    " [%0], [%1, {%2, %3, %4, %5, %6}], [%7];"
                                    :: "r"(replay_up_addr + 65536), "l"((&up_sw_r)), "r"(0), "r"((row_21 - macro_row_offset_1) * 128), "r"(col_64 * 2), "r"(0), "r"(0), "r"(replay_arrived_addr + 16) : "memory");
                            }
                        }
                        if (tile_end_1 > first_tile_2) {
                            mbarrier_wait(replay_arrived_addr, phase_bits_11 & 1);
                            phase_bits_11 = phase_bits_11 ^ 1;
                            int row_22 = first_row_3;
                            int col_65 = first_col_1;
                            if (col_65 >= col_blocks_11) {
                                row_22 = row_22 + 1;
                                col_65 = col_65 - col_blocks_11;
                            }
                            float gate_1[64];
                            float up_1[64];
                            float denominator[64];
                            int warp_0_5 = tid / 32;
                            int local_warp_1 = warp_0_5 / 4 + warp_0_5 % 4 * 2;
                            int lane_7 = tid % 32;
                            #pragma unroll
                            for (int tile_col_5 = 0; tile_col_5 < 8; tile_col_5++) {
                                unsigned int packed_9[4];
                                unsigned int address_35 = replay_gate_addr + (unsigned int)(((tile_col_5 * 16 + lane_7 / 16 * 8) / 64 * 128 * 64 + (local_warp_1 * 16 + lane_7 % 16) * 64 + (tile_col_5 * 16 + lane_7 / 16 * 8) % 64) * 2);
                                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                                    : "=r"(packed_9[0]), "=r"(packed_9[1]), "=r"(packed_9[2]), "=r"(packed_9[3])
                                    : "r"(address_35 ^ (address_35 & 1023) >> 7 << 4)
                                    : "memory");
                                #pragma unroll
                                for (int pair_9 = 0; pair_9 < 4; pair_9++) {
                                    float2 _cvt_f32_11 = __bfloat1622float2(__as_bf16x2(packed_9[pair_9]));
                                    gate_1[tile_col_5 * 8 + pair_9 * 2] = _cvt_f32_11.x;
                                    gate_1[tile_col_5 * 8 + pair_9 * 2 + 1] = _cvt_f32_11.y;
                                }
                            }
                            int warp_1_1 = tid / 32;
                            int local_warp_2_1 = warp_1_1 / 4 + warp_1_1 % 4 * 2;
                            int lane_3_2 = tid % 32;
                            #pragma unroll
                            for (int tile_col_6 = 0; tile_col_6 < 8; tile_col_6++) {
                                unsigned int packed_10[4];
                                unsigned int address_36 = replay_up_addr + (unsigned int)(((tile_col_6 * 16 + lane_3_2 / 16 * 8) / 64 * 128 * 64 + (local_warp_2_1 * 16 + lane_3_2 % 16) * 64 + (tile_col_6 * 16 + lane_3_2 / 16 * 8) % 64) * 2);
                                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                                    : "=r"(packed_10[0]), "=r"(packed_10[1]), "=r"(packed_10[2]), "=r"(packed_10[3])
                                    : "r"(address_36 ^ (address_36 & 1023) >> 7 << 4)
                                    : "memory");
                                #pragma unroll
                                for (int pair_10 = 0; pair_10 < 4; pair_10++) {
                                    float2 _cvt_f32_12 = __bfloat1622float2(__as_bf16x2(packed_10[pair_10]));
                                    up_1[tile_col_6 * 8 + pair_10 * 2] = _cvt_f32_12.x;
                                    up_1[tile_col_6 * 8 + pair_10 * 2 + 1] = _cvt_f32_12.y;
                                }
                            }
                            #pragma unroll
                            for (int elem_11 = 0; elem_11 < 64; elem_11++) {
                                denominator[elem_11] = gate_1[elem_11] * -1.0f;
                            }
                            #pragma unroll
                            for (int elem_12 = 0; elem_12 < 64; elem_12++) {
                                float _exp_1 = expf(denominator[elem_12]);
                                denominator[elem_12] = _exp_1;
                            }
                            #pragma unroll
                            for (int elem_13 = 0; elem_13 < 64; elem_13++) {
                                denominator[elem_13] = denominator[elem_13] + 1.0f;
                            }
                            #pragma unroll
                            for (int elem_14 = 0; elem_14 < 64; elem_14++) {
                                gate_1[elem_14] = gate_1[elem_14] / denominator[elem_14];
                            }
                            #pragma unroll
                            for (int elem_15 = 0; elem_15 < 64; elem_15++) {
                                gate_1[elem_15] = gate_1[elem_15] * up_1[elem_15];
                            }
                            __syncthreads();
                            int warp_4_1 = tid / 32;
                            int local_warp_5_1 = warp_4_1 / 4 + warp_4_1 % 4 * 2;
                            int lane_6_1 = tid % 32;
                            #pragma unroll
                            for (int tile_col_7 = 0; tile_col_7 < 8; tile_col_7++) {
                                unsigned int packed_11[4];
                                #pragma unroll
                                for (int pair_11 = 0; pair_11 < 4; pair_11++) {
                                    __nv_bfloat162 _bf16x2_28 = __float22bfloat162_rn(make_float2(gate_1[tile_col_7 * 8 + pair_11 * 2], gate_1[tile_col_7 * 8 + pair_11 * 2 + 1]));
                                    packed_11[pair_11] = __as_u32(_bf16x2_28);
                                }
                                int row_0_24 = local_warp_5_1 * 16 + lane_6_1 % 16;
                                int col_1_1 = tile_col_7 * 16 + lane_6_1 / 16 * 8;
                                uint32_t _stmatrix_addr_7 = static_cast<uint32_t>(hidden_staging_addr + (unsigned int)((row_0_24 * 136 + col_1_1) * 2));
                                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                    :: "r"(_stmatrix_addr_7), "r"(*reinterpret_cast<const uint32_t*>(&packed_11[0])), "r"(*reinterpret_cast<const uint32_t*>(&packed_11[1])), "r"(*reinterpret_cast<const uint32_t*>(&packed_11[2])), "r"(*reinterpret_cast<const uint32_t*>(&packed_11[3]))
                                    : "memory");
                            }
                            __syncthreads();
                            if (tid < 128) {
                                int row_0_25 = tid;
                                row_0_25 = tid % 64 * 2 + tid / 64;
                                unsigned int scale_word_26 = 0;
                                #pragma unroll 1
                                for (int j_58 = 0; j_58 < 4; j_58++) {
                                    int k_block_24 = (j_58 + tid / 8) % 4;
                                    unsigned int pairs_24[16];
                                    #pragma unroll
                                    for (int k_48 = 0; k_48 < 16; k_48++) {
                                        int col_0 = k_block_24 * 32 + (tid * 4 + k_48 * 2) % 32;
                                        float x0_24 = 0.0f;
                                        float x1_24 = 0.0f;
                                        x0_24 = (float)hidden_staging[col_0 * 136 + row_0_25];
                                        x1_24 = (float)hidden_staging[(col_0 + 1) * 136 + row_0_25];
                                        __nv_bfloat162 _bf16x2_29 = __float22bfloat162_rn(make_float2(x0_24, x1_24));
                                        pairs_24[k_48] = __as_u32(_bf16x2_29);
                                    }
                                    uint32_t _bf16x2_abs_80;
                                    asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_80) : "r"(pairs_24[0]));
                                    unsigned int amax_pair_27 = _bf16x2_abs_80;
                                    #pragma unroll
                                    for (int i_82 = 1; i_82 < 16; i_82++) {
                                        uint32_t _bf16x2_abs_81;
                                        asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_81) : "r"(pairs_24[i_82]));
                                        uint32_t _bf16x2_max_40;
                                        asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_40) : "r"(amax_pair_27), "r"(_bf16x2_abs_81));
                                        amax_pair_27 = _bf16x2_max_40;
                                    }
                                    uint16_t _bf16_max_40;
                                    asm("max.bf16 %0, %1, %2;" : "=h"(_bf16_max_40) : "h"((uint16_t)(amax_pair_27 & 65535)), "h"((uint16_t)(amax_pair_27 >> 16)));
                                    float _cvt_f32_bf16_40;
                                    asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_40) : "h"((uint16_t)(_bf16_max_40)));
                                    float amax_26 = _cvt_f32_bf16_40;
                                    float _fmax_40 = fmaxf(amax_26 * 0.002232142857f, 1e-12f);
                                    float scale_26 = _fmax_40;
                                    uint16_t _ue8m0x2_f32_40;
                                    asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_ue8m0x2_f32_40) : "f"(scale_26), "f"(scale_26));
                                    unsigned int scale_byte_26 = (unsigned int)_ue8m0x2_f32_40 & 255;
                                    unsigned int inverse_lane_26 = 254 - scale_byte_26 << 7;
                                    unsigned int inverse_26 = inverse_lane_26 | inverse_lane_26 << 16;
                                    unsigned int words_26[8];
                                    #pragma unroll
                                    for (int i_83 = 0; i_83 < 8; i_83++) {
                                        uint32_t _bf16x2_mul_80;
                                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_80) : "r"(pairs_24[i_83 * 2]), "r"(inverse_26));
                                        uint16_t _e4m3x2_80;
                                        asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_80) : "r"(_bf16x2_mul_80));
                                        uint32_t _bf16x2_mul_81;
                                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_81) : "r"(pairs_24[i_83 * 2 + 1]), "r"(inverse_26));
                                        uint16_t _e4m3x2_81;
                                        asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_81) : "r"(_bf16x2_mul_81));
                                        words_26[i_83] = (unsigned int)_e4m3x2_80 | (unsigned int)_e4m3x2_81 << 16;
                                    }
                                    scale_word_26 = scale_word_26 | scale_byte_26 << (unsigned int)(k_block_24 * 8);
                                    #pragma unroll
                                    for (int k_49 = 0; k_49 < 8; k_49++) {
                                        int col_0_1 = k_block_24 * 32 + (tid * 4 + k_49 * 4) % 32;
                                        smem_v21[(row_0_25 * 128 + col_0_1) / 4] = words_26[k_49];
                                    }
                                }
                                smem_v22[row_0_25 % 32 * 4 + row_0_25 / 32] = scale_word_26;
                            } else {
                                int row_0_26 = tid - 128;
                                unsigned int scale_word_27 = 0;
                                #pragma unroll 1
                                for (int j_59 = 0; j_59 < 4; j_59++) {
                                    int k_block_25 = (j_59 + (tid - 128) / 8) % 4;
                                    unsigned int pairs_25[16];
                                    #pragma unroll
                                    for (int k_50 = 0; k_50 < 16; k_50++) {
                                        int col_0_2 = k_block_25 * 32 + ((tid - 128) * 4 + k_50 * 2) % 32;
                                        float x0_25 = 0.0f;
                                        float x1_25 = 0.0f;
                                        pairs_25[k_50] = hidden_words[(row_0_26 * 136 + col_0_2) / 2];
                                    }
                                    uint32_t _bf16x2_abs_82;
                                    asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_82) : "r"(pairs_25[0]));
                                    unsigned int amax_pair_28 = _bf16x2_abs_82;
                                    #pragma unroll
                                    for (int i_84 = 1; i_84 < 16; i_84++) {
                                        uint32_t _bf16x2_abs_83;
                                        asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_83) : "r"(pairs_25[i_84]));
                                        uint32_t _bf16x2_max_41;
                                        asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_41) : "r"(amax_pair_28), "r"(_bf16x2_abs_83));
                                        amax_pair_28 = _bf16x2_max_41;
                                    }
                                    uint16_t _bf16_max_41;
                                    asm("max.bf16 %0, %1, %2;" : "=h"(_bf16_max_41) : "h"((uint16_t)(amax_pair_28 & 65535)), "h"((uint16_t)(amax_pair_28 >> 16)));
                                    float _cvt_f32_bf16_41;
                                    asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_41) : "h"((uint16_t)(_bf16_max_41)));
                                    float amax_28 = _cvt_f32_bf16_41;
                                    float _fmax_41 = fmaxf(amax_28 * 0.002232142857f, 1e-12f);
                                    float scale_27 = _fmax_41;
                                    uint16_t _ue8m0x2_f32_41;
                                    asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_ue8m0x2_f32_41) : "f"(scale_27), "f"(scale_27));
                                    unsigned int scale_byte_27 = (unsigned int)_ue8m0x2_f32_41 & 255;
                                    unsigned int inverse_lane_27 = 254 - scale_byte_27 << 7;
                                    unsigned int inverse_27 = inverse_lane_27 | inverse_lane_27 << 16;
                                    unsigned int words_27[8];
                                    #pragma unroll
                                    for (int i_85 = 0; i_85 < 8; i_85++) {
                                        uint32_t _bf16x2_mul_82;
                                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_82) : "r"(pairs_25[i_85 * 2]), "r"(inverse_27));
                                        uint16_t _e4m3x2_82;
                                        asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_82) : "r"(_bf16x2_mul_82));
                                        uint32_t _bf16x2_mul_83;
                                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_83) : "r"(pairs_25[i_85 * 2 + 1]), "r"(inverse_27));
                                        uint16_t _e4m3x2_83;
                                        asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_83) : "r"(_bf16x2_mul_83));
                                        words_27[i_85] = (unsigned int)_e4m3x2_82 | (unsigned int)_e4m3x2_83 << 16;
                                    }
                                    scale_word_27 = scale_word_27 | scale_byte_27 << (unsigned int)(k_block_25 * 8);
                                    #pragma unroll
                                    for (int k_51 = 0; k_51 < 8; k_51++) {
                                        int col_0_3 = k_block_25 * 32 + ((tid - 128) * 4 + k_51 * 4) % 32;
                                        smem_v19[(row_0_26 * 128 + col_0_3) / 4] = words_27[k_51];
                                    }
                                }
                                smem_v20[row_0_26 % 32 * 4 + row_0_26 / 32] = scale_word_27;
                            }
                            __syncthreads();
                            if (tid == 0) {
                                int local_row_3 = row_22 - macro_row_offset_1;
                                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                tma_store_2d((&h_q_store), col_65 * 128, local_row_3 * 128, smem_v19_addr);
                                tma_store_3d((&h_sc_store), 0, 0, local_row_3 * col_blocks_11 + col_65, smem_v20_addr);
                                tma_store_2d((&h_t_store), local_row_3 * 128, col_65 * 128, smem_v21_addr);
                                tma_store_3d((&h_sc_t_store), 0, 0, col_65 * macro_tiles_3 + local_row_3, smem_v22_addr);
                                asm volatile("cp.async.bulk.commit_group;");
                            }
                        }
                        if (tile_end_1 > first_tile_2 + 1) {
                            mbarrier_wait(replay_arrived_addr + 8, phase_bits_11 >> 1 & 1);
                            phase_bits_11 = phase_bits_11 ^ 2;
                            int row_23 = first_row_3;
                            int col_66 = first_col_1 + 1;
                            if (col_66 >= col_blocks_11) {
                                row_23 = row_23 + 1;
                                col_66 = col_66 - col_blocks_11;
                            }
                            float gate_2[64];
                            float up_2[64];
                            float denominator_1[64];
                            int warp_0_6 = tid / 32;
                            int local_warp_3 = warp_0_6 / 4 + warp_0_6 % 4 * 2;
                            int lane_8 = tid % 32;
                            #pragma unroll
                            for (int tile_col_8 = 0; tile_col_8 < 8; tile_col_8++) {
                                unsigned int packed_12[4];
                                unsigned int address_37 = replay_gate_addr + 32768 + (unsigned int)(((tile_col_8 * 16 + lane_8 / 16 * 8) / 64 * 128 * 64 + (local_warp_3 * 16 + lane_8 % 16) * 64 + (tile_col_8 * 16 + lane_8 / 16 * 8) % 64) * 2);
                                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                                    : "=r"(packed_12[0]), "=r"(packed_12[1]), "=r"(packed_12[2]), "=r"(packed_12[3])
                                    : "r"(address_37 ^ (address_37 & 1023) >> 7 << 4)
                                    : "memory");
                                #pragma unroll
                                for (int pair_12 = 0; pair_12 < 4; pair_12++) {
                                    float2 _cvt_f32_13 = __bfloat1622float2(__as_bf16x2(packed_12[pair_12]));
                                    gate_2[tile_col_8 * 8 + pair_12 * 2] = _cvt_f32_13.x;
                                    gate_2[tile_col_8 * 8 + pair_12 * 2 + 1] = _cvt_f32_13.y;
                                }
                            }
                            int warp_1_2 = tid / 32;
                            int local_warp_2_2 = warp_1_2 / 4 + warp_1_2 % 4 * 2;
                            int lane_3_3 = tid % 32;
                            #pragma unroll
                            for (int tile_col_9 = 0; tile_col_9 < 8; tile_col_9++) {
                                unsigned int packed_13[4];
                                unsigned int address_38 = replay_up_addr + 32768 + (unsigned int)(((tile_col_9 * 16 + lane_3_3 / 16 * 8) / 64 * 128 * 64 + (local_warp_2_2 * 16 + lane_3_3 % 16) * 64 + (tile_col_9 * 16 + lane_3_3 / 16 * 8) % 64) * 2);
                                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                                    : "=r"(packed_13[0]), "=r"(packed_13[1]), "=r"(packed_13[2]), "=r"(packed_13[3])
                                    : "r"(address_38 ^ (address_38 & 1023) >> 7 << 4)
                                    : "memory");
                                #pragma unroll
                                for (int pair_13 = 0; pair_13 < 4; pair_13++) {
                                    float2 _cvt_f32_14 = __bfloat1622float2(__as_bf16x2(packed_13[pair_13]));
                                    up_2[tile_col_9 * 8 + pair_13 * 2] = _cvt_f32_14.x;
                                    up_2[tile_col_9 * 8 + pair_13 * 2 + 1] = _cvt_f32_14.y;
                                }
                            }
                            #pragma unroll
                            for (int elem_16 = 0; elem_16 < 64; elem_16++) {
                                denominator_1[elem_16] = gate_2[elem_16] * -1.0f;
                            }
                            #pragma unroll
                            for (int elem_17 = 0; elem_17 < 64; elem_17++) {
                                float _exp_2 = expf(denominator_1[elem_17]);
                                denominator_1[elem_17] = _exp_2;
                            }
                            #pragma unroll
                            for (int elem_18 = 0; elem_18 < 64; elem_18++) {
                                denominator_1[elem_18] = denominator_1[elem_18] + 1.0f;
                            }
                            #pragma unroll
                            for (int elem_19 = 0; elem_19 < 64; elem_19++) {
                                gate_2[elem_19] = gate_2[elem_19] / denominator_1[elem_19];
                            }
                            #pragma unroll
                            for (int elem_20 = 0; elem_20 < 64; elem_20++) {
                                gate_2[elem_20] = gate_2[elem_20] * up_2[elem_20];
                            }
                            __syncthreads();
                            int warp_4_2 = tid / 32;
                            int local_warp_5_2 = warp_4_2 / 4 + warp_4_2 % 4 * 2;
                            int lane_6_2 = tid % 32;
                            #pragma unroll
                            for (int tile_col_10 = 0; tile_col_10 < 8; tile_col_10++) {
                                unsigned int packed_14[4];
                                #pragma unroll
                                for (int pair_14 = 0; pair_14 < 4; pair_14++) {
                                    __nv_bfloat162 _bf16x2_30 = __float22bfloat162_rn(make_float2(gate_2[tile_col_10 * 8 + pair_14 * 2], gate_2[tile_col_10 * 8 + pair_14 * 2 + 1]));
                                    packed_14[pair_14] = __as_u32(_bf16x2_30);
                                }
                                int row_0_27 = local_warp_5_2 * 16 + lane_6_2 % 16;
                                int col_1_2 = tile_col_10 * 16 + lane_6_2 / 16 * 8;
                                uint32_t _stmatrix_addr_8 = static_cast<uint32_t>(hidden_staging_addr + (unsigned int)((row_0_27 * 136 + col_1_2) * 2));
                                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                    :: "r"(_stmatrix_addr_8), "r"(*reinterpret_cast<const uint32_t*>(&packed_14[0])), "r"(*reinterpret_cast<const uint32_t*>(&packed_14[1])), "r"(*reinterpret_cast<const uint32_t*>(&packed_14[2])), "r"(*reinterpret_cast<const uint32_t*>(&packed_14[3]))
                                    : "memory");
                            }
                            __syncthreads();
                            if (tid < 128) {
                                int row_0_28 = tid;
                                row_0_28 = tid % 64 * 2 + tid / 64;
                                unsigned int scale_word_28 = 0;
                                #pragma unroll 1
                                for (int j_60 = 0; j_60 < 4; j_60++) {
                                    int k_block_26 = (j_60 + tid / 8) % 4;
                                    unsigned int pairs_26[16];
                                    #pragma unroll
                                    for (int k_52 = 0; k_52 < 16; k_52++) {
                                        int col_0_4 = k_block_26 * 32 + (tid * 4 + k_52 * 2) % 32;
                                        float x0_26 = 0.0f;
                                        float x1_26 = 0.0f;
                                        x0_26 = (float)hidden_staging[col_0_4 * 136 + row_0_28];
                                        x1_26 = (float)hidden_staging[(col_0_4 + 1) * 136 + row_0_28];
                                        __nv_bfloat162 _bf16x2_31 = __float22bfloat162_rn(make_float2(x0_26, x1_26));
                                        pairs_26[k_52] = __as_u32(_bf16x2_31);
                                    }
                                    uint32_t _bf16x2_abs_84;
                                    asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_84) : "r"(pairs_26[0]));
                                    unsigned int amax_pair_29 = _bf16x2_abs_84;
                                    #pragma unroll
                                    for (int i_86 = 1; i_86 < 16; i_86++) {
                                        uint32_t _bf16x2_abs_85;
                                        asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_85) : "r"(pairs_26[i_86]));
                                        uint32_t _bf16x2_max_42;
                                        asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_42) : "r"(amax_pair_29), "r"(_bf16x2_abs_85));
                                        amax_pair_29 = _bf16x2_max_42;
                                    }
                                    uint16_t _bf16_max_42;
                                    asm("max.bf16 %0, %1, %2;" : "=h"(_bf16_max_42) : "h"((uint16_t)(amax_pair_29 & 65535)), "h"((uint16_t)(amax_pair_29 >> 16)));
                                    float _cvt_f32_bf16_42;
                                    asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_42) : "h"((uint16_t)(_bf16_max_42)));
                                    float amax_29 = _cvt_f32_bf16_42;
                                    float _fmax_42 = fmaxf(amax_29 * 0.002232142857f, 1e-12f);
                                    float scale_29 = _fmax_42;
                                    uint16_t _ue8m0x2_f32_42;
                                    asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_ue8m0x2_f32_42) : "f"(scale_29), "f"(scale_29));
                                    unsigned int scale_byte_28 = (unsigned int)_ue8m0x2_f32_42 & 255;
                                    unsigned int inverse_lane_28 = 254 - scale_byte_28 << 7;
                                    unsigned int inverse_28 = inverse_lane_28 | inverse_lane_28 << 16;
                                    unsigned int words_28[8];
                                    #pragma unroll
                                    for (int i_87 = 0; i_87 < 8; i_87++) {
                                        uint32_t _bf16x2_mul_84;
                                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_84) : "r"(pairs_26[i_87 * 2]), "r"(inverse_28));
                                        uint16_t _e4m3x2_84;
                                        asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_84) : "r"(_bf16x2_mul_84));
                                        uint32_t _bf16x2_mul_85;
                                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_85) : "r"(pairs_26[i_87 * 2 + 1]), "r"(inverse_28));
                                        uint16_t _e4m3x2_85;
                                        asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_85) : "r"(_bf16x2_mul_85));
                                        words_28[i_87] = (unsigned int)_e4m3x2_84 | (unsigned int)_e4m3x2_85 << 16;
                                    }
                                    scale_word_28 = scale_word_28 | scale_byte_28 << (unsigned int)(k_block_26 * 8);
                                    #pragma unroll
                                    for (int k_53 = 0; k_53 < 8; k_53++) {
                                        int col_0_5 = k_block_26 * 32 + (tid * 4 + k_53 * 4) % 32;
                                        smem_v25[(row_0_28 * 128 + col_0_5) / 4] = words_28[k_53];
                                    }
                                }
                                smem_v26[row_0_28 % 32 * 4 + row_0_28 / 32] = scale_word_28;
                            } else {
                                int row_0_29 = tid - 128;
                                unsigned int scale_word_29 = 0;
                                #pragma unroll 1
                                for (int j_61 = 0; j_61 < 4; j_61++) {
                                    int k_block_27 = (j_61 + (tid - 128) / 8) % 4;
                                    unsigned int pairs_27[16];
                                    #pragma unroll
                                    for (int k_54 = 0; k_54 < 16; k_54++) {
                                        int col_0_6 = k_block_27 * 32 + ((tid - 128) * 4 + k_54 * 2) % 32;
                                        float x0_27 = 0.0f;
                                        float x1_27 = 0.0f;
                                        pairs_27[k_54] = hidden_words[(row_0_29 * 136 + col_0_6) / 2];
                                    }
                                    uint32_t _bf16x2_abs_86;
                                    asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_86) : "r"(pairs_27[0]));
                                    unsigned int amax_pair_30 = _bf16x2_abs_86;
                                    #pragma unroll
                                    for (int i_88 = 1; i_88 < 16; i_88++) {
                                        uint32_t _bf16x2_abs_87;
                                        asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_87) : "r"(pairs_27[i_88]));
                                        uint32_t _bf16x2_max_43;
                                        asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_43) : "r"(amax_pair_30), "r"(_bf16x2_abs_87));
                                        amax_pair_30 = _bf16x2_max_43;
                                    }
                                    uint16_t _bf16_max_43;
                                    asm("max.bf16 %0, %1, %2;" : "=h"(_bf16_max_43) : "h"((uint16_t)(amax_pair_30 & 65535)), "h"((uint16_t)(amax_pair_30 >> 16)));
                                    float _cvt_f32_bf16_43;
                                    asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_43) : "h"((uint16_t)(_bf16_max_43)));
                                    float amax_30 = _cvt_f32_bf16_43;
                                    float _fmax_43 = fmaxf(amax_30 * 0.002232142857f, 1e-12f);
                                    float scale_30 = _fmax_43;
                                    uint16_t _ue8m0x2_f32_43;
                                    asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_ue8m0x2_f32_43) : "f"(scale_30), "f"(scale_30));
                                    unsigned int scale_byte_30 = (unsigned int)_ue8m0x2_f32_43 & 255;
                                    unsigned int inverse_lane_29 = 254 - scale_byte_30 << 7;
                                    unsigned int inverse_29 = inverse_lane_29 | inverse_lane_29 << 16;
                                    unsigned int words_29[8];
                                    #pragma unroll
                                    for (int i_89 = 0; i_89 < 8; i_89++) {
                                        uint32_t _bf16x2_mul_86;
                                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_86) : "r"(pairs_27[i_89 * 2]), "r"(inverse_29));
                                        uint16_t _e4m3x2_86;
                                        asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_86) : "r"(_bf16x2_mul_86));
                                        uint32_t _bf16x2_mul_87;
                                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_87) : "r"(pairs_27[i_89 * 2 + 1]), "r"(inverse_29));
                                        uint16_t _e4m3x2_87;
                                        asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_87) : "r"(_bf16x2_mul_87));
                                        words_29[i_89] = (unsigned int)_e4m3x2_86 | (unsigned int)_e4m3x2_87 << 16;
                                    }
                                    scale_word_29 = scale_word_29 | scale_byte_30 << (unsigned int)(k_block_27 * 8);
                                    #pragma unroll
                                    for (int k_55 = 0; k_55 < 8; k_55++) {
                                        int col_0_7 = k_block_27 * 32 + ((tid - 128) * 4 + k_55 * 4) % 32;
                                        smem_v23[(row_0_29 * 128 + col_0_7) / 4] = words_29[k_55];
                                    }
                                }
                                smem_v24[row_0_29 % 32 * 4 + row_0_29 / 32] = scale_word_29;
                            }
                            __syncthreads();
                            if (tid == 0) {
                                int local_row_4 = row_23 - macro_row_offset_1;
                                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                tma_store_2d((&h_q_store), col_66 * 128, local_row_4 * 128, smem_v23_addr);
                                tma_store_3d((&h_sc_store), 0, 0, local_row_4 * col_blocks_11 + col_66, smem_v24_addr);
                                tma_store_2d((&h_t_store), local_row_4 * 128, col_66 * 128, smem_v25_addr);
                                tma_store_3d((&h_sc_t_store), 0, 0, col_66 * macro_tiles_3 + local_row_4, smem_v26_addr);
                                asm volatile("cp.async.bulk.commit_group;");
                            }
                        }
                        if (tile_end_1 > first_tile_2 + 2) {
                            mbarrier_wait(replay_arrived_addr + 16, phase_bits_11 >> 2 & 1);
                            phase_bits_11 = phase_bits_11 ^ 4;
                            int row_24 = first_row_3;
                            int col_67 = first_col_1 + 2;
                            if (col_67 >= col_blocks_11) {
                                row_24 = row_24 + 1;
                                col_67 = col_67 - col_blocks_11;
                            }
                            float gate_3[64];
                            float up_3[64];
                            float denominator_2[64];
                            int warp_0_7 = tid / 32;
                            int local_warp_4 = warp_0_7 / 4 + warp_0_7 % 4 * 2;
                            int lane_10 = tid % 32;
                            #pragma unroll
                            for (int tile_col_11 = 0; tile_col_11 < 8; tile_col_11++) {
                                unsigned int packed_15[4];
                                unsigned int address_39 = replay_gate_addr + 65536 + (unsigned int)(((tile_col_11 * 16 + lane_10 / 16 * 8) / 64 * 128 * 64 + (local_warp_4 * 16 + lane_10 % 16) * 64 + (tile_col_11 * 16 + lane_10 / 16 * 8) % 64) * 2);
                                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                                    : "=r"(packed_15[0]), "=r"(packed_15[1]), "=r"(packed_15[2]), "=r"(packed_15[3])
                                    : "r"(address_39 ^ (address_39 & 1023) >> 7 << 4)
                                    : "memory");
                                #pragma unroll
                                for (int pair_15 = 0; pair_15 < 4; pair_15++) {
                                    float2 _cvt_f32_15 = __bfloat1622float2(__as_bf16x2(packed_15[pair_15]));
                                    gate_3[tile_col_11 * 8 + pair_15 * 2] = _cvt_f32_15.x;
                                    gate_3[tile_col_11 * 8 + pair_15 * 2 + 1] = _cvt_f32_15.y;
                                }
                            }
                            int warp_1_3 = tid / 32;
                            int local_warp_2_3 = warp_1_3 / 4 + warp_1_3 % 4 * 2;
                            int lane_3_4 = tid % 32;
                            #pragma unroll
                            for (int tile_col_12 = 0; tile_col_12 < 8; tile_col_12++) {
                                unsigned int packed_16[4];
                                unsigned int address_40 = replay_up_addr + 65536 + (unsigned int)(((tile_col_12 * 16 + lane_3_4 / 16 * 8) / 64 * 128 * 64 + (local_warp_2_3 * 16 + lane_3_4 % 16) * 64 + (tile_col_12 * 16 + lane_3_4 / 16 * 8) % 64) * 2);
                                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                                    : "=r"(packed_16[0]), "=r"(packed_16[1]), "=r"(packed_16[2]), "=r"(packed_16[3])
                                    : "r"(address_40 ^ (address_40 & 1023) >> 7 << 4)
                                    : "memory");
                                #pragma unroll
                                for (int pair_16 = 0; pair_16 < 4; pair_16++) {
                                    float2 _cvt_f32_16 = __bfloat1622float2(__as_bf16x2(packed_16[pair_16]));
                                    up_3[tile_col_12 * 8 + pair_16 * 2] = _cvt_f32_16.x;
                                    up_3[tile_col_12 * 8 + pair_16 * 2 + 1] = _cvt_f32_16.y;
                                }
                            }
                            #pragma unroll
                            for (int elem_21 = 0; elem_21 < 64; elem_21++) {
                                denominator_2[elem_21] = gate_3[elem_21] * -1.0f;
                            }
                            #pragma unroll
                            for (int elem_22 = 0; elem_22 < 64; elem_22++) {
                                float _exp_3 = expf(denominator_2[elem_22]);
                                denominator_2[elem_22] = _exp_3;
                            }
                            #pragma unroll
                            for (int elem_23 = 0; elem_23 < 64; elem_23++) {
                                denominator_2[elem_23] = denominator_2[elem_23] + 1.0f;
                            }
                            #pragma unroll
                            for (int elem_24 = 0; elem_24 < 64; elem_24++) {
                                gate_3[elem_24] = gate_3[elem_24] / denominator_2[elem_24];
                            }
                            #pragma unroll
                            for (int elem_25 = 0; elem_25 < 64; elem_25++) {
                                gate_3[elem_25] = gate_3[elem_25] * up_3[elem_25];
                            }
                            __syncthreads();
                            int warp_4_3 = tid / 32;
                            int local_warp_5_3 = warp_4_3 / 4 + warp_4_3 % 4 * 2;
                            int lane_6_3 = tid % 32;
                            #pragma unroll
                            for (int tile_col_13 = 0; tile_col_13 < 8; tile_col_13++) {
                                unsigned int packed_17[4];
                                #pragma unroll
                                for (int pair_17 = 0; pair_17 < 4; pair_17++) {
                                    __nv_bfloat162 _bf16x2_32 = __float22bfloat162_rn(make_float2(gate_3[tile_col_13 * 8 + pair_17 * 2], gate_3[tile_col_13 * 8 + pair_17 * 2 + 1]));
                                    packed_17[pair_17] = __as_u32(_bf16x2_32);
                                }
                                int row_0_30 = local_warp_5_3 * 16 + lane_6_3 % 16;
                                int col_1_3 = tile_col_13 * 16 + lane_6_3 / 16 * 8;
                                uint32_t _stmatrix_addr_9 = static_cast<uint32_t>(hidden_staging_addr + (unsigned int)((row_0_30 * 136 + col_1_3) * 2));
                                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                    :: "r"(_stmatrix_addr_9), "r"(*reinterpret_cast<const uint32_t*>(&packed_17[0])), "r"(*reinterpret_cast<const uint32_t*>(&packed_17[1])), "r"(*reinterpret_cast<const uint32_t*>(&packed_17[2])), "r"(*reinterpret_cast<const uint32_t*>(&packed_17[3]))
                                    : "memory");
                            }
                            __syncthreads();
                            if (tid < 128) {
                                int row_0_31 = tid;
                                row_0_31 = tid % 64 * 2 + tid / 64;
                                unsigned int scale_word_30 = 0;
                                #pragma unroll 1
                                for (int j_62 = 0; j_62 < 4; j_62++) {
                                    int k_block_28 = (j_62 + tid / 8) % 4;
                                    unsigned int pairs_28[16];
                                    #pragma unroll
                                    for (int k_56 = 0; k_56 < 16; k_56++) {
                                        int col_0_8 = k_block_28 * 32 + (tid * 4 + k_56 * 2) % 32;
                                        float x0_28 = 0.0f;
                                        float x1_28 = 0.0f;
                                        x0_28 = (float)hidden_staging[col_0_8 * 136 + row_0_31];
                                        x1_28 = (float)hidden_staging[(col_0_8 + 1) * 136 + row_0_31];
                                        __nv_bfloat162 _bf16x2_33 = __float22bfloat162_rn(make_float2(x0_28, x1_28));
                                        pairs_28[k_56] = __as_u32(_bf16x2_33);
                                    }
                                    uint32_t _bf16x2_abs_88;
                                    asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_88) : "r"(pairs_28[0]));
                                    unsigned int amax_pair_31 = _bf16x2_abs_88;
                                    #pragma unroll
                                    for (int i_90 = 1; i_90 < 16; i_90++) {
                                        uint32_t _bf16x2_abs_89;
                                        asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_89) : "r"(pairs_28[i_90]));
                                        uint32_t _bf16x2_max_44;
                                        asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_44) : "r"(amax_pair_31), "r"(_bf16x2_abs_89));
                                        amax_pair_31 = _bf16x2_max_44;
                                    }
                                    uint16_t _bf16_max_44;
                                    asm("max.bf16 %0, %1, %2;" : "=h"(_bf16_max_44) : "h"((uint16_t)(amax_pair_31 & 65535)), "h"((uint16_t)(amax_pair_31 >> 16)));
                                    float _cvt_f32_bf16_44;
                                    asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_44) : "h"((uint16_t)(_bf16_max_44)));
                                    float amax_31 = _cvt_f32_bf16_44;
                                    float _fmax_44 = fmaxf(amax_31 * 0.002232142857f, 1e-12f);
                                    float scale_31 = _fmax_44;
                                    uint16_t _ue8m0x2_f32_44;
                                    asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_ue8m0x2_f32_44) : "f"(scale_31), "f"(scale_31));
                                    unsigned int scale_byte_31 = (unsigned int)_ue8m0x2_f32_44 & 255;
                                    unsigned int inverse_lane_31 = 254 - scale_byte_31 << 7;
                                    unsigned int inverse_30 = inverse_lane_31 | inverse_lane_31 << 16;
                                    unsigned int words_30[8];
                                    #pragma unroll
                                    for (int i_91 = 0; i_91 < 8; i_91++) {
                                        uint32_t _bf16x2_mul_88;
                                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_88) : "r"(pairs_28[i_91 * 2]), "r"(inverse_30));
                                        uint16_t _e4m3x2_88;
                                        asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_88) : "r"(_bf16x2_mul_88));
                                        uint32_t _bf16x2_mul_89;
                                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_89) : "r"(pairs_28[i_91 * 2 + 1]), "r"(inverse_30));
                                        uint16_t _e4m3x2_89;
                                        asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_89) : "r"(_bf16x2_mul_89));
                                        words_30[i_91] = (unsigned int)_e4m3x2_88 | (unsigned int)_e4m3x2_89 << 16;
                                    }
                                    scale_word_30 = scale_word_30 | scale_byte_31 << (unsigned int)(k_block_28 * 8);
                                    #pragma unroll
                                    for (int k_57 = 0; k_57 < 8; k_57++) {
                                        int col_0_9 = k_block_28 * 32 + (tid * 4 + k_57 * 4) % 32;
                                        smem_v29[(row_0_31 * 128 + col_0_9) / 4] = words_30[k_57];
                                    }
                                }
                                smem_v30[row_0_31 % 32 * 4 + row_0_31 / 32] = scale_word_30;
                            } else {
                                int row_0_32 = tid - 128;
                                unsigned int scale_word_31 = 0;
                                #pragma unroll 1
                                for (int j_63 = 0; j_63 < 4; j_63++) {
                                    int k_block_29 = (j_63 + (tid - 128) / 8) % 4;
                                    unsigned int pairs_29[16];
                                    #pragma unroll
                                    for (int k_58 = 0; k_58 < 16; k_58++) {
                                        int col_0_10 = k_block_29 * 32 + ((tid - 128) * 4 + k_58 * 2) % 32;
                                        float x0_29 = 0.0f;
                                        float x1_29 = 0.0f;
                                        pairs_29[k_58] = hidden_words[(row_0_32 * 136 + col_0_10) / 2];
                                    }
                                    uint32_t _bf16x2_abs_90;
                                    asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_90) : "r"(pairs_29[0]));
                                    unsigned int amax_pair_32 = _bf16x2_abs_90;
                                    #pragma unroll
                                    for (int i_92 = 1; i_92 < 16; i_92++) {
                                        uint32_t _bf16x2_abs_91;
                                        asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_91) : "r"(pairs_29[i_92]));
                                        uint32_t _bf16x2_max_45;
                                        asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_45) : "r"(amax_pair_32), "r"(_bf16x2_abs_91));
                                        amax_pair_32 = _bf16x2_max_45;
                                    }
                                    uint16_t _bf16_max_45;
                                    asm("max.bf16 %0, %1, %2;" : "=h"(_bf16_max_45) : "h"((uint16_t)(amax_pair_32 & 65535)), "h"((uint16_t)(amax_pair_32 >> 16)));
                                    float _cvt_f32_bf16_45;
                                    asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_45) : "h"((uint16_t)(_bf16_max_45)));
                                    float amax_32 = _cvt_f32_bf16_45;
                                    float _fmax_45 = fmaxf(amax_32 * 0.002232142857f, 1e-12f);
                                    float scale_32 = _fmax_45;
                                    uint16_t _ue8m0x2_f32_45;
                                    asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_ue8m0x2_f32_45) : "f"(scale_32), "f"(scale_32));
                                    unsigned int scale_byte_32 = (unsigned int)_ue8m0x2_f32_45 & 255;
                                    unsigned int inverse_lane_32 = 254 - scale_byte_32 << 7;
                                    unsigned int inverse_32 = inverse_lane_32 | inverse_lane_32 << 16;
                                    unsigned int words_31[8];
                                    #pragma unroll
                                    for (int i_93 = 0; i_93 < 8; i_93++) {
                                        uint32_t _bf16x2_mul_90;
                                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_90) : "r"(pairs_29[i_93 * 2]), "r"(inverse_32));
                                        uint16_t _e4m3x2_90;
                                        asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_90) : "r"(_bf16x2_mul_90));
                                        uint32_t _bf16x2_mul_91;
                                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_91) : "r"(pairs_29[i_93 * 2 + 1]), "r"(inverse_32));
                                        uint16_t _e4m3x2_91;
                                        asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_91) : "r"(_bf16x2_mul_91));
                                        words_31[i_93] = (unsigned int)_e4m3x2_90 | (unsigned int)_e4m3x2_91 << 16;
                                    }
                                    scale_word_31 = scale_word_31 | scale_byte_32 << (unsigned int)(k_block_29 * 8);
                                    #pragma unroll
                                    for (int k_59 = 0; k_59 < 8; k_59++) {
                                        int col_0_11 = k_block_29 * 32 + ((tid - 128) * 4 + k_59 * 4) % 32;
                                        smem_v27[(row_0_32 * 128 + col_0_11) / 4] = words_31[k_59];
                                    }
                                }
                                smem_v28[row_0_32 % 32 * 4 + row_0_32 / 32] = scale_word_31;
                            }
                            __syncthreads();
                            if (tid == 0) {
                                int local_row_5 = row_24 - macro_row_offset_1;
                                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                tma_store_2d((&h_q_store), col_67 * 128, local_row_5 * 128, smem_v27_addr);
                                tma_store_3d((&h_sc_store), 0, 0, local_row_5 * col_blocks_11 + col_67, smem_v28_addr);
                                tma_store_2d((&h_t_store), local_row_5 * 128, col_67 * 128, smem_v29_addr);
                                tma_store_3d((&h_sc_t_store), 0, 0, col_67 * macro_tiles_3 + local_row_5, smem_v30_addr);
                                asm volatile("cp.async.bulk.commit_group;");
                            }
                        }
                        if (tid == 0) {
                            asm volatile("cp.async.bulk.wait_group 0;");
                            if (tile_end_1 > first_tile_2) {
                                int row_25 = first_row_3;
                                if (col_blocks_11 <= first_col_1) {
                                    row_25 = row_25 + 1;
                                }
                                asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(replay_h)) + (row_25 / 2))), "r"(static_cast<unsigned int>(1)) : "memory");
                            }
                            if (tile_end_1 > first_tile_2 + 1) {
                                int row_26 = first_row_3;
                                if (col_blocks_11 <= first_col_1 + 1) {
                                    row_26 = row_26 + 1;
                                }
                                asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(replay_h)) + (row_26 / 2))), "r"(static_cast<unsigned int>(1)) : "memory");
                            }
                            if (tile_end_1 > first_tile_2 + 2) {
                                int row_27 = first_row_3;
                                if (col_blocks_11 <= first_col_1 + 2) {
                                    row_27 = row_27 + 1;
                                }
                                asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(replay_h)) + (row_27 / 2))), "r"(static_cast<unsigned int>(1)) : "memory");
                            }
                        }
                    }
                    replay_phase = phase_bits_11;
                } else {
                    if (kind == 0) {
                        int col_blocks_12 = (intermediate + 256 - 1) / 256;
                        int x_7 = -1;
                        int y_7 = -1;
                        int expert_7 = -1;
                        int k_start_7 = 0;
                        int k_end_7 = 0;
                        int first_7 = 0;
                        int first_block_2 = (macro_1 * (macro_size / mini_size) + mini_1) * (mini_size / 256);
                        int _min_49 = ((first_block_2 + mini_size / 256) < (tokens / 256) ? (first_block_2 + mini_size / 256) : (tokens / 256));
                        int end_block_2 = _min_49;
                        int block_3 = first_block_2 + task_3 / col_blocks_12;
                        if (block_3 < end_block_2) {
                            int index_4 = counts[3 * experts + block_3];
                            int offset_8 = counts[experts + index_4] / 256;
                            int _max_17 = ((first_block_2) > (offset_8) ? (first_block_2) : (offset_8));
                            int first_row_4 = _max_17;
                            int _min_50 = ((end_block_2) < (offset_8 + counts[index_4] / 256) ? (end_block_2) : (offset_8 + counts[index_4] / 256));
                            int rows_5 = _min_50 - first_row_4;
                            int supergroup_7 = (task_3 - (first_row_4 - first_block_2) * col_blocks_12) / (rows_5 * 8);
                            int full_cols_7 = col_blocks_12 / 8 * 8;
                            int row_28 = 0;
                            int col_68 = 0;
                            if (task_3 - (first_row_4 - first_block_2) * col_blocks_12 < rows_5 * full_cols_7) {
                                row_28 = (task_3 - (first_row_4 - first_block_2) * col_blocks_12) % (rows_5 * 8) / 8;
                                col_68 = supergroup_7 * 8 + (task_3 - (first_row_4 - first_block_2) * col_blocks_12) % 8;
                            } else {
                                row_28 = (task_3 - (first_row_4 - first_block_2) * col_blocks_12 - rows_5 * full_cols_7) / (col_blocks_12 - full_cols_7);
                                col_68 = full_cols_7 + (task_3 - (first_row_4 - first_block_2) * col_blocks_12 - rows_5 * full_cols_7) % (col_blocks_12 - full_cols_7);
                            }
                            if ((supergroup_7 & 1) != 0) {
                                row_28 = rows_5 - row_28 - 1;
                            }
                            x_7 = first_row_4 + row_28 - macro_1 * (macro_size / 256);
                            y_7 = col_68;
                            expert_7 = index_4;
                        }
                        unsigned int phase_bits_12 = gemm_phase;
                        int has_hi_7 = 0;
                        int global_mini_8 = macro_1 * (macro_size / mini_size) + mini_1;
                        int macro_rows_7 = macro_1 * (macro_size / 256);
                        int iterations_7 = hidden / 128;
                        int macro_k_2 = macro_1 * (macro_size / 128);
                        if (expert_7 < 0) {
                            if (tid == 0) {
                                bool enabled_value_12 = macros > 1;
                                if (enabled_value_12 != 0) {
                                    asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(buffers_done)) + (macro_1))), "r"(static_cast<unsigned int>(1)) : "memory");
                                }
                            }
                        } else if (tid / 32 == 7) {
                            if (warp == 7) {
                                if (elect_sync()) {
                                    {
                                        int _min_51 = ((mini_size) < (tokens - global_mini_8 * mini_size) ? (mini_size) : (tokens - global_mini_8 * mini_size));
                                        int _max_18 = ((0) > (_min_51) ? (0) : (_min_51));
                                        int mini_rows_10 = _max_18;
                                        int required_9 = (mini_rows_10 + 127) / 128 * ((hidden + 511) / 512);
                                        bool enabled_value_13 = 1;
                                        if (enabled_value_13 != 0) {
                                            int32_t _relaxed_ld_36;
                                            asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_36) : "l"(dy_ready + global_mini_8) : "memory");
                                            int value_18 = _relaxed_ld_36;
                                            while (value_18 < required_9) {
                                                asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                                int32_t _relaxed_ld_37;
                                                asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_37) : "l"(dy_ready + global_mini_8) : "memory");
                                                value_18 = _relaxed_ld_37;
                                            }
                                            asm volatile("fence.acquire.gpu;" ::: "memory");
                                        }
                                    }
                                    unsigned int previous_9 = phase_bits_12 >> 7 & 1;
                                    unsigned int bits_9 = phase_bits_12;
                                    if (previous_9 != 1) {
                                        mbarrier_wait(gemm_finished_addr, bits_9 >> 16 & 1);
                                        mbarrier_wait(gemm_finished_addr + 8, bits_9 >> 17 & 1);
                                        mbarrier_wait(gemm_finished_addr + 16, bits_9 >> 18 & 1);
                                        mbarrier_wait(gemm_finished_addr + 24, bits_9 >> 19 & 1);
                                        mbarrier_wait(gemm_finished_addr + 32, bits_9 >> 20 & 1);
                                        mbarrier_wait(gemm_finished_addr + 40, bits_9 >> 21 & 1);
                                        bits_9 = bits_9 ^ 128;
                                    }
                                    phase_bits_12 = bits_9;
                                    int ring_16 = 0;
                                    #pragma unroll 1
                                    for (int idx_16 = 0; idx_16 < iterations_7; idx_16++) {
                                        mbarrier_wait(gemm_finished_addr + (ring_16) * 8, phase_bits_12 >> (unsigned int)(16 + ring_16) & 1);
                                        asm volatile(
                                            "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                            " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                            :: "r"(smem_v52_addr + (unsigned int)(ring_16 * 16384)), "l"((&dy_q)), "r"(0), "r"(x_7 * 256 + cta_rank_0 * 128), "r"(idx_16), "r"(0), "r"(0),
                                               "r"(((gemm_arrived_addr + (ring_16) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                        asm volatile(
                                            "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                            " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                            :: "r"(smem_v53_addr + (unsigned int)(ring_16 * 16384)), "l"((&wd_t_q)), "r"(0), "r"(y_7 * 256 + cta_rank_0 * 128), "r"(idx_16), "r"(expert_7), "r"(0),
                                               "r"(((gemm_arrived_addr + (ring_16) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                        phase_bits_12 = phase_bits_12 ^ (unsigned int)(1 << 16 + ring_16);
                                        ring_16 = (ring_16 + 1) % 6;
                                    }
                                }
                            }
                        } else {
                            if (tid / 32 == 6) {
                                if (warp == 6) {
                                    if (elect_sync()) {
                                        {
                                            int _min_52 = ((mini_size) < (tokens - global_mini_8 * mini_size) ? (mini_size) : (tokens - global_mini_8 * mini_size));
                                            int _max_19 = ((0) > (_min_52) ? (0) : (_min_52));
                                            int mini_rows_11 = _max_19;
                                            int required_10 = (mini_rows_11 + 127) / 128 * ((hidden + 511) / 512);
                                            bool enabled_value_14 = 1;
                                            if (enabled_value_14 != 0) {
                                                int32_t _relaxed_ld_38;
                                                asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_38) : "l"(dy_ready + global_mini_8) : "memory");
                                                int value_19 = _relaxed_ld_38;
                                                while (value_19 < required_10) {
                                                    asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                                    int32_t _relaxed_ld_39;
                                                    asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_39) : "l"(dy_ready + global_mini_8) : "memory");
                                                    value_19 = _relaxed_ld_39;
                                                }
                                                asm volatile("fence.acquire.gpu;" ::: "memory");
                                            }
                                        }
                                        unsigned int previous_10 = phase_bits_12 >> 7 & 1;
                                        unsigned int bits_10 = phase_bits_12;
                                        if (previous_10 != 1) {
                                            mbarrier_wait(scales_finished_addr, bits_10 >> 16 & 1);
                                            mbarrier_wait(scales_finished_addr + 8, bits_10 >> 17 & 1);
                                            mbarrier_wait(scales_finished_addr + 16, bits_10 >> 18 & 1);
                                            mbarrier_wait(scales_finished_addr + 24, bits_10 >> 19 & 1);
                                            mbarrier_wait(scales_finished_addr + 32, bits_10 >> 20 & 1);
                                            mbarrier_wait(scales_finished_addr + 40, bits_10 >> 21 & 1);
                                            bits_10 = bits_10 ^ 128;
                                        }
                                        phase_bits_12 = bits_10;
                                        int ring_17 = 0;
                                        #pragma unroll 1
                                        for (int idx_17 = 0; idx_17 < iterations_7; idx_17++) {
                                            mbarrier_wait(scales_finished_addr + (ring_17) * 8, phase_bits_12 >> (unsigned int)(16 + ring_17) & 1);
                                            asm volatile(
                                                "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                                " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                                :: "r"(smem_v54_addr + (unsigned int)(ring_17 * 512)), "l"((&dy_sc)), "r"(0), "r"(0), "r"((x_7 * 2 + cta_rank_0) * (hidden / 128) + idx_17),
                                                   "r"(((scales_arrived_addr + (ring_17) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                            asm volatile(
                                                "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                                " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                                :: "r"(smem_v55_addr + (unsigned int)(ring_17 * 1024) + (unsigned int)(cta_rank_0 * 512)), "l"((&wd_t_sc)), "r"(0), "r"(0), "r"((expert_7 * i_tiles_5 + y_7 * 2 + cta_rank_0) * (hidden / 128) + idx_17),
                                                   "r"(((scales_arrived_addr + (ring_17) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(3)) : "memory");
                                            phase_bits_12 = phase_bits_12 ^ (unsigned int)(1 << 16 + ring_17);
                                            ring_17 = (ring_17 + 1) % 6;
                                        }
                                    }
                                }
                            } else if (tid / 32 == 4 && cta_rank_0 == 0) {
                                if (warp == 4) {
                                    if (elect_sync()) {
                                        int ring_18 = 0;
                                        mbarrier_wait(output_finished_addr, phase_bits_12 >> 22 & 1);
                                        phase_bits_12 = phase_bits_12 ^ 4194304;
                                        asm volatile("tcgen05.fence::after_thread_sync;");
                                        #pragma unroll 1
                                        for (int idx_18 = 0; idx_18 < iterations_7; idx_18++) {
                                            mbarrier_arrive_expect_tx(scales_arrived_addr + (ring_18) * 8, 3072);
                                            mbarrier_wait(scales_arrived_addr + (ring_18) * 8, phase_bits_12 >> (unsigned int)(8 + ring_18) & 1);
                                            int buffer_2 = idx_18 % 3;
                                            tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfa + buffer_2 * 4, make_sf_cp_desc_sbo128(smem_v54_addr + (unsigned int)(ring_18 * 512)));
                                            tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfb + buffer_2 * 8, make_sf_cp_desc_sbo128(smem_v55_addr + (unsigned int)(ring_18 * 1024)));
                                            tcgen05_cp_32x128b_warpx4_cta2((tmem_tmem_sfb + buffer_2 * 8 + 4), make_sf_cp_desc_sbo128((smem_v55_addr + (unsigned int)(ring_18 * 1024) + 512)));
                                            tcgen05_commit_cg2_multicast(scales_finished_addr + (ring_18) * 8, (uint16_t)(3));
                                            mbarrier_arrive_expect_tx(gemm_arrived_addr + (ring_18) * 8, 65536);
                                            mbarrier_wait(gemm_arrived_addr + (ring_18) * 8, phase_bits_12 >> (unsigned int)ring_18 & 1);
                                            int _mma_a_lo_12 = (((smem_v52_addr) >> 4) & 0x3FFF) + (ring_18) * 1024;
                                            int _mma_b_lo_12 = (((smem_v53_addr) >> 4) & 0x3FFF) + (ring_18) * 1024;
                                            {
                                                uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_12) | ((uint64_t)0x40004040 << 32);
                                                uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_12) | ((uint64_t)0x40004040 << 32);

                                                tcgen05_mma_mxf8_bs_cta2(tmem_accumulator, a_desc + 0, b_desc + 0,
                                                    0x10c00000U, tmem_tmem_sfa + buffer_2 * 4, tmem_tmem_sfb + buffer_2 * 8, ((idx_18 == 0) ? 0 : 1));
                                                tcgen05_mma_mxf8_bs_cta2(tmem_accumulator, a_desc + 2, b_desc + 2,
                                                    0x30c00010U, tmem_tmem_sfa + buffer_2 * 4, tmem_tmem_sfb + buffer_2 * 8, 1);
                                                tcgen05_mma_mxf8_bs_cta2(tmem_accumulator, a_desc + 4, b_desc + 4,
                                                    0x50c00020U, tmem_tmem_sfa + buffer_2 * 4, tmem_tmem_sfb + buffer_2 * 8, 1);
                                                tcgen05_mma_mxf8_bs_cta2(tmem_accumulator, a_desc + 6, b_desc + 6,
                                                    0x70c00030U, tmem_tmem_sfa + buffer_2 * 4, tmem_tmem_sfb + buffer_2 * 8, 1);
                                            }
                                            tcgen05_commit_cg2_multicast(gemm_finished_addr + (ring_18) * 8, (uint16_t)(3));
                                            phase_bits_12 = phase_bits_12 ^ (unsigned int)(1 << ring_18) ^ (unsigned int)(1 << 8 + ring_18);
                                            ring_18 = (ring_18 + 1) % 6;
                                        }
                                        tcgen05_commit_cg2_multicast(output_arrived_addr, (uint16_t)(3));
                                    }
                                }
                            } else {
                                if (tid < 128) {
                                    mbarrier_wait(output_arrived_addr, phase_bits_12 >> 6 & 1);
                                    unsigned int packed_18[128];
                                    #pragma unroll
                                    for (int chunk_14 = 0; chunk_14 < 8; chunk_14++) {
                                        #pragma unroll
                                        for (int sub_4 = 0; sub_4 < 2; sub_4++) {
                                            unsigned int address_41 = taddr_1 + (unsigned int)(tid / 32 * 32 + sub_4 * 16 << 16) + (unsigned int)(chunk_14 * 32);
                                            float _tmem_load_12[16];
                                            asm volatile(
                                                "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_12[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_12[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_12[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_12[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_12[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_12[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_12[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_12[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_12[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_12[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_12[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_12[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_12[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_12[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_12[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_12[15]))
                                                : "r"(address_41));
                                            #pragma unroll
                                            for (int pair_18 = 0; pair_18 < 8; pair_18++) {
                                                __nv_bfloat162 _bf16x2_34 = __float22bfloat162_rn(make_float2(_tmem_load_12[pair_18 * 2], _tmem_load_12[pair_18 * 2 + 1]));
                                                packed_18[chunk_14 * 16 + sub_4 * 8 + pair_18] = __as_u32(_bf16x2_34);
                                            }
                                        }
                                    }
                                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                                    int last_3 = 1;
                                    if (last_3 != 0) {
                                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                                        if (tid == 0) {
                                            asm volatile(
                                                "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                                                :: "r"((output_finished_addr) & 0xFEFFFFFF) : "memory");
                                        }
                                    }
                                    if (tid == 0) {
                                        int previous_offset_5 = (macro_1 + 1) * macro_size;
                                        int output_row_2 = x_7 * 256 + cta_rank_0 * 128;
                                        int _min_53 = ((macro_size) < (tokens - previous_offset_5) ? (macro_size) : (tokens - previous_offset_5));
                                        if (output_row_2 < _min_53) {
                                        }
                                    }
                                    #pragma unroll
                                    for (int chunk_15 = 0; chunk_15 < 8; chunk_15++) {
                                        if (tid == 0) {
                                            asm volatile("cp.async.bulk.wait_group.read 2;");
                                        }
                                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                                        int warp_0_8 = tid / 32;
                                        int lane_11 = tid % 32;
                                        #pragma unroll
                                        for (int half_4 = 0; half_4 < 2; half_4++) {
                                            #pragma unroll
                                            for (int col_tile_16 = 0; col_tile_16 < 2; col_tile_16++) {
                                                int row_29 = warp_0_8 * 32 + half_4 * 16 + lane_11 % 16;
                                                int col_69 = col_tile_16 * 16 + lane_11 / 16 * 8;
                                                unsigned int address_42 = d_smem_addr + (unsigned int)(chunk_15 % 3 * 8192) + (unsigned int)((row_29 * 32 + col_69) * 2);
                                                address_42 = address_42 ^ (address_42 & 511) >> 7 << 4;
                                                int offset_9 = chunk_15 * 16 + half_4 * 8 + col_tile_16 * 4;
                                                uint32_t _stmatrix_addr_10 = static_cast<uint32_t>(address_42);
                                                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                                    :: "r"(_stmatrix_addr_10), "r"(*reinterpret_cast<const uint32_t*>(&packed_18[offset_9])), "r"(*reinterpret_cast<const uint32_t*>(&packed_18[offset_9 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&packed_18[offset_9 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&packed_18[offset_9 + 3]))
                                                    : "memory");
                                            }
                                        }
                                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                                        if (tid == 0) {
                                            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                            asm volatile(
                                                "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                                " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                                :: "l"((&dh_r)), "r"(0), "r"(x_7 * 256 + cta_rank_0 * 128), "r"(y_7 * 8 + chunk_15), "r"(0), "r"(0), "r"(d_smem_addr + (unsigned int)(chunk_15 % 3 * 8192)), "l"(0x12F0000000000000ULL) : "memory");
                                            asm volatile("cp.async.bulk.commit_group;");
                                        }
                                    }
                                    if (has_hi_7 != 0) {
                                        unsigned int packed_0_2[128];
                                        #pragma unroll
                                        for (int chunk_16 = 0; chunk_16 < 8; chunk_16++) {
                                            #pragma unroll
                                            for (int sub_5 = 0; sub_5 < 2; sub_5++) {
                                                unsigned int address_43 = taddr_1 + (unsigned int)(tid / 32 * 32 + sub_5 * 16 << 16) + 256 + (unsigned int)(chunk_16 * 32);
                                                float _tmem_load_13[16];
                                                asm volatile(
                                                    "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                                                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                                    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[15]))
                                                    : "r"(address_43));
                                                #pragma unroll
                                                for (int pair_19 = 0; pair_19 < 8; pair_19++) {
                                                    __nv_bfloat162 _bf16x2_35 = __float22bfloat162_rn(make_float2(_tmem_load_13[pair_19 * 2], _tmem_load_13[pair_19 * 2 + 1]));
                                                    packed_0_2[chunk_16 * 16 + sub_5 * 8 + pair_19] = __as_u32(_bf16x2_35);
                                                }
                                            }
                                        }
                                        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                                        int last_1_2 = 1;
                                        if (last_1_2 != 0) {
                                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                                            if (tid == 0) {
                                                asm volatile(
                                                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                                                    :: "r"((output_finished_addr) & 0xFEFFFFFF) : "memory");
                                            }
                                        }
                                        #pragma unroll
                                        for (int chunk_17 = 0; chunk_17 < 8; chunk_17++) {
                                            if (tid == 0) {
                                                asm volatile("cp.async.bulk.wait_group.read 2;");
                                            }
                                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                                            int warp_0_9 = tid / 32;
                                            int lane_13 = tid % 32;
                                            #pragma unroll
                                            for (int half_5 = 0; half_5 < 2; half_5++) {
                                                #pragma unroll
                                                for (int col_tile_17 = 0; col_tile_17 < 2; col_tile_17++) {
                                                    int row_30 = warp_0_9 * 32 + half_5 * 16 + lane_13 % 16;
                                                    int col_70 = col_tile_17 * 16 + lane_13 / 16 * 8;
                                                    unsigned int address_44 = d_smem_addr + (unsigned int)((8 + chunk_17) % 3 * 8192) + (unsigned int)((row_30 * 32 + col_70) * 2);
                                                    address_44 = address_44 ^ (address_44 & 511) >> 7 << 4;
                                                    int offset_10 = chunk_17 * 16 + half_5 * 8 + col_tile_17 * 4;
                                                    uint32_t _stmatrix_addr_11 = static_cast<uint32_t>(address_44);
                                                    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                                        :: "r"(_stmatrix_addr_11), "r"(*reinterpret_cast<const uint32_t*>(&packed_0_2[offset_10])), "r"(*reinterpret_cast<const uint32_t*>(&packed_0_2[offset_10 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&packed_0_2[offset_10 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&packed_0_2[offset_10 + 3]))
                                                        : "memory");
                                                }
                                            }
                                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                                            if (tid == 0) {
                                                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                                asm volatile(
                                                    "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                                    " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                                    :: "l"((&dh_r)), "r"(0), "r"(x_7 * 256 + cta_rank_0 * 128), "r"((y_7 + 1) * 8 + chunk_17), "r"(0), "r"(0), "r"(d_smem_addr + (unsigned int)((8 + chunk_17) % 3 * 8192)), "l"(0x12F0000000000000ULL) : "memory");
                                                asm volatile("cp.async.bulk.commit_group;");
                                            }
                                        }
                                    }
                                    if (tid == 0) {
                                        asm volatile("cp.async.bulk.wait_group.read 0;");
                                    }
                                    asm volatile("barrier.sync 4, 128;" ::: "memory");
                                    phase_bits_12 = phase_bits_12 ^ 64;
                                    if (tid / 32 == 0) {
                                        if (warp == 0) {
                                            if (elect_sync()) {
                                                asm volatile("cp.async.bulk.wait_group 0;");
                                                bool enabled_value_15 = 1;
                                                if (enabled_value_15 != 0) {
                                                    asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(dh_ready)) + (shared_down_4 + (macro_rows_7 + x_7) * (intermediate / 256) + y_7))), "r"(static_cast<unsigned int>(1)) : "memory");
                                                }
                                                bool enabled_value_0_1 = macros > 1;
                                                if (enabled_value_0_1 != 0) {
                                                    asm volatile("cp.async.bulk.wait_group 0;");
                                                    bool enabled_value_1_1 = 1;
                                                    if (enabled_value_1_1 != 0) {
                                                        asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(buffers_done)) + (macro_1))), "r"(static_cast<unsigned int>(1)) : "memory");
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        gemm_phase = phase_bits_12;
                    } else if (kind == 1) {
                        unsigned int phase_bits_13 = swiglu_phase;
                        int col_blocks_13 = intermediate / 128;
                        int num_tiles_2 = tokens / 128 * col_blocks_13;
                        int macro_row_offset_2 = macro_1 * (macro_size / 128);
                        int macro_tiles_4 = macro_size / 128;
                        int first_tile_3 = task_3 * 4 + cta_rank_0 * 2;
                        int global_mini_9 = macro_1 * (macro_size / mini_size) + mini_1;
                        int mini_tiles_1 = mini_size / 128 * col_blocks_13;
                        first_tile_3 = first_tile_3 + global_mini_9 * mini_tiles_1;
                        int _min_54 = ((num_tiles_2) < ((global_mini_9 + 1) * mini_tiles_1) ? (num_tiles_2) : ((global_mini_9 + 1) * mini_tiles_1));
                        int tile_end_2 = _min_54;
                        if (first_tile_3 < tile_end_2) {
                            int first_row_5 = first_tile_3 / col_blocks_13;
                            int first_col_2 = first_tile_3 % col_blocks_13;
                            if (tid == 0) {
                                if (tile_end_2 > first_tile_3) {
                                    int row_31 = first_row_5;
                                    int col_71 = first_col_2;
                                    if (col_71 >= col_blocks_13) {
                                        row_31 = row_31 + 1;
                                        col_71 = col_71 - col_blocks_13;
                                    }
                                    mbarrier_arrive_expect_tx(swiglu_arrived_addr, 66560);
                                    int parent_5 = row_31 / 2 * (intermediate / 256) + col_71 / 2;
                                    int32_t _relaxed_ld_40;
                                    asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_40) : "l"(dh_ready + (shared_down_4 + parent_5)) : "memory");
                                    int value_20 = _relaxed_ld_40;
                                    while (value_20 < 2) {
                                        asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                        int32_t _relaxed_ld_41;
                                        asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_41) : "l"(dh_ready + (shared_down_4 + parent_5)) : "memory");
                                        value_20 = _relaxed_ld_41;
                                    }
                                    asm volatile("fence.acquire.gpu;" ::: "memory");
                                    bool enabled_value_16 = macro_1 > 0;
                                    if (enabled_value_16 != 0) {
                                        int32_t _relaxed_ld_42;
                                        asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_42) : "l"(replay_gu + parent_5) : "memory");
                                        int value_0 = _relaxed_ld_42;
                                        while (value_0 < 4) {
                                            asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                            int32_t _relaxed_ld_43;
                                            asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_43) : "l"(replay_gu + parent_5) : "memory");
                                            value_0 = _relaxed_ld_43;
                                        }
                                        asm volatile("fence.acquire.gpu;" ::: "memory");
                                    }
                                    int local_row_6 = row_31 - macro_row_offset_2;
                                    tma_2d_gmem2smem(smem_v60_addr, (&dh_tile), col_71 * 128, local_row_6 * 128, swiglu_arrived_addr);
                                    tma_2d_gmem2smem(smem_v64_addr, (&gate_tile), col_71 * 128, local_row_6 * 128, swiglu_arrived_addr);
                                    tma_2d_gmem2smem(smem_v66_addr, (&up_tile), col_71 * 128, local_row_6 * 128, swiglu_arrived_addr);
                                    tma_3d_gmem2smem(smem_v68_addr, (&gate_sc), 0, 0, local_row_6 * col_blocks_13 + col_71, swiglu_arrived_addr);
                                    tma_3d_gmem2smem(smem_v70_addr, (&up_sc), 0, 0, local_row_6 * col_blocks_13 + col_71, swiglu_arrived_addr);
                                }
                                if (tile_end_2 > first_tile_3 + 1) {
                                    int row_32 = first_row_5;
                                    int col_72 = first_col_2 + 1;
                                    if (col_72 >= col_blocks_13) {
                                        row_32 = row_32 + 1;
                                        col_72 = col_72 - col_blocks_13;
                                    }
                                    mbarrier_arrive_expect_tx(swiglu_arrived_addr + 8, 66560);
                                    int parent_6 = row_32 / 2 * (intermediate / 256) + col_72 / 2;
                                    int32_t _relaxed_ld_44;
                                    asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_44) : "l"(dh_ready + (shared_down_4 + parent_6)) : "memory");
                                    int value_21 = _relaxed_ld_44;
                                    while (value_21 < 2) {
                                        asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                        int32_t _relaxed_ld_45;
                                        asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_45) : "l"(dh_ready + (shared_down_4 + parent_6)) : "memory");
                                        value_21 = _relaxed_ld_45;
                                    }
                                    asm volatile("fence.acquire.gpu;" ::: "memory");
                                    bool enabled_value_17 = macro_1 > 0;
                                    if (enabled_value_17 != 0) {
                                        int32_t _relaxed_ld_46;
                                        asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_46) : "l"(replay_gu + parent_6) : "memory");
                                        int value_0_1 = _relaxed_ld_46;
                                        while (value_0_1 < 4) {
                                            asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                            int32_t _relaxed_ld_47;
                                            asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_47) : "l"(replay_gu + parent_6) : "memory");
                                            value_0_1 = _relaxed_ld_47;
                                        }
                                        asm volatile("fence.acquire.gpu;" ::: "memory");
                                    }
                                    int local_row_7 = row_32 - macro_row_offset_2;
                                    tma_2d_gmem2smem(smem_v61_addr, (&dh_tile), col_72 * 128, local_row_7 * 128, swiglu_arrived_addr + 8);
                                    tma_2d_gmem2smem(smem_v65_addr, (&gate_tile), col_72 * 128, local_row_7 * 128, swiglu_arrived_addr + 8);
                                    tma_2d_gmem2smem(smem_v67_addr, (&up_tile), col_72 * 128, local_row_7 * 128, swiglu_arrived_addr + 8);
                                    tma_3d_gmem2smem(smem_v69_addr, (&gate_sc), 0, 0, local_row_7 * col_blocks_13 + col_72, swiglu_arrived_addr + 8);
                                    tma_3d_gmem2smem(smem_v71_addr, (&up_sc), 0, 0, local_row_7 * col_blocks_13 + col_72, swiglu_arrived_addr + 8);
                                }
                            }
                            if (tile_end_2 > first_tile_3) {
                                mbarrier_wait(swiglu_arrived_addr, phase_bits_13 & 1);
                                phase_bits_13 = phase_bits_13 ^ 1;
                                int row_33 = first_row_5;
                                int col_73 = first_col_2;
                                if (col_73 >= col_blocks_13) {
                                    row_33 = row_33 + 1;
                                    col_73 = col_73 - col_blocks_13;
                                }
                                int tile_row = tid % 128;
                                int half_6 = tid / 128;
                                int local_token = (row_33 - macro_row_offset_2) * 128 + tile_row;
                                int peer_5 = schedule_rank[row_33 * 128 + tile_row];
                                float weight = weights[local_token];
                                float inverse_33 = ((weight > 0.0f) ? 1.0f / weight : 0.0f);
                                float router_gradient = 0.0f;
                                int scale_index = tile_row % 32 * 4 + tile_row / 32;
                                int k_pair = tile_row >> 1 & 1 ^ half_6;
                                unsigned int gate_scales = (unsigned int)smem_v72[scale_index * 2 + k_pair];
                                unsigned int up_scales = (unsigned int)smem_v74[scale_index * 2 + k_pair];
                                unsigned int dgate_bytes[2];
                                unsigned int dup_bytes[2];
                                #pragma unroll
                                for (int j_64 = 0; j_64 < 2; j_64++) {
                                    int k_block_30 = k_pair * 2 + (tile_row + j_64 & 1);
                                    int pair_byte = tile_row + j_64 & 1;
                                    float gate_scale = __uint_as_float((gate_scales >> (unsigned int)(pair_byte * 8) & 255) << 23);
                                    float up_scale = __uint_as_float((up_scales >> (unsigned int)(pair_byte * 8) & 255) << 23);
                                    float dgate_values[32];
                                    float dup_values[32];
                                    #pragma unroll
                                    for (int k_60 = 0; k_60 < 8; k_60++) {
                                        int col_idx = k_block_30 * 32 + (tile_row / 4 + k_60) % 8 * 4;
                                        unsigned int gate_word = smem_v64[(tile_row * 128 + col_idx) / 4];
                                        unsigned int up_word = smem_v66[(tile_row * 128 + col_idx) / 4];
                                        float2 _fp8x2_decode_0;
                                        asm("{ .reg .b32 pair; .reg .b16 lo, hi;\n"
                                            "cvt.rn.f16x2.e4m3x2 pair, %2;\n"
                                            "mov.b32 {lo, hi}, pair;\n"
                                            "cvt.f32.f16 %0, lo; cvt.f32.f16 %1, hi; }"
                                            : "=f"(_fp8x2_decode_0.x), "=f"(_fp8x2_decode_0.y) : "h"((uint16_t)(gate_word & 65535)));
                                        float2 _fp8x2_decode_1;
                                        asm("{ .reg .b32 pair; .reg .b16 lo, hi;\n"
                                            "cvt.rn.f16x2.e4m3x2 pair, %2;\n"
                                            "mov.b32 {lo, hi}, pair;\n"
                                            "cvt.f32.f16 %0, lo; cvt.f32.f16 %1, hi; }"
                                            : "=f"(_fp8x2_decode_1.x), "=f"(_fp8x2_decode_1.y) : "h"((uint16_t)(gate_word >> 16)));
                                        float2 _fp8x2_decode_2;
                                        asm("{ .reg .b32 pair; .reg .b16 lo, hi;\n"
                                            "cvt.rn.f16x2.e4m3x2 pair, %2;\n"
                                            "mov.b32 {lo, hi}, pair;\n"
                                            "cvt.f32.f16 %0, lo; cvt.f32.f16 %1, hi; }"
                                            : "=f"(_fp8x2_decode_2.x), "=f"(_fp8x2_decode_2.y) : "h"((uint16_t)(up_word & 65535)));
                                        float2 _fp8x2_decode_3;
                                        asm("{ .reg .b32 pair; .reg .b16 lo, hi;\n"
                                            "cvt.rn.f16x2.e4m3x2 pair, %2;\n"
                                            "mov.b32 {lo, hi}, pair;\n"
                                            "cvt.f32.f16 %0, lo; cvt.f32.f16 %1, hi; }"
                                            : "=f"(_fp8x2_decode_3.x), "=f"(_fp8x2_decode_3.y) : "h"((uint16_t)(up_word >> 16)));
                                        float2 _cvt_f32_17 = __bfloat1622float2(__as_bf16x2(smem_v62[(tile_row * 128 + col_idx) / 2]));
                                        float2 _cvt_f32_18 = __bfloat1622float2(__as_bf16x2(smem_v62[(tile_row * 128 + col_idx) / 2 + 1]));
                                        float dg = 0.0f;
                                        float du = 0.0f;
                                        float h = 0.0f;
                                        float _exp_4 = expf(-(_fp8x2_decode_0.x * gate_scale));
                                        float sigmoid = 1.0f / (1.0f + _exp_4);
                                        float silu = _fp8x2_decode_0.x * gate_scale * sigmoid;
                                        float dsilu = (1.0f - silu) * sigmoid + silu;
                                        dg = dsilu * (_fp8x2_decode_2.x * up_scale) * _cvt_f32_17.x;
                                        du = silu * _cvt_f32_17.x;
                                        h = silu * (_fp8x2_decode_2.x * up_scale);
                                        router_gradient = router_gradient + _cvt_f32_17.x * inverse_33 * h;
                                        dgate_values[k_60 * 4] = dg;
                                        dup_values[k_60 * 4] = du;
                                        float dg_0 = 0.0f;
                                        float du_1 = 0.0f;
                                        float h_2 = 0.0f;
                                        float _exp_5 = expf(-(_fp8x2_decode_0.y * gate_scale));
                                        float sigmoid_3 = 1.0f / (1.0f + _exp_5);
                                        float silu_4 = _fp8x2_decode_0.y * gate_scale * sigmoid_3;
                                        float dsilu_5 = (1.0f - silu_4) * sigmoid_3 + silu_4;
                                        dg_0 = dsilu_5 * (_fp8x2_decode_2.y * up_scale) * _cvt_f32_17.y;
                                        du_1 = silu_4 * _cvt_f32_17.y;
                                        h_2 = silu_4 * (_fp8x2_decode_2.y * up_scale);
                                        router_gradient = router_gradient + _cvt_f32_17.y * inverse_33 * h_2;
                                        dgate_values[k_60 * 4 + 1] = dg_0;
                                        dup_values[k_60 * 4 + 1] = du_1;
                                        float dg_6 = 0.0f;
                                        float du_7 = 0.0f;
                                        float h_8 = 0.0f;
                                        float _exp_6 = expf(-(_fp8x2_decode_1.x * gate_scale));
                                        float sigmoid_9 = 1.0f / (1.0f + _exp_6);
                                        float silu_10 = _fp8x2_decode_1.x * gate_scale * sigmoid_9;
                                        float dsilu_11 = (1.0f - silu_10) * sigmoid_9 + silu_10;
                                        dg_6 = dsilu_11 * (_fp8x2_decode_3.x * up_scale) * _cvt_f32_18.x;
                                        du_7 = silu_10 * _cvt_f32_18.x;
                                        h_8 = silu_10 * (_fp8x2_decode_3.x * up_scale);
                                        router_gradient = router_gradient + _cvt_f32_18.x * inverse_33 * h_8;
                                        dgate_values[k_60 * 4 + 2] = dg_6;
                                        dup_values[k_60 * 4 + 2] = du_7;
                                        float dg_12 = 0.0f;
                                        float du_13 = 0.0f;
                                        float h_14 = 0.0f;
                                        float _exp_7 = expf(-(_fp8x2_decode_1.y * gate_scale));
                                        float sigmoid_15 = 1.0f / (1.0f + _exp_7);
                                        float silu_16 = _fp8x2_decode_1.y * gate_scale * sigmoid_15;
                                        float dsilu_17 = (1.0f - silu_16) * sigmoid_15 + silu_16;
                                        dg_12 = dsilu_17 * (_fp8x2_decode_3.y * up_scale) * _cvt_f32_18.y;
                                        du_13 = silu_16 * _cvt_f32_18.y;
                                        h_14 = silu_16 * (_fp8x2_decode_3.y * up_scale);
                                        router_gradient = router_gradient + _cvt_f32_18.y * inverse_33 * h_14;
                                        dgate_values[k_60 * 4 + 3] = dg_12;
                                        dup_values[k_60 * 4 + 3] = du_13;
                                    }
                                    unsigned int dgate_bf16[16];
                                    unsigned int dup_bf16[16];
                                    #pragma unroll
                                    for (int p = 0; p < 16; p++) {
                                        __nv_bfloat162 _bf16x2_36 = __float22bfloat162_rn(make_float2(dgate_values[2 * p], dgate_values[2 * p + 1]));
                                        dgate_bf16[p] = __as_u32(_bf16x2_36);
                                        __nv_bfloat162 _bf16x2_37 = __float22bfloat162_rn(make_float2(dup_values[2 * p], dup_values[2 * p + 1]));
                                        dup_bf16[p] = __as_u32(_bf16x2_37);
                                    }
                                    uint32_t _bf16x2_abs_92;
                                    asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_92) : "r"(dgate_bf16[0]));
                                    unsigned int amax_pair_33 = _bf16x2_abs_92;
                                    #pragma unroll
                                    for (int i_94 = 1; i_94 < 16; i_94++) {
                                        uint32_t _bf16x2_abs_93;
                                        asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_93) : "r"(dgate_bf16[i_94]));
                                        uint32_t _bf16x2_max_46;
                                        asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_46) : "r"(amax_pair_33), "r"(_bf16x2_abs_93));
                                        amax_pair_33 = _bf16x2_max_46;
                                    }
                                    uint16_t _bf16_max_46;
                                    asm("max.bf16 %0, %1, %2;" : "=h"(_bf16_max_46) : "h"((uint16_t)(amax_pair_33 & 65535)), "h"((uint16_t)(amax_pair_33 >> 16)));
                                    float _cvt_f32_bf16_46;
                                    asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_46) : "h"((uint16_t)(_bf16_max_46)));
                                    float amax_33 = _cvt_f32_bf16_46;
                                    float _fmax_46 = fmaxf(amax_33 * 0.002232142857f, 1e-12f);
                                    float scale_33 = _fmax_46;
                                    uint16_t _ue8m0x2_f32_46;
                                    asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_ue8m0x2_f32_46) : "f"(scale_33), "f"(scale_33));
                                    unsigned int scale_byte_33 = (unsigned int)_ue8m0x2_f32_46 & 255;
                                    unsigned int inverse_lane_33 = 254 - scale_byte_33 << 7;
                                    unsigned int inverse_0 = inverse_lane_33 | inverse_lane_33 << 16;
                                    unsigned int words_33[8];
                                    #pragma unroll
                                    for (int i_95 = 0; i_95 < 8; i_95++) {
                                        uint32_t _bf16x2_mul_92;
                                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_92) : "r"(dgate_bf16[i_95 * 2]), "r"(inverse_0));
                                        uint16_t _e4m3x2_92;
                                        asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_92) : "r"(_bf16x2_mul_92));
                                        uint32_t _bf16x2_mul_93;
                                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_93) : "r"(dgate_bf16[i_95 * 2 + 1]), "r"(inverse_0));
                                        uint16_t _e4m3x2_93;
                                        asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_93) : "r"(_bf16x2_mul_93));
                                        words_33[i_95] = (unsigned int)_e4m3x2_92 | (unsigned int)_e4m3x2_93 << 16;
                                    }
                                    uint32_t _bf16x2_abs_94;
                                    asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_94) : "r"(dup_bf16[0]));
                                    unsigned int amax_pair_1_1 = _bf16x2_abs_94;
                                    #pragma unroll
                                    for (int i_96 = 1; i_96 < 16; i_96++) {
                                        uint32_t _bf16x2_abs_95;
                                        asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_95) : "r"(dup_bf16[i_96]));
                                        uint32_t _bf16x2_max_47;
                                        asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_47) : "r"(amax_pair_1_1), "r"(_bf16x2_abs_95));
                                        amax_pair_1_1 = _bf16x2_max_47;
                                    }
                                    uint16_t _bf16_max_47;
                                    asm("max.bf16 %0, %1, %2;" : "=h"(_bf16_max_47) : "h"((uint16_t)(amax_pair_1_1 & 65535)), "h"((uint16_t)(amax_pair_1_1 >> 16)));
                                    float _cvt_f32_bf16_47;
                                    asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_47) : "h"((uint16_t)(_bf16_max_47)));
                                    float amax_2_1 = _cvt_f32_bf16_47;
                                    float _fmax_47 = fmaxf(amax_2_1 * 0.002232142857f, 1e-12f);
                                    float scale_3_1 = _fmax_47;
                                    uint16_t _ue8m0x2_f32_47;
                                    asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_ue8m0x2_f32_47) : "f"(scale_3_1), "f"(scale_3_1));
                                    unsigned int scale_byte_4_1 = (unsigned int)_ue8m0x2_f32_47 & 255;
                                    unsigned int inverse_lane_5_1 = 254 - scale_byte_4_1 << 7;
                                    unsigned int inverse_6_1 = inverse_lane_5_1 | inverse_lane_5_1 << 16;
                                    unsigned int words_7_1[8];
                                    #pragma unroll
                                    for (int i_97 = 0; i_97 < 8; i_97++) {
                                        uint32_t _bf16x2_mul_94;
                                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_94) : "r"(dup_bf16[i_97 * 2]), "r"(inverse_6_1));
                                        uint16_t _e4m3x2_94;
                                        asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_94) : "r"(_bf16x2_mul_94));
                                        uint32_t _bf16x2_mul_95;
                                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_95) : "r"(dup_bf16[i_97 * 2 + 1]), "r"(inverse_6_1));
                                        uint16_t _e4m3x2_95;
                                        asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_95) : "r"(_bf16x2_mul_95));
                                        words_7_1[i_97] = (unsigned int)_e4m3x2_94 | (unsigned int)_e4m3x2_95 << 16;
                                    }
                                    dgate_bytes[j_64] = scale_byte_33;
                                    dup_bytes[j_64] = scale_byte_4_1;
                                    #pragma unroll
                                    for (int k_61 = 0; k_61 < 8; k_61++) {
                                        int col_idx_1 = k_block_30 * 32 + (tile_row / 4 + k_61) % 8 * 4;
                                        smem_v64[(tile_row * 128 + col_idx_1) / 4] = words_33[k_61];
                                        smem_v66[(tile_row * 128 + col_idx_1) / 4] = words_7_1[k_61];
                                        smem_v62[(tile_row * 128 + col_idx_1) / 2] = dgate_bf16[2 * k_61];
                                        smem_v62[(tile_row * 128 + col_idx_1) / 2 + 1] = dgate_bf16[2 * k_61 + 1];
                                        smem_v77[(tile_row * 128 + col_idx_1) / 2] = dup_bf16[2 * k_61];
                                        smem_v77[(tile_row * 128 + col_idx_1) / 2 + 1] = dup_bf16[2 * k_61 + 1];
                                    }
                                }
                                unsigned int pair_g = dgate_bytes[0] | dgate_bytes[1] << 8;
                                unsigned int pair_u = dup_bytes[0] | dup_bytes[1] << 8;
                                if ((tile_row & 1) != 0) {
                                    pair_g = dgate_bytes[1] | dgate_bytes[0] << 8;
                                    pair_u = dup_bytes[1] | dup_bytes[0] << 8;
                                }
                                int half_index = (tile_row % 32 * 16 + tile_row / 32 * 4 + k_pair * 2) / 2;
                                smem_v72[half_index] = (uint16_t)pair_g;
                                smem_v74[half_index] = (uint16_t)pair_u;
                                __syncthreads();
                                if (tid == 0) {
                                    int local_row_8 = row_33 - macro_row_offset_2;
                                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                    tma_store_2d((&dg_store), col_73 * 128, local_row_8 * 128, smem_v64_addr);
                                    tma_store_3d((&dg_sc_store), 0, 0, local_row_8 * col_blocks_13 + col_73, smem_v68_addr);
                                    tma_store_2d((&du_store), col_73 * 128, local_row_8 * 128, smem_v66_addr);
                                    tma_store_3d((&du_sc_store), 0, 0, local_row_8 * col_blocks_13 + col_73, smem_v70_addr);
                                    asm volatile("cp.async.bulk.commit_group;");
                                }
                                if (tid < 128) {
                                    if (tid < 128) {
                                        int row_0_33 = tid;
                                        row_0_33 = tid % 64 * 2 + tid / 64;
                                        unsigned int scale_word_32 = 0;
                                        #pragma unroll 1
                                        for (int j_65 = 0; j_65 < 4; j_65++) {
                                            int k_block_31 = (j_65 + tid / 8) % 4;
                                            unsigned int pairs_30[16];
                                            #pragma unroll
                                            for (int k_62 = 0; k_62 < 16; k_62++) {
                                                int col_0_12 = k_block_31 * 32 + (tid * 4 + k_62 * 2) % 32;
                                                float x0_30 = 0.0f;
                                                float x1_30 = 0.0f;
                                                x0_30 = (float)smem_v60[col_0_12 * 128 + row_0_33];
                                                x1_30 = (float)smem_v60[(col_0_12 + 1) * 128 + row_0_33];
                                                __nv_bfloat162 _bf16x2_38 = __float22bfloat162_rn(make_float2(x0_30, x1_30));
                                                pairs_30[k_62] = __as_u32(_bf16x2_38);
                                            }
                                            uint32_t _bf16x2_abs_96;
                                            asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_96) : "r"(pairs_30[0]));
                                            unsigned int amax_pair_35 = _bf16x2_abs_96;
                                            #pragma unroll
                                            for (int i_98 = 1; i_98 < 16; i_98++) {
                                                uint32_t _bf16x2_abs_97;
                                                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_97) : "r"(pairs_30[i_98]));
                                                uint32_t _bf16x2_max_48;
                                                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_48) : "r"(amax_pair_35), "r"(_bf16x2_abs_97));
                                                amax_pair_35 = _bf16x2_max_48;
                                            }
                                            uint16_t _bf16_max_48;
                                            asm("max.bf16 %0, %1, %2;" : "=h"(_bf16_max_48) : "h"((uint16_t)(amax_pair_35 & 65535)), "h"((uint16_t)(amax_pair_35 >> 16)));
                                            float _cvt_f32_bf16_48;
                                            asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_48) : "h"((uint16_t)(_bf16_max_48)));
                                            float amax_34 = _cvt_f32_bf16_48;
                                            float _fmax_48 = fmaxf(amax_34 * 0.002232142857f, 1e-12f);
                                            float scale_34 = _fmax_48;
                                            uint16_t _ue8m0x2_f32_48;
                                            asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_ue8m0x2_f32_48) : "f"(scale_34), "f"(scale_34));
                                            unsigned int scale_byte_34 = (unsigned int)_ue8m0x2_f32_48 & 255;
                                            unsigned int inverse_lane_34 = 254 - scale_byte_34 << 7;
                                            unsigned int inverse_0_1 = inverse_lane_34 | inverse_lane_34 << 16;
                                            unsigned int words_34[8];
                                            #pragma unroll
                                            for (int i_99 = 0; i_99 < 8; i_99++) {
                                                uint32_t _bf16x2_mul_96;
                                                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_96) : "r"(pairs_30[i_99 * 2]), "r"(inverse_0_1));
                                                uint16_t _e4m3x2_96;
                                                asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_96) : "r"(_bf16x2_mul_96));
                                                uint32_t _bf16x2_mul_97;
                                                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_97) : "r"(pairs_30[i_99 * 2 + 1]), "r"(inverse_0_1));
                                                uint16_t _e4m3x2_97;
                                                asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_97) : "r"(_bf16x2_mul_97));
                                                words_34[i_99] = (unsigned int)_e4m3x2_96 | (unsigned int)_e4m3x2_97 << 16;
                                            }
                                            scale_word_32 = scale_word_32 | scale_byte_34 << (unsigned int)(k_block_31 * 8);
                                            #pragma unroll
                                            for (int k_63 = 0; k_63 < 8; k_63++) {
                                                int col_0_13 = k_block_31 * 32 + (tid * 4 + k_63 * 4) % 32;
                                                smem_v78[(row_0_33 * 128 + col_0_13) / 4] = words_34[k_63];
                                            }
                                        }
                                        smem_v80[row_0_33 % 32 * 4 + row_0_33 / 32] = scale_word_32;
                                    }
                                } else if (tid - 128 < 128) {
                                    int row_0_34 = tid - 128;
                                    row_0_34 = (tid - 128) % 64 * 2 + (tid - 128) / 64;
                                    unsigned int scale_word_33 = 0;
                                    #pragma unroll 1
                                    for (int j_66 = 0; j_66 < 4; j_66++) {
                                        int k_block_32 = (j_66 + (tid - 128) / 8) % 4;
                                        unsigned int pairs_31[16];
                                        #pragma unroll
                                        for (int k_64 = 0; k_64 < 16; k_64++) {
                                            int col_0_14 = k_block_32 * 32 + ((tid - 128) * 4 + k_64 * 2) % 32;
                                            float x0_31 = 0.0f;
                                            float x1_31 = 0.0f;
                                            x0_31 = (float)smem_v76[col_0_14 * 128 + row_0_34];
                                            x1_31 = (float)smem_v76[(col_0_14 + 1) * 128 + row_0_34];
                                            __nv_bfloat162 _bf16x2_39 = __float22bfloat162_rn(make_float2(x0_31, x1_31));
                                            pairs_31[k_64] = __as_u32(_bf16x2_39);
                                        }
                                        uint32_t _bf16x2_abs_98;
                                        asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_98) : "r"(pairs_31[0]));
                                        unsigned int amax_pair_36 = _bf16x2_abs_98;
                                        #pragma unroll
                                        for (int i_100 = 1; i_100 < 16; i_100++) {
                                            uint32_t _bf16x2_abs_99;
                                            asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_99) : "r"(pairs_31[i_100]));
                                            uint32_t _bf16x2_max_49;
                                            asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_49) : "r"(amax_pair_36), "r"(_bf16x2_abs_99));
                                            amax_pair_36 = _bf16x2_max_49;
                                        }
                                        uint16_t _bf16_max_49;
                                        asm("max.bf16 %0, %1, %2;" : "=h"(_bf16_max_49) : "h"((uint16_t)(amax_pair_36 & 65535)), "h"((uint16_t)(amax_pair_36 >> 16)));
                                        float _cvt_f32_bf16_49;
                                        asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_49) : "h"((uint16_t)(_bf16_max_49)));
                                        float amax_36 = _cvt_f32_bf16_49;
                                        float _fmax_49 = fmaxf(amax_36 * 0.002232142857f, 1e-12f);
                                        float scale_35 = _fmax_49;
                                        uint16_t _ue8m0x2_f32_49;
                                        asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_ue8m0x2_f32_49) : "f"(scale_35), "f"(scale_35));
                                        unsigned int scale_byte_35 = (unsigned int)_ue8m0x2_f32_49 & 255;
                                        unsigned int inverse_lane_35 = 254 - scale_byte_35 << 7;
                                        unsigned int inverse_0_2 = inverse_lane_35 | inverse_lane_35 << 16;
                                        unsigned int words_35[8];
                                        #pragma unroll
                                        for (int i_101 = 0; i_101 < 8; i_101++) {
                                            uint32_t _bf16x2_mul_98;
                                            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_98) : "r"(pairs_31[i_101 * 2]), "r"(inverse_0_2));
                                            uint16_t _e4m3x2_98;
                                            asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_98) : "r"(_bf16x2_mul_98));
                                            uint32_t _bf16x2_mul_99;
                                            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_99) : "r"(pairs_31[i_101 * 2 + 1]), "r"(inverse_0_2));
                                            uint16_t _e4m3x2_99;
                                            asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_99) : "r"(_bf16x2_mul_99));
                                            words_35[i_101] = (unsigned int)_e4m3x2_98 | (unsigned int)_e4m3x2_99 << 16;
                                        }
                                        scale_word_33 = scale_word_33 | scale_byte_35 << (unsigned int)(k_block_32 * 8);
                                        #pragma unroll
                                        for (int k_65 = 0; k_65 < 8; k_65++) {
                                            int col_0_15 = k_block_32 * 32 + ((tid - 128) * 4 + k_65 * 4) % 32;
                                            smem_v79[(row_0_34 * 128 + col_0_15) / 4] = words_35[k_65];
                                        }
                                    }
                                    smem_v81[row_0_34 % 32 * 4 + row_0_34 / 32] = scale_word_33;
                                }
                                if (half_6 != 0) {
                                    smem_v82[tile_row] = router_gradient;
                                }
                                __syncthreads();
                                if (tid == 0) {
                                    int local_row_9 = row_33 - macro_row_offset_2;
                                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                    tma_store_2d((&dg_t_store), local_row_9 * 128, col_73 * 128, smem_v78_addr);
                                    tma_store_3d((&dg_sc_t_store), 0, 0, col_73 * macro_tiles_4 + local_row_9, smem_v80_addr);
                                    tma_store_2d((&du_t_store), local_row_9 * 128, col_73 * 128, smem_v79_addr);
                                    tma_store_3d((&du_sc_t_store), 0, 0, col_73 * macro_tiles_4 + local_row_9, smem_v81_addr);
                                    asm volatile("cp.async.bulk.commit_group;");
                                    asm volatile("cp.async.bulk.wait_group.read 0;");
                                }
                                if (half_6 == 0 && peer_5 >= 0) {
                                    router_gradient = router_gradient + smem_v82[tile_row];
                                    partials[local_token * col_blocks_13 + col_73] = router_gradient;
                                }
                                __syncthreads();
                            }
                            if (tile_end_2 > first_tile_3 + 1) {
                                mbarrier_wait(swiglu_arrived_addr + 8, phase_bits_13 >> 1 & 1);
                                phase_bits_13 = phase_bits_13 ^ 2;
                                int row_34 = first_row_5;
                                int col_74 = first_col_2 + 1;
                                if (col_74 >= col_blocks_13) {
                                    row_34 = row_34 + 1;
                                    col_74 = col_74 - col_blocks_13;
                                }
                                int tile_row_1 = tid % 128;
                                int half_7 = tid / 128;
                                int local_token_1 = (row_34 - macro_row_offset_2) * 128 + tile_row_1;
                                int peer_6 = schedule_rank[row_34 * 128 + tile_row_1];
                                float weight_1 = weights[local_token_1];
                                float inverse_34 = ((weight_1 > 0.0f) ? 1.0f / weight_1 : 0.0f);
                                float router_gradient_1 = 0.0f;
                                int scale_index_1 = tile_row_1 % 32 * 4 + tile_row_1 / 32;
                                int k_pair_1 = tile_row_1 >> 1 & 1 ^ half_7;
                                unsigned int gate_scales_1 = (unsigned int)smem_v73[scale_index_1 * 2 + k_pair_1];
                                unsigned int up_scales_1 = (unsigned int)smem_v75[scale_index_1 * 2 + k_pair_1];
                                unsigned int dgate_bytes_1[2];
                                unsigned int dup_bytes_1[2];
                                #pragma unroll
                                for (int j_67 = 0; j_67 < 2; j_67++) {
                                    int k_block_33 = k_pair_1 * 2 + (tile_row_1 + j_67 & 1);
                                    int pair_byte_1 = tile_row_1 + j_67 & 1;
                                    float gate_scale_1 = __uint_as_float((gate_scales_1 >> (unsigned int)(pair_byte_1 * 8) & 255) << 23);
                                    float up_scale_1 = __uint_as_float((up_scales_1 >> (unsigned int)(pair_byte_1 * 8) & 255) << 23);
                                    float dgate_values_1[32];
                                    float dup_values_1[32];
                                    #pragma unroll
                                    for (int k_66 = 0; k_66 < 8; k_66++) {
                                        int col_idx_2 = k_block_33 * 32 + (tile_row_1 / 4 + k_66) % 8 * 4;
                                        unsigned int gate_word_1 = smem_v65[(tile_row_1 * 128 + col_idx_2) / 4];
                                        unsigned int up_word_1 = smem_v67[(tile_row_1 * 128 + col_idx_2) / 4];
                                        float2 _fp8x2_decode_4;
                                        asm("{ .reg .b32 pair; .reg .b16 lo, hi;\n"
                                            "cvt.rn.f16x2.e4m3x2 pair, %2;\n"
                                            "mov.b32 {lo, hi}, pair;\n"
                                            "cvt.f32.f16 %0, lo; cvt.f32.f16 %1, hi; }"
                                            : "=f"(_fp8x2_decode_4.x), "=f"(_fp8x2_decode_4.y) : "h"((uint16_t)(gate_word_1 & 65535)));
                                        float2 _fp8x2_decode_5;
                                        asm("{ .reg .b32 pair; .reg .b16 lo, hi;\n"
                                            "cvt.rn.f16x2.e4m3x2 pair, %2;\n"
                                            "mov.b32 {lo, hi}, pair;\n"
                                            "cvt.f32.f16 %0, lo; cvt.f32.f16 %1, hi; }"
                                            : "=f"(_fp8x2_decode_5.x), "=f"(_fp8x2_decode_5.y) : "h"((uint16_t)(gate_word_1 >> 16)));
                                        float2 _fp8x2_decode_6;
                                        asm("{ .reg .b32 pair; .reg .b16 lo, hi;\n"
                                            "cvt.rn.f16x2.e4m3x2 pair, %2;\n"
                                            "mov.b32 {lo, hi}, pair;\n"
                                            "cvt.f32.f16 %0, lo; cvt.f32.f16 %1, hi; }"
                                            : "=f"(_fp8x2_decode_6.x), "=f"(_fp8x2_decode_6.y) : "h"((uint16_t)(up_word_1 & 65535)));
                                        float2 _fp8x2_decode_7;
                                        asm("{ .reg .b32 pair; .reg .b16 lo, hi;\n"
                                            "cvt.rn.f16x2.e4m3x2 pair, %2;\n"
                                            "mov.b32 {lo, hi}, pair;\n"
                                            "cvt.f32.f16 %0, lo; cvt.f32.f16 %1, hi; }"
                                            : "=f"(_fp8x2_decode_7.x), "=f"(_fp8x2_decode_7.y) : "h"((uint16_t)(up_word_1 >> 16)));
                                        float2 _cvt_f32_19 = __bfloat1622float2(__as_bf16x2(smem_v63[(tile_row_1 * 128 + col_idx_2) / 2]));
                                        float2 _cvt_f32_20 = __bfloat1622float2(__as_bf16x2(smem_v63[(tile_row_1 * 128 + col_idx_2) / 2 + 1]));
                                        float dg_1 = 0.0f;
                                        float du_2 = 0.0f;
                                        float h_1 = 0.0f;
                                        float _exp_8 = expf(-(_fp8x2_decode_4.x * gate_scale_1));
                                        float sigmoid_1 = 1.0f / (1.0f + _exp_8);
                                        float silu_1 = _fp8x2_decode_4.x * gate_scale_1 * sigmoid_1;
                                        float dsilu_1 = (1.0f - silu_1) * sigmoid_1 + silu_1;
                                        dg_1 = dsilu_1 * (_fp8x2_decode_6.x * up_scale_1) * _cvt_f32_19.x;
                                        du_2 = silu_1 * _cvt_f32_19.x;
                                        h_1 = silu_1 * (_fp8x2_decode_6.x * up_scale_1);
                                        router_gradient_1 = router_gradient_1 + _cvt_f32_19.x * inverse_34 * h_1;
                                        dgate_values_1[k_66 * 4] = dg_1;
                                        dup_values_1[k_66 * 4] = du_2;
                                        float dg_0_1 = 0.0f;
                                        float du_1_1 = 0.0f;
                                        float h_2_1 = 0.0f;
                                        float _exp_9 = expf(-(_fp8x2_decode_4.y * gate_scale_1));
                                        float sigmoid_3_1 = 1.0f / (1.0f + _exp_9);
                                        float silu_4_1 = _fp8x2_decode_4.y * gate_scale_1 * sigmoid_3_1;
                                        float dsilu_5_1 = (1.0f - silu_4_1) * sigmoid_3_1 + silu_4_1;
                                        dg_0_1 = dsilu_5_1 * (_fp8x2_decode_6.y * up_scale_1) * _cvt_f32_19.y;
                                        du_1_1 = silu_4_1 * _cvt_f32_19.y;
                                        h_2_1 = silu_4_1 * (_fp8x2_decode_6.y * up_scale_1);
                                        router_gradient_1 = router_gradient_1 + _cvt_f32_19.y * inverse_34 * h_2_1;
                                        dgate_values_1[k_66 * 4 + 1] = dg_0_1;
                                        dup_values_1[k_66 * 4 + 1] = du_1_1;
                                        float dg_6_1 = 0.0f;
                                        float du_7_1 = 0.0f;
                                        float h_8_1 = 0.0f;
                                        float _exp_10 = expf(-(_fp8x2_decode_5.x * gate_scale_1));
                                        float sigmoid_9_1 = 1.0f / (1.0f + _exp_10);
                                        float silu_10_1 = _fp8x2_decode_5.x * gate_scale_1 * sigmoid_9_1;
                                        float dsilu_11_1 = (1.0f - silu_10_1) * sigmoid_9_1 + silu_10_1;
                                        dg_6_1 = dsilu_11_1 * (_fp8x2_decode_7.x * up_scale_1) * _cvt_f32_20.x;
                                        du_7_1 = silu_10_1 * _cvt_f32_20.x;
                                        h_8_1 = silu_10_1 * (_fp8x2_decode_7.x * up_scale_1);
                                        router_gradient_1 = router_gradient_1 + _cvt_f32_20.x * inverse_34 * h_8_1;
                                        dgate_values_1[k_66 * 4 + 2] = dg_6_1;
                                        dup_values_1[k_66 * 4 + 2] = du_7_1;
                                        float dg_12_1 = 0.0f;
                                        float du_13_1 = 0.0f;
                                        float h_14_1 = 0.0f;
                                        float _exp_11 = expf(-(_fp8x2_decode_5.y * gate_scale_1));
                                        float sigmoid_15_1 = 1.0f / (1.0f + _exp_11);
                                        float silu_16_1 = _fp8x2_decode_5.y * gate_scale_1 * sigmoid_15_1;
                                        float dsilu_17_1 = (1.0f - silu_16_1) * sigmoid_15_1 + silu_16_1;
                                        dg_12_1 = dsilu_17_1 * (_fp8x2_decode_7.y * up_scale_1) * _cvt_f32_20.y;
                                        du_13_1 = silu_16_1 * _cvt_f32_20.y;
                                        h_14_1 = silu_16_1 * (_fp8x2_decode_7.y * up_scale_1);
                                        router_gradient_1 = router_gradient_1 + _cvt_f32_20.y * inverse_34 * h_14_1;
                                        dgate_values_1[k_66 * 4 + 3] = dg_12_1;
                                        dup_values_1[k_66 * 4 + 3] = du_13_1;
                                    }
                                    unsigned int dgate_bf16_1[16];
                                    unsigned int dup_bf16_1[16];
                                    #pragma unroll
                                    for (int p_1 = 0; p_1 < 16; p_1++) {
                                        __nv_bfloat162 _bf16x2_40 = __float22bfloat162_rn(make_float2(dgate_values_1[2 * p_1], dgate_values_1[2 * p_1 + 1]));
                                        dgate_bf16_1[p_1] = __as_u32(_bf16x2_40);
                                        __nv_bfloat162 _bf16x2_41 = __float22bfloat162_rn(make_float2(dup_values_1[2 * p_1], dup_values_1[2 * p_1 + 1]));
                                        dup_bf16_1[p_1] = __as_u32(_bf16x2_41);
                                    }
                                    uint32_t _bf16x2_abs_100;
                                    asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_100) : "r"(dgate_bf16_1[0]));
                                    unsigned int amax_pair_37 = _bf16x2_abs_100;
                                    #pragma unroll
                                    for (int i_102 = 1; i_102 < 16; i_102++) {
                                        uint32_t _bf16x2_abs_101;
                                        asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_101) : "r"(dgate_bf16_1[i_102]));
                                        uint32_t _bf16x2_max_50;
                                        asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_50) : "r"(amax_pair_37), "r"(_bf16x2_abs_101));
                                        amax_pair_37 = _bf16x2_max_50;
                                    }
                                    uint16_t _bf16_max_50;
                                    asm("max.bf16 %0, %1, %2;" : "=h"(_bf16_max_50) : "h"((uint16_t)(amax_pair_37 & 65535)), "h"((uint16_t)(amax_pair_37 >> 16)));
                                    float _cvt_f32_bf16_50;
                                    asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_50) : "h"((uint16_t)(_bf16_max_50)));
                                    float amax_37 = _cvt_f32_bf16_50;
                                    float _fmax_50 = fmaxf(amax_37 * 0.002232142857f, 1e-12f);
                                    float scale_37 = _fmax_50;
                                    uint16_t _ue8m0x2_f32_50;
                                    asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_ue8m0x2_f32_50) : "f"(scale_37), "f"(scale_37));
                                    unsigned int scale_byte_36 = (unsigned int)_ue8m0x2_f32_50 & 255;
                                    unsigned int inverse_lane_36 = 254 - scale_byte_36 << 7;
                                    unsigned int inverse_0_3 = inverse_lane_36 | inverse_lane_36 << 16;
                                    unsigned int words_36[8];
                                    #pragma unroll
                                    for (int i_103 = 0; i_103 < 8; i_103++) {
                                        uint32_t _bf16x2_mul_100;
                                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_100) : "r"(dgate_bf16_1[i_103 * 2]), "r"(inverse_0_3));
                                        uint16_t _e4m3x2_100;
                                        asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_100) : "r"(_bf16x2_mul_100));
                                        uint32_t _bf16x2_mul_101;
                                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_101) : "r"(dgate_bf16_1[i_103 * 2 + 1]), "r"(inverse_0_3));
                                        uint16_t _e4m3x2_101;
                                        asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_101) : "r"(_bf16x2_mul_101));
                                        words_36[i_103] = (unsigned int)_e4m3x2_100 | (unsigned int)_e4m3x2_101 << 16;
                                    }
                                    uint32_t _bf16x2_abs_102;
                                    asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_102) : "r"(dup_bf16_1[0]));
                                    unsigned int amax_pair_1_2 = _bf16x2_abs_102;
                                    #pragma unroll
                                    for (int i_104 = 1; i_104 < 16; i_104++) {
                                        uint32_t _bf16x2_abs_103;
                                        asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_103) : "r"(dup_bf16_1[i_104]));
                                        uint32_t _bf16x2_max_51;
                                        asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_51) : "r"(amax_pair_1_2), "r"(_bf16x2_abs_103));
                                        amax_pair_1_2 = _bf16x2_max_51;
                                    }
                                    uint16_t _bf16_max_51;
                                    asm("max.bf16 %0, %1, %2;" : "=h"(_bf16_max_51) : "h"((uint16_t)(amax_pair_1_2 & 65535)), "h"((uint16_t)(amax_pair_1_2 >> 16)));
                                    float _cvt_f32_bf16_51;
                                    asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_51) : "h"((uint16_t)(_bf16_max_51)));
                                    float amax_2_2 = _cvt_f32_bf16_51;
                                    float _fmax_51 = fmaxf(amax_2_2 * 0.002232142857f, 1e-12f);
                                    float scale_3_2 = _fmax_51;
                                    uint16_t _ue8m0x2_f32_51;
                                    asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_ue8m0x2_f32_51) : "f"(scale_3_2), "f"(scale_3_2));
                                    unsigned int scale_byte_4_2 = (unsigned int)_ue8m0x2_f32_51 & 255;
                                    unsigned int inverse_lane_5_2 = 254 - scale_byte_4_2 << 7;
                                    unsigned int inverse_6_2 = inverse_lane_5_2 | inverse_lane_5_2 << 16;
                                    unsigned int words_7_2[8];
                                    #pragma unroll
                                    for (int i_105 = 0; i_105 < 8; i_105++) {
                                        uint32_t _bf16x2_mul_102;
                                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_102) : "r"(dup_bf16_1[i_105 * 2]), "r"(inverse_6_2));
                                        uint16_t _e4m3x2_102;
                                        asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_102) : "r"(_bf16x2_mul_102));
                                        uint32_t _bf16x2_mul_103;
                                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_103) : "r"(dup_bf16_1[i_105 * 2 + 1]), "r"(inverse_6_2));
                                        uint16_t _e4m3x2_103;
                                        asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_103) : "r"(_bf16x2_mul_103));
                                        words_7_2[i_105] = (unsigned int)_e4m3x2_102 | (unsigned int)_e4m3x2_103 << 16;
                                    }
                                    dgate_bytes_1[j_67] = scale_byte_36;
                                    dup_bytes_1[j_67] = scale_byte_4_2;
                                    #pragma unroll
                                    for (int k_67 = 0; k_67 < 8; k_67++) {
                                        int col_idx_3 = k_block_33 * 32 + (tile_row_1 / 4 + k_67) % 8 * 4;
                                        smem_v65[(tile_row_1 * 128 + col_idx_3) / 4] = words_36[k_67];
                                        smem_v67[(tile_row_1 * 128 + col_idx_3) / 4] = words_7_2[k_67];
                                        smem_v63[(tile_row_1 * 128 + col_idx_3) / 2] = dgate_bf16_1[2 * k_67];
                                        smem_v63[(tile_row_1 * 128 + col_idx_3) / 2 + 1] = dgate_bf16_1[2 * k_67 + 1];
                                        smem_v77[(tile_row_1 * 128 + col_idx_3) / 2] = dup_bf16_1[2 * k_67];
                                        smem_v77[(tile_row_1 * 128 + col_idx_3) / 2 + 1] = dup_bf16_1[2 * k_67 + 1];
                                    }
                                }
                                unsigned int pair_g_1 = dgate_bytes_1[0] | dgate_bytes_1[1] << 8;
                                unsigned int pair_u_1 = dup_bytes_1[0] | dup_bytes_1[1] << 8;
                                if ((tile_row_1 & 1) != 0) {
                                    pair_g_1 = dgate_bytes_1[1] | dgate_bytes_1[0] << 8;
                                    pair_u_1 = dup_bytes_1[1] | dup_bytes_1[0] << 8;
                                }
                                int half_index_1 = (tile_row_1 % 32 * 16 + tile_row_1 / 32 * 4 + k_pair_1 * 2) / 2;
                                smem_v73[half_index_1] = (uint16_t)pair_g_1;
                                smem_v75[half_index_1] = (uint16_t)pair_u_1;
                                __syncthreads();
                                if (tid == 0) {
                                    int local_row_10 = row_34 - macro_row_offset_2;
                                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                    tma_store_2d((&dg_store), col_74 * 128, local_row_10 * 128, smem_v65_addr);
                                    tma_store_3d((&dg_sc_store), 0, 0, local_row_10 * col_blocks_13 + col_74, smem_v69_addr);
                                    tma_store_2d((&du_store), col_74 * 128, local_row_10 * 128, smem_v67_addr);
                                    tma_store_3d((&du_sc_store), 0, 0, local_row_10 * col_blocks_13 + col_74, smem_v71_addr);
                                    asm volatile("cp.async.bulk.commit_group;");
                                }
                                if (tid < 128) {
                                    if (tid < 128) {
                                        int row_0_35 = tid;
                                        row_0_35 = tid % 64 * 2 + tid / 64;
                                        unsigned int scale_word_34 = 0;
                                        #pragma unroll 1
                                        for (int j_68 = 0; j_68 < 4; j_68++) {
                                            int k_block_34 = (j_68 + tid / 8) % 4;
                                            unsigned int pairs_32[16];
                                            #pragma unroll
                                            for (int k_68 = 0; k_68 < 16; k_68++) {
                                                int col_0_16 = k_block_34 * 32 + (tid * 4 + k_68 * 2) % 32;
                                                float x0_32 = 0.0f;
                                                float x1_32 = 0.0f;
                                                x0_32 = (float)smem_v61[col_0_16 * 128 + row_0_35];
                                                x1_32 = (float)smem_v61[(col_0_16 + 1) * 128 + row_0_35];
                                                __nv_bfloat162 _bf16x2_42 = __float22bfloat162_rn(make_float2(x0_32, x1_32));
                                                pairs_32[k_68] = __as_u32(_bf16x2_42);
                                            }
                                            uint32_t _bf16x2_abs_104;
                                            asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_104) : "r"(pairs_32[0]));
                                            unsigned int amax_pair_38 = _bf16x2_abs_104;
                                            #pragma unroll
                                            for (int i_106 = 1; i_106 < 16; i_106++) {
                                                uint32_t _bf16x2_abs_105;
                                                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_105) : "r"(pairs_32[i_106]));
                                                uint32_t _bf16x2_max_52;
                                                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_52) : "r"(amax_pair_38), "r"(_bf16x2_abs_105));
                                                amax_pair_38 = _bf16x2_max_52;
                                            }
                                            uint16_t _bf16_max_52;
                                            asm("max.bf16 %0, %1, %2;" : "=h"(_bf16_max_52) : "h"((uint16_t)(amax_pair_38 & 65535)), "h"((uint16_t)(amax_pair_38 >> 16)));
                                            float _cvt_f32_bf16_52;
                                            asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_52) : "h"((uint16_t)(_bf16_max_52)));
                                            float amax_38 = _cvt_f32_bf16_52;
                                            float _fmax_52 = fmaxf(amax_38 * 0.002232142857f, 1e-12f);
                                            float scale_38 = _fmax_52;
                                            uint16_t _ue8m0x2_f32_52;
                                            asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_ue8m0x2_f32_52) : "f"(scale_38), "f"(scale_38));
                                            unsigned int scale_byte_38 = (unsigned int)_ue8m0x2_f32_52 & 255;
                                            unsigned int inverse_lane_37 = 254 - scale_byte_38 << 7;
                                            unsigned int inverse_0_4 = inverse_lane_37 | inverse_lane_37 << 16;
                                            unsigned int words_37[8];
                                            #pragma unroll
                                            for (int i_107 = 0; i_107 < 8; i_107++) {
                                                uint32_t _bf16x2_mul_104;
                                                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_104) : "r"(pairs_32[i_107 * 2]), "r"(inverse_0_4));
                                                uint16_t _e4m3x2_104;
                                                asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_104) : "r"(_bf16x2_mul_104));
                                                uint32_t _bf16x2_mul_105;
                                                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_105) : "r"(pairs_32[i_107 * 2 + 1]), "r"(inverse_0_4));
                                                uint16_t _e4m3x2_105;
                                                asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_105) : "r"(_bf16x2_mul_105));
                                                words_37[i_107] = (unsigned int)_e4m3x2_104 | (unsigned int)_e4m3x2_105 << 16;
                                            }
                                            scale_word_34 = scale_word_34 | scale_byte_38 << (unsigned int)(k_block_34 * 8);
                                            #pragma unroll
                                            for (int k_69 = 0; k_69 < 8; k_69++) {
                                                int col_0_17 = k_block_34 * 32 + (tid * 4 + k_69 * 4) % 32;
                                                smem_v78[(row_0_35 * 128 + col_0_17) / 4] = words_37[k_69];
                                            }
                                        }
                                        smem_v80[row_0_35 % 32 * 4 + row_0_35 / 32] = scale_word_34;
                                    }
                                } else if (tid - 128 < 128) {
                                    int row_0_36 = tid - 128;
                                    row_0_36 = (tid - 128) % 64 * 2 + (tid - 128) / 64;
                                    unsigned int scale_word_35 = 0;
                                    #pragma unroll 1
                                    for (int j_69 = 0; j_69 < 4; j_69++) {
                                        int k_block_35 = (j_69 + (tid - 128) / 8) % 4;
                                        unsigned int pairs_33[16];
                                        #pragma unroll
                                        for (int k_70 = 0; k_70 < 16; k_70++) {
                                            int col_0_18 = k_block_35 * 32 + ((tid - 128) * 4 + k_70 * 2) % 32;
                                            float x0_33 = 0.0f;
                                            float x1_33 = 0.0f;
                                            x0_33 = (float)smem_v76[col_0_18 * 128 + row_0_36];
                                            x1_33 = (float)smem_v76[(col_0_18 + 1) * 128 + row_0_36];
                                            __nv_bfloat162 _bf16x2_43 = __float22bfloat162_rn(make_float2(x0_33, x1_33));
                                            pairs_33[k_70] = __as_u32(_bf16x2_43);
                                        }
                                        uint32_t _bf16x2_abs_106;
                                        asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_106) : "r"(pairs_33[0]));
                                        unsigned int amax_pair_39 = _bf16x2_abs_106;
                                        #pragma unroll
                                        for (int i_108 = 1; i_108 < 16; i_108++) {
                                            uint32_t _bf16x2_abs_107;
                                            asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_107) : "r"(pairs_33[i_108]));
                                            uint32_t _bf16x2_max_53;
                                            asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_53) : "r"(amax_pair_39), "r"(_bf16x2_abs_107));
                                            amax_pair_39 = _bf16x2_max_53;
                                        }
                                        uint16_t _bf16_max_53;
                                        asm("max.bf16 %0, %1, %2;" : "=h"(_bf16_max_53) : "h"((uint16_t)(amax_pair_39 & 65535)), "h"((uint16_t)(amax_pair_39 >> 16)));
                                        float _cvt_f32_bf16_53;
                                        asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_53) : "h"((uint16_t)(_bf16_max_53)));
                                        float amax_39 = _cvt_f32_bf16_53;
                                        float _fmax_53 = fmaxf(amax_39 * 0.002232142857f, 1e-12f);
                                        float scale_39 = _fmax_53;
                                        uint16_t _ue8m0x2_f32_53;
                                        asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_ue8m0x2_f32_53) : "f"(scale_39), "f"(scale_39));
                                        unsigned int scale_byte_39 = (unsigned int)_ue8m0x2_f32_53 & 255;
                                        unsigned int inverse_lane_39 = 254 - scale_byte_39 << 7;
                                        unsigned int inverse_0_5 = inverse_lane_39 | inverse_lane_39 << 16;
                                        unsigned int words_38[8];
                                        #pragma unroll
                                        for (int i_109 = 0; i_109 < 8; i_109++) {
                                            uint32_t _bf16x2_mul_106;
                                            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_106) : "r"(pairs_33[i_109 * 2]), "r"(inverse_0_5));
                                            uint16_t _e4m3x2_106;
                                            asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_106) : "r"(_bf16x2_mul_106));
                                            uint32_t _bf16x2_mul_107;
                                            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_107) : "r"(pairs_33[i_109 * 2 + 1]), "r"(inverse_0_5));
                                            uint16_t _e4m3x2_107;
                                            asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_107) : "r"(_bf16x2_mul_107));
                                            words_38[i_109] = (unsigned int)_e4m3x2_106 | (unsigned int)_e4m3x2_107 << 16;
                                        }
                                        scale_word_35 = scale_word_35 | scale_byte_39 << (unsigned int)(k_block_35 * 8);
                                        #pragma unroll
                                        for (int k_71 = 0; k_71 < 8; k_71++) {
                                            int col_0_19 = k_block_35 * 32 + ((tid - 128) * 4 + k_71 * 4) % 32;
                                            smem_v79[(row_0_36 * 128 + col_0_19) / 4] = words_38[k_71];
                                        }
                                    }
                                    smem_v81[row_0_36 % 32 * 4 + row_0_36 / 32] = scale_word_35;
                                }
                                if (half_7 != 0) {
                                    smem_v82[128 + tile_row_1] = router_gradient_1;
                                }
                                __syncthreads();
                                if (tid == 0) {
                                    int local_row_11 = row_34 - macro_row_offset_2;
                                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                    tma_store_2d((&dg_t_store), local_row_11 * 128, col_74 * 128, smem_v78_addr);
                                    tma_store_3d((&dg_sc_t_store), 0, 0, col_74 * macro_tiles_4 + local_row_11, smem_v80_addr);
                                    tma_store_2d((&du_t_store), local_row_11 * 128, col_74 * 128, smem_v79_addr);
                                    tma_store_3d((&du_sc_t_store), 0, 0, col_74 * macro_tiles_4 + local_row_11, smem_v81_addr);
                                    asm volatile("cp.async.bulk.commit_group;");
                                    asm volatile("cp.async.bulk.wait_group.read 0;");
                                }
                                if (half_7 == 0 && peer_6 >= 0) {
                                    router_gradient_1 = router_gradient_1 + smem_v82[128 + tile_row_1];
                                    partials[local_token_1 * col_blocks_13 + col_74] = router_gradient_1;
                                }
                                __syncthreads();
                            }
                            if (tid == 0) {
                                asm volatile("cp.async.bulk.wait_group 0;");
                                if (tile_end_2 > first_tile_3) {
                                    int row_35 = first_row_5;
                                    if (col_blocks_13 <= first_col_2) {
                                        row_35 = row_35 + 1;
                                    }
                                    asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(dg_ready)) + (shared_rows + row_35 / 2))), "r"(static_cast<unsigned int>(1)) : "memory");
                                }
                                if (tile_end_2 > first_tile_3 + 1) {
                                    int row_36 = first_row_5;
                                    if (col_blocks_13 <= first_col_2 + 1) {
                                        row_36 = row_36 + 1;
                                    }
                                    asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(dg_ready)) + (shared_rows + row_36 / 2))), "r"(static_cast<unsigned int>(1)) : "memory");
                                }
                            }
                        }
                        if (tid == 0) {
                            bool enabled_value_18 = macros > 1;
                            if (enabled_value_18 != 0) {
                                asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(buffers_done)) + (macro_1))), "r"(static_cast<unsigned int>(1)) : "memory");
                            }
                        }
                        swiglu_phase = phase_bits_13;
                    } else {
                        if (kind == 2) {
                            int col_blocks_14 = (hidden + 256 - 1) / 256;
                            int x_8 = -1;
                            int y_8 = -1;
                            int expert_8 = -1;
                            int k_start_8 = 0;
                            int k_end_8 = 0;
                            int first_8 = 0;
                            int first_block_3 = (macro_1 * (macro_size / mini_size) + mini_1) * (mini_size / 256);
                            int _min_55 = ((first_block_3 + mini_size / 256) < (tokens / 256) ? (first_block_3 + mini_size / 256) : (tokens / 256));
                            int end_block_3 = _min_55;
                            int block_4 = first_block_3 + task_3 / col_blocks_14;
                            if (block_4 < end_block_3) {
                                int index_5 = counts[3 * experts + block_4];
                                int offset_11 = counts[experts + index_5] / 256;
                                int _max_20 = ((first_block_3) > (offset_11) ? (first_block_3) : (offset_11));
                                int first_row_6 = _max_20;
                                int _min_56 = ((end_block_3) < (offset_11 + counts[index_5] / 256) ? (end_block_3) : (offset_11 + counts[index_5] / 256));
                                int rows_6 = _min_56 - first_row_6;
                                int supergroup_8 = (task_3 - (first_row_6 - first_block_3) * col_blocks_14) / (rows_6 * 8);
                                int full_cols_8 = col_blocks_14 / 8 * 8;
                                int row_37 = 0;
                                int col_75 = 0;
                                if (task_3 - (first_row_6 - first_block_3) * col_blocks_14 < rows_6 * full_cols_8) {
                                    row_37 = (task_3 - (first_row_6 - first_block_3) * col_blocks_14) % (rows_6 * 8) / 8;
                                    col_75 = supergroup_8 * 8 + (task_3 - (first_row_6 - first_block_3) * col_blocks_14) % 8;
                                } else {
                                    row_37 = (task_3 - (first_row_6 - first_block_3) * col_blocks_14 - rows_6 * full_cols_8) / (col_blocks_14 - full_cols_8);
                                    col_75 = full_cols_8 + (task_3 - (first_row_6 - first_block_3) * col_blocks_14 - rows_6 * full_cols_8) % (col_blocks_14 - full_cols_8);
                                }
                                if ((supergroup_8 & 1) != 0) {
                                    row_37 = rows_6 - row_37 - 1;
                                }
                                x_8 = first_row_6 + row_37 - macro_1 * (macro_size / 256);
                                y_8 = col_75;
                                expert_8 = index_5;
                            }
                            unsigned int phase_bits_14 = gemm_phase;
                            int has_hi_8 = 0;
                            int global_mini_10 = macro_1 * (macro_size / mini_size) + mini_1;
                            int macro_rows_8 = macro_1 * (macro_size / 256);
                            int iterations_8 = intermediate / 128 + intermediate / 128;
                            int macro_k_3 = macro_1 * (macro_size / 128);
                            if (expert_8 < 0) {
                                if (tid == 0) {
                                    bool enabled_value_19 = macros > 1;
                                    if (enabled_value_19 != 0) {
                                        asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(buffers_done)) + (macro_1))), "r"(static_cast<unsigned int>(1)) : "memory");
                                    }
                                }
                            } else if (tid / 32 == 7) {
                                if (warp == 7) {
                                    if (elect_sync()) {
                                        {
                                            bool enabled_value_20 = 1;
                                            if (enabled_value_20 != 0) {
                                                int32_t _relaxed_ld_48;
                                                asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_48) : "l"(dg_ready + (shared_rows + macro_rows_8 + x_8)) : "memory");
                                                int value_22 = _relaxed_ld_48;
                                                while (value_22 < row_count) {
                                                    asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                                    int32_t _relaxed_ld_49;
                                                    asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_49) : "l"(dg_ready + (shared_rows + macro_rows_8 + x_8)) : "memory");
                                                    value_22 = _relaxed_ld_49;
                                                }
                                                asm volatile("fence.acquire.gpu;" ::: "memory");
                                            }
                                            int _min_57 = ((mini_size) < (tokens - global_mini_10 * mini_size) ? (mini_size) : (tokens - global_mini_10 * mini_size));
                                            int _max_21 = ((0) > (_min_57) ? (0) : (_min_57));
                                            int mini_rows_12 = _max_21;
                                            int required_11 = (mini_rows_12 + 127) / 128 * ((intermediate + 511) / 512);
                                        }
                                        unsigned int previous_11 = phase_bits_14 >> 7 & 1;
                                        unsigned int bits_11 = phase_bits_14;
                                        if (previous_11 != 1) {
                                            mbarrier_wait(gemm_finished_addr, bits_11 >> 16 & 1);
                                            mbarrier_wait(gemm_finished_addr + 8, bits_11 >> 17 & 1);
                                            mbarrier_wait(gemm_finished_addr + 16, bits_11 >> 18 & 1);
                                            mbarrier_wait(gemm_finished_addr + 24, bits_11 >> 19 & 1);
                                            mbarrier_wait(gemm_finished_addr + 32, bits_11 >> 20 & 1);
                                            mbarrier_wait(gemm_finished_addr + 40, bits_11 >> 21 & 1);
                                            bits_11 = bits_11 ^ 128;
                                        }
                                        phase_bits_14 = bits_11;
                                        int ring_19 = 0;
                                        #pragma unroll 1
                                        for (int idx_19 = 0; idx_19 < iterations_8; idx_19++) {
                                            mbarrier_wait(gemm_finished_addr + (ring_19) * 8, phase_bits_14 >> (unsigned int)(16 + ring_19) & 1);
                                            if (idx_19 < intermediate / 128) {
                                                asm volatile(
                                                    "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                                    " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                                    :: "r"(smem_v52_addr + (unsigned int)(ring_19 * 16384)), "l"((&dg_q)), "r"(0), "r"(x_8 * 256 + cta_rank_0 * 128), "r"(idx_19), "r"(0), "r"(0),
                                                       "r"(((gemm_arrived_addr + (ring_19) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                                asm volatile(
                                                    "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                                    " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                                    :: "r"(smem_v53_addr + (unsigned int)(ring_19 * 16384)), "l"((&wg_t_q)), "r"(0), "r"(y_8 * 256 + cta_rank_0 * 128), "r"(idx_19), "r"(expert_8), "r"(0),
                                                       "r"(((gemm_arrived_addr + (ring_19) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                            } else {
                                                asm volatile(
                                                    "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                                    " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                                    :: "r"(smem_v52_addr + (unsigned int)(ring_19 * 16384)), "l"((&du_q)), "r"(0), "r"(x_8 * 256 + cta_rank_0 * 128), "r"(idx_19 - intermediate / 128), "r"(0), "r"(0),
                                                       "r"(((gemm_arrived_addr + (ring_19) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                                asm volatile(
                                                    "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                                    " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                                    :: "r"(smem_v53_addr + (unsigned int)(ring_19 * 16384)), "l"((&wu_t_q)), "r"(0), "r"(y_8 * 256 + cta_rank_0 * 128), "r"(idx_19 - intermediate / 128), "r"(expert_8), "r"(0),
                                                       "r"(((gemm_arrived_addr + (ring_19) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                            }
                                            phase_bits_14 = phase_bits_14 ^ (unsigned int)(1 << 16 + ring_19);
                                            ring_19 = (ring_19 + 1) % 6;
                                        }
                                    }
                                }
                            } else {
                                if (tid / 32 == 6) {
                                    if (warp == 6) {
                                        if (elect_sync()) {
                                            {
                                                bool enabled_value_21 = 1;
                                                if (enabled_value_21 != 0) {
                                                    int32_t _relaxed_ld_50;
                                                    asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_50) : "l"(dg_ready + (shared_rows + macro_rows_8 + x_8)) : "memory");
                                                    int value_23 = _relaxed_ld_50;
                                                    while (value_23 < row_count) {
                                                        asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                                        int32_t _relaxed_ld_51;
                                                        asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_51) : "l"(dg_ready + (shared_rows + macro_rows_8 + x_8)) : "memory");
                                                        value_23 = _relaxed_ld_51;
                                                    }
                                                    asm volatile("fence.acquire.gpu;" ::: "memory");
                                                }
                                                int _min_58 = ((mini_size) < (tokens - global_mini_10 * mini_size) ? (mini_size) : (tokens - global_mini_10 * mini_size));
                                                int _max_22 = ((0) > (_min_58) ? (0) : (_min_58));
                                                int mini_rows_13 = _max_22;
                                                int required_12 = (mini_rows_13 + 127) / 128 * ((intermediate + 511) / 512);
                                            }
                                            unsigned int previous_12 = phase_bits_14 >> 7 & 1;
                                            unsigned int bits_12 = phase_bits_14;
                                            if (previous_12 != 1) {
                                                mbarrier_wait(scales_finished_addr, bits_12 >> 16 & 1);
                                                mbarrier_wait(scales_finished_addr + 8, bits_12 >> 17 & 1);
                                                mbarrier_wait(scales_finished_addr + 16, bits_12 >> 18 & 1);
                                                mbarrier_wait(scales_finished_addr + 24, bits_12 >> 19 & 1);
                                                mbarrier_wait(scales_finished_addr + 32, bits_12 >> 20 & 1);
                                                mbarrier_wait(scales_finished_addr + 40, bits_12 >> 21 & 1);
                                                bits_12 = bits_12 ^ 128;
                                            }
                                            phase_bits_14 = bits_12;
                                            int ring_20 = 0;
                                            #pragma unroll 1
                                            for (int idx_20 = 0; idx_20 < iterations_8; idx_20++) {
                                                mbarrier_wait(scales_finished_addr + (ring_20) * 8, phase_bits_14 >> (unsigned int)(16 + ring_20) & 1);
                                                if (idx_20 < intermediate / 128) {
                                                    asm volatile(
                                                        "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                                        " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                                        :: "r"(smem_v54_addr + (unsigned int)(ring_20 * 512)), "l"((&dg_sc)), "r"(0), "r"(0), "r"((x_8 * 2 + cta_rank_0) * (intermediate / 128) + idx_20),
                                                           "r"(((scales_arrived_addr + (ring_20) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                                    asm volatile(
                                                        "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                                        " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                                        :: "r"(smem_v55_addr + (unsigned int)(ring_20 * 1024) + (unsigned int)(cta_rank_0 * 512)), "l"((&wg_t_sc)), "r"(0), "r"(0), "r"((expert_8 * h_tiles + y_8 * 2 + cta_rank_0) * (intermediate / 128) + idx_20),
                                                           "r"(((scales_arrived_addr + (ring_20) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(3)) : "memory");
                                                } else {
                                                    asm volatile(
                                                        "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                                        " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                                        :: "r"(smem_v54_addr + (unsigned int)(ring_20 * 512)), "l"((&du_sc)), "r"(0), "r"(0), "r"((x_8 * 2 + cta_rank_0) * (intermediate / 128) + (idx_20 - intermediate / 128)),
                                                           "r"(((scales_arrived_addr + (ring_20) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                                    asm volatile(
                                                        "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                                        " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                                        :: "r"(smem_v55_addr + (unsigned int)(ring_20 * 1024) + (unsigned int)(cta_rank_0 * 512)), "l"((&wu_t_sc)), "r"(0), "r"(0), "r"((expert_8 * h_tiles + y_8 * 2 + cta_rank_0) * (intermediate / 128) + (idx_20 - intermediate / 128)),
                                                           "r"(((scales_arrived_addr + (ring_20) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(3)) : "memory");
                                                }
                                                phase_bits_14 = phase_bits_14 ^ (unsigned int)(1 << 16 + ring_20);
                                                ring_20 = (ring_20 + 1) % 6;
                                            }
                                        }
                                    }
                                } else if (tid / 32 == 4 && cta_rank_0 == 0) {
                                    if (warp == 4) {
                                        if (elect_sync()) {
                                            int ring_21 = 0;
                                            mbarrier_wait(output_finished_addr, phase_bits_14 >> 22 & 1);
                                            phase_bits_14 = phase_bits_14 ^ 4194304;
                                            asm volatile("tcgen05.fence::after_thread_sync;");
                                            #pragma unroll 1
                                            for (int idx_21 = 0; idx_21 < iterations_8; idx_21++) {
                                                mbarrier_arrive_expect_tx(scales_arrived_addr + (ring_21) * 8, 3072);
                                                mbarrier_wait(scales_arrived_addr + (ring_21) * 8, phase_bits_14 >> (unsigned int)(8 + ring_21) & 1);
                                                int buffer_3 = idx_21 % 3;
                                                tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfa + buffer_3 * 4, make_sf_cp_desc_sbo128(smem_v54_addr + (unsigned int)(ring_21 * 512)));
                                                tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfb + buffer_3 * 8, make_sf_cp_desc_sbo128(smem_v55_addr + (unsigned int)(ring_21 * 1024)));
                                                tcgen05_cp_32x128b_warpx4_cta2((tmem_tmem_sfb + buffer_3 * 8 + 4), make_sf_cp_desc_sbo128((smem_v55_addr + (unsigned int)(ring_21 * 1024) + 512)));
                                                tcgen05_commit_cg2_multicast(scales_finished_addr + (ring_21) * 8, (uint16_t)(3));
                                                mbarrier_arrive_expect_tx(gemm_arrived_addr + (ring_21) * 8, 65536);
                                                mbarrier_wait(gemm_arrived_addr + (ring_21) * 8, phase_bits_14 >> (unsigned int)ring_21 & 1);
                                                int _mma_a_lo_13 = (((smem_v52_addr) >> 4) & 0x3FFF) + (ring_21) * 1024;
                                                int _mma_b_lo_13 = (((smem_v53_addr) >> 4) & 0x3FFF) + (ring_21) * 1024;
                                                {
                                                    uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_13) | ((uint64_t)0x40004040 << 32);
                                                    uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_13) | ((uint64_t)0x40004040 << 32);

                                                    tcgen05_mma_mxf8_bs_cta2(tmem_accumulator, a_desc + 0, b_desc + 0,
                                                        0x10c00000U, tmem_tmem_sfa + buffer_3 * 4, tmem_tmem_sfb + buffer_3 * 8, ((idx_21 == 0) ? 0 : 1));
                                                    tcgen05_mma_mxf8_bs_cta2(tmem_accumulator, a_desc + 2, b_desc + 2,
                                                        0x30c00010U, tmem_tmem_sfa + buffer_3 * 4, tmem_tmem_sfb + buffer_3 * 8, 1);
                                                    tcgen05_mma_mxf8_bs_cta2(tmem_accumulator, a_desc + 4, b_desc + 4,
                                                        0x50c00020U, tmem_tmem_sfa + buffer_3 * 4, tmem_tmem_sfb + buffer_3 * 8, 1);
                                                    tcgen05_mma_mxf8_bs_cta2(tmem_accumulator, a_desc + 6, b_desc + 6,
                                                        0x70c00030U, tmem_tmem_sfa + buffer_3 * 4, tmem_tmem_sfb + buffer_3 * 8, 1);
                                                }
                                                tcgen05_commit_cg2_multicast(gemm_finished_addr + (ring_21) * 8, (uint16_t)(3));
                                                phase_bits_14 = phase_bits_14 ^ (unsigned int)(1 << ring_21) ^ (unsigned int)(1 << 8 + ring_21);
                                                ring_21 = (ring_21 + 1) % 6;
                                            }
                                            tcgen05_commit_cg2_multicast(output_arrived_addr, (uint16_t)(3));
                                        }
                                    }
                                } else {
                                    if (tid < 128) {
                                        mbarrier_wait(output_arrived_addr, phase_bits_14 >> 6 & 1);
                                        unsigned int packed_19[128];
                                        #pragma unroll
                                        for (int chunk_18 = 0; chunk_18 < 8; chunk_18++) {
                                            #pragma unroll
                                            for (int sub_6 = 0; sub_6 < 2; sub_6++) {
                                                unsigned int address_45 = taddr_1 + (unsigned int)(tid / 32 * 32 + sub_6 * 16 << 16) + (unsigned int)(chunk_18 * 32);
                                                float _tmem_load_14[16];
                                                asm volatile(
                                                    "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                                                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                                    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_14[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_14[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_14[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_14[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_14[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_14[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_14[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_14[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_14[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_14[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_14[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_14[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_14[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_14[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_14[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_14[15]))
                                                    : "r"(address_45));
                                                #pragma unroll
                                                for (int pair_20 = 0; pair_20 < 8; pair_20++) {
                                                    __nv_bfloat162 _bf16x2_44 = __float22bfloat162_rn(make_float2(_tmem_load_14[pair_20 * 2], _tmem_load_14[pair_20 * 2 + 1]));
                                                    packed_19[chunk_18 * 16 + sub_6 * 8 + pair_20] = __as_u32(_bf16x2_44);
                                                }
                                            }
                                        }
                                        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                                        int last_4 = 1;
                                        if (last_4 != 0) {
                                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                                            if (tid == 0) {
                                                asm volatile(
                                                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                                                    :: "r"((output_finished_addr) & 0xFEFFFFFF) : "memory");
                                            }
                                        }
                                        if (tid == 0) {
                                            int previous_offset_6 = (macro_1 + 1) * macro_size;
                                            int output_row_3 = x_8 * 256 + cta_rank_0 * 128;
                                            int _min_59 = ((macro_size) < (tokens - previous_offset_6) ? (macro_size) : (tokens - previous_offset_6));
                                            if (output_row_3 < _min_59) {
                                            }
                                        }
                                        #pragma unroll
                                        for (int chunk_19 = 0; chunk_19 < 8; chunk_19++) {
                                            if (tid == 0) {
                                                asm volatile("cp.async.bulk.wait_group.read 2;");
                                            }
                                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                                            int warp_0_10 = tid / 32;
                                            int lane_14 = tid % 32;
                                            #pragma unroll
                                            for (int half_8 = 0; half_8 < 2; half_8++) {
                                                #pragma unroll
                                                for (int col_tile_18 = 0; col_tile_18 < 2; col_tile_18++) {
                                                    int row_38 = warp_0_10 * 32 + half_8 * 16 + lane_14 % 16;
                                                    int col_76 = col_tile_18 * 16 + lane_14 / 16 * 8;
                                                    unsigned int address_46 = d_smem_addr + (unsigned int)(chunk_19 % 3 * 8192) + (unsigned int)((row_38 * 32 + col_76) * 2);
                                                    address_46 = address_46 ^ (address_46 & 511) >> 7 << 4;
                                                    int offset_12 = chunk_19 * 16 + half_8 * 8 + col_tile_18 * 4;
                                                    uint32_t _stmatrix_addr_12 = static_cast<uint32_t>(address_46);
                                                    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                                        :: "r"(_stmatrix_addr_12), "r"(*reinterpret_cast<const uint32_t*>(&packed_19[offset_12])), "r"(*reinterpret_cast<const uint32_t*>(&packed_19[offset_12 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&packed_19[offset_12 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&packed_19[offset_12 + 3]))
                                                        : "memory");
                                                }
                                            }
                                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                                            if (tid == 0) {
                                                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                                asm volatile(
                                                    "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                                    " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                                    :: "l"((&dx_r)), "r"(0), "r"(x_8 * 256 + cta_rank_0 * 128), "r"(y_8 * 8 + chunk_19), "r"(0), "r"(0), "r"(d_smem_addr + (unsigned int)(chunk_19 % 3 * 8192)), "l"(0x12F0000000000000ULL) : "memory");
                                                asm volatile("cp.async.bulk.commit_group;");
                                            }
                                        }
                                        if (has_hi_8 != 0) {
                                            unsigned int packed_0_3[128];
                                            #pragma unroll
                                            for (int chunk_20 = 0; chunk_20 < 8; chunk_20++) {
                                                #pragma unroll
                                                for (int sub_7 = 0; sub_7 < 2; sub_7++) {
                                                    unsigned int address_47 = taddr_1 + (unsigned int)(tid / 32 * 32 + sub_7 * 16 << 16) + 256 + (unsigned int)(chunk_20 * 32);
                                                    float _tmem_load_15[16];
                                                    asm volatile(
                                                        "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                                                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                                        : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_15[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_15[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_15[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_15[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_15[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_15[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_15[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_15[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_15[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_15[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_15[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_15[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_15[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_15[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_15[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_15[15]))
                                                        : "r"(address_47));
                                                    #pragma unroll
                                                    for (int pair_21 = 0; pair_21 < 8; pair_21++) {
                                                        __nv_bfloat162 _bf16x2_45 = __float22bfloat162_rn(make_float2(_tmem_load_15[pair_21 * 2], _tmem_load_15[pair_21 * 2 + 1]));
                                                        packed_0_3[chunk_20 * 16 + sub_7 * 8 + pair_21] = __as_u32(_bf16x2_45);
                                                    }
                                                }
                                            }
                                            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                                            int last_1_3 = 1;
                                            if (last_1_3 != 0) {
                                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                                if (tid == 0) {
                                                    asm volatile(
                                                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                                                        :: "r"((output_finished_addr) & 0xFEFFFFFF) : "memory");
                                                }
                                            }
                                            #pragma unroll
                                            for (int chunk_21 = 0; chunk_21 < 8; chunk_21++) {
                                                if (tid == 0) {
                                                    asm volatile("cp.async.bulk.wait_group.read 2;");
                                                }
                                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                                int warp_0_11 = tid / 32;
                                                int lane_15 = tid % 32;
                                                #pragma unroll
                                                for (int half_9 = 0; half_9 < 2; half_9++) {
                                                    #pragma unroll
                                                    for (int col_tile_19 = 0; col_tile_19 < 2; col_tile_19++) {
                                                        int row_39 = warp_0_11 * 32 + half_9 * 16 + lane_15 % 16;
                                                        int col_77 = col_tile_19 * 16 + lane_15 / 16 * 8;
                                                        unsigned int address_48 = d_smem_addr + (unsigned int)((8 + chunk_21) % 3 * 8192) + (unsigned int)((row_39 * 32 + col_77) * 2);
                                                        address_48 = address_48 ^ (address_48 & 511) >> 7 << 4;
                                                        int offset_13 = chunk_21 * 16 + half_9 * 8 + col_tile_19 * 4;
                                                        uint32_t _stmatrix_addr_13 = static_cast<uint32_t>(address_48);
                                                        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                                            :: "r"(_stmatrix_addr_13), "r"(*reinterpret_cast<const uint32_t*>(&packed_0_3[offset_13])), "r"(*reinterpret_cast<const uint32_t*>(&packed_0_3[offset_13 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&packed_0_3[offset_13 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&packed_0_3[offset_13 + 3]))
                                                            : "memory");
                                                    }
                                                }
                                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                                if (tid == 0) {
                                                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                                    asm volatile(
                                                        "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                                        " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                                        :: "l"((&dx_r)), "r"(0), "r"(x_8 * 256 + cta_rank_0 * 128), "r"((y_8 + 1) * 8 + chunk_21), "r"(0), "r"(0), "r"(d_smem_addr + (unsigned int)((8 + chunk_21) % 3 * 8192)), "l"(0x12F0000000000000ULL) : "memory");
                                                    asm volatile("cp.async.bulk.commit_group;");
                                                }
                                            }
                                        }
                                        if (tid == 0) {
                                            asm volatile("cp.async.bulk.wait_group.read 0;");
                                        }
                                        asm volatile("barrier.sync 4, 128;" ::: "memory");
                                        phase_bits_14 = phase_bits_14 ^ 64;
                                        if (tid / 32 == 0) {
                                            if (warp == 0) {
                                                if (elect_sync()) {
                                                    asm volatile("cp.async.bulk.wait_group 0;");
                                                    bool enabled_value_22 = 1;
                                                    if (enabled_value_22 != 0) {
                                                        asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(dx_ready)) + (global_mini_10))), "r"(static_cast<unsigned int>(1)) : "memory");
                                                    }
                                                    bool enabled_value_0_2 = macros > 1;
                                                    if (enabled_value_0_2 != 0) {
                                                        asm volatile("cp.async.bulk.wait_group 0;");
                                                        bool enabled_value_1_2 = 1;
                                                        if (enabled_value_1_2 != 0) {
                                                            asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(buffers_done)) + (macro_1))), "r"(static_cast<unsigned int>(1)) : "memory");
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                            gemm_phase = phase_bits_14;
                        } else if (kind == 3) {
                            int col_blocks_15 = (macro_size + 256 - 1) / 256;
                            {
                                col_blocks_15 = (intermediate + 256 - 1) / 256;
                            }
                            int x_9 = -1;
                            int y_9 = -1;
                            int expert_9 = -1;
                            int k_start_9 = 0;
                            int k_end_9 = 0;
                            int first_9 = 0;
                            int row_blocks_5 = hidden / 256;
                            int expert_idx_3 = 0;
                            int local_task_3 = task_3;
                            int _max_23 = ((row_blocks_5 * col_blocks_15) > (intermediate / 256 * ((hidden + 256 - 1) / 256)) ? (row_blocks_5 * col_blocks_15) : (intermediate / 256 * ((hidden + 256 - 1) / 256)));
                            int stride_3 = _max_23;
                            expert_idx_3 = task_3 / stride_3;
                            local_task_3 = task_3 % stride_3;
                            int offset_14 = counts[experts + expert_idx_3];
                            int real = counts[2 * experts + expert_idx_3];
                            real = (real + 128 - 1) / 128 * 128;
                            int _max_24 = ((offset_14) > (macro_1 * macro_size) ? (offset_14) : (macro_1 * macro_size));
                            k_start_9 = _max_24;
                            int _min_60 = (((macro_1 + 1) * macro_size) < (tokens) ? ((macro_1 + 1) * macro_size) : (tokens));
                            int _min_61 = ((offset_14 + real) < (_min_60) ? (offset_14 + real) : (_min_60));
                            k_end_9 = _min_61;
                            first_9 = (int)(k_start_9 == offset_14);
                            if (k_start_9 < k_end_9 && local_task_3 < row_blocks_5 * col_blocks_15) {
                                int supergroup_9 = local_task_3 / (row_blocks_5 * 8);
                                int full_cols_9 = col_blocks_15 / 8 * 8;
                                int row_40 = 0;
                                int col_78 = 0;
                                if (local_task_3 < row_blocks_5 * full_cols_9) {
                                    row_40 = local_task_3 % (row_blocks_5 * 8) / 8;
                                    col_78 = supergroup_9 * 8 + local_task_3 % 8;
                                } else {
                                    row_40 = (local_task_3 - row_blocks_5 * full_cols_9) / (col_blocks_15 - full_cols_9);
                                    col_78 = full_cols_9 + (local_task_3 - row_blocks_5 * full_cols_9) % (col_blocks_15 - full_cols_9);
                                }
                                if ((supergroup_9 & 1) != 0) {
                                    row_40 = row_blocks_5 - row_40 - 1;
                                }
                                x_9 = row_40;
                                y_9 = col_78;
                                expert_9 = expert_idx_3;
                            }
                            unsigned int phase_bits_15 = gemm_phase;
                            int has_hi_9 = 0;
                            int global_mini_11 = macro_1 * (macro_size / mini_size);
                            int macro_rows_9 = macro_1 * (macro_size / 256);
                            int iterations_9 = hidden / 128;
                            int macro_k_4 = macro_1 * (macro_size / 128);
                            iterations_9 = (k_end_9 - k_start_9) / 128;
                            if (expert_9 < 0) {
                                if (tid == 0) {
                                    bool enabled_value_23 = macros > 1;
                                    if (enabled_value_23 != 0) {
                                        asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(buffers_done)) + (macro_1))), "r"(static_cast<unsigned int>(1)) : "memory");
                                    }
                                }
                            } else if (tid / 32 == 7) {
                                if (warp == 7) {
                                    if (elect_sync()) {
                                        unsigned int previous_13 = phase_bits_15 >> 7 & 1;
                                        unsigned int bits_13 = phase_bits_15;
                                        if (previous_13 != 1) {
                                            mbarrier_wait(gemm_finished_addr, bits_13 >> 16 & 1);
                                            mbarrier_wait(gemm_finished_addr + 8, bits_13 >> 17 & 1);
                                            mbarrier_wait(gemm_finished_addr + 16, bits_13 >> 18 & 1);
                                            mbarrier_wait(gemm_finished_addr + 24, bits_13 >> 19 & 1);
                                            mbarrier_wait(gemm_finished_addr + 32, bits_13 >> 20 & 1);
                                            mbarrier_wait(gemm_finished_addr + 40, bits_13 >> 21 & 1);
                                            bits_13 = bits_13 ^ 128;
                                        }
                                        phase_bits_15 = bits_13;
                                        int ring_22 = 0;
                                        #pragma unroll 1
                                        for (int idx_22 = 0; idx_22 < iterations_9; idx_22++) {
                                            int token_row_3 = k_start_9 + idx_22 * 128;
                                            if (idx_22 == 0 || token_row_3 % 256 == 0) {
                                                bool enabled_value_24 = macro_1 > 0;
                                                if (enabled_value_24 != 0) {
                                                    int32_t _relaxed_ld_56;
                                                    asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_56) : "l"(replay_h + (token_row_3 / 256)) : "memory");
                                                    int value_24 = _relaxed_ld_56;
                                                    while (value_24 < row_count) {
                                                        asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                                        int32_t _relaxed_ld_57;
                                                        asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_57) : "l"(replay_h + (token_row_3 / 256)) : "memory");
                                                        value_24 = _relaxed_ld_57;
                                                    }
                                                    asm volatile("fence.acquire.gpu;" ::: "memory");
                                                }
                                            }
                                            if (idx_22 == 0 || token_row_3 % mini_size == 0) {
                                                int input_mini_3 = token_row_3 / mini_size;
                                                int _min_63 = ((mini_size) < (tokens - input_mini_3 * mini_size) ? (mini_size) : (tokens - input_mini_3 * mini_size));
                                                int input_rows_3 = _min_63;
                                                int input_count_3 = (input_rows_3 + 127) / 128 * ((hidden + 511) / 512);
                                                bool enabled_value_25 = 1;
                                                if (enabled_value_25 != 0) {
                                                    int32_t _relaxed_ld_58;
                                                    asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_58) : "l"(dy_ready + input_mini_3) : "memory");
                                                    int value_25 = _relaxed_ld_58;
                                                    while (value_25 < input_count_3) {
                                                        asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                                        int32_t _relaxed_ld_59;
                                                        asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_59) : "l"(dy_ready + input_mini_3) : "memory");
                                                        value_25 = _relaxed_ld_59;
                                                    }
                                                    asm volatile("fence.acquire.gpu;" ::: "memory");
                                                }
                                            }
                                            mbarrier_wait(gemm_finished_addr + (ring_22) * 8, phase_bits_15 >> (unsigned int)(16 + ring_22) & 1);
                                            asm volatile(
                                                "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                                " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                                :: "r"(smem_v52_addr + (unsigned int)(ring_22 * 16384)), "l"((&dy_t)), "r"(0), "r"(x_9 * 256 + cta_rank_0 * 128), "r"(k_start_9 / 128 + idx_22 - macro_k_4), "r"(0), "r"(0),
                                                   "r"(((gemm_arrived_addr + (ring_22) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                            asm volatile(
                                                "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                                " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                                :: "r"(smem_v53_addr + (unsigned int)(ring_22 * 16384)), "l"((&h_t)), "r"(0), "r"(y_9 * 256 + cta_rank_0 * 128), "r"(k_start_9 / 128 + idx_22 - macro_k_4), "r"(0), "r"(0),
                                                   "r"(((gemm_arrived_addr + (ring_22) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                            phase_bits_15 = phase_bits_15 ^ (unsigned int)(1 << 16 + ring_22);
                                            ring_22 = (ring_22 + 1) % 6;
                                        }
                                    }
                                }
                            } else {
                                if (tid / 32 == 6) {
                                    if (warp == 6) {
                                        if (elect_sync()) {
                                            unsigned int previous_14 = phase_bits_15 >> 7 & 1;
                                            unsigned int bits_14 = phase_bits_15;
                                            if (previous_14 != 1) {
                                                mbarrier_wait(scales_finished_addr, bits_14 >> 16 & 1);
                                                mbarrier_wait(scales_finished_addr + 8, bits_14 >> 17 & 1);
                                                mbarrier_wait(scales_finished_addr + 16, bits_14 >> 18 & 1);
                                                mbarrier_wait(scales_finished_addr + 24, bits_14 >> 19 & 1);
                                                mbarrier_wait(scales_finished_addr + 32, bits_14 >> 20 & 1);
                                                mbarrier_wait(scales_finished_addr + 40, bits_14 >> 21 & 1);
                                                bits_14 = bits_14 ^ 128;
                                            }
                                            phase_bits_15 = bits_14;
                                            int ring_23 = 0;
                                            #pragma unroll 1
                                            for (int idx_23 = 0; idx_23 < iterations_9; idx_23++) {
                                                int token_row_4 = k_start_9 + idx_23 * 128;
                                                if (idx_23 == 0 || token_row_4 % 256 == 0) {
                                                    bool enabled_value_26 = macro_1 > 0;
                                                    if (enabled_value_26 != 0) {
                                                        int32_t _relaxed_ld_64;
                                                        asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_64) : "l"(replay_h + (token_row_4 / 256)) : "memory");
                                                        int value_26 = _relaxed_ld_64;
                                                        while (value_26 < row_count) {
                                                            asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                                            int32_t _relaxed_ld_65;
                                                            asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_65) : "l"(replay_h + (token_row_4 / 256)) : "memory");
                                                            value_26 = _relaxed_ld_65;
                                                        }
                                                        asm volatile("fence.acquire.gpu;" ::: "memory");
                                                    }
                                                }
                                                if (idx_23 == 0 || token_row_4 % mini_size == 0) {
                                                    int input_mini_4 = token_row_4 / mini_size;
                                                    int _min_65 = ((mini_size) < (tokens - input_mini_4 * mini_size) ? (mini_size) : (tokens - input_mini_4 * mini_size));
                                                    int input_rows_4 = _min_65;
                                                    int input_count_4 = (input_rows_4 + 127) / 128 * ((hidden + 511) / 512);
                                                    bool enabled_value_27 = 1;
                                                    if (enabled_value_27 != 0) {
                                                        int32_t _relaxed_ld_66;
                                                        asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_66) : "l"(dy_ready + input_mini_4) : "memory");
                                                        int value_27 = _relaxed_ld_66;
                                                        while (value_27 < input_count_4) {
                                                            asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                                            int32_t _relaxed_ld_67;
                                                            asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_67) : "l"(dy_ready + input_mini_4) : "memory");
                                                            value_27 = _relaxed_ld_67;
                                                        }
                                                        asm volatile("fence.acquire.gpu;" ::: "memory");
                                                    }
                                                }
                                                mbarrier_wait(scales_finished_addr + (ring_23) * 8, phase_bits_15 >> (unsigned int)(16 + ring_23) & 1);
                                                asm volatile(
                                                    "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                                    " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                                    :: "r"(smem_v54_addr + (unsigned int)(ring_23 * 512)), "l"((&dy_sc_t)), "r"(0), "r"(0), "r"((x_9 * 2 + cta_rank_0) * (macro_size / 128) + (k_start_9 / 128 + idx_23 - macro_k_4)),
                                                       "r"(((scales_arrived_addr + (ring_23) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                                asm volatile(
                                                    "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                                    " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                                    :: "r"(smem_v55_addr + (unsigned int)(ring_23 * 1024) + (unsigned int)(cta_rank_0 * 512)), "l"((&h_sc_t)), "r"(0), "r"(0), "r"((y_9 * 2 + cta_rank_0) * (macro_size / 128) + (k_start_9 / 128 + idx_23 - macro_k_4)),
                                                       "r"(((scales_arrived_addr + (ring_23) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(3)) : "memory");
                                                phase_bits_15 = phase_bits_15 ^ (unsigned int)(1 << 16 + ring_23);
                                                ring_23 = (ring_23 + 1) % 6;
                                            }
                                        }
                                    }
                                } else if (tid / 32 == 4 && cta_rank_0 == 0) {
                                    if (warp == 4) {
                                        if (elect_sync()) {
                                            int ring_24 = 0;
                                            mbarrier_wait(output_finished_addr, phase_bits_15 >> 22 & 1);
                                            phase_bits_15 = phase_bits_15 ^ 4194304;
                                            asm volatile("tcgen05.fence::after_thread_sync;");
                                            #pragma unroll 1
                                            for (int idx_24 = 0; idx_24 < iterations_9; idx_24++) {
                                                mbarrier_arrive_expect_tx(scales_arrived_addr + (ring_24) * 8, 3072);
                                                mbarrier_wait(scales_arrived_addr + (ring_24) * 8, phase_bits_15 >> (unsigned int)(8 + ring_24) & 1);
                                                int buffer_4 = idx_24 % 3;
                                                tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfa + buffer_4 * 4, make_sf_cp_desc_sbo128(smem_v54_addr + (unsigned int)(ring_24 * 512)));
                                                tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfb + buffer_4 * 8, make_sf_cp_desc_sbo128(smem_v55_addr + (unsigned int)(ring_24 * 1024)));
                                                tcgen05_cp_32x128b_warpx4_cta2((tmem_tmem_sfb + buffer_4 * 8 + 4), make_sf_cp_desc_sbo128((smem_v55_addr + (unsigned int)(ring_24 * 1024) + 512)));
                                                tcgen05_commit_cg2_multicast(scales_finished_addr + (ring_24) * 8, (uint16_t)(3));
                                                mbarrier_arrive_expect_tx(gemm_arrived_addr + (ring_24) * 8, 65536);
                                                mbarrier_wait(gemm_arrived_addr + (ring_24) * 8, phase_bits_15 >> (unsigned int)ring_24 & 1);
                                                int _mma_a_lo_14 = (((smem_v52_addr) >> 4) & 0x3FFF) + (ring_24) * 1024;
                                                int _mma_b_lo_14 = (((smem_v53_addr) >> 4) & 0x3FFF) + (ring_24) * 1024;
                                                {
                                                    uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_14) | ((uint64_t)0x40004040 << 32);
                                                    uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_14) | ((uint64_t)0x40004040 << 32);

                                                    tcgen05_mma_mxf8_bs_cta2(tmem_accumulator, a_desc + 0, b_desc + 0,
                                                        0x10c00000U, tmem_tmem_sfa + buffer_4 * 4, tmem_tmem_sfb + buffer_4 * 8, ((idx_24 == 0) ? 0 : 1));
                                                    tcgen05_mma_mxf8_bs_cta2(tmem_accumulator, a_desc + 2, b_desc + 2,
                                                        0x30c00010U, tmem_tmem_sfa + buffer_4 * 4, tmem_tmem_sfb + buffer_4 * 8, 1);
                                                    tcgen05_mma_mxf8_bs_cta2(tmem_accumulator, a_desc + 4, b_desc + 4,
                                                        0x50c00020U, tmem_tmem_sfa + buffer_4 * 4, tmem_tmem_sfb + buffer_4 * 8, 1);
                                                    tcgen05_mma_mxf8_bs_cta2(tmem_accumulator, a_desc + 6, b_desc + 6,
                                                        0x70c00030U, tmem_tmem_sfa + buffer_4 * 4, tmem_tmem_sfb + buffer_4 * 8, 1);
                                                }
                                                tcgen05_commit_cg2_multicast(gemm_finished_addr + (ring_24) * 8, (uint16_t)(3));
                                                phase_bits_15 = phase_bits_15 ^ (unsigned int)(1 << ring_24) ^ (unsigned int)(1 << 8 + ring_24);
                                                ring_24 = (ring_24 + 1) % 6;
                                            }
                                            tcgen05_commit_cg2_multicast(output_arrived_addr, (uint16_t)(3));
                                        }
                                    }
                                } else {
                                    if (tid < 128) {
                                        mbarrier_wait(output_arrived_addr, phase_bits_15 >> 6 & 1);
                                        int warp_row_5 = tid / 32 * 32;
                                        #pragma unroll
                                        for (int chunk_22 = 0; chunk_22 < 16; chunk_22++) {
                                            float _tmem_load_16[16];
                                            tmem_ld_x16(&_tmem_load_16[0], taddr_1 + (unsigned int)(warp_row_5 << 16) + (unsigned int)(chunk_22 * 16));
                                            asm volatile("tcgen05.wait::ld.sync.aligned;");
                                            if (tid == 0) {
                                                asm volatile("cp.async.bulk.wait_group.read 2;");
                                            }
                                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                                            #pragma unroll
                                            for (int vec_9 = 0; vec_9 < 4; vec_9++) {
                                                unsigned int address_49 = d_smem_addr + (unsigned int)(chunk_22 % 3 * 8192) + (unsigned int)(tid * 64) + (unsigned int)(vec_9 * 16);
                                                address_49 = address_49 ^ (address_49 & 511) >> 7 << 4;
                                                asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(d_words_addr + (address_49 - d_words_addr)), "f"(_tmem_load_16[vec_9 * 4]), "f"(_tmem_load_16[vec_9 * 4 + 1]), "f"(_tmem_load_16[vec_9 * 4 + 2]), "f"(_tmem_load_16[vec_9 * 4 + 3]) : "memory");
                                            }
                                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                                            if (tid == 0) {
                                                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 900)
                                                #error "TmaReduceAdd5d requires SM90 or newer"
                                                #endif
                                                asm volatile(
                                                    "cp.reduce.async.bulk.tensor.5d.global.shared::cta.add.tile.bulk_group.L2::cache_hint"
                                                    " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                                    :: "l"((&dwd_r)), "r"(0), "r"(x_9 * 256 + cta_rank_0 * 128), "r"(y_9 * 16 + chunk_22), "r"(expert_9), "r"(0), "r"(d_smem_addr + (unsigned int)(chunk_22 % 3 * 8192)), "l"(0x12F0000000000000ULL) : "memory");
                                                asm volatile("cp.async.bulk.commit_group;");
                                            }
                                        }
                                        if (has_hi_9 != 0) {
                                            #pragma unroll
                                            for (int chunk_23 = 0; chunk_23 < 16; chunk_23++) {
                                                float _tmem_load_17[16];
                                                tmem_ld_x16(&_tmem_load_17[0], taddr_1 + (unsigned int)(warp_row_5 << 16) + 256 + (unsigned int)(chunk_23 * 16));
                                                asm volatile("tcgen05.wait::ld.sync.aligned;");
                                                if (tid == 0) {
                                                    asm volatile("cp.async.bulk.wait_group.read 2;");
                                                }
                                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                                #pragma unroll
                                                for (int vec_10 = 0; vec_10 < 4; vec_10++) {
                                                    unsigned int address_50 = d_smem_addr + (unsigned int)((16 + chunk_23) % 3 * 8192) + (unsigned int)(tid * 64) + (unsigned int)(vec_10 * 16);
                                                    address_50 = address_50 ^ (address_50 & 511) >> 7 << 4;
                                                    asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(d_words_addr + (address_50 - d_words_addr)), "f"(_tmem_load_17[vec_10 * 4]), "f"(_tmem_load_17[vec_10 * 4 + 1]), "f"(_tmem_load_17[vec_10 * 4 + 2]), "f"(_tmem_load_17[vec_10 * 4 + 3]) : "memory");
                                                }
                                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                                if (tid == 0) {
                                                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 900)
                                                    #error "TmaReduceAdd5d requires SM90 or newer"
                                                    #endif
                                                    asm volatile(
                                                        "cp.reduce.async.bulk.tensor.5d.global.shared::cta.add.tile.bulk_group.L2::cache_hint"
                                                        " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                                        :: "l"((&dwd_r)), "r"(0), "r"(x_9 * 256 + cta_rank_0 * 128), "r"((y_9 + 1) * 16 + chunk_23), "r"(expert_9), "r"(0), "r"(d_smem_addr + (unsigned int)((16 + chunk_23) % 3 * 8192)), "l"(0x12F0000000000000ULL) : "memory");
                                                    asm volatile("cp.async.bulk.commit_group;");
                                                }
                                            }
                                        }
                                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                                        if (tid == 0) {
                                            asm volatile(
                                                "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                                                :: "r"((output_finished_addr) & 0xFEFFFFFF) : "memory");
                                            asm volatile("cp.async.bulk.wait_group.read 0;");
                                        }
                                        asm volatile("barrier.sync 4, 128;" ::: "memory");
                                        phase_bits_15 = phase_bits_15 ^ 64;
                                        if (tid / 32 == 0) {
                                            if (warp == 0) {
                                                if (elect_sync()) {
                                                    bool enabled_value_28 = macros > 1;
                                                    if (enabled_value_28 != 0) {
                                                        asm volatile("cp.async.bulk.wait_group 0;");
                                                        bool enabled_value_0_3 = 1;
                                                        if (enabled_value_0_3 != 0) {
                                                            asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(buffers_done)) + (macro_1))), "r"(static_cast<unsigned int>(1)) : "memory");
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                            gemm_phase = phase_bits_15;
                        } else {
                            if (kind == 4) {
                                int col_blocks_16 = (macro_size + 256 - 1) / 256;
                                {
                                    col_blocks_16 = (hidden + 256 - 1) / 256;
                                }
                                int x_10 = -1;
                                int y_10 = -1;
                                int expert_10 = -1;
                                int k_start_10 = 0;
                                int k_end_10 = 0;
                                int first_10 = 0;
                                int row_blocks_6 = intermediate / 256;
                                int expert_idx_4 = 0;
                                int local_task_4 = task_3;
                                int _max_27 = ((row_blocks_6 * col_blocks_16) > (hidden / 256 * ((intermediate + 256 - 1) / 256)) ? (row_blocks_6 * col_blocks_16) : (hidden / 256 * ((intermediate + 256 - 1) / 256)));
                                int stride_4 = _max_27;
                                expert_idx_4 = task_3 / stride_4;
                                local_task_4 = task_3 % stride_4;
                                int offset_15 = counts[experts + expert_idx_4];
                                int real_1 = counts[2 * experts + expert_idx_4];
                                real_1 = (real_1 + 128 - 1) / 128 * 128;
                                int _max_28 = ((offset_15) > (macro_1 * macro_size) ? (offset_15) : (macro_1 * macro_size));
                                k_start_10 = _max_28;
                                int _min_66 = (((macro_1 + 1) * macro_size) < (tokens) ? ((macro_1 + 1) * macro_size) : (tokens));
                                int _min_67 = ((offset_15 + real_1) < (_min_66) ? (offset_15 + real_1) : (_min_66));
                                k_end_10 = _min_67;
                                first_10 = (int)(k_start_10 == offset_15);
                                if (k_start_10 < k_end_10 && local_task_4 < row_blocks_6 * col_blocks_16) {
                                    int supergroup_10 = local_task_4 / (row_blocks_6 * 8);
                                    int full_cols_10 = col_blocks_16 / 8 * 8;
                                    int row_41 = 0;
                                    int col_79 = 0;
                                    if (local_task_4 < row_blocks_6 * full_cols_10) {
                                        row_41 = local_task_4 % (row_blocks_6 * 8) / 8;
                                        col_79 = supergroup_10 * 8 + local_task_4 % 8;
                                    } else {
                                        row_41 = (local_task_4 - row_blocks_6 * full_cols_10) / (col_blocks_16 - full_cols_10);
                                        col_79 = full_cols_10 + (local_task_4 - row_blocks_6 * full_cols_10) % (col_blocks_16 - full_cols_10);
                                    }
                                    if ((supergroup_10 & 1) != 0) {
                                        row_41 = row_blocks_6 - row_41 - 1;
                                    }
                                    x_10 = row_41;
                                    y_10 = col_79;
                                    expert_10 = expert_idx_4;
                                }
                                unsigned int phase_bits_16 = gemm_phase;
                                int has_hi_10 = 0;
                                int global_mini_12 = macro_1 * (macro_size / mini_size);
                                int macro_rows_10 = macro_1 * (macro_size / 256);
                                int iterations_10 = intermediate / 128;
                                int macro_k_5 = macro_1 * (macro_size / 128);
                                iterations_10 = (k_end_10 - k_start_10) / 128;
                                if (expert_10 < 0) {
                                    if (tid == 0) {
                                        bool enabled_value_29 = macros > 1;
                                        if (enabled_value_29 != 0) {
                                            asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(buffers_done)) + (macro_1))), "r"(static_cast<unsigned int>(1)) : "memory");
                                        }
                                    }
                                } else if (tid / 32 == 7) {
                                    if (warp == 7) {
                                        if (elect_sync()) {
                                            unsigned int previous_15 = phase_bits_16 >> 7 & 1;
                                            unsigned int bits_15 = phase_bits_16;
                                            if (previous_15 != 1) {
                                                mbarrier_wait(gemm_finished_addr, bits_15 >> 16 & 1);
                                                mbarrier_wait(gemm_finished_addr + 8, bits_15 >> 17 & 1);
                                                mbarrier_wait(gemm_finished_addr + 16, bits_15 >> 18 & 1);
                                                mbarrier_wait(gemm_finished_addr + 24, bits_15 >> 19 & 1);
                                                mbarrier_wait(gemm_finished_addr + 32, bits_15 >> 20 & 1);
                                                mbarrier_wait(gemm_finished_addr + 40, bits_15 >> 21 & 1);
                                                bits_15 = bits_15 ^ 128;
                                            }
                                            phase_bits_16 = bits_15;
                                            int ring_25 = 0;
                                            #pragma unroll 1
                                            for (int idx_25 = 0; idx_25 < iterations_10; idx_25++) {
                                                int token_row_5 = k_start_10 + idx_25 * 128;
                                                if (idx_25 == 0 || token_row_5 % 256 == 0) {
                                                    bool enabled_value_30 = 1;
                                                    if (enabled_value_30 != 0) {
                                                        int32_t _relaxed_ld_72;
                                                        asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_72) : "l"(dg_ready + (shared_rows + token_row_5 / 256)) : "memory");
                                                        int value_28 = _relaxed_ld_72;
                                                        while (value_28 < row_count) {
                                                            asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                                            int32_t _relaxed_ld_73;
                                                            asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_73) : "l"(dg_ready + (shared_rows + token_row_5 / 256)) : "memory");
                                                            value_28 = _relaxed_ld_73;
                                                        }
                                                        asm volatile("fence.acquire.gpu;" ::: "memory");
                                                    }
                                                }
                                                if (idx_25 == 0 || token_row_5 % mini_size == 0) {
                                                    int input_mini_5 = token_row_5 / mini_size;
                                                    int _min_69 = ((mini_size) < (tokens - input_mini_5 * mini_size) ? (mini_size) : (tokens - input_mini_5 * mini_size));
                                                    int input_rows_5 = _min_69;
                                                    int input_count_5 = (input_rows_5 + 127) / 128 * ((hidden + 511) / 512);
                                                    bool enabled_value_31 = macro_1 > 0;
                                                    if (enabled_value_31 != 0) {
                                                        int32_t _relaxed_ld_74;
                                                        asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_74) : "l"(replay_x + input_mini_5) : "memory");
                                                        int value_29 = _relaxed_ld_74;
                                                        while (value_29 < input_count_5) {
                                                            asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                                            int32_t _relaxed_ld_75;
                                                            asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_75) : "l"(replay_x + input_mini_5) : "memory");
                                                            value_29 = _relaxed_ld_75;
                                                        }
                                                        asm volatile("fence.acquire.gpu;" ::: "memory");
                                                    }
                                                }
                                                mbarrier_wait(gemm_finished_addr + (ring_25) * 8, phase_bits_16 >> (unsigned int)(16 + ring_25) & 1);
                                                asm volatile(
                                                    "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                                    " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                                    :: "r"(smem_v52_addr + (unsigned int)(ring_25 * 16384)), "l"((&dg_t)), "r"(0), "r"(x_10 * 256 + cta_rank_0 * 128), "r"(k_start_10 / 128 + idx_25 - macro_k_5), "r"(0), "r"(0),
                                                       "r"(((gemm_arrived_addr + (ring_25) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                                asm volatile(
                                                    "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                                    " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                                    :: "r"(smem_v53_addr + (unsigned int)(ring_25 * 16384)), "l"((&x_t)), "r"(0), "r"(y_10 * 256 + cta_rank_0 * 128), "r"(k_start_10 / 128 + idx_25 - macro_k_5), "r"(0), "r"(0),
                                                       "r"(((gemm_arrived_addr + (ring_25) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                                phase_bits_16 = phase_bits_16 ^ (unsigned int)(1 << 16 + ring_25);
                                                ring_25 = (ring_25 + 1) % 6;
                                            }
                                        }
                                    }
                                } else {
                                    if (tid / 32 == 6) {
                                        if (warp == 6) {
                                            if (elect_sync()) {
                                                unsigned int previous_16 = phase_bits_16 >> 7 & 1;
                                                unsigned int bits_16 = phase_bits_16;
                                                if (previous_16 != 1) {
                                                    mbarrier_wait(scales_finished_addr, bits_16 >> 16 & 1);
                                                    mbarrier_wait(scales_finished_addr + 8, bits_16 >> 17 & 1);
                                                    mbarrier_wait(scales_finished_addr + 16, bits_16 >> 18 & 1);
                                                    mbarrier_wait(scales_finished_addr + 24, bits_16 >> 19 & 1);
                                                    mbarrier_wait(scales_finished_addr + 32, bits_16 >> 20 & 1);
                                                    mbarrier_wait(scales_finished_addr + 40, bits_16 >> 21 & 1);
                                                    bits_16 = bits_16 ^ 128;
                                                }
                                                phase_bits_16 = bits_16;
                                                int ring_26 = 0;
                                                #pragma unroll 1
                                                for (int idx_26 = 0; idx_26 < iterations_10; idx_26++) {
                                                    int token_row_6 = k_start_10 + idx_26 * 128;
                                                    if (idx_26 == 0 || token_row_6 % 256 == 0) {
                                                        bool enabled_value_32 = 1;
                                                        if (enabled_value_32 != 0) {
                                                            int32_t _relaxed_ld_80;
                                                            asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_80) : "l"(dg_ready + (shared_rows + token_row_6 / 256)) : "memory");
                                                            int value_30 = _relaxed_ld_80;
                                                            while (value_30 < row_count) {
                                                                asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                                                int32_t _relaxed_ld_81;
                                                                asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_81) : "l"(dg_ready + (shared_rows + token_row_6 / 256)) : "memory");
                                                                value_30 = _relaxed_ld_81;
                                                            }
                                                            asm volatile("fence.acquire.gpu;" ::: "memory");
                                                        }
                                                    }
                                                    if (idx_26 == 0 || token_row_6 % mini_size == 0) {
                                                        int input_mini_6 = token_row_6 / mini_size;
                                                        int _min_71 = ((mini_size) < (tokens - input_mini_6 * mini_size) ? (mini_size) : (tokens - input_mini_6 * mini_size));
                                                        int input_rows_6 = _min_71;
                                                        int input_count_6 = (input_rows_6 + 127) / 128 * ((hidden + 511) / 512);
                                                        bool enabled_value_33 = macro_1 > 0;
                                                        if (enabled_value_33 != 0) {
                                                            int32_t _relaxed_ld_82;
                                                            asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_82) : "l"(replay_x + input_mini_6) : "memory");
                                                            int value_31 = _relaxed_ld_82;
                                                            while (value_31 < input_count_6) {
                                                                asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                                                int32_t _relaxed_ld_83;
                                                                asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_83) : "l"(replay_x + input_mini_6) : "memory");
                                                                value_31 = _relaxed_ld_83;
                                                            }
                                                            asm volatile("fence.acquire.gpu;" ::: "memory");
                                                        }
                                                    }
                                                    mbarrier_wait(scales_finished_addr + (ring_26) * 8, phase_bits_16 >> (unsigned int)(16 + ring_26) & 1);
                                                    asm volatile(
                                                        "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                                        " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                                        :: "r"(smem_v54_addr + (unsigned int)(ring_26 * 512)), "l"((&dg_sc_t)), "r"(0), "r"(0), "r"((x_10 * 2 + cta_rank_0) * (macro_size / 128) + (k_start_10 / 128 + idx_26 - macro_k_5)),
                                                           "r"(((scales_arrived_addr + (ring_26) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                                    asm volatile(
                                                        "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                                        " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                                        :: "r"(smem_v55_addr + (unsigned int)(ring_26 * 1024) + (unsigned int)(cta_rank_0 * 512)), "l"((&x_sc_t)), "r"(0), "r"(0), "r"((y_10 * 2 + cta_rank_0) * (macro_size / 128) + (k_start_10 / 128 + idx_26 - macro_k_5)),
                                                           "r"(((scales_arrived_addr + (ring_26) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(3)) : "memory");
                                                    phase_bits_16 = phase_bits_16 ^ (unsigned int)(1 << 16 + ring_26);
                                                    ring_26 = (ring_26 + 1) % 6;
                                                }
                                            }
                                        }
                                    } else if (tid / 32 == 4 && cta_rank_0 == 0) {
                                        if (warp == 4) {
                                            if (elect_sync()) {
                                                int ring_27 = 0;
                                                mbarrier_wait(output_finished_addr, phase_bits_16 >> 22 & 1);
                                                phase_bits_16 = phase_bits_16 ^ 4194304;
                                                asm volatile("tcgen05.fence::after_thread_sync;");
                                                #pragma unroll 1
                                                for (int idx_27 = 0; idx_27 < iterations_10; idx_27++) {
                                                    mbarrier_arrive_expect_tx(scales_arrived_addr + (ring_27) * 8, 3072);
                                                    mbarrier_wait(scales_arrived_addr + (ring_27) * 8, phase_bits_16 >> (unsigned int)(8 + ring_27) & 1);
                                                    int buffer_5 = idx_27 % 3;
                                                    tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfa + buffer_5 * 4, make_sf_cp_desc_sbo128(smem_v54_addr + (unsigned int)(ring_27 * 512)));
                                                    tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfb + buffer_5 * 8, make_sf_cp_desc_sbo128(smem_v55_addr + (unsigned int)(ring_27 * 1024)));
                                                    tcgen05_cp_32x128b_warpx4_cta2((tmem_tmem_sfb + buffer_5 * 8 + 4), make_sf_cp_desc_sbo128((smem_v55_addr + (unsigned int)(ring_27 * 1024) + 512)));
                                                    tcgen05_commit_cg2_multicast(scales_finished_addr + (ring_27) * 8, (uint16_t)(3));
                                                    mbarrier_arrive_expect_tx(gemm_arrived_addr + (ring_27) * 8, 65536);
                                                    mbarrier_wait(gemm_arrived_addr + (ring_27) * 8, phase_bits_16 >> (unsigned int)ring_27 & 1);
                                                    int _mma_a_lo_15 = (((smem_v52_addr) >> 4) & 0x3FFF) + (ring_27) * 1024;
                                                    int _mma_b_lo_15 = (((smem_v53_addr) >> 4) & 0x3FFF) + (ring_27) * 1024;
                                                    {
                                                        uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_15) | ((uint64_t)0x40004040 << 32);
                                                        uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_15) | ((uint64_t)0x40004040 << 32);

                                                        tcgen05_mma_mxf8_bs_cta2(tmem_accumulator, a_desc + 0, b_desc + 0,
                                                            0x10c00000U, tmem_tmem_sfa + buffer_5 * 4, tmem_tmem_sfb + buffer_5 * 8, ((idx_27 == 0) ? 0 : 1));
                                                        tcgen05_mma_mxf8_bs_cta2(tmem_accumulator, a_desc + 2, b_desc + 2,
                                                            0x30c00010U, tmem_tmem_sfa + buffer_5 * 4, tmem_tmem_sfb + buffer_5 * 8, 1);
                                                        tcgen05_mma_mxf8_bs_cta2(tmem_accumulator, a_desc + 4, b_desc + 4,
                                                            0x50c00020U, tmem_tmem_sfa + buffer_5 * 4, tmem_tmem_sfb + buffer_5 * 8, 1);
                                                        tcgen05_mma_mxf8_bs_cta2(tmem_accumulator, a_desc + 6, b_desc + 6,
                                                            0x70c00030U, tmem_tmem_sfa + buffer_5 * 4, tmem_tmem_sfb + buffer_5 * 8, 1);
                                                    }
                                                    tcgen05_commit_cg2_multicast(gemm_finished_addr + (ring_27) * 8, (uint16_t)(3));
                                                    phase_bits_16 = phase_bits_16 ^ (unsigned int)(1 << ring_27) ^ (unsigned int)(1 << 8 + ring_27);
                                                    ring_27 = (ring_27 + 1) % 6;
                                                }
                                                tcgen05_commit_cg2_multicast(output_arrived_addr, (uint16_t)(3));
                                            }
                                        }
                                    } else {
                                        if (tid < 128) {
                                            mbarrier_wait(output_arrived_addr, phase_bits_16 >> 6 & 1);
                                            int warp_row_6 = tid / 32 * 32;
                                            #pragma unroll
                                            for (int chunk_24 = 0; chunk_24 < 16; chunk_24++) {
                                                float _tmem_load_18[16];
                                                tmem_ld_x16(&_tmem_load_18[0], taddr_1 + (unsigned int)(warp_row_6 << 16) + (unsigned int)(chunk_24 * 16));
                                                asm volatile("tcgen05.wait::ld.sync.aligned;");
                                                if (tid == 0) {
                                                    asm volatile("cp.async.bulk.wait_group.read 2;");
                                                }
                                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                                #pragma unroll
                                                for (int vec_11 = 0; vec_11 < 4; vec_11++) {
                                                    unsigned int address_51 = d_smem_addr + (unsigned int)(chunk_24 % 3 * 8192) + (unsigned int)(tid * 64) + (unsigned int)(vec_11 * 16);
                                                    address_51 = address_51 ^ (address_51 & 511) >> 7 << 4;
                                                    asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(d_words_addr + (address_51 - d_words_addr)), "f"(_tmem_load_18[vec_11 * 4]), "f"(_tmem_load_18[vec_11 * 4 + 1]), "f"(_tmem_load_18[vec_11 * 4 + 2]), "f"(_tmem_load_18[vec_11 * 4 + 3]) : "memory");
                                                }
                                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                                if (tid == 0) {
                                                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 900)
                                                    #error "TmaReduceAdd5d requires SM90 or newer"
                                                    #endif
                                                    asm volatile(
                                                        "cp.reduce.async.bulk.tensor.5d.global.shared::cta.add.tile.bulk_group.L2::cache_hint"
                                                        " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                                        :: "l"((&dwg_r)), "r"(0), "r"(x_10 * 256 + cta_rank_0 * 128), "r"(y_10 * 16 + chunk_24), "r"(expert_10), "r"(0), "r"(d_smem_addr + (unsigned int)(chunk_24 % 3 * 8192)), "l"(0x12F0000000000000ULL) : "memory");
                                                    asm volatile("cp.async.bulk.commit_group;");
                                                }
                                            }
                                            if (has_hi_10 != 0) {
                                                #pragma unroll
                                                for (int chunk_25 = 0; chunk_25 < 16; chunk_25++) {
                                                    float _tmem_load_19[16];
                                                    tmem_ld_x16(&_tmem_load_19[0], taddr_1 + (unsigned int)(warp_row_6 << 16) + 256 + (unsigned int)(chunk_25 * 16));
                                                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                                                    if (tid == 0) {
                                                        asm volatile("cp.async.bulk.wait_group.read 2;");
                                                    }
                                                    asm volatile("barrier.sync 1, 128;" ::: "memory");
                                                    #pragma unroll
                                                    for (int vec_12 = 0; vec_12 < 4; vec_12++) {
                                                        unsigned int address_52 = d_smem_addr + (unsigned int)((16 + chunk_25) % 3 * 8192) + (unsigned int)(tid * 64) + (unsigned int)(vec_12 * 16);
                                                        address_52 = address_52 ^ (address_52 & 511) >> 7 << 4;
                                                        asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(d_words_addr + (address_52 - d_words_addr)), "f"(_tmem_load_19[vec_12 * 4]), "f"(_tmem_load_19[vec_12 * 4 + 1]), "f"(_tmem_load_19[vec_12 * 4 + 2]), "f"(_tmem_load_19[vec_12 * 4 + 3]) : "memory");
                                                    }
                                                    asm volatile("barrier.sync 1, 128;" ::: "memory");
                                                    if (tid == 0) {
                                                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 900)
                                                        #error "TmaReduceAdd5d requires SM90 or newer"
                                                        #endif
                                                        asm volatile(
                                                            "cp.reduce.async.bulk.tensor.5d.global.shared::cta.add.tile.bulk_group.L2::cache_hint"
                                                            " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                                            :: "l"((&dwg_r)), "r"(0), "r"(x_10 * 256 + cta_rank_0 * 128), "r"((y_10 + 1) * 16 + chunk_25), "r"(expert_10), "r"(0), "r"(d_smem_addr + (unsigned int)((16 + chunk_25) % 3 * 8192)), "l"(0x12F0000000000000ULL) : "memory");
                                                        asm volatile("cp.async.bulk.commit_group;");
                                                    }
                                                }
                                            }
                                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                                            if (tid == 0) {
                                                asm volatile(
                                                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                                                    :: "r"((output_finished_addr) & 0xFEFFFFFF) : "memory");
                                                asm volatile("cp.async.bulk.wait_group.read 0;");
                                            }
                                            asm volatile("barrier.sync 4, 128;" ::: "memory");
                                            phase_bits_16 = phase_bits_16 ^ 64;
                                            if (tid / 32 == 0) {
                                                if (warp == 0) {
                                                    if (elect_sync()) {
                                                        bool enabled_value_34 = macros > 1;
                                                        if (enabled_value_34 != 0) {
                                                            asm volatile("cp.async.bulk.wait_group 0;");
                                                            bool enabled_value_0_4 = 1;
                                                            if (enabled_value_0_4 != 0) {
                                                                asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(buffers_done)) + (macro_1))), "r"(static_cast<unsigned int>(1)) : "memory");
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                                gemm_phase = phase_bits_16;
                            } else if (kind == 5) {
                                int col_blocks_17 = (macro_size + 256 - 1) / 256;
                                {
                                    col_blocks_17 = (hidden + 256 - 1) / 256;
                                }
                                int x_11 = -1;
                                int y_11 = -1;
                                int expert_11 = -1;
                                int k_start_11 = 0;
                                int k_end_11 = 0;
                                int first_11 = 0;
                                int row_blocks_7 = intermediate / 256;
                                int expert_idx_5 = 0;
                                int local_task_5 = task_3;
                                int _max_31 = ((row_blocks_7 * col_blocks_17) > (hidden / 256 * ((intermediate + 256 - 1) / 256)) ? (row_blocks_7 * col_blocks_17) : (hidden / 256 * ((intermediate + 256 - 1) / 256)));
                                int stride_5 = _max_31;
                                expert_idx_5 = task_3 / stride_5;
                                local_task_5 = task_3 % stride_5;
                                int offset_16 = counts[experts + expert_idx_5];
                                int real_2 = counts[2 * experts + expert_idx_5];
                                real_2 = (real_2 + 128 - 1) / 128 * 128;
                                int _max_32 = ((offset_16) > (macro_1 * macro_size) ? (offset_16) : (macro_1 * macro_size));
                                k_start_11 = _max_32;
                                int _min_72 = (((macro_1 + 1) * macro_size) < (tokens) ? ((macro_1 + 1) * macro_size) : (tokens));
                                int _min_73 = ((offset_16 + real_2) < (_min_72) ? (offset_16 + real_2) : (_min_72));
                                k_end_11 = _min_73;
                                first_11 = (int)(k_start_11 == offset_16);
                                if (k_start_11 < k_end_11 && local_task_5 < row_blocks_7 * col_blocks_17) {
                                    int supergroup_11 = local_task_5 / (row_blocks_7 * 8);
                                    int full_cols_11 = col_blocks_17 / 8 * 8;
                                    int row_42 = 0;
                                    int col_80 = 0;
                                    if (local_task_5 < row_blocks_7 * full_cols_11) {
                                        row_42 = local_task_5 % (row_blocks_7 * 8) / 8;
                                        col_80 = supergroup_11 * 8 + local_task_5 % 8;
                                    } else {
                                        row_42 = (local_task_5 - row_blocks_7 * full_cols_11) / (col_blocks_17 - full_cols_11);
                                        col_80 = full_cols_11 + (local_task_5 - row_blocks_7 * full_cols_11) % (col_blocks_17 - full_cols_11);
                                    }
                                    if ((supergroup_11 & 1) != 0) {
                                        row_42 = row_blocks_7 - row_42 - 1;
                                    }
                                    x_11 = row_42;
                                    y_11 = col_80;
                                    expert_11 = expert_idx_5;
                                }
                                unsigned int phase_bits_17 = gemm_phase;
                                int has_hi_11 = 0;
                                int global_mini_13 = macro_1 * (macro_size / mini_size);
                                int macro_rows_11 = macro_1 * (macro_size / 256);
                                int iterations_11 = intermediate / 128;
                                int macro_k_6 = macro_1 * (macro_size / 128);
                                iterations_11 = (k_end_11 - k_start_11) / 128;
                                if (expert_11 < 0) {
                                    if (tid == 0) {
                                        bool enabled_value_35 = macros > 1;
                                        if (enabled_value_35 != 0) {
                                            asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(buffers_done)) + (macro_1))), "r"(static_cast<unsigned int>(1)) : "memory");
                                        }
                                    }
                                } else if (tid / 32 == 7) {
                                    if (warp == 7) {
                                        if (elect_sync()) {
                                            unsigned int previous_17 = phase_bits_17 >> 7 & 1;
                                            unsigned int bits_17 = phase_bits_17;
                                            if (previous_17 != 1) {
                                                mbarrier_wait(gemm_finished_addr, bits_17 >> 16 & 1);
                                                mbarrier_wait(gemm_finished_addr + 8, bits_17 >> 17 & 1);
                                                mbarrier_wait(gemm_finished_addr + 16, bits_17 >> 18 & 1);
                                                mbarrier_wait(gemm_finished_addr + 24, bits_17 >> 19 & 1);
                                                mbarrier_wait(gemm_finished_addr + 32, bits_17 >> 20 & 1);
                                                mbarrier_wait(gemm_finished_addr + 40, bits_17 >> 21 & 1);
                                                bits_17 = bits_17 ^ 128;
                                            }
                                            phase_bits_17 = bits_17;
                                            int ring_28 = 0;
                                            #pragma unroll 1
                                            for (int idx_28 = 0; idx_28 < iterations_11; idx_28++) {
                                                int token_row_7 = k_start_11 + idx_28 * 128;
                                                if (idx_28 == 0 || token_row_7 % 256 == 0) {
                                                    bool enabled_value_36 = 1;
                                                    if (enabled_value_36 != 0) {
                                                        int32_t _relaxed_ld_88;
                                                        asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_88) : "l"(dg_ready + (shared_rows + token_row_7 / 256)) : "memory");
                                                        int value_32 = _relaxed_ld_88;
                                                        while (value_32 < row_count) {
                                                            asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                                            int32_t _relaxed_ld_89;
                                                            asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_89) : "l"(dg_ready + (shared_rows + token_row_7 / 256)) : "memory");
                                                            value_32 = _relaxed_ld_89;
                                                        }
                                                        asm volatile("fence.acquire.gpu;" ::: "memory");
                                                    }
                                                }
                                                if (idx_28 == 0 || token_row_7 % mini_size == 0) {
                                                    int input_mini_7 = token_row_7 / mini_size;
                                                    int _min_75 = ((mini_size) < (tokens - input_mini_7 * mini_size) ? (mini_size) : (tokens - input_mini_7 * mini_size));
                                                    int input_rows_7 = _min_75;
                                                    int input_count_7 = (input_rows_7 + 127) / 128 * ((hidden + 511) / 512);
                                                    bool enabled_value_37 = macro_1 > 0;
                                                    if (enabled_value_37 != 0) {
                                                        int32_t _relaxed_ld_90;
                                                        asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_90) : "l"(replay_x + input_mini_7) : "memory");
                                                        int value_33 = _relaxed_ld_90;
                                                        while (value_33 < input_count_7) {
                                                            asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                                            int32_t _relaxed_ld_91;
                                                            asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_91) : "l"(replay_x + input_mini_7) : "memory");
                                                            value_33 = _relaxed_ld_91;
                                                        }
                                                        asm volatile("fence.acquire.gpu;" ::: "memory");
                                                    }
                                                }
                                                mbarrier_wait(gemm_finished_addr + (ring_28) * 8, phase_bits_17 >> (unsigned int)(16 + ring_28) & 1);
                                                asm volatile(
                                                    "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                                    " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                                    :: "r"(smem_v52_addr + (unsigned int)(ring_28 * 16384)), "l"((&du_t)), "r"(0), "r"(x_11 * 256 + cta_rank_0 * 128), "r"(k_start_11 / 128 + idx_28 - macro_k_6), "r"(0), "r"(0),
                                                       "r"(((gemm_arrived_addr + (ring_28) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                                asm volatile(
                                                    "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                                    " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                                    :: "r"(smem_v53_addr + (unsigned int)(ring_28 * 16384)), "l"((&x_t)), "r"(0), "r"(y_11 * 256 + cta_rank_0 * 128), "r"(k_start_11 / 128 + idx_28 - macro_k_6), "r"(0), "r"(0),
                                                       "r"(((gemm_arrived_addr + (ring_28) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                                phase_bits_17 = phase_bits_17 ^ (unsigned int)(1 << 16 + ring_28);
                                                ring_28 = (ring_28 + 1) % 6;
                                            }
                                        }
                                    }
                                } else {
                                    if (tid / 32 == 6) {
                                        if (warp == 6) {
                                            if (elect_sync()) {
                                                unsigned int previous_18 = phase_bits_17 >> 7 & 1;
                                                unsigned int bits_18 = phase_bits_17;
                                                if (previous_18 != 1) {
                                                    mbarrier_wait(scales_finished_addr, bits_18 >> 16 & 1);
                                                    mbarrier_wait(scales_finished_addr + 8, bits_18 >> 17 & 1);
                                                    mbarrier_wait(scales_finished_addr + 16, bits_18 >> 18 & 1);
                                                    mbarrier_wait(scales_finished_addr + 24, bits_18 >> 19 & 1);
                                                    mbarrier_wait(scales_finished_addr + 32, bits_18 >> 20 & 1);
                                                    mbarrier_wait(scales_finished_addr + 40, bits_18 >> 21 & 1);
                                                    bits_18 = bits_18 ^ 128;
                                                }
                                                phase_bits_17 = bits_18;
                                                int ring_29 = 0;
                                                #pragma unroll 1
                                                for (int idx_29 = 0; idx_29 < iterations_11; idx_29++) {
                                                    int token_row_8 = k_start_11 + idx_29 * 128;
                                                    if (idx_29 == 0 || token_row_8 % 256 == 0) {
                                                        bool enabled_value_38 = 1;
                                                        if (enabled_value_38 != 0) {
                                                            int32_t _relaxed_ld_96;
                                                            asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_96) : "l"(dg_ready + (shared_rows + token_row_8 / 256)) : "memory");
                                                            int value_34 = _relaxed_ld_96;
                                                            while (value_34 < row_count) {
                                                                asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                                                int32_t _relaxed_ld_97;
                                                                asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_97) : "l"(dg_ready + (shared_rows + token_row_8 / 256)) : "memory");
                                                                value_34 = _relaxed_ld_97;
                                                            }
                                                            asm volatile("fence.acquire.gpu;" ::: "memory");
                                                        }
                                                    }
                                                    if (idx_29 == 0 || token_row_8 % mini_size == 0) {
                                                        int input_mini_8 = token_row_8 / mini_size;
                                                        int _min_77 = ((mini_size) < (tokens - input_mini_8 * mini_size) ? (mini_size) : (tokens - input_mini_8 * mini_size));
                                                        int input_rows_8 = _min_77;
                                                        int input_count_8 = (input_rows_8 + 127) / 128 * ((hidden + 511) / 512);
                                                        bool enabled_value_39 = macro_1 > 0;
                                                        if (enabled_value_39 != 0) {
                                                            int32_t _relaxed_ld_98;
                                                            asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_98) : "l"(replay_x + input_mini_8) : "memory");
                                                            int value_35 = _relaxed_ld_98;
                                                            while (value_35 < input_count_8) {
                                                                asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                                                int32_t _relaxed_ld_99;
                                                                asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_99) : "l"(replay_x + input_mini_8) : "memory");
                                                                value_35 = _relaxed_ld_99;
                                                            }
                                                            asm volatile("fence.acquire.gpu;" ::: "memory");
                                                        }
                                                    }
                                                    mbarrier_wait(scales_finished_addr + (ring_29) * 8, phase_bits_17 >> (unsigned int)(16 + ring_29) & 1);
                                                    asm volatile(
                                                        "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                                        " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                                        :: "r"(smem_v54_addr + (unsigned int)(ring_29 * 512)), "l"((&du_sc_t)), "r"(0), "r"(0), "r"((x_11 * 2 + cta_rank_0) * (macro_size / 128) + (k_start_11 / 128 + idx_29 - macro_k_6)),
                                                           "r"(((scales_arrived_addr + (ring_29) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                                    asm volatile(
                                                        "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                                        " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                                        :: "r"(smem_v55_addr + (unsigned int)(ring_29 * 1024) + (unsigned int)(cta_rank_0 * 512)), "l"((&x_sc_t)), "r"(0), "r"(0), "r"((y_11 * 2 + cta_rank_0) * (macro_size / 128) + (k_start_11 / 128 + idx_29 - macro_k_6)),
                                                           "r"(((scales_arrived_addr + (ring_29) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(3)) : "memory");
                                                    phase_bits_17 = phase_bits_17 ^ (unsigned int)(1 << 16 + ring_29);
                                                    ring_29 = (ring_29 + 1) % 6;
                                                }
                                            }
                                        }
                                    } else if (tid / 32 == 4 && cta_rank_0 == 0) {
                                        if (warp == 4) {
                                            if (elect_sync()) {
                                                int ring_30 = 0;
                                                mbarrier_wait(output_finished_addr, phase_bits_17 >> 22 & 1);
                                                phase_bits_17 = phase_bits_17 ^ 4194304;
                                                asm volatile("tcgen05.fence::after_thread_sync;");
                                                #pragma unroll 1
                                                for (int idx_30 = 0; idx_30 < iterations_11; idx_30++) {
                                                    mbarrier_arrive_expect_tx(scales_arrived_addr + (ring_30) * 8, 3072);
                                                    mbarrier_wait(scales_arrived_addr + (ring_30) * 8, phase_bits_17 >> (unsigned int)(8 + ring_30) & 1);
                                                    int buffer_6 = idx_30 % 3;
                                                    tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfa + buffer_6 * 4, make_sf_cp_desc_sbo128(smem_v54_addr + (unsigned int)(ring_30 * 512)));
                                                    tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfb + buffer_6 * 8, make_sf_cp_desc_sbo128(smem_v55_addr + (unsigned int)(ring_30 * 1024)));
                                                    tcgen05_cp_32x128b_warpx4_cta2((tmem_tmem_sfb + buffer_6 * 8 + 4), make_sf_cp_desc_sbo128((smem_v55_addr + (unsigned int)(ring_30 * 1024) + 512)));
                                                    tcgen05_commit_cg2_multicast(scales_finished_addr + (ring_30) * 8, (uint16_t)(3));
                                                    mbarrier_arrive_expect_tx(gemm_arrived_addr + (ring_30) * 8, 65536);
                                                    mbarrier_wait(gemm_arrived_addr + (ring_30) * 8, phase_bits_17 >> (unsigned int)ring_30 & 1);
                                                    int _mma_a_lo_16 = (((smem_v52_addr) >> 4) & 0x3FFF) + (ring_30) * 1024;
                                                    int _mma_b_lo_16 = (((smem_v53_addr) >> 4) & 0x3FFF) + (ring_30) * 1024;
                                                    {
                                                        uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_16) | ((uint64_t)0x40004040 << 32);
                                                        uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_16) | ((uint64_t)0x40004040 << 32);

                                                        tcgen05_mma_mxf8_bs_cta2(tmem_accumulator, a_desc + 0, b_desc + 0,
                                                            0x10c00000U, tmem_tmem_sfa + buffer_6 * 4, tmem_tmem_sfb + buffer_6 * 8, ((idx_30 == 0) ? 0 : 1));
                                                        tcgen05_mma_mxf8_bs_cta2(tmem_accumulator, a_desc + 2, b_desc + 2,
                                                            0x30c00010U, tmem_tmem_sfa + buffer_6 * 4, tmem_tmem_sfb + buffer_6 * 8, 1);
                                                        tcgen05_mma_mxf8_bs_cta2(tmem_accumulator, a_desc + 4, b_desc + 4,
                                                            0x50c00020U, tmem_tmem_sfa + buffer_6 * 4, tmem_tmem_sfb + buffer_6 * 8, 1);
                                                        tcgen05_mma_mxf8_bs_cta2(tmem_accumulator, a_desc + 6, b_desc + 6,
                                                            0x70c00030U, tmem_tmem_sfa + buffer_6 * 4, tmem_tmem_sfb + buffer_6 * 8, 1);
                                                    }
                                                    tcgen05_commit_cg2_multicast(gemm_finished_addr + (ring_30) * 8, (uint16_t)(3));
                                                    phase_bits_17 = phase_bits_17 ^ (unsigned int)(1 << ring_30) ^ (unsigned int)(1 << 8 + ring_30);
                                                    ring_30 = (ring_30 + 1) % 6;
                                                }
                                                tcgen05_commit_cg2_multicast(output_arrived_addr, (uint16_t)(3));
                                            }
                                        }
                                    } else {
                                        if (tid < 128) {
                                            mbarrier_wait(output_arrived_addr, phase_bits_17 >> 6 & 1);
                                            int warp_row_7 = tid / 32 * 32;
                                            #pragma unroll
                                            for (int chunk_26 = 0; chunk_26 < 16; chunk_26++) {
                                                float _tmem_load_20[16];
                                                tmem_ld_x16(&_tmem_load_20[0], taddr_1 + (unsigned int)(warp_row_7 << 16) + (unsigned int)(chunk_26 * 16));
                                                asm volatile("tcgen05.wait::ld.sync.aligned;");
                                                if (tid == 0) {
                                                    asm volatile("cp.async.bulk.wait_group.read 2;");
                                                }
                                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                                #pragma unroll
                                                for (int vec_13 = 0; vec_13 < 4; vec_13++) {
                                                    unsigned int address_53 = d_smem_addr + (unsigned int)(chunk_26 % 3 * 8192) + (unsigned int)(tid * 64) + (unsigned int)(vec_13 * 16);
                                                    address_53 = address_53 ^ (address_53 & 511) >> 7 << 4;
                                                    asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(d_words_addr + (address_53 - d_words_addr)), "f"(_tmem_load_20[vec_13 * 4]), "f"(_tmem_load_20[vec_13 * 4 + 1]), "f"(_tmem_load_20[vec_13 * 4 + 2]), "f"(_tmem_load_20[vec_13 * 4 + 3]) : "memory");
                                                }
                                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                                if (tid == 0) {
                                                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 900)
                                                    #error "TmaReduceAdd5d requires SM90 or newer"
                                                    #endif
                                                    asm volatile(
                                                        "cp.reduce.async.bulk.tensor.5d.global.shared::cta.add.tile.bulk_group.L2::cache_hint"
                                                        " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                                        :: "l"((&dwu_r)), "r"(0), "r"(x_11 * 256 + cta_rank_0 * 128), "r"(y_11 * 16 + chunk_26), "r"(expert_11), "r"(0), "r"(d_smem_addr + (unsigned int)(chunk_26 % 3 * 8192)), "l"(0x12F0000000000000ULL) : "memory");
                                                    asm volatile("cp.async.bulk.commit_group;");
                                                }
                                            }
                                            if (has_hi_11 != 0) {
                                                #pragma unroll
                                                for (int chunk_27 = 0; chunk_27 < 16; chunk_27++) {
                                                    float _tmem_load_21[16];
                                                    tmem_ld_x16(&_tmem_load_21[0], taddr_1 + (unsigned int)(warp_row_7 << 16) + 256 + (unsigned int)(chunk_27 * 16));
                                                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                                                    if (tid == 0) {
                                                        asm volatile("cp.async.bulk.wait_group.read 2;");
                                                    }
                                                    asm volatile("barrier.sync 1, 128;" ::: "memory");
                                                    #pragma unroll
                                                    for (int vec_14 = 0; vec_14 < 4; vec_14++) {
                                                        unsigned int address_54 = d_smem_addr + (unsigned int)((16 + chunk_27) % 3 * 8192) + (unsigned int)(tid * 64) + (unsigned int)(vec_14 * 16);
                                                        address_54 = address_54 ^ (address_54 & 511) >> 7 << 4;
                                                        asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(d_words_addr + (address_54 - d_words_addr)), "f"(_tmem_load_21[vec_14 * 4]), "f"(_tmem_load_21[vec_14 * 4 + 1]), "f"(_tmem_load_21[vec_14 * 4 + 2]), "f"(_tmem_load_21[vec_14 * 4 + 3]) : "memory");
                                                    }
                                                    asm volatile("barrier.sync 1, 128;" ::: "memory");
                                                    if (tid == 0) {
                                                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 900)
                                                        #error "TmaReduceAdd5d requires SM90 or newer"
                                                        #endif
                                                        asm volatile(
                                                            "cp.reduce.async.bulk.tensor.5d.global.shared::cta.add.tile.bulk_group.L2::cache_hint"
                                                            " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                                            :: "l"((&dwu_r)), "r"(0), "r"(x_11 * 256 + cta_rank_0 * 128), "r"((y_11 + 1) * 16 + chunk_27), "r"(expert_11), "r"(0), "r"(d_smem_addr + (unsigned int)((16 + chunk_27) % 3 * 8192)), "l"(0x12F0000000000000ULL) : "memory");
                                                        asm volatile("cp.async.bulk.commit_group;");
                                                    }
                                                }
                                            }
                                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                                            if (tid == 0) {
                                                asm volatile(
                                                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                                                    :: "r"((output_finished_addr) & 0xFEFFFFFF) : "memory");
                                                asm volatile("cp.async.bulk.wait_group.read 0;");
                                            }
                                            asm volatile("barrier.sync 4, 128;" ::: "memory");
                                            phase_bits_17 = phase_bits_17 ^ 64;
                                            if (tid / 32 == 0) {
                                                if (warp == 0) {
                                                    if (elect_sync()) {
                                                        bool enabled_value_40 = macros > 1;
                                                        if (enabled_value_40 != 0) {
                                                            asm volatile("cp.async.bulk.wait_group 0;");
                                                            bool enabled_value_0_5 = 1;
                                                            if (enabled_value_0_5 != 0) {
                                                                asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(buffers_done)) + (macro_1))), "r"(static_cast<unsigned int>(1)) : "memory");
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                                gemm_phase = phase_bits_17;
                            }
                        }
                    }
                }
            }
            gemm_bits = gemm_phase;
            swiglu_bits = swiglu_phase;
            replay_bits = replay_phase;
            mbarrier_wait(schedule_arrived_addr, iteration % 2);
            uint32_t _clc_valid_0 = 0;
            uint32_t _clc_ctaid_x_0;
            uint32_t _clc_ctaid_y_0;
            uint32_t _clc_ctaid_z_0;
            asm volatile(
                "{\n\t"
                ".reg .pred p1;\n\t"
                ".reg .b128 clc_r;\n\t"
                "ld.shared.b128 clc_r, [%4];\n\t"
                "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                "selp.u32 %3, 1, 0, p1;\n\t"
                "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid.v4.b32.b128 {%0, %1, %2, _}, clc_r;\n\t"
                "}\n"
                : "=r"(_clc_ctaid_x_0), "=r"(_clc_ctaid_y_0), "=r"(_clc_ctaid_z_0), "=r"(_clc_valid_0)
                : "r"(smem + 512 + 0 * 16 + 0 * 16)
                : "memory");
            cluster = -1;
            if (_clc_valid_0 != 0) {
                cluster = (int)(_clc_ctaid_x_0 / 2);
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            }
            __syncwarp();
            bool _elect_sync_0 = elect_sync();
            if (_elect_sync_0) {
                asm volatile(
                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                    :: "r"((schedule_finished_addr) & 0xFEFFFFFF) : "memory");
            }
            int shared_tasks_6 = shared_down + shared_swiglu + shared_dx + 3 * shared_wgrad;
            int mini_bwd_7 = mini_down + mini_swiglu + mini_dx;
            int mini_replay_8 = 2 * mini_replay_gate + mini_replay_swiglu;
            int weight_tasks_9 = experts * routed_wgrad;
            int _min_78 = ((tokens) < (macro_size) ? (tokens) : (macro_size));
            int saved_minis_10 = (_min_78 + mini_size - 1) / mini_size;
            int saved_tasks_11 = saved_minis_10 * mini_bwd_7 + 3 * weight_tasks_9;
            int replay_macro_tasks_12 = macro_size / mini_size * (mini_replay_8 + mini_bwd_7) + 3 * weight_tasks_9;
            int kind_13 = -1;
            int task_14 = 0;
            int macro_15 = 0;
            int mini_16 = 0;
            int shared_17 = 0;
            if (cluster - comm_clusters >= 0 && true_compute > cluster - comm_clusters) {
                if (shared_tasks_6 > cluster - comm_clusters) {
                    shared_17 = 1;
                    if (shared_down > cluster - comm_clusters) {
                        kind_13 = 0;
                        task_14 = cluster - comm_clusters;
                    } else if (cluster - comm_clusters < shared_down + shared_swiglu) {
                        kind_13 = 1;
                        task_14 = cluster - comm_clusters - shared_down;
                    } else {
                        if (cluster - comm_clusters < shared_down + shared_swiglu + shared_dx) {
                            kind_13 = 2;
                            task_14 = cluster - comm_clusters - shared_down - shared_swiglu;
                        } else {
                            int weight_task_2 = cluster - comm_clusters - shared_down - shared_swiglu - shared_dx;
                            kind_13 = 3 + weight_task_2 / shared_wgrad;
                            task_14 = weight_task_2 % shared_wgrad;
                        }
                    }
                } else {
                    int routed_1 = cluster - comm_clusters - shared_tasks_6;
                    int macro_task_1 = routed_1;
                    int replay_tasks_1 = 0;
                    if (routed_1 >= saved_tasks_11) {
                        macro_15 = 1 + (routed_1 - saved_tasks_11) / replay_macro_tasks_12;
                        macro_task_1 = (routed_1 - saved_tasks_11) % replay_macro_tasks_12;
                        int _min_79 = ((tokens - macro_15 * macro_size) < (macro_size) ? (tokens - macro_15 * macro_size) : (macro_size));
                        int macro_minis_2 = (_min_79 + mini_size - 1) / mini_size;
                        replay_tasks_1 = macro_minis_2 * mini_replay_8;
                    }
                    int _min_80 = ((tokens - macro_15 * macro_size) < (macro_size) ? (tokens - macro_15 * macro_size) : (macro_size));
                    int macro_minis_3 = (_min_80 + mini_size - 1) / mini_size;
                    if (macro_task_1 < replay_tasks_1) {
                        mini_16 = macro_task_1 / mini_replay_8;
                        int mini_task_2 = macro_task_1 % mini_replay_8;
                        if (mini_task_2 < mini_replay_gate) {
                            kind_13 = 6;
                            task_14 = mini_task_2;
                        } else if (mini_task_2 < 2 * mini_replay_gate) {
                            kind_13 = 7;
                            task_14 = mini_task_2 - mini_replay_gate;
                        } else {
                            kind_13 = 8;
                            task_14 = mini_task_2 - 2 * mini_replay_gate;
                        }
                    } else {
                        int bwd_task_1 = macro_task_1 - replay_tasks_1;
                        if (bwd_task_1 < macro_minis_3 * mini_bwd_7) {
                            mini_16 = bwd_task_1 / mini_bwd_7;
                            int mini_task_3 = bwd_task_1 % mini_bwd_7;
                            if (mini_task_3 < mini_down) {
                                kind_13 = 0;
                                task_14 = mini_task_3;
                            } else if (mini_task_3 < mini_down + mini_swiglu) {
                                kind_13 = 1;
                                task_14 = mini_task_3 - mini_down;
                            } else {
                                kind_13 = 2;
                                task_14 = mini_task_3 - mini_down - mini_swiglu;
                            }
                        } else {
                            int weight_task_3 = bwd_task_1 - macro_minis_3 * mini_bwd_7;
                            kind_13 = 3 + weight_task_3 / weight_tasks_9;
                            task_14 = weight_task_3 % weight_tasks_9;
                        }
                    }
                }
            }
            if ((kind == 1 || kind == 8) && cluster >= 0 && kind_13 != 1 && kind_13 != 8) {
                asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
                asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
            }
            iteration = iteration + 1;
        }
        asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
        asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
        if (cluster >= 0 && tid == 0) {
            int iteration_0 = 0;
            if (cta_rank_0 == 0) {
                asm volatile(
                    "clusterlaunchcontrol.try_cancel.async.shared::cta"
                        ".mbarrier::complete_tx::bytes.multicast::cluster::all.b128"
                        " [%0], [%1];"
                    :: "r"(smem + 528 + 0 * 16 + 0 * 16), "r"(drain_arrived_0_addr + 0 * 8)
                    : "memory");
            }
            mbarrier_arrive_expect_tx(drain_arrived_0_addr, 16);
            while (1) {
                mbarrier_wait(drain_arrived_0_addr, iteration_0 % 2);
                uint32_t _clc_valid_1 = 0;
                uint32_t _clc_ctaid_x_1;
                uint32_t _clc_ctaid_y_1;
                uint32_t _clc_ctaid_z_1;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%4];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "selp.u32 %3, 1, 0, p1;\n\t"
                    "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid.v4.b32.b128 {%0, %1, %2, _}, clc_r;\n\t"
                    "}\n"
                    : "=r"(_clc_ctaid_x_1), "=r"(_clc_ctaid_y_1), "=r"(_clc_ctaid_z_1), "=r"(_clc_valid_1)
                    : "r"(smem + 528 + 0 * 16 + 0 * 16)
                    : "memory");
                if (_clc_valid_1 != 0) {
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                }
                asm volatile(
                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                    :: "r"((drain_finished_0_addr) & 0xFEFFFFFF) : "memory");
                if (cta_rank_0 == 0) {
                    mbarrier_wait(drain_finished_0_addr, iteration_0 % 2);
                }
                if (_clc_valid_1 == 0) {
                    break;
                }
                if (cta_rank_0 == 0) {
                    asm volatile(
                        "clusterlaunchcontrol.try_cancel.async.shared::cta"
                            ".mbarrier::complete_tx::bytes.multicast::cluster::all.b128"
                            " [%0], [%1];"
                        :: "r"(smem + 528 + 0 * 16 + 0 * 16), "r"(drain_arrived_0_addr + 0 * 8)
                        : "memory");
                }
                mbarrier_arrive_expect_tx(drain_arrived_0_addr, 16);
                iteration_0 = iteration_0 + 1;
            }
        }
        if (cluster >= 0 && tid == 32) {
            int iteration_0_1 = 0;
            if (cta_rank_0 == 0) {
                asm volatile(
                    "clusterlaunchcontrol.try_cancel.async.shared::cta"
                        ".mbarrier::complete_tx::bytes.multicast::cluster::all.b128"
                        " [%0], [%1];"
                    :: "r"(smem + 544 + 0 * 16 + 0 * 16), "r"(drain_arrived_1_addr + 0 * 8)
                    : "memory");
            }
            mbarrier_arrive_expect_tx(drain_arrived_1_addr, 16);
            while (1) {
                mbarrier_wait(drain_arrived_1_addr, iteration_0_1 % 2);
                uint32_t _clc_valid_2 = 0;
                uint32_t _clc_ctaid_x_2;
                uint32_t _clc_ctaid_y_2;
                uint32_t _clc_ctaid_z_2;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%4];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "selp.u32 %3, 1, 0, p1;\n\t"
                    "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid.v4.b32.b128 {%0, %1, %2, _}, clc_r;\n\t"
                    "}\n"
                    : "=r"(_clc_ctaid_x_2), "=r"(_clc_ctaid_y_2), "=r"(_clc_ctaid_z_2), "=r"(_clc_valid_2)
                    : "r"(smem + 544 + 0 * 16 + 0 * 16)
                    : "memory");
                if (_clc_valid_2 != 0) {
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                }
                asm volatile(
                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                    :: "r"((drain_finished_1_addr) & 0xFEFFFFFF) : "memory");
                if (cta_rank_0 == 0) {
                    mbarrier_wait(drain_finished_1_addr, iteration_0_1 % 2);
                }
                if (_clc_valid_2 == 0) {
                    break;
                }
                if (cta_rank_0 == 0) {
                    asm volatile(
                        "clusterlaunchcontrol.try_cancel.async.shared::cta"
                            ".mbarrier::complete_tx::bytes.multicast::cluster::all.b128"
                            " [%0], [%1];"
                        :: "r"(smem + 544 + 0 * 16 + 0 * 16), "r"(drain_arrived_1_addr + 0 * 8)
                        : "memory");
                }
                mbarrier_arrive_expect_tx(drain_arrived_1_addr, 16);
                iteration_0_1 = iteration_0_1 + 1;
            }
        }
        if (cluster >= 0 && tid == 64) {
            int iteration_0_2 = 0;
            if (cta_rank_0 == 0) {
                asm volatile(
                    "clusterlaunchcontrol.try_cancel.async.shared::cta"
                        ".mbarrier::complete_tx::bytes.multicast::cluster::all.b128"
                        " [%0], [%1];"
                    :: "r"(smem + 560 + 0 * 16 + 0 * 16), "r"(drain_arrived_2_addr + 0 * 8)
                    : "memory");
            }
            mbarrier_arrive_expect_tx(drain_arrived_2_addr, 16);
            while (1) {
                mbarrier_wait(drain_arrived_2_addr, iteration_0_2 % 2);
                uint32_t _clc_valid_3 = 0;
                uint32_t _clc_ctaid_x_3;
                uint32_t _clc_ctaid_y_3;
                uint32_t _clc_ctaid_z_3;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%4];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "selp.u32 %3, 1, 0, p1;\n\t"
                    "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid.v4.b32.b128 {%0, %1, %2, _}, clc_r;\n\t"
                    "}\n"
                    : "=r"(_clc_ctaid_x_3), "=r"(_clc_ctaid_y_3), "=r"(_clc_ctaid_z_3), "=r"(_clc_valid_3)
                    : "r"(smem + 560 + 0 * 16 + 0 * 16)
                    : "memory");
                if (_clc_valid_3 != 0) {
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                }
                asm volatile(
                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                    :: "r"((drain_finished_2_addr) & 0xFEFFFFFF) : "memory");
                if (cta_rank_0 == 0) {
                    mbarrier_wait(drain_finished_2_addr, iteration_0_2 % 2);
                }
                if (_clc_valid_3 == 0) {
                    break;
                }
                if (cta_rank_0 == 0) {
                    asm volatile(
                        "clusterlaunchcontrol.try_cancel.async.shared::cta"
                            ".mbarrier::complete_tx::bytes.multicast::cluster::all.b128"
                            " [%0], [%1];"
                        :: "r"(smem + 560 + 0 * 16 + 0 * 16), "r"(drain_arrived_2_addr + 0 * 8)
                        : "memory");
                }
                mbarrier_arrive_expect_tx(drain_arrived_2_addr, 16);
                iteration_0_2 = iteration_0_2 + 1;
            }
        }
        if (cluster >= 0 && tid == 96) {
            int iteration_0_3 = 0;
            if (cta_rank_0 == 0) {
                asm volatile(
                    "clusterlaunchcontrol.try_cancel.async.shared::cta"
                        ".mbarrier::complete_tx::bytes.multicast::cluster::all.b128"
                        " [%0], [%1];"
                    :: "r"(smem + 576 + 0 * 16 + 0 * 16), "r"(drain_arrived_3_addr + 0 * 8)
                    : "memory");
            }
            mbarrier_arrive_expect_tx(drain_arrived_3_addr, 16);
            while (1) {
                mbarrier_wait(drain_arrived_3_addr, iteration_0_3 % 2);
                uint32_t _clc_valid_4 = 0;
                uint32_t _clc_ctaid_x_4;
                uint32_t _clc_ctaid_y_4;
                uint32_t _clc_ctaid_z_4;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%4];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "selp.u32 %3, 1, 0, p1;\n\t"
                    "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid.v4.b32.b128 {%0, %1, %2, _}, clc_r;\n\t"
                    "}\n"
                    : "=r"(_clc_ctaid_x_4), "=r"(_clc_ctaid_y_4), "=r"(_clc_ctaid_z_4), "=r"(_clc_valid_4)
                    : "r"(smem + 576 + 0 * 16 + 0 * 16)
                    : "memory");
                if (_clc_valid_4 != 0) {
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                }
                asm volatile(
                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                    :: "r"((drain_finished_3_addr) & 0xFEFFFFFF) : "memory");
                if (cta_rank_0 == 0) {
                    mbarrier_wait(drain_finished_3_addr, iteration_0_3 % 2);
                }
                if (_clc_valid_4 == 0) {
                    break;
                }
                if (cta_rank_0 == 0) {
                    asm volatile(
                        "clusterlaunchcontrol.try_cancel.async.shared::cta"
                            ".mbarrier::complete_tx::bytes.multicast::cluster::all.b128"
                            " [%0], [%1];"
                        :: "r"(smem + 576 + 0 * 16 + 0 * 16), "r"(drain_arrived_3_addr + 0 * 8)
                        : "memory");
                }
                mbarrier_arrive_expect_tx(drain_arrived_3_addr, 16);
                iteration_0_3 = iteration_0_3 + 1;
            }
        }
        if (cluster >= 0 && tid == 128) {
            int iteration_0_4 = 0;
            if (cta_rank_0 == 0) {
                asm volatile(
                    "clusterlaunchcontrol.try_cancel.async.shared::cta"
                        ".mbarrier::complete_tx::bytes.multicast::cluster::all.b128"
                        " [%0], [%1];"
                    :: "r"(smem + 592 + 0 * 16 + 0 * 16), "r"(drain_arrived_4_addr + 0 * 8)
                    : "memory");
            }
            mbarrier_arrive_expect_tx(drain_arrived_4_addr, 16);
            while (1) {
                mbarrier_wait(drain_arrived_4_addr, iteration_0_4 % 2);
                uint32_t _clc_valid_5 = 0;
                uint32_t _clc_ctaid_x_5;
                uint32_t _clc_ctaid_y_5;
                uint32_t _clc_ctaid_z_5;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%4];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "selp.u32 %3, 1, 0, p1;\n\t"
                    "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid.v4.b32.b128 {%0, %1, %2, _}, clc_r;\n\t"
                    "}\n"
                    : "=r"(_clc_ctaid_x_5), "=r"(_clc_ctaid_y_5), "=r"(_clc_ctaid_z_5), "=r"(_clc_valid_5)
                    : "r"(smem + 592 + 0 * 16 + 0 * 16)
                    : "memory");
                if (_clc_valid_5 != 0) {
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                }
                asm volatile(
                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                    :: "r"((drain_finished_4_addr) & 0xFEFFFFFF) : "memory");
                if (cta_rank_0 == 0) {
                    mbarrier_wait(drain_finished_4_addr, iteration_0_4 % 2);
                }
                if (_clc_valid_5 == 0) {
                    break;
                }
                if (cta_rank_0 == 0) {
                    asm volatile(
                        "clusterlaunchcontrol.try_cancel.async.shared::cta"
                            ".mbarrier::complete_tx::bytes.multicast::cluster::all.b128"
                            " [%0], [%1];"
                        :: "r"(smem + 592 + 0 * 16 + 0 * 16), "r"(drain_arrived_4_addr + 0 * 8)
                        : "memory");
                }
                mbarrier_arrive_expect_tx(drain_arrived_4_addr, 16);
                iteration_0_4 = iteration_0_4 + 1;
            }
        }
        if (cluster >= 0 && tid == 160) {
            int iteration_0_5 = 0;
            if (cta_rank_0 == 0) {
                asm volatile(
                    "clusterlaunchcontrol.try_cancel.async.shared::cta"
                        ".mbarrier::complete_tx::bytes.multicast::cluster::all.b128"
                        " [%0], [%1];"
                    :: "r"(smem + 608 + 0 * 16 + 0 * 16), "r"(drain_arrived_5_addr + 0 * 8)
                    : "memory");
            }
            mbarrier_arrive_expect_tx(drain_arrived_5_addr, 16);
            while (1) {
                mbarrier_wait(drain_arrived_5_addr, iteration_0_5 % 2);
                uint32_t _clc_valid_6 = 0;
                uint32_t _clc_ctaid_x_6;
                uint32_t _clc_ctaid_y_6;
                uint32_t _clc_ctaid_z_6;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%4];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "selp.u32 %3, 1, 0, p1;\n\t"
                    "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid.v4.b32.b128 {%0, %1, %2, _}, clc_r;\n\t"
                    "}\n"
                    : "=r"(_clc_ctaid_x_6), "=r"(_clc_ctaid_y_6), "=r"(_clc_ctaid_z_6), "=r"(_clc_valid_6)
                    : "r"(smem + 608 + 0 * 16 + 0 * 16)
                    : "memory");
                if (_clc_valid_6 != 0) {
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                }
                asm volatile(
                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                    :: "r"((drain_finished_5_addr) & 0xFEFFFFFF) : "memory");
                if (cta_rank_0 == 0) {
                    mbarrier_wait(drain_finished_5_addr, iteration_0_5 % 2);
                }
                if (_clc_valid_6 == 0) {
                    break;
                }
                if (cta_rank_0 == 0) {
                    asm volatile(
                        "clusterlaunchcontrol.try_cancel.async.shared::cta"
                            ".mbarrier::complete_tx::bytes.multicast::cluster::all.b128"
                            " [%0], [%1];"
                        :: "r"(smem + 608 + 0 * 16 + 0 * 16), "r"(drain_arrived_5_addr + 0 * 8)
                        : "memory");
                }
                mbarrier_arrive_expect_tx(drain_arrived_5_addr, 16);
                iteration_0_5 = iteration_0_5 + 1;
            }
        }
        if (cluster >= 0 && tid == 192) {
            int iteration_0_6 = 0;
            if (cta_rank_0 == 0) {
                asm volatile(
                    "clusterlaunchcontrol.try_cancel.async.shared::cta"
                        ".mbarrier::complete_tx::bytes.multicast::cluster::all.b128"
                        " [%0], [%1];"
                    :: "r"(smem + 624 + 0 * 16 + 0 * 16), "r"(drain_arrived_6_addr + 0 * 8)
                    : "memory");
            }
            mbarrier_arrive_expect_tx(drain_arrived_6_addr, 16);
            while (1) {
                mbarrier_wait(drain_arrived_6_addr, iteration_0_6 % 2);
                uint32_t _clc_valid_7 = 0;
                uint32_t _clc_ctaid_x_7;
                uint32_t _clc_ctaid_y_7;
                uint32_t _clc_ctaid_z_7;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%4];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "selp.u32 %3, 1, 0, p1;\n\t"
                    "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid.v4.b32.b128 {%0, %1, %2, _}, clc_r;\n\t"
                    "}\n"
                    : "=r"(_clc_ctaid_x_7), "=r"(_clc_ctaid_y_7), "=r"(_clc_ctaid_z_7), "=r"(_clc_valid_7)
                    : "r"(smem + 624 + 0 * 16 + 0 * 16)
                    : "memory");
                if (_clc_valid_7 != 0) {
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                }
                asm volatile(
                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                    :: "r"((drain_finished_6_addr) & 0xFEFFFFFF) : "memory");
                if (cta_rank_0 == 0) {
                    mbarrier_wait(drain_finished_6_addr, iteration_0_6 % 2);
                }
                if (_clc_valid_7 == 0) {
                    break;
                }
                if (cta_rank_0 == 0) {
                    asm volatile(
                        "clusterlaunchcontrol.try_cancel.async.shared::cta"
                            ".mbarrier::complete_tx::bytes.multicast::cluster::all.b128"
                            " [%0], [%1];"
                        :: "r"(smem + 624 + 0 * 16 + 0 * 16), "r"(drain_arrived_6_addr + 0 * 8)
                        : "memory");
                }
                mbarrier_arrive_expect_tx(drain_arrived_6_addr, 16);
                iteration_0_6 = iteration_0_6 + 1;
            }
        }
        if (cluster >= 0 && tid == 224) {
            int iteration_0_7 = 0;
            if (cta_rank_0 == 0) {
                asm volatile(
                    "clusterlaunchcontrol.try_cancel.async.shared::cta"
                        ".mbarrier::complete_tx::bytes.multicast::cluster::all.b128"
                        " [%0], [%1];"
                    :: "r"(smem + 640 + 0 * 16 + 0 * 16), "r"(drain_arrived_7_addr + 0 * 8)
                    : "memory");
            }
            mbarrier_arrive_expect_tx(drain_arrived_7_addr, 16);
            while (1) {
                mbarrier_wait(drain_arrived_7_addr, iteration_0_7 % 2);
                uint32_t _clc_valid_8 = 0;
                uint32_t _clc_ctaid_x_8;
                uint32_t _clc_ctaid_y_8;
                uint32_t _clc_ctaid_z_8;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%4];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "selp.u32 %3, 1, 0, p1;\n\t"
                    "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid.v4.b32.b128 {%0, %1, %2, _}, clc_r;\n\t"
                    "}\n"
                    : "=r"(_clc_ctaid_x_8), "=r"(_clc_ctaid_y_8), "=r"(_clc_ctaid_z_8), "=r"(_clc_valid_8)
                    : "r"(smem + 640 + 0 * 16 + 0 * 16)
                    : "memory");
                if (_clc_valid_8 != 0) {
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                }
                asm volatile(
                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                    :: "r"((drain_finished_7_addr) & 0xFEFFFFFF) : "memory");
                if (cta_rank_0 == 0) {
                    mbarrier_wait(drain_finished_7_addr, iteration_0_7 % 2);
                }
                if (_clc_valid_8 == 0) {
                    break;
                }
                if (cta_rank_0 == 0) {
                    asm volatile(
                        "clusterlaunchcontrol.try_cancel.async.shared::cta"
                            ".mbarrier::complete_tx::bytes.multicast::cluster::all.b128"
                            " [%0], [%1];"
                        :: "r"(smem + 640 + 0 * 16 + 0 * 16), "r"(drain_arrived_7_addr + 0 * 8)
                        : "memory");
                }
                mbarrier_arrive_expect_tx(drain_arrived_7_addr, 16);
                iteration_0_7 = iteration_0_7 + 1;
            }
        }
    }

    // ---- Role: w0 ----
    if (warp == 0) {
        // idle — no tasks assigned
    }
    // ---- Role: w1 ----
    if (warp == 1) {
        // idle — no tasks assigned
    }
    // ---- Role: w2 ----
    if (warp == 2) {
        // idle — no tasks assigned
    }
    // ---- Role: w3 ----
    if (warp == 3) {
        // idle — no tasks assigned
    }
    // ---- Role: w4 ----
    if (warp == 4) {
        // idle — no tasks assigned
    }
    // ---- Role: w5 ----
    if (warp == 5) {
        // idle — no tasks assigned
    }
    // ---- Role: w6 ----
    if (warp == 6) {
        // idle — no tasks assigned
    }
    // ---- Role: w7 ----
    if (warp == 7) {
        // idle — no tasks assigned
    }

    // Cleanup
    asm volatile("barrier.cluster.arrive.release.aligned;");
    asm volatile("barrier.cluster.wait.acquire.aligned;");

    if (warp == 0) {
        asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(512));
    }
}

} // extern "C"
